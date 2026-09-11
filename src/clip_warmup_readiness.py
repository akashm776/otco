"""Dense baseline warmup measurements, including exact projection-head gradients."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.clip_training import (
    clip_relative_denominator_loss, hardest_real_negative_indices,
    native_clip_contrastive_loss, synthetic_barycentric_weights,
)
from src import clip_train
from src.clip_geometry_v2_metrics import _top_k_support
from src.clip_gradient_stages import GradientStageObserver, ROOT, plot_results


def probe_objectives(images, texts, scale, top_k=8):
    raw = texts @ images.T
    diagonal = torch.eye(len(images), dtype=torch.bool, device=images.device)
    _, support = _top_k_support(raw.detach(), diagonal, top_k)
    weights = synthetic_barycentric_weights(support.to(raw.dtype), support, 'uniform_topk').detach()
    synthetic = F.normalize(weights @ images, dim=-1)
    hardest = hardest_real_negative_indices(raw)
    index = torch.arange(len(images), device=images.device)
    logits = raw * scale.detach()
    return {
        'native_symmetric': native_clip_contrastive_loss(logits),
        'native_row': F.cross_entropy(logits, index),
        'u8': clip_relative_denominator_loss(logits, (texts * synthetic).sum(-1) * scale.detach())[0],
        'real': clip_relative_denominator_loss(logits, logits[index, hardest])[0],
        'margin': (raw.diagonal() - raw[index, hardest]).mean(),
    }


def gradient_vectors(objectives, leaves):
    return {name: tuple(g.detach() for g in torch.autograd.grad(loss, leaves, retain_graph=True))
            for name, loss in objectives.items()}


def vector_metrics(vectors):
    norm = {name: float(v.norm()) for name, v in vectors.items()}
    def cosine(a, b):
        return float(torch.dot(vectors[a] / norm[a], vectors[b] / norm[b])) if min(norm[a], norm[b]) > 1e-12 else None
    result = {f'norm_{name}': value for name, value in norm.items()}
    for auxiliary in ('u8', 'real'):
        for native in ('native_symmetric', 'native_row'):
            result[f'cos_{auxiliary}_{native}'] = cosine(auxiliary, native)
        result[f'margin_change_{auxiliary}'] = (
            float(-torch.dot(vectors['margin'], vectors[auxiliary] / norm[auxiliary]))
            if norm[auxiliary] > 1e-12 else None)
    result['cos_u8_real'] = cosine('u8', 'real')
    result['u8_to_real_norm_ratio'] = norm['u8'] / norm['real'] if norm['real'] > 1e-12 else None
    return result


def projection_probe(visual_projection, text_projection, pooled_images, pooled_texts, scale, top_k=8):
    """Exact head gradients conditional on the current detached encoder outputs.

    Reapplying the live linear heads preserves the full derivative wrt their
    weights; detaching encoder outputs excludes only encoder parameters.
    A matched batch-level embedding probe controls the aggregation difference
    from the existing per-query embedding diagnostic.
    """
    parameters = tuple(visual_projection.parameters()) + tuple(text_projection.parameters())
    visual_count = len(tuple(visual_projection.parameters()))
    images = F.normalize(visual_projection(pooled_images.detach()), dim=-1)
    texts = F.normalize(text_projection(pooled_texts.detach()), dim=-1)
    objectives = probe_objectives(images, texts, scale, top_k)
    gradients = gradient_vectors(objectives, parameters)
    flatten = lambda values: torch.cat([v.reshape(-1) for v in values])
    rows = []
    for space, selection in [('projection_joint', slice(None)),
                             ('visual_projection', slice(None, visual_count)),
                             ('text_projection', slice(visual_count, None))]:
        vectors = {name: flatten(values[selection]) for name, values in gradients.items()}
        rows.append({'space': space, **vector_metrics(vectors)})
    image_leaves, text_leaves = images.detach().clone().requires_grad_(True), texts.detach().clone().requires_grad_(True)
    embedding_objectives = probe_objectives(F.normalize(image_leaves, dim=-1),
                                           F.normalize(text_leaves, dim=-1), scale, top_k)
    embedding_gradients = gradient_vectors(embedding_objectives, (image_leaves, text_leaves))
    rows.append({'space': 'embedding_batch', **vector_metrics({k: flatten(v) for k, v in embedding_gradients.items()})})
    return rows


class WarmupObserver(GradientStageObserver):
    def __init__(self, output_dir, diagnostic, protocol):
        super().__init__(output_dir, diagnostic, 'baseline')
        self.protocol = protocol
        self.projection_rows = []

    def initialize(self, **kwargs):
        config = kwargs['config']
        if config['ot']['enabled']:
            raise ValueError('Warmup readiness must train baseline only')
        if len(kwargs['data'].train_loader) != self.protocol['expected_batches_per_epoch']:
            raise ValueError('Unexpected number of batches per epoch')
        steps = self.protocol['measurement_steps']
        if steps != [0, 100, 250, 500, 750, 1001]:
            raise ValueError('Unexpected preregistered measurement times')
        if steps[-1] != self.protocol['stop_after_epochs'] * len(kwargs['data'].train_loader):
            raise ValueError('Final diagnostic must equal the training stop')
        super().initialize(**kwargs)
        self.stages = {step: ('pretrained' if step == 0 else f'warmup_{step}') for step in steps}
        selected = {name: condition['batches'][:self.protocol['projection_batches_per_partition']]
                    for name, condition in self.conditions.items()}
        clip_train.write_json(self.output_dir / 'projection_probe_batches.json', selected)

    def capture(self, model, epoch, global_step):
        pooled = {'images': [], 'texts': []}
        def collect(name):
            return lambda module, inputs: pooled[name].append(inputs[0].detach().cpu())
        heads = model.clip_model.visual_projection, model.clip_model.text_projection
        hooks = [heads[0].register_forward_pre_hook(collect('images')),
                 heads[1].register_forward_pre_hook(collect('texts'))]
        try:
            super().capture(model, epoch, global_step)
        finally:
            for hook in hooks:
                hook.remove()
        pooled = {name: torch.cat(values).clone() for name, values in pooled.items()}
        if any(len(tensor) != self.diagnostic['holdout_size'] for tensor in pooled.values()):
            raise AssertionError('Projection hook count does not match the fixed holdout')
        stage_dir = self.output_dir / f'{global_step:06d}_{self.stages[global_step]}'
        saved = torch.load(stage_dir / 'features.pt', weights_only=True)
        # Verify captured head inputs reconstruct the exact diagnostic embeddings.
        with torch.no_grad():
            for key, head, feature_key in [('images', heads[0], 'image_features'), ('texts', heads[1], 'text_features')]:
                reconstructed = torch.cat([F.normalize(head(chunk.to(self.device)), dim=-1).cpu()
                    for chunk in pooled[key].split(self.diagnostic['batch_size'])])
                torch.testing.assert_close(reconstructed, saved[feature_key], atol=1e-6, rtol=1e-5)
        torch.save({'pooled_images': pooled['images'], 'pooled_texts': pooled['texts'],
                    'visual_projection': heads[0].state_dict(), 'text_projection': heads[1].state_dict()},
                   stage_dir / 'projection_inputs_and_heads.pt')
        current = []
        scale = model.get_logit_scale().detach().float()
        for name, condition in self.conditions.items():
            for batch_index, positions in enumerate(condition['batches'][:self.protocol['projection_batches_per_partition']]):
                rows = projection_probe(*heads, pooled['images'][positions].to(self.device),
                                        pooled['texts'][positions].to(self.device), scale)
                current.extend({'completed_updates': global_step, 'partition': name,
                                'batch_index': batch_index, **row} for row in rows)
        self.projection_rows.extend(current)
        clip_train.write_json(stage_dir / 'projection_probe.json', current)
        clip_train.write_json(self.output_dir / 'projection_summary.json', self.projection_rows)
        with (self.output_dir / 'projection_summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.projection_rows[0]))
            writer.writeheader()
            writer.writerows(self.projection_rows)
        print(f'[warmup projection] saved {len(current)} space/batch measurements at step {global_step}', flush=True)

    def finish(self):
        super().finish()
        expected = len(self.stages) * len(self.conditions) * self.protocol['projection_batches_per_partition'] * 4
        if len(self.projection_rows) != expected:
            raise AssertionError('Missing projection measurements')


def plot_projection(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.loads((output / 'baseline/diagnostics/projection_summary.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [('cos_u8_native_symmetric', 'U8 vs native symmetric CLIP'),
              ('cos_real_native_symmetric', 'Real vs native symmetric CLIP'),
              ('cos_u8_real', 'U8 vs real auxiliary')]
    for ax, (metric, title) in zip(axes, panels):
        for space in ('projection_joint', 'embedding_batch'):
            selected = [r for r in rows if r['space'] == space and r['partition'] != 'sequential']
            steps = sorted({r['completed_updates'] for r in selected})
            values = np.asarray([[r[metric] for r in selected if r['completed_updates'] == step] for step in steps], dtype=float)
            line, = ax.plot(steps, values.mean(1), marker='o', label=space)
            ax.fill_between(steps, values.min(1), values.max(1), alpha=.15, color=line.get_color())
        ax.axhline(0, color='gray', lw=.7)
        ax.set(title=title, xlabel='Completed baseline updates', ylabel='Gradient cosine')
    axes[0].legend(fontsize=8)
    fig.suptitle('Matched batch objectives: 2 fixed batches × 3 shuffled partitions; min–max bands, not CIs')
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'projection_readiness.{suffix}', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-directory', required=True)
    parser.add_argument('--checkpoint-directory', required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT / 'configs/clip_warmup_readiness.yaml').read_text())
    config = clip_train.load_training_config(ROOT / protocol['baseline_config'])
    diagnostic = yaml.safe_load((ROOT / protocol['diagnostic_config']).read_text())['diagnostic']
    output, checkpoint = Path(args.output_directory), Path(args.checkpoint_directory)
    if output.exists() or checkpoint.exists():
        raise FileExistsError('Choose fresh output/checkpoint paths; never overwrite previous runs')
    output.mkdir(parents=True)
    clip_train.write_json(output / 'protocol.json', protocol)
    observer = WarmupObserver(output / 'baseline/diagnostics', diagnostic, protocol)
    clip_train.run(config, output_directory=output / 'baseline/training', checkpoint_directory=checkpoint,
                   observer=observer, stop_after_epochs=protocol['stop_after_epochs'])
    observer.finish()
    plot_results(output)
    plot_projection(output)
    clip_train.write_json(output / 'completion.json', {'status': 'complete', 'completed_updates': 1001,
        'measurement_steps': protocol['measurement_steps'], 'projection_space_batch_measurements': len(observer.projection_rows)})


if __name__ == '__main__':
    main()
