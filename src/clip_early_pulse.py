"""One-seed causal early-pulse pilot; deterministic prefix replay, audited state."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.clip_training import (CLIPLossOutput, native_clip_contrastive_loss,
    clip_relative_denominator_loss, hardest_real_negative_indices,
    synthetic_barycentric_weights, tensor_gradient_norm)
from src import clip_train
from src.clip_geometry_v2_metrics import _top_k_support
from src.clip_gradient_stages import ROOT, GradientStageObserver, observational_state, plot_results
from src.clip_warmup_readiness import WarmupObserver
from src.clip_training_eval import evaluate_clip


def auxiliary_losses(output):
    # Same relative-denominator candidates as the readiness probe. No OT solver
    # or gap gate: the controlled intervention is a fixed, calibrated pulse.
    raw = output.raw_similarity
    diagonal = torch.eye(len(raw), dtype=torch.bool, device=raw.device)
    _, support = _top_k_support(raw.detach().float(), diagonal, 8)
    weights = synthetic_barycentric_weights(support.to(raw.dtype), support, 'uniform_topk').detach()
    images, texts = output.image_features, output.text_features
    synthetic = F.normalize(weights.to(images.dtype) @ images, dim=-1)
    scale = output.logit_scale.detach()
    logits = raw * scale
    index = torch.arange(len(raw), device=raw.device)
    return {
        'uniform_top8': clip_relative_denominator_loss(logits, (texts * synthetic).sum(-1) * scale)[0],
        'hardest_real': clip_relative_denominator_loss(logits, logits[index, hardest_real_negative_indices(raw)])[0],
    }


class PulseObjective(torch.nn.Module):
    def __init__(self, arm, start=100, stop=200):
        super().__init__()
        if arm not in ('baseline', 'uniform_top8', 'hardest_real') or not 0 <= start < stop:
            raise ValueError('Invalid pulse arm or bounds')
        self.arm, self.start, self.stop = arm, start, stop
        self.coefficient = None
        self.active_steps = []
        self.gradient_records = []
        self.trainable_parameters = ()

    def forward(self, output, *, species_ids, step):
        native = native_clip_contrastive_loss(output.logits)
        active = self.arm != 'baseline' and self.start <= step < self.stop
        aux = weighted = None
        if active:
            if self.coefficient is None:
                raise RuntimeError('Pulse cannot start before training-only calibration')
            aux = auxiliary_losses(output)[self.arm]
            weighted = self.coefficient * aux
            self.active_steps.append(step)
            if self.trainable_parameters:
                norms = {name: tensor_gradient_norm(torch.autograd.grad(loss,
                    self.trainable_parameters, retain_graph=True, allow_unused=True))
                    for name, loss in [('native', native), ('weighted_aux', weighted)]}
                self.gradient_records.append({'objective_step': step, **norms,
                    'ratio': norms['weighted_aux'] / max(norms['native'], 1e-12)})
        total = native if weighted is None else native + weighted
        return CLIPLossOutput(total, native, aux, weighted, {
            'clip_loss': float(native.detach()), 'total_loss': float(total.detach()),
            'pulse_active': int(active), 'alpha_effective': self.coefficient if active else 0.,
            'weighted_ot_loss': float(weighted.detach()) if active else 0.,
        })


def state_digest(value):
    """Canonical recursive digest, including optimizer tensors and RNG state."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
        else:
            digest.update(repr(item).encode() + b'\0')
    visit(value)
    return digest.hexdigest()


def calibration_coefficients(rows, target):
    if not rows or not 0 < target <= 1:
        raise ValueError('Invalid calibration target or empty calibration')
    means = {key: float(np.mean([r[key] for r in rows]))
             for key in ('native', 'uniform_top8', 'hardest_real')}
    if any(not math.isfinite(v) or v <= 1e-12 for v in means.values()):
        raise ValueError('Nonfinite or degenerate calibration gradient')
    return {arm: target * means['native'] / means[arm]
            for arm in ('uniform_top8', 'hardest_real')}


def calibrate(model, data, device, config, protocol):
    """Four disjoint seeded training batches; no training-loader iteration."""
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    size = config['training']['batch_size']
    positions = np.random.default_rng(protocol['calibration_seed']).permutation(len(data.train_dataset))
    count = protocol['calibration_batches'] * size
    if count > len(positions):
        raise ValueError('Insufficient training examples for disjoint calibration')
    records = []
    with observational_state(model):
        model.train()  # Match the actual pulse training path, including bf16.
        for selected in positions[:count].reshape(-1, size):
            batch = data.train_loader.collate_fn([data.train_dataset[int(i)] for i in selected])
            with clip_train._autocast(device, config['training']['mixed_precision']):
                output = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
                losses = {'native': native_clip_contrastive_loss(output.logits), **auxiliary_losses(output)}
            norms = {key: tensor_gradient_norm(torch.autograd.grad(loss, parameters,
                retain_graph=True, allow_unused=True)) for key, loss in losses.items()}
            records.append({'source_indices': batch['source_indices'].tolist(),
                            'captions': batch['captions'], **norms})
    return {'batches': records, 'coefficients': calibration_coefficients(records,
        protocol['calibration_target_ratio']), 'target_ratio': protocol['calibration_target_ratio'],
        'space': 'all_trainable_parameters_before_gradient_clipping',
        'coefficient_rule': 'target * mean(native_norm) / mean(aux_norm)',
        'caption_epoch': data.train_dataset.epoch}


class PulseObserver(WarmupObserver):
    def __init__(self, output, diagnostic, protocol, arm, checkpoint, shared):
        super().__init__(output, diagnostic, protocol)
        self.arm, self.checkpoint, self.shared = arm, Path(checkpoint), Path(shared)

    def initialize(self, **kwargs):
        GradientStageObserver.initialize(self, **kwargs)
        if len(kwargs['data'].train_loader) != self.protocol['expected_batches_per_epoch']:
            raise ValueError('Unexpected epoch size')
        self.stages = {step: f'pulse_{step}' for step in self.protocol['measurement_steps']}
        clip_train.write_json(self.output_dir / 'projection_probe_batches.json', {
            name: part['batches'][:self.protocol['projection_batches_per_partition']]
            for name, part in self.conditions.items()})

    def bind_training_state(self, *, optimizer, scheduler, objective, data):
        self.optimizer, self.scheduler, self.objective, self.data = optimizer, scheduler, objective, data
        # Store parameters in a plain tuple: do not register/reparent the model.
        objective.trainable_parameters = tuple(p for group in optimizer.param_groups for p in group['params'])

    def __call__(self, *, model, epoch, global_step):
        if global_step == self.protocol['pulse_start']:
            state = {'model': model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict(), 'python_rng': random.getstate(),
                     'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     'loader_generator': self.data.train_loader.generator.get_state(),
                     'epoch': epoch, 'completed_updates': global_step,
                     'batches_consumed_in_epoch': global_step % len(self.data.train_loader)}
            hashes = {key: state_digest(value) for key, value in state.items()}
            reference = self.shared / 'prefix_hashes.json'
            if self.arm == 'baseline':
                clip_train.write_json(reference, hashes)
                # Full model/optimizer/RNG snapshot; replay is used for branching
                # because worker prefetch / the mid-epoch iterator are not serialized.
                torch.save(state, self.checkpoint / 'common_step_100.pt')
            elif json.loads(reference.read_text()) != hashes:
                raise AssertionError('Prefix model/optimizer/RNG/loader state differs between arms')
            clip_train.write_json(self.output_dir / 'prefix_hashes.json', hashes)
            if self.arm == 'baseline':
                calibration = calibrate(model, self.data, self.device, self.config, self.protocol)
                clip_train.write_json(self.shared / 'calibration.json', calibration)
            calibration = json.loads((self.shared / 'calibration.json').read_text())
            if self.arm != 'baseline':
                self.objective.coefficient = calibration['coefficients'][self.arm]
            print('[early pulse] prefix verified; calibration:', calibration['coefficients'], flush=True)
        if global_step in self.stages:
            super().__call__(model=model, epoch=epoch, global_step=global_step)
            with observational_state(model):
                evaluation = evaluate_clip(model, self.processor, self.data, self.device,
                    mixed_precision=self.config['training']['mixed_precision'],
                    prompt_template=self.config['evaluation']['species_prompt'],
                    retrieval_chunk_size=self.config['evaluation']['retrieval_chunk_size'])
            clip_train.write_json(self.output_dir / f'evaluation_{global_step}.json', {
                'completed_updates': global_step, 'evaluation': evaluation})
            clip_train.write_json(self.output_dir / 'pulse_audit.json', {
                'active_objective_steps': self.objective.active_steps,
                'coefficient': self.objective.coefficient,
                'full_parameter_gradient_norms': self.objective.gradient_records})
            print(f'[early pulse] {self.arm} evaluation saved at {global_step}', flush=True)

    def finish(self):
        super().finish()
        expected = [] if self.arm == 'baseline' else list(range(self.protocol['pulse_start'], self.protocol['pulse_stop']))
        if self.objective.active_steps != expected:
            raise AssertionError('Pulse timing/count differs from protocol')


def plot_performance(output, arms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    records = []
    for arm in arms:
        rows = [json.loads(line) for line in (output / arm / 'training/metrics.jsonl').read_text().splitlines()]
        for step in (100, 200):
            rows.append(json.loads((output / arm / 'diagnostics' / f'evaluation_{step}.json').read_text()))
            rows[-1]['global_step'] = step
        rows.sort(key=lambda r: r['global_step'])
        values = []
        for row in rows:
            evaluation = row['evaluation']
            retrieval = evaluation['canonical_retrieval']
            record = {'arm': arm, 'step': row['global_step'],
                'canonical_avg_r1_percent': np.mean([retrieval[d]['r_at_1'] for d in ('text_to_image','image_to_text')]),
                'species_top1_percent': 100 * evaluation['species']['top_1_accuracy']}
            records.append(record)
            values.append(record)
        for ax, metric in zip(axes, ('canonical_avg_r1_percent', 'species_top1_percent')):
            ax.plot([v['step'] for v in values], [v[metric] for v in values], marker='.', label=arm)
            ax.set(xlabel='Completed updates', ylabel=metric)
    for ax in axes:
        ax.axvspan(100, 200, color='gray', alpha=.15)
    axes[0].legend()
    fig.suptitle('Early pulse pilot: one seed; shaded interval is the intervention')
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'early_pulse_performance.{suffix}', dpi=170)
    plt.close(fig)
    clip_train.write_json(output / 'performance.json', records)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-directory', required=True)
    parser.add_argument('--checkpoint-directory', required=True)
    args = parser.parse_args()
    output, checkpoint = Path(args.output_directory), Path(args.checkpoint_directory)
    if output.exists() or checkpoint.exists():
        raise FileExistsError('Fresh experiment paths required')
    output.mkdir(parents=True)
    protocol = yaml.safe_load((ROOT / 'configs/clip_early_pulse.yaml').read_text())
    diagnostic = yaml.safe_load((ROOT / protocol['diagnostic_config']).read_text())['diagnostic']
    clip_train.write_json(output / 'protocol.json', protocol)
    for arm in protocol['arms']:
        config = clip_train.load_training_config(ROOT / protocol['baseline_config'])
        config['diagnostics']['separate_projection_gradient_steps'] = 0
        if config['training']['seed'] != protocol['seed']:
            raise ValueError('Unexpected seed')
        observer = PulseObserver(output / arm / 'diagnostics', diagnostic, protocol,
                                 arm, checkpoint / arm, output)
        clip_train.run(config, output_directory=output / arm / 'training',
            checkpoint_directory=checkpoint / arm, observer=observer,
            stop_after_epochs=protocol['stop_after_epochs'],
            objective_factory=lambda cfg: PulseObjective(arm, protocol['pulse_start'], protocol['pulse_stop']))
        observer.finish()
        clip_train.write_json(output / arm / 'completion.json', {'status': 'complete', 'arm': arm})
        del observer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    plot_results(output)
    plot_performance(output, protocol['arms'])
    clip_train.write_json(output / 'completion.json', {'status': 'complete',
        'completed_updates': 1001, 'arms': protocol['arms']})


if __name__ == '__main__':
    main()
