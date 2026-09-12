"""Actual paired AdamW updates from fixed baseline states; no training rollout."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from model.clip_backend import CLIPEncoderBackend
from model.clip_training import (native_clip_contrastive_loss, build_clip_optimizer,
    configure_clip_trainable_parameters, tensor_gradient_norm)
from src import clip_train
from src.clip_early_pulse import auxiliary_losses, state_digest
from src.clip_geometry_diagnostic import CUBCLIPDiagnosticDataset
from src.clip_negative_gradient_geometry_randomized import build_partition_conditions, EXPECTED_PARTITIONS
from src.clip_gradient_stages import ROOT, observational_state


def select_batches(source_indices, excluded, *, seed, count, size):
    excluded = set(excluded)
    positions = [i for i, source in enumerate(source_indices) if source not in excluded]
    if len(positions) < count * size:
        raise ValueError('Not enough disjoint training examples')
    chosen = np.random.default_rng(seed).permutation(positions)[:count * size]
    return chosen.reshape(count, size).tolist()


def cosine(a, b):
    denominator = a.double().norm() * b.double().norm()
    return float(torch.dot(a.double(), b.double()) / denominator) if denominator > 1e-20 else None


def flat_grad(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                      for p, g in zip(parameters, grads)])


def one_update(model, optimizer, batch, config, *, arm, coefficient, seed):
    """Exactly the trainer's forward/backward, clipping, AdamW and scale clamp."""
    device = next(model.parameters()).device
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = tuple(p for _, p in named)
    before = [p.detach().clone() for p in parameters]
    clip_train.set_global_seed(seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with clip_train._autocast(device, config['training']['mixed_precision']):
        output = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
        native = native_clip_contrastive_loss(output.logits)
        auxiliary = None if arm == 'baseline' else auxiliary_losses(output)[arm]
        total = native if auxiliary is None else native + coefficient * auxiliary
    native_gradient = flat_grad(native, parameters)
    metrics = {'native_loss_bf16_training_path': float(native.detach()),
               'native_gradient_norm': float(native_gradient.double().norm())}
    if auxiliary is not None:
        auxiliary_gradient = flat_grad(auxiliary, parameters)
        metrics.update(auxiliary_gradient_norm=float(auxiliary_gradient.double().norm()),
                       full_gradient_alignment=cosine(native_gradient, auxiliary_gradient))
        projection_mask = torch.cat([torch.full((p.numel(),), 'projection' in n, device=device,
                                                dtype=torch.bool) for n,p in named])
        metrics['projection_gradient_alignment'] = cosine(native_gradient[projection_mask], auxiliary_gradient[projection_mask])
        metrics['weighted_gradient_ratio'] = coefficient * metrics['auxiliary_gradient_norm'] / metrics['native_gradient_norm']
    total.backward()
    unclipped_norm = torch.nn.utils.clip_grad_norm_(parameters, config['optimizer']['max_gradient_norm'])
    metrics['total_gradient_norm_before_clip'] = float(unclipped_norm)
    optimizer.step()
    model.clip_model.logit_scale.data.clamp_(max=math.log(100.0))
    delta = torch.cat([(p.detach() - old).float().flatten().cpu() for p, old in zip(parameters, before)])
    metrics['actual_update_norm'] = float(delta.double().norm())
    metrics['logit_scale_after'] = float(model.get_logit_scale().detach())
    metrics['actual_update_norm_by_group'] = {}
    offset = 0
    for name, p in named:
        key = 'projection' if 'projection' in name else ('logit_scale' if 'logit_scale' in name else 'encoder')
        value = float(delta[offset:offset+p.numel()].double().square().sum())
        metrics['actual_update_norm_by_group'][key] = metrics['actual_update_norm_by_group'].get(key, 0.) + value
        offset += p.numel()
    metrics['actual_update_norm_by_group'] = {k: math.sqrt(v) for k,v in metrics['actual_update_norm_by_group'].items()}
    return metrics, delta


def restore_branch(model, optimizer, scheduler, initial, seed):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(initial['model'][name])
        for name, buffer in model.named_buffers():
            if name in initial['model']:
                buffer.copy_(initial['model'][name])
    # load_state_dict may alias CPU tensors: deepcopy is essential to preserve
    # the fixed source moments during CPU tests and repeated branch resets.
    optimizer.load_state_dict(deepcopy(initial['optimizer']))
    scheduler.load_state_dict(deepcopy(initial['scheduler']))
    optimizer.zero_grad(set_to_none=True)
    clip_train.set_global_seed(seed)


@torch.inference_mode()
def encode_cached(model, batches):
    device = next(model.parameters()).device
    images, texts = [], []
    with observational_state(model):
        for batch in batches:
            images.append(model.encode_images(batch['pixel_values'].to(device)).cpu())
            texts.append(model.encode_texts(clip_train._text_batch(batch, device)).cpu())
    return torch.cat(images), torch.cat(texts), float(model.get_logit_scale().detach())


def feature_hash(encoded):
    images, texts, scale = encoded
    return hashlib.sha256(images.numpy().tobytes() + texts.numpy().tobytes() + json.dumps(scale).encode()).hexdigest()


def partition_losses(encoded, conditions):
    images, texts, scale = encoded
    # fp32 encoder outputs, fp64 score/reduction arithmetic to resolve small
    # paired effects; not the bf16 training objective's rounded scalar.
    values = {}
    for name, condition in conditions.items():
        values[name] = [float(native_clip_contrastive_loss(texts[indices].double() @ images[indices].double().T * scale))
                        for indices in condition['batches']]
    return values


def primary_mean(values, primary):
    return float(np.mean([x for name in primary for x in values[name]]))


def train_eval(model, batch):
    n = len(batch['pixel_values'])
    return partition_losses(encode_cached(model, [batch]), {'train': {'batches': [list(range(n))]}})['train'][0]


def normalize_checkpoint(checkpoint, step):
    if step == 100:
        if checkpoint.get('completed_updates') != 100:
            raise ValueError('Incorrect early checkpoint')
        return {k: checkpoint[k] for k in ('model','optimizer','scheduler')}
    if checkpoint.get('epoch') != 13 or checkpoint.get('metrics', {}).get('global_step') != 1001:
        raise ValueError('Incorrect late checkpoint')
    return {key: checkpoint[key + '_state_dict'] for key in ('model','optimizer','scheduler')}


def verified_checkpoint(source, relative, manifest):
    path = source / relative
    expected = manifest['verified_files'][relative]['sha256']
    from colabs.backup_clip_run import checksum
    if checksum(path) != expected:
        raise ValueError(f'Checkpoint checksum mismatch: {relative}')
    # These are our own authenticated, hash-verified checkpoint artifacts.
    return torch.load(path, map_location='cpu', weights_only=False)


def cache_heldout(processor, data, holdout):
    diagnostic = CUBCLIPDiagnosticDataset(data.train_dataset.grouped_split, data.train_dataset.species_ids, holdout)
    batches = []
    for start in range(0, len(diagnostic), 64):
        records = [diagnostic[i] for i in range(start, min(start + 64, len(diagnostic)))]
        batches.append(dict(processor(images=[r['image'] for r in records], text=[r['caption'] for r in records],
                                      padding=True, truncation=True, return_tensors='pt')))
    return batches


def run(source, output, protocol=None):
    from transformers import AutoProcessor
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    output.mkdir(parents=True)
    protocol = deepcopy(protocol) if protocol is not None else yaml.safe_load((ROOT / 'configs/clip_paired_updates.yaml').read_text())
    clip_train.write_json(output / 'protocol.json', protocol)
    if source.name != protocol['source_run']:
        raise ValueError('Unexpected source run')
    config = clip_train.load_training_config(ROOT / 'configs/hf_cub200_clip_vit_b32_baseline.yaml')
    device, info = clip_train.resolve_device(config['runtime'])
    clip_train.set_global_seed(42)
    processor = AutoProcessor.from_pretrained(config['model']['checkpoint'])
    data = clip_train.build_clip_training_data(config, processor, pin_memory=True)
    model = CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
    configure_clip_trainable_parameters(model, config['model']['trainable_policy'])
    optimizer = build_clip_optimizer(model, config['optimizer'])
    scheduler = clip_train.build_scheduler(optimizer, warmup_steps=100, total_steps=3850)
    results = source / 'results'
    read = lambda p: json.loads(p.read_text())
    calibration = read(results / 'calibration.json')
    if calibration['coefficients'] != protocol['coefficients']:
        raise ValueError('Coefficients must remain fixed from the pilot')
    holdout = read(results / 'baseline/diagnostics/diagnostic_holdout_indices.json')
    if set(holdout) & set(data.train_dataset.source_indices):
        raise AssertionError('Training/holdout overlap')
    excluded = [i for batch in calibration['batches'] for i in batch['source_indices']]
    positions = select_batches(data.train_dataset.source_indices, excluded, seed=protocol['selection_seed'],
                               count=protocol['training_batches'], size=protocol['batch_size'])
    data.train_dataset.set_epoch(protocol['caption_epoch'])
    training_batches, identities = [], []
    for selected in positions:
        batch = data.train_loader.collate_fn([data.train_dataset[i] for i in selected])
        identities.append({'source_indices': batch['source_indices'].tolist(), 'captions': batch['captions']})
        training_batches.append(batch)
    heldout_batches = cache_heldout(processor, data, holdout)
    conditions, partitions = build_partition_conditions(EXPECTED_PARTITIONS, len(holdout), 64)
    clip_train.write_json(output / 'protocol.json', protocol)
    clip_train.write_json(output / 'training_batches.json', identities)
    clip_train.write_json(output / 'heldout_partitions.json', partitions)
    for name, expected in protocol.get('fixed_input_sha256', {}).items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise AssertionError('Fixed diagnostic inputs differ from original seed-42 experiment: ' + name)
    clip_train.write_json(output / 'device.json', info)
    manifest = read(source / 'backup_manifest.json')
    all_rows, provenance = [], []
    for step, relative in [(100, 'checkpoints/baseline/common_step_100.pt'), (1001, 'checkpoints/baseline/latest.pt')]:
        checkpoint = verified_checkpoint(source, relative, manifest)
        if 'training_seed' in protocol:
            actual_seed = checkpoint.get('training_seed') if step == 100 else checkpoint['config']['training']['seed']
            if actual_seed != protocol['training_seed']:
                raise AssertionError('Checkpoint training seed differs from replication protocol')
        if step == 100:
            hashes = {k: state_digest(v) for k,v in checkpoint.items()}
            if hashes != read(results / 'prefix_hashes.json'):
                raise AssertionError('Early source checkpoint does not match the verified prefix')
        initial = normalize_checkpoint(checkpoint, step)
        initial_hash = state_digest(initial)
        model.load_state_dict(initial['model'])
        restore_branch(model, optimizer, scheduler, initial, protocol['branch_seed'])
        frozen = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        original_encoded = encode_cached(model, heldout_batches)
        expected_feature_hash = read(results / 'baseline/diagnostics' / f'{step:06d}_pulse_{step}/report.json')['features_and_scale_sha256']
        if feature_hash(original_encoded) != expected_feature_hash:
            raise AssertionError('Checkpoint does not reproduce saved held-out features')
        if feature_hash(encode_cached(model, heldout_batches)) != expected_feature_hash:
            raise AssertionError('No-update re-encoding is nondeterministic')
        initial_losses = partition_losses(original_encoded, conditions)
        initial_mean = primary_mean(initial_losses, protocol['primary_partitions'])
        provenance.append({'step': step, 'checkpoint_sha256': manifest['verified_files'][relative]['sha256'],
            'features_sha256': expected_feature_hash, 'initial_state_sha256': initial_hash,
            'learning_rates': [g['lr'] for g in optimizer.param_groups], 'initial_heldout_loss': initial_losses,
            'no_update_reencoding_exact': True})
        clip_train.write_json(output / 'checkpoint_provenance.json', provenance)
        for trial, batch in enumerate(training_batches):
            seed = protocol['branch_seed'] + trial
            restore_branch(model, optimizer, scheduler, initial, seed)
            train_before = train_eval(model, batch)
            native_delta, native_row = None, None
            for arm in protocol['arms']:
                restore_branch(model, optimizer, scheduler, initial, seed)
                if state_digest(optimizer.state_dict()) != state_digest(initial['optimizer']):
                    raise AssertionError('Optimizer reset failed')
                coefficient = protocol['coefficients'].get(arm, 0.)
                metrics, delta = one_update(model, optimizer, batch, config, arm=arm, coefficient=coefficient, seed=seed)
                scheduler.step()
                heldout_loss = partition_losses(encode_cached(model, heldout_batches), conditions)
                heldout_mean = primary_mean(heldout_loss, protocol['primary_partitions'])
                row = {'checkpoint_step': step, 'trial': trial, 'arm': arm, 'coefficient': coefficient,
                    'heldout_loss': heldout_loss, 'heldout_mean': heldout_mean, 'heldout_change': heldout_mean - initial_mean,
                    'train_before': train_before, 'train_after': train_eval(model, batch), **metrics}
                if arm == 'baseline':
                    native_delta, native_row = delta, row
                    if trial == 0:
                        restore_branch(model, optimizer, scheduler, initial, seed)
                        _, repeated = one_update(model, optimizer, batch, config, arm=arm, coefficient=0., seed=seed)
                        scheduler.step()
                        if not torch.equal(delta, repeated):
                            raise AssertionError('Repeated native update differs: reset or nondeterminism problem')
                        repeated_losses = partition_losses(encode_cached(model, heldout_batches), conditions)
                        if repeated_losses != heldout_loss:
                            raise AssertionError('Repeated native held-out evaluation differs')
                        provenance[-1]['native_update_and_evaluation_replay_exact'] = True
                row.update(incremental_heldout_loss=heldout_mean - native_row['heldout_mean'],
                    incremental_train_loss=row['train_after'] - native_row['train_after'],
                    actual_update_cosine_to_native=cosine(delta, native_delta),
                    actual_update_difference_norm=float((delta - native_delta).double().norm()),
                    actual_update_difference_ratio=float((delta - native_delta).double().norm() / native_delta.double().norm()))
                all_rows.append(row)
                with (output / 'paired_updates.jsonl').open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
                print(f'[paired update] checkpoint={step} trial={trial + 1}/16 arm={arm} incremental_heldout={row["incremental_heldout_loss"]:+.8f}', flush=True)
            clip_train.write_json(output / 'progress.json', {'checkpoint_step': step, 'completed_trials_at_checkpoint': trial+1, 'rows': len(all_rows)})
        assert state_digest(initial) == initial_hash, 'Source model/moments mutated'
        assert all(torch.equal(p.detach().cpu(), frozen[n]) for n,p in model.named_parameters() if n in frozen), 'Frozen parameters mutated'
        provenance[-1].update(source_state_immutable=True, frozen_parameters_unchanged=True,
                              optimizer_reset_checks=48)
        clip_train.write_json(output / 'checkpoint_provenance.json', provenance)
        del checkpoint, initial, frozen
    if len(all_rows) != 96:
        raise AssertionError('Incomplete paired experiment')
    summarize(output, all_rows)
    clip_train.write_json(output / 'completion.json', {'status': 'complete', 'checkpoint_steps': [100,1001], 'trials_per_checkpoint': 16, 'rows': 96})


def summarize(output, rows):
    summaries = []
    for step in [100,1001]:
        for arm in ['uniform_top8','hardest_real']:
            selected = [r for r in rows if r['checkpoint_step'] == step and r['arm'] == arm]
            values = np.array([r['incremental_heldout_loss'] for r in selected])
            summaries.append({'checkpoint_step': step, 'arm': arm, 'mean_incremental_heldout_loss': float(values.mean()),
                'median_incremental_heldout_loss': float(np.median(values)), 'beneficial_trials': int((values < 0).sum()),
                'range': [float(values.min()), float(values.max())],
                **{f'mean_{key}': float(np.mean([r[key] for r in selected])) for key in
                   ['incremental_train_loss','full_gradient_alignment','projection_gradient_alignment','actual_update_difference_ratio','actual_update_cosine_to_native']}})
    clip_train.write_json(output / 'summary.json', summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, step in zip(axes, [100,1001]):
        for arm, color in [('uniform_top8','tab:orange'), ('hardest_real','tab:green')]:
            selected = [r for r in rows if r['checkpoint_step'] == step and r['arm'] == arm]
            ax.plot([r['trial']+1 for r in selected], [r['incremental_heldout_loss'] for r in selected], marker='o', color=color, label=arm)
        ax.axhline(0, color='black', lw=.8)
        ax.set(title=f'Baseline checkpoint: {step} updates', xlabel='Fixed training batch', ylabel='Extra held-out loss vs native update (lower better)')
    axes[0].legend()
    fig.suptitle('Actual AdamW one-step effects; 16 paired inputs, one training seed')
    for suffix in ['png','svg']:
        fig.savefig(output / f'paired_update_effects.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-directory', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    run(args.source_directory, args.output_directory)


if __name__ == '__main__':
    main()
