"""Four-arm CLIP/CUB rollout pilot, with a frozen training-only state gate."""

import gc
import hashlib
import json
import math
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from model.clip_training import CLIPLossOutput, native_clip_contrastive_loss
from colabs.gated_checkpoint_retention import required_paths, save_hybrid_checkpoint
from src import clip_train
from src.clip_early_pulse import auxiliary_losses, state_digest
from src.clip_gradient_stages import ROOT, observational_state
from src.clip_paired_updates import cosine, flat_grad
from src.clip_training_data import CLIPCUBPairDataset

ARMS = ['baseline', 'always_on', 'step_gated', 'alignment_gated']


def load_protocol():
    protocol = json.loads((ROOT / 'configs/clip_gated_training.json').read_text())
    rules = json.loads((ROOT / 'configs/clip_usefulness_frozen_rules.json').read_text())['rules']
    if (protocol['arms'] != ARMS or protocol['training_seeds'] != [789, 2026, 31415]
            or protocol['rules_refitted'] or protocol['completed_updates'] != 1001
            or protocol['stop_after_epochs'] * protocol['expected_batches_per_epoch'] != 1001
            or protocol['alignment_refresh_steps'] != [100, 250, 500, 750]
            or protocol['checkpoint_steps'] != [100, 250, 500, 750, 1001]
            or protocol['checkpoint_retention'] != 'hybrid_v1'
            or protocol['common_native_prefix_updates'] != 100
            or protocol['probe_batch_count'] != 16 or protocol['probe_batch_size'] != 64
            or protocol['alignment_threshold'] != rules['full_gradient_alignment']['threshold']
            or protocol['step_threshold'] != rules['checkpoint_step']['threshold']
            or protocol['uniform_top8_coefficient'] != 0.06005351588542947):
        raise ValueError('Frozen pilot protocol changed')
    return protocol


def gate_active(arm, step, protocol, alignment=None):
    """step is the number of already completed updates (zero-based objective)."""
    if arm not in ARMS or not 0 <= step < protocol['completed_updates']:
        raise ValueError('Invalid arm or objective step')
    if step < protocol['common_native_prefix_updates'] or arm == 'baseline':
        return False
    if arm == 'always_on':
        return True
    if arm == 'step_gated':
        return step <= protocol['step_threshold']
    if alignment is None or not math.isfinite(alignment):
        raise ValueError('Missing/nonfinite alignment: refusing a gate decision')
    return alignment > protocol['alignment_threshold']


class GatedObjective(torch.nn.Module):
    def __init__(self, arm, protocol):
        super().__init__()
        self.arm, self.protocol = arm, protocol
        self.alignment = None
        self.decisions = []

    def forward(self, output, *, species_ids, step):
        active = gate_active(self.arm, step, self.protocol, self.alignment)
        native = native_clip_contrastive_loss(output.logits)
        auxiliary = auxiliary_losses(output)['uniform_top8'] if active else None
        weighted = self.protocol['uniform_top8_coefficient'] * auxiliary if active else None
        total = native + weighted if active else native
        self.decisions.append({'objective_step': step, 'active': active})
        return CLIPLossOutput(total, native, auxiliary, weighted, {
            'clip_loss': float(native.detach()), 'total_loss': float(total.detach()),
            'gate_active': int(active),
            'alpha_effective': self.protocol['uniform_top8_coefficient'] if active else 0.,
        })


def fixed_probe_batches(data, protocol, output):
    """Reconstruct the archived batches, never iterate the training loader."""
    reference = ROOT / 'experiment_results/clip_paired_updates_2026-09'
    indices = json.loads((reference / 'training_source_indices.json').read_text())
    dataset = CLIPCUBPairDataset(data.train_dataset.grouped_split,
        data.train_dataset.species_ids, source_indices=data.train_dataset.source_indices,
        seed=protocol['probe_caption_seed'])
    dataset.set_epoch(protocol['probe_caption_epoch'])
    positions = {source: i for i, source in enumerate(dataset.source_indices)}
    if len(indices) != 16 or any(len(batch) != 64 for batch in indices):
        raise ValueError('Unexpected archived probe layout')
    if len({i for batch in indices for i in batch}) != 1024:
        raise ValueError('Probe identities must be disjoint training examples')
    batches, identities = [], []
    for selected in indices:
        if not set(selected) <= positions.keys():
            raise ValueError('Probe includes excluded or nontraining examples')
        batch = data.train_loader.collate_fn([dataset[positions[i]] for i in selected])
        batches.append(batch)
        identities.append({'source_indices': batch['source_indices'].tolist(),
                           'captions': batch['captions']})
    path = output / 'training_batches.json'
    clip_train.write_json(path, identities)
    expected = json.loads((reference / 'audit.json').read_text())['files']['training_batches.json']['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AssertionError('Probe batch/caption identity differs from frozen-rule inputs')
    return batches


def measure_alignment(model, batches, config, branch_seed):
    """Mean of full-parameter cosines, using the historical bf16 train path.

    No optimizer updates, accumulated .grad writes, loader iteration or held-out
    outcomes. Preserve buffers, modes and global RNG, including on failure.
    """
    device = next(model.parameters()).device
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    values = []
    try:
        with observational_state(model):
            model.train()
            for trial, batch in enumerate(batches):
                clip_train.set_global_seed(branch_seed + trial)
                with clip_train._autocast(device, config['training']['mixed_precision']):
                    output = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
                    native = native_clip_contrastive_loss(output.logits)
                    auxiliary = auxiliary_losses(output)['uniform_top8']
                value = cosine(flat_grad(native, parameters), flat_grad(auxiliary, parameters))
                if value is None or not math.isfinite(value):
                    raise ValueError('Degenerate/nonfinite probe gradient')
                values.append(value)
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
    if not values:
        raise ValueError('Empty probe')
    return {'batch_cosines': values, 'mean_cosine': statistics.fmean(values)}


class RolloutObserver:
    def __init__(self, directory, shared, arm, protocol, sync):
        self.directory, self.shared = Path(directory), Path(shared)
        self.arm, self.protocol, self.sync = arm, protocol, sync
        self.probes = []

    def initialize(self, *, model, processor, data, device, config, total_steps):
        if len(data.train_loader) != 77 or total_steps != self.protocol['scheduler_horizon_updates']:
            raise AssertionError('Unexpected training/LR horizon')
        self.model, self.config, self.data = model, config, data
        # The same setup occurs in every arm; only the alignment arm runs probes.
        with observational_state(model):
            self.batches = fixed_probe_batches(data, self.protocol, self.directory)

    def bind_training_state(self, *, optimizer, scheduler, objective, data):
        self.optimizer, self.scheduler, self.objective = optimizer, scheduler, objective

    def snapshot(self, epoch, step):
        return {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(), 'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'loader_generator': self.data.train_loader.generator.get_state(),
                'epoch': epoch, 'completed_updates': step,
                'batches_consumed_in_epoch': step % len(self.data.train_loader),
                'training_seed': self.config['training']['seed']}

    def __call__(self, *, model, epoch, global_step):
        if global_step not in self.protocol['checkpoint_steps']:
            return
        state = self.snapshot(epoch, global_step)
        if global_step == 100:
            hashes = {key: state_digest(value) for key, value in state.items()}
            reference = self.shared / 'common_prefix_hashes.json'
            if self.arm == 'baseline':
                clip_train.write_json(reference, hashes)
            elif json.loads(reference.read_text()) != hashes:
                raise AssertionError('Native prefix model/optimizer/RNG differs across arms')
            clip_train.write_json(self.directory / 'prefix_hashes.json', hashes)
        if self.arm == 'alignment_gated' and global_step in self.protocol['alignment_refresh_steps']:
            before = state_digest(state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            measured = measure_alignment(model, self.batches, self.config, self.protocol['probe_branch_seed'])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            measured.update(completed_updates=global_step, seconds=time.perf_counter()-started)
            if state_digest(self.snapshot(epoch, global_step)) != before:
                raise AssertionError('Gate probe mutated training state')
            self.objective.alignment = measured['mean_cosine']
            measured['active_until_next_refresh'] = gate_active(self.arm, global_step,
                self.protocol, measured['mean_cosine'])
            self.probes.append(measured)
            print('GATE_PROBE:', json.dumps(measured), flush=True)
        state['gate_state'] = {'alignment': self.objective.alignment, 'probes': self.probes,
                               'objective_decisions': self.objective.decisions}
        state['arm'] = self.arm
        save_hybrid_checkpoint(self.shared.parent, state,
            seed=self.config['training']['seed'], arm=self.arm, step=global_step)
        self.write_audit()
        self.sync()  # Synchronous, fail closed; all checkpoint files are closed.

    def write_audit(self):
        clip_train.write_json(self.directory / 'gate_audit.json', {
            'arm': self.arm, 'probes': self.probes, 'decisions': self.objective.decisions,
            'active_updates': sum(d['active'] for d in self.objective.decisions)})


def verify_decisions(arm, audit, protocol):
    decisions, probes = audit['decisions'], audit['probes']
    if [d['objective_step'] for d in decisions] != list(range(1001)):
        raise AssertionError('Missing/duplicated objective updates')
    expected = protocol['alignment_refresh_steps'] if arm == 'alignment_gated' else []
    if [r['completed_updates'] for r in probes] != expected:
        raise AssertionError('Wrong gate refresh steps')
    for probe in probes:
        values = probe['batch_cosines']
        if len(values) != 16 or not all(math.isfinite(v) for v in values):
            raise AssertionError('Incomplete/nonfinite probe')
        if probe['mean_cosine'] != statistics.fmean(values):
            raise AssertionError('Wrong probe aggregation')
    for decision in decisions:
        previous = [r for r in probes if r['completed_updates'] <= decision['objective_step']]
        value = previous[-1]['mean_cosine'] if previous else None
        if decision['active'] != gate_active(arm, decision['objective_step'], protocol, value):
            raise AssertionError('Actual gate differs from frozen rule')
    if audit['active_updates'] != sum(d['active'] for d in decisions):
        raise AssertionError('Wrong active-update count')


def summarize(directory, protocol):
    rows = []
    for seed in protocol['training_seeds']:
        for arm in ARMS:
            path = directory / f'seed_{seed}' / arm
            summary = json.loads((path / 'training/summary.json').read_text())
            if summary['execution']['completed_updates'] != 1001:
                raise AssertionError('Incomplete arm')
            audit = json.loads((path / 'gate_audit.json').read_text())
            verify_decisions(arm, audit, protocol)
            evaluation = summary['final']['evaluation']
            canonical = evaluation['canonical_retrieval']
            rows.append({'seed': seed, 'arm': arm,
                'canonical_avg_r1_percent': statistics.fmean(canonical[d]['r_at_1']
                    for d in ['text_to_image', 'image_to_text']),
                'species_top1_percent': 100 * evaluation['species']['top_1_accuracy'],
                'active_updates': audit['active_updates'],
                'probe_seconds': sum(p['seconds'] for p in audit['probes']),
                'wall_seconds': json.loads((path / 'timing.json').read_text())['wall_seconds']})
    differences = []
    for seed in protocol['training_seeds']:
        aligned = next(r for r in rows if r['seed'] == seed and r['arm'] == 'alignment_gated')
        for comparator in ['baseline', 'step_gated', 'always_on']:
            other = next(r for r in rows if r['seed'] == seed and r['arm'] == comparator)
            differences.append({'seed': seed, 'comparator': comparator,
                'canonical_r1_difference_pp': aligned['canonical_avg_r1_percent']-other['canonical_avg_r1_percent'],
                'species_top1_difference_pp': aligned['species_top1_percent']-other['species_top1_percent']})
    report = {'rows': rows, 'per_seed_differences': differences,
        'mean_alignment_minus_comparator_r1_pp': {arm: statistics.fmean(
            r['canonical_r1_difference_pp'] for r in differences if r['comparator'] == arm)
            for arm in ['baseline', 'step_gated', 'always_on']},
        'interpretation': 'Exploratory rollout on reused CUB test data and seeds. Not exposure/compute matched; no novelty or significance claim.'}
    clip_train.write_json(directory / 'comparison.json', report)
    return report


def run(directory, sync):
    protocol = load_protocol()
    directory = Path(directory)
    clip_train.write_json(directory / 'protocol.json', protocol)
    # The unmodified trainer also writes latest/best files. Reuse one local-only
    # scratch directory across arms so these cannot multiply or enter Drive.
    scratch = Path(tempfile.mkdtemp(prefix='.otco_gated_work_', dir=directory.parent)) / 'checkpoints'
    clip_train.write_json(directory / 'checkpoint_retention.json', {
        'policy': 'hybrid_v1', 'permanent_checkpoints': required_paths(),
        'rolling_checkpoint': 'rolling_checkpoint.pt',
        'rolling_refresh_steps': protocol['checkpoint_steps'],
        'local_only_trainer_scratch': str(scratch),
        'best_epoch_weights_retained': False,
        'automatic_mid_epoch_resume': False})
    sync()
    for seed in protocol['training_seeds']:
        shared = directory / f'seed_{seed}'
        shared.mkdir(exist_ok=False)
        for arm in ARMS:
            path = shared / arm
            path.mkdir()
            config = clip_train.load_training_config(ROOT / protocol['baseline_config'])
            config['experiment']['seed'] = config['training']['seed'] = seed
            config['experiment']['name'] = f'clip_gated_training_{seed}_{arm}'
            config['diagnostics']['separate_projection_gradient_steps'] = 0
            observer = RolloutObserver(path, shared, arm, protocol, sync)
            print(f'ROLLOUT: seed={seed} arm={arm}', flush=True)
            started = time.perf_counter()
            summary = clip_train.run(config, output_directory=path / 'training',
                checkpoint_directory=scratch, observer=observer,
                stop_after_epochs=13, objective_factory=lambda _: GatedObjective(arm, protocol))
            observer.write_audit()
            if summary['execution']['completed_updates'] != 1001:
                raise AssertionError('Incomplete rollout')
            verify_decisions(arm, json.loads((path / 'gate_audit.json').read_text()), protocol)
            clip_train.write_json(path / 'timing.json', {'wall_seconds': time.perf_counter()-started,
                'includes_evaluation_and_synchronous_backup': True})
            sync()
            del observer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report = summarize(directory, protocol)
    clip_train.write_json(directory / 'completion.json', {'status': 'complete',
        'seeds': protocol['training_seeds'], 'arms': ARMS, 'trajectories': 12,
        'required_state_checkpoints': 24, 'rolling_checkpoint_files': 1,
        'checkpoint_retention': 'hybrid_v1', 'rules_refitted': False})
    sync()
    return report
