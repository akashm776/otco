"""Two new native training trajectories; fixed paired probes, no curriculum tuning."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from colabs.backup_clip_run import checksum
from src import clip_train, clip_paired_updates as paired
from src.clip_early_pulse import state_digest
from src.clip_gradient_stages import ROOT, observational_state


def protocols(seed, source_name):
    study = yaml.safe_load((ROOT / 'configs/clip_paired_seed_replication.yaml').read_text())
    if seed not in study['training_seeds'] or seed == study['reference_training_seed']:
        raise ValueError('Unexpected replication seed')
    config = clip_train.load_training_config(ROOT / study['baseline_config'])
    config['experiment']['seed'] = config['training']['seed'] = seed
    config['diagnostics']['separate_projection_gradient_steps'] = 0
    protocol = yaml.safe_load((ROOT / study['paired_protocol']).read_text())
    protocol.update(source_run=source_name, training_seed=seed, diagnostic_caption_seed=42,
                    reference_run=protocol['source_run'])
    audit = json.loads((ROOT / 'experiment_results/clip_paired_updates_2026-09/audit.json').read_text())
    protocol['fixed_input_sha256'] = {name: audit['files'][name]['sha256'] for name in
                                     ['training_batches.json', 'heldout_partitions.json']}
    return study, config, protocol


def fixed_calibration_exclusions(source_indices, study, protocol):
    """Recreate only the old selection; never recalibrate on the new model."""
    size = protocol['batch_size']
    positions = np.random.default_rng(study['original_calibration_selection_seed']).permutation(len(source_indices))
    selected = positions[:study['original_calibration_batches'] * size].reshape(-1, size)
    return {'coefficients': protocol['coefficients'],
            'batches': [{'source_indices': [source_indices[int(i)] for i in batch]} for batch in selected],
            'recalibrated': False, 'selection_seed': study['original_calibration_selection_seed'],
            'coefficient_source_run': protocol['reference_run']}


class CheckpointObserver:
    def __init__(self, source, study, protocol):
        self.source, self.study, self.protocol = Path(source), study, protocol
        self.seen = []

    def initialize(self, *, model, processor, data, device, config, total_steps):
        if total_steps != self.study['scheduler_horizon_updates'] or len(data.train_loader) != 77:
            raise AssertionError('Changed training/scheduler horizon')
        self.data = data
        self.holdout = json.loads((ROOT / config['dataset']['diagnostic_holdout_indices']).read_text())
        if len(self.holdout) != 1024 or set(self.holdout) & set(data.train_dataset.source_indices):
            raise AssertionError('Invalid holdout exclusion')
        with observational_state(model):
            self.batches = paired.cache_heldout(processor, data, self.holdout)
        self.calibration = fixed_calibration_exclusions(data.train_dataset.source_indices, self.study, self.protocol)
        diagnostic = self.source / 'results/baseline/diagnostics'
        diagnostic.mkdir(parents=True, exist_ok=True)
        clip_train.write_json(diagnostic / 'diagnostic_holdout_indices.json', self.holdout)

    def bind_training_state(self, *, optimizer, scheduler, objective, data):
        self.optimizer, self.scheduler = optimizer, scheduler

    def __call__(self, *, model, epoch, global_step):
        if global_step % 25 == 0:
            print(f'[seed {self.protocol["training_seed"]}] baseline updates {global_step}/1001', flush=True)
        if global_step not in self.study['checkpoint_steps']:
            return
        # No training-loader iteration, optimizer mutation, or global RNG drift.
        with observational_state(model):
            if global_step == 100:
                state = {'model': model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                         'scheduler': self.scheduler.state_dict(), 'completed_updates': 100,
                         'training_seed': self.protocol['training_seed']}
                target = self.source / 'checkpoints/baseline/common_step_100.pt'
                torch.save(state, target)
                clip_train.write_json(self.source / 'results/prefix_hashes.json',
                                      {key: state_digest(value) for key, value in state.items()})
                # Backup completion marker only after the checkpoint has closed.
                clip_train.write_json(self.source / 'results/calibration.json', self.calibration)
            encoded = paired.encode_cached(model, self.batches)
            stage = self.source / f'results/baseline/diagnostics/{global_step:06d}_pulse_{global_step}'
            stage.mkdir(parents=True, exist_ok=False)
            clip_train.write_json(stage / 'report.json', {'completed_updates': global_step,
                'training_seed': self.protocol['training_seed'], 'features_and_scale_sha256': paired.feature_hash(encoded)})
        self.seen.append(global_step)
        print(f'[seed {self.protocol["training_seed"]}] saved checkpoint reference at {global_step}', flush=True)


def source_inventory(source):
    files = {}
    for path in sorted(source.rglob('*')):
        if path.is_file() and path.name != 'backup_manifest.json':
            files[str(path.relative_to(source))] = {'bytes': path.stat().st_size, 'sha256': checksum(path)}
    clip_train.write_json(source / 'backup_manifest.json', {'verified_files': files,
        'scope': 'Local closed-file integrity inventory; no external backup'})


def train_seed(source, seed):
    source = Path(source)
    study, config, protocol = protocols(seed, source.name)
    if source.exists():
        raise FileExistsError('Fresh seed directory required')
    source.mkdir(parents=True)
    observer = CheckpointObserver(source, study, protocol)
    summary = clip_train.run(config, output_directory=source / 'results/baseline/training',
        checkpoint_directory=source / 'checkpoints/baseline', observer=observer,
        stop_after_epochs=study['stop_after_epochs'])
    if observer.seen != [100, 1001] or summary['execution']['completed_updates'] != 1001:
        raise AssertionError('Incomplete baseline trajectory')
    clip_train.write_json(source / 'results/baseline/completion.json',
                         {'status': 'complete', 'training_seed': seed, 'completed_updates': 1001})
    del observer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    source_inventory(source)
    return protocol


def aggregate(output, seeds):
    """Preserve seed-level results; never treat 48 batches as 48 seeds."""
    records = []
    reference = ROOT / 'experiment_results/clip_paired_updates_2026-09'
    for seed, directory in [(42, reference)] + [(seed, output / f'seed_{seed}/results/paired') for seed in seeds]:
        completion = json.loads((directory / 'completion.json').read_text())
        if completion.get('status') != 'complete' or completion.get('rows') != 96:
            raise AssertionError('Incomplete paired seed')
        for row in json.loads((directory / 'summary.json').read_text()):
            records.append({'training_seed': seed, 'previous_reference_run': seed == 42, **row})
    clip_train.write_json(output / 'results/per_seed_summary.json', records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, step in zip(axes, [100, 1001]):
        for arm, color in [('uniform_top8', 'tab:orange'), ('hardest_real', 'tab:blue')]:
            selected = [r for r in records if r['checkpoint_step'] == step and r['arm'] == arm]
            ax.scatter([str(r['training_seed']) for r in selected],
                       [r['mean_incremental_heldout_loss'] * 1e6 for r in selected], marker='o', color=color, label=arm)
        ax.axhline(0, color='black', lw=.8)
        ax.set(title=f'{step:,} updates', xlabel='Training seed (42 = earlier reference)', ylabel='Mean extra held-out loss × 10⁻⁶ (lower better)')
    axes[0].legend()
    fig.suptitle('Seed robustness: each point averages 16 fixed paired inputs; no confidence intervals')
    for suffix in ['png', 'svg']:
        fig.savefig(output / f'results/seed_comparison.{suffix}', dpi=180)
    plt.close(fig)
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError('Choose a fresh run directory')
    (output / 'results').mkdir(parents=True)
    study = yaml.safe_load((ROOT / 'configs/clip_paired_seed_replication.yaml').read_text())
    clip_train.write_json(output / 'results/protocol.json', study)
    for seed in study['training_seeds']:
        source = output / f'seed_{seed}'
        print(f'=== START SEED {seed}: native baseline training ===', flush=True)
        protocol = train_seed(source, seed)
        print(f'=== SEED {seed}: 96 paired comparisons ===', flush=True)
        paired.run(source, source / 'results/paired', protocol)
        clip_train.write_json(output / 'results/progress.json', {'last_completed_seed': seed})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    records = aggregate(output, study['training_seeds'])
    clip_train.write_json(output / 'results/completion.json', {'status': 'complete', 'new_training_seeds': study['training_seeds'],
        'new_branch_rows': 192, 'reference_seed': 42, 'seed_level_summary_rows': len(records)})
    print('COMPLETE: both new seeds; 192 new paired branches.', flush=True)


if __name__ == '__main__':
    main()
