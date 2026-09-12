"""Intermediate paired-update map across existing seeds; no curriculum selection."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import yaml

from src import clip_train, clip_paired_updates as paired
from src.clip_early_pulse import state_digest
from src.clip_gradient_stages import ROOT, observational_state
from src.clip_paired_seed_replication import CheckpointObserver, source_inventory


def reference_directory(seed):
    if seed == 42:
        return ROOT / 'experiment_results/clip_paired_updates_2026-09'
    if seed in [123, 456]:
        return ROOT / f'experiment_results/clip_paired_seeds_2026-09/seed_{seed}/results/paired'
    raise ValueError('Use the existing three seed trajectories')


def protocols(seed, source_name):
    study = yaml.safe_load((ROOT / 'configs/clip_paired_intermediate.yaml').read_text())
    if seed not in study['training_seeds']:
        raise ValueError('Unexpected seed')
    config = clip_train.load_training_config(ROOT / study['baseline_config'])
    config['experiment']['seed'] = config['training']['seed'] = seed
    config['diagnostics']['separate_projection_gradient_steps'] = 0
    protocol = yaml.safe_load((ROOT / study['paired_protocol']).read_text())
    protocol.update(reference_run=protocol['source_run'], source_run=source_name, training_seed=seed,
                    diagnostic_caption_seed=42, checkpoint_steps=study['checkpoint_steps'],
                    endpoint_reference_directory=str(reference_directory(seed).relative_to(ROOT)),
                    endpoint_loss_atol=study['endpoint_loss_atol'])
    audit = json.loads((ROOT / 'experiment_results/clip_paired_updates_2026-09/audit.json').read_text())
    protocol['fixed_input_sha256'] = {name: audit['files'][name]['sha256'] for name in
                                     ['training_batches.json', 'heldout_partitions.json']}
    paired.validate_protocol(protocol)
    return study, config, protocol


def check_reference_state(actual, reference):
    return (actual['initial_state_sha256'] == reference['initial_state_sha256']
            and actual['features_sha256'] == reference['features_sha256']
            and actual['learning_rates'] == reference['learning_rates'])


class IntermediateObserver(CheckpointObserver):
    def __init__(self, source, study, protocol):
        super().__init__(source, study, protocol)
        self.references = {r['step']: r for r in json.loads(
            (ROOT / protocol['endpoint_reference_directory'] / 'checkpoint_provenance.json').read_text())}

    def __call__(self, *, model, epoch, global_step):
        if global_step % 25 == 0:
            print(f'[seed {self.protocol["training_seed"]}] baseline updates {global_step}/1001', flush=True)
        if global_step not in self.study['checkpoint_steps']:
            return
        with observational_state(model):
            initial = {'model': model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                       'scheduler': self.scheduler.state_dict()}
            encoded = paired.encode_cached(model, self.batches)
            actual = {'initial_state_sha256': state_digest(initial),
                      'features_sha256': paired.feature_hash(encoded),
                      'learning_rates': [group['lr'] for group in self.optimizer.param_groups]}
            stage = self.source / f'results/baseline/diagnostics/{global_step:06d}_pulse_{global_step}'
            stage.mkdir(parents=True, exist_ok=False)
            report = {'completed_updates': global_step, 'training_seed': self.protocol['training_seed'],
                      'features_and_scale_sha256': actual['features_sha256'], **actual}
            if global_step in self.references:
                reference = self.references[global_step]
                report.update(historical_endpoint_state_exact=check_reference_state(actual, reference),
                              expected_initial_state_sha256=reference['initial_state_sha256'],
                              expected_features_sha256=reference['features_sha256'])
            clip_train.write_json(stage / 'report.json', report)
            if global_step in self.references and not report['historical_endpoint_state_exact']:
                raise AssertionError(f'Seed {self.protocol["training_seed"]} endpoint {global_step} differs from historical state/features; stopping, not relaxing checks')
            if global_step < 1001:
                snapshot = {**initial, 'completed_updates': global_step, 'training_seed': self.protocol['training_seed']}
                torch.save(snapshot, self.source / paired.checkpoint_relative_path(global_step))
                hashes = {key: state_digest(value) for key, value in snapshot.items()}
                clip_train.write_json(stage / 'snapshot_hashes.json', hashes)
                if global_step == 100:
                    clip_train.write_json(self.source / 'results/prefix_hashes.json', hashes)
                    clip_train.write_json(self.source / 'results/calibration.json', self.calibration)
        self.seen.append(global_step)
        print(f'[seed {self.protocol["training_seed"]}] saved diagnostic state {global_step}', flush=True)


def train_seed(source, seed):
    source = Path(source)
    study, config, protocol = protocols(seed, source.name)
    if source.exists():
        raise FileExistsError('Fresh seed paths required; no partial replay resume')
    source.mkdir(parents=True)
    observer = IntermediateObserver(source, study, protocol)
    summary = clip_train.run(config, output_directory=source / 'results/baseline/training',
        checkpoint_directory=source / 'checkpoints/baseline', observer=observer,
        stop_after_epochs=study['stop_after_epochs'])
    if observer.seen != study['checkpoint_steps'] or summary['execution']['completed_updates'] != 1001:
        raise AssertionError('Missing intermediate checkpoints or incomplete trajectory')
    clip_train.write_json(source / 'results/baseline/completion.json',
                         {'status': 'complete', 'training_seed': seed, 'completed_updates': 1001,
                          'checkpoint_steps': study['checkpoint_steps']})
    del observer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    source_inventory(source)
    return protocol


def verify_seed(directory, study):
    read = lambda name: json.loads((directory / name).read_text())
    completion = read('completion.json')
    expected = {(step, trial, arm) for step in study['checkpoint_steps'] for trial in range(16)
                for arm in ['baseline', 'uniform_top8', 'hardest_real']}
    rows = [json.loads(line) for line in (directory / 'paired_updates.jsonl').read_text().splitlines()]
    if (completion.get('status') != 'complete' or completion.get('rows') != 240
            or completion.get('checkpoint_steps') != study['checkpoint_steps'] or len(rows) != 240
            or {(r['checkpoint_step'], r['trial'], r['arm']) for r in rows} != expected):
        raise AssertionError('Incomplete five-state paired seed')
    checks = read('endpoint_replay_checks.json')
    if len(checks) != 2 or sorted(r['step'] for r in checks) != [100, 1001]:
        raise AssertionError('Missing endpoint replay checks')
    for check in checks:
        if (check.get('passed') is not True or check.get('rows') != 48
                or check.get('loss_absolute_tolerance') != study['endpoint_loss_atol']
                or not 0 <= check.get('maximum_absolute_loss_difference', float('inf')) <= study['endpoint_loss_atol']):
            raise AssertionError('Historical endpoint loss replay failed')
    states = read('checkpoint_provenance.json')
    if len(states) != 5 or sorted(r['step'] for r in states) != study['checkpoint_steps']:
        raise AssertionError('Missing state provenance')
    for state in states:
        if state.get('optimizer_reset_checks') != 48 or not all(state.get(key) is True for key in
            ['no_update_reencoding_exact', 'native_update_and_evaluation_replay_exact', 'source_state_immutable', 'frozen_parameters_unchanged']):
            raise AssertionError('Invalid per-state reset checks')
    lookup = {(r['checkpoint_step'], r['trial'], r['arm']): r for r in rows}
    for row in rows:
        native = lookup[row['checkpoint_step'], row['trial'], 'baseline']
        if set(row['heldout_loss']) != {'sequential', 'shuffle_seed_42', 'shuffle_seed_123', 'shuffle_seed_4242'}:
            raise AssertionError('Changed held-out partitions')
        if any(len(values) != 16 or not all(math.isfinite(v) for v in values) for values in row['heldout_loss'].values()):
            raise AssertionError('Missing/nonfinite held-out batch losses')
        measured = paired.primary_mean(row['heldout_loss'], ['shuffle_seed_42', 'shuffle_seed_123', 'shuffle_seed_4242'])
        checks = [(row['heldout_mean'], measured),
                  (row['incremental_heldout_loss'], measured - native['heldout_mean']),
                  (row['incremental_train_loss'], row['train_after'] - native['train_after'])]
        if not all(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a,b in checks):
            raise AssertionError('Invalid paired loss arithmetic')
    return rows


def aggregate(output, study):
    summaries = []
    for seed in study['training_seeds']:
        directory = output / f'seed_{seed}/results/paired'
        rows = verify_seed(directory, study)
        seed_summary = json.loads((directory / 'summary.json').read_text())
        if len(seed_summary) != 10 or {(r['checkpoint_step'], r['arm']) for r in seed_summary} != {
            (step, arm) for step in study['checkpoint_steps'] for arm in ['uniform_top8', 'hardest_real']}:
            raise AssertionError('Missing or duplicated seed summaries')
        for row in seed_summary:
            values = [r['incremental_heldout_loss'] for r in rows if r['checkpoint_step'] == row['checkpoint_step'] and r['arm'] == row['arm']]
            if not math.isclose(sum(values)/16, row['mean_incremental_heldout_loss'], rel_tol=0, abs_tol=1e-12) or sum(v < 0 for v in values) != row['beneficial_trials']:
                raise AssertionError('Seed summary does not match paired rows')
            summaries.append({'training_seed': seed, 'historical_endpoint_replay': row['checkpoint_step'] in [100, 1001], **row})
        if len(rows) != 240:
            raise AssertionError('Unexpected number of seed rows')
    if len(summaries) != 30:
        raise AssertionError('Missing seed/stage/arm summaries')
    clip_train.write_json(output / 'results/per_seed_summary.json', summaries)
    plot(output / 'results', summaries)
    return summaries


def plot(directory, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, arm in zip(axes, ['uniform_top8', 'hardest_real']):
        for seed, color in [(42, '#0072B2'), (123, '#D55E00'), (456, '#009E73')]:
            rows = sorted([r for r in summaries if r['training_seed'] == seed and r['arm'] == arm], key=lambda r: r['checkpoint_step'])
            x = [r['checkpoint_step'] for r in rows]
            y = [r['mean_incremental_heldout_loss'] * 1e6 for r in rows]
            ax.plot(x, y, marker='o', color=color, label=f'Seed {seed}')
            ax.fill_between(x, [r['range'][0] * 1e6 for r in rows], [r['range'][1] * 1e6 for r in rows], color=color, alpha=.08)
        ax.axhline(0, color='black', lw=.8)
        ax.set(title=arm, xlabel='Completed native training updates', ylabel='Extra held-out loss vs native × 10⁻⁶ (lower better)', xticks=[100,250,500,750,1001])
    axes[0].legend()
    fig.suptitle('Paired-update usefulness across stages; lines connect measured points only\nShading: range of 16 fixed input batches, not confidence intervals')
    for suffix in ['png', 'svg']:
        fig.savefig(directory / f'intermediate_usefulness.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    (output / 'results').mkdir(parents=True)
    study = yaml.safe_load((ROOT / 'configs/clip_paired_intermediate.yaml').read_text())
    clip_train.write_json(output / 'results/protocol.json', study)
    for seed in study['training_seeds']:
        print(f'=== SEED {seed}: native trajectory replay and five checkpoints ===', flush=True)
        source = output / f'seed_{seed}'
        protocol = train_seed(source, seed)
        print(f'=== SEED {seed}: 240 paired branches (144 intermediate, 96 endpoint replays) ===', flush=True)
        paired.run(source, source / 'results/paired', protocol)
        verify_seed(source / 'results/paired', study)
        clip_train.write_json(output / 'results/progress.json', {'last_completed_seed': seed})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregate(output, study)
    clip_train.write_json(output / 'results/completion.json', {'status': 'complete', 'training_seeds': study['training_seeds'],
        'checkpoint_steps': study['checkpoint_steps'], 'total_branch_rows': 720,
        'new_intermediate_branch_rows': 432, 'endpoint_replay_branch_rows': 288})
    print('COMPLETE: 432 intermediate comparisons and 288 verified endpoint replays.', flush=True)


if __name__ == '__main__':
    main()
