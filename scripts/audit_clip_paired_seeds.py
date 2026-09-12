"""Reproduce the completed two-seed ZIP audit and export caption-free evidence."""

import argparse
import tempfile
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as stats
import zipfile
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = 'clip_paired_seeds_20260912T050100_795138Z'
ARCHIVE_SHA256 = '6a14b339686712ef1f9d8354b2200627642e79a892b681a1a6db32362f7f996e'
REF = ROOT / 'experiment_results/clip_paired_updates_2026-09'
PARTS = ['shuffle_seed_42', 'shuffle_seed_123', 'shuffle_seed_4242']
ARMS = ['baseline', 'uniform_top8', 'hardest_real']
read = lambda path: json.loads(path.read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
mean = lambda row, parts=PARTS: stats.fmean(x for name in parts for x in row[name])
def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def audit_extracted(DATA, ARCHIVE):
    manifest = read(DATA / 'results/run_manifest.json')
    assert manifest['status'] == 'complete' and manifest['training_seeds'] == [123, 456]
    assert manifest['source_commit'] == '2c4c610576c5167114bb6778cef526f0a8d64df8'
    assert manifest['packages'] == read(REF / 'run_manifest.json')['packages']
    completion = read(DATA / 'results/completion.json')
    assert completion['status'] == 'complete' and completion['new_branch_rows'] == 192
    original_audit = read(REF / 'audit.json')
    original_protocol = read(REF / 'protocol.json')
    combined = read(DATA / 'results/per_seed_summary.json')
    assert len(combined) == 12
    details = []
    identities = []
    for seed in [42, 123, 456]:
        directory = REF if seed == 42 else DATA / f'seed_{seed}/results/paired'
        protocol = read(directory / 'protocol.json')
        states = read(directory / 'checkpoint_provenance.json')
        state_map = {r['step']: r for r in states}
        assert len(states) == 2 and set(state_map) == {100, 1001}
        rows = [json.loads(line) for line in (directory / 'paired_updates.jsonl').read_text().splitlines()]
        lookup = {(r['checkpoint_step'], r['trial'], r['arm']): r for r in rows}
        assert len(rows) == len(lookup) == 96
        assert set(lookup) == set(itertools.product([100, 1001], range(16), ARMS))
        assert protocol['coefficients'] == original_protocol['coefficients']
        for state in states:
            for key in ['no_update_reencoding_exact', 'native_update_and_evaluation_replay_exact',
                        'source_state_immutable', 'frozen_parameters_unchanged']:
                assert state[key] is True
            assert state['optimizer_reset_checks'] == 48
            identities.append((seed, state['step'], state['features_sha256']))
        if seed != 42:
            source = DATA / f'seed_{seed}'
            inventory = read(source / 'backup_manifest.json')['verified_files']
            for relative, entry in inventory.items():
                file = source / relative
                if not relative.endswith('.pt'):
                    assert digest(file) == entry['sha256']
            config = yaml.safe_load((source / 'results/baseline/training/resolved_config.yaml').read_text())
            assert config['training']['seed'] == config['experiment']['seed'] == seed
            assert config['ot']['enabled'] is False and config['training']['epochs'] == 50
            assert read(source / 'results/baseline/training/summary.json')['execution'] == dict(
                completed_epochs=13, completed_updates=1001, scheduler_horizon_updates=3850, planned_short_run=True)
            metrics = [json.loads(line) for line in (source / 'results/baseline/training/metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in metrics] == list(range(14))
            assert [r['global_step'] for r in metrics] == [i*77 for i in range(14)]
            assert protocol['training_seed'] == seed and protocol['diagnostic_caption_seed'] == 42
            for name in ['training_batches.json', 'heldout_partitions.json']:
                assert digest(directory / name) == original_audit['files'][name]['sha256']
            holdout = read(source / 'results/baseline/diagnostics/diagnostic_holdout_indices.json')
            assert holdout == read(ROOT / 'configs/cub200_clip_diagnostic_holdout_indices.json')
            batches = read(directory / 'training_batches.json')
            selected = [i for b in batches for i in b['source_indices']]
            calibration = read(source / 'results/calibration.json')
            excluded = {i for b in calibration['batches'] for i in b['source_indices']}
            assert len(selected) == len(set(selected)) == 1024
            assert not set(selected) & set(holdout) and not set(selected) & excluded
            assert calibration['recalibrated'] is False and calibration['coefficients'] == original_protocol['coefficients']
            for step, state in state_map.items():
                relative = 'checkpoints/baseline/' + ('common_step_100.pt' if step == 100 else 'latest.pt')
                assert state['checkpoint_sha256'] == inventory[relative]['sha256']
                observed = read(source / f'results/baseline/diagnostics/{step:06d}_pulse_{step}/report.json')
                assert state['features_sha256'] == observed['features_and_scale_sha256']
        for row in rows:
            step, trial, arm = row['checkpoint_step'], row['trial'], row['arm']
            native = lookup[step, trial, 'baseline']
            assert set(row['heldout_loss']) == set(PARTS + ['sequential'])
            assert all(len(values) == 16 for values in row['heldout_loss'].values())
            close(row['heldout_mean'], mean(row['heldout_loss']))
            close(row['heldout_change'], row['heldout_mean'] - mean(state_map[step]['initial_heldout_loss']))
            close(row['incremental_heldout_loss'], row['heldout_mean'] - native['heldout_mean'])
            close(row['incremental_train_loss'], row['train_after'] - native['train_after'])
            close(row['train_before'], native['train_before'])
            close(row['coefficient'], original_protocol['coefficients'].get(arm, 0.))
            close(row['actual_update_difference_ratio'], row['actual_update_difference_norm'] / native['actual_update_norm'])
        summaries = read(directory / 'summary.json')
        assert len(summaries) == 4
        for summary in summaries:
            step, arm = summary['checkpoint_step'], summary['arm']
            chosen = [lookup[step, trial, arm] for trial in range(16)]
            values = [r['incremental_heldout_loss'] for r in chosen]
            close(summary['mean_incremental_heldout_loss'], stats.fmean(values))
            close(summary['median_incremental_heldout_loss'], stats.median(values))
            assert summary['range'] == [min(values), max(values)]
            assert summary['beneficial_trials'] == sum(v < 0 for v in values)
            for key in ['incremental_train_loss', 'full_gradient_alignment', 'projection_gradient_alignment',
                        'actual_update_difference_ratio', 'actual_update_cosine_to_native']:
                close(summary['mean_' + key], stats.fmean(r[key] for r in chosen))
            assert {'training_seed': seed, 'previous_reference_run': seed == 42, **summary} in combined
            partition_effects = {}
            for part in PARTS + ['sequential']:
                effects = [mean(r['heldout_loss'], [part]) - mean(lookup[step, r['trial'], 'baseline']['heldout_loss'], [part]) for r in chosen]
                partition_effects[part] = dict(mean=stats.fmean(effects), beneficial_trials=sum(e < 0 for e in effects))
            native_change = stats.fmean(lookup[step, trial, 'baseline']['heldout_change'] for trial in range(16))
            details.append(dict(training_seed=seed, checkpoint_step=step, arm=arm, **{
                'mean_incremental_heldout_loss': stats.fmean(values), 'beneficial_trials': sum(v < 0 for v in values),
                'mean_native_heldout_change': native_change,
                'mean_treatment_heldout_change': stats.fmean(r['heldout_change'] for r in chosen),
                'incremental_benefit_percent_of_abs_native_mean_change': -stats.fmean(values)/abs(native_change)*100,
                'mean_weighted_gradient_ratio': stats.fmean(r['weighted_gradient_ratio'] for r in chosen),
                'partition_effects': partition_effects}))
    assert len(set(value for _, _, value in identities)) == 6, 'Expected distinct seed/stage checkpoints'
    report = dict(run_id=RUN, zip_sha256=digest(ARCHIVE), zip_files=54, zip_crc_and_extracted_bytes_verified=True,
        new_rows_verified=192, historical_rows_verified=96, fixed_inputs_exact=True, different_seed_stage_features=True,
        same_package_versions_as_reference=True, provenance_checks_passed=True,
        checkpoint_note='Checkpoints not in ZIP: hashes matched inventory and runtime provenance, not independently re-encoded offline.',
        details=details)

    report['files'] = {str(p.relative_to(DATA)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                       for p in sorted(DATA.rglob('*')) if p.is_file()}
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    assert digest(args.archive) == ARCHIVE_SHA256, 'Unexpected source archive bytes'
    with tempfile.TemporaryDirectory(prefix='otco-seed-audit-') as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(args.archive) as bundle:
            names = bundle.namelist()
            assert bundle.testzip() is None
            assert len(names) == len(set(names)) == 54
            assert all(name.startswith(RUN + '/') and '..' not in Path(name).parts and not name.endswith('.pt') for name in names)
            bundle.extractall(extracted)
            assert all((extracted / name).read_bytes() == bundle.read(name) for name in names)
        data = extracted / RUN
        report = audit_extracted(data, args.archive)
        args.output_directory.mkdir(parents=True, exist_ok=False)
        for path in sorted(data.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json', '.jsonl', '.yaml'}:
                continue
            if path.name in {'training_batches.json', 'heldout_partitions.json', 'diagnostic_holdout_indices.json'}:
                continue  # Exact reference identities/partitions already committed; omit raw captions.
            destination = args.output_directory / path.relative_to(data)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        (args.output_directory / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key not in {'details', 'files'}}, indent=2))


if __name__ == '__main__':
    main()
