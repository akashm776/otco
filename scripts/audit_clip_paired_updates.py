"""Audit the completed paired-update ZIP and export compact, caption-free evidence."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import zipfile


RUN = 'clip_paired_updates_20260912T042828_516737Z'
ARCHIVE_SHA256 = '8045ff1d9cae92e0685c9e6381479826b3f00ea216393ea3d7da71d0bc8957c4'
PRIMARY = ['shuffle_seed_42', 'shuffle_seed_123', 'shuffle_seed_4242']
ARMS = ['baseline', 'uniform_top8', 'hardest_real']


def mean_loss(losses, partitions=PRIMARY):
    return statistics.fmean(x for name in partitions for x in losses[name])


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def audit(archive):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA256, 'Unexpected archive bytes'
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist()) == len(set(bundle.namelist())) == 13
        blobs = {Path(name).name: bundle.read(name) for name in bundle.namelist()}
    read = lambda name: json.loads(blobs[name])
    rows = [json.loads(line) for line in blobs['paired_updates.jsonl'].splitlines()]
    completion, protocol = read('completion.json'), read('protocol.json')
    manifest, provenance = read('run_manifest.json'), read('checkpoint_provenance.json')
    assert manifest['run_id'] == RUN and manifest['status'] == 'complete'
    assert manifest['source_commit'] == 'd22b9d7cec94d5b044b814d774911cab5e37d560'
    assert completion == dict(status='complete', checkpoint_steps=[100, 1001], trials_per_checkpoint=16, rows=96)
    assert protocol['primary_partitions'] == PRIMARY
    expected = set(itertools.product([100, 1001], range(16), ARMS))
    lookup = {(r['checkpoint_step'], r['trial'], r['arm']): r for r in rows}
    assert len(rows) == len(lookup) == 96 and set(lookup) == expected
    states = {p['step']: p for p in provenance}
    assert len(provenance) == 2 and set(states) == {100, 1001}
    for state in states.values():
        assert state['optimizer_reset_checks'] == 48
        for key in ['no_update_reencoding_exact', 'native_update_and_evaluation_replay_exact',
                    'source_state_immutable', 'frozen_parameters_unchanged']:
            assert state[key] is True
    for row in rows:
        native = lookup[row['checkpoint_step'], row['trial'], 'baseline']
        assert set(row['heldout_loss']) == set(PRIMARY + ['sequential'])
        assert all(len(values) == 16 for values in row['heldout_loss'].values())
        close(row['heldout_mean'], mean_loss(row['heldout_loss']))
        close(row['heldout_change'], row['heldout_mean'] - mean_loss(states[row['checkpoint_step']]['initial_heldout_loss']))
        close(row['incremental_heldout_loss'], row['heldout_mean'] - native['heldout_mean'])
        close(row['incremental_train_loss'], row['train_after'] - native['train_after'])
        close(row['train_before'], native['train_before'])
        close(row['coefficient'], protocol['coefficients'].get(row['arm'], 0.))
        close(row['actual_update_difference_ratio'], row['actual_update_difference_norm'] / native['actual_update_norm'])
        close(row['actual_update_norm'] ** 2, sum(x*x for x in row['actual_update_norm_by_group'].values()))
        assert -1.00000001 <= row['actual_update_cosine_to_native'] <= 1.00000001
    summaries = read('summary.json')
    assert len(summaries) == 4
    assert {(s['checkpoint_step'], s['arm']) for s in summaries} == set(itertools.product([100, 1001], ARMS[1:]))
    details = []
    for summary in summaries:
        step, arm = summary['checkpoint_step'], summary['arm']
        selected = [lookup[step, trial, arm] for trial in range(16)]
        values = [r['incremental_heldout_loss'] for r in selected]
        close(summary['mean_incremental_heldout_loss'], statistics.fmean(values))
        close(summary['median_incremental_heldout_loss'], statistics.median(values))
        assert summary['beneficial_trials'] == sum(x < 0 for x in values)
        assert summary['range'] == [min(values), max(values)]
        for key in ['incremental_train_loss', 'full_gradient_alignment', 'projection_gradient_alignment',
                    'actual_update_difference_ratio', 'actual_update_cosine_to_native']:
            close(summary['mean_' + key], statistics.fmean(r[key] for r in selected))
        partitions = {}
        for partition in PRIMARY + ['sequential']:
            effects = [mean_loss(r['heldout_loss'], [partition]) -
                       mean_loss(lookup[step, r['trial'], 'baseline']['heldout_loss'], [partition]) for r in selected]
            partitions[partition] = dict(mean=statistics.fmean(effects), beneficial_trials=sum(x < 0 for x in effects),
                                         range=[min(effects), max(effects)])
        details.append(dict(checkpoint_step=step, arm=arm, partition_effects=partitions,
            mean_initial_heldout_loss=mean_loss(states[step]['initial_heldout_loss']),
            mean_native_heldout_change=statistics.fmean(lookup[step, trial, 'baseline']['heldout_change'] for trial in range(16)),
            mean_treatment_heldout_change=statistics.fmean(r['heldout_change'] for r in selected),
            mean_native_train_change=statistics.fmean(lookup[step, trial, 'baseline']['train_after'] - r['train_before'] for trial, r in enumerate(selected)),
            mean_treatment_train_change=statistics.fmean(r['train_after'] - r['train_before'] for r in selected),
            mean_weighted_gradient_ratio=statistics.fmean(r['weighted_gradient_ratio'] for r in selected)))
    batches = read('training_batches.json')
    indices = [i for batch in batches for i in batch['source_indices']]
    assert len(batches) == 16 and all(len(batch['source_indices']) == 64 for batch in batches)
    assert len(indices) == len(set(indices)) == 1024
    report = dict(run_id=RUN, archive_sha256=digest, zip_crc_verified=True, unique_branch_rows=96,
        recomputed_branch_losses_and_summaries=True, unique_training_examples=1024,
        runtime_provenance_checks_passed=True,
        scope='Offline arithmetic/integrity audit; GPU replay and checkpoint checks are recorded by the runner, not rerun here.',
        files={name: dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()) for name, blob in blobs.items()},
        details=details)
    return report, blobs, [batch['source_indices'] for batch in batches]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    report, blobs, indices = audit(args.archive)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    for name in ['paired_updates.jsonl', 'summary.json', 'checkpoint_provenance.json', 'protocol.json',
                 'run_manifest.json', 'completion.json', 'heldout_partitions.json']:
        (args.output_directory / name).write_bytes(blobs[name])
    (args.output_directory / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_directory / 'training_source_indices.json').write_text(json.dumps(indices, indent=2) + '\n')
    print(json.dumps(report['details'], indent=2))


if __name__ == '__main__':
    main()
