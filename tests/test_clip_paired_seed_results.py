"""Guard the reported replication conclusions and byte-preserved numeric evidence."""

import hashlib
import itertools
import json
from pathlib import Path
import statistics

import pytest

from scripts.audit_clip_paired_seeds import close, mean


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'experiment_results/clip_paired_seeds_2026-09'
READ = lambda path: json.loads(path.read_text())


@pytest.mark.parametrize('seed', [123, 456])
def test_all_new_branch_losses_and_summaries_recompute(seed):
    directory = EVIDENCE / f'seed_{seed}/results/paired'
    rows = [json.loads(line) for line in (directory / 'paired_updates.jsonl').read_text().splitlines()]
    lookup = {(r['checkpoint_step'], r['trial'], r['arm']): r for r in rows}
    assert len(rows) == len(lookup) == 96
    assert set(lookup) == set(itertools.product([100, 1001], range(16), ['baseline', 'uniform_top8', 'hardest_real']))
    for row in rows:
        native = lookup[row['checkpoint_step'], row['trial'], 'baseline']
        close(row['heldout_mean'], mean(row['heldout_loss']))
        close(row['incremental_heldout_loss'], row['heldout_mean'] - native['heldout_mean'])
        close(row['incremental_train_loss'], row['train_after'] - native['train_after'])
    for summary in READ(directory / 'summary.json'):
        values = [lookup[summary['checkpoint_step'], trial, summary['arm']]['incremental_heldout_loss'] for trial in range(16)]
        close(summary['mean_incremental_heldout_loss'], statistics.fmean(values))
        assert summary['beneficial_trials'] == sum(value < 0 for value in values)
        assert summary['range'] == [min(values), max(values)]


def test_early_replication_but_mixed_later_signs_are_preserved():
    summaries = READ(EVIDENCE / 'results/per_seed_summary.json')
    assert len(summaries) == 12
    assert sum(s['previous_reference_run'] for s in summaries) == 4
    lookup = {(r['training_seed'], r['checkpoint_step'], r['arm']): r for r in summaries}
    for seed, later_beneficial in [(42, 0), (123, 15), (456, 3)]:
        early = lookup[seed, 100, 'uniform_top8']
        later = lookup[seed, 1001, 'uniform_top8']
        assert early['mean_incremental_heldout_loss'] < 0 and early['beneficial_trials'] == 16
        assert later['beneficial_trials'] == later_beneficial
        assert (later['mean_incremental_heldout_loss'] < 0) == (seed == 123)
        assert lookup[seed, 100, 'hardest_real']['mean_incremental_heldout_loss'] > 0
        assert lookup[seed, 1001, 'hardest_real']['mean_incremental_heldout_loss'] > 0


def test_relative_benefit_is_not_mislabeled_as_absolute_improvement():
    row = next(r for r in READ(EVIDENCE / 'audit.json')['details']
               if (r['training_seed'], r['checkpoint_step'], r['arm']) == (123, 1001, 'uniform_top8'))
    assert row['mean_native_heldout_change'] > row['mean_treatment_heldout_change'] > 0
    assert row['mean_incremental_heldout_loss'] < 0


def test_retained_files_match_source_inventory_and_contain_no_captions_or_tensors():
    inventory = READ(EVIDENCE / 'audit.json')['files']
    retained = [path for path in EVIDENCE.rglob('*') if path.is_file() and path.name not in {'audit.json', 'README.md'}]
    assert len(retained) > 20
    for path in retained:
        entry = inventory[str(path.relative_to(EVIDENCE))]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        assert path.stat().st_size == entry['bytes']
        assert path.suffix != '.pt' and '"captions"' not in path.read_text()


def test_historical_reference_not_counted_as_new_training():
    completion = READ(EVIDENCE / 'results/completion.json')
    assert completion['new_training_seeds'] == [123, 456] and completion['new_branch_rows'] == 192
    reference = READ(ROOT / 'experiment_results/clip_paired_updates_2026-09/summary.json')
    historical = [{key: value for key, value in r.items() if key not in {'training_seed', 'previous_reference_run'}}
                  for r in READ(EVIDENCE / 'results/per_seed_summary.json') if r['training_seed'] == 42]
    assert historical == reference
