"""Replay the prospective seeds and observe a fixed, disjoint CUB test pool.

The extra evaluation is observational, never a training objective or gate.
The test split was previously used for retrieval monitoring, so this is NOT
an untouched-test confirmation, architecture transfer, or a curriculum trial.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

import torch

from src import clip_train, clip_paired_updates as paired
from src import clip_usefulness_prospective as prospective
from src.clip_geometry_diagnostic import CUBCLIPDiagnosticDataset, stratified_holdout_indices
from src.clip_gradient_stages import ROOT
from src.clip_negative_gradient_geometry_randomized import build_partition_conditions, EXPECTED_PARTITIONS

REFERENCE = 'experiment_results/clip_usefulness_prospective_2026-09'
DESIGN = dict(training_seeds=prospective.SEEDS, checkpoint_steps=prospective.STEPS,
              selection_seed=20260913, pool_size=1024, batch_size=64,
              pool='CUB test split, canonical first captions',
              primary_partitions=prospective.PARTS, near_zero_absolute_band=1e-6,
              rules_refitted=False, gate_driven_training=False,
              historical_loss_atol=1e-8,
              interpretation='Evaluation-pool transfer; test split previously used for retrieval monitoring.')


def read(path):
    return json.loads(Path(path).read_text())


def select_pool(data):
    validation = data.validation_canonical_dataset
    train_keys = {g.image_key for g in data.train_dataset.grouped_split.groups}
    groups = validation.grouped_split.groups
    indices = stratified_holdout_indices(validation.species_ids, fraction=1.,
                                        seed=DESIGN['selection_seed'], max_samples=1024)
    keys = [groups[i].image_key for i in indices]
    if len(indices) != 1024 or len(set(keys)) != 1024 or set(keys) & train_keys:
        raise AssertionError('Transfer pool must contain 1024 unique images disjoint from the entire train split')
    return indices


def transfer_states(rows, seed):
    lookup = {(r['checkpoint_step'], r['trial'], r['arm']): r for r in rows}
    expected = {(s,t,a) for s in prospective.STEPS for t in range(16)
                for a in ['baseline', 'uniform_top8', 'hardest_real']}
    if len(rows) != 240 or set(lookup) != expected:
        raise AssertionError('Incomplete or duplicate transfer branches')
    states = []
    for row in rows:
        losses = row['heldout_loss']
        if set(losses) != set(prospective.PARTS + ['sequential']) or any(
                len(v) != 16 or not all(math.isfinite(x) for x in v) for v in losses.values()):
            raise AssertionError('Invalid transfer losses')
        native = lookup[row['checkpoint_step'], row['trial'], 'baseline']
        measured = paired.primary_mean(losses, prospective.PARTS)
        for actual, expected_value in [
            (row['heldout_mean'], measured),
            (row['incremental_heldout_loss'], measured - native['heldout_mean']),
            (row['heldout_change'], measured - row['initial_heldout_mean'])]:
            if not math.isfinite(actual) or not math.isclose(actual, expected_value, rel_tol=0, abs_tol=1e-12):
                raise AssertionError('Invalid transfer arithmetic')
    for step in prospective.STEPS:
        chosen = [lookup[step,t,'uniform_top8'] for t in range(16)]
        states.append(dict(training_seed=seed, checkpoint_step=step,
            full_gradient_alignment=stats.fmean(r['full_gradient_alignment'] for r in chosen),
            effect=stats.fmean(r['incremental_heldout_loss'] for r in chosen),
            beneficial_trials=sum(r['incremental_heldout_loss'] < 0 for r in chosen),
            native_heldout_change=stats.fmean(lookup[step,t,'baseline']['heldout_change'] for t in range(16)),
            synthetic_heldout_change=stats.fmean(r['heldout_change'] for r in chosen)))
    return states


class TransferObserver:
    def __init__(self, output, seed):
        self.output, self.seed = Path(output), seed
        self.output.mkdir(parents=True, exist_ok=False)
        self.rows, self.initials, self.native = [], {}, {}

    def initialize(self, processor, data):
        indices = select_pool(data)
        validation = data.validation_canonical_dataset
        diagnostic = CUBCLIPDiagnosticDataset(validation.grouped_split, validation.species_ids, indices)
        self.batches, metadata = [], []
        for start in range(0, 1024, 64):
            records = [diagnostic[i] for i in range(start, start+64)]
            self.batches.append(dict(processor(images=[r['image'] for r in records],
                text=[r['caption'] for r in records], padding=True, truncation=True, return_tensors='pt')))
            metadata.extend({k:r[k] for k in ['source_index','image_key','species_id','caption']} for r in records)
        self.conditions, partitions = build_partition_conditions(EXPECTED_PARTITIONS, 1024, 64)
        clip_train.write_json(self.output/'pool.json', dict(design=DESIGN, records=metadata,
            disjoint_from_entire_cub_train_split=True))
        clip_train.write_json(self.output/'partitions.json', partitions)

    def evaluate(self, model):
        return paired.partition_losses(paired.encode_cached(model, self.batches), self.conditions)

    def start_checkpoint(self, model, step):
        encoded = paired.encode_cached(model, self.batches)
        digest = paired.feature_hash(encoded)
        if paired.feature_hash(paired.encode_cached(model, self.batches)) != digest:
            raise AssertionError('Transfer no-update encoding is not exact')
        self.initials[str(step)] = dict(features_sha256=digest,
            heldout_loss=paired.partition_losses(encoded, self.conditions), no_update_reencoding_exact=True)
        clip_train.write_json(self.output/'initial_states.json', self.initials)

    def record(self, row, losses):
        step, trial, arm = row['checkpoint_step'], row['trial'], row['arm']
        mean = paired.primary_mean(losses, prospective.PARTS)
        if arm == 'baseline':
            self.native[step,trial] = mean
        initial = paired.primary_mean(self.initials[str(step)]['heldout_loss'], prospective.PARTS)
        transferred = dict(row, heldout_loss=losses, heldout_mean=mean,
            initial_heldout_mean=initial, heldout_change=mean-initial,
            incremental_heldout_loss=mean-self.native[step,trial])
        self.rows.append(transferred)
        with (self.output/'paired_updates.jsonl').open('a') as handle:
            handle.write(json.dumps(transferred)+'\n')
        if trial == 15 and arm == 'hardest_real' and getattr(self, 'on_checkpoint', None) is not None:
            self.on_checkpoint()

    def finish(self, provenance):
        states = transfer_states(self.rows, self.seed)
        if len(provenance) != 5 or not all(p.get('native_update_and_evaluation_replay_exact')
                and p.get('source_state_immutable') and p.get('frozen_parameters_unchanged') for p in provenance):
            raise AssertionError('Underlying paired state checks did not pass')
        clip_train.write_json(self.output/'states.json', states)
        clip_train.write_json(self.output/'completion.json', dict(status='complete', rows=240,
            checkpoint_steps=prospective.STEPS, native_transfer_replay_exact=True))


def verify_completion(output):
    output = Path(output)
    if read(output/'results/design.json') != DESIGN:
        raise AssertionError('Transfer design changed')
    _, frozen = prospective.load_study()
    if read(output/'results/frozen_rules.json') != frozen:
        raise AssertionError('Frozen rules changed')
    states, pool_hashes = [], set()
    for seed in prospective.SEEDS:
        source = output/f'seed_{seed}'
        prospective.state_records(source, seed)
        reference_directory = ROOT/REFERENCE/f'seed_{seed}/results/paired'
        historical = {r['step']:r for r in read(reference_directory/'checkpoint_provenance.json')}
        provenance = read(source/'results/paired/checkpoint_provenance.json')
        for state in provenance:
            original = historical[state['step']]
            if any(state[k] != original[k] for k in
                   ['initial_state_sha256','features_sha256','learning_rates']):
                raise AssertionError('Historical state/features/LR do not match')
        replay_rows = [json.loads(line) for line in (source/'results/paired/paired_updates.jsonl').read_text().splitlines()]
        reference_rows = [json.loads(line) for line in (reference_directory/'paired_updates.jsonl').read_text().splitlines()]
        checks = read(source/'results/paired/endpoint_replay_checks.json')
        recomputed_checks = [paired.compare_endpoint_losses(replay_rows, reference_rows, step,
                            DESIGN['historical_loss_atol']) for step in prospective.STEPS]
        if checks != recomputed_checks:
            raise AssertionError('Missing historical replay checks')
        transfer = source/'results/transfer'
        marker = read(transfer/'completion.json')
        if marker != dict(status='complete', rows=240, checkpoint_steps=prospective.STEPS, native_transfer_replay_exact=True):
            raise AssertionError('Incomplete transfer')
        pool_hashes.add(hashlib.sha256((transfer/'pool.json').read_bytes()).hexdigest())
        rows = [json.loads(line) for line in (transfer/'paired_updates.jsonl').read_text().splitlines()]
        by_key = {(r['checkpoint_step'],r['trial'],r['arm']):r for r in replay_rows}
        for row in rows:
            original = by_key[row['checkpoint_step'],row['trial'],row['arm']]
            for key in ['coefficient','full_gradient_alignment','train_before','train_after',
                        'actual_update_norm','actual_update_difference_ratio']:
                if row.get(key) != original.get(key):
                    raise AssertionError('Transfer and original pool did not observe the same update')
        states.extend(transfer_states(rows, seed))
    if len(pool_hashes) != 1:
        raise AssertionError('Transfer examples/captions differ between seeds')
    if prospective.score(states, frozen) != read(output/'results/prediction_report.json'):
        raise AssertionError('Transfer scores do not reproduce')
    if read(output/'results/completion.json') != dict(status='complete', training_seeds=prospective.SEEDS,
            checkpoint_steps=prospective.STEPS, replay_branch_rows=720, transfer_branch_rows=720,
            evaluated_states=15, rules_refitted=False):
        raise AssertionError('Incomplete replay/transfer study')


def plot(output, old, new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13,4), constrained_layout=True)
    for ax, seed in zip(axes, prospective.SEEDS):
        for states, label in [(old, 'Original train-split holdout'), (new, 'CUB test-split pool')]:
            rows = sorted([r for r in states if r['training_seed']==seed], key=lambda r:r['checkpoint_step'])
            ax.plot([r['checkpoint_step'] for r in rows], [r['effect']*1e6 for r in rows], marker='o', label=label)
        ax.axhline(0, color='black', lw=.8)
        ax.axhspan(-1, 1, color='grey', alpha=.12)
        ax.set(title=f'Seed {seed}', xlabel='Completed updates', ylabel='Extra loss × 10⁻⁶ (lower better)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed-rule evaluation-pool transfer; lines connect measured states only')
    for suffix in ['png','svg']:
        fig.savefig(output/f'pool_transfer.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-directory', required=True)
    parser.add_argument('--backup-directory')
    args = parser.parse_args()
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    results = output/'results'
    results.mkdir()
    _, frozen = prospective.load_study()
    clip_train.write_json(results/'design.json', DESIGN)
    clip_train.write_json(results/'frozen_rules.json', frozen)
    mirror = None
    if args.backup_directory:
        from colabs.transfer_drive_backup import DriveMirror
        mirror = DriveMirror(output, args.backup_directory)
        mirror.sync()
    print('FROZEN TRANSFER DESIGN:', json.dumps(DESIGN), flush=True)
    old, new = [], []
    for seed in prospective.SEEDS:
        print(f'REPLAY SEED {seed}: training, 240 original branches and 240 extra pool observations', flush=True)
        source = output/f'seed_{seed}'
        protocol = prospective.train_seed(source, seed, on_checkpoint=mirror.sync if mirror else None)
        if mirror:
            mirror.sync()
        protocol.update(endpoint_reference_directory=f'{REFERENCE}/seed_{seed}/results/paired',
                        endpoint_loss_atol=DESIGN['historical_loss_atol'])
        observer = TransferObserver(source/'results/transfer', seed)
        observer.on_checkpoint = mirror.sync if mirror else None
        paired.run(source, source/'results/paired', protocol, evaluation_observer=observer)
        old.extend(prospective.state_records(source, seed))
        new.extend(transfer_states(observer.rows, seed))
        clip_train.write_json(results/'progress.json', dict(last_completed_seed=seed, evaluated_states=len(new)))
        if mirror:
            mirror.sync()
        del observer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    clip_train.write_json(results/'original_states.json', old)
    clip_train.write_json(results/'states.json', new)
    clip_train.write_json(results/'prediction_report.json', prospective.score(new, frozen))
    clip_train.write_json(results/'original_prediction_report.json', prospective.score(old, frozen))
    plot(results, old, new)
    clip_train.write_json(results/'completion.json', dict(status='complete', training_seeds=prospective.SEEDS,
        checkpoint_steps=prospective.STEPS, replay_branch_rows=720, transfer_branch_rows=720,
        evaluated_states=15, rules_refitted=False))
    verify_completion(output)
    if mirror:
        mirror.sync(final=True)
    print('COMPLETE: exact historical states, original losses within 1e-8, 720 new-pool observations.', flush=True)


if __name__ == '__main__':
    main()
