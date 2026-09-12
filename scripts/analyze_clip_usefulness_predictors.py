"""Offline, leave-one-training-seed-out screening; Python standard library only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

SEEDS = [42, 123, 456]
STEPS = [100, 250, 500, 750, 1001]
# Fixed candidate list: no held-out loss or training-loss differences as predictors.
FEATURES = ['checkpoint_step', 'full_gradient_alignment', 'projection_gradient_alignment',
            'weighted_gradient_ratio', 'actual_update_difference_ratio',
            'actual_update_cosine_to_native']
DEADBAND = 1e-6  # Fixed sensitivity flag, not fitted to these results.


def balanced_accuracy(truth, predicted):
    recalls = [sum(p == label for y, p in zip(truth, predicted) if y == label) / truth.count(label)
               for label in [False, True] if label in truth]
    return statistics.fmean(recalls) if recalls else None


def fit_rule(rows, feature):
    """Fit one decision stump on training seeds ONLY; deterministic tie breaking."""
    values = sorted({r[feature] for r in rows})
    thresholds = [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates = [dict(constant=False), dict(constant=True)]
    candidates += [dict(feature=feature, threshold=t, lower_is_helpful=lower)
                   for t in thresholds for lower in [True, False]]
    truth = [r['effect'] < 0 for r in rows]
    return max(candidates, key=lambda rule: balanced_accuracy(truth, [predict(rule, r) for r in rows]))


def predict(rule, row):
    if 'constant' in rule:
        return rule['constant']
    return (row[rule['feature']] <= rule['threshold']) == rule['lower_is_helpful']


def load_archive(archive):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or bundle.testzip() is not None:
            raise ValueError('Duplicate members or corrupt ZIP')
        markers = [n for n in names if n.count('/') == 2 and n.endswith('/results/completion.json')]
        if len(markers) != 1:
            raise ValueError('Expected one intermediate-study completion marker')
        root = markers[0].split('/')[0]
        read = lambda name: json.loads(bundle.read(root + '/' + name))
        expected = dict(status='complete', training_seeds=SEEDS, checkpoint_steps=STEPS,
                        total_branch_rows=720, new_intermediate_branch_rows=432, endpoint_replay_branch_rows=288)
        if read('results/completion.json') != expected:
            raise ValueError('Incomplete or wrong study')
        manifest = read('results/run_manifest.json')
        if manifest['status'] != 'complete' or manifest['source_commit'] != 'cce20072fe61be22bfcfd3b17ac8605d8bdce78b':
            raise ValueError('Unexpected experiment source')
        states = []
        for seed in SEEDS:
            prefix = f'seed_{seed}/results/paired/'
            checks = read(prefix + 'endpoint_replay_checks.json')
            if len(checks) != 2 or sorted(c['step'] for c in checks) != [100, 1001]:
                raise ValueError('Missing endpoint replay checks')
            if not all(c['passed'] is True and c['rows'] == 48 and c['loss_absolute_tolerance'] == 1e-8
                       and 0 <= c['maximum_absolute_loss_difference'] <= 1e-8 for c in checks):
                raise ValueError('Failed endpoint replay')
            rows = [json.loads(line) for line in bundle.read(root + '/' + prefix + 'paired_updates.jsonl').decode().splitlines()]
            lookup = {(r['checkpoint_step'], r['trial'], r['arm']): r for r in rows}
            keys = {(step, trial, arm) for step in STEPS for trial in range(16)
                    for arm in ['baseline', 'uniform_top8', 'hardest_real']}
            if len(rows) != 240 or set(lookup) != keys:
                raise ValueError('Missing/duplicated branch rows')
            # Collapse fixed diagnostic batches BEFORE fitting: 15 states, not 240 independent seeds.
            for step in STEPS:
                selected = [lookup[step, trial, 'uniform_top8'] for trial in range(16)]
                for r in selected:
                    native = lookup[step, r['trial'], 'baseline']
                    for row in [r, native]:
                        parts = ['shuffle_seed_42', 'shuffle_seed_123', 'shuffle_seed_4242']
                        losses = [v for p in parts for v in row['heldout_loss'][p]]
                        if len(losses) != 48 or not all(math.isfinite(v) for v in losses):
                            raise ValueError('Invalid primary partition losses')
                        if not math.isclose(statistics.fmean(losses), row['heldout_mean'], abs_tol=1e-12, rel_tol=0):
                            raise ValueError('Held-out mean arithmetic mismatch')
                    effect = r['heldout_mean'] - native['heldout_mean']
                    if not math.isfinite(effect) or not math.isclose(effect, r['incremental_heldout_loss'], abs_tol=1e-12, rel_tol=0):
                        raise ValueError('Paired loss arithmetic mismatch')
                    if not all(math.isfinite(r[f]) for f in FEATURES):
                        raise ValueError('Nonfinite predictor')
                state = dict(training_seed=seed, checkpoint_step=step,
                             effect=statistics.fmean(r['incremental_heldout_loss'] for r in selected))
                state.update({f: statistics.fmean(r[f] for r in selected) for f in FEATURES if f != 'checkpoint_step'})
                state['near_zero'] = abs(state['effect']) <= DEADBAND
                states.append(state)
    return states, manifest


def analyze(states):
    predictions, folds = [], []
    for seed in SEEDS:
        train = [r for r in states if r['training_seed'] != seed]
        test = [r for r in states if r['training_seed'] == seed]
        if len(train) != 10 or len(test) != 5:
            raise ValueError('Expected ten training and five held-out states')
        for feature in ['majority_baseline'] + FEATURES:
            rule = (dict(constant=sum(r['effect'] < 0 for r in train) > len(train)/2)
                    if feature == 'majority_baseline' else fit_rule(train, feature))
            labels = [predict(rule, r) for r in test]
            truth = [r['effect'] < 0 for r in test]
            folds.append(dict(held_out_seed=seed, predictor=feature, rule=rule,
                              correct=sum(a == b for a,b in zip(labels,truth)), count=5,
                              balanced_accuracy=balanced_accuracy(truth,labels)))
            for r, label in zip(test, labels):
                predictions.append(dict(predictor=feature, training_seed=seed, checkpoint_step=r['checkpoint_step'],
                    effect=r['effect'], near_zero=r['near_zero'], actual_helpful=r['effect'] < 0, predicted_helpful=label))
    return folds, predictions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, help='Defaults to newest intermediate complete ZIP in Downloads')
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args()
    archive = args.archive
    if archive is None:
        candidates = list((Path.home() / 'Downloads').glob('clip_paired_intermediate_*_complete.zip'))
        if not candidates:
            parser.error('No matching ZIP in Downloads; supply --archive')
        archive = max(candidates, key=lambda p: p.stat().st_mtime)
    output = args.output_directory or archive.with_name(archive.stem + '_predictor_analysis')
    if output.exists():
        parser.error('Output directory already exists; choose a fresh --output-directory')
    states, manifest = load_archive(archive)
    folds, predictions = analyze(states)
    lines = ['# Exploratory usefulness predictor screening', '',
             'Leave one training seed out: fit on ten states, evaluate five states from the remaining seed.',
             'Synthetic uniform-top-8 only. Fixed diagnostic batches are averaged within each seed/state.', '',
             '| Predictor | Seed 42 correct/5 | Seed 123 correct/5 | Seed 456 correct/5 | Mean seed balanced accuracy | Correct outside ±1e-6 |',
             '|---|---:|---:|---:|---:|---:|']
    for feature in ['majority_baseline'] + FEATURES:
        chosen = [r for r in folds if r['predictor'] == feature]
        robust = [r for r in predictions if r['predictor'] == feature and not r['near_zero']]
        lines.append('| ' + feature + ' | ' + ' | '.join(str(r['correct']) for r in chosen)
                     + f" | {statistics.fmean(r['balanced_accuracy'] for r in chosen):.3f}"
                     + f" | {sum(r['actual_helpful'] == r['predicted_helpful'] for r in robust)}/{len(robust)} |")
    lines += ['', '## Interpretation limits', '',
        '- Compare every candidate with the majority and checkpoint-step baselines; do not select a winning rule as a validated curriculum.',
        '- Threshold and direction are fitted using training seeds only. Actual held-out effects never enter predictor features.',
        '- Near-zero sensitivity excludes |effect| ≤ 1e-6 only from the secondary score; it does not change training labels or fit.',
        '- Only three seeds and fifteen dependent states; no significance tests or confidence intervals. The same holdout and diagnostic inputs are reused.',
        '- Actual-update features require trial updates and resets; these are retrospective diagnostics, not free pre-update signals.',
        '- This analysis was proposed after inspecting stage outcomes. Leave-one-seed-out is a screening check, not untouched prospective validation.',
        '- If a feature looks promising, preregister it for new seeds/independent evaluation before a curriculum training experiment.', '',
        'Outputs: `states.csv`, `predictions.csv`, `analysis.json`. Preserve these with the source ZIP.']
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('states.csv', states), ('predictions.csv', predictions)]:
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / 'analysis.json').write_text(json.dumps(dict(
        archive=str(archive.resolve()), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_manifest=manifest, features=FEATURES, near_zero_band=DEADBAND,
        folds=folds, predictions=predictions, states=states), indent=2) + '\n')
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nSAVED:', output.resolve())


if __name__ == '__main__':
    main()
