from scripts.analyze_clip_usefulness_predictors import FEATURES, SEEDS, STEPS, analyze, fit_rule, predict


def fixture():
    return [dict(training_seed=seed, effect=-1. if step < 500 else 1., near_zero=False,
                 **{f: float(step) for f in FEATURES}) for seed in SEEDS for step in STEPS]


def test_threshold_learns_both_directions():
    rows = [{'x': 1., 'effect': -1.}, {'x': 2., 'effect': 1.}]
    assert [predict(fit_rule(rows, 'x'), r) for r in rows] == [True, False]
    rows = [{**r, 'effect': -r['effect']} for r in rows]
    assert [predict(fit_rule(rows, 'x'), r) for r in rows] == [False, True]


def test_all_folds_have_only_five_test_states():
    folds, predictions = analyze(fixture())
    assert len(folds) == 21 and len(predictions) == 105
    assert all(r['count'] == 5 for r in folds)
    assert all(r['correct'] == 5 for r in folds if r['predictor'] != 'majority_baseline')


def test_held_out_labels_cannot_change_fitted_rules():
    states = fixture()
    first, _ = analyze(states)
    changed = [{**r, 'effect': -r['effect']} if r['training_seed'] == 42 else r for r in states]
    second, _ = analyze(changed)
    assert [r['rule'] for r in first if r['held_out_seed'] == 42] == [r['rule'] for r in second if r['held_out_seed'] == 42]
