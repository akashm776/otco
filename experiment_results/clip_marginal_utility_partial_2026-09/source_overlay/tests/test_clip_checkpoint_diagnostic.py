from copy import deepcopy
import hashlib
import json

import pytest
import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_train
from src.clip_early_pulse import state_digest
from src.clip_gradient_stages import observational_state
from src.clip_paired_updates import one_update, restore_branch
from tests.test_clip_paired_updates import fixture


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


def setup():
    model, optimizer, scheduler, batch, config, initial = fixture()
    initial.update(deepcopy(d.rng_state()))
    return model, optimizer, scheduler, batch, config, initial


def test_protocol_and_fixed_budget():
    p = d.load_protocol()
    assert len(p['checkpoints']) == 12
    assert 12*(3*p['horizon']+2) == 1824
    assert all(len(h) == 64 for h in p['checkpoints'].values())
    assert d.checkpoint_path(789, 100) == 'seed_789/shared/checkpoints/step_000100.pt'
    assert d.checkpoint_path(2026, 250) == 'seed_2026/alignment_gated/checkpoints/step_000250.pt'


def test_split_and_plans_are_deterministic_image_disjoint():
    p = d.load_protocol()
    train, heldout = list(range(4970)), list(range(4970, 5994))
    a = d.split_pool(train, heldout, p)
    assert a == d.split_pool(list(reversed(train)), list(reversed(heldout)), p)
    assert len(a['meta']) == len(a['report']) == 512
    assert not set(a['meta']) & set(a['report'])
    assert set(a['meta'] + a['report']) == set(heldout)
    plan = d.branch_plan(train, 789, 250, p)
    assert plan == d.branch_plan(list(reversed(train)), 789, 250, p)
    assert plan != d.branch_plan(train, 2026, 250, p)
    assert plan != d.branch_plan(train, 789, 500, p)
    assert len(plan) == 50 and all(len(b) == 64 for b in plan)
    assert len({i for b in plan for i in b}) == 3200
    assert not {i for b in plan for i in b} & set(heldout)
    with pytest.raises(ValueError, match='overlap'):
        d.split_pool(train + [4970], heldout, p)
    with pytest.raises(ValueError, match='Duplicate'):
        d.split_pool(train, heldout[:-1] + [heldout[0]], p)
    with pytest.raises(ValueError, match='enough'):
        d.branch_plan(list(range(10)), 1, 100, p)


@pytest.mark.parametrize('arm,count', [('native', 0), ('pulse', 1), ('sustained', 50)])
def test_branch_exposure(arm, count):
    assert sum(d.active(arm, i) for i in range(50)) == count
    with pytest.raises(ValueError):
        d.active('unknown', 0)


def test_report_partitions_cover_same_512_images_exactly():
    conditions = d.report_conditions(d.load_protocol())
    assert len(conditions) == 2
    for condition in conditions.values():
        batches = condition['batches']
        assert len(batches) == 8 and all(len(b) == 64 for b in batches)
        assert sorted(i for b in batches for i in b) == list(range(512))
    assert len({json.dumps(v) for v in conditions.values()}) == 2


@pytest.mark.parametrize('coefficient', [0., .06005351588542947])
def test_update_matches_existing_paired_trainer_path(coefficient):
    model, optimizer, scheduler, batch, config, initial = setup()
    restore_branch(model, optimizer, scheduler, initial, 7)
    one_update(model, optimizer, batch, config,
               arm='uniform_top8' if coefficient else 'baseline', coefficient=coefficient, seed=7)
    scheduler.step()
    expected = d.live_digest(model, optimizer, scheduler)
    initial.update(deepcopy(d.rng_state()))
    d.restore(model, optimizer, scheduler, initial)
    clip_train.set_global_seed(7)
    d.step_update(model, optimizer, scheduler, batch, config, coefficient)
    assert d.live_digest(model, optimizer, scheduler) == expected


def test_restore_is_exact_and_source_moments_do_not_alias():
    model, optimizer, scheduler, batch, config, initial = setup()
    source = state_digest(initial)
    d.restore(model, optimizer, scheduler, initial)
    before = d.live_digest(model, optimizer, scheduler)
    d.step_update(model, optimizer, scheduler, batch, config, .06)
    d.restore(model, optimizer, scheduler, initial)
    assert d.live_digest(model, optimizer, scheduler) == before
    assert state_digest(initial) == source


def test_observer_restores_buffers_modes_rng_and_leaves_grads():
    model, optimizer, scheduler, batch, config, initial = setup()
    model.register_buffer('counter', torch.tensor(0.))
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    before = d.live_digest(model, optimizer, scheduler)
    gradients = state_digest([p.grad for p in model.parameters()])
    with pytest.raises(RuntimeError):
        with d.observe(model):
            model.counter.add_(5)
            torch.rand(10)
            raise RuntimeError('probe failed')
    assert model.training
    assert d.live_digest(model, optimizer, scheduler) == before
    d.training_gradients(model, batch, config)
    d.MetaPool([batch]).gradient(model)
    assert d.live_digest(model, optimizer, scheduler) == before
    assert state_digest([p.grad for p in model.parameters()]) == gradients


def test_meta_gradient_agrees_with_directional_finite_difference():
    model, optimizer, scheduler, batch, config, initial = setup()
    # Double parameters remove FP32 finite-difference quantization in this test.
    model.double()
    batch = {k: v.double() if v.is_floating_point() else v for k,v in batch.items()}
    meta = d.MetaPool([batch])
    gradient, _ = meta.gradient(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    directions = [torch.randn_like(p) for p in parameters]
    vector = torch.cat([v.flatten() for v in directions])
    epsilon = 1e-5
    with torch.no_grad():
        for p,v in zip(parameters, directions):
            p.add_(epsilon*v)
    _, plus = meta.gradient(model)
    with torch.no_grad():
        for p,v in zip(parameters, directions):
            p.add_(-2*epsilon*v)
    _, minus = meta.gradient(model)
    assert float(gradient @ vector) == pytest.approx((plus-minus)/(2*epsilon), rel=2e-5, abs=2e-5)


def test_predictors_reproduce_displacement_score_and_sign():
    model, optimizer, scheduler, batch, config, initial = setup()
    meta = d.MetaPool([batch])
    scores, hashes, before = d.predictors(model, optimizer, scheduler, initial, batch, config, meta, .06)
    aux = d.trainable_vector(model)
    d.restore(model, optimizer, scheduler, initial)
    g0, _ = meta.gradient(model)
    d.step_update(model, optimizer, scheduler, batch, config, 0.)
    native = d.trainable_vector(model)
    gn, _ = meta.gradient(model)
    delta = (aux-native).double()
    assert scores['adamw_s0'] == float(g0 @ delta)
    assert scores['adamw_sn'] == float(gn @ delta)
    assert scores['decisions']['adamw_sn'] == (scores['adamw_sn'] < 0)
    assert hashes['native'] == d.live_digest(model, optimizer, scheduler)


def test_matched_branches_commit_before_report_and_replay():
    model, optimizer, scheduler, batch, config, initial = setup()
    committed = []
    conditions = {'fixed': {'batches': [list(range(10))]}}
    class GuardedReport(d.ReportPool):
        def evaluate(self, model):
            assert len(committed) == 1, 'report outcomes accessed before predictor commitment'
            return super().evaluate(model)
    result = d.run_state(model, optimizer, scheduler, initial, [batch]*3, config,
        d.MetaPool([batch]), GuardedReport([batch], conditions), coefficient=.06,
        commit_predictions=lambda scores: committed.append(deepcopy(scores)))
    assert committed[0] == result['predictors']
    assert len(set(result['audit']['branch_start_hashes'].values())) == 1
    assert result['audit']['actual_optimizer_steps'] == 11
    assert result['first_reports']['pulse'] == result['first_reports']['sustained']
    assert result['endpoints']['pulse'] != result['endpoints']['sustained']
    assert [r['coefficient'] for r in result['traces']['pulse']] == [.06, 0., 0.]
    assert all(result['audit'][k] for k in ['matched_rng_and_lrs', 'exact_trial_replay', 'source_immutable'])
    expected = result['endpoints']['sustained']['native_loss'] - result['endpoints']['native']['native_loss']
    assert result['differences']['sustained']['step50_loss'] == expected


def test_commit_failure_prevents_report_access():
    model, optimizer, scheduler, batch, config, initial = setup()
    class ForbiddenReport:
        def evaluate(self, model):
            raise AssertionError('Report should not be touched')
    def fail(_):
        raise OSError('Drive unavailable')
    with pytest.raises(OSError, match='Drive'):
        d.run_state(model, optimizer, scheduler, initial, [batch]*2, config,
            d.MetaPool([batch]), ForbiddenReport(), coefficient=.06, commit_predictions=fail)


def test_matching_survives_stochastic_training_forward(monkeypatch):
    model, optimizer, scheduler, batch, config, initial = setup()
    original_forward = model.forward
    def stochastic_forward(images, texts):
        if model.training:
            images = images + .01 * torch.randn_like(images)
        return original_forward(images, texts)
    monkeypatch.setattr(model, 'forward', stochastic_forward)
    result = d.run_state(model, optimizer, scheduler, initial, [batch]*3, config,
        d.MetaPool([batch]), d.ReportPool([batch], {'fixed': {'batches': [list(range(10))]}}),
        coefficient=.06, commit_predictions=lambda _: None)
    assert result['audit']['exact_trial_replay']
    assert result['audit']['matched_rng_and_lrs']
    assert result['first_reports']['pulse'] == result['first_reports']['sustained']


def test_report_loss_is_observational_and_retrieval_units_are_percent():
    model, optimizer, scheduler, batch, config, initial = setup()
    reporter = d.ReportPool([batch], {'fixed': {'batches': [list(range(10))]}})
    before = d.live_digest(model, optimizer, scheduler)
    actual = reporter.evaluate(model)
    assert d.live_digest(model, optimizer, scheduler) == before
    assert actual['native_loss'] == actual['partition_losses']['fixed'][0]
    assert 0 <= actual['pool_retrieval']['mean_r1_percent'] <= 100


def test_summary_groups_by_seed_and_rejects_incomplete_states():
    p = d.load_protocol()
    rows = []
    for seed in p['training_seeds']:
        for step in p['checkpoint_steps']:
            rows.append(dict(training_seed=seed, checkpoint_step=step,
                differences={arm: dict(step1_loss=-1e-7, step50_loss=.02, step50_pool_r1_pp=-.1)
                             for arm in ['pulse', 'sustained']},
                predictors={'decisions': {'adamw_sn': True}}, audit={'actual_optimizer_steps': 152}))
    result = d.summarize(rows, p)
    assert result['training_seed_count'] == 3 and result['states'] == 12
    assert result['optimizer_steps'] == 1824
    assert result['seed_mean']['sustained']['step50_loss'] == pytest.approx(.02)
    assert result['predictor_descriptives']['adamw_sn']['immediate']['outside_sensitivity_band'] == 0
    assert result['predictor_descriptives']['adamw_sn']['sustained50']['mean_decision_regret'] == pytest.approx(.02)
    with pytest.raises(ValueError):
        d.summarize(rows[:-1], p)
    with pytest.raises(ValueError):
        d.summarize(rows[:-1]+rows[:1], p)


def test_json_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        d.write_json(tmp_path/'bad.json', {'loss': float('nan')})


def test_cell_bundle_integrity_and_pinned_inputs():
    # Build checks separately; no network and no execution of the Colab cell.
    import ast
    import base64
    import zlib
    path = d.ROOT/'colabs/clip_checkpoint_diagnostic_drive_one_cell.py'
    if not path.exists():
        pytest.skip('Generated launcher not included in its own recursive source bundle')
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ['BASE_COMMIT', 'BUNDLE_SHA256', 'PAYLOAD']:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == values['BUNDLE_SHA256']
    assert values['BASE_COMMIT'] == '7e7cfea90b60415dc9561efbe97bcd383cc1580e'
    sources = json.loads(raw)
    manifest = json.loads(sources['diagnostic_source_manifest.json'])
    assert 'tests/__init__.py' in sources
    for name, expected in manifest.items():
        assert hashlib.sha256(sources[name].encode()).hexdigest() == expected
        assert sources[name] == (d.ROOT/name).read_text()
    assert json.loads(sources['configs/clip_checkpoint_diagnostic.json'])['checkpoints'] == d.load_protocol()['checkpoints']
