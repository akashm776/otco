from copy import deepcopy
import ast
import base64
import hashlib
import json
import zlib

import pytest
import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_continuation_timecourse as t
from src.clip_gated_training import load_protocol as gate_protocol
from src.clip_gradient_stages import observational_state
from tests.test_clip_checkpoint_diagnostic import setup


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


def make_pair():
    model, optimizer, scheduler, batch, config, initial = setup()
    meta = d.MetaPool([batch])
    probe = t.GradientReprobe([batch, batch], config, meta, gate_protocol())
    report = d.ReportPool([batch], {'fixed': {'batches': [list(range(10))]}})
    return model, optimizer, scheduler, batch, config, initial, probe, report


def test_frozen_budget_and_parent_unchanged():
    p, parent = t.load_protocol(), d.load_protocol()
    assert p['arms'] == ['native', 'sustained']
    assert p['observation_steps'] == [0, 1, 5, 10, 25, 50]
    assert 3*4*2*2*50 == p['optimizer_step_budget'] == 2400
    assert parent['arms'] == ['native', 'pulse', 'sustained']
    assert p['checkpoints'] == parent['checkpoints']
    assert p['split_seed'] == parent['split_seed']
    assert p['report_partition_seeds'] == parent['report_partition_seeds']


def test_fresh_matched_plans_and_roles():
    p = t.load_protocol()
    training, heldout = list(range(4970)), list(range(4970, 5994))
    plans = t.make_plans(training, p)
    assert len(plans) == 24
    assert plans == t.make_plans(list(reversed(training)), p)
    assert len({d.state_digest(plan) for plan in plans.values()}) == 24
    for plan in plans.values():
        assert len(plan) == 50 and all(len(b) == 64 for b in plan)
        ids = [i for b in plan for i in b]
        assert len(set(ids)) == 3200
        assert not set(ids) & set(heldout)
    roles = d.split_pool(training, heldout, p)
    assert roles == d.split_pool(training, heldout, d.load_protocol())
    assert not set(roles['meta']) & set(roles['report'])
    with pytest.raises(AssertionError, match='Repeated'):
        t.make_plans(training, {**p, 'stream_seeds': [p['previous_branch_data_seed']]})


def test_reprobe_is_deterministic_observational_and_gradient_only(monkeypatch):
    model, optimizer, scheduler, batch, config, initial, probe, report = make_pair()
    model.register_buffer('probe_counter', torch.tensor(0.))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    before = d.live_digest(model, optimizer, scheduler)
    grads = d.state_digest([p.grad for p in model.parameters()])
    modes = [m.training for m in model.modules()]
    def forbidden(*_):
        raise AssertionError('Re-probe must not perform optimizer steps')
    monkeypatch.setattr(optimizer, 'step', forbidden)
    a = probe.measure(model)
    torch.rand(7)
    b = probe.measure(model)
    assert a == b
    # Check all state except the deliberately consumed RNG above.
    assert model.probe_counter == 0
    assert d.state_digest([p.grad for p in model.parameters()]) == grads
    assert [m.training for m in model.modules()] == modes
    assert a['diagnostic_only']
    assert len(a['historical_batch_cosines']) == 2
    current = d.live_digest(model, optimizer, scheduler)
    probe.measure(model)
    assert d.live_digest(model, optimizer, scheduler) == current
    assert current != before  # Only the explicit external torch.rand changed it.


def test_pair_grid_commit_order_budget_and_unobserved_equivalence(monkeypatch):
    model, optimizer, scheduler, batch, config, initial, probe, report = make_pair()
    source = d.state_digest(initial)
    committed, saved = [], []
    evaluate = report.evaluate
    def guarded(model):
        assert committed
        return evaluate(model)
    monkeypatch.setattr(report, 'evaluate', guarded)
    calls = []
    step = optimizer.step
    def counted(*args, **kwargs):
        calls.append(1)
        return step(*args, **kwargs)
    monkeypatch.setattr(optimizer, 'step', counted)
    times = [0, 1, 5, 10, 25, 50]
    result = t.run_pair(model, optimizer, scheduler, initial, [batch]*50, config, probe, report,
        coefficient=.06, observation_steps=times,
        commit_probe=lambda label, values: committed.append((label, deepcopy(values))),
        save_point=lambda arm, offset, value: saved.append((arm, offset)))
    assert len(calls) == result['audit']['actual_optimizer_steps'] == 100
    assert [label for label, _ in committed] == [('common', 0)] + [
        (arm, time) for arm in t.ARMS for time in times[1:]]
    assert saved == [label for label, _ in committed]
    assert len(set(result['audit']['branch_start_hashes'].values())) == 1
    assert result['points']['native']['0'] == result['points']['sustained']['0']
    assert d.state_digest(initial) == source
    for arm in t.ARMS:
        assert set(result['points'][arm]) == {str(time) for time in times}
        d.restore(model, optimizer, scheduler, initial)
        for _ in range(50):
            d.step_update(model, optimizer, scheduler, batch, config, .06 if arm == 'sustained' else 0.)
        assert d.live_digest(model, optimizer, scheduler) == result['points'][arm]['50']['training_state_sha256']
    for offset in times:
        native, treated = (result['points'][arm][str(offset)] for arm in t.ARMS)
        assert result['differences'][str(offset)]['report_loss'] == pytest.approx(
            treated['report']['native_loss'] - native['report']['native_loss'])


def test_stochastic_forward_still_matches_and_reprobes_do_not_gate(monkeypatch):
    model, optimizer, scheduler, batch, config, initial, probe, report = make_pair()
    forward = model.forward
    def noisy(images, texts):
        if model.training:
            images = images + .01*torch.randn_like(images)
        return forward(images, texts)
    monkeypatch.setattr(model, 'forward', noisy)
    measure = probe.measure
    def all_off(model):
        result = measure(model)
        result['historical_above_threshold'] = False
        return result
    monkeypatch.setattr(probe, 'measure', all_off)
    result = t.run_pair(model, optimizer, scheduler, initial, [batch]*3, config, probe, report,
        coefficient=.06, observation_steps=[0, 1, 3], commit_probe=lambda *_: None)
    assert result['audit']['matched_rng_and_lrs']
    assert [row['coefficient'] for row in result['traces']['sustained']] == [.06]*3
    assert [row['coefficient'] for row in result['traces']['native']] == [0.]*3


def test_probe_commit_failure_stops_before_report(monkeypatch):
    model, optimizer, scheduler, batch, config, initial, probe, report = make_pair()
    def no_report(_):
        raise AssertionError('Report accessed after failed prediction persistence')
    monkeypatch.setattr(report, 'evaluate', no_report)
    def fail(*_):
        raise OSError('Drive failure')
    with pytest.raises(OSError, match='Drive'):
        t.run_pair(model, optimizer, scheduler, initial, [batch], config, probe, report,
            coefficient=.06, observation_steps=[0, 1], commit_probe=fail)


def test_mutating_probe_is_rejected(monkeypatch):
    model, optimizer, scheduler, batch, config, initial, probe, report = make_pair()
    measure = probe.measure
    def bad(model):
        result = measure(model)
        with torch.no_grad():
            next(model.parameters()).add_(1.)
        return result
    monkeypatch.setattr(probe, 'measure', bad)
    with pytest.raises(AssertionError, match='Re-probe changed'):
        t.run_pair(model, optimizer, scheduler, initial, [batch], config, probe, report,
            coefficient=.06, observation_steps=[0, 1], commit_probe=lambda *_: None)


def test_summary_averages_streams_states_and_seeds_without_pseudoreplication():
    p = t.load_protocol()
    rows = []
    for seed_index, seed in enumerate(p['training_seeds']):
        for step in p['checkpoint_steps']:
            for stream_index, stream in enumerate(p['stream_seeds']):
                effect = (-2 if step == 100 else 1) + seed_index + 2*stream_index
                rows.append(dict(training_seed=seed, checkpoint_step=step, stream_seed=stream,
                    differences={str(offset): {k: effect if offset else 0. for k in
                        ['report_loss', 'pool_r1_pp', 'historical_cosine', 'meta_aux_cosine']}
                        for offset in p['observation_steps']}, audit={'actual_optimizer_steps': 100}))
    summary = t.summarize(rows, p)
    assert summary['primary_value'] == 3.
    assert summary['training_seed_count'] == 3
    assert summary['pairs'] == 24 and summary['branches'] == 48
    assert summary['optimizer_steps'] == 2400
    assert summary['seed_mean_timecourse']['0']['later_minus_early'] == 0.
    for invalid in [rows[:-1], rows[:-1]+rows[:1]]:
        with pytest.raises(ValueError, match='Incomplete or duplicate'):
            t.summarize(invalid, p)


def test_preflight_blocks_any_existing_timecourse_but_not_parent(tmp_path):
    from colabs.run_clip_continuation_timecourse import require_no_previous_run
    parent = tmp_path/'clip_checkpoint_diagnostic_old'
    parent.mkdir()
    (parent/'run_manifest.json').write_text(json.dumps({'experiment': 'clip_checkpoint_diagnostic_v1'}))
    require_no_previous_run(tmp_path, t.load_protocol()['experiment'])
    run = tmp_path/'clip_continuation_timecourse_old'
    run.mkdir()
    (run/'run_manifest.json').write_text(json.dumps({'experiment': t.load_protocol()['experiment'], 'status': 'failed'}))
    with pytest.raises(RuntimeError, match='no automatic resume'):
        require_no_previous_run(tmp_path, t.load_protocol()['experiment'])


def test_generated_cell_integrity():
    path = d.ROOT/'colabs/clip_continuation_timecourse_drive_one_cell.py'
    if not path.exists():
        pytest.skip('Cell is not recursively included inside its own payload')
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {'BASE_COMMIT', 'BUNDLE_SHA256', 'PAYLOAD'}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == values['BUNDLE_SHA256']
    assert values['BASE_COMMIT'] == 'b36bcbf7c2124bbcbe0cf372df3853dadcf79bd2'
    sources = json.loads(raw)
    for name, sha in json.loads(sources['timecourse_source_manifest.json']).items():
        assert hashlib.sha256(sources[name].encode()).hexdigest() == sha
        assert sources[name] == (d.ROOT/name).read_text()
