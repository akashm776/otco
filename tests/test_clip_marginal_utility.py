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
from src import clip_marginal_utility as m
from src.clip_gradient_stages import observational_state
from tests.test_clip_continuation_timecourse import make_pair


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


def test_fixed_design_and_budget():
    p = m.load_protocol()
    assert p['checkpoint_steps'] == [100, 500]
    assert p['observation_steps'] == [0, 10, 25, 50]
    assert p['primary_time'] == 25
    assert 3*2*2*2*50 == p['continuation_updates'] == 1200
    assert 3*2*2*(1+2*3)*2*2 == p['trial_updates'] == 336
    assert p['optimizer_step_budget'] == 1536
    assert p['checkpoints'] == d.load_protocol()['checkpoints']
    assert p['split_seed'] == d.load_protocol()['split_seed']
    assert t.load_protocol()['checkpoint_steps'] == [100, 250, 500, 750]


def test_trial_plans_disjoint_and_frozen():
    p = m.load_protocol()
    training = list(range(4970))
    continuation = t.make_plans(training, p)
    for key, plan in continuation.items():
        seed, step, stream = map(int, key.split(':'))
        trial = m.trial_plan(training, plan, range(1024), seed, step, stream, p)
        assert trial == m.trial_plan(training[::-1], plan, range(1024), seed, step, stream, p)
        ids = [i for b in trial for i in b]
        assert len(trial) == 2 and all(len(b) == 64 for b in trial)
        assert len(set(ids)) == 128
        assert not set(ids) & {i for b in plan for i in b}
        assert not set(ids) & set(range(1024))
    with pytest.raises(ValueError, match='Insufficient'):
        m.trial_plan(range(128), [], range(128), 1, 100, 1, p)


def test_snapshot_exact_and_immutable_with_modes_gradients_buffers():
    model, opt, sched, batch, config, initial, probe, report = make_pair()
    model.register_buffer('counter', torch.tensor(3.))
    model.eval()
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    before = m.snapshot(model, opt, sched)
    stamp = d.state_digest(before)
    d.step_update(model, opt, sched, batch, config, .06)
    model.counter.add_(2)
    assert d.state_digest(before) == stamp
    m.restore_snapshot(model, opt, sched, before)
    m.assert_snapshot(model, opt, sched, before)
    assert not model.training


def test_actual_trial_updates_match_direct_execution_and_rollback(monkeypatch):
    model, opt, sched, batch, config, initial, probe, report = make_pair()
    meta = d.MetaPool([batch])
    before = m.snapshot(model, opt, sched)
    commits = []
    evaluate = report.evaluate
    def guarded(model):
        assert len(commits) == 1
        return evaluate(model)
    monkeypatch.setattr(report, 'evaluate', guarded)
    calls = []
    step = opt.step
    def counted(*args, **kwargs):
        calls.append(1)
        return step(*args, **kwargs)
    monkeypatch.setattr(opt, 'step', counted)
    result = m.marginal_audit(model, opt, sched, [batch, batch], config, meta, probe, report,
        coefficient=.06, commit_predictions=lambda p: commits.append(deepcopy(p)))
    assert len(calls) == result['trial_updates'] == 4
    m.assert_snapshot(model, opt, sched, before)
    assert result['trials'][0]['reports'] == result['trials'][1]['reports']
    assert result['predictions'] == commits[0]
    for action, coefficient in [('native', 0.), ('auxiliary', .06)]:
        m.restore_snapshot(model, opt, sched, before)
        d.step_update(model, opt, sched, batch, config, coefficient)
        assert evaluate(model) == result['trials'][0]['reports'][action]
        assert d.live_digest(model, opt, sched) == result['trials'][0]['state_hashes'][action]
    trial = result['trials'][0]
    assert trial['report_loss'] == trial['reports']['auxiliary']['native_loss']-trial['reports']['native']['native_loss']


@pytest.mark.parametrize('failure', ['commit', 'report', 'probe', 'step'])
def test_failure_restores_state_and_failed_commit_never_reports(monkeypatch, failure):
    model, opt, sched, batch, config, initial, probe, report = make_pair()
    before = m.snapshot(model, opt, sched)
    calls = []
    def bad(*_):
        with torch.no_grad():
            next(model.parameters()).add_(1.)
        torch.rand(3)
        raise OSError('deliberate failure')
    def forbidden(*_):
        calls.append(1)
        raise AssertionError('Report before commitment')
    commit = bad if failure == 'commit' else lambda _: None
    if failure == 'commit':
        monkeypatch.setattr(report, 'evaluate', forbidden)
    if failure == 'report':
        monkeypatch.setattr(report, 'evaluate', bad)
    if failure == 'probe':
        monkeypatch.setattr(probe, 'measure', bad)
    if failure == 'step':
        monkeypatch.setattr(opt, 'step', bad)
    with pytest.raises(OSError, match='deliberate'):
        m.marginal_audit(model, opt, sched, [batch], config, d.MetaPool([batch]), probe, report,
                         coefficient=.06, commit_predictions=commit)
    assert not calls
    m.assert_snapshot(model, opt, sched, before)


def test_observer_mutation_rejected_and_rolled_back(monkeypatch):
    model, opt, sched, batch, config, initial, probe, report = make_pair()
    before = m.snapshot(model, opt, sched)
    evaluate = report.evaluate
    def bad(model):
        result = evaluate(model)
        with torch.no_grad():
            next(model.parameters()).add_(1.)
        return result
    monkeypatch.setattr(report, 'evaluate', bad)
    with pytest.raises(AssertionError, match='Audit changed'):
        m.marginal_audit(model, opt, sched, [batch], config, d.MetaPool([batch]), probe, report,
                         coefficient=.06, commit_predictions=lambda _: None)
    m.assert_snapshot(model, opt, sched, before)


def test_history_prefixes_exact_with_trials_and_stochastic_forward(monkeypatch):
    model, opt, sched, batch, config, initial, probe, report = make_pair()
    forward = model.forward
    def stochastic(images, texts):
        return forward(images + .01*torch.randn_like(images) if model.training else images, texts)
    monkeypatch.setattr(model, 'forward', stochastic)
    expected = {}
    d.restore(model, opt, sched, initial)
    expected['common:0'] = d.live_digest(model, opt, sched)
    for arm in t.ARMS:
        d.restore(model, opt, sched, initial)
        for offset in range(1, 4):
            d.step_update(model, opt, sched, batch, config, .06 if arm == 'sustained' else 0.)
            if offset in [1, 3]:
                expected[f'{arm}:{offset}'] = d.live_digest(model, opt, sched)
    commits, saves = [], []
    result = m.run_pair(model, opt, sched, initial, [batch]*3, [batch]*2, config,
        d.MetaPool([batch]), probe, report, coefficient=.06, observation_steps=[0, 1, 3],
        commit_predictions=lambda label, p: commits.append(label),
        save_point=lambda arm, offset, value: saves.append((arm, offset)), expected_hashes=expected)
    assert commits == saves == [('common', 0), ('native', 1), ('native', 3), ('sustained', 1), ('sustained', 3)]
    assert result['audit']['actual_optimizer_steps'] == 6+5*4
    assert result['history_contrast']['0'] == 0.
    assert result['points']['native']['0'] == result['points']['sustained']['0']
    assert all(x['coefficient'] == .06 for x in result['traces']['sustained'])
    expected['common:0'] = 'wrong'
    with pytest.raises(AssertionError, match='replay differs'):
        m.run_pair(model, opt, sched, initial, [batch]*3, [batch], config,
            d.MetaPool([batch]), probe, report, coefficient=.06, observation_steps=[0, 3],
            commit_predictions=lambda *_: pytest.fail('Must stop before trials'), expected_hashes=expected)


def test_summary_hierarchy_and_shared_zero_not_duplicated():
    p = m.load_protocol()
    rows = []
    for seed in p['training_seeds']:
        for step in p['checkpoint_steps']:
            for stream in p['stream_seeds']:
                points = {}
                for arm in t.ARMS:
                    points[arm] = {}
                    for time in p['observation_steps']:
                        effect = 1. if arm == 'native' or time == 0 else 4.
                        points[arm][str(time)] = dict(mean_marginal_report_loss=effect,
                            mean_marginal_pool_r1_pp=0., predictions={'batches': [
                                {'decisions': {'historical': False, 'raw': False, 'meta': False}}]*2},
                            trials=[{'report_loss': effect}]*2)
                rows.append(dict(training_seed=seed, checkpoint_step=step, stream_seed=stream, points=points,
                    audit=dict(continuation_updates=100, trial_updates=28, actual_optimizer_steps=128)))
    result = m.summarize(rows, p)
    assert result['primary_value'] == 3.
    assert result['optimizer_steps'] == 1536
    assert result['predictor_descriptives']['historical']['trials'] == 168
    assert result['predictor_descriptives']['historical']['correct'] == 168
    assert result['training_seed_count'] == 3
    for invalid in [rows[:-1], rows[:-1]+rows[:1]]:
        with pytest.raises(ValueError, match='Incomplete or duplicate'):
            m.summarize(invalid, p)


def test_reference_grid_and_pins():
    p = m.load_protocol()
    ref = json.loads((d.ROOT/p['reference_file']).read_text())
    assert ref['run_id'] == 'clip_continuation_timecourse_20260929T135631_790370Z'
    assert ref['source_files_verified'] == 615
    assert set(ref['pairs']) == {f'{s}:{k}:{u}' for s in p['training_seeds']
                               for k in p['checkpoint_steps'] for u in p['stream_seeds']}
    for pair in ref['pairs'].values():
        assert set(pair['anchor_hashes']) == {'common:0'} | {f'{a}:{h}' for a in t.ARMS for h in [10,25,50]}
        assert all(len(v) == 64 for v in pair['anchor_hashes'].values())


def test_new_preflight_does_not_block_old_studies(tmp_path):
    from colabs.run_clip_marginal_utility import require_no_previous_run
    for name, experiment in [('clip_continuation_timecourse_old', t.load_protocol()['experiment']),
                             ('clip_checkpoint_diagnostic_old', d.load_protocol()['experiment'])]:
        folder = tmp_path/name
        folder.mkdir()
        (folder/'run_manifest.json').write_text(json.dumps({'experiment': experiment}))
    require_no_previous_run(tmp_path, m.load_protocol()['experiment'])
    folder = tmp_path/'clip_marginal_utility_failed'
    folder.mkdir()
    (folder/'run_manifest.json').write_text(json.dumps({'experiment': m.load_protocol()['experiment']}))
    with pytest.raises(RuntimeError, match='no automatic resume'):
        require_no_previous_run(tmp_path, m.load_protocol()['experiment'])


def test_generated_cell_integrity():
    path = d.ROOT/'colabs/clip_marginal_utility_drive_one_cell.py'
    if not path.exists():
        pytest.skip('Generated cell not recursively included in itself')
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {'BASE_COMMIT', 'BUNDLE_SHA256', 'PAYLOAD'}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == values['BUNDLE_SHA256']
    assert values['BASE_COMMIT'] == 'b36bcbf7c2124bbcbe0cf372df3853dadcf79bd2'
    sources = json.loads(raw)
    for name, sha in json.loads(sources['marginal_source_manifest.json']).items():
        assert hashlib.sha256(sources[name].encode()).hexdigest() == sha
        assert sources[name] == (d.ROOT/name).read_text()
    assert sources['tests/__init__.py'] == (d.ROOT/'tests/__init__.py').read_text()
