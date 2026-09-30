from copy import deepcopy
import ast
import base64
import hashlib
import json
import subprocess
import sys
import zlib

import pytest
import torch

from src import clip_policy_test as p
from src import clip_checkpoint_diagnostic as d
from src.clip_gradient_stages import observational_state
from tests.test_clip_continuation_timecourse import make_pair


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


class Scores:
    def __init__(self, values):
        self.values = iter(values)

    def measure(self, model):
        value = next(self.values)
        return dict(mean_cosine=value, batch_cosines=[value]*16)


def execute(probe=None, report_override=None, commit=None):
    model, opt, sched, batch, config, initial, _, report = make_pair()
    result = p.run_pair(model, opt, sched, initial, [batch]*4, config,
        probe or Scores([1., -1., 1.]), report_override or report,
        coefficient=.06, threshold=0., refresh_offsets=[0, 1, 3],
        observation_steps=[0, 1, 3, 4], permutation=[1, 2, 3, 0],
        commit=commit or (lambda *_: None))
    return result


def test_protocol_and_new_matched_streams():
    protocol = p.load_protocol()
    assert 3*2*2*4*50 == protocol['optimizer_step_budget'] == 2400
    assert protocol['checkpoints'] == d.load_protocol()['checkpoints']
    assert protocol['split_seed'] == d.load_protocol()['split_seed']
    plans, orders = p.make_plans(range(4970), protocol)
    assert (plans, orders) == p.make_plans(list(reversed(range(4970))), protocol)
    assert len(plans) == len(orders) == 12
    assert len({d.state_digest(x) for x in plans.values()}) == 12
    for key, batches in plans.items():
        assert len(batches) == 50 and all(len(b) == 64 for b in batches)
        assert len({i for b in batches for i in b}) == 3200
        assert sorted(orders[key]) == list(range(50))
    with pytest.raises(AssertionError, match='Repeated'):
        p.make_plans(range(4970), {**protocol, 'stream_seeds': [2026092601]})


@pytest.mark.parametrize('count', [0, 1, 17, 49, 50])
def test_exact_exposure_matching_including_degenerate_doses(count):
    actions = [True]*count+[False]*(50-count)
    schedule = p.matched_schedule(actions, list(reversed(range(50))))
    assert sum(schedule) == count
    assert schedule == list(reversed(actions))
    if count in [0, 50]:
        assert schedule == actions  # No redraw to manufacture a timing difference.


def test_invalid_schedule_fails():
    for actions, order in [([], []), ([True, False], [0, 0]), ([1, 0], [0, 1])]:
        with pytest.raises(ValueError):
            p.matched_schedule(actions, order)


def test_decisions_committed_before_reports_and_updates_and_exact_branch_paths(monkeypatch):
    model, opt, sched, batch, config, initial, _, report = make_pair()
    events = []
    forward = model.forward
    def stochastic(images, texts):
        return forward(images + .01*torch.randn_like(images) if model.training else images, texts)
    monkeypatch.setattr(model, 'forward', stochastic)
    evaluate = report.evaluate
    def logged(model):
        events.append('report')
        return evaluate(model)
    monkeypatch.setattr(report, 'evaluate', logged)
    update = opt.step
    def counted(*args, **kwargs):
        events.append('update')
        return update(*args, **kwargs)
    monkeypatch.setattr(opt, 'step', counted)
    commits = {}
    result = p.run_pair(model, opt, sched, initial, [batch]*4, config,
        Scores([1., -1., 1.]), report, coefficient=.06, threshold=0.,
        refresh_offsets=[0, 1, 3], observation_steps=[0, 1, 3, 4], permutation=[1, 2, 3, 0],
        commit=lambda label, value: (events.append(label), commits.update({label: deepcopy(value)})))
    assert events[:5] == ['decision_000', 'report', 'update', 'decision_001', 'report']
    assert events.count('update') == result['audit']['actual_optimizer_steps'] == 16
    assert result['actions'] == dict(gated=[True, False, False, True], native=[False]*4,
                                     sustained=[True]*4, random_matched=[False, True, True, False])
    assert commits['schedule_random_matched']['dose'] == 2
    assert commits['schedule_random_matched']['actions'] == result['actions']['random_matched']
    assert len(set(result['audit']['branch_start_hashes'].values())) == 1
    for arm in p.ARMS:
        d.restore(model, opt, sched, initial)
        for on in result['actions'][arm]:
            d.step_update(model, opt, sched, batch, config, .06 if on else 0.)
        assert d.live_digest(model, opt, sched) == result['points'][arm]['4']['state_sha256']
        assert evaluate(model) == result['points'][arm]['4']['report']
    for time in ['0', '1', '3', '4']:
        for arm in p.ARMS[1:]:
            assert result['differences'][time][arm]['report_loss'] == (
                result['points']['gated'][time]['report']['native_loss']-
                result['points'][arm][time]['report']['native_loss'])


def test_strict_threshold_and_reactivation():
    result = execute(Scores([0., 1., -1.]))
    assert result['actions']['gated'] == [False, True, True, False]


def test_real_training_only_probe_preserves_full_state():
    from src.clip_gated_training import load_protocol
    model, opt, sched, batch, config, initial, _, _ = make_pair()
    probe = p.AlignmentGate([batch, batch], config, load_protocol())
    before = p.state_stamp(model, opt, sched)
    measured = p.guarded(model, opt, sched, lambda: probe.measure(model))
    assert measured == probe.measure(model)
    assert len(measured['batch_cosines']) == 2
    assert p.state_stamp(model, opt, sched) == before
    assert not hasattr(probe, 'report') and not hasattr(probe, 'meta')


def test_reporting_values_cannot_change_gate_actions():
    class Reporting:
        def __init__(self, value):
            self.value = value
        def evaluate(self, model):
            return dict(native_loss=self.value, pool_retrieval={'mean_r1_percent': 0.},
                        partition_losses={'fixed': [self.value]})
    a = execute(report_override=Reporting(-1e6))
    b = execute(report_override=Reporting(1e6))
    assert a['actions'] == b['actions']
    assert a['decisions'] == b['decisions']


@pytest.mark.parametrize('failure', ['nan', 'commit'])
def test_gate_failure_prevents_optimizer_updates(monkeypatch, failure):
    model, opt, sched, batch, config, initial, _, report = make_pair()
    monkeypatch.setattr(opt, 'step', lambda *_: pytest.fail('Must stop before training'))
    def bad(*_):
        raise OSError('disk unavailable')
    with pytest.raises(ValueError if failure == 'nan' else OSError):
        p.run_pair(model, opt, sched, initial, [batch]*4, config,
            Scores([float('nan')] if failure == 'nan' else [1.]), report,
            coefficient=.06, threshold=0., refresh_offsets=[0, 1, 3],
            observation_steps=[0, 4], permutation=[1, 2, 3, 0],
            commit=bad if failure == 'commit' else lambda *_: None)


@pytest.mark.parametrize('mutation', ['parameter', 'gradient', 'mode', 'rng'])
def test_mutating_observer_is_rejected(mutation):
    model, opt, sched, batch, config, initial, _, report = make_pair()
    def bad():
        parameter = next(x for x in model.parameters() if x.requires_grad)
        if mutation == 'parameter':
            with torch.no_grad():
                parameter.add_(1)
        elif mutation == 'gradient':
            parameter.grad = torch.ones_like(parameter)
        elif mutation == 'mode':
            model.eval()
        else:
            torch.rand(1)
    with pytest.raises(AssertionError, match='changed training state'):
        p.guarded(model, opt, sched, bad)


def test_summary_equal_seed_state_stream_weight_and_grid_validation():
    protocol = p.load_protocol()
    rows = []
    for si, seed in enumerate(protocol['training_seeds']):
        for ki, step in enumerate(protocol['checkpoint_steps']):
            for ui, stream in enumerate(protocol['stream_seeds']):
                effect = si*10+ki*2+ui
                rows.append(dict(training_seed=seed, checkpoint_step=step, stream_seed=stream,
                    differences={str(t): {a: dict(report_loss=effect, pool_r1_pp=0.)
                        for a in p.ARMS[1:]} for t in protocol['observation_steps']},
                    exposure={a: 0 for a in p.ARMS}, random_schedule_equals_gate=True,
                    audit={'actual_optimizer_steps': 200}))
    summary = p.summarize(rows, protocol)
    assert summary['primary_value'] == 11.5
    assert [s['primary'] for s in summary['per_seed']] == [1.5, 11.5, 21.5]
    assert summary['optimizer_steps'] == 2400
    for invalid in [rows[:-1], rows[:-1]+rows[:1]]:
        with pytest.raises(ValueError, match='Incomplete or duplicate'):
            p.summarize(invalid, protocol)


def test_completed_marginal_and_failed_runs_do_not_block_new_policy(tmp_path):
    from colabs.run_clip_policy_test import require_no_previous_run
    for relative in ['clip_marginal_utility_complete', 'failed runs/clip_marginal_utility_partial']:
        folder = tmp_path/relative
        folder.mkdir(parents=True)
        (folder/'run_manifest.json').write_text(json.dumps({'experiment': 'clip_marginal_utility_v1'}))
    require_no_previous_run(tmp_path, 'clip_policy_test_v1')
    folder = tmp_path/'clip_policy_test_failed'
    folder.mkdir()
    (folder/'run_manifest.json').write_text(json.dumps({'experiment': 'clip_policy_test_v1'}))
    with pytest.raises(RuntimeError, match='no automatic resume'):
        require_no_previous_run(tmp_path, 'clip_policy_test_v1')


@pytest.mark.parametrize('code', [0, 1])
def test_child_logging_retains_stderr_and_exit_code(tmp_path, code):
    from colabs.policy_test_logging import run_logged
    target = tmp_path/'run.log'
    cmd = [sys.executable, '-u', '-c', f'import sys; print("stdout"); print("root cause", file=sys.stderr); sys.exit({code})']
    if code:
        with pytest.raises(subprocess.CalledProcessError) as error:
            run_logged(cmd, tmp_path, target)
        assert 'root cause' in error.value.output
    else:
        assert run_logged(cmd, tmp_path, target) == 0
    text = target.read_text()
    assert 'stdout' in text and 'root cause' in text and f'CHILD_EXIT_CODE: {code}' in text
    with pytest.raises(FileExistsError):
        run_logged(cmd, tmp_path, target)


def test_generated_bundle_is_pinned_and_matches_sources():
    path = d.ROOT/'colabs/clip_policy_test_drive_one_cell.py'
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
    for name, sha in json.loads(sources['policy_source_manifest.json']).items():
        assert hashlib.sha256(sources[name].encode()).hexdigest() == sha
        assert sources[name] == (d.ROOT/name).read_text()
