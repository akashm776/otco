from copy import deepcopy
import ast
import base64
import hashlib
import json
import zlib

import pytest
import torch

from src import clip_policy_pool_test as pool
from src import clip_policy_test as policy
from src import clip_checkpoint_diagnostic as d
from src.clip_gradient_stages import observational_state
from tests.test_clip_continuation_timecourse import make_pair
from tests.test_clip_policy_test import Scores


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


def fixture():
    model, opt, sched, batch, config, initial, _, report = make_pair()
    reference = policy.run_pair(model, opt, sched, initial, [batch]*4, config,
        Scores([1., -1., 1.]), report, coefficient=.06, threshold=0.,
        refresh_offsets=[0, 1, 3], observation_steps=[0, 1, 3, 4], permutation=[1, 2, 3, 0],
        commit=lambda *_: None)
    return model, opt, sched, batch, config, initial, report, reference


def replay(parts, arm='gated'):
    model, opt, sched, batch, config, initial, _, reference = parts
    return pool.replay_branch(model, opt, sched, initial, [batch]*4, config,
        reference, arm, .06, checkpoint_offsets=[0, 1, 3, 4])


def test_frozen_protocol_and_partition_grid():
    p = pool.load_protocol()
    assert p['optimizer_step_budget'] == 3*2*2*4*50 == 2400
    assert p['checkpoints'] == d.load_protocol()['checkpoints']
    assert p['split_seed'] == d.load_protocol()['split_seed']
    conditions = pool.make_conditions(p)
    assert conditions == pool.make_conditions(p)
    assert len(conditions) == 32
    assert set(conditions).isdisjoint(map(str, p['report_partition_seeds']))
    for condition in conditions.values():
        assert len(condition['batches']) == 8
        assert all(len(b) == 64 for b in condition['batches'])
        assert sorted(i for b in condition['batches'] for i in b) == list(range(512))
    with pytest.raises(ValueError, match='Reused'):
        pool.make_conditions({**p, 'new_partition_seed_start': p['report_partition_seeds'][0]})


def test_all_recorded_branches_replay_and_reports_and_cache(tmp_path, monkeypatch):
    parts = fixture()
    model, opt, sched, batch, _, _, report, reference = parts
    # No re-probe / decision recomputation is allowed during replay.
    monkeypatch.setattr(policy.AlignmentGate, 'measure', lambda *_: pytest.fail('Gate recomputed'))
    for arm in policy.ARMS:
        audit = replay(parts, arm)
        assert audit['actual_optimizer_steps'] == 4
        assert audit['final_state_sha256'] == reference['points'][arm]['4']['state_sha256']
        assert set(audit['checked_state_hashes']) == {'0', '1', '3', '4'}
        before = policy.state_stamp(model, opt, sched)
        encoded = pool.encode_verified(model, opt, sched, [batch], reference['points'][arm]['4']['report'], report.conditions)
        assert policy.state_stamp(model, opt, sched) == before
        metadata = pool.save_cache(tmp_path/(arm+'.npz'), encoded, 'a'*64)
        restored = pool.load_cache(tmp_path/(arm+'.npz'), metadata)
        assert d.feature_hash(restored) == d.feature_hash(encoded)
        assert pool.evaluate_encoded(restored, report.conditions) == pool.evaluate_encoded(encoded, report.conditions)
        assert policy.state_stamp(model, opt, sched) == before


@pytest.mark.parametrize('offset', ['0', '1', '4'])
def test_wrong_reference_state_fails(offset):
    parts = fixture()
    parts[-1]['points']['gated'][offset]['state_sha256'] = '0'*64
    with pytest.raises(AssertionError, match='Exact policy replay mismatch'):
        replay(parts)


@pytest.mark.parametrize('field', ['coefficient', 'learning_rates', 'rng_after_sha256'])
def test_trace_mismatch_fails(field):
    parts = fixture()
    parts[-1]['traces']['gated'][0][field] = None
    with pytest.raises(AssertionError, match='action/LR/RNG'):
        replay(parts)


def test_changed_action_and_incomplete_schedule_fail():
    parts = fixture()
    parts[-1]['actions']['gated'][0] = False
    with pytest.raises(AssertionError, match='action/LR/RNG'):
        replay(parts)
    parts[-1]['actions']['gated'].pop()
    with pytest.raises(ValueError, match='schedule'):
        replay(parts)


@pytest.mark.parametrize('field', ['features_sha256', 'native_loss', 'partition_losses', 'pool_retrieval'])
def test_original_feature_or_report_mismatch_rejected(field):
    parts = fixture()
    replay(parts)
    model, opt, sched, batch, _, _, report, reference = parts
    original = deepcopy(reference['points']['gated']['4']['report'])
    original[field] = None
    with pytest.raises(AssertionError, match='hash differs|did not replay'):
        pool.encode_verified(model, opt, sched, [batch], original, report.conditions)


def test_full_pool_loss_matches_closed_form_and_row_permutation():
    encoded = (torch.eye(8), torch.eye(8), 3.)
    conditions = {'x': {'batches': [list(range(4)), list(range(4, 8))]}}
    result = pool.evaluate_encoded(encoded, conditions)
    expected = torch.log(torch.exp(torch.tensor(3., dtype=torch.float64))+7)-3
    assert result['full_pool_loss'] == pytest.approx(float(expected), abs=1e-14)
    assert result['pool_retrieval']['mean_r1_percent'] == 100.
    order = torch.tensor([2, 7, 4, 1, 3, 0, 6, 5])
    permuted = (encoded[0][order], encoded[1][order], 3.)
    assert pool.evaluate_encoded(permuted, conditions)['full_pool_loss'] == pytest.approx(result['full_pool_loss'])
    with pytest.raises(ValueError, match='exactly once'):
        pool.evaluate_encoded(encoded, {'bad': {'batches': [[0]*8]}})
    with pytest.raises(ValueError, match='Unequal'):
        pool.evaluate_encoded(encoded, {'bad': {'batches': [[0], list(range(1, 8))]}})


def test_cache_corruption_and_metadata_mismatch_fail(tmp_path):
    path = tmp_path/'features.npz'
    metadata = pool.save_cache(path, (torch.eye(4), torch.eye(4), 2.), 'a'*64)
    with pytest.raises(FileExistsError):
        pool.save_cache(path, (torch.eye(4), torch.eye(4), 2.), 'a'*64)
    with pytest.raises(ValueError, match='checksum'):
        pool.load_cache(path, {**metadata, 'npz_sha256': '0'*64})
    with pytest.raises(ValueError, match='hash/shape'):
        pool.load_cache(path, {**metadata, 'feature_sha256': '0'*64})
    with pytest.raises(ValueError, match='Invalid'):
        pool.validate_encoded((torch.eye(4), torch.eye(4), float('nan')))


def test_summary_keeps_seed_hierarchy_and_partitions_are_sensitivity():
    p = pool.load_protocol()
    names = pool.make_conditions(p)
    rows = []
    for si, seed in enumerate(p['training_seeds']):
        for step in p['checkpoint_steps']:
            for stream in p['stream_seeds']:
                endpoints = {a: dict(full_pool_loss=float(si+1) if a == 'gated' else 0.,
                    pool_retrieval={'mean_r1_percent': 0.},
                    partition_means={n: (-.01 if a == 'gated' else 0.) for n in names}) for a in policy.ARMS}
                rows.append(dict(training_seed=seed, checkpoint_step=step, stream_seed=stream,
                    endpoints=endpoints, replay={a: {'actual_optimizer_steps': 50} for a in policy.ARMS}))
    result = pool.summarize(rows, p)
    assert result['primary_value'] == 2.
    assert result['optimizer_steps'] == 2400
    sensitivity = result['comparisons']['random_matched']['partition_sensitivity']
    assert sensitivity['favorable'] == 32 and 'not independent' in sensitivity['note']
    with pytest.raises(ValueError, match='Incomplete'):
        pool.summarize(rows[:-1], p)


def test_reference_location_and_previous_run_guard(tmp_path):
    from colabs.run_clip_policy_pool_test import locate_policy_reference, require_no_previous_run
    from scripts.analyze_clip_policy_robustness import SPEC
    with pytest.raises(FileNotFoundError, match='exactly one'):
        locate_policy_reference([tmp_path])
    source = tmp_path/SPEC['source_run']
    source.mkdir()
    assert locate_policy_reference([tmp_path]) == source
    require_no_previous_run(tmp_path, 'clip_policy_pool_test_v1')
    marker = tmp_path/'clip_policy_pool_test_example'
    marker.mkdir()
    (marker/'run_manifest.json').write_text(json.dumps({'experiment': 'clip_policy_pool_test_v1'}))
    with pytest.raises(RuntimeError, match='Existing'):
        require_no_previous_run(tmp_path, 'clip_policy_pool_test_v1')


def test_embedded_cell_exact_sources():
    cell = d.ROOT/'colabs/clip_policy_pool_test_drive_one_cell.py'
    if not cell.exists():
        pytest.skip('Generated delivery cell not included inside its own overlay')
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(cell.read_text()).body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in {'BUNDLE_SHA256', 'PAYLOAD'}}
    raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == values['BUNDLE_SHA256']
    sources = json.loads(raw)
    manifest = json.loads(sources.pop('policy_pool_source_manifest.json'))
    assert set(manifest) == set(sources)
    for name, source in sources.items():
        assert source == (d.ROOT/name).read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == manifest[name]
