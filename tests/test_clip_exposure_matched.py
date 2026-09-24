"""Offline fabricated-data checks, not scientific measurements."""

import ast
import base64
from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace
import zlib

import numpy as np
import pytest
import torch

from src import clip_exposure_matched as exposure
from src.clip_early_pulse import auxiliary_losses, state_digest
from src.clip_gradient_stages import observational_state
from model.clip_training import native_clip_contrastive_loss
from tests.test_clip_early_pulse import TinyModel, output_fixture
from tests.test_clip_gated_training import fabricated_audit, toy_batches


@pytest.fixture(autouse=True)
def isolate_rng():
    with observational_state(torch.nn.Identity()):
        yield


def test_protocol_keeps_original_scientific_constants():
    p, original = exposure.load_protocol(), exposure.gated.load_protocol()
    for key in ['training_seeds', 'baseline_config', 'completed_updates',
                'common_native_prefix_updates', 'alignment_refresh_steps',
                'scheduler_horizon_updates', 'uniform_top8_coefficient',
                'alignment_threshold', 'probe_branch_seed']:
        assert p[key] == original[key]
    assert p['exposure_matched'] and not p['rules_refitted']


@pytest.mark.parametrize('seed', exposure.SEEDS)
def test_fixed_rank_randomization_exact_dose_and_rng_isolation(seed):
    p = exposure.load_protocol()
    audit = fabricated_audit('alignment_gated')
    before = state_digest({'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()})
    plan = exposure.matched_schedules(seed, audit, p)
    assert plan == exposure.matched_schedules(seed, audit, p)
    assert before == state_digest({'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()})
    assert plan['active_updates'] == 400
    assert len({tuple(s) for s in plan['schedules'].values()}) == 3
    orders = exposure.rank_orders(seed, p)
    for arm, schedule in plan['schedules'].items():
        assert len(schedule) == len(set(schedule)) == 400
        assert min(schedule) >= 100 and max(schedule) <= 1000
        assert schedule == sorted(orders[arm][:400])
        assert sorted(orders[arm]) == list(range(100, 1001))
    # Outcome values are not required or accessed, even if supplied as extras.
    poisoned = deepcopy(audit)
    poisoned['evaluation'] = {'r_at_1': 9999999, 'species': -9999999}
    assert exposure.matched_schedules(seed, poisoned, p) == plan


def test_randomization_not_shared_across_training_seeds():
    p = exposure.load_protocol()
    assert exposure.rank_orders(789, p) != exposure.rank_orders(2026, p)


@pytest.mark.parametrize('active', [False, True])
def test_degenerate_dose_stops_instead_of_rerolling(active):
    p = exposure.load_protocol()
    audit = fabricated_audit('alignment_gated')
    for probe in audit['probes']:
        probe['batch_cosines'] = [0.04 if active else 0.02] * 16
        probe['mean_cosine'] = probe['batch_cosines'][0]
    for d in audit['decisions']:
        d['active'] = active and d['objective_step'] >= 100
    audit['active_updates'] = sum(d['active'] for d in audit['decisions'])
    with pytest.raises(ValueError, match='Degenerate'):
        exposure.matched_schedules(789, audit, p)


def test_invalid_reference_is_not_used_to_choose_dose():
    audit = fabricated_audit('alignment_gated')
    audit['decisions'][500]['active'] = False
    with pytest.raises(AssertionError):
        exposure.matched_schedules(789, audit, exposure.load_protocol())


@pytest.mark.parametrize('steps', [[99], [1001], [100, 100], [True], [100.0]])
def test_scheduled_objective_rejects_invalid_steps(steps):
    with pytest.raises(ValueError):
        exposure.ScheduledObjective(steps, exposure.load_protocol())


@pytest.mark.parametrize('step', [0, 99, 100, 101, 103, 1000])
def test_scheduled_objective_loss_and_gradients_identical_to_original(step):
    p = exposure.load_protocol()
    obj = exposure.ScheduledObjective([100, 103, 1000], p)
    obj.decisions = [{'objective_step': s, 'active': s in obj.active_steps} for s in range(step)]
    out = output_fixture()
    actual = obj(out, species_ids=None, step=step)
    expected = native_clip_contrastive_loss(out.logits)
    if step in obj.active_steps:
        expected = expected + p['uniform_top8_coefficient'] * auxiliary_losses(out)['uniform_top8']
    torch.testing.assert_close(actual.total_loss, expected, rtol=0, atol=0)
    params = (out.image_features, out.text_features, out.logit_scale)
    for a, b in zip(torch.autograd.grad(actual.total_loss, params, retain_graph=True),
                    torch.autograd.grad(expected, params)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='Missing/repeated'):
        obj(out, species_ids=None, step=step)


def random_audit(schedule):
    return {'probes': [], 'active_updates': len(schedule),
        'decisions': [{'objective_step': s, 'active': s in schedule} for s in range(1001)]}


def test_control_audit_checks_schedule_not_just_count():
    p = exposure.load_protocol()
    audit = random_audit([100, 200])
    exposure.verify_audit('random_matched_1', audit, p, [100, 200])
    audit['decisions'][200]['active'] = False
    audit['decisions'][201]['active'] = True
    with pytest.raises(AssertionError, match='committed'):
        exposure.verify_audit('random_matched_1', audit, p, [100, 200])


def write_summary_fixture(root):
    p = exposure.load_protocol()
    for seed in exposure.SEEDS:
        shared = root / f'seed_{seed}'
        shared.mkdir()
        ref = fabricated_audit('alignment_gated')
        plan = exposure.matched_schedules(seed, ref, p)
        (shared / 'matched_schedules.json').write_text(json.dumps(plan))
        (shared / 'common_prefix_hashes.json').write_text('{"model":"same"}')
        for i, arm in enumerate(exposure.ARMS):
            path = shared / arm
            (path / 'training').mkdir(parents=True)
            (path / 'prefix_hashes.json').write_text('{"model":"same"}')
            audit = ref if i == 0 else random_audit(plan['schedules'][arm])
            (path / 'gate_audit.json').write_text(json.dumps(audit))
            summary = {'execution': {'completed_updates': 1001}, 'final': {'evaluation': {
                'canonical_retrieval': {d: {'r_at_1': [4., 1., 3., 5.][i]} for d in ['text_to_image', 'image_to_text']},
                'species': {'top_1_accuracy': .5}}}}
            (path / 'training/summary.json').write_text(json.dumps(summary))
            (path / 'timing.json').write_text('{"wall_seconds": 10}')
    return p


def test_primary_contrast_averages_all_controls_within_seed(tmp_path):
    result = exposure.summarize(tmp_path, write_summary_fixture(tmp_path))
    assert len(result['rows']) == 12
    assert len(result['per_seed_primary_contrasts']) == 3
    assert result['mean_alignment_minus_mean_random_r1_pp'] == 1.
    assert result['mean_alignment_minus_mean_random_species_pp'] == 0.


def test_report_rejects_modified_plan(tmp_path):
    p = write_summary_fixture(tmp_path)
    path = tmp_path / 'seed_789/matched_schedules.json'
    plan = json.loads(path.read_text())
    plan['schedules']['random_matched_1'].pop()
    path.write_text(json.dumps(plan))
    with pytest.raises(AssertionError, match='plan changed'):
        exposure.summarize(tmp_path, p)


def test_actual_trainer_twelve_toy_trajectories_and_backup_order(tmp_path, monkeypatch):
    """Full orchestration/real trainer, tiny model and fabricated probe scores."""
    trainer = exposure.gated.clip_train
    batch = toy_batches()[0]
    batch['species_ids'] = torch.arange(10)
    class Loader:
        def __init__(self):
            self.generator = torch.Generator().manual_seed(789)
        def __len__(self):
            return 77
        def __iter__(self):
            yield from [batch] * 77
    class Dataset:
        def set_epoch(self, epoch):
            self.epoch = epoch
    class Model(TinyModel):
        @property
        def clip_model(self):
            return SimpleNamespace(logit_scale=self.scale)
        def get_logit_scale(self):
            return self.scale.exp()
    import transformers
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda *a, **k: None)
    monkeypatch.setattr(trainer.CLIPEncoderBackend, 'from_pretrained', lambda *a, **k: Model())
    monkeypatch.setattr(trainer, 'resolve_device', lambda _: (torch.device('cpu'), {}))
    monkeypatch.setattr(trainer, 'build_clip_training_data', lambda *a, **k:
        SimpleNamespace(train_loader=Loader(), train_dataset=Dataset(), exclusion_report={}))
    monkeypatch.setattr(trainer, 'configure_clip_trainable_parameters', lambda *a: {})
    monkeypatch.setattr(trainer, 'build_clip_optimizer', lambda model, config:
        torch.optim.AdamW(model.parameters(), lr=.001))
    monkeypatch.setattr(trainer, 'evaluate_clip', lambda *a, **k: {
        'canonical_retrieval': {d: {'r_at_1': 1.} for d in ['text_to_image', 'image_to_text']},
        'species': {'top_1_accuracy': .5}})
    monkeypatch.setattr(exposure.gated, 'fixed_probe_batches', lambda *a: [batch] * 16)
    calls = []
    def measure(*args):
        value = [.04, .02, .04, .02][len(calls) % 4]
        calls.append(value)
        return {'batch_cosines': [value] * 16, 'mean_cosine': value}
    monkeypatch.setattr(exposure.gated, 'measure_alignment', measure)
    load_config = trainer.load_training_config
    def config(path):
        result = load_config(path)
        result['training']['mixed_precision'] = 'none'
        return result
    monkeypatch.setattr(trainer, 'load_training_config', config)
    synced_plans, snapshots = set(), set()
    def sync():
        assert (tmp_path / 'randomization_commitment.json').exists()
        rolling = tmp_path / 'rolling_checkpoint.pt'
        if rolling.exists():
            state = torch.load(rolling, weights_only=False)
            identity = (state['training_seed'], state['arm'], state['completed_updates'])
            if state['arm'] != 'alignment_gated':
                assert state['training_seed'] in synced_plans
            assert len(state['gate_state']['objective_decisions']) == state['completed_updates']
            assert 'optimizer' in state and 'scheduler' in state
            snapshots.add(identity)
        for seed in exposure.SEEDS:
            if (tmp_path / f'seed_{seed}/matched_schedules.json').exists():
                synced_plans.add(seed)
    result = exposure.run(tmp_path, sync)
    assert len(calls) == 12  # reference-only probes
    assert len(snapshots) == 60  # 12 trajectories x 5 backup boundaries
    assert len(result['rows']) == 12
    assert all(r['active_updates'] == 400 for r in result['rows'])
    assert result['mean_alignment_minus_mean_random_r1_pp'] == 0.
    assert {str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*.pt')} == set(exposure.required_paths()) | {'rolling_checkpoint.pt'}
    assert json.loads((tmp_path / 'completion.json').read_text())['status'] == 'complete'


def test_launcher_payload_pins_base_and_contains_package_fix():
    path = exposure.ROOT / 'colabs/clip_exposure_matched_drive_one_cell.py'
    if not path.exists():
        pytest.skip('Launcher not present in source-only test staging')
    text = path.read_text()
    module = ast.parse(text)
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)}
    raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == values['BUNDLE_SHA256']
    assert values['BASE_COMMIT'] == '7e7cfea90b60415dc9561efbe97bcd383cc1580e'
    sources = json.loads(raw)
    assert 'tests/__init__.py' in sources
    manifest = json.loads(sources['exposure_source_manifest.json'])
    for name, digest in manifest.items():
        assert sources[name] == (exposure.ROOT / name).read_text()
        assert hashlib.sha256(sources[name].encode()).hexdigest() == digest
    assert 'stdout=subprocess.PIPE' not in text
    assert 'drive.flush_and_unmount()' in text and 'PARTIAL_DRIVE_SYNC_COMPLETE' in text


def test_test_package_resolves_to_repo():
    import tests
    assert exposure.ROOT / 'tests/__init__.py' == __import__('pathlib').Path(tests.__file__)
