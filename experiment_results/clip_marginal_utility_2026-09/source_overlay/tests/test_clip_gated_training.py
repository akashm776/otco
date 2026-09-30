"""Offline tests with fabricated tensors; never claimed as research results."""

from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import clip_gated_training as gated
from src.clip_early_pulse import auxiliary_losses, state_digest
from src.clip_gradient_stages import observational_state
from model.clip_training import native_clip_contrastive_loss
from tests.test_clip_early_pulse import TinyModel, output_fixture


@pytest.fixture(autouse=True)
def isolate_rng():
    with observational_state(torch.nn.Identity()):
        yield


@pytest.mark.parametrize('arm,count', [('baseline', 0), ('always_on', 901),
    ('step_gated', 76), ('alignment_gated', 901)])
def test_exact_schedule(arm, count):
    p = gated.load_protocol()
    active = [s for s in range(1001) if gated.gate_active(arm, s, p, .04)]
    assert len(active) == count
    assert all(s >= 100 for s in active)
    if arm == 'step_gated':
        assert active == list(range(100, 176))


def test_frozen_gate_strict_boundary_and_no_outcome_input():
    p = gated.load_protocol()
    t = p['alignment_threshold']
    assert not gated.gate_active('alignment_gated', 100, p, t)
    assert gated.gate_active('alignment_gated', 100, p, t + 1e-10)
    assert not gated.gate_active('alignment_gated', 99, p)
    for value in [None, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            gated.gate_active('alignment_gated', 100, p, value)
    for arm, step in [('unknown', 100), ('baseline', -1), ('baseline', 1001)]:
        with pytest.raises(ValueError):
            gated.gate_active(arm, step, p)


@pytest.mark.parametrize('arm', gated.ARMS)
def test_objective_exact_loss_and_gradients(arm):
    p = gated.load_protocol()
    objective = gated.GatedObjective(arm, p)
    objective.alignment = .04
    for step in [0, 99, 100, 175, 176, 999, 1000]:
        out = output_fixture()
        actual = objective(out, species_ids=None, step=step)
        active = gated.gate_active(arm, step, p, .04)
        expected = native_clip_contrastive_loss(out.logits)
        if active:
            expected = expected + p['uniform_top8_coefficient'] * auxiliary_losses(out)['uniform_top8']
        torch.testing.assert_close(actual.total_loss, expected, rtol=0, atol=0)
        params = (out.image_features, out.text_features, out.logit_scale)
        a = torch.autograd.grad(actual.total_loss, params, retain_graph=True)
        b = torch.autograd.grad(expected, params)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def toy_batches():
    return [{'pixel_values': torch.randn(10, 6), 'input_ids': torch.randn(10, 6)}
            for _ in range(16)]


def test_probe_preserves_modes_rng_grads_and_buffers_and_is_repeatable():
    model = TinyModel()
    model.text.eval()
    model.register_buffer('counter', torch.zeros(()))
    def count(m, args, out):
        m.counter.add_(1)
    model.register_forward_hook(count)
    batches = toy_batches()
    config = {'training': {'mixed_precision': 'none'}}
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    def snapshot():
        return state_digest({'model': model.state_dict(),
            'grad': [p.grad for p in model.parameters()], 'python': random.getstate(),
            'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'modes': [m.training for m in model.modules()]})
    before = snapshot()
    first = gated.measure_alignment(model, batches, config, 123)
    assert snapshot() == before
    assert first == gated.measure_alignment(model, batches, config, 123)
    assert len(first['batch_cosines']) == 16
    assert snapshot() == before


def test_probe_does_not_change_subsequent_optimizer_update():
    model = TinyModel()
    clone = deepcopy(model)
    batches = toy_batches()
    config = {'training': {'mixed_precision': 'none'}}
    gated.measure_alignment(model, batches, config, 123)
    for current in (model, clone):
        opt = torch.optim.AdamW(current.parameters(), lr=.01)
        out = current(batches[0]['pixel_values'], {'input_ids': batches[0]['input_ids']})
        native_clip_contrastive_loss(out.logits).backward()
        opt.step()
    assert state_digest(model.state_dict()) == state_digest(clone.state_dict())


def test_probe_failure_restores_rng_and_buffers(monkeypatch):
    model = TinyModel()
    model.register_buffer('counter', torch.zeros(()))
    def fail(m, args, out):
        m.counter.add_(1)
        torch.rand(4)
        raise RuntimeError('fabricated probe failure')
    model.register_forward_hook(fail)
    batches = toy_batches()
    before = state_digest({'state': model.state_dict(), 'rng': torch.get_rng_state()})
    with pytest.raises(RuntimeError, match='fabricated'):
        gated.measure_alignment(model, batches, {'training': {'mixed_precision': 'none'}}, 123)
    assert before == state_digest({'state': model.state_dict(), 'rng': torch.get_rng_state()})
    assert model.training


def fabricated_audit(arm):
    p = gated.load_protocol()
    probes = [dict(completed_updates=s, batch_cosines=[v]*16, mean_cosine=v, seconds=0.)
              for s, v in zip(p['alignment_refresh_steps'], [.04, .02, .04, .02])]
    if arm != 'alignment_gated':
        probes = []
    decisions = []
    for step in range(1001):
        previous = [r for r in probes if r['completed_updates'] <= step]
        alignment = previous[-1]['mean_cosine'] if previous else None
        decisions.append(dict(objective_step=step, active=gated.gate_active(arm, step, p, alignment)))
    return dict(probes=probes, decisions=decisions, active_updates=sum(d['active'] for d in decisions))


@pytest.mark.parametrize('arm', gated.ARMS)
def test_audit_verification_and_tamper_rejection(arm):
    p, audit = gated.load_protocol(), fabricated_audit(arm)
    gated.verify_decisions(arm, audit, p)
    audit['decisions'][500]['active'] = not audit['decisions'][500]['active']
    with pytest.raises(AssertionError):
        gated.verify_decisions(arm, audit, p)


def test_audit_requires_sixteen_probes_and_all_updates():
    p = gated.load_protocol()
    audit = fabricated_audit('alignment_gated')
    audit['probes'][0]['batch_cosines'].pop()
    with pytest.raises(AssertionError):
        gated.verify_decisions('alignment_gated', audit, p)
    audit = fabricated_audit('baseline')
    audit['decisions'].pop()
    with pytest.raises(AssertionError):
        gated.verify_decisions('baseline', audit, p)


def test_inventory_requires_hybrid_states_and_rejects_duplicates():
    from colabs.run_clip_gated_training import verify_inventory
    from colabs.gated_checkpoint_retention import required_paths, ROLLING
    files = {name: {'bytes': 1} for name in required_paths() + [ROLLING]}
    assert verify_inventory(files) == 24
    extra = dict(files, **{'seed_789/baseline/checkpoints/latest.pt': {'bytes': 1}})
    with pytest.raises(AssertionError, match='duplicates'):
        verify_inventory(extra)
    files.pop(next(iter(files)))
    with pytest.raises(AssertionError):
        verify_inventory(files)


def test_preflight_rejects_unmounted_drive_before_gpu(monkeypatch):
    from colabs import run_clip_gated_training as runner
    def fail(_):
        raise RuntimeError('Drive not mounted')
    monkeypatch.setattr(runner, 'require_drive', fail)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: pytest.fail('GPU check before mount'))
    with pytest.raises(RuntimeError, match='not mounted'):
        runner.preflight('0'*64)


def test_final_comparison_reports_paired_seed_effects(tmp_path):
    p = gated.load_protocol()
    for seed in p['training_seeds']:
        for i, arm in enumerate(gated.ARMS):
            path = tmp_path / f'seed_{seed}' / arm
            (path / 'training').mkdir(parents=True)
            summary = {'execution': {'completed_updates': 1001}, 'final': {'evaluation': {
                'canonical_retrieval': {d: {'r_at_1': 1.+i} for d in ['text_to_image','image_to_text']},
                'species': {'top_1_accuracy': .5}}}}
            (path / 'training/summary.json').write_text(json.dumps(summary))
            (path / 'gate_audit.json').write_text(json.dumps(fabricated_audit(arm)))
            (path / 'timing.json').write_text('{"wall_seconds":10}')
    result = gated.summarize(tmp_path, p)
    assert result['mean_alignment_minus_comparator_r1_pp'] == {
        'baseline': 3., 'step_gated': 1., 'always_on': 2.}
    assert len(result['rows']) == 12 and len(result['per_seed_differences']) == 9


def test_real_trainer_four_arm_toy_rollout_prefix_and_backups(tmp_path, monkeypatch):
    """Exercise actual trainer/observer/objective ordering without downloads."""
    trainer = gated.clip_train
    p = gated.load_protocol()
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
    monkeypatch.setattr(trainer, 'evaluate_clip', lambda *a, **k: {'species': {'top_1_accuracy': .5}})
    monkeypatch.setattr(gated, 'fixed_probe_batches', lambda *a: [batch]*16)
    backups = []
    shared = tmp_path / 'seed_789'
    shared.mkdir()
    scratch = tmp_path.parent / (tmp_path.name + '_trainer_scratch')
    for arm in gated.ARMS:
        directory = shared / arm
        directory.mkdir()
        config = trainer.load_training_config(gated.ROOT / p['baseline_config'])
        config['training']['seed'] = config['experiment']['seed'] = 789
        config['training']['mixed_precision'] = 'none'
        config['diagnostics']['separate_projection_gradient_steps'] = 0
        observer = gated.RolloutObserver(directory, shared, arm, p,
            lambda: backups.append((arm, torch.load(tmp_path/'rolling_checkpoint.pt',
                weights_only=False)['completed_updates'])))
        summary = trainer.run(config, output_directory=directory/'training',
            checkpoint_directory=scratch, observer=observer,
            stop_after_epochs=13, objective_factory=lambda _: gated.GatedObjective(arm, p))
        assert summary['execution']['completed_updates'] == 1001
        observer.write_audit()
        audit = json.loads((directory/'gate_audit.json').read_text())
        gated.verify_decisions(arm, audit, p)
        assert [c for a,c in backups if a == arm] == [100,250,500,750,1001]
        state = torch.load(directory/'checkpoints/step_001001.pt', weights_only=False)
        assert state['completed_updates'] == 1001
        assert len(state['gate_state']['objective_decisions']) == 1001
        assert 'optimizer' in state and 'scheduler' in state
    prefix = json.loads((shared/'baseline/prefix_hashes.json').read_text())
    assert all(json.loads((shared/arm/'prefix_hashes.json').read_text()) == prefix for arm in gated.ARMS)
    from colabs.gated_checkpoint_retention import required_paths, ROLLING
    assert {str(f.relative_to(tmp_path)) for f in tmp_path.rglob('*.pt')} == {
        name for name in required_paths() if name.startswith('seed_789/')} | {ROLLING}


def test_launcher_payload_matches_source_and_pins_base():
    # No cell execution, downloads or training.
    import ast
    import base64
    import zlib
    path = gated.ROOT / 'colabs/clip_gated_training_drive_one_cell.py'
    if not path.exists():
        pytest.skip('Generate launcher after source tests pass')
    module = ast.parse(path.read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body
              if isinstance(node, ast.Assign)}
    raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == values['BUNDLE_SHA256']
    assert values['BASE_COMMIT'] == '7e7cfea90b60415dc9561efbe97bcd383cc1580e'
    for name, source in json.loads(raw).items():
        if name != 'gated_source_manifest.json':
            assert source == (gated.ROOT / name).read_text()
