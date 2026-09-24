"""Hybrid storage tests using tiny fabricated states, never GPU training."""

import json
from types import SimpleNamespace

import pytest
import torch

from colabs import gated_checkpoint_retention as retention
from colabs import transfer_drive_backup as backup
from colabs.run_clip_gated_training import verify_inventory


def state(seed=789, arm='baseline', step=100):
    return {'training_seed': seed, 'arm': arm, 'completed_updates': step,
            'model': {'weight': torch.tensor([float(step)])},
            'optimizer': {'state': {0: {'exp_avg': torch.ones(1)}}},
            'scheduler': {'last_epoch': step}, 'gate_state': {}}


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    mount = tmp_path / 'drive'
    root = mount / 'MyDrive/OTCO/evaluation_transfer'
    root.mkdir(parents=True)
    monkeypatch.setattr(backup, 'DRIVE_MOUNT', mount)
    monkeypatch.setattr(backup, 'DRIVE_ROOT', root)
    monkeypatch.setattr(backup.os.path, 'ismount', lambda p: p == mount)
    return root


def test_exact_permanent_layout():
    paths = retention.required_paths()
    assert len(paths) == len(set(paths)) == 24
    assert sum('/shared/' in p for p in paths) == 3
    assert sum(p.endswith('step_001001.pt') for p in paths) == 12
    assert sum('/alignment_gated/' in p and not p.endswith('step_001001.pt') for p in paths) == 9
    for arm in ['always_on', 'step_gated', 'alignment_gated']:
        assert retention.permanent_path(789, arm, 100) is None
    for step in [250, 500, 750]:
        for arm in ['baseline', 'always_on', 'step_gated']:
            assert retention.permanent_path(789, arm, step) is None


def test_sixty_save_events_leave_only_24_permanent_and_one_rolling(tmp_path, mounted):
    source = tmp_path / 'run'
    source.mkdir()
    old = mounted / 'unrelated_completed_run'
    old.mkdir()
    (old / 'checkpoint.pt').write_bytes(b'Existing results must not be touched')
    mirror = backup.DriveMirror(source, mounted / 'hybrid_run')
    for seed in retention.SEEDS:
        for arm in retention.ARMS:
            for step in retention.STEPS:
                value = state(seed, arm, step)
                before = torch.get_rng_state().clone()
                retention.save_hybrid_checkpoint(source, value, seed=seed, arm=arm, step=step)
                assert torch.equal(before, torch.get_rng_state())
                mirror.sync()
                restored = torch.load(mirror.destination / retention.ROLLING, weights_only=False)
                assert (restored['training_seed'], restored['arm'], restored['completed_updates']) == (seed, arm, step)
                torch.testing.assert_close(restored['optimizer']['state'][0]['exp_avg'], torch.ones(1))
    manifest = mirror.sync(final=True)
    assert verify_inventory(manifest['files']) == 24
    assert len(list(source.rglob('*.pt'))) == len(list(mirror.destination.rglob('*.pt'))) == 25
    assert (old / 'checkpoint.pt').read_bytes() == b'Existing results must not be touched'
    saved = json.loads((mirror.destination / 'DRIVE_BACKUP_MANIFEST.json').read_text())
    assert saved['status'] == 'verified_tree' and len(saved['files']) == 25


def test_rolling_write_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    retention.save_hybrid_checkpoint(tmp_path, state(), seed=789, arm='baseline', step=100)
    before = (tmp_path / retention.ROLLING).read_bytes()
    def fail(value, path):
        path.write_bytes(b'partial checkpoint')
        raise OSError('fabricated disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        retention.save_hybrid_checkpoint(tmp_path, state(step=250), seed=789, arm='baseline', step=250)
    assert (tmp_path / retention.ROLLING).read_bytes() == before
    assert not list(tmp_path.glob('.rolling-*'))


def test_permanent_copy_failure_does_not_publish_partial_state(tmp_path, monkeypatch):
    def fail(source, destination):
        destination.write_bytes(b'partial')
        raise OSError('fabricated disk full')
    monkeypatch.setattr(retention.shutil, 'copyfile', fail)
    with pytest.raises(OSError):
        retention.save_hybrid_checkpoint(tmp_path, state(), seed=789, arm='baseline', step=100)
    assert not (tmp_path / retention.permanent_path(789, 'baseline', 100)).exists()
    assert not list(tmp_path.rglob('.snapshot-*'))
    assert torch.load(tmp_path / retention.ROLLING, weights_only=False)['completed_updates'] == 100


def test_rolling_drive_failure_preserves_last_verified_backup(tmp_path, mounted, monkeypatch):
    source = tmp_path / 'run'
    source.mkdir()
    mirror = backup.DriveMirror(source, mounted / 'run')
    retention.save_hybrid_checkpoint(source, state(), seed=789, arm='baseline', step=100)
    mirror.sync()
    before = (mirror.destination / retention.ROLLING).read_bytes()
    retention.save_hybrid_checkpoint(source, state(step=250), seed=789, arm='baseline', step=250)
    def fail(*args):
        raise OSError('fabricated Drive quota failure')
    monkeypatch.setattr(backup, 'copy_verified', fail)
    with pytest.raises(OSError, match='quota'):
        mirror.sync()
    assert (mirror.destination / retention.ROLLING).read_bytes() == before


def test_permanent_states_and_identity_are_protected(tmp_path):
    retention.save_hybrid_checkpoint(tmp_path, state(), seed=789, arm='baseline', step=100)
    with pytest.raises(FileExistsError):
        retention.save_hybrid_checkpoint(tmp_path, state(), seed=789, arm='baseline', step=100)
    with pytest.raises(ValueError, match='metadata'):
        retention.save_hybrid_checkpoint(tmp_path, state(), seed=2026, arm='baseline', step=100)


def test_missing_rolling_file_is_not_a_complete_inventory():
    with pytest.raises(AssertionError, match='Incomplete'):
        verify_inventory({p: {'bytes': 1} for p in retention.required_paths()})


def test_local_space_check_is_40_gib_not_80(tmp_path, monkeypatch):
    from colabs import run_clip_gated_training as runner
    monkeypatch.setattr(runner, 'require_drive', lambda _: None)
    monkeypatch.setattr(runner, 'DRIVE_ROOT', tmp_path)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'NVIDIA A100-SXM4-40GB, 40960\n')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=40*1024**3))
    assert 'A100' in runner.preflight('0'*64)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=39*1024**3))
    with pytest.raises(RuntimeError, match='40 GiB'):
        runner.preflight('0'*64)
