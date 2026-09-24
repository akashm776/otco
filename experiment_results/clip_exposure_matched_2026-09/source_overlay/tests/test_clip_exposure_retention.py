"""Tiny checkpoint, Drive failure and duplicate-run guards for next-stage runner."""

from types import SimpleNamespace

import pytest
import torch

from colabs import exposure_checkpoint_retention as retention
from colabs import run_clip_exposure_matched as runner
from colabs import transfer_drive_backup as backup
from tests.test_clip_gated_retention import mounted


def state(seed, arm, step):
    return {'training_seed': seed, 'arm': arm, 'completed_updates': step,
            'model': {'value': torch.ones(1)}, 'optimizer': {'moment': torch.zeros(1)}}


def test_exact_layout_and_all_sixty_atomic_backups(tmp_path, mounted):
    source = tmp_path / 'source'
    source.mkdir()
    old = mounted / 'old_run'
    old.mkdir()
    (old / 'keep.pt').write_bytes(b'preserve')
    mirror = backup.DriveMirror(source, mounted / 'new_run')
    for seed in retention.SEEDS:
        for arm in retention.ARMS:
            for step in retention.STEPS:
                before = torch.get_rng_state().clone()
                retention.save_checkpoint(source, state(seed, arm, step), seed=seed, arm=arm, step=step)
                assert torch.equal(before, torch.get_rng_state())
                mirror.sync()
    result = mirror.sync(final=True)
    assert retention.verify_inventory(result['files']) == 24
    assert len(list(source.rglob('*.pt'))) == len(list(mirror.destination.rglob('*.pt'))) == 25
    assert (old / 'keep.pt').read_bytes() == b'preserve'
    paths = retention.required_paths()
    assert sum('/shared/' in p for p in paths) == 3
    assert sum(p.endswith('step_001001.pt') for p in paths) == 12
    assert sum('/alignment_gated/' in p and not p.endswith('step_001001.pt') for p in paths) == 9


def test_failed_rolling_save_preserves_previous(tmp_path, monkeypatch):
    retention.save_checkpoint(tmp_path, state(789, 'alignment_gated', 100), seed=789, arm='alignment_gated', step=100)
    before = (tmp_path / retention.ROLLING).read_bytes()
    def fail(value, path):
        path.write_bytes(b'partial')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        retention.save_checkpoint(tmp_path, state(789, 'alignment_gated', 250), seed=789, arm='alignment_gated', step=250)
    assert (tmp_path / retention.ROLLING).read_bytes() == before
    assert not list(tmp_path.glob('.rolling-*'))


def test_failed_permanent_copy_does_not_publish(tmp_path, monkeypatch):
    def fail(source, target):
        target.write_bytes(b'partial')
        raise OSError('disk full')
    monkeypatch.setattr(retention.shutil, 'copyfile', fail)
    with pytest.raises(OSError):
        retention.save_checkpoint(tmp_path, state(789, 'alignment_gated', 100), seed=789, arm='alignment_gated', step=100)
    assert not (tmp_path / retention.permanent_path(789, 'alignment_gated', 100)).exists()
    assert not list(tmp_path.rglob('.snapshot-*'))


def test_checkpoint_identity_and_permanent_overwrite_rejected(tmp_path):
    s = state(789, 'alignment_gated', 100)
    retention.save_checkpoint(tmp_path, s, seed=789, arm='alignment_gated', step=100)
    with pytest.raises(FileExistsError):
        retention.save_checkpoint(tmp_path, s, seed=789, arm='alignment_gated', step=100)
    with pytest.raises(ValueError, match='metadata'):
        retention.save_checkpoint(tmp_path, s, seed=2026, arm='alignment_gated', step=100)


def test_inventory_rejects_missing_and_extra_weights():
    files = {p: {'bytes': 1} for p in retention.required_paths() + [retention.ROLLING]}
    assert retention.verify_inventory(files) == 24
    with pytest.raises(AssertionError):
        retention.verify_inventory({p: v for p, v in files.items() if p != retention.ROLLING})
    with pytest.raises(AssertionError):
        retention.verify_inventory({**files, 'best.pt': {'bytes': 1}})


def test_preflight_checks_mount_before_gpu(monkeypatch):
    monkeypatch.setattr(runner, 'require_drive', lambda _: (_ for _ in ()).throw(RuntimeError('not mounted')))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: pytest.fail('GPU checked too early'))
    with pytest.raises(RuntimeError, match='not mounted'):
        runner.preflight('a' * 64)


def test_duplicate_experiment_refused_even_with_different_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'require_drive', lambda _: None)
    monkeypatch.setattr(runner, 'DRIVE_ROOT', tmp_path)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'NVIDIA A100-SXM4-40GB, 40960\n')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=40 * 1024**3))
    assert 'A100' in runner.preflight('a' * 64)
    old = tmp_path / 'clip_exposure_matched_existing'
    old.mkdir()
    (old / 'run_manifest.json').write_text('{"experiment":"clip_exposure_matched_v1","bundle_id":"old"}')
    with pytest.raises(RuntimeError, match='Existing exposure'):
        runner.preflight('b' * 64)
