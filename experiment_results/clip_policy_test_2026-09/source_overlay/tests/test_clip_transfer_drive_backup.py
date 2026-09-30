"""Tiny fabricated file tests, never model results."""
import json
import random

import pytest

from colabs import transfer_drive_backup as backup


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    mount = tmp_path/'drive'
    (mount/'MyDrive').mkdir(parents=True)
    root = mount/'MyDrive/OTCO/evaluation_transfer'
    monkeypatch.setattr(backup, 'DRIVE_MOUNT', mount)
    monkeypatch.setattr(backup, 'DRIVE_ROOT', root)
    monkeypatch.setattr(backup.os.path, 'ismount', lambda p: p == mount)
    return root


def test_unmounted_drive_never_creates_local_imitation(tmp_path, monkeypatch):
    monkeypatch.setattr(backup.os.path, 'ismount', lambda _: False)
    target = tmp_path/'fake_drive/run'
    with pytest.raises(RuntimeError, match='not mounted'):
        backup.DriveMirror(tmp_path/'source', target).sync()
    assert not target.exists()


def test_mirror_incremental_and_final_readback(tmp_path, mounted):
    source = tmp_path/'run'
    source.mkdir()
    (source/'checkpoint.pt').write_bytes(b'fabricated closed checkpoint')
    (source/'report.json').write_text('{"step":100}')
    worker = backup.DriveMirror(source, mounted/'run')
    rng = random.getstate()
    first = worker.sync()
    assert random.getstate() == rng
    assert len(first['files']) == 2
    assert (mounted/'run/checkpoint.pt').read_bytes() == (source/'checkpoint.pt').read_bytes()
    (source/'report.json').write_text('{"step":250}')
    manifest = worker.sync(final=True)
    assert manifest['status'] == 'verified_tree'
    assert json.loads((mounted/'run/report.json').read_text()) == {'step':250}
    assert not list((mounted/'run').glob('.manifest-*'))
    (mounted/'run/checkpoint.pt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='verification failed'):
        worker.sync(final=True)


def test_copy_failure_stops_and_preserves_previous_file(tmp_path, mounted):
    source = tmp_path/'run'
    source.mkdir()
    (source/'report.json').write_text('{"complete":true}')
    worker = backup.DriveMirror(source, mounted/'run')
    worker.sync()
    before = (mounted/'run/report.json').read_bytes()
    (source/'report.json').write_text('{')
    with pytest.raises(ValueError):
        worker.sync()
    assert (mounted/'run/report.json').read_bytes() == before


def test_backup_cannot_escape_experiment_folder(tmp_path, mounted):
    with pytest.raises(ValueError, match='inside'):
        backup.require_drive(tmp_path/'unrelated')


def test_diagnostic_backup_follows_closed_snapshot_and_not_early_latest(monkeypatch):
    from src.clip_usefulness_prospective import ProspectiveObserver, IntermediateObserver
    calls = []
    monkeypatch.setattr(IntermediateObserver, '__call__',
                        lambda self, **kw: calls.append(('closed', kw['global_step'])))
    observer = object.__new__(ProspectiveObserver)
    observer.on_checkpoint = lambda: calls.append(('backup', None))
    for step in [0,100,250,500,750,1001]:
        observer(model=None, epoch=1, global_step=step)
    assert calls == [('closed',0)] + [item for step in [100,250,500,750]
        for item in [('closed',step),('backup',None)]] + [('closed',1001)]


def test_archive_copy_is_readback_verified(tmp_path, mounted):
    source = tmp_path/'archive.zip'
    source.write_bytes(b'archive fixture')
    record = backup.save_archive(source, mounted/'run/archive.zip')
    assert record['sha256'] == backup.checksum(mounted/'run/archive.zip')
