import json

import pytest

from colabs.backup_clip_run import RunBackup, backup_sources, copy_verified, checksum


def test_verified_copy_and_checksum(tmp_path):
    source, target = tmp_path / 'source.json', tmp_path / 'drive/result.json'
    source.write_text('{"step": 100}')
    entry, _ = copy_verified(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert entry['sha256'] == checksum(target)
    assert not list(target.parent.glob('.upload-*'))


def test_partial_json_keeps_previous_good_copy(tmp_path):
    source, target = tmp_path / 'source.json', tmp_path / 'copy.json'
    source.write_text('{"complete": true}')
    copy_verified(source, target)
    previous = target.read_bytes()
    source.write_text('{"in_progress":')
    with pytest.raises(ValueError):
        copy_verified(source, target)
    assert target.read_bytes() == previous


def test_changing_source_is_not_published(tmp_path, monkeypatch):
    import colabs.backup_clip_run as backup
    source, target = tmp_path / 'source.txt', tmp_path / 'copy.txt'
    source.write_text('initial')
    original = backup.shutil.copyfile
    def racing_copy(src, dst):
        original(src, dst)
        src.write_text('changed during upload')
    monkeypatch.setattr(backup.shutil, 'copyfile', racing_copy)
    with pytest.raises(RuntimeError):
        copy_verified(source, target)
    assert not target.exists()


def test_sync_retries_and_updates_manifest(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    source = output / 'report.json'
    source.write_text('{"step":100}')
    worker = RunBackup(output, tmp_path / 'checkpoints', tmp_path / 'drive')
    assert worker.sync()['copied_this_pass'] == 1
    assert worker.sync()['copied_this_pass'] == 0
    source.write_text('{')
    assert worker.sync()['errors']
    source.write_text('{"step":200}')
    assert worker.sync()['copied_this_pass'] == 1
    manifest = json.loads((tmp_path / 'drive/backup_manifest.json').read_text())
    assert not manifest['errors']
    assert manifest['verified_files']['results/report.json']['sha256'] == checksum(source)


def test_checkpoint_and_feature_readiness_markers(tmp_path):
    output, checkpoint = tmp_path / 'output', tmp_path / 'checkpoint'
    stage = output / 'baseline/diagnostics/stage'
    stage.mkdir(parents=True)
    (stage / 'features.pt').write_bytes(b'tensor fixture')
    arm = checkpoint / 'baseline'
    arm.mkdir(parents=True)
    (arm / 'common_step_100.pt').write_bytes(b'common fixture')
    (arm / 'latest.pt').write_bytes(b'latest fixture')
    assert not list(backup_sources(output, checkpoint))
    (stage / 'projection_probe.json').write_text('[]')
    (output / 'calibration.json').write_text('{}')
    selected = {str(relative) for _, relative in backup_sources(output, checkpoint)}
    assert 'results/baseline/diagnostics/stage/features.pt' in selected
    assert 'checkpoints/baseline/common_step_100.pt' in selected
    assert 'checkpoints/baseline/latest.pt' not in selected
    (output / 'baseline/completion.json').write_text('{"status":"complete"}')
    assert 'checkpoints/baseline/latest.pt' in {str(r) for _, r in backup_sources(output, checkpoint)}
