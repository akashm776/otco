"""Small, CPU-only tests of archive recovery; never deserialize checkpoints."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from colabs.checkpoint_diagnostic_inputs import recover_inputs


def fixture_archive(tmp_path, change=None):
    payload = b'opaque checkpoint bytes'
    relative = 'seed_789/shared/checkpoints/step_000100.pt'
    sha = hashlib.sha256(payload).hexdigest()
    protocol = dict(source_run='original_run', source_bundle='a' * 64,
                    checkpoints={relative: sha})
    manifest = dict(run_id='original_run', bundle_id='a' * 64,
                    status='complete_pending_drive_flush')
    backup = dict(run_id='original_run', status='verified_tree',
                  files={relative: dict(bytes=len(payload), sha256=sha)})
    members = {relative: payload, 'run_manifest.json': json.dumps(manifest).encode(),
               'DRIVE_BACKUP_MANIFEST.json': json.dumps(backup).encode()}
    if change:
        change(members, relative)
    archive = tmp_path/'inputs.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, value in members.items():
            z.writestr('original_run/' + name, value)
    return archive, tmp_path/'restored', protocol, members, relative


def test_recovery_preserves_bytes_and_records_subset(tmp_path):
    archive, destination, protocol, members, relative = fixture_archive(tmp_path)
    assert recover_inputs(archive, destination, protocol) == destination
    for name, expected in members.items():
        assert (destination/name).read_bytes() == expected
    scope = json.loads((destination/'RESTORED_SUBSET_MANIFEST.json').read_text())
    assert scope['restored_subset_only'] is True
    assert scope['files'][relative]['sha256'] == protocol['checkpoints'][relative]
    before = (destination/relative).stat().st_mtime_ns
    recover_inputs(archive, destination, protocol)
    assert (destination/relative).stat().st_mtime_ns == before


def test_corrupt_checkpoint_never_published(tmp_path):
    def change(members, relative):
        members[relative] = b'x' * len(members[relative])
    archive, destination, protocol, _, relative = fixture_archive(tmp_path, change)
    with pytest.raises(ValueError, match='checksum mismatch'):
        recover_inputs(archive, destination, protocol)
    assert not (destination/relative).exists()
    assert not list(destination.rglob('.recover-*'))


@pytest.mark.parametrize('kind', ['missing', 'traversal', 'wrong_bundle', 'wrong_pin', 'wrong_size'])
def test_invalid_archive_rejected_before_extraction(tmp_path, kind):
    def change(members, relative):
        if kind == 'missing':
            del members[relative]
        elif kind == 'traversal':
            members['../../outside'] = b'bad'
        elif kind == 'wrong_bundle':
            m = json.loads(members['run_manifest.json'])
            m['bundle_id'] = 'b' * 64
            members['run_manifest.json'] = json.dumps(m).encode()
        else:
            m = json.loads(members['DRIVE_BACKUP_MANIFEST.json'])
            m['files'][relative]['sha256' if kind == 'wrong_pin' else 'bytes'] = (
                'b' * 64 if kind == 'wrong_pin' else 999)
            members['DRIVE_BACKUP_MANIFEST.json'] = json.dumps(m).encode()
    archive, destination, protocol, _, _ = fixture_archive(tmp_path, change)
    with pytest.raises(ValueError):
        recover_inputs(archive, destination, protocol)
    assert not destination.exists()
    assert not (tmp_path/'outside').exists()


def test_duplicate_member_rejected(tmp_path):
    archive, destination, protocol, members, relative = fixture_archive(tmp_path)
    with zipfile.ZipFile(archive, 'a') as z, pytest.warns(UserWarning, match='Duplicate'):
        z.writestr('original_run/'+relative, members[relative])
    with pytest.raises(ValueError, match='exactly'):
        recover_inputs(archive, destination, protocol)


def test_changed_existing_file_not_overwritten(tmp_path):
    archive, destination, protocol, _, relative = fixture_archive(tmp_path)
    target = destination/relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b'user data')
    with pytest.raises(ValueError, match='overwrite changed'):
        recover_inputs(archive, destination, protocol)
    assert target.read_bytes() == b'user data'


def test_symlink_destination_refused(tmp_path):
    archive, destination, protocol, _, _ = fixture_archive(tmp_path)
    outside = tmp_path/'outside'
    outside.mkdir()
    destination.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        recover_inputs(archive, destination, protocol)
    assert not list(outside.iterdir())


def test_disk_capacity_checked_before_copy(tmp_path, monkeypatch):
    archive, destination, protocol, _, relative = fixture_archive(tmp_path)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr('colabs.checkpoint_diagnostic_inputs.shutil.disk_usage',
                        lambda _: usage._replace(free=0))
    with pytest.raises(RuntimeError, match='Insufficient local disk'):
        recover_inputs(archive, destination, protocol)
    assert not (destination/relative).exists()


def test_launcher_uses_zip_when_original_drive_folder_missing(tmp_path, monkeypatch):
    from colabs import run_clip_checkpoint_diagnostic as launcher
    from colabs.checkpoint_diagnostic_inputs import ARCHIVE_NAME
    archive, _, protocol, _, relative = fixture_archive(tmp_path)
    protocol['experiment'] = 'test_diagnostic'
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/clip_checkpoint_diagnostic.json').write_text(json.dumps(protocol))
    archive.rename(tmp_path/ARCHIVE_NAME)
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'DRIVE_ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'require_drive', lambda _: None)
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: 'NVIDIA A100, 40960\n')
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda _: usage._replace(free=100 * 1024**3))
    actual_recover = recover_inputs
    def local_recover(zip_path, destination, p):
        assert destination == Path('/content/otco_diagnostic_inputs/original_run')
        return actual_recover(zip_path, tmp_path/'local', p)
    monkeypatch.setattr(launcher, 'recover_inputs', local_recover)
    source, manifest, _ = launcher.preflight('f' * 64)
    assert source == tmp_path/'local'
    assert (source/relative).is_file()
    assert manifest['run_id'] == protocol['source_run']
    assert not (tmp_path/'original_run').exists()
