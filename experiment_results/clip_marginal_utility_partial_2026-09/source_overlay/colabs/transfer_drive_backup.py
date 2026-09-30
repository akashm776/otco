"""Synchronous closed-file backups. No training/RNG operations or deletions."""

import json
import os
from pathlib import Path
import tempfile

from colabs.backup_clip_run import checksum, copy_verified, signature

DRIVE_MOUNT = Path('/content/drive')
DRIVE_ROOT = DRIVE_MOUNT / 'MyDrive/OTCO/evaluation_transfer'


def require_drive(destination):
    destination = Path(destination)
    if not os.path.ismount(DRIVE_MOUNT) or not (DRIVE_MOUNT / 'MyDrive').is_dir():
        raise RuntimeError('Google Drive is not mounted; refusing a fake local backup. Mount Drive first.')
    if not destination.resolve().is_relative_to(DRIVE_ROOT.resolve()):
        raise ValueError('Backups must stay inside MyDrive/OTCO/evaluation_transfer')


class DriveMirror:
    def __init__(self, source, destination):
        self.source, self.destination = Path(source), Path(destination)
        self.signatures, self.inventory = {}, {}

    def sync(self, *, final=False):
        require_drive(self.destination)
        for source in sorted(self.source.rglob('*')):
            if not source.is_file() or source.is_symlink() or any(
                    part.startswith('.') for part in source.relative_to(self.source).parts):
                continue
            relative = str(source.relative_to(self.source))
            target = self.destination / relative
            if self.signatures.get(relative) != signature(source) or not target.is_file():
                record, stamp = copy_verified(source, target)
                self.inventory[relative], self.signatures[relative] = record, stamp
                if source.suffix == '.pt':
                    print('DRIVE_CHECKPOINT_VERIFIED:', relative, record['sha256'], flush=True)
            if final and (target.stat().st_size != self.inventory[relative]['bytes']
                          or checksum(target) != self.inventory[relative]['sha256']):
                raise RuntimeError(f'Drive read-back verification failed: {relative}')
        self.destination.mkdir(parents=True, exist_ok=True)
        manifest = dict(run_id=self.source.name, status='verified_tree' if final else 'in_progress',
                        verification='Mounted Drive read-back SHA256; final Drive flush is separate',
                        files=self.inventory)
        with tempfile.NamedTemporaryFile(mode='w', dir=self.destination,
                prefix='.manifest-', delete=False) as handle:
            json.dump(manifest, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(self.destination / 'DRIVE_BACKUP_MANIFEST.json')
        return manifest


def save_archive(archive, destination):
    require_drive(destination)
    record, _ = copy_verified(Path(archive), Path(destination))
    print('DRIVE_FILE_VERIFIED:', destination, record['sha256'], flush=True)
    return record
