"""Recover only authenticated diagnostic inputs from a user-uploaded ZIP.

No generic extractall, pickle loading, source replacement, or training here.
The old Drive backup manifest is retained as provenance, not as a claim that
the entire original run has been restored.
"""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ARCHIVE_NAME = 'clip_checkpoint_diagnostic_inputs.zip'
METADATA = ['run_manifest.json', 'DRIVE_BACKUP_MANIFEST.json']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def recover_inputs(archive, destination, protocol):
    """Write a verified local subset; refuse symlinks/changed existing files."""
    archive, destination = Path(archive), Path(destination)
    run = protocol['source_run']
    prefix = run + '/'
    required = METADATA + list(protocol['checkpoints'])
    # Protocol paths are embedded in the source bundle, not supplied by the ZIP.
    for relative in required:
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe protocol path')
    if destination.is_symlink():
        raise ValueError('Refusing symlink recovery destination')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        wanted = {prefix + relative for relative in required}
        if set(names) != wanted or len(names) != len(wanted):
            raise ValueError('Recovery ZIP must contain exactly the required 12 checkpoints and two manifests')
        if any(z.getinfo(prefix+n).file_size > 5 * 1024**2 for n in METADATA):
            raise ValueError('Oversized metadata')
        manifest = json.loads(z.read(prefix + 'run_manifest.json'))
        backup = json.loads(z.read(prefix + 'DRIVE_BACKUP_MANIFEST.json'))
        if (manifest.get('run_id') != run or manifest.get('bundle_id') != protocol['source_bundle']
                or manifest.get('status') != 'complete_pending_drive_flush'
                or backup.get('run_id') != run or backup.get('status') != 'verified_tree'):
            raise ValueError('Recovery archive is not the required completed source run')
        records = {}
        for relative, expected in protocol['checkpoints'].items():
            record = backup['files'][relative]
            if record['sha256'] != expected or z.getinfo(prefix+relative).file_size != record['bytes']:
                raise ValueError(f'Checkpoint manifest/size mismatch: {relative}')
            records[relative] = dict(record)
        for relative in METADATA:
            data = z.read(prefix+relative)
            records[relative] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        missing_bytes = 0
        for relative, record in records.items():
            target = destination/relative
            if any(parent.is_symlink() for parent in [target, *target.parents]):
                raise ValueError('Refusing symlink in recovery path')
            if target.exists():
                if not target.is_file() or target.stat().st_size != record['bytes'] or digest(target) != record['sha256']:
                    raise ValueError(f'Refusing to overwrite changed recovery file: {target}')
            else:
                missing_bytes += record['bytes']
        destination.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination).free < missing_bytes + 1024**3:
            raise RuntimeError('Insufficient local disk for checkpoint recovery plus 1 GiB safety margin')
        for relative, record in records.items():
            target = destination/relative
            if target.exists():
                print('INPUT_ALREADY_VERIFIED:', relative, flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                result = hashlib.sha256()
                size = 0
                with z.open(prefix+relative) as src, tempfile.NamedTemporaryFile(
                        dir=target.parent, prefix='.recover-', delete=False) as out:
                    temporary = Path(out.name)
                    for chunk in iter(lambda: src.read(8 * 1024**2), b''):
                        size += len(chunk)
                        if size > record['bytes']:
                            raise ValueError('ZIP member exceeded declared size')
                        result.update(chunk)
                        out.write(chunk)
                if size != record['bytes'] or result.hexdigest() != record['sha256']:
                    raise ValueError(f'Checkpoint checksum mismatch: {relative}')
                # Verify the closed on-disk file too, before publishing it.
                if digest(temporary) != record['sha256']:
                    raise ValueError(f'Recovery read-back mismatch: {relative}')
                temporary.replace(target)
                print('INPUT_RECOVERED_VERIFIED:', relative, flush=True)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        scope = dict(source_run=run, archive=str(archive), restored_subset_only=True,
                     original_manifests_preserved=True, files=records,
                     verification='SHA256 pins before any checkpoint deserialization')
        (destination/'RESTORED_SUBSET_MANIFEST.json').write_text(json.dumps(scope, indent=2))
    return destination
