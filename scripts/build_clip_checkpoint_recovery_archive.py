"""Select and SHA256-verify the 12 diagnostic checkpoints from Downloads ZIPs."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from colabs.checkpoint_diagnostic_inputs import ARCHIVE_NAME, METADATA

ROOT = Path(__file__).resolve().parents[1]


def build(downloads, output):
    p = json.loads((ROOT/'configs/clip_checkpoint_diagnostic.json').read_text())
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    archives = sorted(Path(downloads).glob(p['source_run']+'-*.zip'))
    required = METADATA + list(p['checkpoints'])
    with ExitStack() as stack:
        sources = [stack.enter_context(zipfile.ZipFile(path)) for path in archives]
        found = {}
        for path, z in zip(archives, sources):
            for name in z.namelist():
                for relative in required:
                    if name == p['source_run']+'/'+relative:
                        if relative in found:
                            raise ValueError(f'Duplicate input member: {relative}')
                        found[relative] = (path, z, name)
        if set(found) != set(required):
            raise FileNotFoundError(f'Missing input members: {set(required)-set(found)}')
        read = lambda relative: found[relative][1].read(found[relative][2])
        backup = json.loads(read('DRIVE_BACKUP_MANIFEST.json'))
        manifest = json.loads(read('run_manifest.json'))
        if manifest['bundle_id'] != p['source_bundle'] or manifest['run_id'] != p['source_run']:
            raise ValueError('Wrong source run')
        records = {}
        for relative in required:
            if relative in p['checkpoints']:
                records[relative] = backup['files'][relative]
                if records[relative]['sha256'] != p['checkpoints'][relative]:
                    raise ValueError('Checkpoint manifest disagrees with frozen diagnostic pins')
            else:
                raw = read(relative)
                records[relative] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.recovery-', suffix='.zip', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            # ZIP_STORED avoids spending minutes recompressing dense FP32 weights.
            # The reduction comes from retaining 12 checkpoints instead of 25.
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as target:
                for relative in required:
                    path, z, name = found[relative]
                    expected = records[relative]
                    result, size = hashlib.sha256(), 0
                    with z.open(name) as src, target.open(name, 'w', force_zip64=True) as out:
                        for chunk in iter(lambda: src.read(8*1024**2), b''):
                            size += len(chunk)
                            result.update(chunk)
                            out.write(chunk)
                    if size != expected['bytes'] or result.hexdigest() != expected['sha256']:
                        raise ValueError(f'Input checkpoint verification failed: {relative}')
                    print('PACKED_VERIFIED:', relative, 'from', path.name, flush=True)
            with zipfile.ZipFile(temporary) as z:
                for relative, expected in records.items():
                    result = hashlib.sha256()
                    with z.open(p['source_run']+'/'+relative) as src:
                        for chunk in iter(lambda: src.read(8*1024**2), b''):
                            result.update(chunk)
                    if result.hexdigest() != expected['sha256']:
                        raise ValueError(f'Output archive read-back failed: {relative}')
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print('RECOVERY_ARCHIVE_VERIFIED:', output, 'bytes=', output.stat().st_size, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--downloads', type=Path, default=Path('/Users/akashmittal/Downloads'))
    parser.add_argument('--output', type=Path, default=ROOT/'output/recovery'/ARCHIVE_NAME)
    args = parser.parse_args()
    build(args.downloads, args.output)
