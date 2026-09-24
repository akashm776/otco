"""Generate the immutable-base, self-contained Drive-backed rollout cell."""

import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
BASE = '7e7cfea90b60415dc9561efbe97bcd383cc1580e'
FILES = [
    'configs/clip_gated_training.json', 'src/clip_gated_training.py',
    'colabs/run_clip_gated_training.py', 'colabs/transfer_drive_backup.py',
    'colabs/gated_checkpoint_retention.py', 'tests/test_clip_gated_retention.py',
    'tests/test_clip_gated_training.py', 'tests/test_clip_transfer_drive_backup.py',
    'src/clip_usefulness_prospective.py', 'src/clip_paired_updates.py',
    'src/clip_evaluation_transfer.py',
    'tests/test_clip_paired_updates.py', 'docs/clip-gated-training.md',
]


def build():
    sources = {name: (ROOT / name).read_text() for name in FILES}
    manifest = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in sources.items()}
    sources['gated_source_manifest.json'] = json.dumps(manifest, indent=2)
    payload = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(zlib.compress(payload, 9)).decode()
    cell = '''"""ONE Colab A100 cell: four-arm rollout, three seeds, mandatory Drive.
Hybrid retention: 24 permanent checkpoints plus one rolling backup.
Allow 25–30 GB free Google Drive quota and 40 GiB local runtime disk.
No browser download is needed; wait for DRIVE_SYNC_COMPLETE.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

BASE_COMMIT = __BASE__
BUNDLE_SHA256 = __DIGEST__
PAYLOAD = __PAYLOAD__

def run_all():
    if not Path('/content').is_dir():
        raise RuntimeError('Run in Google Colab, A100 runtime.')
    raw = zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Source bundle checksum mismatch')
    from google.colab import drive
    drive.mount('/content/drive')
    staging = Path(tempfile.mkdtemp(prefix='otco_gated_source_', dir='/content'))
    repo = staging / 'repo'
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/akashm776/otco.git', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', BASE_COMMIT], check=True)
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != BASE_COMMIT:
        raise RuntimeError('Wrong pinned source')
    for name, source in json.loads(raw).items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe source path')
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    # Separate interpreter: no stale notebook imports or previous model objects.
    subprocess.run([sys.executable, '-u', str(repo / 'colabs/run_clip_gated_training.py'),
                    '--bundle-id', BUNDLE_SHA256], cwd=repo, check=True)
    # Flush from the authenticated notebook process, not a child interpreter.
    drive.flush_and_unmount()
    print('DRIVE_SYNC_COMPLETE: all rollout weights and results backed up.')
    print('Confirm the run folder in Google Drive before releasing this runtime.')

run_all()
'''
    cell = cell.replace('__BASE__', repr(BASE)).replace('__DIGEST__', repr(digest)).replace('__PAYLOAD__', repr(encoded))
    destination = ROOT / 'colabs/clip_gated_training_drive_one_cell.py'
    destination.write_text(cell)
    print(destination, 'bundle_sha256=', digest)
    return cell


if __name__ == '__main__':
    build()
