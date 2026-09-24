"""Build a pinned-base, fully embedded, paste-ready diagnostic Colab cell."""

import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.build_clip_gated_training_cell import BASE, FILES as GATED_FILES

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(set(GATED_FILES + [
    'tests/__init__.py', 'configs/clip_checkpoint_diagnostic.json',
    'src/clip_checkpoint_diagnostic.py', 'colabs/run_clip_checkpoint_diagnostic.py',
    'tests/test_clip_checkpoint_diagnostic.py', 'docs/clip-checkpoint-diagnostic.md',
    'colabs/checkpoint_diagnostic_inputs.py', 'tests/test_clip_checkpoint_recovery.py',
]))


def build():
    sources = {name: (ROOT/name).read_text() for name in FILES}
    sources['diagnostic_source_manifest.json'] = json.dumps({
        name: hashlib.sha256(content.encode()).hexdigest() for name, content in sources.items()}, indent=2)
    payload = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(zlib.compress(payload, 9)).decode()
    cell = '''"""ONE Colab A100 cell: saved-checkpoint mechanistic diagnostic.
Requires the original clip_gated_training_20260922T114312_608272Z folder OR
clip_checkpoint_diagnostic_inputs.zip in MyDrive/OTCO/evaluation_transfer.
The recovery ZIP can alternatively be placed in /content. ZIP inputs are
verified and extracted locally, not copied back into a large Drive folder.
No retraining of old runs. 12 states x 3 branches x 50 updates + 24 trial updates.
Keep 25 GiB free runtime disk for ZIP recovery (15 GiB with existing Drive folder)
and 100 MB additional Drive space for results. Existing inputs stay read-only.
Paste this ENTIRE file into one Colab code cell. Wait for DRIVE_SYNC_COMPLETE.
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
        raise RuntimeError('Run this cell in Google Colab with an A100 GPU.')
    raw = zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Source bundle checksum mismatch')
    from google.colab import drive
    drive.mount('/content/drive')
    staging = Path(tempfile.mkdtemp(prefix='otco_checkpoint_diagnostic_source_', dir='/content'))
    repo = staging/'repo'
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/akashm776/otco.git', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', BASE_COMMIT], check=True)
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != BASE_COMMIT:
        raise RuntimeError('Wrong pinned base')
    for name, source in json.loads(raw).items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe source path')
        path = repo/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
    try:
        subprocess.run([sys.executable, '-u', str(repo/'colabs/run_clip_checkpoint_diagnostic.py'),
                        '--bundle-id', BUNDLE_SHA256], cwd=repo, check=True)
    except BaseException:
        print('DIAGNOSTIC_FAILED: inspect the partial output; do not restart automatically.')
        try:
            drive.flush_and_unmount()
            print('PARTIAL_DRIVE_SYNC_COMPLETE: this does NOT mean the diagnostic completed.')
        except BaseException as error:
            print('DRIVE_FLUSH_FAILED:', repr(error))
        raise
    drive.flush_and_unmount()
    print('DRIVE_SYNC_COMPLETE: diagnostic results saved and Drive flushed.')
    print('Download the new clip_checkpoint_diagnostic run folder; no large new checkpoints are written.')

run_all()
'''
    cell = cell.replace('__BASE__', repr(BASE)).replace('__DIGEST__', repr(digest)).replace('__PAYLOAD__', repr(encoded))
    target = ROOT/'colabs/clip_checkpoint_diagnostic_drive_one_cell.py'
    target.write_text(cell)
    print(target, 'bundle_sha256=', digest)
    return cell


if __name__ == '__main__':
    build()
