"""Build a paste-ready, pinned-base, checksummed Colab source overlay."""

import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.build_clip_gated_training_cell import BASE, FILES as GATED_FILES

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(set(GATED_FILES + [
    'tests/__init__.py', 'configs/clip_exposure_matched.json',
    'src/clip_exposure_matched.py', 'colabs/run_clip_exposure_matched.py',
    'colabs/exposure_checkpoint_retention.py', 'tests/test_clip_exposure_matched.py',
    'tests/test_clip_exposure_retention.py', 'docs/clip-exposure-matched.md',
]))


def build():
    sources = {name: (ROOT / name).read_text() for name in FILES}
    sources['exposure_source_manifest.json'] = json.dumps({
        name: hashlib.sha256(content.encode()).hexdigest() for name, content in sources.items()}, indent=2)
    payload = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(zlib.compress(payload, 9)).decode()
    cell = '''"""ONE Colab A100 cell: equal-exposure timing controls, 3 seeds x 4 arms.
Select A100 (40 GB); keep 40 GiB runtime disk and 30 GB ADDITIONAL Drive quota free.
No old runs/checkpoints are deleted or modified. No download/upload inputs needed.
Runs frozen alignment gate + 3 dose-matched randomized controls per seed.
Do not rerun; wait for DRIVE_SYNC_COMPLETE.
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
        raise RuntimeError('Run in Google Colab with an A100 GPU.')
    raw = zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Source bundle checksum mismatch')
    from google.colab import drive
    drive.mount('/content/drive')
    staging = Path(tempfile.mkdtemp(prefix='otco_exposure_source_', dir='/content'))
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
        path.write_text(source, encoding='utf-8')
    # Stream live output; never buffer the whole experiment in memory.
    try:
        subprocess.run([sys.executable, '-u', str(repo / 'colabs/run_clip_exposure_matched.py'),
                        '--bundle-id', BUNDLE_SHA256], cwd=repo, check=True)
    except BaseException:
        print('RUN_FAILED: do not restart training automatically; inspect saved artifacts.')
        try:
            drive.flush_and_unmount()
            print('PARTIAL_DRIVE_SYNC_COMPLETE: this does NOT mean training completed.')
        except BaseException as error:
            print('DRIVE_FLUSH_FAILED:', repr(error))
        raise
    # Authenticated notebook process, not child interpreter.
    drive.flush_and_unmount()
    print('DRIVE_SYNC_COMPLETE: exposure-matched run completed and Drive flushed.')
    print('Confirm the run folder in Drive. The stored manifest was written before flush.')

run_all()
'''
    cell = cell.replace('__BASE__', repr(BASE)).replace('__DIGEST__', repr(digest)).replace('__PAYLOAD__', repr(encoded))
    target = ROOT / 'colabs/clip_exposure_matched_drive_one_cell.py'
    target.write_text(cell)
    print(target, 'bundle_sha256=', digest)
    return cell


if __name__ == '__main__':
    build()
