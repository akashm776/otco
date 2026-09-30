"""Build a self-contained, checksum-pinned fresh-stream Colab time course."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.build_clip_checkpoint_diagnostic_cell import FILES as PARENT_FILES

ROOT = Path(__file__).resolve().parents[1]
BASE = 'b36bcbf7c2124bbcbe0cf372df3853dadcf79bd2'
FILES = sorted(set(PARENT_FILES + [
    'configs/clip_continuation_timecourse.json', 'src/clip_continuation_timecourse.py',
    'colabs/run_clip_continuation_timecourse.py', 'tests/test_clip_continuation_timecourse.py',
    'docs/clip-continuation-timecourse.md',
]))


def build():
    sources = {name: (ROOT/name).read_text() for name in FILES}
    sources['timecourse_source_manifest.json'] = json.dumps({
        name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()}, indent=2)
    raw = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(zlib.compress(raw, 9)).decode()
    cell = '''"""ONE A100 Colab cell: fresh matched continuation streams and fixed time course.
Requires the existing 12-checkpoint source folder OR clip_checkpoint_diagnostic_inputs.zip
in MyDrive/OTCO/evaluation_transfer (ZIP may alternatively be in /content).
24 matched pairs, 48 branches, 2400 optimizer updates, no trial updates or retraining.
Reports/re-probes at 0, 1, 5, 10, 25, 50; signals NEVER control either branch.
Allow ~25 GiB local recovery space, 15 GiB free after recovery, 100 MB new Drive space.
No large new checkpoints. Paste this ENTIRE file and wait for DRIVE_SYNC_COMPLETE.
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
        raise RuntimeError('Run in Google Colab with one A100 GPU.')
    raw = zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Embedded bundle checksum mismatch')
    from google.colab import drive
    drive.mount('/content/drive')
    staging = Path(tempfile.mkdtemp(prefix='otco_timecourse_source_', dir='/content'))
    repo = staging/'repo'
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/akashm776/otco.git', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', BASE_COMMIT], check=True)
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != BASE_COMMIT:
        raise RuntimeError('Wrong pinned repository base')
    for name, source in json.loads(raw).items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Unsafe source overlay path')
        target = repo/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding='utf-8')
    try:
        subprocess.run([sys.executable, '-u', str(repo/'colabs/run_clip_continuation_timecourse.py'),
                        '--bundle-id', BUNDLE_SHA256], cwd=repo, check=True)
    except BaseException:
        print('TIMECOURSE_FAILED: inspect partial output; do not restart automatically.')
        try:
            drive.flush_and_unmount()
            print('PARTIAL_DRIVE_SYNC_COMPLETE: this does NOT mean the experiment completed.')
        except BaseException as error:
            print('DRIVE_FLUSH_FAILED:', repr(error))
        raise
    drive.flush_and_unmount()
    print('DRIVE_SYNC_COMPLETE: fresh-stream time-course results saved and Drive flushed.')
    print('Download the new clip_continuation_timecourse folder; no large new checkpoints were written.')

run_all()
'''
    cell = cell.replace('__BASE__', repr(BASE)).replace('__DIGEST__', repr(digest)).replace('__PAYLOAD__', repr(payload))
    target = ROOT/'colabs/clip_continuation_timecourse_drive_one_cell.py'
    target.write_text(cell)
    print(target, 'bundle_sha256=', digest)
    return cell


if __name__ == '__main__':
    build()
