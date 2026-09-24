"""Generate a single pasteable cell with a hash-checked source overlay."""

import base64
import argparse
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
BASE = '7e7cfea90b60415dc9561efbe97bcd383cc1580e'
FILES = ['src/clip_paired_updates.py', 'src/clip_evaluation_transfer.py',
         'colabs/run_clip_evaluation_transfer.py', 'tests/test_clip_evaluation_transfer.py',
         'tests/test_clip_paired_updates.py',
         'colabs/recover_clip_evaluation_transfer_one_cell.py',
         'tests/test_clip_transfer_recovery.py', 'src/clip_usefulness_prospective.py',
         'colabs/transfer_drive_backup.py', 'tests/test_clip_transfer_drive_backup.py']


def build(*, drive=False):
    payload = json.dumps({name:(ROOT/name).read_text() for name in FILES}, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(zlib.compress(payload, 9)).decode()
    cell = '''"""Paste this ENTIRE file into ONE A100 Colab cell. No Google Drive use.

Three-seed replay + fixed evaluation-pool transfer + actual checkpoint download.
Allow 50 GiB local runtime disk and roughly 13 GB on your computer for the ZIP.
"""
import base64
import hashlib
import json
from pathlib import Path
import runpy
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
    payload = zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(payload).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Embedded source checksum mismatch')
    sources = json.loads(payload)
    __DRIVE_SETUP__
    staging = Path(tempfile.mkdtemp(prefix='otco_transfer_source_', dir='/content'))
    repo = staging/'repo'
    subprocess.run(['git','clone','--no-checkout','https://github.com/akashm776/otco.git',str(repo)], check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',BASE_COMMIT], check=True)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip() != BASE_COMMIT:
        raise RuntimeError('Wrong pinned checkout')
    for name, source in sources.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe overlay path')
        path = repo/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    print('Three seeds: 789, 2026, 31415; checkpoints: 100, 250, 500, 750, 1001.')
    print('Original-state replay checks + different CUB evaluation pool; frozen rules, no curriculum training.')
    print('No Drive. The final verified ZIP INCLUDES model and optimizer checkpoints (~13 GB).')
    print('Download results first, then individual checkpoints using the selector. Full ZIP retained.')
    print('A100 required. Keep the runtime until the browser download has finished.')
    previous = sys.argv[:]
    try:
        script = repo/'colabs/run_clip_evaluation_transfer.py'
        sys.argv = [str(script), '--bundle-id', BUNDLE_SHA256] + __DRIVE_ARGS__
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous

run_all()
'''
    cell = cell.replace('__BASE__', repr(BASE)).replace('__DIGEST__', repr(digest)).replace('__PAYLOAD__', repr(encoded))
    cell = cell.replace('__DRIVE_SETUP__',
        "from google.colab import drive\n    drive.mount('/content/drive')" if drive else '')
    cell = cell.replace('__DRIVE_ARGS__',
        repr(['--drive-root', '/content/drive/MyDrive/OTCO/evaluation_transfer']) if drive else '[]')
    if drive:
        cell = cell.replace('No Google Drive use.', 'Google Drive authorization required before training.')
        cell = cell.replace('roughly 13 GB on your computer for the ZIP.',
                            'at least 30 GB free in Google Drive for checkpoint copies + the ZIP.')
        cell = cell.replace('No Drive. The final verified ZIP INCLUDES model and optimizer checkpoints (~13 GB).',
                            'Drive backups enabled: checkpoints copied as they close; full verified ZIP saved too.')
        cell = cell.replace('Download results first, then individual checkpoints using the selector. Full ZIP retained.',
                            'Destination: MyDrive/OTCO/evaluation_transfer/<run_id> (roughly 25 GB total).')
        cell = cell.replace('Keep the runtime until the browser download has finished.',
                            'Keep the runtime until DRIVE_SYNC_COMPLETE. No browser download required.')
    cell = '\n'.join(line.rstrip() for line in cell.splitlines()) + '\n'
    destination = ROOT/('colabs/clip_evaluation_transfer_drive_one_cell.py' if drive
                        else 'colabs/clip_evaluation_transfer_one_cell.py')
    destination.write_text(cell)
    print(destination, 'bytes=', len(cell.encode()), 'bundle_sha256=', digest)
    return cell


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--drive', action='store_true')
    build(drive=parser.parse_args().drive)
