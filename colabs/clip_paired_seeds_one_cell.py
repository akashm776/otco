"""Paste this entire file into ONE A100 Colab code cell. No Google Drive writes."""

from pathlib import Path
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


SOURCE_COMMIT = '2c4c610576c5167114bb6778cef526f0a8d64df8'
RUNNER_SHA256 = '32358c0a2bf9f230cf3ef72d69d95884ab80957dee8ebbeb45d6f13957ecf4df'


def run_two_seeds():
    from google.colab import files
    # Rerunning after successful completion downloads the existing result;
    # it does not unnecessarily repeat both baseline training trajectories.
    archives = sorted(Path('/content').glob('otco_control_clip_paired_seeds_*/*_complete.zip'),
                      key=lambda path: path.stat().st_mtime, reverse=True)
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            run_id = archive.name.removesuffix('_complete.zip')
            manifest = json.loads(bundle.read(f'{run_id}/results/run_manifest.json'))
            completion = json.loads(bundle.read(f'{run_id}/results/completion.json'))
            if manifest.get('source_commit') != SOURCE_COMMIT:
                continue
            if (bundle.testzip() is not None or manifest.get('status') != 'complete'
                    or completion.get('status') != 'complete' or completion.get('new_branch_rows') != 192
                    or completion.get('new_training_seeds') != [123, 456]):
                raise RuntimeError('Existing result ZIP failed verification; inspect it before starting a new run.')
        print('Already complete. Downloading:', archive)
        print('SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())
        files.download(str(archive))
        return
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('Select Runtime → Change runtime type → A100 GPU, then rerun this cell.')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    if 'A100' not in gpu:
        raise RuntimeError('This experiment requires an A100 GPU runtime.')
    url = f'https://raw.githubusercontent.com/akashm776/otco/{SOURCE_COMMIT}/colabs/run_clip_paired_seed_replication.py'
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != RUNNER_SHA256:
        raise RuntimeError('Downloaded runner checksum mismatch')
    script = Path(tempfile.mkdtemp(prefix='otco_two_seed_cell_', dir='/content')) / 'runner.py'
    script.write_bytes(source)
    print('Starting seeds 123 and 456 sequentially. No Drive mount or storage.')
    print('Keep this cell running; one combined results ZIP will download when both seeds finish.')
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(script), '--source-commit', SOURCE_COMMIT]
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous_argv


run_two_seeds()
