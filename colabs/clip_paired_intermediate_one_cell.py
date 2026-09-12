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


SOURCE_COMMIT = 'cce20072fe61be22bfcfd3b17ac8605d8bdce78b'
RUNNER_SHA256 = 'f97e6bf7c403e42b02ee5f2a05e51350f8274e0f7dffd2808cbbd44e6d06702a'


def run_intermediate_study():
    from google.colab import files
    # Successful reruns retrieve results without repeating training.
    archives = sorted(Path('/content').glob('otco_control_clip_paired_intermediate_*/*_complete.zip'),
                      key=lambda path: path.stat().st_mtime, reverse=True)
    expected = dict(status='complete', training_seeds=[42, 123, 456],
                    checkpoint_steps=[100, 250, 500, 750, 1001], total_branch_rows=720,
                    new_intermediate_branch_rows=432, endpoint_replay_branch_rows=288)
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            run_id = archive.name.removesuffix('_complete.zip')
            manifest = json.loads(bundle.read(f'{run_id}/results/run_manifest.json'))
            completion = json.loads(bundle.read(f'{run_id}/results/completion.json'))
            if manifest.get('source_commit') != SOURCE_COMMIT:
                continue
            if bundle.testzip() is not None or manifest.get('status') != 'complete' or completion != expected:
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
    url = f'https://raw.githubusercontent.com/akashm776/otco/{SOURCE_COMMIT}/colabs/run_clip_paired_intermediate.py'
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != RUNNER_SHA256:
        raise RuntimeError('Downloaded runner checksum mismatch')
    script = Path(tempfile.mkdtemp(prefix='otco_intermediate_cell_', dir='/content')) / 'runner.py'
    script.write_bytes(source)
    print('Starting seeds 42, 123 and 456 at five checkpoints. No Drive mount or storage.')
    print('720 branches: 432 intermediate records and 288 endpoint replays. No cutoff selection.')
    print('Keep this cell running; one combined results ZIP will download when all seeds finish.')
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(script), '--source-commit', SOURCE_COMMIT]
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous_argv


run_intermediate_study()
