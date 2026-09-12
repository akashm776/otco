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


SOURCE_COMMIT = 'd907f67968b4ead5e06d416b2f675283cdba2438'
RUNNER_SHA256 = 'a66a70bbc100f103c8c6fd47634bfdfc2fac59a44d5165a30a9594ffad5b2038'


def run_prospective_study():
    from google.colab import files
    # Successful reruns retrieve results without repeating training.
    archives = sorted(Path('/content').glob('otco_control_clip_usefulness_prospective_*/*_complete.zip'),
                      key=lambda path: path.stat().st_mtime, reverse=True)
    expected = dict(status='complete', training_seeds=[789, 2026, 31415],
                    checkpoint_steps=[100, 250, 500, 750, 1001], total_branch_rows=720,
                    evaluated_states=15, rules_refitted=False)
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
    url = f'https://raw.githubusercontent.com/akashm776/otco/{SOURCE_COMMIT}/colabs/run_clip_usefulness_prospective.py'
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != RUNNER_SHA256:
        raise RuntimeError('Downloaded runner checksum mismatch')
    script = Path(tempfile.mkdtemp(prefix='otco_prospective_cell_', dir='/content')) / 'runner.py'
    script.write_bytes(source)
    print('Starting new seeds 789, 2026 and 31415 at five checkpoints. No Drive mount or storage.')
    print('720 new branches: frozen alignment vs step rules. No refitting or gate-driven training.')
    print('Keep this cell running; one combined results ZIP will download when all seeds finish.')
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(script), '--source-commit', SOURCE_COMMIT]
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous_argv


run_prospective_study()
