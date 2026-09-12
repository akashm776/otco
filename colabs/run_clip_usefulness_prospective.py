"""One foreground Colab job: three new seeds, frozen rules, 720 probes, local-only ZIP."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


def verify_completion(output):
    from src.clip_usefulness_prospective import verify_completion as verify
    verify(output)


def report_archive(output, archive):
    """No model checkpoints or dataset tensors in the browser download."""
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix != '.pt' and not path.name.startswith('.'):
                bundle.write(path, arcname=path.relative_to(output.parent))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise AssertionError('ZIP verification failed')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise ValueError('Full immutable source commit required')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip()
    if 'A100' not in gpu:
        raise RuntimeError('Select an A100 runtime')
    active = [line for line in subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
              if ' -m src.clip_' in line]
    if active:
        raise RuntimeError('Another CLIP experiment is running; refusing a duplicate')
    if shutil.disk_usage('/content').free < 25 * 1024**3:
        raise RuntimeError('At least 25 GiB of local runtime disk space is required')
    run_id = 'clip_usefulness_prospective_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    control = Path('/content') / ('otco_control_' + run_id)
    control.mkdir(exist_ok=False)
    repo = control / 'source'
    output = Path('/content/otco_outputs') / run_id
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/akashm776/otco.git', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', args.source_commit], check=True)
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != args.source_commit:
        raise AssertionError('Wrong source checkout')
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # Keep model/dataset caches on runtime disk even if a prior notebook used an external cache.
    os.environ['HF_HOME'] = '/content/otco_hf_cache'
    os.environ['HF_DATASETS_CACHE'] = '/content/otco_hf_cache/datasets'
    os.environ['HF_HUB_CACHE'] = '/content/otco_hf_cache/hub'
    os.environ['TRANSFORMERS_CACHE'] = '/content/otco_hf_cache/hub'
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==2.21.0', 'transformers==4.57.3',
                    'numpy==2.1.3', 'PyYAML==6.0.3', 'pytest', 'matplotlib'], check=True)
    reference = json.loads((repo / 'experiment_results/clip_paired_updates_2026-09/run_manifest.json').read_text())
    for package in ['torch', 'torchvision']:
        actual = importlib.metadata.version(package)
        expected = reference['packages'][package]
        if actual != expected:
            raise RuntimeError(f'Prospective comparison pins {package}=={expected}; runtime has {actual}. No training started.')
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_clip_usefulness_prospective.py', 'tests/test_clip_usefulness_predictors.py', 'tests/test_clip_paired_seed_replication.py',
                    'tests/test_clip_paired_updates.py', 'tests/test_clip_early_pulse.py',
                    'tests/test_clip_training.py', 'tests/test_clip_run_backup.py'], check=True)
    from colabs.backup_clip_run import checksum
    manifest = {'run_id': run_id, 'source_commit': args.source_commit, 'training_seeds': [789, 2026, 31415], 'checkpoint_steps': [100,250,500,750,1001],
                'status': 'running', 'storage': 'colab_local_only_no_drive', 'gpu': gpu, 'python': sys.version,
                'packages': {name: importlib.metadata.version(name) for name in
                             ['torch', 'torchvision', 'transformers', 'datasets', 'numpy', 'PyYAML']}}
    (control / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    command = [sys.executable, '-u', '-m', 'src.clip_usefulness_prospective', '--output-directory', str(output)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print('PROSPECTIVE_OUTPUT:', output, '\nSTORAGE: local Colab disk only; no Drive writes.', flush=True)
    failure = None
    try:
        with (control / 'stdout.txt').open('w') as log:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
        verify_completion(output)
        manifest['status'] = 'complete'
    except BaseException as error:
        failure = error
        manifest.update(status='interrupted_or_failed', error=repr(error))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        results = output / 'results'
        results.mkdir(parents=True, exist_ok=True)
        (results / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        if (control / 'stdout.txt').exists():
            shutil.copyfile(control / 'stdout.txt', results / 'stdout.txt')
        archive = control / f'{run_id}_{manifest["status"]}.zip'
        report_archive(output, archive)
        print('LOCAL_ARCHIVE:', archive, '\nARCHIVE_SHA256:', checksum(archive), flush=True)
        try:
            from google.colab import files
            files.download(str(archive))
        except Exception as error:
            print('Automatic download unavailable:', repr(error), '\nDownload LOCAL_ARCHIVE from the Colab Files panel.', flush=True)
    if failure is not None:
        raise failure
    print('DONE: all three seeds and five checkpoints completed. Keep the combined ZIP for analysis.', flush=True)


if __name__ == '__main__':
    main()
