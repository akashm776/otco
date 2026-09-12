"""Foreground A100 runner. Mount Drive first; supply an immutable source commit."""

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
import threading
import zipfile


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise ValueError('An immutable full source commit is required')
    if not os.path.ismount('/content/drive'):
        raise RuntimeError('Mount Google Drive in the notebook before running this script')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'], text=True).strip()
    if 'A100' not in gpu:
        raise RuntimeError('Select an A100 runtime first')
    active = [line for line in subprocess.check_output(['ps','-eo','pid,args'], text=True).splitlines()
              if ' -m src.clip_' in line]
    if active:
        raise RuntimeError('Another CLIP experiment is running; refusing a duplicate')
    run_id = 'clip_paired_updates_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    control = Path('/content') / ('otco_control_' + run_id)
    control.mkdir(exist_ok=False)
    repo = control / 'source'
    output = Path('/content/otco_outputs') / run_id
    destination = Path('/content/drive/MyDrive/OTCO/paired_updates') / run_id
    source_name = 'clip_early_pulse_20260911T233840_978774Z'
    source = Path('/content/drive/MyDrive/OTCO/early_pulse') / source_name
    if not (source / 'checkpoints/baseline/latest.pt').exists():
        raise FileNotFoundError(f'Missing baseline checkpoint backup: {source}')
    subprocess.run(['git','clone','--no-checkout','https://github.com/akashm776/otco.git',str(repo)], check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',args.source_commit], check=True)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    subprocess.run([sys.executable,'-m','pip','install','datasets>=2.21.0,<3.0.0','transformers==4.57.3','pyyaml>=6.0.1','pytest','matplotlib'], check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_clip_paired_updates.py',
        'tests/test_clip_early_pulse.py','tests/test_clip_warmup_readiness.py','tests/test_clip_training.py',
        'tests/test_clip_run_backup.py'], check=True)
    from colabs.backup_clip_run import RunBackup, copy_verified
    original_manifest = json.loads((source / 'backup_manifest.json').read_text())
    local_source = control / source_name
    required = ['backup_manifest.json','checkpoints/baseline/common_step_100.pt','checkpoints/baseline/latest.pt',
                'results/calibration.json','results/prefix_hashes.json',
                'results/baseline/diagnostics/diagnostic_holdout_indices.json',
                'results/baseline/diagnostics/000100_pulse_100/report.json',
                'results/baseline/diagnostics/001001_pulse_1001/report.json']
    for relative in required:
        entry, _ = copy_verified(source / relative, local_source / relative)
        if relative != 'backup_manifest.json' and entry['sha256'] != original_manifest['verified_files'][relative]['sha256']:
            raise ValueError('Source backup inventory mismatch: ' + relative)
        print('Verified input:', relative, flush=True)
    manifest = {'run_id': run_id, 'source_commit': args.source_commit, 'input_run': source_name,
        'gpu': gpu, 'python': sys.version, 'status': 'running',
        'packages': {name: importlib.metadata.version(name) for name in
                     ['torch','torchvision','transformers','datasets','numpy','PyYAML']}}
    worker = RunBackup(output, control / 'no_new_checkpoints', destination)
    stop = threading.Event()
    def periodic_backup():
        while not stop.is_set():
            try:
                if output.exists():
                    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
                    result = worker.sync()
                    print(f'[backup] {len(result["verified_files"])} files; {len(result["errors"])} errors', flush=True)
            except Exception as error:
                print('[backup retry]', repr(error), flush=True)
            stop.wait(30)
    thread = threading.Thread(target=periodic_backup, daemon=True)
    thread.start()
    command = [sys.executable,'-u','-m','src.clip_paired_updates','--source-directory',str(local_source),'--output-directory',str(output)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print('PAIRED_OUTPUT:', output, flush=True)
    error = None
    try:
        with (control / 'stdout.txt').open('w') as log:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
        result = json.loads((output / 'completion.json').read_text())
        if result.get('rows') != 96 or result.get('status') != 'complete':
            raise AssertionError('Incomplete paired result')
        manifest['status'] = 'complete'
    except BaseException as caught:
        error = caught
        manifest.update(status='interrupted_or_failed', error=repr(caught))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stop.set()
        thread.join(timeout=60)
        if thread.is_alive():
            raise RuntimeError('Drive backup is still blocked; local results remain in ' + str(output))
        output.mkdir(parents=True, exist_ok=True)
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        shutil.copyfile(control / 'stdout.txt', output / 'stdout.txt')
        archive = control / f'{run_id}_{manifest["status"]}.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    bundle.write(path, arcname=path.relative_to(output.parent))
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
        print('LOCAL_ARCHIVE:', archive, flush=True)
        try:
            worker.sync()
            copied, _ = copy_verified(archive, destination / archive.name)
            print('VERIFIED_DRIVE_ZIP:', destination / archive.name, copied['sha256'], flush=True)
        except Exception as backup_error:
            print('DRIVE_BACKUP_FAILED:', repr(backup_error), flush=True)
        try:
            from google.colab import files
            files.download(str(archive))
        except Exception as download_error:
            print('Automatic browser download unavailable:', repr(download_error), flush=True)
            print('Download LOCAL_ARCHIVE from Colab Files, or the ZIP from Drive.', flush=True)
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
