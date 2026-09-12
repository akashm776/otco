"""Foreground Colab handoff: attach/start, show progress, back up, download."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

TRAINING_COMMIT = '030c081f9c6d6c8e03740c7424c0f26af0ec81bb'
BACKUP_COMMIT = 'dc6b4daefe113355c5a17927976bb357a243e7fe'


def download_source(commit, filename, expected_hash, destination):
    url = f'https://raw.githubusercontent.com/akashm776/otco/{commit}/colabs/{filename}'
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise RuntimeError(f'Source checksum mismatch: {filename}')
    destination.write_bytes(content)


def processes(proc_root=Path('/proc')):
    result = {}
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            # comm may contain spaces; fields after the last ')' start at state.
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            args = (path / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if fields[0] != 'Z' and args != ['']:
                result[int(path.name)] = {'parent': int(fields[1]), 'args': args}
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return result


def find_training_jobs(table):
    jobs = []
    for pid, process in table.items():
        args = process['args']
        if 'src.clip_early_pulse' not in args or '--output-directory' not in args:
            continue
        parent = table.get(process['parent'], {})
        if 'src.clip_early_pulse' in parent.get('args', []):
            continue  # DataLoader worker, not a second training run.
        output = Path(args[args.index('--output-directory') + 1])
        parent_args = parent.get('args', [])
        is_runner = any(Path(a).name in ('runner.py', 'run_clip_early_pulse.py') for a in parent_args)
        jobs.append((process['parent'] if is_runner else pid, output))
    return jobs


def completed(output):
    path = output / 'completion.json'
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
        return (result.get('status') == 'complete' and result.get('completed_updates') == 1001
                and result.get('arms') == ['baseline', 'uniform_top8', 'hardest_real'])
    except (OSError, ValueError):
        return False


def main():
    from google.colab import drive, files

    print('Connecting Google Drive. Approve the authorization prompt if shown.', flush=True)
    drive.mount('/content/drive')
    if not os.path.ismount('/content/drive'):
        raise RuntimeError('A real Drive mount is required before this script starts training.')
    control = Path(tempfile.mkdtemp(prefix='otco_one_cell_', dir='/content'))
    backup_script = control / 'backup_clip_run.py'
    download_source(BACKUP_COMMIT, 'backup_clip_run.py',
        'ac90ba5ee64b27dc1e2c1b1228aa1e8295ca88e7649d5ccac0277cd7e86eea17', backup_script)
    spec = importlib.util.spec_from_file_location('otco_verified_backup', backup_script)
    backup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backup)

    table = processes()
    jobs = find_training_jobs(table)
    if len(jobs) > 1:
        raise RuntimeError(f'Multiple training runs detected; refusing to launch another: {jobs}')
    output, watched_pid, launched = None, None, None
    if jobs:
        watched_pid, output = jobs[0]
        print(f'ATTACHED to existing run: {output.name}. No duplicate launched.', flush=True)
        log_path = Path('/content') / (output.name + '_stdout.txt')
    else:
        # Reuse a completed latest result, including after a runtime reset.
        candidates = list(Path('/content/otco_outputs').glob('clip_early_pulse_*'))
        candidates += list(Path('/content/drive/MyDrive/OTCO/early_pulse').glob('clip_early_pulse_*'))
        if candidates:
            newest = max(candidates, key=lambda p: p.name)
            source = newest / 'results' if (newest / 'results').is_dir() else newest
            if completed(source):
                output = source
                print('Found completed results; preparing download.', flush=True)
        if output is None:
            # Also guard setup-phase launchers which have no training child yet.
            for process in table.values():
                for arg in process['args']:
                    path = Path(arg)
                    if path.name not in ('runner.py', 'run_clip_early_pulse.py') or not path.is_file():
                        continue
                    if 'src.clip_early_pulse' in path.read_text(errors='replace'):
                        raise RuntimeError('An early-pulse launcher is still setting up. Wait one minute and rerun THIS cell.')
            runner = control / 'runner.py'
            download_source(TRAINING_COMMIT, 'run_clip_early_pulse.py',
                'e6cee0a713f55702c1abea2f0380c2d7779ce572464cbf43c9d5501547708f9b', runner)
            log_path = control / 'master.log'
            with log_path.open('w') as log:
                launched = subprocess.Popen([sys.executable, '-u', str(runner), '--source-commit', TRAINING_COMMIT],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            watched_pid = launched.pid
            print('STARTED a fresh three-arm run with the unchanged protocol.', flush=True)

    log_offset, last_progress, last_heartbeat, backup_started = 0, None, 0., False
    backup_process = None
    while watched_pid is not None:
        if log_path.exists():
            with log_path.open() as log:
                log.seek(log_offset)
                chunk = log.read()
                log_offset = log.tell()
            if chunk:
                print(chunk, end='', flush=True)
            if output is None:
                for line in log_path.read_text(errors='replace').splitlines():
                    if line.startswith('EARLY_PULSE_OUTPUT: '):
                        output = Path(line.split(': ', 1)[1])
        if output is not None and not backup_started:
            running_backup = any('backup_clip_run.py' in ' '.join(p['args']) and str(output) in p['args']
                                 for p in processes().values())
            if not running_backup:
                with (control / 'backup.log').open('w') as log:
                    backup_process = subprocess.Popen([sys.executable, '-u', str(backup_script),
                        '--output-directory', str(output), '--checkpoint-directory',
                        str(Path('/content/otco_checkpoints') / output.name)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            backup_started = True
            print('\nDrive backup worker active for', output.name, flush=True)
        if time.monotonic() - last_heartbeat >= 30:
            print('\n[Monitoring]', datetime.now(timezone.utc).isoformat(),
                  '— cell stays running until completion and download.', flush=True)
            if output is not None:
                manifest = Path('/content/drive/MyDrive/OTCO/early_pulse') / output.name / 'backup_manifest.json'
                if manifest.exists():
                    try:
                        state = json.loads(manifest.read_text())
                        print('Drive:', len(state['verified_files']), 'verified files; errors:', state['errors'], flush=True)
                    except ValueError:
                        pass
            last_heartbeat = time.monotonic()
        if launched is not None:
            alive = launched.poll() is None
        else:
            alive = watched_pid in processes()
        if not alive:
            break
        time.sleep(10)

    if output is None or not completed(output):
        raise RuntimeError('Training stopped before all three arms completed. Review the visible log; partial results are on Drive.')
    run_id = output.parent.name if output.name == 'results' else output.name
    archive = control / f'{run_id}_complete.zip'
    print('\nAll three arms complete. Packaging and verifying results…', flush=True)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(output.rglob('*')):
            if source.is_file() and not source.name.startswith('.'):
                bundle.write(source, arcname=Path(run_id) / source.relative_to(output))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Result ZIP verification failed')
    target = Path('/content/drive/MyDrive/OTCO/early_pulse') / run_id / archive.name
    entry, _ = backup.copy_verified(archive, target)
    print('COMPLETE — verified ZIP backed up to:', target, flush=True)
    print('ZIP SHA256:', entry['sha256'], flush=True)
    files.download(str(archive))
    print('Download requested. If your browser blocks it, the same ZIP is in Drive.', flush=True)


if __name__ == '__main__':
    main()
