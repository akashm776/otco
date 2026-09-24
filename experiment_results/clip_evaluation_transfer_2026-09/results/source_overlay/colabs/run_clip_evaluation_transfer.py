"""Foreground replay + evaluation-pool transfer, with a real checkpoint ZIP."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SEEDS = [789, 2026, 31415]
STEPS = [100, 250, 500, 750, 1001]


def digest(handle):
    h = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8*1024**2), b''):
        h.update(chunk)
    return h.hexdigest()


def required_checkpoints():
    return [f'seed_{seed}/checkpoints/baseline/{name}' for seed in SEEDS
            for name in ['common_step_100.pt', 'step_000250.pt', 'step_000500.pt',
                         'step_000750.pt', 'latest.pt']]


def verify_archive(archive, *, require_complete=True, bundle_id=None):
    with zipfile.ZipFile(archive) as z:
        packed = json.loads(z.read('PACKING_MANIFEST.json'))
        if bundle_id is not None and packed['bundle_id'] != bundle_id:
            raise AssertionError('Archive belongs to different source code')
        if require_complete and packed['status'] != 'complete':
            raise AssertionError('Archive is not complete')
        if set(z.namelist()) != set(packed['files']) | {'PACKING_MANIFEST.json'} or len(z.namelist()) != len(set(z.namelist())):
            raise AssertionError('Archive member inventory mismatch')
        for name, record in packed['files'].items():
            if z.getinfo(name).file_size != record['bytes']:
                raise AssertionError(f'Archive size mismatch: {name}')
            with z.open(name) as handle:
                if digest(handle) != record['sha256']:
                    raise AssertionError(f'Archive checksum mismatch: {name}')
        if require_complete:
            for relative in required_checkpoints():
                if packed['run_id']+'/'+relative not in packed['files']:
                    raise AssertionError(f'Checkpoint missing from download: {relative}')
            marker = json.loads(z.read(packed['run_id']+'/results/completion.json'))
            if marker != dict(status='complete', training_seeds=SEEDS, checkpoint_steps=STEPS,
                    replay_branch_rows=720, transfer_branch_rows=720, evaluated_states=15, rules_refitted=False):
                raise AssertionError('Invalid study completion marker')
    return packed


def report_archive(output, archive, *, status, bundle_id):
    """Include actual tensors, not just their inventory. Stream without RAM spikes."""
    output, archive = Path(output), Path(archive)
    paths = [p for p in sorted(output.rglob('*')) if p.is_file() and not p.is_symlink()]
    if status == 'complete' and any(not (output/p).is_file() for p in required_checkpoints()):
        raise AssertionError('Cannot mark download complete without all 15 checkpoints')
    size = sum(p.stat().st_size for p in paths)
    if shutil.disk_usage(archive.parent).free < size + 1024**3:
        raise RuntimeError('Insufficient local space for full archive; source files have been retained')
    packed = dict(run_id=output.name, status=status, bundle_id=bundle_id, files={},
                  checkpoint_policy='Actual model + optimizer + scheduler files included')
    temporary = archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in paths:
            name = str(path.relative_to(output.parent))
            before = path.stat()
            with path.open('rb') as source, z.open(name, 'w', force_zip64=True) as target:
                h = hashlib.sha256()
                for chunk in iter(lambda: source.read(8*1024**2), b''):
                    h.update(chunk)
                    target.write(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise AssertionError(f'Source changed during packaging: {path}')
            packed['files'][name] = dict(bytes=before.st_size, sha256=h.hexdigest())
        z.writestr('PACKING_MANIFEST.json', json.dumps(packed, indent=2))
    verify_archive(temporary, require_complete=status=='complete', bundle_id=bundle_id)
    temporary.replace(archive)
    return packed


def download(archive):
    print('DOWNLOAD:', archive, '\nUse the Colab Files panel if the browser download fails.', flush=True)
    print('Keep the runtime until the ZIP is fully saved on your computer. No Drive copy exists.', flush=True)
    try:
        from google.colab import files
        files.download(str(archive))
    except Exception as error:
        print('Automatic download unavailable:', repr(error), flush=True)


def download_complete(archive):
    """Keep the full verified ZIP, but request the small results download first."""
    from colabs.recover_clip_evaluation_transfer_one_cell import run_recovery
    with zipfile.ZipFile(archive) as bundle:
        run_id = json.loads(bundle.read('PACKING_MANIFEST.json'))['run_id']
    print('Full checkpoint archive retained:', archive, flush=True)
    run_recovery(archive, run_id)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle-id', required=True)
    parser.add_argument('--drive-root', help='Mounted MyDrive/OTCO/evaluation_transfer; enables persistent backups')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    drive_root = Path(args.drive_root) if args.drive_root else None
    if drive_root:
        from colabs.transfer_drive_backup import require_drive, DriveMirror, save_archive
        require_drive(drive_root)
        # Only inspect this experiment's folder, never unrelated Drive data.
        if drive_root.exists():
            for previous_path in sorted(drive_root.glob('*/results/run_manifest.json')):
                previous = json.loads(previous_path.read_text())
                if previous.get('bundle_id') == args.bundle_id:
                    raise RuntimeError(f'Persistent run already exists: {previous_path.parent.parent}. '
                                       'Inspect/recover it before starting another training run.')
    # Repeat clicks retrieve a verified finished archive, never silently retrain.
    for archive in sorted(Path('/content').glob('otco_transfer_*/*_complete.zip'), reverse=True):
        with zipfile.ZipFile(archive) as z:
            packed = json.loads(z.read('PACKING_MANIFEST.json'))
        if packed.get('bundle_id') == args.bundle_id:
            print('Verifying the existing complete download (including checkpoints)...', flush=True)
            verify_archive(archive, bundle_id=args.bundle_id)
            sys.path.insert(0, str(repo))
            if drive_root:
                save_archive(archive, drive_root / packed['run_id'] / archive.name)
                from google.colab import drive
                drive.flush_and_unmount()
                print('DRIVE_SYNC_COMPLETE: existing archive saved; no retraining.', flush=True)
            else:
                download_complete(archive)
            return
    active = [line for line in subprocess.check_output(['ps','-eo','pid,args'], text=True).splitlines()
              if ' -m src.clip_' in line]
    if active:
        raise RuntimeError('Another CLIP experiment is running; refusing a duplicate')
    unfinished = list(Path('/content').glob('otco_transfer_*/run_manifest.json'))
    for path in unfinished:
        previous = json.loads(path.read_text())
        if previous.get('bundle_id') == args.bundle_id:
            raise RuntimeError(f'Previous run exists at {path.parent}; inspect/recover it before retraining.')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], text=True).strip()
    if 'A100' not in gpu:
        raise RuntimeError('Select an A100 GPU runtime')
    if shutil.disk_usage('/content').free < 50*1024**3:
        raise RuntimeError('At least 50 GiB free local runtime disk required, including checkpoint ZIP space')
    reference = json.loads((repo/'experiment_results/clip_usefulness_prospective_2026-09/results/run_manifest.json').read_text())
    for package in ['torch','torchvision']:
        actual, expected = importlib.metadata.version(package), reference['packages'][package]
        if actual != expected:
            raise RuntimeError(f'Replay requires {package}=={expected}; runtime has {actual}. No training started.')
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.update(TOKENIZERS_PARALLELISM='false', HF_HOME='/content/otco_hf_cache',
        HF_DATASETS_CACHE='/content/otco_hf_cache/datasets', HF_HUB_CACHE='/content/otco_hf_cache/hub',
        TRANSFORMERS_CACHE='/content/otco_hf_cache/hub')
    subprocess.run([sys.executable,'-m','pip','install','datasets==2.21.0','transformers==4.57.3',
                    'numpy==2.1.3','PyYAML==6.0.3','pytest','matplotlib'], check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_clip_evaluation_transfer.py',
        'tests/test_clip_paired_updates.py','tests/test_clip_usefulness_prospective.py',
        'tests/test_clip_paired_intermediate.py','tests/test_clip_transfer_recovery.py',
        'tests/test_clip_transfer_drive_backup.py'], check=True)
    run_id = 'clip_evaluation_transfer_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    control = Path('/content')/('otco_transfer_'+run_id)
    control.mkdir()
    output = control/run_id
    drive_destination = drive_root / run_id if drive_root else None
    if drive_destination:
        require_drive(drive_destination)
        drive_destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_id=run_id, bundle_id=args.bundle_id, status='running', gpu=gpu,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        storage='local_with_verified_drive_copies' if drive_root else 'colab_local_only_no_drive',
        drive_destination=str(drive_destination) if drive_destination else None, python=sys.version,
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','datasets','transformers','numpy','PyYAML']})
    (control/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    if drive_destination:
        save_archive(control/'run_manifest.json', drive_destination/'results/run_manifest.json')
    command = [sys.executable,'-u','-m','src.clip_evaluation_transfer','--output-directory',str(output)]
    if drive_destination:
        command += ['--backup-directory', str(drive_destination)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    failure = None
    try:
        with (control/'stdout.txt').open('w') as log:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
        subprocess.run([sys.executable, '-c',
            'from pathlib import Path; from src.clip_evaluation_transfer import verify_completion; '
            'import sys; verify_completion(Path(sys.argv[1]))', str(output)], check=True)
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
        results = output/'results'
        results.mkdir(parents=True, exist_ok=True)
        (control/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        (results/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        shutil.copyfile(control/'stdout.txt', results/'stdout.txt')
        # Preserve the exact overlay source alongside the pinned base commit.
        for relative in ['src/clip_paired_updates.py','src/clip_evaluation_transfer.py',
                         'colabs/run_clip_evaluation_transfer.py','tests/test_clip_evaluation_transfer.py',
                         'tests/test_clip_paired_updates.py',
                         'colabs/recover_clip_evaluation_transfer_one_cell.py',
                         'tests/test_clip_transfer_recovery.py',
                         'src/clip_usefulness_prospective.py',
                         'colabs/transfer_drive_backup.py', 'tests/test_clip_transfer_drive_backup.py']:
            destination = results/'source_overlay'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/relative, destination)
        if drive_destination:
            print('Verifying persistent output tree before ZIP packaging...', flush=True)
            DriveMirror(output, drive_destination).sync(final=True)
        archive = control/f'{run_id}_{manifest["status"]}.zip'
        print('Packaging and verifying actual checkpoints + all results. This may take several minutes.', flush=True)
        report_archive(output, archive, status=manifest['status'], bundle_id=args.bundle_id)
        print('ARCHIVE_GB:', round(archive.stat().st_size/1e9, 2), flush=True)
        if drive_destination:
            save_archive(archive, drive_destination/archive.name)
            print('Flushing pending Drive writes; keep this runtime until DRIVE_SYNC_COMPLETE.', flush=True)
            from google.colab import drive
            drive.flush_and_unmount()
            print('DRIVE_SYNC_COMPLETE:', str(drive_destination).replace('/content/drive/', ''), flush=True)
        elif manifest['status'] == 'complete':
            download_complete(archive)
        else:
            download(archive)
    if failure is not None:
        raise failure
    print('DONE: replay and transfer passed; ZIP includes all 15 required checkpoints.', flush=True)


if __name__ == '__main__':
    main()
