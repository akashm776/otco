"""Drive-backed launcher for the bounded saved-state diagnostic; no new rollout."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colabs.backup_clip_run import checksum
from colabs.checkpoint_diagnostic_inputs import ARCHIVE_NAME, recover_inputs
from colabs.transfer_drive_backup import DRIVE_ROOT, DriveMirror, require_drive

TEST_FILES = ['tests/test_clip_checkpoint_diagnostic.py', 'tests/test_clip_checkpoint_recovery.py',
              'tests/test_clip_paired_updates.py',
              'tests/test_clip_early_pulse.py', 'tests/test_clip_training.py',
              'tests/test_clip_gated_training.py', 'tests/test_clip_transfer_drive_backup.py']


def preflight(bundle_id):
    if len(bundle_id) != 64 or any(c not in '0123456789abcdef' for c in bundle_id):
        raise ValueError('Expected bundle SHA256')
    require_drive(DRIVE_ROOT / 'preflight')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                  '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    if 'A100' not in gpu or int(gpu.split(',')[1].strip()) < 39000:
        raise RuntimeError(f'Requires one A100 with at least 39 GB; found {gpu}')
    if shutil.disk_usage('/content').free < 15 * 1024**3:
        raise RuntimeError('At least 15 GiB free runtime disk required for model/data caches')
    p = json.loads((ROOT / 'configs/clip_checkpoint_diagnostic.json').read_text())
    for marker in DRIVE_ROOT.glob('clip_checkpoint_diagnostic_*/run_manifest.json'):
        if json.loads(marker.read_text()).get('experiment') == p['experiment']:
            raise RuntimeError(f'Existing diagnostic: {marker.parent}. Inspect it before rerunning; no automatic resume.')
    source = DRIVE_ROOT / p['source_run']
    if not source.is_dir():
        candidates = [DRIVE_ROOT / ARCHIVE_NAME, Path('/content') / ARCHIVE_NAME]
        archive = next((path for path in candidates if path.is_file()), None)
        if archive is None:
            raise FileNotFoundError(f'Missing original saved run: {source}. Upload {ARCHIVE_NAME} '
                                    f'to {DRIVE_ROOT} or /content. No old training will be rerun.')
        print('RECOVERING_INPUTS_FROM:', archive, flush=True)
        source = recover_inputs(archive, Path('/content/otco_diagnostic_inputs') / p['source_run'], p)
        if shutil.disk_usage('/content').free < 15 * 1024**3:
            raise RuntimeError('Recovery verified, but 15 GiB additional runtime disk is required for caches')
    manifest = json.loads((source/'run_manifest.json').read_text())
    backup = json.loads((source/'DRIVE_BACKUP_MANIFEST.json').read_text())
    if (manifest['run_id'] != p['source_run'] or manifest['bundle_id'] != p['source_bundle']
            or manifest['status'] != 'complete_pending_drive_flush'
            or backup['status'] != 'verified_tree'):
        raise ValueError('Unexpected/incomplete source run')
    for relative, expected in p['checkpoints'].items():
        path = source / relative
        if not path.is_file() or backup['files'][relative]['sha256'] != expected:
            raise ValueError(f'Missing checkpoint or manifest mismatch: {relative}')
        if path.stat().st_size != backup['files'][relative]['bytes']:
            raise ValueError(f'Checkpoint size mismatch: {relative}')
    return source, manifest, gpu


def main(bundle_id):
    os.chdir(ROOT)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sources = json.loads((ROOT/'diagnostic_source_manifest.json').read_text())
    for relative, expected in sources.items():
        if checksum(ROOT/relative) != expected:
            raise AssertionError(f'Source overlay changed: {relative}')
    source, source_manifest, gpu = preflight(bundle_id)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==2.21.0',
                    'transformers==4.57.3', 'pyyaml>=6.0.1', 'pytest', 'matplotlib'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *TEST_FILES], check=True)
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device is required to restore source RNG')
    from src.clip_checkpoint_diagnostic import run, write_json
    # Use the source PyTorch version for AdamW state/update semantics. Do not
    # replace Colab's CUDA stack behind the user's back if it has changed.
    current_torch = importlib.metadata.version('torch')
    if current_torch.split('+')[0] != source_manifest['packages']['torch'].split('+')[0]:
        raise RuntimeError(f"Source torch={source_manifest['packages']['torch']}; runtime={current_torch}. "
                           'Version mismatch: inspect before running a changed optimizer implementation.')
    run_id = 'clip_checkpoint_diagnostic_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path('/content') / run_id
    output.mkdir(exist_ok=False)
    destination = DRIVE_ROOT/run_id
    mirror = DriveMirror(output, destination)
    manifest = dict(run_id=run_id, experiment='clip_checkpoint_diagnostic_v1', bundle_id=bundle_id,
        status='running', source_run=str(source), source_bundle=source_manifest['bundle_id'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gpu=gpu, python=sys.version, packages={n: importlib.metadata.version(n) for n in
            ['torch', 'torchvision', 'transformers', 'datasets', 'numpy', 'PyYAML']},
        independent_test_set=False, rules_refitted=False, source_checkpoints_read_only=True,
        drive_destination=str(destination), literal_training_loader_resume=False)
    for relative in sources:
        target = output/'source_overlay'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    write_json(output/'source_manifest.json', sources)
    write_json(output/'source_run_manifest.json', source_manifest)
    if (source/'RESTORED_SUBSET_MANIFEST.json').is_file():
        shutil.copyfile(source/'RESTORED_SUBSET_MANIFEST.json', output/'RESTORED_SUBSET_MANIFEST.json')
    write_json(output/'run_manifest.json', manifest)
    mirror.sync()
    print('PERSISTENT_DESTINATION:', destination, flush=True)
    print('12 saved states; 36 matched branches; 1824 total optimizer steps. No source checkpoints modified.', flush=True)
    print('Exploratory: reused excluded pool split 512 meta / 512 report; no fitting or adaptive branch selection.', flush=True)
    try:
        summary = run(source, output, mirror.sync)
        mirror.sync(final=True)
        manifest['status'] = 'complete_pending_drive_flush'
        write_json(output/'run_manifest.json', manifest)
        mirror.sync(final=True)
    except BaseException as error:
        manifest.update(status='failed_or_sync_incomplete', error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json', manifest)
        try:
            mirror.sync()
        except BaseException as backup_error:
            print('BACKUP_ALSO_FAILED:', repr(backup_error), flush=True)
        raise
    print('FINAL_COMPARISON:', json.dumps(summary, indent=2), flush=True)
    print('DRIVE_TREE_VERIFIED:', destination, flush=True)
    print('Diagnostic complete; notebook must now flush Drive.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle-id', required=True)
    main(parser.parse_args().bundle_id)
