"""A100-only rollout runner; mandatory synchronous Google Drive backups."""

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
from colabs.gated_checkpoint_retention import required_paths, ROLLING
from colabs.transfer_drive_backup import DriveMirror, DRIVE_ROOT, require_drive


def preflight(bundle_id):
    if not bundle_id or len(bundle_id) != 64 or any(c not in '0123456789abcdef' for c in bundle_id):
        raise ValueError('Expected source bundle SHA256')
    require_drive(DRIVE_ROOT / 'preflight')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
        '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    if 'A100' not in gpu or int(gpu.split(',')[1].strip()) < 39000:
        raise RuntimeError(f'A100 with at least 39 GB required; found {gpu}')
    if shutil.disk_usage('/content').free < 40 * 1024**3:
        raise RuntimeError('At least 40 GiB free Colab disk required for hybrid checkpoints and caches')
    # Drive FUSE can report misleading quota. The user must also check quota in
    # Drive; write/read-back failures always stop training.
    for existing in DRIVE_ROOT.glob('clip_gated_training_*/run_manifest.json'):
        manifest = json.loads(existing.read_text())
        if manifest.get('bundle_id') == bundle_id:
            raise RuntimeError(f'Existing run for this bundle: {existing.parent}. Inspect it; no automatic duplicate/resume.')
    return gpu


def verify_inventory(inventory):
    required = required_paths()
    allowed = set(required) | {ROLLING}
    if any(name not in inventory or inventory[name]['bytes'] <= 0 for name in allowed):
        raise AssertionError('Incomplete persistent checkpoint inventory')
    if {name for name in inventory if name.endswith('.pt')} != allowed:
        raise AssertionError('Unexpected checkpoint duplicates violate hybrid retention')
    return len(required)


def main(bundle_id):
    gpu = preflight(bundle_id)
    os.chdir(ROOT)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets>=2.21.0,<3.0.0',
        'transformers==4.57.3', 'pyyaml>=6.0.1', 'pytest', 'matplotlib'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q',
        'tests/test_clip_gated_training.py', 'tests/test_clip_early_pulse.py',
        'tests/test_clip_gated_retention.py',
        'tests/test_clip_training.py', 'tests/test_clip_transfer_drive_backup.py',
        'tests/test_clip_paired_updates.py'], check=True)
    from src.clip_gated_training import run
    run_id = 'clip_gated_training_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path('/content') / run_id
    output.mkdir(exist_ok=False)
    destination = DRIVE_ROOT / run_id
    mirror = DriveMirror(output, destination)
    manifest = {'run_id': run_id, 'bundle_id': bundle_id, 'status': 'running',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'gpu': gpu, 'python': sys.version,
        'packages': {name: importlib.metadata.version(name) for name in
            ['torch', 'torchvision', 'transformers', 'datasets', 'numpy', 'PyYAML']},
        'drive_destination': str(destination), 'rules_refitted': False,
        'checkpoint_retention': 'hybrid_v1'}
    source_list = json.loads((ROOT / 'gated_source_manifest.json').read_text())
    for relative, expected in source_list.items():
        path = ROOT / relative
        if checksum(path) != expected:
            raise AssertionError(f'Source hash changed: {relative}')
        target = output / 'source_overlay' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (output / 'source_manifest.json').write_text(json.dumps(source_list, indent=2))
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    mirror.sync()  # Verified actual Drive write before the first trajectory.
    print('PERSISTENT_DESTINATION:', destination, flush=True)
    print('12 trajectories: 24 permanent checkpoints + one rolling backup. Allow 25–30 GB free Drive quota.', flush=True)
    print('No browser download needed. Wait for DRIVE_SYNC_COMPLETE.', flush=True)
    try:
        run(output, mirror.sync)
        verified = mirror.sync(final=True)
        manifest['required_checkpoints_verified'] = verify_inventory(verified['files'])
        manifest['status'] = 'complete_pending_drive_flush'
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        mirror.sync(final=True)
    except BaseException as error:
        manifest.update(status='failed_or_sync_incomplete', error=f'{type(error).__name__}: {error}')
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        # Best effort preserves closed artifacts; never masks the original error.
        try:
            mirror.sync()
        except BaseException as backup_error:
            print('BACKUP_ALSO_FAILED:', repr(backup_error), flush=True)
        raise
    print('DRIVE_TREE_VERIFIED:', destination, flush=True)
    print('Training and audit complete. The notebook must now flush Drive.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle-id', required=True)
    main(parser.parse_args().bundle_id)
