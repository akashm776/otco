"""A100-only equal-exposure timing experiment with mandatory Drive backups."""

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
from colabs.exposure_checkpoint_retention import verify_inventory
from colabs.transfer_drive_backup import DriveMirror, DRIVE_ROOT, require_drive

TEST_FILES = [
    'tests/test_clip_exposure_matched.py', 'tests/test_clip_exposure_retention.py',
    'tests/test_clip_gated_training.py', 'tests/test_clip_gated_retention.py',
    'tests/test_clip_early_pulse.py', 'tests/test_clip_training.py',
    'tests/test_clip_transfer_drive_backup.py', 'tests/test_clip_paired_updates.py',
]


def preflight(bundle_id):
    if len(bundle_id) != 64 or any(c not in '0123456789abcdef' for c in bundle_id):
        raise ValueError('Expected source bundle SHA256')
    require_drive(DRIVE_ROOT / 'preflight')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
        '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    if 'A100' not in gpu or int(gpu.split(',')[1].strip()) < 39000:
        raise RuntimeError(f'A100 with at least 39 GB required; found {gpu}')
    if shutil.disk_usage('/content').free < 40 * 1024**3:
        raise RuntimeError('At least 40 GiB free Colab disk required')
    # Refuse a duplicate study even if a formatting-only bundle rebuild changes
    # its hash. Failed runs require inspection, not automatic retries/resumes.
    for existing in DRIVE_ROOT.glob('clip_exposure_matched_*/run_manifest.json'):
        manifest = json.loads(existing.read_text())
        if manifest.get('experiment') == 'clip_exposure_matched_v1':
            raise RuntimeError(f'Existing exposure-matched study: {existing.parent}. Inspect before rerunning; no automatic resume.')
    return gpu


def main(bundle_id):
    gpu = preflight(bundle_id)
    os.chdir(ROOT)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sources = json.loads((ROOT / 'exposure_source_manifest.json').read_text())
    for relative, expected in sources.items():
        if checksum(ROOT / relative) != expected:
            raise AssertionError(f'Source hash changed: {relative}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets>=2.21.0,<3.0.0',
        'transformers==4.57.3', 'pyyaml>=6.0.1', 'pytest', 'matplotlib'], check=True)
    # Explicit local tests package is part of the checksummed source bundle.
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *TEST_FILES], check=True)
    from src.clip_exposure_matched import run
    run_id = 'clip_exposure_matched_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path('/content') / run_id
    output.mkdir(exist_ok=False)
    destination = DRIVE_ROOT / run_id
    mirror = DriveMirror(output, destination)
    manifest = {'run_id': run_id, 'experiment': 'clip_exposure_matched_v1',
        'bundle_id': bundle_id, 'status': 'running',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'gpu': gpu, 'python': sys.version,
        'packages': {name: importlib.metadata.version(name) for name in
            ['torch', 'torchvision', 'transformers', 'datasets', 'numpy', 'PyYAML']},
        'drive_destination': str(destination), 'rules_refitted': False,
        'checkpoint_retention': 'exposure_hybrid_v1', 'exposure_matched': True,
        'compute_matched': False, 'random_controls_per_seed': 3}
    for relative in sources:
        target = output / 'source_overlay' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (output / 'source_manifest.json').write_text(json.dumps(sources, indent=2))
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    mirror.sync()
    print('PERSISTENT_DESTINATION:', destination, flush=True)
    print('12 trajectories; 24 permanent checkpoints + one rolling backup. Keep 30 GB additional Drive quota free.', flush=True)
    print('Do not rerun this cell. Wait for DRIVE_SYNC_COMPLETE before releasing the runtime.', flush=True)
    try:
        report = run(output, mirror.sync)
        verified = mirror.sync(final=True)
        manifest['required_checkpoints_verified'] = verify_inventory(verified['files'])
        manifest['status'] = 'complete_pending_drive_flush'
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        mirror.sync(final=True)
    except BaseException as error:
        manifest.update(status='failed_or_sync_incomplete', error=f'{type(error).__name__}: {error}')
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        try:
            mirror.sync()
        except BaseException as backup_error:
            print('BACKUP_ALSO_FAILED:', repr(backup_error), flush=True)
        raise
    # Compact summary at the end survives verbose trainer output truncation.
    print('FINAL_COMPARISON:', json.dumps(report, indent=2), flush=True)
    print('DRIVE_TREE_VERIFIED:', destination, flush=True)
    print('Training and audit complete; notebook must now flush Drive.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle-id', required=True)
    main(parser.parse_args().bundle_id)
