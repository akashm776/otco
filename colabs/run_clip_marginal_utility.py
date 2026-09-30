"""Fail-closed Drive launcher for rollback-only marginal-utility diagnostics."""
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

TEST_FILES = ['tests/test_clip_marginal_utility.py', 'tests/test_clip_continuation_timecourse.py', 'tests/test_clip_checkpoint_diagnostic.py',
    'tests/test_clip_checkpoint_recovery.py', 'tests/test_clip_paired_updates.py',
    'tests/test_clip_gated_training.py', 'tests/test_clip_transfer_drive_backup.py']


def require_no_previous_run(root, experiment):
    for marker in Path(root).glob('clip_marginal_utility_*/run_manifest.json'):
        if json.loads(marker.read_text()).get('experiment') == experiment:
            raise RuntimeError(f'Existing marginal-utility run: {marker.parent}. Inspect before rerun; no automatic resume.')


def preflight(bundle_id, p):
    if len(bundle_id) != 64 or any(c not in '0123456789abcdef' for c in bundle_id):
        raise ValueError('Expected source-bundle SHA256')
    require_drive(DRIVE_ROOT/'preflight')
    require_no_previous_run(DRIVE_ROOT, p['experiment'])
    require_no_previous_run(Path('/content'), p['experiment'])
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
        '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    if 'A100' not in gpu or int(gpu.split(',')[1].strip()) < 39000:
        raise RuntimeError(f'Requires A100 with at least 39 GB; found {gpu}')
    source = DRIVE_ROOT/p['source_run']
    if not source.is_dir():
        candidates = [DRIVE_ROOT/ARCHIVE_NAME, Path('/content')/ARCHIVE_NAME]
        archive = next((path for path in candidates if path.is_file()), None)
        if archive is None:
            raise FileNotFoundError(f'Upload {ARCHIVE_NAME} to {DRIVE_ROOT} or /content. No old training is rerun.')
        source = recover_inputs(archive, Path('/content/otco_diagnostic_inputs')/p['source_run'], p)
    if shutil.disk_usage('/content').free < 15*1024**3:
        raise RuntimeError('15 GiB free runtime disk required after input recovery')
    manifest = json.loads((source/'run_manifest.json').read_text())
    backup = json.loads((source/'DRIVE_BACKUP_MANIFEST.json').read_text())
    if (manifest['run_id'] != p['source_run'] or manifest['bundle_id'] != p['source_bundle']
            or manifest['status'] != 'complete_pending_drive_flush'
            or backup['run_id'] != p['source_run'] or backup['status'] != 'verified_tree'):
        raise ValueError('Wrong or incomplete source run')
    for relative, sha in p['checkpoints'].items():
        path = source/relative
        if not path.is_file() or backup['files'][relative]['sha256'] != sha:
            raise ValueError(f'Missing or unpinned checkpoint: {relative}')
        if path.stat().st_size != backup['files'][relative]['bytes']:
            raise ValueError(f'Checkpoint size mismatch: {relative}')
    # Every tensor file is SHA256 checked again immediately before unpickling.
    return source, manifest, gpu


def main(bundle_id):
    os.chdir(ROOT)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sources = json.loads((ROOT/'marginal_source_manifest.json').read_text())
    for relative, expected in sources.items():
        if checksum(ROOT/relative) != expected:
            raise AssertionError(f'Source overlay changed: {relative}')
    # Load configuration without importing optional ML dependencies before installation.
    parent = json.loads((ROOT/'configs/clip_checkpoint_diagnostic.json').read_text())
    p = {**parent, **json.loads((ROOT/'configs/clip_marginal_utility.json').read_text())}
    source, source_manifest, gpu = preflight(bundle_id, p)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==2.21.0',
        'transformers==4.57.3', 'pyyaml>=6.0.1', 'pytest', 'matplotlib'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *TEST_FILES], check=True)
    import torch
    from src.clip_marginal_utility import load_protocol, run
    from src.clip_checkpoint_diagnostic import write_json
    p = load_protocol()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device required')
    current = importlib.metadata.version('torch')
    if current.split('+')[0] != source_manifest['packages']['torch'].split('+')[0]:
        raise RuntimeError(f"Source torch={source_manifest['packages']['torch']}; current={current}. "
                           'Inspect optimizer-version mismatch; CUDA stack will not be replaced automatically.')
    run_id = 'clip_marginal_utility_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path('/content')/run_id
    output.mkdir(exist_ok=False)
    mirror = DriveMirror(output, DRIVE_ROOT/run_id)
    manifest = dict(run_id=run_id, experiment=p['experiment'], bundle_id=bundle_id, status='running',
        source_run=str(source), source_bundle=source_manifest['bundle_id'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gpu=gpu, python=sys.version, packages={n: importlib.metadata.version(n) for n in
            ['torch', 'torchvision', 'transformers', 'datasets', 'numpy', 'PyYAML']},
        source_checkpoints_read_only=True, independent_test_set=False, rules_refitted=False,
        literal_training_loader_resume=False, optimizer_step_budget=1536, continuation_updates=1200, trial_updates=336,
        trial_steps_rolled_back=True, exploratory_followup=True, drive_destination=str(mirror.destination))
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
    print('PERSISTENT_DESTINATION:', mirror.destination, flush=True)
    print('12 history pairs / 1200 continuation updates + 336 rollback trial updates = 1536.', flush=True)
    print('Matched next-step native vs auxiliary trials; scores and reports NEVER control continuation.', flush=True)
    try:
        summary = run(source, output, mirror.sync, p)
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
    print('DRIVE_TREE_VERIFIED:', mirror.destination, flush=True)
    print('Marginal-utility audit complete; notebook must now flush Drive.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle-id', required=True)
    main(parser.parse_args().bundle_id)
