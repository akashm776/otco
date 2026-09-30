"""Bounded policy test with pinned input recovery and durable failure details."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colabs.backup_clip_run import checksum
from colabs.run_clip_continuation_timecourse import preflight as source_preflight
from colabs.transfer_drive_backup import DRIVE_ROOT, DriveMirror

TEST_FILES = ['tests/test_clip_policy_test.py', 'tests/test_clip_continuation_timecourse.py',
    'tests/test_clip_checkpoint_diagnostic.py', 'tests/test_clip_checkpoint_recovery.py',
    'tests/test_clip_paired_updates.py', 'tests/test_clip_gated_training.py',
    'tests/test_clip_transfer_drive_backup.py']


def require_no_previous_run(root, experiment):
    for marker in Path(root).glob('clip_policy_test_*/run_manifest.json'):
        if json.loads(marker.read_text()).get('experiment') == experiment:
            raise RuntimeError(f'Existing policy run: {marker.parent}. Inspect first; no automatic resume.')


def main(bundle_id):
    os.chdir(ROOT)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sources = json.loads((ROOT/'policy_source_manifest.json').read_text())
    for relative, expected in sources.items():
        if checksum(ROOT/relative) != expected:
            raise AssertionError(f'Source overlay changed: {relative}')
    parent = json.loads((ROOT/'configs/clip_checkpoint_diagnostic.json').read_text())
    p = {**parent, **json.loads((ROOT/'configs/clip_policy_test.json').read_text())}
    for root in [Path('/content'), DRIVE_ROOT]:
        require_no_previous_run(root, p['experiment'])
    # Reuse the executed time-course's input/GPU validation, without changing it.
    # Its experiment-specific old-run guard does not reject this new experiment.
    source, source_manifest, gpu = source_preflight(bundle_id, p)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==2.21.0',
        'transformers==4.57.3', 'pyyaml>=6.0.1', 'pytest', 'matplotlib'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *TEST_FILES], check=True)
    import torch
    from src.clip_policy_test import load_protocol, run
    from src.clip_checkpoint_diagnostic import write_json
    p = load_protocol()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device required')
    current = importlib.metadata.version('torch')
    if current.split('+')[0] != source_manifest['packages']['torch'].split('+')[0]:
        raise RuntimeError(f"Source torch={source_manifest['packages']['torch']}; current={current}. "
                           'Inspect version mismatch; CUDA stack is not replaced automatically.')
    run_id = 'clip_policy_test_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path('/content')/run_id
    output.mkdir(exist_ok=False)
    mirror = DriveMirror(output, DRIVE_ROOT/run_id)
    manifest = dict(run_id=run_id, experiment=p['experiment'], bundle_id=bundle_id, status='running',
        source_run=str(source), source_bundle=source_manifest['bundle_id'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gpu=gpu, python=sys.version, packages={n: importlib.metadata.version(n) for n in
            ['torch', 'torchvision', 'transformers', 'datasets', 'numpy', 'PyYAML']},
        source_checkpoints_read_only=True, independent_test_set=False, rules_refitted=False,
        literal_training_loader_resume=False, optimizer_step_budget=2400, trial_updates=0,
        exploratory_followup=True, reports_control_treatment=False,
        subprocess_log=os.environ.get('OTCO_POLICY_LOG_PATH'), drive_destination=str(mirror.destination))
    try:
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
        print('12 matched sets / 48 branches / exactly 2400 updates; no optimizer trials.', flush=True)
        summary = run(source, output, mirror.sync, p)
        manifest['status'] = 'complete_pending_drive_flush'
        write_json(output/'run_manifest.json', manifest)
        mirror.sync(final=True)
    except BaseException as error:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        manifest.update(status='failed_or_sync_incomplete', error=f'{type(error).__name__}: {error}')
        try:
            write_json(output/'failure.json', dict(error=manifest['error'], traceback=detail))
            write_json(output/'run_manifest.json', manifest)
            mirror.sync()
        except BaseException as backup_error:
            print('BACKUP_ALSO_FAILED:', repr(backup_error), flush=True)
        raise
    print('FINAL_COMPARISON:', json.dumps(summary, indent=2), flush=True)
    print('DRIVE_TREE_VERIFIED:', mirror.destination, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle-id', required=True)
    main(parser.parse_args().bundle_id)
