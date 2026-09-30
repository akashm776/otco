"""Exact endpoint replay with pinned archived actions and durable failure details."""
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

TEST_FILES = ['tests/test_clip_policy_pool_test.py', 'tests/test_clip_policy_test.py', 'tests/test_clip_policy_robustness.py', 'tests/test_clip_continuation_timecourse.py',
    'tests/test_clip_checkpoint_diagnostic.py', 'tests/test_clip_checkpoint_recovery.py',
    'tests/test_clip_paired_updates.py', 'tests/test_clip_gated_training.py',
    'tests/test_clip_transfer_drive_backup.py']


def require_no_previous_run(root, experiment):
    for marker in Path(root).glob('clip_policy_pool_test_*/run_manifest.json'):
        if json.loads(marker.read_text()).get('experiment') == experiment:
            raise RuntimeError(f'Existing policy run: {marker.parent}. Inspect first; no automatic resume.')


def locate_policy_reference(roots):
    from scripts.analyze_clip_policy_robustness import SPEC
    # Prefer the exact complete folder; never search failed runs or marginal runs.
    folders = [Path(root)/SPEC['source_run'] for root in roots
               if (Path(root)/SPEC['source_run']).is_dir()]
    if len(folders) == 1:
        return folders[0]
    if len(folders) > 1:
        raise RuntimeError('Multiple policy source folders; retain one explicit source.')
    archives = sorted({p for root in roots for p in Path(root).glob(SPEC['source_run']+'*.zip')})
    if len(archives) != 1:
        raise FileNotFoundError('Need exactly one completed policy folder or single ZIP: '
            + SPEC['source_run'] + ' in MyDrive/OTCO/evaluation_transfer or /content. '
            'Do not use a marginal-utility run or a partial/multipart ZIP.')
    return archives[0]


def main(bundle_id):
    os.chdir(ROOT)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sources = json.loads((ROOT/'policy_pool_source_manifest.json').read_text())
    for relative, expected in sources.items():
        if checksum(ROOT/relative) != expected:
            raise AssertionError(f'Source overlay changed: {relative}')
    parent = json.loads((ROOT/'configs/clip_checkpoint_diagnostic.json').read_text())
    p = {**parent, **json.loads((ROOT/'configs/clip_policy_pool_test.json').read_text())}
    for root in [Path('/content'), DRIVE_ROOT]:
        require_no_previous_run(root, p['experiment'])
    # Reuse the executed time-course's input/GPU validation, without changing it.
    # Its experiment-specific old-run guard does not reject this new experiment.
    from scripts.analyze_clip_policy_robustness import verify_input, extract_rows, checked_json
    policy_source = locate_policy_reference([DRIVE_ROOT, Path('/content')])
    reference_files, provenance = verify_input(policy_source)
    extract_rows(reference_files)  # Recompute and validate the complete archived grid.
    archived_manifest = checked_json(reference_files['run_manifest.json'])
    source, source_manifest, gpu = source_preflight(bundle_id, p)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==2.21.0',
        'transformers==4.57.3', 'numpy==2.1.3', 'pyyaml==6.0.3', 'pytest', 'matplotlib'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *TEST_FILES], check=True)
    import torch
    from src.clip_policy_pool_test import load_protocol, run
    from src.clip_checkpoint_diagnostic import write_json
    p = load_protocol()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device required')
    current = importlib.metadata.version('torch')
    # Exact replay is only attempted in the same relevant numerical stack.
    for name in ['torch', 'torchvision', 'transformers', 'datasets', 'numpy', 'PyYAML']:
        if importlib.metadata.version(name) != archived_manifest['packages'][name]:
            raise RuntimeError(f'Policy replay package mismatch: {name}; inspect before running.')
    if gpu != archived_manifest['gpu']:
        raise RuntimeError('Policy replay requires the archived A100 GPU identity.')
    if current.split('+')[0] != source_manifest['packages']['torch'].split('+')[0]:
        raise RuntimeError(f"Source torch={source_manifest['packages']['torch']}; current={current}. "
                           'Inspect version mismatch; CUDA stack is not replaced automatically.')
    run_id = 'clip_policy_pool_test_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path('/content')/run_id
    output.mkdir(exist_ok=False)
    mirror = DriveMirror(output, DRIVE_ROOT/run_id)
    manifest = dict(run_id=run_id, experiment=p['experiment'], bundle_id=bundle_id, status='running',
        source_run=str(source), source_bundle=source_manifest['bundle_id'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gpu=gpu, python=sys.version, packages={n: importlib.metadata.version(n) for n in
            ['torch', 'torchvision', 'transformers', 'datasets', 'numpy', 'PyYAML']},
        source_checkpoints_read_only=True, independent_test_set=False, rules_refitted=False,
        policy_reference=str(policy_source), policy_reference_verified=True, gate_decisions_recomputed=False,
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
        print('12 archived matched sets / 48 branches / 2400 replay updates; full-512 plus 32 B64 partitions.', flush=True)
        summary = run(source, output, mirror.sync, reference_files, provenance, p)
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
