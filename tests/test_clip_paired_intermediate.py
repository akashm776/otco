from copy import deepcopy
import ast
import hashlib
import json
import random
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch
import yaml

from src import clip_paired_intermediate as study_module
from src import clip_paired_updates as paired
from src.clip_early_pulse import state_digest
from colabs.run_clip_paired_intermediate import verify_completion


ROOT = study_module.ROOT
REF = ROOT / 'experiment_results/clip_paired_updates_2026-09'
STEPS = [100, 250, 500, 750, 1001]


def test_fixed_three_seed_five_state_protocol():
    for seed in [42, 123, 456]:
        study, config, protocol = study_module.protocols(seed, f'seed_{seed}')
        assert study['training_seeds'] == [42, 123, 456]
        assert paired.validate_protocol(protocol) == (STEPS, 240)
        assert config['training']['seed'] == config['experiment']['seed'] == seed
        assert config['training']['epochs'] == 50 and study['stop_after_epochs'] == 13
        assert not config['ot']['enabled'] and not study['recalibrate']
        assert protocol['endpoint_loss_atol'] == 1e-8
        original = json.loads((REF / 'protocol.json').read_text())
        for key in ['coefficients', 'training_batches', 'batch_size', 'selection_seed', 'branch_seed', 'caption_epoch', 'primary_partitions']:
            assert protocol[key] == original[key]
        assert (ROOT / protocol['endpoint_reference_directory'] / 'paired_updates.jsonl').exists()


@pytest.mark.parametrize('step', [250, 500, 750])
def test_intermediate_snapshot_metadata_must_match_requested_step(step):
    snapshot = {'completed_updates': step, 'model': {}, 'optimizer': {}, 'scheduler': {}}
    assert paired.normalize_checkpoint(snapshot, step) == {'model': {}, 'optimizer': {}, 'scheduler': {}}
    assert paired.checkpoint_relative_path(step) == f'checkpoints/baseline/step_{step:06d}.pt'
    with pytest.raises(ValueError):
        paired.normalize_checkpoint(snapshot, step + 1)


@pytest.mark.parametrize('steps', [[], [100, 100], [500, 100], [0, 1001], [100, 3850], [100.0, 1001]])
def test_reject_invalid_stage_lists(steps):
    protocol = json.loads((REF / 'protocol.json').read_text())
    protocol['checkpoint_steps'] = steps
    with pytest.raises(ValueError):
        paired.validate_protocol(protocol)


def test_endpoint_loss_replay_has_fixed_tolerance_and_rejects_missing_or_nonfinite_rows():
    original = [json.loads(line) for line in (REF / 'paired_updates.jsonl').read_text().splitlines()]
    assert paired.compare_endpoint_losses(original, original, 100, 1e-8)['maximum_absolute_loss_difference'] == 0
    changed = deepcopy(original)
    changed[0]['heldout_mean'] += 5e-9
    assert paired.compare_endpoint_losses(changed, original, 100, 1e-8)['passed']
    for bad in ['too_large', 'nan', 'duplicate']:
        changed = deepcopy(original)
        if bad == 'duplicate':
            changed[1] = changed[0]
        else:
            changed[0]['heldout_mean'] += 2e-8 if bad == 'too_large' else float('nan')
        with pytest.raises(AssertionError):
            paired.compare_endpoint_losses(changed, original, 100, 1e-8)


def test_dense_observation_saves_snapshots_without_changing_training_state(tmp_path, monkeypatch):
    study, config, protocol = study_module.protocols(42, 'seed_42')
    holdout = set(json.loads((ROOT / config['dataset']['diagnostic_holdout_indices']).read_text()))
    data = SimpleNamespace(train_loader=[None] * 77, train_dataset=SimpleNamespace(source_indices=[i for i in range(5994) if i not in holdout]))
    monkeypatch.setattr(paired, 'cache_heldout', lambda *args: [])
    encoded = (torch.ones(2, 3), torch.ones(2, 3), 1.)
    def encode(*args):
        random.random(); np.random.rand(); torch.rand(1)
        return encoded
    monkeypatch.setattr(paired, 'encode_cached', encode)
    model = torch.nn.Linear(3, 3).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    (tmp_path / 'checkpoints/baseline').mkdir(parents=True)
    observer = study_module.IntermediateObserver(tmp_path, study, protocol)
    observer.initialize(model=model, processor=None, data=data, device='cpu', config=config, total_steps=3850)
    observer.bind_training_state(optimizer=optimizer, scheduler=scheduler, objective=None, data=data)
    initial = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    reference = {'initial_state_sha256': state_digest(initial), 'features_sha256': paired.feature_hash(encoded), 'learning_rates': [.001]}
    observer.references = {100: reference, 1001: reference}
    snapshot = lambda: state_digest({'initial': initial, 'rng': [random.getstate(), np.random.get_state(), torch.get_rng_state()]})
    before = snapshot()
    for step in STEPS:
        observer(model=model, epoch=2, global_step=step)
        assert snapshot() == before and model.training
        report = json.loads((tmp_path / f'results/baseline/diagnostics/{step:06d}_pulse_{step}/report.json').read_text())
        if step in [100, 1001]:
            assert report['historical_endpoint_state_exact'] is True
        else:
            saved = torch.load(tmp_path / paired.checkpoint_relative_path(step), weights_only=False)
            assert saved['completed_updates'] == step and saved['training_seed'] == 42
    assert observer.seen == STEPS


@pytest.mark.parametrize('key', ['initial_state_sha256', 'features_sha256', 'learning_rates'])
def test_historical_endpoint_state_mismatch_is_not_accepted(key):
    actual = {'initial_state_sha256': 'state', 'features_sha256': 'features', 'learning_rates': [.001]}
    wrong = {**actual, key: 'different'}
    assert study_module.check_reference_state(actual, actual)
    assert not study_module.check_reference_state(actual, wrong)


def populate(output):
    study = yaml.safe_load((ROOT / 'configs/clip_paired_intermediate.yaml').read_text())
    (output / 'results').mkdir(parents=True)
    (output / 'results/protocol.json').write_text(json.dumps(study))
    (output / 'results/completion.json').write_text(json.dumps(dict(status='complete', training_seeds=[42,123,456],
        checkpoint_steps=STEPS, total_branch_rows=720, new_intermediate_branch_rows=432, endpoint_replay_branch_rows=288)))
    original = [json.loads(line) for line in (REF / 'paired_updates.jsonl').read_text().splitlines()]
    originals = json.loads((REF / 'summary.json').read_text())
    provenance = json.loads((REF / 'checkpoint_provenance.json').read_text())
    for seed in [42, 123, 456]:
        directory = output / f'seed_{seed}/results/paired'
        directory.mkdir(parents=True)
        # Synthetic completion fixtures for plumbing tests, not new scientific results.
        rows = [{**r, 'checkpoint_step': step} for step in STEPS for r in original if r['checkpoint_step'] == (1001 if step == 1001 else 100)]
        summaries = [{**r, 'checkpoint_step': step} for step in STEPS for r in originals if r['checkpoint_step'] == (1001 if step == 1001 else 100)]
        states = [{**provenance[1 if step == 1001 else 0], 'step': step} for step in STEPS]
        (directory / 'paired_updates.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        for name, value in [('summary.json', summaries), ('checkpoint_provenance.json', states),
            ('completion.json', dict(status='complete', checkpoint_steps=STEPS, rows=240)),
            ('endpoint_replay_checks.json', [dict(step=step, rows=48, passed=True, loss_absolute_tolerance=1e-8, maximum_absolute_loss_difference=0.) for step in [100,1001]])]:
            (directory / name).write_text(json.dumps(value))
    return study


def test_complete_study_has_720_rows_and_three_seed_curves(tmp_path):
    study = populate(tmp_path)
    verify_completion(tmp_path)
    records = study_module.aggregate(tmp_path, study)
    assert len(records) == 30 and sum(r['historical_endpoint_replay'] for r in records) == 12
    assert (tmp_path / 'results/intermediate_usefulness.png').exists()


@pytest.mark.parametrize('failure', ['duplicate_branch', 'missing_replay', 'failed_replay', 'nonfinite_loss'])
def test_invalid_complete_results_are_rejected(tmp_path, failure):
    populate(tmp_path)
    directory = tmp_path / 'seed_456/results/paired'
    if failure in ['duplicate_branch', 'nonfinite_loss']:
        path = directory / 'paired_updates.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if failure == 'duplicate_branch':
            rows[-1] = rows[0]
        else:
            rows[0]['heldout_loss']['sequential'][0] = float('nan')
        path.write_text('\n'.join(json.dumps(r) for r in rows))
    else:
        path = directory / 'endpoint_replay_checks.json'
        rows = json.loads(path.read_text())
        if failure == 'missing_replay':
            rows.pop()
        else:
            rows[0]['maximum_absolute_loss_difference'] = 1e-5
        path.write_text(json.dumps(rows))
    with pytest.raises(AssertionError):
        verify_completion(tmp_path)


def test_intermediate_runner_stays_local_and_pins_compute_stack():
    source = (ROOT / 'colabs/run_clip_paired_intermediate.py').read_text()
    for forbidden in ['/content/drive', 'drive.mount', 'RunBackup', 'copy_verified']:
        assert forbidden not in source
    assert '25 * 1024**3' in source
    assert "['torch', 'torchvision']" in source and 'No training started.' in source


def handoff_namespace():
    source = (ROOT / 'colabs/clip_paired_intermediate_one_cell.py').read_text()
    tree = ast.parse(source)
    assert isinstance(tree.body[-1], ast.Expr) and tree.body[-1].value.func.id == 'run_intermediate_study'
    tree.body.pop()  # Inspect definitions without starting GPU work or networking.
    namespace = {'__name__': 'handoff_test'}
    exec(compile(tree, '<intermediate handoff test>', 'exec'), namespace)
    return namespace


def test_intermediate_cell_pins_committed_runner():
    namespace = handoff_namespace()
    source = subprocess.check_output(['git', 'show', namespace['SOURCE_COMMIT'] + ':colabs/run_clip_paired_intermediate.py'], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == namespace['RUNNER_SHA256']
    cell = (ROOT / 'colabs/clip_paired_intermediate_one_cell.py').read_text()
    assert '/content/drive' not in cell and 'drive.mount' not in cell


@pytest.mark.parametrize('wrong_total', [False, True])
def test_intermediate_cell_redownloads_only_complete_matching_archive(tmp_path, monkeypatch, wrong_total):
    namespace = handoff_namespace()
    run_id = 'clip_paired_intermediate_fixture'
    control = tmp_path / ('otco_control_' + run_id)
    control.mkdir()
    archive = control / (run_id + '_complete.zip')
    completion = dict(status='complete', training_seeds=[42, 123, 456], checkpoint_steps=STEPS,
                      total_branch_rows=719 if wrong_total else 720,
                      new_intermediate_branch_rows=432, endpoint_replay_branch_rows=288)
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr(run_id + '/results/run_manifest.json', json.dumps(dict(source_commit=namespace['SOURCE_COMMIT'], status='complete')))
        bundle.writestr(run_id + '/results/completion.json', json.dumps(completion))
    downloads = []
    monkeypatch.setitem(sys.modules, 'google.colab', SimpleNamespace(files=SimpleNamespace(download=downloads.append)))
    namespace['Path'] = lambda _: tmp_path
    monkeypatch.setattr(namespace['shutil'], 'which', lambda _: pytest.fail('Must not train or require GPU for existing archive'))
    if wrong_total:
        with pytest.raises(RuntimeError, match='failed verification'):
            namespace['run_intermediate_study']()
        assert downloads == []
    else:
        namespace['run_intermediate_study']()
        assert downloads == [str(archive)]
