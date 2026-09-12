import json
import random
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from colabs.run_clip_paired_seed_replication import report_archive, verify_completion
from src import clip_paired_seed_replication as replication
from src import clip_paired_updates as paired
from src.clip_early_pulse import state_digest


def test_new_training_seeds_keep_fixed_probe_protocol():
    for seed in [123, 456]:
        study, config, protocol = replication.protocols(seed, f'seed_{seed}')
        assert config['experiment']['seed'] == config['training']['seed'] == seed
        assert config['training']['epochs'] == 50 and study['stop_after_epochs'] == 13
        assert protocol['diagnostic_caption_seed'] == 42
        assert protocol['selection_seed'] == 20260912 and protocol['caption_epoch'] == 2
        assert protocol['coefficients'] == {'uniform_top8': 0.06005351588542947, 'hardest_real': 0.3753711620568176}
        assert set(protocol['fixed_input_sha256']) == {'training_batches.json', 'heldout_partitions.json'}
        assert not config['ot']['enabled'] and not study['recalibrate']
    with pytest.raises(ValueError):
        replication.protocols(42, 'seed_42')


def test_recreated_exclusions_select_exact_original_training_batches():
    study, config, protocol = replication.protocols(123, 'seed_123')
    holdout = set(json.loads((replication.ROOT / config['dataset']['diagnostic_holdout_indices']).read_text()))
    indices = [i for i in range(5994) if i not in holdout]
    calibration = replication.fixed_calibration_exclusions(indices, study, protocol)
    excluded = [i for batch in calibration['batches'] for i in batch['source_indices']]
    assert len(set(excluded)) == 256 and not set(excluded) & holdout
    positions = paired.select_batches(indices, excluded, seed=protocol['selection_seed'], count=16, size=64)
    selected = [[indices[i] for i in batch] for batch in positions]
    original = json.loads((replication.ROOT / 'experiment_results/clip_paired_updates_2026-09/training_source_indices.json').read_text())
    assert selected == original


def test_checkpoint_observer_is_observational_and_saves_closed_states(tmp_path, monkeypatch):
    study, config, protocol = replication.protocols(123, 'seed_123')
    holdout = set(json.loads((replication.ROOT / config['dataset']['diagnostic_holdout_indices']).read_text()))
    data = SimpleNamespace(train_loader=[None] * 77, train_dataset=SimpleNamespace(
        source_indices=[i for i in range(5994) if i not in holdout]))
    monkeypatch.setattr(paired, 'cache_heldout', lambda *args: [])
    # Deliberately consume global RNGs inside observational evaluation.
    def encode(*args):
        random.random()
        np.random.rand()
        return torch.rand(2, 3), torch.rand(2, 3), 1.
    monkeypatch.setattr(paired, 'encode_cached', encode)
    model = torch.nn.Linear(3, 3).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    (tmp_path / 'checkpoints/baseline').mkdir(parents=True)
    observer = replication.CheckpointObserver(tmp_path, study, protocol)
    observer.initialize(model=model, processor=None, data=data, device='cpu', config=config, total_steps=3850)
    observer.bind_training_state(optimizer=optimizer, scheduler=scheduler, objective=None, data=data)
    snapshot = lambda: state_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
        'python': random.getstate()})
    before = snapshot()
    for step in [100, 1001]:
        observer(model=model, epoch=2 if step == 100 else 13, global_step=step)
        assert snapshot() == before and model.training
    state = torch.load(tmp_path / 'checkpoints/baseline/common_step_100.pt', weights_only=False)
    assert state['training_seed'] == 123 and state['completed_updates'] == 100
    hashes = json.loads((tmp_path / 'results/prefix_hashes.json').read_text())
    assert hashes == {key: state_digest(value) for key, value in state.items()}
    assert json.loads((tmp_path / 'results/calibration.json').read_text())['recalibrated'] is False
    assert observer.seen == [100, 1001]


def populate_completed(output):
    evidence = replication.ROOT / 'experiment_results/clip_paired_updates_2026-09'
    (output / 'results').mkdir(parents=True)
    (output / 'results/completion.json').write_text(json.dumps(dict(
        status='complete', new_training_seeds=[123, 456], new_branch_rows=192)))
    for seed in [123, 456]:
        directory = output / f'seed_{seed}/results/paired'
        directory.mkdir(parents=True)
        for name in ['completion.json', 'paired_updates.jsonl', 'summary.json']:
            (directory / name).write_bytes((evidence / name).read_bytes())


def test_combined_completion_and_seed_level_aggregation(tmp_path):
    populate_completed(tmp_path)
    verify_completion(tmp_path)
    records = replication.aggregate(tmp_path, [123, 456])
    assert len(records) == 12
    assert {r['training_seed'] for r in records} == {42, 123, 456}
    assert sum(r['previous_reference_run'] for r in records) == 4
    assert (tmp_path / 'results/seed_comparison.png').exists()


@pytest.mark.parametrize('failure', ['duplicate_branch', 'missing_seed', 'wrong_total'])
def test_reject_false_completion(tmp_path, failure):
    populate_completed(tmp_path)
    if failure == 'duplicate_branch':
        path = tmp_path / 'seed_456/results/paired/paired_updates.jsonl'
        rows = path.read_text().splitlines()
        rows[-1] = rows[0]
        path.write_text('\n'.join(rows))
    elif failure == 'missing_seed':
        path = tmp_path / 'seed_456/results/paired/completion.json'
        path.write_text('{"status": "running", "rows": 48}')
    else:
        path = tmp_path / 'results/completion.json'
        path.write_text('{"status": "complete", "new_training_seeds": [123, 456], "new_branch_rows": 96}')
    with pytest.raises(AssertionError):
        verify_completion(tmp_path)


def test_browser_archive_excludes_checkpoints_keeps_partial_reports(tmp_path):
    output = tmp_path / 'run'
    (output / 'seed_123/checkpoints').mkdir(parents=True)
    (output / 'seed_123/checkpoints/state.pt').write_bytes(b'checkpoint')
    (output / 'progress.json').write_text('{"status":"running"}')
    archive = tmp_path / 'partial.zip'
    report_archive(output, archive)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ['run/progress.json'] and bundle.testzip() is None


def test_overnight_runner_has_no_drive_writes_or_mount():
    source = (replication.ROOT / 'colabs/run_clip_paired_seed_replication.py').read_text()
    for forbidden in ['/content/drive', 'drive.mount', 'RunBackup', 'copy_verified']:
        assert forbidden not in source
    assert 'colab_local_only_no_drive' in source
