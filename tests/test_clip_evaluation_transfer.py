"""CPU checks; fabricated values here are fixtures, never experiment results."""

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from colabs import run_clip_evaluation_transfer as runner
from src import clip_evaluation_transfer as transfer
from src import clip_paired_updates as paired
from src.clip_early_pulse import state_digest


def test_pool_selection_is_fixed_unique_and_disjoint():
    groups = [SimpleNamespace(image_key=f'test/{i}') for i in range(1200)]
    data = SimpleNamespace(validation_canonical_dataset=SimpleNamespace(
        grouped_split=SimpleNamespace(groups=groups), species_ids=[i%200 for i in range(1200)]),
        train_dataset=SimpleNamespace(grouped_split=SimpleNamespace(groups=[SimpleNamespace(image_key='train/1')])))
    selected = transfer.select_pool(data)
    assert selected == transfer.select_pool(data)
    assert len(set(selected)) == 1024
    data.train_dataset.grouped_split.groups.append(groups[selected[0]])
    with pytest.raises(AssertionError, match='disjoint'):
        transfer.select_pool(data)


def rows_fixture():
    rows = []
    for step in transfer.prospective.STEPS:
        for trial in range(16):
            for arm in ['baseline','uniform_top8','hardest_real']:
                mean = 1. if arm == 'baseline' else .99999
                rows.append(dict(checkpoint_step=step, trial=trial, arm=arm,
                    heldout_loss={p:[mean]*16 for p in transfer.prospective.PARTS+['sequential']},
                    heldout_mean=mean, incremental_heldout_loss=mean-1.,
                    initial_heldout_mean=1.01, heldout_change=mean-1.01,
                    full_gradient_alignment=.05 if arm != 'baseline' else None))
    return rows


def test_transfer_aggregation_preserves_state_unit_and_absolute_changes():
    states = transfer.transfer_states(rows_fixture(), 789)
    assert len(states) == 5
    assert all(s['beneficial_trials']==16 and s['effect']==pytest.approx(-1e-5) for s in states)
    assert all(s['native_heldout_change']==pytest.approx(-.01) for s in states)


@pytest.mark.parametrize('kind', ['missing','duplicate','nonfinite','arithmetic'])
def test_bad_transfer_rows_fail(kind):
    rows = rows_fixture()
    if kind == 'missing':
        rows.pop()
    elif kind == 'duplicate':
        rows[-1] = rows[0]
    elif kind == 'nonfinite':
        rows[0]['heldout_loss']['sequential'][0] = float('nan')
    else:
        rows[0]['incremental_heldout_loss'] = 1.
    with pytest.raises(AssertionError):
        transfer.transfer_states(rows, 789)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3,3)

    def encode_images(self, x):
        return self.projection(x)

    def encode_texts(self, batch):
        return self.projection(batch['input_ids'])

    def get_logit_scale(self):
        return torch.tensor(1.)


def test_transfer_observation_preserves_rng_parameters_gradients_and_modes(tmp_path):
    model = Encoder().train()
    sum(p.sum() for p in model.parameters()).backward()
    observer = transfer.TransferObserver(tmp_path/'transfer', 789)
    observer.batches = [dict(pixel_values=torch.randn(4,3), input_ids=torch.randn(4,3))]
    observer.conditions = {'sequential':dict(batches=[list(range(4))])}
    before = state_digest(model.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    observer.start_checkpoint(model, 100)
    assert observer.evaluate(model) == observer.evaluate(model)
    assert before == state_digest(model.state_dict()) and model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p.grad, g) for p,g in zip(model.parameters(), gradients))


def complete_fixture(output):
    output.mkdir()
    for relative in runner.required_checkpoints():
        path = output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'checkpoint fixture')
    (output/'results').mkdir()
    (output/'results/completion.json').write_text(json.dumps(dict(status='complete', training_seeds=runner.SEEDS,
        checkpoint_steps=runner.STEPS, replay_branch_rows=720, transfer_branch_rows=720,
        evaluated_states=15, rules_refitted=False)))


def test_export_contains_actual_15_checkpoint_members_and_rejects_tampering(tmp_path):
    output, archive = tmp_path/'run', tmp_path/'complete.zip'
    complete_fixture(output)
    packed = runner.report_archive(output, archive, status='complete', bundle_id='test')
    assert len([name for name in packed['files'] if name.endswith('.pt')]) == 15
    runner.verify_archive(archive, bundle_id='test')
    corrupt = tmp_path/'corrupt.zip'
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(corrupt,'w') as changed:
        for name in original.namelist():
            content = original.read(name)
            if name.endswith('common_step_100.pt'):
                content = b'x'*len(content)
            changed.writestr(name, content)
    with pytest.raises(AssertionError, match='checksum'):
        runner.verify_archive(corrupt)


def test_missing_checkpoint_cannot_be_exported_as_complete(tmp_path):
    output = tmp_path/'run'
    complete_fixture(output)
    (output/runner.required_checkpoints()[0]).unlink()
    with pytest.raises(AssertionError, match='all 15'):
        runner.report_archive(output, tmp_path/'bad.zip', status='complete', bundle_id='test')


def test_completed_download_uses_actual_run_for_small_exports(tmp_path, monkeypatch):
    from colabs import recover_clip_evaluation_transfer_one_cell as recovery
    output, archive = tmp_path/'new_retraining_run', tmp_path/'complete.zip'
    complete_fixture(output)
    runner.report_archive(output, archive, status='complete', bundle_id='test')
    calls = []
    monkeypatch.setattr(recovery, 'run_recovery', lambda path, run_id: calls.append((path, run_id)))
    monkeypatch.setattr(runner, 'download', lambda _: pytest.fail('Full ZIP must not auto-download'))
    runner.download_complete(archive)
    assert calls == [(archive, output.name)]


def test_partial_archive_retains_available_checkpoints_but_is_not_complete(tmp_path):
    output = tmp_path/'run'
    output.mkdir()
    (output/'saved.pt').write_bytes(b'partial checkpoint')
    archive = tmp_path/'failed.zip'
    runner.report_archive(output, archive, status='interrupted_or_failed', bundle_id='test')
    assert 'run/saved.pt' in runner.verify_archive(archive, require_complete=False)['files']
    with pytest.raises(AssertionError, match='not complete'):
        runner.verify_archive(archive)


def test_runner_drive_is_opt_in_and_pins_replay_environment():
    source = Path(runner.__file__).read_text()
    assert '50*1024**3' in source
    assert "parser.add_argument('--drive-root'" in source
    assert 'drive.mount' not in source
    assert 'torchvision' in source and "reference['packages'][package]" in source


def test_replay_observer_is_opt_in_and_replays_new_pool_too():
    import inspect
    assert inspect.signature(paired.run).parameters['evaluation_observer'].default is None
    source = inspect.getsource(paired.run)
    assert 'evaluation_observer.evaluate(model) != transfer_losses' in source
    assert 'evaluation_observer.record(row, transfer_losses)' in source
