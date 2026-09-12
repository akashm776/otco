from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from model.clip_training import CLIPTrainingObjective
from src import clip_train
from src.clip_early_pulse import state_digest
from src.clip_gradient_stages import observational_state
from src.clip_paired_updates import (select_batches, one_update, restore_branch,
    partition_losses, primary_mean, encode_cached, normalize_checkpoint)


@pytest.fixture(autouse=True)
def rng_isolation():
    with observational_state(torch.nn.Identity()):
        yield


class TinyCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = torch.nn.Module()
        self.clip_model.logit_scale = torch.nn.Parameter(torch.tensor(1.))
        self.clip_model.visual_projection = torch.nn.Linear(6, 4)
        self.clip_model.text_projection = torch.nn.Linear(6, 4)
        self.frozen = torch.nn.Parameter(torch.randn(6), requires_grad=False)

    def encode_images(self, x):
        return F.normalize(self.clip_model.visual_projection(x + self.frozen), dim=-1)

    def encode_texts(self, x):
        return F.normalize(self.clip_model.text_projection(x['input_ids']), dim=-1)

    def get_logit_scale(self):
        return self.clip_model.logit_scale.exp()

    def forward(self, images, texts):
        i, t = self.encode_images(images), self.encode_texts(texts)
        scale = self.get_logit_scale()
        raw = t @ i.T
        return SimpleNamespace(image_features=i, text_features=t, raw_similarity=raw,
                               logit_scale=scale, logits=raw * scale)


def fixture():
    torch.manual_seed(44)
    model = TinyCLIP()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    scheduler = clip_train.build_scheduler(optimizer, warmup_steps=2, total_steps=100)
    batch = {'pixel_values': torch.randn(10, 6), 'input_ids': torch.randn(10, 6), 'species_ids': torch.arange(10)}
    config = {'training': {'mixed_precision': 'none'}, 'optimizer': {'max_gradient_norm': 1.}}
    one_update(model, optimizer, batch, config, arm='baseline', coefficient=0., seed=1)
    scheduler.step()
    initial = deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()})
    return model, optimizer, scheduler, batch, config, initial


def test_selection_is_disjoint_reproducible_and_excludes_calibration():
    sources = list(range(100, 200))
    chosen = select_batches(sources, [100,101,102], seed=4, count=4, size=10)
    assert chosen == select_batches(sources, [100,101,102], seed=4, count=4, size=10)
    indices = [sources[i] for batch in chosen for i in batch]
    assert len(indices) == len(set(indices)) == 40
    assert not set(indices) & {100,101,102}
    with pytest.raises(ValueError):
        select_batches(sources, [], seed=1, count=11, size=10)


def test_one_step_matches_existing_trainer_exactly():
    model, optimizer, scheduler, batch, config, initial = fixture()
    restore_branch(model, optimizer, scheduler, initial, 9)
    one_update(model, optimizer, batch, config, arm='baseline', coefficient=0., seed=9)
    scheduler.step()
    expected = state_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()})
    restore_branch(model, optimizer, scheduler, initial, 9)
    data = SimpleNamespace(train_loader=[batch], train_dataset=SimpleNamespace(set_epoch=lambda x: None))
    clip_train.train_epoch(model, CLIPTrainingObjective({'enabled': False}), data, optimizer, scheduler,
        torch.device('cpu'), config, epoch=1, global_step=1, remaining_gradient_diagnostics=0)
    assert expected == state_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()})


def test_branch_reset_preserves_source_optimizer_and_reproduces_update():
    model, optimizer, scheduler, batch, config, initial = fixture()
    before = state_digest(initial)
    deltas = []
    for _ in range(2):
        restore_branch(model, optimizer, scheduler, initial, 5)
        assert state_digest(optimizer.state_dict()) == state_digest(initial['optimizer'])
        _, delta = one_update(model, optimizer, batch, config, arm='uniform_top8', coefficient=.06, seed=5)
        deltas.append(delta)
    assert torch.equal(*deltas)
    assert state_digest(initial) == before
    assert torch.equal(model.frozen, initial['model']['frozen'])


@pytest.mark.parametrize('arm', ['uniform_top8','hardest_real'])
def test_zero_auxiliary_reproduces_native_step(arm):
    model, optimizer, scheduler, batch, config, initial = fixture()
    restore_branch(model, optimizer, scheduler, initial, 5)
    _, native = one_update(model, optimizer, batch, config, arm='baseline', coefficient=0., seed=5)
    restore_branch(model, optimizer, scheduler, initial, 5)
    _, other = one_update(model, optimizer, batch, config, arm=arm, coefficient=0., seed=5)
    torch.testing.assert_close(native, other, rtol=0, atol=0)


def test_auxiliary_directions_and_actual_update_are_measured():
    model, optimizer, scheduler, batch, config, initial = fixture()
    restore_branch(model, optimizer, scheduler, initial, 2)
    metrics, delta = one_update(model, optimizer, batch, config, arm='hardest_real', coefficient=.375, seed=2)
    assert -1 <= metrics['full_gradient_alignment'] <= 1
    assert -1 <= metrics['projection_gradient_alignment'] <= 1
    assert metrics['weighted_gradient_ratio'] > 0
    expected = torch.cat([(p.detach() - initial['model'][name]).flatten() for name,p in model.named_parameters() if p.requires_grad])
    assert torch.equal(delta, expected)


def test_evaluation_is_symmetric_and_observational():
    model, optimizer, scheduler, batch, config, initial = fixture()
    rng = torch.get_rng_state().clone()
    modes = [m.training for m in model.modules()]
    encoded = encode_cached(model, [batch])
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    conditions = {'sequential': {'batches': [list(range(10))]}}
    losses = partition_losses(encoded, conditions)
    images, texts, scale = encoded
    logits = texts.double() @ images.double().T * scale
    targets = torch.arange(10)
    expected = .5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    assert primary_mean(losses, ['sequential']) == pytest.approx(float(expected), abs=1e-12)


def test_checkpoint_types_require_exact_stage():
    with pytest.raises(ValueError):
        normalize_checkpoint({'completed_updates': 99}, 100)
    with pytest.raises(ValueError):
        normalize_checkpoint({'epoch': 50, 'metrics': {'global_step': 3850}}, 1001)
