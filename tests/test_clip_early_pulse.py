from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.clip_early_pulse import (PulseObjective, auxiliary_losses, calibrate,
    calibration_coefficients, state_digest)
from src.clip_gradient_stages import observational_state
from src.clip_warmup_readiness import probe_objectives
from model.clip_training import native_clip_contrastive_loss


@pytest.mark.parametrize('arm', ['uniform_top8', 'hardest_real'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_candidate_loss_matches_existing_training_implementation(arm, dtype):
    import yaml
    from src.clip_gradient_stages import ROOT
    from model.clip_training import CLIPTrainingObjective
    name = 'uniform_top8' if arm == 'uniform_top8' else 'hardest_real'
    config = yaml.safe_load((ROOT / f'configs/hf_cub200_clip_vit_b32_{name}_relative_native_strength.yaml').read_text())
    out = output_fixture()
    out.image_features = out.image_features.to(dtype)
    out.text_features = out.text_features.to(dtype)
    out.raw_similarity = out.text_features @ out.image_features.T
    out.logits = out.raw_similarity * out.logit_scale
    original = CLIPTrainingObjective(config['ot'])(out, species_ids=torch.arange(10), step=2000)
    torch.testing.assert_close(auxiliary_losses(out)[arm], original.ot_loss, rtol=0, atol=0)


@pytest.fixture(autouse=True)
def isolate_rng():
    with observational_state(torch.nn.Identity()):
        yield


def output_fixture():
    torch.manual_seed(91)
    i = F.normalize(torch.randn(10, 6), dim=-1).requires_grad_()
    t = F.normalize(torch.randn(10, 6), dim=-1).requires_grad_()
    scale = torch.tensor(4., requires_grad=True)
    raw = t @ i.T
    return SimpleNamespace(image_features=i, text_features=t, raw_similarity=raw,
                           logit_scale=scale, logits=raw * scale)


@pytest.mark.parametrize('arm,key', [('uniform_top8', 'u8'), ('hardest_real', 'real')])
def test_auxiliary_matches_readiness_loss_and_gradients(arm, key):
    out = output_fixture()
    actual = auxiliary_losses(out)[arm]
    expected = probe_objectives(out.image_features, out.text_features, out.logit_scale)[key]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    leaves = (out.image_features, out.text_features, out.logit_scale)
    a = torch.autograd.grad(actual, leaves, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(expected, leaves, allow_unused=True)
    assert a[-1] is None and b[-1] is None
    for x, y in zip(a[:2], b[:2]):
        torch.testing.assert_close(x, y)
        assert x.norm() > 0


@pytest.mark.parametrize('arm', ['baseline', 'uniform_top8', 'hardest_real'])
def test_pulse_exact_boundaries_and_native_outside(arm):
    objective = PulseObjective(arm)
    objective.coefficient = .03
    out = output_fixture()
    for step in (0, 99, 100, 199, 200, 1000):
        loss = objective(out, species_ids=None, step=step)
        active = arm != 'baseline' and step in (100, 199)
        assert bool(loss.metrics['pulse_active']) == active
        if not active:
            assert loss.weighted_ot_loss is None
            torch.testing.assert_close(loss.total_loss, native_clip_contrastive_loss(out.logits), rtol=0, atol=0)
        else:
            torch.testing.assert_close(loss.total_loss, loss.clip_loss + .03 * auxiliary_losses(out)[arm])
    assert objective.active_steps == ([] if arm == 'baseline' else [100, 199])


def test_uncalibrated_pulse_fails_closed():
    with pytest.raises(RuntimeError, match='calibration'):
        PulseObjective('uniform_top8')(output_fixture(), species_ids=None, step=100)


def test_coefficients_match_mean_norms_not_mean_ratios():
    rows = [{'native': 2., 'uniform_top8': 10., 'hardest_real': 1.},
            {'native': 6., 'uniform_top8': 30., 'hardest_real': 3.}]
    coefficients = calibration_coefficients(rows, .1)
    assert coefficients == pytest.approx({'uniform_top8': .02, 'hardest_real': .2})
    for arm, alpha in coefficients.items():
        assert alpha * np.mean([r[arm] for r in rows]) == pytest.approx(.4)
    with pytest.raises(ValueError):
        calibration_coefficients([{'native': 1., 'uniform_top8': 0., 'hardest_real': 1.}], .1)


def test_state_digest_tracks_optimizer_rng_and_scalar_tensors():
    state = {'weights': torch.ones(2, 3), 'step': torch.tensor(100.),
             'rng': np.random.get_state(), 'nested': [torch.tensor([1], dtype=torch.int64)]}
    assert state_digest(state) == state_digest(deepcopy(state))
    changed = deepcopy(state)
    changed['step'] += 1
    assert state_digest(state) != state_digest(changed)
    assert state_digest({'b': 2, 'a': 1}) == state_digest({'a': 1, 'b': 2})


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image = torch.nn.Linear(6, 6)
        self.text = torch.nn.Linear(6, 6)
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, images, texts):
        i = F.normalize(self.image(images), dim=-1)
        t = F.normalize(self.text(texts['input_ids']), dim=-1)
        raw = t @ i.T
        return SimpleNamespace(image_features=i, text_features=t, raw_similarity=raw,
            logit_scale=self.scale.exp(), logits=raw * self.scale.exp())


def test_calibration_is_training_only_rng_and_grad_preserving():
    model = TinyModel()
    class Dataset:
        epoch = 2
        def __len__(self):
            return 40
        def __getitem__(self, i):
            return i
    class Loader:
        def __iter__(self):
            raise AssertionError('Must not iterate training loader')
        def collate_fn(self, indices):
            return {'pixel_values': torch.randn(10, 6), 'input_ids': torch.randn(10, 6),
                    'source_indices': torch.tensor(indices), 'captions': [str(i) for i in indices]}
    data = SimpleNamespace(train_dataset=Dataset(), train_loader=Loader())
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = state_digest({'weights': model.state_dict(), 'rng': torch.get_rng_state(),
                           'grads': [p.grad for p in model.parameters()]})
    result = calibrate(model, data, torch.device('cpu'),
        {'training': {'batch_size': 10, 'mixed_precision': 'none'}},
        {'calibration_seed': 123, 'calibration_batches': 4, 'calibration_target_ratio': .1})
    after = state_digest({'weights': model.state_dict(), 'rng': torch.get_rng_state(),
                          'grads': [p.grad for p in model.parameters()]})
    assert before == after and model.training
    indices = [i for r in result['batches'] for i in r['source_indices']]
    assert len(indices) == len(set(indices)) == 40
    assert result['caption_epoch'] == 2


def test_deterministic_replayed_prefix_then_causal_divergence():
    prefix, endpoints = {}, {}
    for arm in ('baseline', 'uniform_top8', 'hardest_real'):
        torch.manual_seed(123)
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        objective = PulseObjective(arm, start=2, stop=4)
        objective.coefficient = .02
        objective.trainable_parameters = tuple(model.parameters())
        for step in range(6):
            optimizer.zero_grad(set_to_none=True)
            out = model(torch.randn(10, 6), {'input_ids': torch.randn(10, 6)})
            losses = objective(out, species_ids=None, step=step)
            losses.total_loss.backward()
            optimizer.step()
            if step == 1:
                prefix[arm] = state_digest({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state()})
        endpoints[arm] = state_digest(model.state_dict())
        assert objective.active_steps == ([] if arm == 'baseline' else [2, 3])
        assert len(objective.gradient_records) == len(objective.active_steps)
    assert len(set(prefix.values())) == 1
    assert len(set(endpoints.values())) == 3
