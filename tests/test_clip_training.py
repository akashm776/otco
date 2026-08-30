import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import model.clip_training as clip_training_module
from model.clip_backend import CLIPEncoderBackend, CLIPSimilarityOutput
from model.clip_training import (
    CLIPTrainingObjective,
    FreshBatchRawCosineOTCO,
    clip_relative_denominator_loss,
    compute_clip_alpha_effective,
    configure_clip_trainable_parameters,
    native_clip_contrastive_loss,
)
from src.clip_training_data import (
    exclude_diagnostic_indices,
    load_diagnostic_holdout,
)
from src.clip_training_eval import chunked_bidirectional_retrieval
from src.clip_train import consider_species_checkpoint, load_training_config


ROOT = Path(__file__).resolve().parents[1]


def ot_config(**overrides):
    config = {
        "enabled": True,
        "loss_type": "historical_absolute_sigmoid",
        "cost_space": "raw_cosine",
        "top_k": 32,
        "ot_eps": 0.049,
        "sinkhorn_iters": 30,
        "update_freq": 1,
        "solver": "historical_sparse_ot",
        "alpha_max": 0.05,
        "warmup_steps": 0,
        "ramp_steps": 0,
        "entropy_gate_enabled": False,
        "entropy_threshold": 3.0,
        "gap_suppress_easy": 0.10,
        "gap_downweight_hard": -0.07,
        "hard_alpha_scale": 0.25,
        "synthetic_logit_gate_enabled": True,
        "synthetic_logit_gate": -4.0,
    }
    config.update(overrides)
    return config


def relative_ot_config(**overrides):
    return ot_config(
        loss_type="clip_relative_denominator",
        synthetic_logit_gate_enabled=False,
        **overrides,
    )


def feature_output(seed=0, count=8, dimension=6, logit_scale=25.0):
    generator = torch.Generator().manual_seed(seed)
    image = F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)
    text = F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)
    raw = text @ image.T
    scale = torch.tensor(logit_scale)
    return CLIPSimilarityOutput(raw, raw * scale, image, text, scale)


def test_native_clip_loss_uses_symmetric_diagonal_targets():
    logits = torch.tensor([[4.0, 1.0, -1.0], [0.0, 3.0, 1.0], [-2.0, 0.0, 5.0]])
    targets = torch.arange(3)
    expected = 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    )
    assert torch.allclose(native_clip_contrastive_loss(logits), expected)


def test_relative_denominator_loss_matches_exact_manual_calculation():
    base_logits = torch.tensor([[2.0, 0.5], [-1.0, 1.5]])
    synthetic_logits = torch.tensor([1.25, 0.75])
    loss, base_lse, augmented_lse = clip_relative_denominator_loss(
        base_logits, synthetic_logits
    )
    expected_base = torch.logsumexp(base_logits, dim=1)
    expected_augmented = torch.logsumexp(
        torch.cat((base_logits, synthetic_logits[:, None]), dim=1), dim=1
    )
    assert torch.equal(base_lse, expected_base)
    assert torch.equal(augmented_lse, expected_augmented)
    assert torch.allclose(
        loss, (expected_augmented - expected_base).mean(), atol=1e-7, rtol=1e-7
    )
    assert loss.item() >= 0


def test_relative_denominator_loss_is_invariant_to_common_logit_shift():
    base_logits = torch.tensor([[3.0, 1.0, -2.0], [0.5, -1.0, 2.5]])
    synthetic_logits = torch.tensor([2.0, 1.25])
    original = clip_relative_denominator_loss(base_logits, synthetic_logits)[0]
    shifted = clip_relative_denominator_loss(
        base_logits - 37.0, synthetic_logits - 37.0
    )[0]
    assert torch.allclose(original, shifted, atol=1e-6, rtol=1e-6)


def test_relative_denominator_loss_tracks_synthetic_competitiveness():
    base_logits = torch.tensor([[2.0, 0.0], [0.5, 1.0]])
    synthetic_logits = torch.tensor([0.25, 0.75])
    lower = clip_relative_denominator_loss(base_logits, synthetic_logits - 2.0)[0]
    middle = clip_relative_denominator_loss(base_logits, synthetic_logits)[0]
    higher = clip_relative_denominator_loss(base_logits, synthetic_logits + 2.0)[0]
    assert lower < middle < higher


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(2.0).log())

    def get_image_features(self, pixel_values):
        return pixel_values

    def get_text_features(self, input_ids, attention_mask=None):
        return input_ids.float()


def test_clip_logit_scale_is_applied_exactly_once():
    backend = CLIPEncoderBackend(FakeCLIP())
    image = F.normalize(torch.tensor([[1.0, 2.0], [2.0, -1.0]]), dim=-1)
    text = F.normalize(torch.tensor([[2.0, 1.0], [-1.0, 2.0]]), dim=-1)
    output = backend(image, {"input_ids": text})
    assert torch.allclose(output.raw_similarity, text @ image.T)
    assert torch.allclose(output.logits, output.raw_similarity * 2.0)


def test_raw_cosine_plan_is_independent_of_clip_logit_scale():
    otco = FreshBatchRawCosineOTCO(ot_config())
    output = feature_output(seed=3)
    otco(
        raw_similarity=output.raw_similarity,
        image_features=output.image_features,
        text_features=output.text_features,
        logit_scale=torch.tensor(5.0),
        species_ids=torch.arange(8),
        step=0,
    )
    first = otco.last_plan.clone()
    otco(
        raw_similarity=output.raw_similarity,
        image_features=output.image_features,
        text_features=output.text_features,
        logit_scale=torch.tensor(100.0),
        species_ids=torch.arange(8),
        step=0,
    )
    assert torch.equal(first, otco.last_plan)


def test_ot_planner_receives_raw_cosine_scores(monkeypatch):
    captured = []
    real_builder = clip_training_module.build_audited_transport_plan

    def capture(scores, *args, **kwargs):
        captured.append(scores.clone())
        return real_builder(scores, *args, **kwargs)

    monkeypatch.setattr(
        clip_training_module, "build_audited_transport_plan", capture
    )
    output = feature_output(seed=13, logit_scale=100.0)
    FreshBatchRawCosineOTCO(ot_config())(
        raw_similarity=output.raw_similarity,
        image_features=output.image_features,
        text_features=output.text_features,
        logit_scale=output.logit_scale,
        species_ids=torch.arange(8),
        step=0,
    )
    assert torch.equal(captured[0], output.raw_similarity.float())


def test_update_freq_one_recomputes_the_current_live_batch_plan():
    otco = FreshBatchRawCosineOTCO(ot_config())
    first = feature_output(seed=4)
    second = feature_output(seed=5)
    for output in (first, second):
        otco(
            raw_similarity=output.raw_similarity,
            image_features=output.image_features,
            text_features=output.text_features,
            logit_scale=output.logit_scale,
            species_ids=torch.arange(8),
            step=0,
        )
        if output is first:
            first_plan = otco.last_plan.clone()
    assert otco.plan_build_count == 2
    assert not torch.equal(first_plan, otco.last_plan)


def test_entropy_is_observational_when_entropy_gate_is_disabled():
    alpha, state = compute_clip_alpha_effective(
        0.05,
        coupling_entropy=1000.0,
        positive_selected_gap=0.0,
        entropy_gate_enabled=False,
        entropy_threshold=3.0,
        gap_suppress_easy=0.10,
        gap_downweight_hard=-0.07,
        hard_alpha_scale=0.25,
    )
    assert alpha == pytest.approx(0.05)
    assert state == 0


@pytest.mark.parametrize(
    ("gap", "expected_alpha", "expected_state"),
    [(0.11, 0.0, 1), (-0.08, 0.0125, 2), (0.0, 0.05, 0)],
)
def test_gap_thresholds_control_effective_alpha(gap, expected_alpha, expected_state):
    alpha, state = compute_clip_alpha_effective(
        0.05,
        coupling_entropy=3.5,
        positive_selected_gap=gap,
        entropy_gate_enabled=False,
        entropy_threshold=3.0,
        gap_suppress_easy=0.10,
        gap_downweight_hard=-0.07,
        hard_alpha_scale=0.25,
    )
    assert alpha == pytest.approx(expected_alpha)
    assert state == expected_state


def load_pair_configs():
    paths = [
        ROOT / "configs/hf_cub200_clip_vit_b32_baseline.yaml",
        ROOT / "configs/hf_cub200_clip_vit_b32_otco_rawcos_gap_gate.yaml",
    ]
    return [yaml.safe_load(path.read_text()) for path in paths]


def test_baseline_and_otco_configs_share_trainable_policy_and_controls():
    baseline, treatment = load_pair_configs()
    assert baseline["model"]["trainable_policy"] == treatment["model"]["trainable_policy"]
    normalized_baseline = copy.deepcopy(baseline)
    normalized_treatment = copy.deepcopy(treatment)
    normalized_baseline["experiment"]["name"] = "arm"
    normalized_treatment["experiment"]["name"] = "arm"
    normalized_baseline["ot"]["enabled"] = True
    assert normalized_baseline == normalized_treatment
    load_training_config(
        ROOT / "configs/hf_cub200_clip_vit_b32_baseline.yaml"
    )
    load_training_config(
        ROOT / "configs/hf_cub200_clip_vit_b32_otco_rawcos_gap_gate.yaml"
    )


def test_baseline_and_relative_otco_configs_differ_only_by_treatment():
    baseline = yaml.safe_load(
        (ROOT / "configs/hf_cub200_clip_vit_b32_baseline.yaml").read_text()
    )
    treatment = yaml.safe_load(
        (
            ROOT
            / "configs/hf_cub200_clip_vit_b32_otco_relative_denominator.yaml"
        ).read_text()
    )
    normalized_baseline = copy.deepcopy(baseline)
    normalized_treatment = copy.deepcopy(treatment)
    normalized_baseline["experiment"]["name"] = "arm"
    normalized_treatment["experiment"]["name"] = "arm"
    normalized_baseline["ot"]["enabled"] = True
    normalized_baseline["ot"]["loss_type"] = "clip_relative_denominator"
    normalized_baseline["ot"]["synthetic_logit_gate_enabled"] = False
    assert normalized_baseline == normalized_treatment
    load_training_config(
        ROOT / "configs/hf_cub200_clip_vit_b32_otco_relative_denominator.yaml"
    )


def test_exact_diagnostic_holdout_is_excluded_from_training():
    path = ROOT / "configs/cub200_clip_diagnostic_holdout_indices.json"
    holdout = load_diagnostic_holdout(path, original_count=10_000)
    training = exclude_diagnostic_indices(10_000, holdout)
    assert len(holdout) == 1024
    assert set(training).isdisjoint(holdout)
    assert len(training) + len(holdout) == 10_000


def test_baseline_objective_has_no_ot_path_or_term():
    objective = CLIPTrainingObjective(ot_config(enabled=False))
    result = objective(feature_output(), species_ids=torch.arange(8), step=0)
    assert objective.otco is None
    assert result.ot_loss is None
    assert result.weighted_ot_loss is None
    assert result.total_loss is result.clip_loss


def test_otco_total_reduces_exactly_to_baseline_when_alpha_is_zero():
    output = feature_output(seed=8)
    baseline = CLIPTrainingObjective(ot_config(enabled=False))(
        output, species_ids=torch.arange(8), step=0
    )
    treatment = CLIPTrainingObjective(
        ot_config(enabled=True, warmup_steps=100)
    )(output, species_ids=torch.arange(8), step=0)
    assert treatment.metrics["alpha_effective"] == 0
    assert torch.equal(treatment.total_loss, baseline.total_loss)


def test_relative_otco_total_reduces_exactly_to_baseline_when_alpha_is_zero():
    output = feature_output(seed=18)
    baseline = CLIPTrainingObjective(ot_config(enabled=False))(
        output, species_ids=torch.arange(8), step=0
    )
    treatment = CLIPTrainingObjective(
        relative_ot_config(enabled=True, warmup_steps=100)
    )(output, species_ids=torch.arange(8), step=0)
    assert treatment.metrics["alpha_effective"] == 0
    assert torch.equal(treatment.total_loss, baseline.total_loss)


def test_transport_mass_diagnostics_are_emitted():
    result = CLIPTrainingObjective(ot_config())(
        feature_output(seed=9), species_ids=torch.arange(8), step=0
    )
    for key in (
        "transport_total_mass",
        "transport_max_row_marginal_error",
        "transport_max_column_marginal_error",
        "transport_fraction_mass_removed",
    ):
        assert key in result.metrics
        assert math_is_finite(result.metrics[key])


def math_is_finite(value):
    return not (value != value or value in {float("inf"), float("-inf")})


class FakeTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


class FakePolicyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_projection = nn.Linear(4, 3, bias=False)
        self.text_projection = nn.Linear(4, 3, bias=False)
        self.vision_model = FakeTower()
        self.text_model = FakeTower()
        self.logit_scale = nn.Parameter(torch.tensor(1.0))


def test_partial_unfreezing_selects_only_the_declared_policy():
    model = FakePolicyCLIP()
    inventory = configure_clip_trainable_parameters(
        model, "projections_last_blocks_logit_scale"
    )
    names = {item["name"] for item in inventory["parameters"]}
    assert "vision_model.encoder.layers.0.weight" not in names
    assert "text_model.encoder.layers.0.weight" not in names
    assert "vision_model.encoder.layers.1.weight" in names
    assert "text_model.encoder.layers.1.weight" in names
    assert "visual_projection.weight" in names
    assert "text_projection.weight" in names
    assert "logit_scale" in names


def test_chunked_retrieval_handles_multiple_captions_per_image():
    images = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    texts = F.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        dim=-1,
    )
    metrics = chunked_bidirectional_retrieval(
        texts, images, [0, 0, 1, 1], chunk_size=1
    )
    assert metrics["text_to_image"]["r_at_1"] == pytest.approx(100.0)
    assert metrics["image_to_text"]["r_at_1"] == pytest.approx(100.0)


def test_ot_loss_detaches_logit_scale_but_native_clip_loss_trains_it():
    generator = torch.Generator().manual_seed(21)
    image = F.normalize(
        torch.randn(8, 6, generator=generator, requires_grad=True), dim=-1
    )
    text = F.normalize(
        torch.randn(8, 6, generator=generator, requires_grad=True), dim=-1
    )
    raw = text @ image.T
    logit_scale = torch.tensor(25.0, requires_grad=True)
    output = CLIPSimilarityOutput(
        raw_similarity=raw,
        logits=raw * logit_scale,
        image_features=image,
        text_features=text,
        logit_scale=logit_scale,
    )
    result = CLIPTrainingObjective(ot_config())(
        output, species_ids=torch.arange(8), step=0
    )
    native_gradient = torch.autograd.grad(
        result.clip_loss, logit_scale, retain_graph=True
    )[0]
    ot_gradient = torch.autograd.grad(
        result.ot_loss, logit_scale, allow_unused=True
    )[0]
    assert native_gradient.abs().item() > 0
    assert ot_gradient is None


def test_relative_ot_loss_detaches_scale_but_keeps_representation_gradients_live():
    generator = torch.Generator().manual_seed(31)
    image_source = torch.randn(8, 6, generator=generator, requires_grad=True)
    text_source = torch.randn(8, 6, generator=generator, requires_grad=True)
    image = F.normalize(image_source, dim=-1)
    text = F.normalize(text_source, dim=-1)
    raw = text @ image.T
    logit_scale = torch.tensor(25.0, requires_grad=True)
    output = CLIPSimilarityOutput(
        raw_similarity=raw,
        logits=raw * logit_scale,
        image_features=image,
        text_features=text,
        logit_scale=logit_scale,
    )
    result = CLIPTrainingObjective(relative_ot_config())(
        output, species_ids=torch.arange(8), step=0
    )
    scale_gradient = torch.autograd.grad(
        result.ot_loss, logit_scale, retain_graph=True, allow_unused=True
    )[0]
    image_gradient, text_gradient = torch.autograd.grad(
        result.ot_loss, (image_source, text_source)
    )
    assert scale_gradient is None
    assert image_gradient.norm().item() > 0
    assert text_gradient.norm().item() > 0
    assert result.metrics["ot_relative_loss"] == pytest.approx(
        result.ot_loss.detach().item()
    )
    assert result.metrics["weighted_ot_relative_loss"] == pytest.approx(
        result.weighted_ot_loss.detach().item()
    )


def test_relative_loss_bypasses_absolute_gate_while_historical_mode_retains_it():
    output = feature_output(seed=37)
    species = torch.arange(8)
    relative_low = CLIPTrainingObjective(
        relative_ot_config(synthetic_logit_gate=-1.0e9)
    )(output, species_ids=species, step=0)
    relative_high = CLIPTrainingObjective(
        relative_ot_config(synthetic_logit_gate=1.0e9)
    )(output, species_ids=species, step=0)
    assert torch.equal(relative_low.ot_loss, relative_high.ot_loss)
    assert relative_low.metrics["synthetic_gate_active_fraction"] == 1.0
    assert relative_high.metrics["synthetic_gate_active_fraction"] == 1.0

    historical_active = CLIPTrainingObjective(
        ot_config(synthetic_logit_gate=-1.0e9)
    )(output, species_ids=species, step=0)
    historical_suppressed = CLIPTrainingObjective(
        ot_config(synthetic_logit_gate=1.0e9)
    )(output, species_ids=species, step=0)
    assert historical_active.metrics["synthetic_gate_active_fraction"] == 1.0
    assert historical_active.ot_loss.item() > 0
    assert historical_suppressed.metrics["synthetic_gate_active_fraction"] == 0.0
    assert historical_suppressed.ot_loss.item() == 0.0


def test_epoch_zero_can_remain_the_best_checkpoint():
    epoch_zero = {
        "epoch": 0,
        "evaluation": {"species": {"top_1_accuracy": 0.5352}},
    }
    best_score, best_epoch, should_save = consider_species_checkpoint(epoch_zero)
    assert should_save
    assert best_epoch == 0
    assert best_score == pytest.approx(0.5352)

    worse_epoch = {
        "epoch": 1,
        "evaluation": {"species": {"top_1_accuracy": 0.50}},
    }
    best_score, best_epoch, should_save = consider_species_checkpoint(
        worse_epoch, best_score, best_epoch
    )
    assert not should_save
    assert best_epoch == 0
