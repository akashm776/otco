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
    hardest_real_negative_indices,
    native_clip_contrastive_loss,
    scheduled_ot_alpha,
    synthetic_barycentric_weights,
)
from src.clip_training_data import (
    exclude_diagnostic_indices,
    load_diagnostic_holdout,
)
from src.clip_training_eval import chunked_bidirectional_retrieval
from src.clip_train import (
    add_projection_gradient_aliases,
    consider_species_checkpoint,
    diagnostic_gradient_norms,
    load_training_config,
)


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


def uniform_relative_config(**overrides):
    return relative_ot_config(synthetic_weighting="uniform_topk", **overrides)


def hardest_real_relative_config(**overrides):
    return uniform_relative_config(
        extra_negative_mode="hardest_real", top_k=8, **overrides
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


def test_hardest_real_selection_excludes_positive_and_uses_expected_index():
    similarity = torch.tensor(
        [
            [10.0, 0.2, 0.8, 0.1],
            [0.4, 10.0, 0.3, 0.9],
            [0.7, 0.6, 10.0, 0.5],
            [0.2, 0.8, 0.4, 10.0],
        ]
    )
    selected = hardest_real_negative_indices(similarity)
    assert torch.equal(selected, torch.tensor([2, 3, 0, 1]))
    assert torch.all(selected != torch.arange(4))


def test_hardest_real_selection_ties_use_first_pytorch_argmax_index():
    similarity = torch.tensor(
        [
            [9.0, 0.8, 0.8],
            [0.5, 9.0, 0.5],
            [0.7, 0.7, 9.0],
        ]
    )
    assert torch.equal(
        hardest_real_negative_indices(similarity), torch.tensor([1, 0, 0])
    )


def test_hardest_real_relative_loss_matches_duplicate_logit_calculation():
    base_logits = torch.tensor(
        [[5.0, 2.0, 1.0], [0.5, 4.0, 2.5], [3.0, 1.0, 6.0]]
    )
    selected = hardest_real_negative_indices(base_logits)
    row = torch.arange(3)
    hardest_logits = base_logits[row, selected]
    actual = clip_relative_denominator_loss(base_logits, hardest_logits)[0]
    expected = torch.stack(
        [
            torch.logsumexp(
                torch.cat((base_logits[i], hardest_logits[i : i + 1])), dim=0
            )
            - torch.logsumexp(base_logits[i], dim=0)
            for i in range(3)
        ]
    ).mean()
    assert torch.allclose(actual, expected)


def test_uniform_weights_use_exact_existing_support_and_exclude_positive():
    support = torch.tensor(
        [
            [False, True, True, False],
            [True, False, True, True],
        ]
    )
    plan = support.float() * torch.tensor(
        [[0.0, 0.7, 0.3, 0.0], [0.2, 0.0, 0.3, 0.5]]
    )
    weights = synthetic_barycentric_weights(plan, support, mode="uniform_topk")

    assert torch.equal(weights != 0, support)
    assert torch.equal(weights[0], torch.tensor([0.0, 0.5, 0.5, 0.0]))
    assert torch.allclose(
        weights[1], torch.tensor([1 / 3, 0.0, 1 / 3, 1 / 3])
    )
    assert torch.allclose(weights.sum(1), torch.ones(2))
    assert weights[0, 0] == 0
    assert weights[1, 1] == 0


def test_uniform_barycentric_normalization_can_exceed_both_contributors():
    query = torch.tensor([1.0, 0.0])
    images = torch.tensor([[0.8, 0.6], [0.8, -0.6]])
    support = torch.ones((1, 2), dtype=torch.bool)
    weights = synthetic_barycentric_weights(
        torch.ones((1, 2)), support, mode="uniform_topk"
    )
    synthetic = F.normalize(weights @ images, dim=-1)[0]

    assert torch.allclose(synthetic, query)
    assert query @ synthetic > (images @ query).max()


def test_uniform_top8_support_weights_and_synthetic_regression():
    support = torch.zeros((1, 64), dtype=torch.bool)
    support[0, torch.tensor([1, 7, 12, 19, 23, 31, 44, 58])] = True
    plan = support.to(torch.float32)
    weights = synthetic_barycentric_weights(plan, support, mode="uniform_topk")

    assert int((weights != 0).sum()) == 8
    assert torch.equal(weights != 0, support)
    assert torch.all(weights[support] == torch.tensor(1.0 / 8))
    assert torch.all(weights[~support] == 0)
    assert torch.allclose(weights.sum(1), torch.ones(1))

    images = torch.zeros((64, 2), dtype=torch.float32)
    images[1] = torch.tensor([0.8, 0.6])
    images[7] = torch.tensor([0.8, -0.6])
    images[12] = torch.tensor([1.0, 0.0])
    images[19] = torch.tensor([0.9, 0.1])
    images[23] = torch.tensor([0.9, -0.1])
    images[31] = torch.tensor([0.7, 0.2])
    images[44] = torch.tensor([0.7, -0.2])
    images[58] = torch.tensor([0.8, 0.0])
    expected = F.normalize(weights @ images, dim=-1)
    assert torch.allclose(expected, F.normalize(images[support[0]].mean(0), dim=0))


def test_ot_weighting_expression_is_unchanged_and_matches_uniform_when_equal():
    support = torch.tensor([[False, True, True]])
    nonuniform_plan = torch.tensor([[0.0, 0.2, 0.8]])
    ot_weights = synthetic_barycentric_weights(
        nonuniform_plan, support, mode="ot"
    )
    historical = nonuniform_plan / nonuniform_plan.sum(
        1, keepdim=True
    ).clamp_min(1e-8)
    assert torch.equal(ot_weights, historical)

    uniform_plan = torch.tensor([[0.0, 0.5, 0.5]])
    equal_ot = synthetic_barycentric_weights(uniform_plan, support, mode="ot")
    equal_uniform = synthetic_barycentric_weights(
        uniform_plan, support, mode="uniform_topk"
    )
    images = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.8, -0.6]])
    assert torch.allclose(equal_ot, equal_uniform)
    assert torch.allclose(
        F.normalize(equal_ot @ images, dim=-1),
        F.normalize(equal_uniform @ images, dim=-1),
    )


def test_relative_loss_is_identical_for_identical_synthetic_logits():
    base_logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
    synthetic_logits = torch.tensor([1.25, 0.75])

    ot_result = clip_relative_denominator_loss(base_logits, synthetic_logits)
    uniform_result = clip_relative_denominator_loss(base_logits, synthetic_logits)
    for ot_value, uniform_value in zip(ot_result, uniform_result):
        assert torch.equal(ot_value, uniform_value)


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


def test_native_strength_differs_from_weak_relative_only_by_name_and_alpha():
    weak = yaml.safe_load(
        (
            ROOT
            / "configs/hf_cub200_clip_vit_b32_otco_relative_denominator.yaml"
        ).read_text()
    )
    native = yaml.safe_load(
        (
            ROOT
            / "configs/hf_cub200_clip_vit_b32_otco_relative_native_strength.yaml"
        ).read_text()
    )
    normalized_weak = copy.deepcopy(weak)
    normalized_native = copy.deepcopy(native)
    normalized_weak["experiment"]["name"] = "arm"
    normalized_native["experiment"]["name"] = "arm"
    normalized_weak["ot"]["alpha_max"] = 0.5
    normalized_weak["ot"]["synthetic_weighting"] = "ot"
    assert normalized_weak == normalized_native
    load_training_config(
        ROOT
        / "configs/hf_cub200_clip_vit_b32_otco_relative_native_strength.yaml"
    )


def test_uniform_native_strength_differs_from_ot_only_by_name_and_weighting():
    ot = yaml.safe_load(
        (
            ROOT
            / "configs/hf_cub200_clip_vit_b32_otco_relative_native_strength.yaml"
        ).read_text()
    )
    uniform = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "hf_cub200_clip_vit_b32_uniform_barycentric_relative_native_strength.yaml"
        ).read_text()
    )
    normalized_ot = copy.deepcopy(ot)
    normalized_uniform = copy.deepcopy(uniform)
    normalized_ot["experiment"]["name"] = "arm"
    normalized_uniform["experiment"]["name"] = "arm"
    normalized_ot["ot"]["synthetic_weighting"] = "uniform_topk"
    assert normalized_ot == normalized_uniform
    load_training_config(
        ROOT
        / "configs/"
        "hf_cub200_clip_vit_b32_uniform_barycentric_relative_native_strength.yaml"
    )


def test_uniform_top8_native_strength_differs_from_top32_only_by_name_and_top_k():
    top32 = yaml.safe_load(
        (ROOT / "configs/hf_cub200_clip_vit_b32_uniform_barycentric_relative_native_strength.yaml").read_text()
    )
    top8 = yaml.safe_load(
        (ROOT / "configs/hf_cub200_clip_vit_b32_uniform_top8_relative_native_strength.yaml").read_text()
    )
    normalized_top32 = copy.deepcopy(top32)
    normalized_top8 = copy.deepcopy(top8)
    normalized_top32["experiment"]["name"] = "arm"
    normalized_top8["experiment"]["name"] = "arm"
    normalized_top32["ot"]["top_k"] = 8
    assert normalized_top32 == normalized_top8
    assert top32["ot"]["top_k"] == 32
    assert top8["ot"]["top_k"] == 8
    load_training_config(
        ROOT / "configs/hf_cub200_clip_vit_b32_uniform_top8_relative_native_strength.yaml"
    )


def test_pressure_matched_u8_differs_from_native_u8_only_by_name_and_alpha():
    native = yaml.safe_load(
        (
            ROOT
            / "configs/hf_cub200_clip_vit_b32_uniform_top8_relative_native_strength.yaml"
        ).read_text()
    )
    matched = yaml.safe_load(
        (
            ROOT
            / "configs/hf_cub200_clip_vit_b32_uniform_top8_relative_pressure_matched.yaml"
        ).read_text()
    )
    normalized_native = copy.deepcopy(native)
    normalized_matched = copy.deepcopy(matched)
    normalized_native["experiment"]["name"] = "arm"
    normalized_matched["experiment"]["name"] = "arm"
    normalized_native["ot"]["alpha_max"] = 0.134
    assert normalized_native == normalized_matched
    assert native["ot"]["alpha_max"] == 0.5
    assert matched["ot"]["alpha_max"] == 0.134
    load_training_config(
        ROOT
        / "configs/hf_cub200_clip_vit_b32_uniform_top8_relative_pressure_matched.yaml"
    )


def test_pressure_matched_u8_uses_existing_alpha_schedule():
    config = load_training_config(
        ROOT
        / "configs/hf_cub200_clip_vit_b32_uniform_top8_relative_pressure_matched.yaml"
    )
    ot = config["ot"]
    schedule = lambda step: scheduled_ot_alpha(
        step,
        alpha_max=ot["alpha_max"],
        warmup_steps=ot["warmup_steps"],
        ramp_steps=ot["ramp_steps"],
    )
    assert schedule(999) == 0.0
    assert schedule(1000) == 0.0
    assert schedule(2000) == pytest.approx(0.134)
    assert schedule(2001) == pytest.approx(0.134)


def test_hardest_real_config_differs_from_uniform_top8_only_by_extra_mode():
    top8 = yaml.safe_load(
        (
            ROOT
            / "configs/hf_cub200_clip_vit_b32_uniform_top8_relative_native_strength.yaml"
        ).read_text()
    )
    hardest = yaml.safe_load(
        (
            ROOT
            / "configs/hf_cub200_clip_vit_b32_hardest_real_relative_native_strength.yaml"
        ).read_text()
    )
    normalized_top8 = copy.deepcopy(top8)
    normalized_hardest = copy.deepcopy(hardest)
    normalized_top8["experiment"]["name"] = "arm"
    normalized_hardest["experiment"]["name"] = "arm"
    normalized_top8["ot"]["extra_negative_mode"] = "hardest_real"
    assert normalized_top8 == normalized_hardest
    load_training_config(
        ROOT
        / "configs/hf_cub200_clip_vit_b32_hardest_real_relative_native_strength.yaml"
    )


@pytest.mark.parametrize(
    ("config_name", "incorrect_weighting"),
    [
        (
            "hf_cub200_clip_vit_b32_otco_relative_native_strength.yaml",
            "uniform_topk",
        ),
        (
            "hf_cub200_clip_vit_b32_uniform_barycentric_relative_native_strength.yaml",
            "ot",
        ),
    ],
)
def test_experiment_name_strictly_validates_weighting_mode(
    tmp_path, config_name, incorrect_weighting
):
    config = yaml.safe_load((ROOT / "configs" / config_name).read_text())
    config["ot"]["synthetic_weighting"] = incorrect_weighting
    path = tmp_path / config_name
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="synthetic weighting mode"):
        load_training_config(path)


def test_projection_gradient_ratio_is_weighted_ot_norm_over_clip_norm():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    loss_output = type(
        "LossOutput",
        (),
        {
            "clip_loss": (2.0 * parameter).sum(),
            "weighted_ot_loss": (0.5 * parameter).sum(),
        },
    )()
    metrics = diagnostic_gradient_norms(loss_output, [parameter])
    assert metrics["projection_gradient_norm_clip"] == pytest.approx(2.0**1.5)
    assert metrics["projection_gradient_norm_weighted_ot"] == pytest.approx(
        0.5 * 2.0**0.5
    )
    assert metrics["projection_gradient_ratio_weighted_ot_to_clip"] == (
        pytest.approx(0.25)
    )


def test_hardest_real_gradient_alias_is_not_mislabeled_as_synthetic():
    generic = {
        "projection_gradient_norm_weighted_ot": 2.0,
        "projection_gradient_ratio_weighted_ot_to_clip": 0.25,
    }
    hardest = add_projection_gradient_aliases(
        {
            "synthetic_weighting_mode": "uniform_topk",
            "extra_negative_mode": "hardest_real",
        },
        dict(generic),
    )
    assert hardest["projection_gradient_norm_weighted_hardest_real"] == 2.0
    assert hardest[
        "projection_gradient_ratio_weighted_hardest_real_to_clip"
    ] == 0.25
    assert "projection_gradient_norm_weighted_synthetic" not in hardest
    assert "projection_gradient_ratio_weighted_synthetic_to_clip" not in hardest

    barycentric = add_projection_gradient_aliases(
        {"synthetic_weighting_mode": "uniform_topk"}, dict(generic)
    )
    assert barycentric["projection_gradient_norm_weighted_synthetic"] == 2.0
    assert "projection_gradient_norm_weighted_hardest_real" not in barycentric


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


@pytest.mark.parametrize("weighting", ["ot", "uniform_topk"])
def test_relative_alpha_zero_equivalence_for_both_weighting_modes(weighting):
    output = feature_output(seed=181)
    baseline = CLIPTrainingObjective(ot_config(enabled=False))(
        output, species_ids=torch.arange(8), step=0
    )
    treatment = CLIPTrainingObjective(
        relative_ot_config(
            synthetic_weighting=weighting,
            enabled=True,
            warmup_steps=100,
        )
    )(output, species_ids=torch.arange(8), step=0)

    assert treatment.metrics["alpha_effective"] == 0
    assert torch.equal(treatment.total_loss, baseline.total_loss)


def test_pressure_matched_u8_alpha_zero_is_exact_native_clip():
    config = load_training_config(
        ROOT
        / "configs/hf_cub200_clip_vit_b32_uniform_top8_relative_pressure_matched.yaml"
    )
    output = feature_output(seed=184)
    baseline = CLIPTrainingObjective(ot_config(enabled=False))(
        output, species_ids=torch.arange(8), step=0
    )
    treatment = CLIPTrainingObjective(config["ot"])(
        output, species_ids=torch.arange(8), step=999
    )
    assert treatment.metrics["alpha_effective"] == 0
    assert treatment.weighted_ot_loss.item() == 0
    assert torch.equal(treatment.total_loss, treatment.clip_loss)
    assert torch.equal(treatment.total_loss, baseline.total_loss)


def test_uniform_treatment_preserves_ot_plan_gap_and_gate_semantics():
    output = feature_output(seed=182)
    species = torch.arange(8)
    shared = {
        "alpha_max": 0.5,
        "warmup_steps": 0,
        "ramp_steps": 0,
    }
    ot_objective = CLIPTrainingObjective(relative_ot_config(**shared))
    uniform_objective = CLIPTrainingObjective(uniform_relative_config(**shared))
    ot_result = ot_objective(output, species_ids=species, step=2000)
    uniform_result = uniform_objective(output, species_ids=species, step=2000)

    assert torch.equal(ot_objective.otco.last_plan, uniform_objective.otco.last_plan)
    for key in (
        "positive_selected_gap",
        "coupling_entropy",
        "normalized_coupling_entropy",
        "coupling_peak_mass",
        "selected_rank",
        "same_species_selected_rate",
        "alpha_scheduled",
        "alpha_effective",
        "gate_state_id",
    ):
        assert uniform_result.metrics[key] == pytest.approx(ot_result.metrics[key])


def test_hardest_real_preserves_uniform_top8_ot_plan_gap_and_gate_semantics():
    output = feature_output(seed=183)
    species = torch.arange(8)
    shared = {"alpha_max": 0.5, "warmup_steps": 0, "ramp_steps": 0, "top_k": 8}
    uniform_objective = CLIPTrainingObjective(uniform_relative_config(**shared))
    hardest_objective = CLIPTrainingObjective(
        hardest_real_relative_config(
            alpha_max=0.5, warmup_steps=0, ramp_steps=0
        )
    )
    uniform_result = uniform_objective(output, species_ids=species, step=2000)
    hardest_result = hardest_objective(output, species_ids=species, step=2000)

    assert torch.equal(
        uniform_objective.otco.last_plan, hardest_objective.otco.last_plan
    )
    for key in (
        "positive_selected_gap",
        "coupling_entropy",
        "normalized_coupling_entropy",
        "coupling_peak_mass",
        "selected_rank",
        "same_species_selected_rate",
        "alpha_scheduled",
        "alpha_effective",
        "gate_state_id",
    ):
        assert hardest_result.metrics[key] == pytest.approx(
            uniform_result.metrics[key]
        )


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


def test_hardest_real_branch_reuses_relative_loss_and_keeps_live_gradients():
    generator = torch.Generator().manual_seed(312)
    image_source = torch.randn(8, 6, generator=generator, requires_grad=True)
    text_source = torch.randn(8, 6, generator=generator, requires_grad=True)
    image = F.normalize(image_source, dim=-1)
    text = F.normalize(text_source, dim=-1)
    raw = text @ image.T
    scale = torch.tensor(25.0, requires_grad=True)
    output = CLIPSimilarityOutput(raw, raw * scale, image, text, scale)
    result = CLIPTrainingObjective(hardest_real_relative_config())(
        output, species_ids=torch.arange(8), step=0
    )

    row = torch.arange(8)
    selected = hardest_real_negative_indices(raw)
    auxiliary_base = raw * scale.detach()
    expected = clip_relative_denominator_loss(
        auxiliary_base, auxiliary_base[row, selected]
    )[0]
    assert torch.allclose(result.ot_loss, expected)

    scale_gradient = torch.autograd.grad(
        result.ot_loss, scale, retain_graph=True, allow_unused=True
    )[0]
    image_gradient, text_gradient = torch.autograd.grad(
        result.ot_loss, (image_source, text_source)
    )
    assert scale_gradient is None
    assert image_gradient.norm().item() > 0
    assert text_gradient.norm().item() > 0
    assert result.metrics["extra_negative_mode"] == "hardest_real"
    assert result.metrics["hardest_real_rank"] == 1.0
    assert result.metrics["hardest_real_relative_loss"] == pytest.approx(
        expected.detach().item()
    )
    assert result.metrics[
        "weighted_hardest_real_relative_loss"
    ] == pytest.approx(result.weighted_ot_loss.detach().item())
    for key in (
        "counterfactual_uniform_top8_synthetic_similarity",
        "hardest_real_minus_uniform_top8_similarity",
        "fraction_hardest_real_gt_uniform_top8",
        "uniform_top8_synthetic_vs_hardest_real_delta",
        "fraction_uniform_top8_gt_hardest_real",
        "hardest_real_similarity",
        "hardest_real_logit",
        "hardest_real_minus_positive_similarity",
        "fraction_hardest_real_gt_positive",
        "hardest_real_logit_minus_base_lse_mean",
    ):
        assert key in result.metrics


def test_hardest_real_alpha_zero_reduces_exactly_to_native_clip():
    output = feature_output(seed=313)
    baseline = CLIPTrainingObjective(ot_config(enabled=False))(
        output, species_ids=torch.arange(8), step=0
    )
    treatment = CLIPTrainingObjective(
        hardest_real_relative_config(warmup_steps=100)
    )(output, species_ids=torch.arange(8), step=0)
    assert treatment.metrics["alpha_effective"] == 0
    assert torch.equal(treatment.total_loss, baseline.total_loss)


def test_uniform_relative_loss_keeps_features_live_and_emits_counterfactuals():
    generator = torch.Generator().manual_seed(311)
    image_source = torch.randn(8, 6, generator=generator, requires_grad=True)
    text_source = torch.randn(8, 6, generator=generator, requires_grad=True)
    image = F.normalize(image_source, dim=-1)
    text = F.normalize(text_source, dim=-1)
    raw = text @ image.T
    scale = torch.tensor(25.0, requires_grad=True)
    output = CLIPSimilarityOutput(
        raw_similarity=raw,
        logits=raw * scale,
        image_features=image,
        text_features=text,
        logit_scale=scale,
    )
    result = CLIPTrainingObjective(uniform_relative_config())(
        output, species_ids=torch.arange(8), step=0
    )
    image_gradient, text_gradient = torch.autograd.grad(
        result.ot_loss, (image_source, text_source)
    )

    assert image_gradient.norm().item() > 0
    assert text_gradient.norm().item() > 0
    assert result.metrics["synthetic_weighting_mode"] == "uniform_topk"
    for key in (
        "active_uniform_synthetic_similarity",
        "counterfactual_ot_synthetic_similarity",
        "uniform_minus_ot_synthetic_similarity",
        "fraction_uniform_gt_ot_synthetic",
        "uniform_synthetic_vs_hardest_real_delta",
        "ot_synthetic_vs_hardest_real_delta",
        "fraction_uniform_synthetic_gt_hardest_real",
        "fraction_ot_synthetic_gt_hardest_real",
        "weighted_synthetic_relative_loss",
    ):
        assert key in result.metrics


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
