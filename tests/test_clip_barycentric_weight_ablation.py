"""Focused tests for the frozen OT-vs-uniform barycentric ablation."""

import torch
import torch.nn.functional as F

from src.clip_barycentric_weight_metrics import (
    compute_barycentric_weight_ablation,
    construct_normalized_barycenters,
    paired_hardness_counts,
    summarize_barycentric_weight_ablation,
    uniform_weights_from_support,
)


def _three_point_fixture(plan=None):
    image_features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.8, -0.6]], dtype=torch.float32
    )
    text_features = image_features.clone()
    positive_mask = torch.eye(3, dtype=torch.bool)
    support_mask = ~positive_mask
    if plan is None:
        plan = torch.tensor(
            [[0.0, 0.2, 0.3], [0.4, 0.0, 0.1], [0.25, 0.25, 0.0]]
        )
    return compute_barycentric_weight_ablation(
        image_features=image_features,
        text_features=text_features,
        plan=plan,
        support_mask=support_mask,
        positive_mask=positive_mask,
    )


def test_ot_and_uniform_use_identical_support():
    diagnostics = _three_point_fixture()

    assert torch.equal(
        diagnostics["support_mask"], diagnostics["ot_support_mask"]
    )
    assert torch.equal(
        diagnostics["support_mask"], diagnostics["uniform_support_mask"]
    )
    assert torch.all(diagnostics["ot_weights"][~diagnostics["support_mask"]] == 0)
    assert torch.all(
        diagnostics["uniform_weights"][~diagnostics["support_mask"]] == 0
    )


def test_uniform_weighting_is_one_over_support_size_and_zero_elsewhere():
    support = torch.tensor(
        [[True, False, True, False], [False, True, True, True]]
    )
    weights = uniform_weights_from_support(support)

    assert torch.equal(weights[0], torch.tensor([0.5, 0.0, 0.5, 0.0]))
    assert torch.equal(
        weights[1], torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3])
    )
    assert torch.all(weights[~support] == 0)


def test_synthetic_construction_matches_normalized_weighted_sum():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    weights = torch.tensor([[0.75, 0.25, 0.0], [0.0, 0.5, 0.5]])

    actual = construct_normalized_barycenters(weights, features)
    expected = F.normalize(weights @ F.normalize(features, dim=-1), dim=-1)

    assert torch.allclose(actual, expected)


def test_uniform_barycenter_can_be_harder_than_every_real_vector():
    query = torch.tensor([1.0, 0.0])
    contributors = torch.tensor([[0.8, 0.6], [0.8, -0.6]])
    weights = torch.tensor([[0.5, 0.5]])

    synthetic = construct_normalized_barycenters(weights, contributors)[0]
    synthetic_similarity = query @ synthetic
    hardest_real_similarity = (contributors @ query).max()

    assert torch.isclose(synthetic, query).all()
    assert torch.isclose(synthetic_similarity, torch.tensor(1.0))
    assert torch.isclose(hardest_real_similarity, torch.tensor(0.8))
    assert synthetic_similarity > hardest_real_similarity


def test_ot_and_uniform_synthetics_match_when_ot_weights_are_uniform():
    uniform_plan = torch.tensor(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    )
    diagnostics = _three_point_fixture(plan=uniform_plan)

    assert torch.allclose(
        diagnostics["ot_weights"], diagnostics["uniform_weights"]
    )
    assert torch.allclose(
        diagnostics["ot_synthetic_features"],
        diagnostics["uniform_synthetic_features"],
    )
    assert torch.allclose(
        diagnostics["ot_minus_uniform_synthetic_similarity"],
        torch.zeros(3),
        atol=1e-7,
    )


def test_paired_metrics_include_disagreement_table_and_exact_deltas():
    counts = paired_hardness_counts(
        torch.tensor([True, True, False, False]),
        torch.tensor([True, False, True, False]),
    )
    assert counts == {
        "both_harder_than_hardest_real": 1,
        "ot_only_harder_than_hardest_real": 1,
        "uniform_only_harder_than_hardest_real": 1,
        "neither_harder_than_hardest_real": 1,
    }

    diagnostics = _three_point_fixture()
    expected_delta = (
        diagnostics["ot_synthetic_similarity"]
        - diagnostics["uniform_synthetic_similarity"]
    )
    assert torch.allclose(
        diagnostics["ot_minus_uniform_synthetic_similarity"], expected_delta
    )

    summary = summarize_barycentric_weight_ablation(
        diagnostics, bootstrap_seed=42, bootstrap_samples=50
    )
    paired = summary["paired_comparison"]
    assert sum(paired["paired_hardness_2x2"].values()) == 3
    assert (
        paired["fraction_ot_harder_than_uniform"]
        + paired["fraction_uniform_harder_than_ot"]
        + paired["fraction_tied_within_tolerance"]
        == 1.0
    )
    assert paired["ot_minus_uniform_synthetic_similarity"]["mean"] == float(
        expected_delta.double().mean().item()
    )
