"""Focused tests for the frozen B64 CLIP support-breadth ablation."""

import pytest
import torch
import torch.nn.functional as F

from src.clip_barycentric_weight_metrics import (
    construct_normalized_barycenters,
    uniform_weights_from_support,
)
from src.clip_geometry_v2_metrics import compute_transport_variant
from src.clip_support_breadth_metrics import (
    build_support_breadth_curve,
    compute_support_breadth_batch,
    paired_bootstrap_uniform_minus_ot,
)


K_VALUES = [2, 4, 8, 16, 32]


def _batch_fixture(count=40):
    # Unique angles give deterministic, non-tied cosine rankings.
    angles = torch.linspace(0.03, 2.6, count)
    image_features = torch.stack((angles.cos(), angles.sin()), dim=1)
    text_angles = angles + torch.linspace(0.001, 0.011, count)
    text_features = torch.stack((text_angles.cos(), text_angles.sin()), dim=1)
    return {
        "image_features": image_features,
        "text_features": text_features,
        "image_ids": list(range(count)),
        "species_ids": [f"species-{index % 5}" for index in range(count)],
    }


def _sweep(k_values=(2, 4, 8, 16, 32)):
    return compute_support_breadth_batch(
        **_batch_fixture(),
        k_values=k_values,
        ot_eps=0.049,
        sinkhorn_iters=30,
        solver="historical_sparse_ot",
        tie_tolerance=1e-6,
    )


def test_same_batch_candidate_pool_and_raw_geometry_are_used_across_k():
    sweep = _sweep()
    reference = sweep[2]["transport"]
    for top_k in K_VALUES:
        transport = sweep[top_k]["transport"]
        assert torch.equal(
            transport["positive_mask"], reference["positive_mask"]
        )
        assert torch.equal(
            transport["raw_similarity"], reference["raw_similarity"]
        )


def test_top_k_supports_are_nested_for_non_tied_scores():
    sweep = _sweep()
    for narrow, broad in zip(K_VALUES, K_VALUES[1:]):
        narrow_support = sweep[narrow]["transport"]["support_mask"]
        broad_support = sweep[broad]["transport"]["support_mask"]
        assert torch.all(~narrow_support | broad_support)


def test_ot_and_uniform_have_identical_support_at_every_k():
    sweep = _sweep()
    for result in sweep.values():
        diagnostics = result["diagnostics"]
        support = result["transport"]["support_mask"]
        assert torch.equal(diagnostics["ot_support_mask"], support)
        assert torch.equal(diagnostics["uniform_support_mask"], support)


def test_same_image_positive_is_excluded_at_every_k():
    sweep = _sweep()
    for result in sweep.values():
        transport = result["transport"]
        assert not torch.any(
            transport["support_mask"] & transport["positive_mask"]
        )


def test_uniform_weights_are_exactly_one_over_k_on_support():
    sweep = _sweep()
    for top_k, result in sweep.items():
        diagnostics = result["diagnostics"]
        support = result["transport"]["support_mask"]
        weights = diagnostics["uniform_weights"]
        assert torch.all(weights[~support] == 0)
        assert torch.allclose(
            weights[support], torch.full_like(weights[support], 1 / top_k)
        )
        assert torch.allclose(weights.sum(dim=1), torch.ones(weights.shape[0]))


def test_ot_weights_are_exact_row_normalization_of_existing_historical_plan():
    sweep = _sweep((4,))
    result = sweep[4]
    plan = result["transport"]["plan"]
    expected = plan / plan.sum(dim=1, keepdim=True)

    assert torch.allclose(result["diagnostics"]["ot_weights"], expected)
    assert result["max_abs_ot_similarity_vs_existing_v2"] < 1e-6

    direct = compute_transport_variant(
        **_batch_fixture(),
        logit_scale=1.0,
        cost_space="raw_cosine",
        top_k=4,
        ot_eps=0.049,
        sinkhorn_iters=30,
        solver="historical_sparse_ot",
    )
    assert torch.equal(result["transport"]["support_mask"], direct["support_mask"])
    assert torch.allclose(result["transport"]["plan"], direct["plan"])


def test_synthetic_construction_is_normalized_weighted_feature_sum():
    sweep = _sweep((4,))
    diagnostics = sweep[4]["diagnostics"]
    features = F.normalize(_batch_fixture()["image_features"], dim=-1)
    expected_ot = F.normalize(diagnostics["ot_weights"] @ features, dim=-1)
    expected_uniform = F.normalize(
        diagnostics["uniform_weights"] @ features, dim=-1
    )
    assert torch.allclose(diagnostics["ot_synthetic_features"], expected_ot)
    assert torch.allclose(
        diagnostics["uniform_synthetic_features"], expected_uniform
    )


def test_k2_uniform_cancellation_can_create_harder_barycenter():
    query = torch.tensor([1.0, 0.0])
    candidates = torch.tensor([[0.8, 0.6], [0.8, -0.6]])
    weights = uniform_weights_from_support(
        torch.ones((1, 2), dtype=torch.bool)
    )
    synthetic = construct_normalized_barycenters(weights, candidates)[0]
    assert torch.isclose(query @ synthetic, torch.tensor(1.0))
    assert query @ synthetic > (candidates @ query).max()


def test_broad_irrelevant_support_can_hurt_uniform_but_not_concentrated_weights():
    query = torch.tensor([1.0, 0.0])
    candidates = F.normalize(
        torch.tensor(
            [[0.9, 0.1], [0.85, -0.1], [0.2, 1.0], [0.25, 1.0]]
        ),
        dim=-1,
    )
    uniform = torch.full((1, 4), 0.25)
    concentrated = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    uniform_synthetic = construct_normalized_barycenters(uniform, candidates)[0]
    concentrated_synthetic = construct_normalized_barycenters(
        concentrated, candidates
    )[0]
    assert query @ concentrated_synthetic > query @ uniform_synthetic


def test_paired_bootstrap_estimates_paired_uniform_minus_ot_effects():
    ot_harder = torch.tensor([True, False, True, False])
    uniform_harder = torch.tensor([True, True, False, False])
    uniform_minus_ot = torch.tensor([0.3, 0.2, -0.1, 0.0])
    first = paired_bootstrap_uniform_minus_ot(
        ot_harder,
        uniform_harder,
        uniform_minus_ot,
        seed=42,
        samples=100,
    )
    second = paired_bootstrap_uniform_minus_ot(
        ot_harder,
        uniform_harder,
        uniform_minus_ot,
        seed=42,
        samples=100,
    )
    assert first == second
    assert first["hardness_fraction_uniform_minus_ot"]["estimate"] == 0.0
    assert first["mean_synthetic_similarity_uniform_minus_ot"][
        "estimate"
    ] == pytest.approx(0.1)
    assert "not an iid population guarantee" in first["interpretation"]


def test_cross_k_summary_contains_each_requested_k_once_and_sorted():
    per_k = {}
    for top_k in reversed(K_VALUES):
        method = {
            "fraction_synthetic_harder_than_hardest_real": top_k / 100,
            "weight_normalized_entropy": {"mean": 0.5},
            "peak_weight": {"mean": 0.25},
        }
        per_k[top_k] = {
            "support_fraction_of_available_negatives": {"mean": top_k / 63},
            "ot_weighted": method,
            "uniform_weighted": {
                **method,
                "fraction_synthetic_harder_than_hardest_real": top_k / 90,
            },
            "delta_hardness_fraction_uniform_minus_ot": top_k / 900,
            "paired_comparison": {
                "uniform_minus_ot_synthetic_similarity": {"mean": 0.01},
                "fraction_uniform_harder_than_ot": 0.6,
                "fraction_ot_harder_than_uniform": 0.4,
                "cosine_between_ot_and_uniform_synthetic": {"mean": 0.99},
            },
        }
    curve = build_support_breadth_curve(per_k)
    assert [row["top_k"] for row in curve] == K_VALUES
