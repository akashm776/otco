"""Focused tests for the frozen geometry-adaptive neighborhood diagnostic."""

import inspect

import pytest
import torch
import torch.nn.functional as F

import src.clip_geometry_adaptive_neighborhood as runner
from src.clip_barycentric_weight_metrics import (
    construct_normalized_barycenters,
    uniform_weights_from_support,
)
from src.clip_geometry_adaptive_metrics import (
    compute_geometry_adaptive_batch,
    eligible_negative_similarity_spectrum,
    select_geometry_boundary_k,
    select_oracle_k,
    summarize_geometry_adaptive,
)
from src.clip_geometry_v2_metrics import _top_k_support


K_VALUES = [2, 4, 8, 16, 32]


def _batch_fixture(count=40):
    angles = torch.linspace(0.03, 2.6, count)
    images = torch.stack((angles.cos(), angles.sin()), dim=1)
    text_angles = angles + torch.linspace(0.001, 0.011, count)
    texts = torch.stack((text_angles.cos(), text_angles.sin()), dim=1)
    return {
        "image_features": images,
        "text_features": texts,
        "image_ids": list(range(count)),
        "allowed_k_values": K_VALUES,
        "selector_tie_tolerance": 1e-12,
        "comparison_tie_tolerance": 1e-6,
    }


def test_similarity_spectrum_sorts_descending_after_positive_exclusion():
    similarities = torch.tensor(
        [[0.9, 1.0, 0.2, 0.7], [0.3, 0.8, 1.0, 0.4]]
    )
    positive = torch.tensor(
        [[False, True, False, False], [False, False, True, False]]
    )
    spectrum, counts = eligible_negative_similarity_spectrum(
        similarities, positive
    )
    assert torch.equal(counts, torch.tensor([3, 3]))
    assert torch.equal(spectrum[0], torch.tensor([0.9, 0.7, 0.2]))
    assert torch.equal(spectrum[1], torch.tensor([0.8, 0.4, 0.3]))


def test_boundary_gap_indexing_uses_rank_k_minus_rank_k_plus_one():
    spectrum = torch.tensor([[0.99, 0.90, 0.80, 0.50, 0.49, 0.20]])
    result = select_geometry_boundary_k(spectrum, [2, 4])
    expected = torch.tensor([[0.90 - 0.80, 0.50 - 0.49]])
    assert torch.allclose(result["boundary_gaps"], expected)


def test_geometry_selector_chooses_largest_gap_after_rank_eight():
    spectrum = torch.linspace(1.0, 0.4, 40).unsqueeze(0)
    spectrum[:, 8:] -= 0.25
    result = select_geometry_boundary_k(spectrum, K_VALUES)
    assert result["selected_k"].item() == 8


def test_geometry_selector_breaks_equal_largest_gap_toward_smaller_k():
    spectrum = torch.linspace(1.0, 0.4, 40).unsqueeze(0)
    # Produce equal additional drops after ranks 4 and 8.
    spectrum[:, 4:] -= 0.2
    spectrum[:, 8:] -= 0.2
    result = select_geometry_boundary_k(
        spectrum, K_VALUES, tie_tolerance=1e-7
    )
    assert result["selected_k"].item() == 4


def test_geometry_selector_api_cannot_receive_forbidden_inputs():
    parameters = inspect.signature(select_geometry_boundary_k).parameters
    assert list(parameters) == [
        "negative_similarities",
        "allowed_k_values",
        "tie_tolerance",
    ]
    forbidden = {
        "synthetic_similarities",
        "species_ids",
        "ot_plan",
        "retrieval_metrics",
        "loss",
    }
    assert forbidden.isdisjoint(parameters)


def test_adaptive_uniform_barycenter_equals_existing_fixed_k_construction():
    diagnostics = compute_geometry_adaptive_batch(**_batch_fixture())
    for query, selected_k in enumerate(
        diagnostics["geometry_selected_k"].tolist()
    ):
        expected = diagnostics["fixed_by_k"][selected_k][
            "synthetic_features"
        ][query]
        assert torch.allclose(
            diagnostics["geometry_synthetic_features"][query], expected
        )
        assert torch.equal(
            diagnostics["geometry_support_mask"][query],
            diagnostics["fixed_by_k"][selected_k]["support_mask"][query],
        )


def test_oracle_chooses_maximum_synthetic_and_smaller_k_on_tie():
    similarities = torch.tensor(
        [[0.1, 0.8, 0.2, 0.3, 0.4], [0.9, 0.9, 0.2, 0.1, 0.0]]
    )
    oracle = select_oracle_k(similarities, K_VALUES, tie_tolerance=1e-6)
    assert torch.equal(oracle["selected_k"], torch.tensor([4, 2]))
    assert oracle["oracle_is_evaluation_only"] is True


def test_fixed_k8_matches_exact_existing_uniform_construction():
    fixture = _batch_fixture()
    diagnostics = compute_geometry_adaptive_batch(**fixture)
    images = F.normalize(fixture["image_features"], dim=-1)
    texts = F.normalize(fixture["text_features"], dim=-1)
    raw = texts @ images.T
    positive = torch.eye(images.shape[0], dtype=torch.bool)
    _, support = _top_k_support(raw, positive, 8)
    weights = uniform_weights_from_support(support).to(images.dtype)
    synthetic = construct_normalized_barycenters(weights, images)
    similarity = (texts * synthetic).sum(dim=1)
    fixed = diagnostics["fixed_by_k"][8]
    assert torch.equal(fixed["support_mask"], support)
    assert torch.allclose(fixed["weights"], weights)
    assert torch.allclose(fixed["synthetic_features"], synthetic)
    assert torch.allclose(fixed["synthetic_similarity"], similarity)


def test_geometry_oracle_and_controls_share_one_paired_observation_identity():
    diagnostics = compute_geometry_adaptive_batch(**_batch_fixture())
    metadata = [
        {
            "source_index": index + 100,
            "image_key": f"image-{index}",
            "species_id": f"species-{index % 5}",
        }
        for index in range(40)
    ]
    rows = runner.build_observation_rows(
        batch_index=7,
        pool_indices=torch.arange(40),
        metadata=metadata,
        diagnostics=diagnostics,
    )
    assert len(rows) == 40
    assert rows[3]["observation_id"] == "batch_0007_query_03"
    assert rows[3]["holdout_pool_index"] == 3
    assert {
        "geometry_synthetic_similarity",
        "oracle_synthetic_similarity",
        "uniform_k8_similarity",
    }.issubset(rows[3])


def test_runner_has_no_training_optimizer_scheduler_or_backward_path():
    source = inspect.getsource(runner.run).lower()
    assert "optimizer" not in source
    assert "scheduler" not in source
    assert "backward" not in source
    assert "training loss" not in source


def test_same_image_positive_never_enters_any_fixed_or_adaptive_support():
    diagnostics = compute_geometry_adaptive_batch(**_batch_fixture())
    positive = diagnostics["positive_mask"]
    for fixed in diagnostics["fixed_by_k"].values():
        assert not torch.any(fixed["support_mask"] & positive)
    assert not torch.any(diagnostics["geometry_support_mask"] & positive)
    assert not torch.any(diagnostics["oracle_support_mask"] & positive)


def test_paired_summary_reports_oracle_headroom_and_geometry_recovery():
    rows = []
    for index, (geometry, oracle, fixed8) in enumerate(
        [(0.8, 0.9, 0.7), (0.6, 0.8, 0.7), (0.75, 0.75, 0.75)]
    ):
        row = {
            "positive_similarity": 0.72,
            "hardest_real_similarity": 0.69,
            "geometry_selected_k": [4, 8, 16][index],
            "oracle_selected_k": [2, 4, 16][index],
            "geometry_selected_support_fraction": [4 / 63, 8 / 63, 16 / 63][
                index
            ],
            "selected_boundary_gap": 0.02 + index * 0.01,
            "geometry_synthetic_similarity": geometry,
            "oracle_synthetic_similarity": oracle,
        }
        for top_k in K_VALUES:
            row[f"gap_after_k{top_k}"] = top_k / 1000
            row[f"uniform_k{top_k}_similarity"] = (
                fixed8 if top_k == 8 else fixed8 - 0.01
            )
        rows.append(row)

    summary = summarize_geometry_adaptive(
        rows,
        allowed_k_values=K_VALUES,
        bootstrap_seed=42,
        bootstrap_samples=100,
    )
    expected_oracle_gain = ((0.9 + 0.8 + 0.75) / 3) - (
        (0.7 + 0.7 + 0.75) / 3
    )
    expected_geometry_gain = ((0.8 + 0.6 + 0.75) / 3) - (
        (0.7 + 0.7 + 0.75) / 3
    )
    headroom = summary["adaptive_headroom"]
    assert headroom["oracle_gain_over_fixed_k8_mean_similarity"] == pytest.approx(
        expected_oracle_gain
    )
    assert headroom["geometry_gain_over_fixed_k8_mean_similarity"] == pytest.approx(
        expected_geometry_gain
    )
    assert headroom["fraction_of_oracle_gain_recovered"] == pytest.approx(
        expected_geometry_gain / expected_oracle_gain
    )
    assert summary["bootstrap"]["samples"] == 100
    assert sum(
        sum(row) for row in summary["geometry_vs_oracle"]["confusion_matrix"]["counts"]
    ) == 3
