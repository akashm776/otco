"""Focused tests for the additive frozen CLIP V2 diagnostics."""

import torch

from model.loss import SoftmaxMixLoss, compute_alpha_effective
from src.clip_geometry_metrics import build_sparse_transport_plan
from src.clip_geometry_v2_metrics import (
    audit_transport_mass,
    build_audited_transport_plan,
    compare_transport_variants,
    compute_transport_variant,
    compute_zero_shot_species_evaluation,
    emulate_batch_local_gates,
    summarize_transport_variant,
)


def _toy_geometry():
    image_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
            [0.7, 0.7, 0.0],
            [0.0, 0.7, 0.7],
        ]
    )
    text_features = image_features.clone()
    image_ids = list(range(len(image_features)))
    species_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    return image_features, text_features, image_ids, species_ids


def test_historical_audited_solver_preserves_v1_and_training_behavior():
    torch.manual_seed(7)
    scores = torch.randn(12, 12)
    positives = torch.eye(12, dtype=torch.bool)
    dummy_features = torch.randn(12, 4)

    v1 = build_sparse_transport_plan(
        scores, positives, top_k=3, ot_eps=0.7, sinkhorn_iters=30
    )
    audited = build_audited_transport_plan(
        scores,
        positives,
        top_k=3,
        ot_eps=0.7,
        sinkhorn_iters=30,
        solver="historical_sparse_ot",
    )
    training_plan, training_support = SoftmaxMixLoss(
        top_k=3, ot_eps=0.7, sinkhorn_iters=30
    )._make_plan(dummy_features, dummy_features, scores)

    assert torch.equal(audited.support_mask, v1.support_mask)
    assert torch.equal(audited.support_mask, training_support)
    assert torch.allclose(audited.plan, v1.plan)
    assert torch.allclose(audited.plan, training_plan)


def test_raw_cosine_and_scale_matched_logits_are_equivalent():
    image_features, text_features, image_ids, species_ids = _toy_geometry()
    common = {
        "image_features": image_features,
        "text_features": text_features,
        "logit_scale": 100.0,
        "image_ids": image_ids,
        "species_ids": species_ids,
        "top_k": 4,
        "sinkhorn_iters": 100,
        "solver": "historical_sparse_ot",
    }
    raw = compute_transport_variant(
        **common, cost_space="raw_cosine", ot_eps=0.049
    )
    scaled = compute_transport_variant(
        **common, cost_space="clip_scaled_logits", ot_eps=4.9
    )
    comparison = compare_transport_variants(raw, scaled)

    assert torch.equal(raw["support_mask"], scaled["support_mask"])
    assert comparison["selected_index_agreement"] == 1.0
    assert comparison["max_abs_row_normalized_plan_difference"] < 1e-6
    assert comparison["max_abs_entropy_difference"] < 1e-6


def test_historical_sparse_solver_exposes_mass_removed_after_masking():
    torch.manual_seed(0)
    scores = torch.randn(16, 16)
    positives = torch.eye(16, dtype=torch.bool)
    transport = build_audited_transport_plan(
        scores,
        positives,
        top_k=2,
        ot_eps=0.7,
        sinkhorn_iters=30,
        solver="historical_sparse_ot",
    )
    audit = audit_transport_mass(transport)

    assert torch.all(transport.plan[~transport.support_mask] == 0)
    assert audit["total_mass_before_final_mask"] > audit["total_mass_returned"]
    assert audit["fraction_mass_removed_by_final_mask"] > 0.01
    assert audit["max_row_marginal_error"] > 0
    assert audit["max_column_marginal_error"] > 0


def test_support_preserving_sparse_solver_retains_mass_and_marginals():
    count = 8
    scores = torch.full((count, count), -10.0)
    for row in range(count):
        scores[row, (row - 1) % count] = 1.0
        scores[row, (row + 1) % count] = 0.9
    scores.fill_diagonal_(10.0)
    positives = torch.eye(count, dtype=torch.bool)
    transport = build_audited_transport_plan(
        scores,
        positives,
        top_k=2,
        ot_eps=1.0,
        sinkhorn_iters=500,
        solver="support_preserving_sparse_ot",
    )
    audit = audit_transport_mass(transport)

    assert torch.all(transport.plan[~transport.support_mask] == 0)
    assert audit["fraction_mass_removed_by_final_mask"] == 0.0
    assert abs(audit["total_mass_returned"] - 1.0) < 1e-6
    assert audit["max_row_marginal_error"] < 1e-6
    assert audit["max_column_marginal_error"] < 1e-6


def test_batch_emulation_uses_actual_batch_mean_gate_semantics():
    image_features, text_features, image_ids, species_ids = _toy_geometry()
    thresholds = {
        "gate_sim": -4.0,
        "entropy_threshold": 3.0,
        "gap_suppress_easy": 0.10,
        "gap_downweight_hard": -0.07,
        "hard_alpha_scale": 0.25,
    }
    records = emulate_batch_local_gates(
        image_features=image_features,
        text_features=text_features,
        logit_scale=100.0,
        image_ids=image_ids,
        species_ids=species_ids,
        variants={
            "raw_cosine_ot": {
                "cost_space": "raw_cosine",
                "top_k": 3,
                "ot_eps": 0.049,
                "sinkhorn_iters": 50,
                "solver": "historical_sparse_ot",
            }
        },
        batch_size=4,
        num_batches=2,
        seed=42,
        scheduled_alpha=0.05,
        thresholds=thresholds,
    )

    assert len(records) == 2
    for record in records:
        expected_alpha, expected_bucket = compute_alpha_effective(
            0.05,
            record["mean_coupling_entropy"],
            record["mean_positive_selected_gap"],
            entropy_threshold=thresholds["entropy_threshold"],
            gap_suppress_easy=thresholds["gap_suppress_easy"],
            gap_downweight_hard=thresholds["gap_downweight_hard"],
            hard_alpha_scale=thresholds["hard_alpha_scale"],
        )
        assert record["alpha_effective"] == expected_alpha
        assert record["gap_bucket_id"] == expected_bucket
        assert "mean_synthetic_vs_hardest_real_similarity_delta" in record
        assert "mean_synthetic_vs_hardest_real_logit_delta" in record
        assert (
            "fraction_synthetic_harder_than_hardest_real_contributor" in record
        )


def test_species_evaluation_and_report_fields_are_explicit():
    image_features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    species_text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    evaluation = compute_zero_shot_species_evaluation(
        image_features,
        species_text_features,
        ["a", "a", "b", "b"],
        ["a", "b"],
    )
    diagnostics = compute_transport_variant(
        image_features=image_features,
        text_features=image_features,
        logit_scale=100.0,
        image_ids=[0, 1, 2, 3],
        species_ids=["a", "a", "b", "b"],
        cost_space="raw_cosine",
        top_k=3,
        ot_eps=0.049,
        sinkhorn_iters=100,
        historical_thresholds={
            "entropy_threshold": 3.0,
            "gap_suppress_easy": 0.10,
            "gap_downweight_hard": -0.07,
        },
    )
    summary = summarize_transport_variant(diagnostics)
    contributor_mask = diagnostics["plan"] > 0
    expected_hardest_real = diagnostics["raw_similarity"].masked_fill(
        ~contributor_mask, float("-inf")
    ).max(1).values
    expected_delta = diagnostics["synthetic_similarity"] - expected_hardest_real

    assert evaluation["top_1_accuracy"] == 1.0
    assert evaluation["top_5_accuracy"] == 1.0
    assert torch.allclose(
        diagnostics["hardest_real_contributor_similarity"],
        expected_hardest_real,
    )
    assert torch.allclose(
        diagnostics["synthetic_vs_hardest_real_similarity_delta"],
        expected_delta,
    )
    assert torch.allclose(
        diagnostics["synthetic_vs_hardest_real_logit_delta"],
        expected_delta * 100.0,
    )
    assert "fraction_positive_selected_gap_lt_0" in summary["extreme_gap_behavior"]
    assert (
        "fraction_synthetic_similarity_gt_positive_similarity"
        in summary["extreme_gap_behavior"]
    )
    assert (
        "fraction_synthetic_harder_than_hardest_real_contributor"
        in summary["extreme_gap_behavior"]
    )
    assert "top-k" in summary["strict_hardest_comparison"]["selected_rank_context"]
    assert "Observational" in summary[
        "per_query_historical_threshold_overlay"
    ]["interpretation"]


def test_barycenter_can_be_harder_than_every_real_contributor():
    # The two contributors are symmetric around the anchor. Their normalized
    # barycenter aligns perfectly with it even though each real contributor has
    # cosine similarity 0.8.
    features = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.8, -0.6]])
    diagnostics = compute_transport_variant(
        image_features=features,
        text_features=features,
        logit_scale=100.0,
        image_ids=[0, 1, 2],
        species_ids=["a", "b", "c"],
        cost_space="raw_cosine",
        top_k=2,
        ot_eps=10.0,
        sinkhorn_iters=100,
    )

    assert torch.isclose(
        diagnostics["hardest_real_contributor_similarity"][0],
        torch.tensor(0.8),
    )
    assert torch.isclose(
        diagnostics["synthetic_similarity"][0], torch.tensor(1.0)
    )
    assert torch.isclose(
        diagnostics["synthetic_vs_hardest_real_similarity_delta"][0],
        torch.tensor(0.2),
    )
