"""Focused tests for randomized B64 negative-gradient geometry."""

import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.clip_negative_gradient_geometry_randomized import (
    EXPECTED_PARTITIONS,
    build_partition_conditions,
    load_config,
    run_feature_partition,
)
from src.clip_negative_gradient_randomized_metrics import (
    annotate_species_composition,
    evaluate_robustness_criteria,
    make_partition_indices,
    pairwise_query_correlations,
    validate_partition,
)


ROOT = Path(__file__).resolve().parents[1]


def test_same_partition_seed_is_deterministic_and_different_seeds_differ():
    first = make_partition_indices(1024, 64, seed=42)
    repeated = make_partition_indices(1024, 64, seed=42)
    different = make_partition_indices(1024, 64, seed=123)
    assert first == repeated
    assert first != different


def test_each_partition_is_exhaustive_unique_and_has_exact_b64_shape():
    for seed in (None, 42, 123, 4242):
        batches = make_partition_indices(1024, 64, seed=seed)
        invariants = validate_partition(batches, 1024, 64)
        flattened = [index for batch in batches for index in batch]
        assert len(batches) == 16
        assert all(len(batch) == 64 for batch in batches)
        assert len(flattened) == len(set(flattened)) == 1024
        assert set(flattened) == set(range(1024))
        assert invariants["contains_exactly_range_0_to_count_minus_1"]


def test_sequential_partition_is_exact_contiguous_chunks():
    batches = make_partition_indices(1024, 64)
    assert batches[0] == list(range(0, 64))
    assert batches[1] == list(range(64, 128))
    assert batches[-1] == list(range(960, 1024))


def test_partition_helper_api_has_no_label_or_species_input():
    assert list(inspect.signature(make_partition_indices).parameters) == [
        "count",
        "batch_size",
        "seed",
    ]


def test_all_four_conditions_contain_each_source_position_once():
    conditions, artifact = build_partition_conditions(
        EXPECTED_PARTITIONS, 1024, 64
    )
    assert list(conditions) == [spec["name"] for spec in EXPECTED_PARTITIONS]
    for name, condition in conditions.items():
        flattened = [index for batch in condition["batches"] for index in batch]
        assert sorted(flattened) == list(range(1024))
        assert artifact["invariants"][name]["unique_entry_count"] == 1024
        assert len(artifact["flattened_sha256"][name]) == 64


def test_toy_output_cardinality_is_partition_count_times_sample_count():
    count = 20
    batch_size = 10
    generator = torch.Generator().manual_seed(7)
    images = F.normalize(torch.randn(count, 6, generator=generator), dim=-1)
    texts = F.normalize(
        images + 0.1 * torch.randn(count, 6, generator=generator), dim=-1
    )
    metadata = [
        {
            "source_index": index + 100,
            "image_key": f"image-{index}",
            "species_id": index % 4,
        }
        for index in range(count)
    ]
    specs = [
        {"name": "sequential", "mode": "sequential"},
        {"name": "shuffle_seed_42", "mode": "shuffled", "seed": 42},
    ]
    conditions, _ = build_partition_conditions(specs, count, batch_size)
    all_rows = []
    for condition in conditions.values():
        rows, audit = run_feature_partition(
            condition=condition,
            image_features=images,
            text_features=texts,
            metadata=metadata,
            logit_scale=torch.tensor(1.0),
            top_k=8,
            tangent_tolerance=1e-5,
        )
        assert audit["query_pass"] and audit["image_pass"]
        assert len({row["query_index"] for row in rows}) == count
        all_rows.extend(rows)
    assert len(all_rows) == len(specs) * count


def test_randomized_config_has_exact_pre_registered_conditions():
    config = load_config(
        ROOT
        / "configs/hf_cub200_clip_negative_gradient_geometry_randomized_batches.yaml"
    )
    assert config["diagnostic"]["partitions"] == EXPECTED_PARTITIONS
    assert config["diagnostic"]["batch_size"] == 64
    assert config["diagnostic"]["top_k"] == 8
    assert config["diagnostic"]["tangent_tolerance"] == 1e-5


def test_species_annotations_are_observational_counts_on_fixed_support():
    row = {
        "query_index_within_batch": 0,
        "u8_support_indices": "[1, 2, 3]",
        "hardest_real_index": 2,
    }
    annotated = annotate_species_composition(row, [5, 5, 6, 5])
    assert annotated["same_species_negatives_available_count"] == 2
    assert annotated["u8_support_same_species_count"] == 2
    assert annotated["u8_support_contains_same_species"]
    assert annotated["u8_support_contains_at_least_two_same_species"]
    assert not annotated["hardest_real_same_species"]


def _robustness_report(joint_mean=-0.2):
    return {
        "joint_gradient_alignment": {
            "cosine_u8_vs_hardest_real": {
                "mean": joint_mean,
                "fraction_lt_0": 0.8,
            }
        },
        "gradient_footprint": {
            "image_gradient_effective_support_u8": {"mean": 12.0},
            "image_gradient_effective_support_real": {"mean": 3.0},
        },
        "native_objective_alignment": {
            "u8": {"mean": 0.1},
            "hardest_real": {"mean": 0.3},
        },
        "margin_directional_change": {
            "u8": {"mean": -0.1},
            "hardest_real": {"mean": 1.0},
            "fraction_hardest_real_better": 0.95,
        },
    }


def test_robustness_criteria_are_exactly_pre_registered_and_fail_plainly():
    names = ["shuffle_seed_42", "shuffle_seed_123", "shuffle_seed_4242"]
    reports = {name: _robustness_report() for name in names}
    passing = evaluate_robustness_criteria(reports, names)
    assert passing["all_three_shuffled_partitions_pass_all_criteria"]
    reports["shuffle_seed_123"] = _robustness_report(joint_mean=0.01)
    failing = evaluate_robustness_criteria(reports, names)
    assert not failing["all_three_shuffled_partitions_pass_all_criteria"]
    assert not failing["by_shuffled_partition"]["shuffle_seed_123"][
        "criteria"
    ]["joint_gradient_cosine_mean_lt_0"]


def test_pairwise_query_correlations_pair_stable_query_identity():
    names = ["shuffle_seed_42", "shuffle_seed_123"]
    rows = {}
    for multiplier, name in enumerate(names, start=1):
        rows[name] = [
            {
                "query_index": index,
                "u8_minus_hardest_real": multiplier * (index + 1),
                "joint_gradient_cosine_u8_real": multiplier * (index + 2),
                "native_alignment_u8": multiplier * (index + 3),
                "margin_directional_change_u8": multiplier * (index + 4),
            }
            for index in range(5)
        ]
    correlations = pairwise_query_correlations(rows, names)
    pair = correlations["shuffle_seed_42_vs_shuffle_seed_123"]
    assert pair["paired_query_count"] == 5
    for metric in pair["metrics"].values():
        assert metric["pearson"] == pytest.approx(1.0)
        assert metric["spearman"] == pytest.approx(1.0)


def test_randomized_diagnostic_has_no_training_operations():
    from src import clip_negative_gradient_geometry_randomized as diagnostic

    source = inspect.getsource(diagnostic)
    assert "optimizer.step(" not in source
    assert "scheduler.step(" not in source
    assert ".backward(" not in source
