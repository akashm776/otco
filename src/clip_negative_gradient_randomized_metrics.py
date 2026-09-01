"""Partition and aggregation helpers for randomized CLIP gradient geometry."""

import hashlib
import itertools
import json

import numpy as np
import torch

from src.clip_negative_gradient_geometry_metrics import (
    correlation_pair,
    distribution_summary,
)


PRIMARY_STABILITY_PATHS = {
    "fraction_u8_gt_hardest_real": (
        "hardness_summary",
        "fraction_uniform_top8_gt_hardest_real",
    ),
    "query_gradient_cosine_mean": (
        "query_gradient_alignment",
        "cosine_u8_vs_hardest_real",
        "mean",
    ),
    "image_gradient_cosine_mean": (
        "image_gradient_alignment",
        "cosine_u8_vs_hardest_real",
        "mean",
    ),
    "joint_gradient_cosine_mean": (
        "joint_gradient_alignment",
        "cosine_u8_vs_hardest_real",
        "mean",
    ),
    "joint_fraction_cosine_lt_0": (
        "joint_gradient_alignment",
        "cosine_u8_vs_hardest_real",
        "fraction_lt_0",
    ),
    "u8_effective_image_gradient_support": (
        "gradient_footprint",
        "image_gradient_effective_support_u8",
        "mean",
    ),
    "hardest_real_effective_image_gradient_support": (
        "gradient_footprint",
        "image_gradient_effective_support_real",
        "mean",
    ),
    "u8_native_alignment": (
        "native_objective_alignment",
        "u8",
        "mean",
    ),
    "hardest_real_native_alignment": (
        "native_objective_alignment",
        "hardest_real",
        "mean",
    ),
    "u8_margin_directional_change": (
        "margin_directional_change",
        "u8",
        "mean",
    ),
    "hardest_real_margin_directional_change": (
        "margin_directional_change",
        "hardest_real",
        "mean",
    ),
    "fraction_hardest_real_better_on_margin": (
        "margin_directional_change",
        "fraction_hardest_real_better",
    ),
}


PAIRWISE_QUERY_FIELDS = {
    "u8_minus_hardest_real_hardness": "u8_minus_hardest_real",
    "joint_gradient_cosine": "joint_gradient_cosine_u8_real",
    "u8_native_alignment": "native_alignment_u8",
    "u8_margin_directional_change": "margin_directional_change_u8",
}


STRATIFIED_FIELDS = {
    "joint_gradient_cosine": "joint_gradient_cosine_u8_real",
    "native_alignment_u8": "native_alignment_u8",
    "native_alignment_hardest_real": "native_alignment_hardest_real",
    "margin_directional_change_u8": "margin_directional_change_u8",
    "margin_directional_change_hardest_real": (
        "margin_directional_change_hardest_real"
    ),
}


def make_partition_indices(count, batch_size, seed=None):
    """Partition ``range(count)`` without labels, resampling, or replacement."""
    if count < 1 or batch_size < 1 or count % batch_size:
        raise ValueError("count must be positive and divisible by batch_size")
    if seed is None:
        order = torch.arange(count)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        order = torch.randperm(count, generator=generator)
    return [
        order[start : start + batch_size].tolist()
        for start in range(0, count, batch_size)
    ]


def validate_partition(batches, count, batch_size):
    """Return explicit exhaustive/non-overlapping partition invariants."""
    flattened = [index for batch in batches for index in batch]
    expected = set(range(count))
    invariants = {
        "flattened_length": len(flattened),
        "unique_entry_count": len(set(flattened)),
        "contains_exactly_range_0_to_count_minus_1": set(flattened) == expected,
        "batch_count": len(batches),
        "all_batches_exact_size": all(len(batch) == batch_size for batch in batches),
    }
    if not (
        invariants["flattened_length"] == count
        and invariants["unique_entry_count"] == count
        and invariants["contains_exactly_range_0_to_count_minus_1"]
        and invariants["batch_count"] == count // batch_size
        and invariants["all_batches_exact_size"]
    ):
        raise ValueError(f"Invalid exhaustive partition: {invariants}")
    return invariants


def flattened_partition_sha256(batches):
    """SHA256 of compact-JSON encoded flattened holdout positions."""
    flattened = [index for batch in batches for index in batch]
    payload = json.dumps(flattened, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def annotate_species_composition(row, batch_species_ids):
    """Add label-only observations without affecting selection or gradients."""
    query_position = row["query_index_within_batch"]
    anchor_species = batch_species_ids[query_position]
    support = json.loads(row["u8_support_indices"])
    available_same_species = sum(
        species == anchor_species
        for index, species in enumerate(batch_species_ids)
        if index != query_position
    )
    support_same_species = sum(
        batch_species_ids[index] == anchor_species for index in support
    )
    hardest_same_species = (
        batch_species_ids[row["hardest_real_index"]] == anchor_species
    )
    annotated = dict(row)
    annotated.update(
        {
            "same_species_negatives_available_count": available_same_species,
            "has_same_species_negative_in_batch": available_same_species >= 1,
            "u8_support_same_species_count": support_same_species,
            "u8_support_contains_same_species": support_same_species >= 1,
            "u8_support_contains_at_least_two_same_species": (
                support_same_species >= 2
            ),
            "hardest_real_same_species": hardest_same_species,
        }
    )
    return annotated


def summarize_species_composition(rows):
    """Summarize batch and candidate species composition observationally."""
    count = len(rows)
    available = np.asarray(
        [row["same_species_negatives_available_count"] for row in rows],
        dtype=np.float64,
    )
    support_count = np.asarray(
        [row["u8_support_same_species_count"] for row in rows],
        dtype=np.float64,
    )
    return {
        "labels_used_for_partition_selection_or_loss": False,
        "query_count": count,
        "fraction_queries_with_same_species_negative_in_batch": float(
            (available >= 1).mean()
        ),
        "same_species_negatives_available_count": distribution_summary(available),
        "u8_support_same_species_fraction": distribution_summary(
            [row["u8_support_same_species_fraction"] for row in rows]
        ),
        "u8_support_same_species_count": distribution_summary(support_count),
        "u8_support_unique_species_count": distribution_summary(
            [row["u8_support_unique_species_count"] for row in rows]
        ),
        "fraction_u8_supports_containing_same_species": float(
            (support_count >= 1).mean()
        ),
        "fraction_u8_supports_containing_at_least_two_same_species": float(
            (support_count >= 2).mean()
        ),
        "fraction_hardest_real_same_species": float(
            np.mean([row["hardest_real_same_species"] for row in rows])
        ),
    }


def _nested_value(record, path):
    value = record
    for key in path:
        value = value[key]
    return float(value)


def build_cross_partition_stability(partition_reports, shuffled_names):
    """Collect pre-registered population metrics across all partitions."""
    result = {}
    for metric, path in PRIMARY_STABILITY_PATHS.items():
        values = {
            name: _nested_value(report, path)
            for name, report in partition_reports.items()
        }
        shuffled = np.asarray([values[name] for name in shuffled_names])
        result[metric] = {
            "by_partition": values,
            "shuffled_partition_summary": {
                "mean": float(shuffled.mean()),
                "std": float(shuffled.std()),
                "min": float(shuffled.min()),
                "max": float(shuffled.max()),
            },
        }
    return result


def evaluate_robustness_criteria(partition_reports, shuffled_names):
    """Evaluate the pre-registered qualitative falsification criteria."""
    criteria = {
        "joint_gradient_cosine_mean_lt_0": (
            lambda report: _nested_value(
                report,
                PRIMARY_STABILITY_PATHS["joint_gradient_cosine_mean"],
            )
            < 0
        ),
        "joint_fraction_cosine_lt_0_gt_0_60": (
            lambda report: _nested_value(
                report,
                PRIMARY_STABILITY_PATHS["joint_fraction_cosine_lt_0"],
            )
            > 0.60
        ),
        "u8_effective_support_gt_hardest_real": (
            lambda report: _nested_value(
                report,
                PRIMARY_STABILITY_PATHS["u8_effective_image_gradient_support"],
            )
            > _nested_value(
                report,
                PRIMARY_STABILITY_PATHS[
                    "hardest_real_effective_image_gradient_support"
                ],
            )
        ),
        "u8_native_alignment_lt_hardest_real": (
            lambda report: _nested_value(
                report, PRIMARY_STABILITY_PATHS["u8_native_alignment"]
            )
            < _nested_value(
                report,
                PRIMARY_STABILITY_PATHS["hardest_real_native_alignment"],
            )
        ),
        "u8_margin_change_lt_hardest_real": (
            lambda report: _nested_value(
                report,
                PRIMARY_STABILITY_PATHS["u8_margin_directional_change"],
            )
            < _nested_value(
                report,
                PRIMARY_STABILITY_PATHS[
                    "hardest_real_margin_directional_change"
                ],
            )
        ),
        "fraction_hardest_real_better_margin_gt_0_90": (
            lambda report: _nested_value(
                report,
                PRIMARY_STABILITY_PATHS[
                    "fraction_hardest_real_better_on_margin"
                ],
            )
            > 0.90
        ),
    }
    by_partition = {}
    for name in shuffled_names:
        results = {
            criterion: bool(check(partition_reports[name]))
            for criterion, check in criteria.items()
        }
        by_partition[name] = {
            "criteria": results,
            "all_criteria_pass": all(results.values()),
        }
    return {
        "nature": "pre-specified qualitative falsification criteria, not significance tests",
        "thresholds_were_not_selected_from_results": True,
        "by_shuffled_partition": by_partition,
        "all_three_shuffled_partitions_pass_all_criteria": all(
            result["all_criteria_pass"] for result in by_partition.values()
        ),
    }


def pairwise_query_correlations(rows_by_partition, shuffled_names):
    """Pair same holdout examples across shuffled partition contexts."""
    indexed = {
        name: {row["query_index"]: row for row in rows_by_partition[name]}
        for name in shuffled_names
    }
    result = {}
    for left_name, right_name in itertools.combinations(shuffled_names, 2):
        left = indexed[left_name]
        right = indexed[right_name]
        if set(left) != set(right):
            raise ValueError("Partitions do not contain the same query identities")
        query_indices = sorted(left)
        pair = {}
        for metric, field in PAIRWISE_QUERY_FIELDS.items():
            pair[metric] = correlation_pair(
                [left[index][field] for index in query_indices],
                [right[index][field] for index in query_indices],
            )
        result[f"{left_name}_vs_{right_name}"] = {
            "paired_query_count": len(query_indices),
            "metrics": pair,
        }
    return result


def _stratum_summary(rows):
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        **{
            name: distribution_summary([row[field] for row in rows])
            for name, field in STRATIFIED_FIELDS.items()
        },
    }


def same_species_stratification(rows):
    """Describe shuffled gradients by two observational same-species flags."""
    result = {}
    for field in (
        "hardest_real_same_species",
        "u8_support_contains_same_species",
    ):
        result[field] = {
            "false": _stratum_summary([row for row in rows if not row[field]]),
            "true": _stratum_summary([row for row in rows if row[field]]),
        }
    return result
