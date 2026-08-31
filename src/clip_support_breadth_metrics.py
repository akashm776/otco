"""Frozen B64 OT-vs-uniform diagnostics across top-k support breadth.

The transport and batching operators are imported from the existing CLIP V2
diagnostic.  This module only coordinates a paired top-k sweep and summarizes
the resulting observations; it does not define a new transport solver.
"""

from collections import defaultdict

import numpy as np
import torch

from src.clip_barycentric_weight_metrics import (
    compute_barycentric_weight_ablation,
    paired_hardness_counts,
    summarize_barycentric_weight_ablation,
)
from src.clip_geometry_metrics import quantile_summary
from src.clip_geometry_v2_metrics import compute_transport_variant


OBSERVATION_TENSOR_KEYS = (
    "positive_similarity",
    "hardest_real_similarity",
    "ot_synthetic_similarity",
    "uniform_synthetic_similarity",
    "ot_vs_hardest_delta",
    "uniform_vs_hardest_delta",
    "ot_vs_positive_delta",
    "uniform_vs_positive_delta",
    "ot_minus_uniform_synthetic_similarity",
    "ot_harder_than_hardest_real",
    "uniform_harder_than_hardest_real",
    "ot_harder_than_uniform",
    "uniform_harder_than_ot",
    "tied_within_tolerance",
    "ot_weight_entropy",
    "ot_weight_normalized_entropy",
    "ot_peak_weight",
    "uniform_weight_entropy",
    "uniform_weight_normalized_entropy",
    "uniform_peak_weight",
    "l1_distance_ot_vs_uniform_weights",
    "cosine_between_ot_and_uniform_synthetic",
)


def compute_support_breadth_batch(
    *,
    image_features,
    text_features,
    image_ids,
    species_ids,
    k_values,
    ot_eps=0.049,
    sinkhorn_iters=30,
    solver="historical_sparse_ot",
    tie_tolerance=1e-6,
):
    """Evaluate every k on one unchanged batch candidate pool.

    ``compute_transport_variant`` is the existing V2 batch-local operator.  It
    performs same-image positive exclusion, raw-cosine top-k selection, and
    the historical sparse OT solve.  The paired helper then constructs OT and
    uniform barycenters from that exact returned support.
    """
    ordered_k = list(k_values)
    if ordered_k != sorted(set(ordered_k)) or not ordered_k:
        raise ValueError("k_values must be unique, non-empty, and sorted")

    results = {}
    reference_similarity = None
    reference_positive_mask = None
    for top_k in ordered_k:
        transport = compute_transport_variant(
            image_features=image_features,
            text_features=text_features,
            logit_scale=1.0,
            image_ids=image_ids,
            species_ids=species_ids,
            cost_space="raw_cosine",
            top_k=top_k,
            ot_eps=ot_eps,
            sinkhorn_iters=sinkhorn_iters,
            solver=solver,
        )
        diagnostics = compute_barycentric_weight_ablation(
            image_features=image_features,
            text_features=text_features,
            plan=transport["plan"],
            support_mask=transport["support_mask"],
            positive_mask=transport["positive_mask"],
            tie_tolerance=tie_tolerance,
        )

        if reference_similarity is None:
            reference_similarity = transport["raw_similarity"]
            reference_positive_mask = transport["positive_mask"]
        else:
            if not torch.equal(transport["positive_mask"], reference_positive_mask):
                raise AssertionError("Changing k changed the candidate/positive pool")
            if not torch.equal(transport["raw_similarity"], reference_similarity):
                raise AssertionError("Changing k changed the raw batch geometry")

        eligible = (~transport["positive_mask"]).sum(dim=1)
        support_size = transport["support_mask"].sum(dim=1)
        support_fraction = support_size.to(torch.float64) / eligible.to(torch.float64)
        reference_error = (
            diagnostics["ot_synthetic_similarity"]
            - transport["synthetic_similarity"]
        ).abs()
        results[top_k] = {
            "transport": transport,
            "diagnostics": diagnostics,
            "eligible_negative_count": eligible,
            "support_fraction": support_fraction,
            "max_abs_ot_similarity_vs_existing_v2": float(
                reference_error.max().item()
            ),
        }
    return results


def paired_bootstrap_uniform_minus_ot(
    ot_harder,
    uniform_harder,
    uniform_minus_ot_similarity,
    *,
    seed=42,
    samples=10000,
    chunk_size=256,
):
    """Observation-level paired bootstrap for uniform-minus-OT effects."""
    if samples < 1 or chunk_size < 1:
        raise ValueError("samples and chunk_size must be positive")
    ot = np.asarray(torch.as_tensor(ot_harder).cpu(), dtype=np.float64).reshape(-1)
    uniform = np.asarray(
        torch.as_tensor(uniform_harder).cpu(), dtype=np.float64
    ).reshape(-1)
    similarity = np.asarray(
        torch.as_tensor(uniform_minus_ot_similarity).cpu(), dtype=np.float64
    ).reshape(-1)
    if not (ot.shape == uniform.shape == similarity.shape) or ot.size == 0:
        raise ValueError("Bootstrap inputs must be non-empty paired observations")

    fraction_differences = np.empty(samples, dtype=np.float64)
    similarity_differences = np.empty(samples, dtype=np.float64)
    generator = np.random.default_rng(seed)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = generator.integers(0, ot.size, size=(stop - start, ot.size))
        fraction_differences[start:stop] = (
            uniform[indices] - ot[indices]
        ).mean(axis=1)
        similarity_differences[start:stop] = similarity[indices].mean(axis=1)

    def interval(values, estimate):
        low, high = np.quantile(values, [0.025, 0.975])
        return {
            "estimate": float(estimate),
            "ci_95_percentile": [float(low), float(high)],
        }

    return {
        "seed": seed,
        "samples": samples,
        "unit": "batch-query observation",
        "interpretation": (
            "Observation-level descriptive paired CI; deterministic batches reuse "
            "underlying holdout examples, so this is not an iid population guarantee."
        ),
        "hardness_fraction_uniform_minus_ot": interval(
            fraction_differences, (uniform - ot).mean()
        ),
        "mean_synthetic_similarity_uniform_minus_ot": interval(
            similarity_differences, similarity.mean()
        ),
    }


def concatenate_diagnostics(batch_diagnostics):
    """Concatenate per-batch observation tensors for one k."""
    if not batch_diagnostics:
        raise ValueError("At least one batch diagnostic is required")
    merged = {
        key: torch.cat([record[key].detach().cpu() for record in batch_diagnostics])
        for key in OBSERVATION_TENSOR_KEYS
    }
    merged["tie_tolerance"] = batch_diagnostics[0]["tie_tolerance"]
    return merged


def summarize_support_breadth_k(
    *,
    top_k,
    batch_diagnostics,
    eligible_counts,
    support_fractions,
    mass_audits,
    reference_errors,
    bootstrap_seed=42,
    bootstrap_samples=10000,
):
    """Build the complete report section for one support breadth."""
    diagnostics = concatenate_diagnostics(batch_diagnostics)
    existing = summarize_barycentric_weight_ablation(
        diagnostics,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=max(1, min(bootstrap_samples, 10)),
    )
    # Replace the generic OT-minus-uniform bootstrap with the direction tested
    # by this ablation, while retaining the existing exact method summaries.
    uniform_minus_ot = -diagnostics["ot_minus_uniform_synthetic_similarity"]
    paired = {
        "tie_tolerance": diagnostics["tie_tolerance"],
        "fraction_uniform_harder_than_ot": float(
            diagnostics["uniform_harder_than_ot"].float().mean().item()
        ),
        "fraction_ot_harder_than_uniform": float(
            diagnostics["ot_harder_than_uniform"].float().mean().item()
        ),
        "fraction_tied_within_tolerance": float(
            diagnostics["tied_within_tolerance"].float().mean().item()
        ),
        "uniform_minus_ot_synthetic_similarity": quantile_summary(uniform_minus_ot),
        "l1_distance_ot_vs_uniform_weights": quantile_summary(
            diagnostics["l1_distance_ot_vs_uniform_weights"]
        ),
        "cosine_between_ot_and_uniform_synthetic": quantile_summary(
            diagnostics["cosine_between_ot_and_uniform_synthetic"]
        ),
        "paired_hardness_2x2": paired_hardness_counts(
            diagnostics["ot_harder_than_hardest_real"],
            diagnostics["uniform_harder_than_hardest_real"],
        ),
        "bootstrap": paired_bootstrap_uniform_minus_ot(
            diagnostics["ot_harder_than_hardest_real"],
            diagnostics["uniform_harder_than_hardest_real"],
            uniform_minus_ot,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
    }

    audit_fields = (
        "total_mass_before_final_mask",
        "total_mass_returned",
        "mass_removed_by_final_mask",
        "fraction_mass_removed_by_final_mask",
        "max_row_marginal_error",
        "mean_row_marginal_error",
        "max_column_marginal_error",
        "mean_column_marginal_error",
    )
    audits = {
        field: quantile_summary([record[field] for record in mass_audits])
        for field in audit_fields
    }
    return {
        "top_k": top_k,
        "eligible_negative_count": quantile_summary(torch.cat(eligible_counts)),
        "support_fraction_of_available_negatives": quantile_summary(
            torch.cat(support_fractions)
        ),
        "ot_weighted": existing["ot_weighted"],
        "uniform_weighted": existing["uniform_weighted"],
        "paired_comparison": paired,
        "delta_hardness_fraction_uniform_minus_ot": (
            existing["uniform_weighted"]
            ["fraction_synthetic_harder_than_hardest_real"]
            - existing["ot_weighted"]
            ["fraction_synthetic_harder_than_hardest_real"]
        ),
        "transport_mass_marginal_audit_across_batches": audits,
        "existing_v2_ot_construction_regression": {
            "helper": "src.clip_geometry_v2_metrics.compute_transport_variant",
            "max_abs_synthetic_similarity_difference": max(reference_errors),
        },
    }


def build_support_breadth_curve(per_k):
    """Return one compact, sorted cross-k row per requested breadth."""
    rows = []
    for top_k in sorted(per_k):
        summary = per_k[top_k]
        paired = summary["paired_comparison"]
        ot = summary["ot_weighted"]
        uniform = summary["uniform_weighted"]
        rows.append(
            {
                "top_k": top_k,
                "mean_support_fraction": summary[
                    "support_fraction_of_available_negatives"
                ]["mean"],
                "ot_fraction_gt_hardest": ot[
                    "fraction_synthetic_harder_than_hardest_real"
                ],
                "uniform_fraction_gt_hardest": uniform[
                    "fraction_synthetic_harder_than_hardest_real"
                ],
                "delta_hardness_fraction_uniform_minus_ot": summary[
                    "delta_hardness_fraction_uniform_minus_ot"
                ],
                "uniform_minus_ot_mean_similarity": paired[
                    "uniform_minus_ot_synthetic_similarity"
                ]["mean"],
                "fraction_uniform_gt_ot": paired[
                    "fraction_uniform_harder_than_ot"
                ],
                "fraction_ot_gt_uniform": paired[
                    "fraction_ot_harder_than_uniform"
                ],
                "ot_normalized_entropy_mean": ot[
                    "weight_normalized_entropy"
                ]["mean"],
                "ot_peak_weight_mean": ot["peak_weight"]["mean"],
                "cosine_ot_uniform_synthetic_mean": paired[
                    "cosine_between_ot_and_uniform_synthetic"
                ]["mean"],
            }
        )
    return rows


def initialize_k_accumulators(k_values):
    """Create per-k containers while preserving a sorted sweep contract."""
    return {
        top_k: defaultdict(list)
        for top_k in sorted(k_values)
    }
