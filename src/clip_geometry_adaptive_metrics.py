"""Geometry-only adaptive-neighborhood metrics for frozen CLIP batches."""

import numpy as np
import torch
import torch.nn.functional as F

from src.clip_barycentric_weight_metrics import (
    construct_normalized_barycenters,
    uniform_weights_from_support,
)
from src.clip_geometry_metrics import quantile_summary
from src.clip_geometry_v2_metrics import _top_k_support


def eligible_negative_similarity_spectrum(raw_similarity, positive_mask):
    """Return eligible negative similarities sorted descending per query."""
    if raw_similarity.ndim != 2 or raw_similarity.shape != positive_mask.shape:
        raise ValueError("raw_similarity and positive_mask must be equal 2-D shapes")
    if positive_mask.dtype != torch.bool:
        raise TypeError("positive_mask must be Boolean")
    eligible_counts = (~positive_mask).sum(dim=1)
    if torch.any(eligible_counts == 0):
        raise ValueError("Every query needs at least one eligible negative")
    if not torch.all(eligible_counts == eligible_counts[0]):
        raise ValueError("This B64 diagnostic requires equal eligible counts per row")
    count = int(eligible_counts[0].item())
    sorted_values = raw_similarity.masked_fill(
        positive_mask, float("-inf")
    ).sort(dim=1, descending=True).values[:, :count]
    return sorted_values, eligible_counts


def select_geometry_boundary_k(
    negative_similarities,
    allowed_k_values,
    tie_tolerance=1e-12,
):
    """Select the largest local rank-boundary gap using geometry alone.

    The API intentionally accepts only eligible negative similarities and the
    allowed support grid.  Synthetic scores, labels, OT state, and downstream
    metrics cannot enter this selector.
    """
    if negative_similarities.ndim != 2:
        raise ValueError("negative_similarities must have shape [queries, negatives]")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    choices = list(allowed_k_values)
    if not choices or choices != sorted(set(choices)):
        raise ValueError("allowed_k_values must be unique, non-empty, and sorted")
    if choices[-1] >= negative_similarities.shape[1]:
        raise ValueError("Every k needs both rank k and rank k+1")

    sorted_values = negative_similarities.sort(dim=1, descending=True).values
    boundary_gaps = torch.stack(
        [sorted_values[:, top_k - 1] - sorted_values[:, top_k] for top_k in choices],
        dim=1,
    )
    maxima = boundary_gaps.max(dim=1, keepdim=True).values
    near_maximum = boundary_gaps >= maxima - tie_tolerance
    # Choices are sorted, so argmax over this Boolean mask implements the
    # deterministic smaller-k tie break.
    selected_choice_indices = near_maximum.to(torch.int64).argmax(dim=1)
    choice_tensor = torch.tensor(
        choices, device=negative_similarities.device, dtype=torch.int64
    )
    selected_k = choice_tensor.index_select(0, selected_choice_indices)
    selected_gap = boundary_gaps.gather(
        1, selected_choice_indices.unsqueeze(1)
    ).squeeze(1)
    return {
        "selected_k": selected_k,
        "selected_boundary_gap": selected_gap,
        "boundary_gaps": boundary_gaps,
        "sorted_negative_similarities": sorted_values,
        "allowed_k_values": choices,
        "tie_tolerance": tie_tolerance,
    }


def select_oracle_k(
    synthetic_similarities,
    allowed_k_values,
    tie_tolerance=1e-6,
):
    """Evaluation-only maximum-synthetic selector with smaller-k tie break."""
    choices = list(allowed_k_values)
    if synthetic_similarities.ndim != 2:
        raise ValueError("synthetic_similarities must have shape [queries, choices]")
    if synthetic_similarities.shape[1] != len(choices):
        raise ValueError("One synthetic-similarity column is required per k")
    if choices != sorted(set(choices)) or not choices:
        raise ValueError("allowed_k_values must be unique, non-empty, and sorted")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    maxima = synthetic_similarities.max(dim=1, keepdim=True).values
    near_maximum = synthetic_similarities >= maxima - tie_tolerance
    selected_choice_indices = near_maximum.to(torch.int64).argmax(dim=1)
    choices_tensor = torch.tensor(
        choices, device=synthetic_similarities.device, dtype=torch.int64
    )
    return {
        "selected_k": choices_tensor.index_select(0, selected_choice_indices),
        "selected_choice_indices": selected_choice_indices,
        "selected_similarity": synthetic_similarities.gather(
            1, selected_choice_indices.unsqueeze(1)
        ).squeeze(1),
        "tie_tolerance": tie_tolerance,
        "oracle_is_evaluation_only": True,
    }


def gather_choice_rows(stacked_values, choice_indices):
    """Gather one precomputed fixed-k value/vector per query."""
    if stacked_values.ndim < 2 or stacked_values.shape[0] != choice_indices.shape[0]:
        raise ValueError("stacked_values and choice_indices have incompatible rows")
    row_indices = torch.arange(stacked_values.shape[0], device=stacked_values.device)
    return stacked_values[row_indices, choice_indices]


def compute_geometry_adaptive_batch(
    *,
    image_features,
    text_features,
    image_ids,
    allowed_k_values=(2, 4, 8, 16, 32),
    selector_tie_tolerance=1e-12,
    comparison_tie_tolerance=1e-6,
):
    """Compute fixed-uniform, geometry-adaptive, and oracle controls on one batch."""
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError("Image and text features must have the same [B, d] shape")
    if len(image_ids) != image_features.shape[0]:
        raise ValueError("image_ids must align with the batch")
    choices = list(allowed_k_values)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    raw_similarity = text_features @ image_features.T
    ids = torch.as_tensor(image_ids, device=raw_similarity.device)
    positive_mask = ids.unsqueeze(1) == ids.unsqueeze(0)
    negative_spectrum, eligible_counts = eligible_negative_similarity_spectrum(
        raw_similarity, positive_mask
    )
    geometry = select_geometry_boundary_k(
        negative_spectrum,
        choices,
        tie_tolerance=selector_tie_tolerance,
    )

    fixed_supports = []
    fixed_weights = []
    fixed_synthetic_features = []
    fixed_synthetic_similarities = []
    for top_k in choices:
        _, support = _top_k_support(raw_similarity, positive_mask, top_k)
        weights = uniform_weights_from_support(support).to(
            device=image_features.device, dtype=image_features.dtype
        )
        synthetic = construct_normalized_barycenters(weights, image_features)
        similarity = (text_features * synthetic).sum(dim=1)
        fixed_supports.append(support)
        fixed_weights.append(weights)
        fixed_synthetic_features.append(synthetic)
        fixed_synthetic_similarities.append(similarity)

    support_tensor = torch.stack(fixed_supports, dim=1)
    weights_tensor = torch.stack(fixed_weights, dim=1)
    synthetic_tensor = torch.stack(fixed_synthetic_features, dim=1)
    similarity_tensor = torch.stack(fixed_synthetic_similarities, dim=1)
    choice_tensor = torch.tensor(choices, device=raw_similarity.device)
    geometry_indices = (
        geometry["selected_k"].unsqueeze(1) == choice_tensor.unsqueeze(0)
    ).to(torch.int64).argmax(dim=1)
    geometry_synthetic = gather_choice_rows(synthetic_tensor, geometry_indices)
    geometry_similarity = gather_choice_rows(similarity_tensor, geometry_indices)
    geometry_support = gather_choice_rows(support_tensor, geometry_indices)
    geometry_weights = gather_choice_rows(weights_tensor, geometry_indices)

    oracle = select_oracle_k(
        similarity_tensor,
        choices,
        tie_tolerance=comparison_tie_tolerance,
    )
    oracle_synthetic = gather_choice_rows(
        synthetic_tensor, oracle["selected_choice_indices"]
    )
    oracle_support = gather_choice_rows(
        support_tensor, oracle["selected_choice_indices"]
    )

    positive_similarity = raw_similarity.masked_fill(
        ~positive_mask, float("-inf")
    ).max(dim=1).values
    hardest_real_similarity = negative_spectrum[:, 0]
    fixed_by_k = {
        top_k: {
            "support_mask": fixed_supports[index],
            "weights": fixed_weights[index],
            "synthetic_features": fixed_synthetic_features[index],
            "synthetic_similarity": fixed_synthetic_similarities[index],
        }
        for index, top_k in enumerate(choices)
    }
    return {
        "raw_similarity": raw_similarity,
        "positive_mask": positive_mask,
        "eligible_negative_count": eligible_counts,
        "negative_similarity_spectrum": negative_spectrum,
        "boundary_gaps": geometry["boundary_gaps"],
        "selected_boundary_gap": geometry["selected_boundary_gap"],
        "geometry_selected_k": geometry["selected_k"],
        "geometry_choice_indices": geometry_indices,
        "oracle_selected_k": oracle["selected_k"],
        "oracle_choice_indices": oracle["selected_choice_indices"],
        "positive_similarity": positive_similarity,
        "hardest_real_similarity": hardest_real_similarity,
        "fixed_by_k": fixed_by_k,
        "fixed_similarity_tensor": similarity_tensor,
        "geometry_support_mask": geometry_support,
        "geometry_weights": geometry_weights,
        "geometry_synthetic_features": geometry_synthetic,
        "geometry_synthetic_similarity": geometry_similarity,
        "oracle_support_mask": oracle_support,
        "oracle_synthetic_features": oracle_synthetic,
        "oracle_synthetic_similarity": oracle["selected_similarity"],
        "oracle_is_evaluation_only": True,
        "allowed_k_values": choices,
        "selector_tie_tolerance": selector_tie_tolerance,
        "comparison_tie_tolerance": comparison_tie_tolerance,
    }


def summarize_adaptive_method(similarity, positive, hardest):
    """Summarize one synthetic method against positive and hardest real."""
    vs_hardest = similarity - hardest
    vs_positive = similarity - positive
    return {
        "synthetic_similarity": quantile_summary(similarity),
        "fraction_synthetic_harder_than_hardest_real": float(
            (vs_hardest > 0).to(torch.float32).mean().item()
        ),
        "synthetic_minus_hardest_real": quantile_summary(vs_hardest),
        "fraction_synthetic_harder_than_paired_positive": float(
            (vs_positive > 0).to(torch.float32).mean().item()
        ),
        "synthetic_minus_paired_positive": quantile_summary(vs_positive),
    }


def paired_bootstrap_adaptive(
    geometry_minus_k8,
    oracle_minus_geometry,
    geometry_harder,
    k8_harder,
    *,
    seed=42,
    samples=10000,
    chunk_size=256,
):
    """Observation-level paired bootstrap for the adaptive comparisons."""
    arrays = [
        np.asarray(torch.as_tensor(value).cpu(), dtype=np.float64).reshape(-1)
        for value in (
            geometry_minus_k8,
            oracle_minus_geometry,
            geometry_harder,
            k8_harder,
        )
    ]
    if not arrays[0].size or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("Bootstrap inputs must be non-empty paired observations")
    if samples < 1 or chunk_size < 1:
        raise ValueError("samples and chunk_size must be positive")
    boot = [np.empty(samples, dtype=np.float64) for _ in range(3)]
    generator = np.random.default_rng(seed)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = generator.integers(
            0, arrays[0].size, size=(stop - start, arrays[0].size)
        )
        boot[0][start:stop] = arrays[0][indices].mean(axis=1)
        boot[1][start:stop] = arrays[1][indices].mean(axis=1)
        boot[2][start:stop] = (
            arrays[2][indices] - arrays[3][indices]
        ).mean(axis=1)

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
        "mean_geometry_minus_fixed_k8_similarity": interval(
            boot[0], arrays[0].mean()
        ),
        "mean_oracle_minus_geometry_similarity": interval(
            boot[1], arrays[1].mean()
        ),
        "harder_than_hardest_fraction_geometry_minus_fixed_k8": interval(
            boot[2], (arrays[2] - arrays[3]).mean()
        ),
    }


def confusion_matrix(selected_rows, selected_columns, allowed_k_values):
    """Return a labeled geometry-row/oracle-column count matrix."""
    choices = list(allowed_k_values)
    rows = torch.as_tensor(selected_rows).flatten()
    columns = torch.as_tensor(selected_columns).flatten()
    if rows.shape != columns.shape:
        raise ValueError("Confusion inputs must be paired")
    matrix = []
    for row_k in choices:
        matrix.append(
            [int(((rows == row_k) & (columns == col_k)).sum()) for col_k in choices]
        )
    return {
        "row_labels_geometry_k": choices,
        "column_labels_oracle_k": choices,
        "counts": matrix,
    }


def summarize_geometry_adaptive(
    rows,
    *,
    allowed_k_values,
    tie_tolerance=1e-6,
    bootstrap_seed=42,
    bootstrap_samples=10000,
):
    """Create the JSON-ready aggregate report from paired observation rows."""
    if not rows:
        raise ValueError("At least one observation is required")

    def tensor(field, dtype=torch.float64):
        return torch.tensor([row[field] for row in rows], dtype=dtype)

    positive = tensor("positive_similarity")
    hardest = tensor("hardest_real_similarity")
    geometry_similarity = tensor("geometry_synthetic_similarity")
    oracle_similarity = tensor("oracle_synthetic_similarity")
    fixed = {
        top_k: tensor(f"uniform_k{top_k}_similarity")
        for top_k in allowed_k_values
    }
    geometry_k = tensor("geometry_selected_k", dtype=torch.int64)
    oracle_k = tensor("oracle_selected_k", dtype=torch.int64)
    geometry_minus_k8 = geometry_similarity - fixed[8]
    oracle_minus_geometry = oracle_similarity - geometry_similarity
    geometry_harder = geometry_similarity > hardest
    k8_harder = fixed[8] > hardest

    selection_counts = {
        f"fraction_k_{top_k}": float(
            (geometry_k == top_k).to(torch.float64).mean().item()
        )
        for top_k in allowed_k_values
    }
    boundary_by_selection = {
        str(top_k): quantile_summary(
            tensor("selected_boundary_gap")[geometry_k == top_k]
        )
        for top_k in allowed_k_values
    }
    exact_agreement = float((geometry_k == oracle_k).to(torch.float64).mean().item())
    geometry_oracle_gap = oracle_minus_geometry
    k8_comparison_delta = geometry_minus_k8
    oracle_gain = float((oracle_similarity.mean() - fixed[8].mean()).item())
    geometry_gain = float((geometry_similarity.mean() - fixed[8].mean()).item())
    recovered = geometry_gain / oracle_gain if oracle_gain > 0 else None

    return {
        "geometry_gap_adaptive": {
            "selected_neighborhood_distribution": selection_counts,
            "mean_selected_k": float(geometry_k.to(torch.float64).mean().item()),
            "median_selected_k": float(geometry_k.to(torch.float64).median().item()),
            "mean_selected_support_fraction": float(
                tensor("geometry_selected_support_fraction").mean().item()
            ),
            **summarize_adaptive_method(
                geometry_similarity, positive, hardest
            ),
            "selected_boundary_gap": quantile_summary(
                tensor("selected_boundary_gap")
            ),
            "selected_boundary_gap_conditioned_on_k": boundary_by_selection,
        },
        "fixed_uniform_controls": {
            str(top_k): summarize_adaptive_method(
                fixed[top_k], positive, hardest
            )
            for top_k in allowed_k_values
        },
        "oracle_evaluation_only": {
            "oracle_is_evaluation_only": True,
            "selected_neighborhood_distribution": {
                f"fraction_k_{top_k}": float(
                    (oracle_k == top_k).to(torch.float64).mean().item()
                )
                for top_k in allowed_k_values
            },
            **summarize_adaptive_method(oracle_similarity, positive, hardest),
        },
        "geometry_vs_oracle": {
            "exact_k_agreement_geometry_vs_oracle": exact_agreement,
            "confusion_matrix": confusion_matrix(
                geometry_k, oracle_k, allowed_k_values
            ),
            "oracle_minus_geometry_synthetic_similarity": quantile_summary(
                geometry_oracle_gap
            ),
            "fraction_geometry_matches_oracle_synthetic_within_1e-6": float(
                (geometry_oracle_gap <= 1e-6).to(torch.float64).mean().item()
            ),
            "fraction_geometry_within_0.001_of_oracle": float(
                (geometry_oracle_gap <= 0.001).to(torch.float64).mean().item()
            ),
            "fraction_geometry_within_0.005_of_oracle": float(
                (geometry_oracle_gap <= 0.005).to(torch.float64).mean().item()
            ),
        },
        "geometry_vs_fixed_k8": {
            "fraction_geometry_harder_than_fixed_k8": float(
                (k8_comparison_delta > tie_tolerance).to(torch.float64).mean().item()
            ),
            "fraction_fixed_k8_harder_than_geometry": float(
                (k8_comparison_delta < -tie_tolerance).to(torch.float64).mean().item()
            ),
            "fraction_tied_within_1e-6": float(
                (k8_comparison_delta.abs() <= tie_tolerance)
                .to(torch.float64).mean().item()
            ),
            "geometry_minus_fixed_k8_synthetic_similarity": quantile_summary(
                k8_comparison_delta
            ),
            "geometry_hardness": summarize_adaptive_method(
                geometry_similarity, positive, hardest
            ),
            "fixed_k8_hardness": summarize_adaptive_method(
                fixed[8], positive, hardest
            ),
        },
        "adaptive_headroom": {
            "oracle_gain_over_fixed_k8_mean_similarity": oracle_gain,
            "geometry_gain_over_fixed_k8_mean_similarity": geometry_gain,
            "fraction_of_oracle_gain_recovered": recovered,
            "null_explanation": (
                None
                if recovered is not None
                else "Oracle mean similarity did not exceed fixed k=8."
            ),
        },
        "boundary_gap_distributions": {
            f"gap_after_k{top_k}": quantile_summary(
                tensor(f"gap_after_k{top_k}")
            )
            for top_k in allowed_k_values
        },
        "bootstrap": paired_bootstrap_adaptive(
            geometry_minus_k8,
            oracle_minus_geometry,
            geometry_harder,
            k8_harder,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
    }
