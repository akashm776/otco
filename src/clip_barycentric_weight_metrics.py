"""Paired OT-vs-uniform barycentric diagnostics on one fixed support.

This module does not solve transport or select candidates.  It consumes the
plan and support produced by the existing frozen CLIP V2 diagnostic so the
only experimental variable is the weighting within that support.
"""

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from src.clip_geometry_metrics import quantile_summary


def row_normalize_plan(plan):
    """Normalize each returned transport row without changing its support."""
    mass = plan.sum(dim=1, keepdim=True)
    if torch.any(mass <= 0):
        raise ValueError("Transport plan contains a row with no returned mass")
    return plan / mass


def uniform_weights_from_support(support_mask):
    """Assign exactly 1/k to each retained support edge and zero elsewhere."""
    if support_mask.ndim != 2 or support_mask.dtype != torch.bool:
        raise TypeError("support_mask must be a two-dimensional Boolean tensor")
    support_size = support_mask.sum(dim=1, keepdim=True)
    if torch.any(support_size == 0):
        raise ValueError("Every query needs at least one supported candidate")
    return support_mask.to(torch.float32) / support_size


def construct_normalized_barycenters(weights, image_features):
    """Construct ``normalize(weights @ image_features)`` row by row."""
    if weights.ndim != 2 or image_features.ndim != 2:
        raise ValueError("weights and image_features must be two-dimensional")
    if weights.shape[1] != image_features.shape[0]:
        raise ValueError("Weight columns must match the image feature count")
    return F.normalize(weights @ F.normalize(image_features, dim=-1), dim=-1)


def _weight_entropy(weights):
    return -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)


def compute_barycentric_weight_ablation(
    *,
    image_features,
    text_features,
    plan,
    support_mask,
    positive_mask,
    tie_tolerance=1e-6,
):
    """Compare OT and uniform barycenters on the identical retained support."""
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError("Image and text features must have the same [N, d] shape")
    count = image_features.shape[0]
    expected_shape = (count, count)
    if plan.shape != expected_shape:
        raise ValueError("plan must have shape [N, N]")
    if support_mask.shape != expected_shape or positive_mask.shape != expected_shape:
        raise ValueError("support_mask and positive_mask must have shape [N, N]")
    if support_mask.dtype != torch.bool or positive_mask.dtype != torch.bool:
        raise TypeError("support_mask and positive_mask must be Boolean")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    if torch.any(support_mask & positive_mask):
        raise ValueError("The retained support must exclude paired positives")
    if torch.any(plan.masked_select(~support_mask) != 0):
        raise ValueError("Transport plan has returned mass outside its support")

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    raw_similarity = text_features @ image_features.T

    ot_weights = row_normalize_plan(plan)
    uniform_weights = uniform_weights_from_support(support_mask).to(
        device=plan.device, dtype=plan.dtype
    )
    if not torch.equal(ot_weights != 0, support_mask):
        raise ValueError("Returned OT weights do not occupy the complete support")
    if not torch.equal(uniform_weights != 0, support_mask):
        raise AssertionError("Uniform weights changed the retained support")

    ot_synthetic = construct_normalized_barycenters(ot_weights, image_features)
    uniform_synthetic = construct_normalized_barycenters(
        uniform_weights, image_features
    )
    ot_similarity = (text_features * ot_synthetic).sum(dim=1)
    uniform_similarity = (text_features * uniform_synthetic).sum(dim=1)

    positive_similarity = raw_similarity.masked_fill(
        ~positive_mask, float("-inf")
    ).max(dim=1).values
    supported_similarity = raw_similarity.masked_fill(
        ~support_mask, float("-inf")
    )
    hardest_real_similarity, hardest_real_indices = supported_similarity.max(dim=1)

    ot_vs_hardest = ot_similarity - hardest_real_similarity
    uniform_vs_hardest = uniform_similarity - hardest_real_similarity
    ot_vs_positive = ot_similarity - positive_similarity
    uniform_vs_positive = uniform_similarity - positive_similarity
    ot_minus_uniform = ot_similarity - uniform_similarity

    support_size = support_mask.sum(dim=1).to(plan.dtype)
    ot_entropy = _weight_entropy(ot_weights)
    uniform_entropy = _weight_entropy(uniform_weights)
    log_support_size = support_size.log().clamp_min(1e-12)

    return {
        "support_mask": support_mask,
        "ot_support_mask": ot_weights != 0,
        "uniform_support_mask": uniform_weights != 0,
        "support_size": support_size,
        "ot_weights": ot_weights,
        "uniform_weights": uniform_weights,
        "positive_similarity": positive_similarity,
        "hardest_real_indices": hardest_real_indices,
        "hardest_real_similarity": hardest_real_similarity,
        "ot_synthetic_features": ot_synthetic,
        "uniform_synthetic_features": uniform_synthetic,
        "ot_synthetic_similarity": ot_similarity,
        "uniform_synthetic_similarity": uniform_similarity,
        "ot_vs_hardest_delta": ot_vs_hardest,
        "uniform_vs_hardest_delta": uniform_vs_hardest,
        "ot_vs_positive_delta": ot_vs_positive,
        "uniform_vs_positive_delta": uniform_vs_positive,
        "ot_minus_uniform_synthetic_similarity": ot_minus_uniform,
        "ot_harder_than_hardest_real": ot_vs_hardest > 0,
        "uniform_harder_than_hardest_real": uniform_vs_hardest > 0,
        "ot_harder_than_uniform": ot_minus_uniform > tie_tolerance,
        "uniform_harder_than_ot": ot_minus_uniform < -tie_tolerance,
        "tied_within_tolerance": ot_minus_uniform.abs() <= tie_tolerance,
        "ot_weight_entropy": ot_entropy,
        "ot_weight_normalized_entropy": ot_entropy / log_support_size,
        "ot_peak_weight": ot_weights.max(dim=1).values,
        "uniform_weight_entropy": uniform_entropy,
        "uniform_weight_normalized_entropy": uniform_entropy / log_support_size,
        "uniform_peak_weight": uniform_weights.max(dim=1).values,
        "l1_distance_ot_vs_uniform_weights": (
            ot_weights - uniform_weights
        ).abs().sum(dim=1),
        "cosine_between_ot_and_uniform_synthetic": (
            ot_synthetic * uniform_synthetic
        ).sum(dim=1),
        "tie_tolerance": tie_tolerance,
    }


def paired_hardness_counts(ot_harder, uniform_harder):
    """Return the paired 2x2 hardness table."""
    ot_harder = torch.as_tensor(ot_harder, dtype=torch.bool).flatten()
    uniform_harder = torch.as_tensor(uniform_harder, dtype=torch.bool).flatten()
    if ot_harder.shape != uniform_harder.shape:
        raise ValueError("Paired hardness arrays must have the same shape")
    return {
        "both_harder_than_hardest_real": int((ot_harder & uniform_harder).sum()),
        "ot_only_harder_than_hardest_real": int(
            (ot_harder & ~uniform_harder).sum()
        ),
        "uniform_only_harder_than_hardest_real": int(
            (~ot_harder & uniform_harder).sum()
        ),
        "neither_harder_than_hardest_real": int(
            (~ot_harder & ~uniform_harder).sum()
        ),
    }


def deterministic_paired_bootstrap(
    ot_harder,
    uniform_harder,
    ot_minus_uniform_similarity,
    *,
    seed=42,
    samples=10000,
    chunk_size=256,
):
    """Bootstrap paired query differences without additional dependencies."""
    if samples < 1 or chunk_size < 1:
        raise ValueError("samples and chunk_size must be positive")
    ot = np.asarray(torch.as_tensor(ot_harder).cpu(), dtype=np.float64).reshape(-1)
    uniform = np.asarray(
        torch.as_tensor(uniform_harder).cpu(), dtype=np.float64
    ).reshape(-1)
    similarity_delta = np.asarray(
        torch.as_tensor(ot_minus_uniform_similarity).cpu(), dtype=np.float64
    ).reshape(-1)
    if not (ot.shape == uniform.shape == similarity_delta.shape) or ot.size == 0:
        raise ValueError("Bootstrap inputs must be non-empty paired arrays")

    fraction_differences = np.empty(samples, dtype=np.float64)
    mean_similarity_differences = np.empty(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, ot.size, size=(stop - start, ot.size))
        fraction_differences[start:stop] = (ot[indices] - uniform[indices]).mean(1)
        mean_similarity_differences[start:stop] = similarity_delta[indices].mean(1)

    def interval(values, estimate):
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {
            "estimate": float(estimate),
            "ci_95_percentile": [float(lower), float(upper)],
        }

    return {
        "seed": seed,
        "samples": samples,
        "fraction_harder_difference_ot_minus_uniform": interval(
            fraction_differences, (ot - uniform).mean()
        ),
        "mean_synthetic_similarity_difference_ot_minus_uniform": interval(
            mean_similarity_differences, similarity_delta.mean()
        ),
    }


def summarize_barycentric_weight_ablation(
    diagnostics, *, bootstrap_seed=42, bootstrap_samples=10000
):
    """Create JSON-ready OT, uniform, and paired report sections."""
    ot_harder = diagnostics["ot_harder_than_hardest_real"]
    uniform_harder = diagnostics["uniform_harder_than_hardest_real"]
    count = int(ot_harder.numel())

    def method_summary(prefix):
        similarity = diagnostics[f"{prefix}_synthetic_similarity"]
        vs_hardest = diagnostics[f"{prefix}_vs_hardest_delta"]
        vs_positive = diagnostics[f"{prefix}_vs_positive_delta"]
        harder = diagnostics[f"{prefix}_harder_than_hardest_real"]
        return {
            "fraction_synthetic_harder_than_hardest_real": float(
                harder.float().mean().item()
            ),
            "fraction_synthetic_harder_than_paired_positive": float(
                (vs_positive > 0).float().mean().item()
            ),
            "synthetic_similarity": quantile_summary(similarity.detach().cpu()),
            "synthetic_minus_hardest_real": quantile_summary(
                vs_hardest.detach().cpu()
            ),
            "synthetic_minus_paired_positive": quantile_summary(
                vs_positive.detach().cpu()
            ),
            "weight_entropy": quantile_summary(
                diagnostics[f"{prefix}_weight_entropy"].detach().cpu()
            ),
            "weight_normalized_entropy": quantile_summary(
                diagnostics[f"{prefix}_weight_normalized_entropy"].detach().cpu()
            ),
            "peak_weight": quantile_summary(
                diagnostics[f"{prefix}_peak_weight"].detach().cpu()
            ),
        }

    outcome_counts = Counter(
        "ot_harder"
        if value > diagnostics["tie_tolerance"]
        else "uniform_harder"
        if value < -diagnostics["tie_tolerance"]
        else "tied"
        for value in diagnostics[
            "ot_minus_uniform_synthetic_similarity"
        ].detach().cpu().tolist()
    )
    paired = {
        "tie_tolerance": diagnostics["tie_tolerance"],
        "fraction_ot_harder_than_uniform": outcome_counts["ot_harder"] / count,
        "fraction_uniform_harder_than_ot": outcome_counts["uniform_harder"] / count,
        "fraction_tied_within_tolerance": outcome_counts["tied"] / count,
        "ot_minus_uniform_synthetic_similarity": quantile_summary(
            diagnostics["ot_minus_uniform_synthetic_similarity"].detach().cpu()
        ),
        "l1_distance_ot_vs_uniform_weights": quantile_summary(
            diagnostics["l1_distance_ot_vs_uniform_weights"].detach().cpu()
        ),
        "cosine_between_ot_and_uniform_synthetic": quantile_summary(
            diagnostics["cosine_between_ot_and_uniform_synthetic"].detach().cpu()
        ),
        "paired_hardness_2x2": paired_hardness_counts(ot_harder, uniform_harder),
        "bootstrap": deterministic_paired_bootstrap(
            ot_harder,
            uniform_harder,
            diagnostics["ot_minus_uniform_synthetic_similarity"],
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
    }
    return {
        "ot_weighted": method_summary("ot"),
        "uniform_weighted": method_summary("uniform"),
        "paired_comparison": paired,
    }
