"""Metrics for the frozen U8-vs-hardest-real embedding-gradient diagnostic."""

import math

import numpy as np
import torch


def cosine_alignment(left, right, eps=1e-12):
    """Cosine between arbitrary equally shaped tensors, flattened."""
    left = torch.as_tensor(left).reshape(-1)
    right = torch.as_tensor(right).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("Gradient tensors must have identical flattened shapes")
    denominator = left.norm() * right.norm()
    if denominator <= eps:
        raise ValueError("Gradient cosine is undefined for a zero-norm tensor")
    return torch.dot(left, right) / denominator


def gradient_footprint(image_gradient, eps=1e-12):
    """Summarize how image-gradient norm is distributed across batch images."""
    if image_gradient.ndim != 2:
        raise ValueError("image_gradient must have shape [B, D]")
    norms = image_gradient.norm(dim=1)
    total = norms.sum()
    if total <= eps:
        raise ValueError("Gradient footprint is undefined for zero total norm")
    shares = norms / total
    entropy = -(shares * shares.clamp_min(eps).log()).sum()
    batch_size = image_gradient.shape[0]
    sorted_shares = shares.sort(descending=True).values

    def top_share(count):
        return sorted_shares[: min(count, batch_size)].sum()

    return {
        "image_norms": norms,
        "image_shares": shares,
        "effective_support": 1.0 / shares.square().sum(),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(batch_size),
        "largest_share": top_share(1),
        "top2_share": top_share(2),
        "top4_share": top_share(4),
        "top8_share": top_share(8),
    }


def gradient_share_on_indices(image_gradient, indices, eps=1e-12):
    """Fraction of total per-image gradient norm assigned to selected indices."""
    norms = image_gradient.norm(dim=1)
    total = norms.sum()
    if total <= eps:
        raise ValueError("Gradient share is undefined for zero total norm")
    indices = torch.as_tensor(indices, device=norms.device, dtype=torch.long)
    return norms.index_select(0, indices).sum() / total


def margin_directional_change(margin_gradient, auxiliary_gradient, eps=1e-12):
    """First-order margin change under a unit-norm auxiliary descent step."""
    margin_gradient = torch.as_tensor(margin_gradient).reshape(-1)
    auxiliary_gradient = torch.as_tensor(auxiliary_gradient).reshape(-1)
    if margin_gradient.shape != auxiliary_gradient.shape:
        raise ValueError("Margin and auxiliary gradients must have equal shapes")
    norm = auxiliary_gradient.norm()
    if norm <= eps:
        raise ValueError("Auxiliary descent direction has zero norm")
    return -torch.dot(margin_gradient, auxiliary_gradient / norm)


def distribution_summary(values):
    """Stable JSON-ready distribution summary."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty distribution")
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "max": float(array.max()),
    }


def alignment_summary(values):
    """Distribution plus the pre-registered cosine-alignment buckets."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    result = distribution_summary(array)
    result.update(
        {
            "fraction_lt_0": float((array < 0).mean()),
            "fraction_lt_0_25": float((array < 0.25).mean()),
            "fraction_lt_0_50": float((array < 0.50).mean()),
            "fraction_gt_0_90": float((array > 0.90).mean()),
        }
    )
    return result


def _average_ranks(values):
    """Return one-based average ranks with deterministic handling of ties."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def pearson_correlation(left, right):
    """Pearson correlation, returning null for a constant input."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Correlation inputs must be paired and contain >=2 values")
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def spearman_correlation(left, right):
    """Spearman correlation using average ranks and no SciPy dependency."""
    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def correlation_pair(left, right):
    return {
        "pearson": pearson_correlation(left, right),
        "spearman": spearman_correlation(left, right),
    }
