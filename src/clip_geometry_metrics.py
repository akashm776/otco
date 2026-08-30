"""Pure metrics for deciding whether frozen CLIP geometry merits an OT trial.

This module contains no dataset or model-loading code.  The functions operate
on already normalized embeddings, which makes the diagnostic reusable with
CLIP variants and with the repository's historical encoder backend.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TransportPlan:
    plan: torch.Tensor
    support_mask: torch.Tensor


def build_sparse_transport_plan(
    logits,
    positive_mask,
    *,
    top_k=32,
    ot_eps=0.7,
    sinkhorn_iters=30,
):
    """Build the same top-k, logit-cost Sinkhorn plan used by OT-Mix.

    ``positive_mask`` can exclude one or several valid matches per query.  It
    is intentionally explicit rather than assuming that positives are on the
    diagonal, so this function also works with future external candidate pools.
    """
    if logits.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("logits and positive_mask must be equally shaped 2-D tensors")
    if positive_mask.dtype != torch.bool:
        raise TypeError("positive_mask must be Boolean")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if ot_eps <= 0:
        raise ValueError("ot_eps must be positive")
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be positive")

    num_queries, num_candidates = logits.shape
    available = (~positive_mask).sum(dim=1)
    if torch.any(available == 0):
        raise ValueError("Every query needs at least one non-positive candidate")
    k = min(top_k, int(available.min().item()))

    masked_logits = logits.masked_fill(positive_mask, float("-inf"))
    topk_indices = torch.topk(masked_logits, k=k, dim=1).indices
    support_mask = torch.zeros_like(positive_mask)
    support_mask.scatter_(1, topk_indices, True)
    support_mask &= ~positive_mask

    # This deliberately mirrors SoftmaxMixLoss: high scaled similarity is low
    # cost, and the same small dense floor stabilizes Sinkhorn before the final
    # sparse support is restored.
    max_logit = masked_logits.max()
    cost = (max_logit - logits).clamp_min(0.0)
    kernel = torch.exp(-cost / ot_eps) * support_mask.to(logits.dtype)
    kernel = kernel.clamp_min(1e-12)

    query_marginal = torch.full(
        (num_queries,), 1.0 / num_queries, device=logits.device, dtype=logits.dtype
    )
    candidate_marginal = torch.full(
        (num_candidates,),
        1.0 / num_candidates,
        device=logits.device,
        dtype=logits.dtype,
    )
    u = torch.ones_like(query_marginal)
    v = torch.ones_like(candidate_marginal)
    for _ in range(sinkhorn_iters):
        u = query_marginal / (kernel @ v + 1e-8)
        v = candidate_marginal / (kernel.T @ u + 1e-8)

    plan = (u.unsqueeze(1) * kernel) * v.unsqueeze(0)
    plan *= support_mask.to(plan.dtype)
    return TransportPlan(plan=plan, support_mask=support_mask)


def _safe_row_distribution(plan):
    row_mass = plan.sum(dim=1, keepdim=True)
    if torch.any(row_mass <= 0):
        raise ValueError("Transport plan contains a row with no retained mass")
    return plan / row_mass


def transport_marginal_errors(plan):
    """Return residuals after sparse support is restored."""
    rows, columns = plan.shape
    row_target = 1.0 / rows
    column_target = 1.0 / columns
    return {
        "total_mass": float(plan.sum().item()),
        "max_row_error": float((plan.sum(1) - row_target).abs().max().item()),
        "mean_row_error": float((plan.sum(1) - row_target).abs().mean().item()),
        "max_column_error": float(
            (plan.sum(0) - column_target).abs().max().item()
        ),
        "mean_column_error": float(
            (plan.sum(0) - column_target).abs().mean().item()
        ),
    }


def compute_retrieval_metrics(raw_similarity, positive_mask, ks=(1, 5, 10)):
    """Bidirectional recall where any explicitly marked match is correct."""
    if raw_similarity.shape != positive_mask.shape:
        raise ValueError("raw_similarity and positive_mask shapes must match")

    def direction(scores, matches):
        order = scores.argsort(dim=1, descending=True)
        ranked_matches = matches.gather(1, order)
        return {
            f"r_at_{k}": float(
                ranked_matches[:, : min(k, scores.shape[1])]
                .any(dim=1)
                .float()
                .mean()
                .item()
            )
            for k in ks
        }

    return {
        "text_to_image": direction(raw_similarity, positive_mask),
        "image_to_text": direction(raw_similarity.T, positive_mask.T),
    }


def quantile_summary(values):
    """JSON-ready distribution summary for a one-dimensional tensor."""
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        finite, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=finite.dtype)
    )
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "q05": float(quantiles[0].item()),
        "q25": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "q75": float(quantiles[3].item()),
        "q95": float(quantiles[4].item()),
        "max": float(finite.max().item()),
    }


def classify_gate_overlay(
    entropy,
    gap,
    *,
    entropy_threshold=3.0,
    gap_suppress_easy=0.10,
    gap_downweight_hard=-0.07,
):
    """Apply historical thresholds as labels only; this does not gate a loss."""
    labels = []
    for entropy_i, gap_i in zip(entropy.tolist(), gap.tolist()):
        if entropy_i > entropy_threshold:
            labels.append("diffuse")
        elif gap_i > gap_suppress_easy:
            labels.append("too_easy")
        elif gap_i < gap_downweight_hard:
            labels.append("too_hard")
        else:
            labels.append("useful")
    return labels


def compute_geometry_diagnostics(
    *,
    image_features,
    text_features,
    logit_scale,
    image_ids,
    species_ids,
    top_k=32,
    ot_eps=0.7,
    sinkhorn_iters=30,
    historical_thresholds=None,
):
    """Compute per-query evidence about the usefulness of OT on frozen geometry."""
    if image_features.shape != text_features.shape:
        raise ValueError("This canonical diagnostic requires one text per image")
    if image_features.ndim != 2:
        raise ValueError("Embeddings must have shape [N, d]")
    count = image_features.shape[0]
    if len(image_ids) != count or len(species_ids) != count:
        raise ValueError("Metadata length must match the number of embedding pairs")

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    raw_similarity = text_features @ image_features.T
    logits = raw_similarity * torch.as_tensor(
        logit_scale, device=raw_similarity.device, dtype=raw_similarity.dtype
    )

    image_id_tensor = torch.as_tensor(image_ids, device=raw_similarity.device)
    positive_mask = image_id_tensor.unsqueeze(1) == image_id_tensor.unsqueeze(0)
    transport = build_sparse_transport_plan(
        logits,
        positive_mask,
        top_k=top_k,
        ot_eps=ot_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    row_distribution = _safe_row_distribution(transport.plan)
    selected_indices = row_distribution.argmax(dim=1)
    batch_indices = torch.arange(count, device=raw_similarity.device)

    positive_similarity = (
        raw_similarity.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    )
    selected_similarity = raw_similarity[batch_indices, selected_indices]
    positive_selected_gap = positive_similarity - selected_similarity

    negative_similarity = raw_similarity.masked_fill(positive_mask, float("-inf"))
    selected_rank = (
        (negative_similarity > selected_similarity.unsqueeze(1)).sum(dim=1).float()
        + 1.0
    )
    negative_count = (~positive_mask).sum(dim=1).float()
    selected_hardness_percentile = 1.0 - (selected_rank - 1.0) / negative_count.clamp_min(1)

    entropy = -(row_distribution * row_distribution.clamp_min(1e-12).log()).sum(1)
    support_size = transport.support_mask.sum(1).clamp_min(1).float()
    normalized_entropy = entropy / support_size.log().clamp_min(1e-12)
    peak_mass = row_distribution.max(dim=1).values

    species_lookup = {
        species: index for index, species in enumerate(dict.fromkeys(species_ids))
    }
    species_tensor = torch.tensor(
        [species_lookup[species] for species in species_ids],
        device=raw_similarity.device,
    )
    species_equal = species_tensor.unsqueeze(1) == species_tensor.unsqueeze(0)
    wrong_class_mask = (~species_equal) & (~positive_mask)
    hardest_wrong_similarity = raw_similarity.masked_fill(
        ~wrong_class_mask, float("-inf")
    ).max(dim=1).values
    wrong_class_margin = positive_similarity - hardest_wrong_similarity
    same_species_selected = species_equal[batch_indices, selected_indices]

    hardest_indices = negative_similarity.argmax(dim=1)
    hardest_similarity = negative_similarity[batch_indices, hardest_indices]
    hardest_same_species = species_equal[batch_indices, hardest_indices]

    synthetic_features = F.normalize(row_distribution @ image_features, dim=-1)
    synthetic_similarity = (text_features * synthetic_features).sum(dim=1)
    synthetic_logits = synthetic_similarity * torch.as_tensor(
        logit_scale, device=raw_similarity.device, dtype=raw_similarity.dtype
    )

    thresholds = historical_thresholds or {}
    overlay = classify_gate_overlay(
        entropy,
        positive_selected_gap,
        entropy_threshold=thresholds.get("entropy_threshold", 3.0),
        gap_suppress_easy=thresholds.get("gap_suppress_easy", 0.10),
        gap_downweight_hard=thresholds.get("gap_downweight_hard", -0.07),
    )

    return {
        "raw_similarity": raw_similarity,
        "logits": logits,
        "positive_mask": positive_mask,
        "plan": transport.plan,
        "support_mask": transport.support_mask,
        "selected_indices": selected_indices,
        "hardest_indices": hardest_indices,
        "positive_similarity": positive_similarity,
        "selected_similarity": selected_similarity,
        "positive_selected_gap": positive_selected_gap,
        "selected_rank": selected_rank,
        "selected_hardness_percentile": selected_hardness_percentile,
        "coupling_entropy": entropy,
        "normalized_coupling_entropy": normalized_entropy,
        "coupling_peak_mass": peak_mass,
        "hardest_negative_similarity": hardest_similarity,
        "hardest_wrong_class_similarity": hardest_wrong_similarity,
        "wrong_class_margin": wrong_class_margin,
        "same_species_selected": same_species_selected,
        "same_species_hardest": hardest_same_species,
        "synthetic_similarity": synthetic_similarity,
        "synthetic_logits": synthetic_logits,
        "gate_overlay": overlay,
        "retrieval": compute_retrieval_metrics(raw_similarity, positive_mask),
        "marginal_errors": transport_marginal_errors(transport.plan),
    }


def summarize_geometry(diagnostics):
    """Collapse tensor diagnostics into a compact, JSON-ready report."""
    distributions = {}
    keys = (
        "positive_similarity",
        "selected_similarity",
        "positive_selected_gap",
        "selected_rank",
        "selected_hardness_percentile",
        "coupling_entropy",
        "normalized_coupling_entropy",
        "coupling_peak_mass",
        "hardest_negative_similarity",
        "hardest_wrong_class_similarity",
        "wrong_class_margin",
        "synthetic_similarity",
        "synthetic_logits",
    )
    for key in keys:
        distributions[key] = quantile_summary(diagnostics[key].detach().cpu())

    overlay_counts = {}
    for label in diagnostics["gate_overlay"]:
        overlay_counts[label] = overlay_counts.get(label, 0) + 1
    total = max(len(diagnostics["gate_overlay"]), 1)

    def pearson(left, right):
        left = left.detach().float().flatten()
        right = right.detach().float().flatten()
        finite = torch.isfinite(left) & torch.isfinite(right)
        left = left[finite]
        right = right[finite]
        if left.numel() < 2 or left.std(unbiased=False) == 0 or right.std(unbiased=False) == 0:
            return None
        return float(torch.corrcoef(torch.stack((left, right)))[0, 1].item())

    same_species = diagnostics["same_species_selected"]
    nonpositive_gap = diagnostics["positive_selected_gap"] <= 0
    selected_matches_hardest = (
        diagnostics["selected_indices"] == diagnostics["hardest_indices"]
    )

    return {
        "retrieval": diagnostics["retrieval"],
        "distributions": distributions,
        "same_species_selected_rate": float(
            diagnostics["same_species_selected"].float().mean().item()
        ),
        "same_species_hardest_negative_rate": float(
            diagnostics["same_species_hardest"].float().mean().item()
        ),
        "ot_selected_matches_strict_hardest_rate": float(
            selected_matches_hardest.float().mean().item()
        ),
        "potential_false_negative_rates": {
            "selected_same_species_and_gap_nonpositive": float(
                (same_species & nonpositive_gap).float().mean().item()
            ),
            "selected_different_species_and_gap_nonpositive": float(
                ((~same_species) & nonpositive_gap).float().mean().item()
            ),
        },
        "correlations": {
            "entropy_vs_positive_selected_gap": pearson(
                diagnostics["coupling_entropy"],
                diagnostics["positive_selected_gap"],
            ),
            "entropy_vs_selected_rank": pearson(
                diagnostics["coupling_entropy"], diagnostics["selected_rank"]
            ),
            "positive_selected_gap_vs_wrong_class_margin": pearson(
                diagnostics["positive_selected_gap"],
                diagnostics["wrong_class_margin"],
            ),
        },
        "historical_gate_overlay": {
            key: {"count": value, "fraction": value / total}
            for key, value in sorted(overlay_counts.items())
        },
        "transport_marginal_errors": diagnostics["marginal_errors"],
    }
