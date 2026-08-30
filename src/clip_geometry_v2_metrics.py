"""Scale-controlled and batch-emulated frozen CLIP geometry diagnostics.

V1 intentionally reproduces the historical OT-Mix planner.  This module adds
experimental modes around that frozen implementation; it does not change the
training loss or the V1 output contract.
"""

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.loss import compute_alpha_effective
from src.clip_geometry_metrics import (
    classify_gate_overlay,
    compute_retrieval_metrics,
    quantile_summary,
)


GATE_STATE_NAMES = {
    0: "fully_active",
    1: "easy_gap_suppressed",
    2: "too_hard_downweighted",
    3: "entropy_suppressed",
    4: "inactive_scheduled_alpha",
}


@dataclass(frozen=True)
class AuditedTransportPlan:
    """Returned plan plus the state needed to audit sparse mass removal."""

    plan: torch.Tensor
    support_mask: torch.Tensor
    plan_before_final_mask: torch.Tensor
    solver: str


def _validate_transport_inputs(scores, positive_mask, top_k, ot_eps, sinkhorn_iters):
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must be equally shaped 2-D tensors")
    if positive_mask.dtype != torch.bool:
        raise TypeError("positive_mask must be Boolean")
    if top_k < 1 or ot_eps <= 0 or sinkhorn_iters < 1:
        raise ValueError("top_k, ot_eps, and sinkhorn_iters must be positive")
    available = (~positive_mask).sum(dim=1)
    if torch.any(available == 0):
        raise ValueError("Every query needs at least one non-positive candidate")
    return min(top_k, int(available.min().item()))


def _top_k_support(scores, positive_mask, top_k):
    masked_scores = scores.masked_fill(positive_mask, float("-inf"))
    indices = torch.topk(masked_scores, k=top_k, dim=1).indices
    support = torch.zeros_like(positive_mask)
    support.scatter_(1, indices, True)
    support &= ~positive_mask
    return masked_scores, support


def build_audited_transport_plan(
    scores,
    positive_mask,
    *,
    top_k=32,
    ot_eps=0.7,
    sinkhorn_iters=30,
    solver="historical_sparse_ot",
):
    """Solve historical or support-preserving top-k entropic transport.

    ``historical_sparse_ot`` exactly retains the old dense-floor-then-mask
    behavior. ``support_preserving_sparse_ot`` keeps unsupported edges at zero
    throughout Sinkhorn and is diagnostic-only until a training ablation is
    explicitly authorized.
    """
    actual_top_k = _validate_transport_inputs(
        scores, positive_mask, top_k, ot_eps, sinkhorn_iters
    )
    masked_scores, support = _top_k_support(
        scores, positive_mask, actual_top_k
    )
    maximum = masked_scores.max()
    cost = (maximum - scores).clamp_min(0.0)
    affinity = torch.exp(-cost / ot_eps)

    rows, columns = scores.shape
    row_target = torch.full(
        (rows,), 1.0 / rows, device=scores.device, dtype=scores.dtype
    )
    column_target = torch.full(
        (columns,), 1.0 / columns, device=scores.device, dtype=scores.dtype
    )

    if solver == "historical_sparse_ot":
        # Exact historical behavior: the clamp reintroduces unsupported edges
        # during Sinkhorn, and the final multiplication removes their mass.
        kernel = (affinity * support.to(scores.dtype)).clamp_min(1e-12)
        u = torch.ones_like(row_target)
        v = torch.ones_like(column_target)
        for _ in range(sinkhorn_iters):
            u = row_target / (kernel @ v + 1e-8)
            v = column_target / (kernel.T @ u + 1e-8)
        before_mask = (u.unsqueeze(1) * kernel) * v.unsqueeze(0)
        plan = before_mask * support.to(before_mask.dtype)
    elif solver == "support_preserving_sparse_ot":
        if torch.any(~support.any(dim=0)):
            unsupported = int((~support.any(dim=0)).sum().item())
            raise ValueError(
                f"Sparse support has {unsupported} candidate columns with no edge; "
                "uniform column marginals are infeasible"
            )
        kernel = torch.where(
            support,
            affinity.clamp_min(torch.finfo(scores.dtype).tiny),
            torch.zeros_like(affinity),
        )
        u = torch.ones_like(row_target)
        v = torch.ones_like(column_target)
        for _ in range(sinkhorn_iters):
            row_denom = kernel @ v
            if torch.any(row_denom <= 0):
                raise ValueError("Sparse Sinkhorn encountered an unsupported row")
            u = row_target / row_denom
            column_denom = kernel.T @ u
            if torch.any(column_denom <= 0):
                raise ValueError("Sparse Sinkhorn encountered an unsupported column")
            v = column_target / column_denom
        before_mask = (u.unsqueeze(1) * kernel) * v.unsqueeze(0)
        plan = before_mask
    else:
        raise ValueError(f"Unknown transport solver: {solver}")

    return AuditedTransportPlan(
        plan=plan,
        support_mask=support,
        plan_before_final_mask=before_mask,
        solver=solver,
    )


def audit_transport_mass(transport):
    """Report marginal residuals and mass removed by final sparsification."""
    plan = transport.plan
    before = transport.plan_before_final_mask
    rows, columns = plan.shape
    row_target = 1.0 / rows
    column_target = 1.0 / columns
    before_mass = before.sum()
    returned_mass = plan.sum()
    removed_mass = (before_mass - returned_mass).clamp_min(0)
    return {
        "solver": transport.solver,
        "total_mass_before_final_mask": float(before_mass.item()),
        "total_mass_returned": float(returned_mass.item()),
        "mass_removed_by_final_mask": float(removed_mass.item()),
        "fraction_mass_removed_by_final_mask": float(
            (removed_mass / before_mass.clamp_min(1e-12)).item()
        ),
        "max_row_marginal_error": float(
            (plan.sum(1) - row_target).abs().max().item()
        ),
        "mean_row_marginal_error": float(
            (plan.sum(1) - row_target).abs().mean().item()
        ),
        "max_column_marginal_error": float(
            (plan.sum(0) - column_target).abs().max().item()
        ),
        "mean_column_marginal_error": float(
            (plan.sum(0) - column_target).abs().mean().item()
        ),
    }


def _row_distribution(plan):
    mass = plan.sum(dim=1, keepdim=True)
    if torch.any(mass <= 0):
        raise ValueError("Transport plan has a row with no returned mass")
    return plan / mass


def _species_matrix(species_ids, device):
    lookup = {
        species: index for index, species in enumerate(dict.fromkeys(species_ids))
    }
    encoded = torch.tensor([lookup[value] for value in species_ids], device=device)
    return encoded.unsqueeze(1) == encoded.unsqueeze(0)


def _plan_scores(raw_similarity, logit_scale, cost_space):
    if cost_space == "raw_cosine":
        return raw_similarity
    if cost_space == "clip_scaled_logits":
        return raw_similarity * torch.as_tensor(
            logit_scale,
            device=raw_similarity.device,
            dtype=raw_similarity.dtype,
        )
    raise ValueError(f"Unknown cost space: {cost_space}")


def compute_transport_variant(
    *,
    image_features,
    text_features,
    logit_scale,
    image_ids,
    species_ids,
    cost_space,
    top_k,
    ot_eps,
    sinkhorn_iters,
    solver="historical_sparse_ot",
    historical_thresholds=None,
):
    """Compute a frozen full-pool or batch-local transport condition."""
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError("Image and text features must have the same [N, d] shape")
    count = image_features.shape[0]
    if len(image_ids) != count or len(species_ids) != count:
        raise ValueError("Metadata lengths must match the number of feature pairs")
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    raw_similarity = text_features @ image_features.T
    device = raw_similarity.device
    image_id_tensor = torch.as_tensor(image_ids, device=device)
    positive_mask = image_id_tensor.unsqueeze(1) == image_id_tensor.unsqueeze(0)
    scores = _plan_scores(raw_similarity, logit_scale, cost_space)
    transport = build_audited_transport_plan(
        scores,
        positive_mask,
        top_k=top_k,
        ot_eps=ot_eps,
        sinkhorn_iters=sinkhorn_iters,
        solver=solver,
    )
    weights = _row_distribution(transport.plan)
    selected = weights.argmax(dim=1)
    row_indices = torch.arange(count, device=device)

    positive_similarity = raw_similarity.masked_fill(
        ~positive_mask, float("-inf")
    ).max(dim=1).values
    negative_similarity = raw_similarity.masked_fill(
        positive_mask, float("-inf")
    )
    selected_similarity = raw_similarity[row_indices, selected]
    gap = positive_similarity - selected_similarity
    selected_rank = (
        (negative_similarity > selected_similarity.unsqueeze(1)).sum(1).float()
        + 1
    )
    hardest = negative_similarity.argmax(dim=1)

    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(1)
    support_size = transport.support_mask.sum(1).float().clamp_min(1)
    normalized_entropy = entropy / support_size.log().clamp_min(1e-12)
    peak_mass = weights.max(dim=1).values

    species_equal = _species_matrix(species_ids, device)
    same_species_negative = species_equal & ~positive_mask
    wrong_species_negative = ~species_equal & ~positive_mask
    selected_same_species = species_equal[row_indices, selected]
    hardest_same_species = species_equal[row_indices, hardest]

    best_same_similarity = raw_similarity.masked_fill(
        ~same_species_negative, float("-inf")
    ).max(1).values
    best_wrong_similarity = raw_similarity.masked_fill(
        ~wrong_species_negative, float("-inf")
    ).max(1).values
    best_same_rank = (
        (negative_similarity > best_same_similarity.unsqueeze(1)).sum(1).float()
        + 1
    )
    best_wrong_rank = (
        (negative_similarity > best_wrong_similarity.unsqueeze(1)).sum(1).float()
        + 1
    )
    best_same_rank = best_same_rank.masked_fill(
        ~torch.isfinite(best_same_similarity), float("nan")
    )
    wrong_species_margin = positive_similarity - best_wrong_similarity

    random_same_species_chance = (
        same_species_negative.sum(1).float()
        / (~positive_mask).sum(1).float().clamp_min(1)
    )
    support_same_species_rate = (
        (transport.support_mask & same_species_negative).sum(1).float()
        / transport.support_mask.sum(1).float().clamp_min(1)
    )

    synthetic_features = F.normalize(weights @ image_features, dim=-1)
    synthetic_similarity = (text_features * synthetic_features).sum(1)
    synthetic_logits = synthetic_similarity * torch.as_tensor(
        logit_scale, device=device, dtype=raw_similarity.dtype
    )

    thresholds = historical_thresholds or {}
    per_query_overlay = classify_gate_overlay(
        entropy,
        gap,
        entropy_threshold=thresholds.get("entropy_threshold", 3.0),
        gap_suppress_easy=thresholds.get("gap_suppress_easy", 0.10),
        gap_downweight_hard=thresholds.get("gap_downweight_hard", -0.07),
    )

    return {
        "cost_space": cost_space,
        "ot_eps": ot_eps,
        "solver": solver,
        "raw_similarity": raw_similarity,
        "plan": transport.plan,
        "support_mask": transport.support_mask,
        "positive_mask": positive_mask,
        "selected_indices": selected,
        "hardest_indices": hardest,
        "positive_similarity": positive_similarity,
        "selected_similarity": selected_similarity,
        "positive_selected_gap": gap,
        "selected_rank": selected_rank,
        "coupling_entropy": entropy,
        "normalized_coupling_entropy": normalized_entropy,
        "coupling_peak_mass": peak_mass,
        "synthetic_similarity": synthetic_similarity,
        "synthetic_logits": synthetic_logits,
        "selected_same_species": selected_same_species,
        "hardest_same_species": hardest_same_species,
        "random_same_species_chance": random_same_species_chance,
        "support_same_species_rate": support_same_species_rate,
        "best_same_species_rank": best_same_rank,
        "best_wrong_species_rank": best_wrong_rank,
        "wrong_species_margin": wrong_species_margin,
        "per_query_historical_threshold_overlay": per_query_overlay,
        "retrieval": compute_retrieval_metrics(raw_similarity, positive_mask),
        "transport_mass_audit": audit_transport_mass(transport),
    }


def summarize_transport_variant(diagnostics):
    distribution_keys = (
        "positive_similarity",
        "selected_similarity",
        "positive_selected_gap",
        "selected_rank",
        "coupling_entropy",
        "normalized_coupling_entropy",
        "coupling_peak_mass",
        "synthetic_similarity",
        "synthetic_logits",
        "best_same_species_rank",
        "best_wrong_species_rank",
        "wrong_species_margin",
    )
    distributions = {
        key: quantile_summary(diagnostics[key].detach().cpu())
        for key in distribution_keys
    }
    overlay_counts = Counter(
        diagnostics["per_query_historical_threshold_overlay"]
    )
    count = len(diagnostics["per_query_historical_threshold_overlay"])
    selected_same_rate = diagnostics["selected_same_species"].float().mean()
    random_same_chance = diagnostics["random_same_species_chance"].mean()
    support_same_rate = diagnostics["support_same_species_rate"].mean()

    return {
        "retrieval": diagnostics["retrieval"],
        "distributions": distributions,
        "transport_mass_audit": diagnostics["transport_mass_audit"],
        "extreme_gap_behavior": {
            "fraction_positive_selected_gap_lt_0": float(
                (diagnostics["positive_selected_gap"] < 0).float().mean().item()
            ),
            "fraction_synthetic_similarity_gt_positive_similarity": float(
                (
                    diagnostics["synthetic_similarity"]
                    > diagnostics["positive_similarity"]
                )
                .float()
                .mean()
                .item()
            ),
        },
        "strict_hardest_comparison": {
            "ot_selected_equals_strict_hardest_rate": float(
                (
                    diagnostics["selected_indices"]
                    == diagnostics["hardest_indices"]
                )
                .float()
                .mean()
                .item()
            ),
            "selected_rank_context": (
                "Rank is global, but OT support was prefiltered to top-k; high "
                "global hardness is partly guaranteed by candidate preprocessing."
            ),
        },
        "species_neighborhood": {
            "ot_selected_same_species_rate": float(selected_same_rate.item()),
            "strict_hardest_same_species_rate": float(
                diagnostics["hardest_same_species"].float().mean().item()
            ),
            "random_same_species_chance": float(random_same_chance.item()),
            "ot_selected_same_species_enrichment": float(
                (selected_same_rate / random_same_chance.clamp_min(1e-12)).item()
            ),
            "top_k_support_same_species_rate": float(support_same_rate.item()),
            "top_k_support_same_species_enrichment": float(
                (support_same_rate / random_same_chance.clamp_min(1e-12)).item()
            ),
        },
        "per_query_historical_threshold_overlay": {
            "interpretation": (
                "Observational per-query labels only; the training gate uses "
                "batch-level mean entropy and mean gap."
            ),
            "states": {
                key: {"count": value, "fraction": value / count}
                for key, value in sorted(overlay_counts.items())
            },
        },
    }


def compare_transport_variants(left, right):
    """Quantify whether two parameterizations induce the same transport."""
    left_weights = _row_distribution(left["plan"])
    right_weights = _row_distribution(right["plan"])
    return {
        "max_abs_row_normalized_plan_difference": float(
            (left_weights - right_weights).abs().max().item()
        ),
        "mean_abs_row_normalized_plan_difference": float(
            (left_weights - right_weights).abs().mean().item()
        ),
        "selected_index_agreement": float(
            (left["selected_indices"] == right["selected_indices"])
            .float()
            .mean()
            .item()
        ),
        "max_abs_entropy_difference": float(
            (
                left["coupling_entropy"] - right["coupling_entropy"]
            ).abs().max().item()
        ),
    }


def deterministic_batches(count, batch_size, num_batches, seed):
    """Yield full batches from deterministic epoch-like random permutations."""
    if batch_size > count:
        raise ValueError("batch_size cannot exceed the diagnostic pool")
    generator = torch.Generator().manual_seed(seed)
    batches = []
    while len(batches) < num_batches:
        permutation = torch.randperm(count, generator=generator)
        for start in range(0, count - batch_size + 1, batch_size):
            batches.append(permutation[start : start + batch_size])
            if len(batches) == num_batches:
                break
    return batches


def emulate_batch_local_gates(
    *,
    image_features,
    text_features,
    logit_scale,
    image_ids,
    species_ids,
    variants,
    batch_size,
    num_batches,
    seed,
    scheduled_alpha,
    thresholds,
):
    """Run actual batch-level alpha gating over frozen deterministic batches."""
    records = []
    batches = deterministic_batches(
        len(image_ids), batch_size, num_batches, seed
    )
    for batch_number, cpu_indices in enumerate(batches):
        indices = cpu_indices.to(image_features.device)
        batch_image_ids = [image_ids[index] for index in cpu_indices.tolist()]
        batch_species = [species_ids[index] for index in cpu_indices.tolist()]
        for name, config in variants.items():
            diagnostics = compute_transport_variant(
                image_features=image_features.index_select(0, indices),
                text_features=text_features.index_select(0, indices),
                logit_scale=logit_scale,
                image_ids=batch_image_ids,
                species_ids=batch_species,
                cost_space=config["cost_space"],
                top_k=config["top_k"],
                ot_eps=config["ot_eps"],
                sinkhorn_iters=config["sinkhorn_iters"],
                solver=config.get("solver", "historical_sparse_ot"),
                historical_thresholds=thresholds,
            )
            mean_entropy = float(diagnostics["coupling_entropy"].mean().item())
            mean_gap = float(diagnostics["positive_selected_gap"].mean().item())
            alpha_effective, bucket_id = compute_alpha_effective(
                scheduled_alpha,
                mean_entropy,
                mean_gap,
                entropy_threshold=thresholds["entropy_threshold"],
                gap_suppress_easy=thresholds["gap_suppress_easy"],
                gap_downweight_hard=thresholds["gap_downweight_hard"],
                hard_alpha_scale=thresholds["hard_alpha_scale"],
            )
            records.append(
                {
                    "batch_number": batch_number,
                    "variant": name,
                    "mean_coupling_entropy": mean_entropy,
                    "mean_normalized_entropy": float(
                        diagnostics["normalized_coupling_entropy"].mean().item()
                    ),
                    "mean_positive_selected_gap": mean_gap,
                    "mean_coupling_peak_mass": float(
                        diagnostics["coupling_peak_mass"].mean().item()
                    ),
                    "mean_selected_rank": float(
                        diagnostics["selected_rank"].mean().item()
                    ),
                    "median_selected_rank": float(
                        diagnostics["selected_rank"].median().item()
                    ),
                    "mean_synthetic_similarity": float(
                        diagnostics["synthetic_similarity"].mean().item()
                    ),
                    "mean_synthetic_logit": float(
                        diagnostics["synthetic_logits"].mean().item()
                    ),
                    "fraction_synthetics_passing_gate_sim": float(
                        (
                            diagnostics["synthetic_logits"]
                            > thresholds["gate_sim"]
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                    "scheduled_alpha": scheduled_alpha,
                    "alpha_effective": alpha_effective,
                    "gap_bucket_id": bucket_id,
                    "gate_state": GATE_STATE_NAMES[bucket_id],
                    "total_mass_returned": diagnostics[
                        "transport_mass_audit"
                    ]["total_mass_returned"],
                    "fraction_mass_removed": diagnostics[
                        "transport_mass_audit"
                    ]["fraction_mass_removed_by_final_mask"],
                }
            )
    return records


def summarize_batch_emulation(records):
    by_variant = {}
    for variant in sorted({record["variant"] for record in records}):
        subset = [record for record in records if record["variant"] == variant]
        states = Counter(record["gate_state"] for record in subset)
        numeric_fields = (
            "mean_coupling_entropy",
            "mean_normalized_entropy",
            "mean_positive_selected_gap",
            "mean_coupling_peak_mass",
            "mean_selected_rank",
            "median_selected_rank",
            "mean_synthetic_similarity",
            "mean_synthetic_logit",
            "fraction_synthetics_passing_gate_sim",
            "alpha_effective",
            "total_mass_returned",
            "fraction_mass_removed",
        )
        by_variant[variant] = {
            "num_batches": len(subset),
            "gate_state_fractions": {
                state_name: {
                    "count": states.get(state_name, 0),
                    "fraction": states.get(state_name, 0) / len(subset),
                }
                for state_name in GATE_STATE_NAMES.values()
            },
            "distributions": {
                field: quantile_summary([record[field] for record in subset])
                for field in numeric_fields
            },
        }
    return by_variant


def compute_zero_shot_species_evaluation(
    image_features,
    species_text_features,
    image_species_ids,
    prompt_species_ids,
):
    """Evaluate fixed-prompt zero-shot CUB species classification."""
    if image_features.ndim != 2 or species_text_features.ndim != 2:
        raise ValueError("Species evaluation features must be two-dimensional")
    if image_features.shape[1] != species_text_features.shape[1]:
        raise ValueError("Image and species text embedding dimensions must match")
    if len(image_species_ids) != image_features.shape[0]:
        raise ValueError("Each image needs one species ID")
    if len(prompt_species_ids) != species_text_features.shape[0]:
        raise ValueError("Each species prompt needs one species ID")
    if len(set(prompt_species_ids)) != len(prompt_species_ids):
        raise ValueError("Prompt species IDs must be unique")
    missing = set(image_species_ids) - set(prompt_species_ids)
    if missing:
        raise ValueError(f"Missing prompts for species IDs: {sorted(missing)}")
    image_features = F.normalize(image_features, dim=-1)
    species_text_features = F.normalize(species_text_features, dim=-1)
    scores = image_features @ species_text_features.T
    prompt_lookup = {
        species: index for index, species in enumerate(prompt_species_ids)
    }
    targets = torch.tensor(
        [prompt_lookup[species] for species in image_species_ids],
        device=scores.device,
    )
    order = scores.argsort(dim=1, descending=True)
    positive_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
    wrong_mask = torch.ones_like(scores, dtype=torch.bool)
    wrong_mask.scatter_(1, targets.unsqueeze(1), False)
    nearest_wrong = scores.masked_fill(~wrong_mask, float("-inf")).max(1).values
    margin = positive_scores - nearest_wrong
    return {
        "top_1_accuracy": float((order[:, 0] == targets).float().mean().item()),
        "top_5_accuracy": float(
            (order[:, :5] == targets.unsqueeze(1)).any(1).float().mean().item()
        ),
        "positive_species_vs_nearest_wrong_margin": quantile_summary(
            margin.detach().cpu()
        ),
        "fraction_positive_species_margin_lt_0": float(
            (margin < 0).float().mean().item()
        ),
    }
