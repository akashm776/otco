"""Native CLIP fine-tuning objective and fresh batch-local OTCO treatment.

This module is intentionally independent of the historical SigLIP/OTLIP loss
path. The baseline and treatment share the native symmetric CLIP objective;
the treatment adds only a fresh raw-cosine barycentric negative term.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip_backend import CLIPEncoderBackend
from src.clip_geometry_v2_metrics import (
    audit_transport_mass,
    build_audited_transport_plan,
)


GAP_GATE_STATES = {
    0: "fully_active",
    1: "easy_gap_suppressed",
    2: "too_hard_downweighted",
    3: "entropy_suppressed",
    4: "inactive_scheduled_alpha",
}

OT_LOSS_TYPES = {
    "historical_absolute_sigmoid",
    "clip_relative_denominator",
}
SYNTHETIC_WEIGHTING_MODES = {"ot", "uniform_topk"}


@dataclass
class CLIPLossOutput:
    total_loss: torch.Tensor
    clip_loss: torch.Tensor
    ot_loss: torch.Tensor | None
    weighted_ot_loss: torch.Tensor | None
    metrics: dict


def native_clip_contrastive_loss(logits):
    """Standard symmetric CLIP cross-entropy with diagonal pair targets."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Native CLIP loss requires square [B, B] logits")
    targets = torch.arange(logits.shape[0], device=logits.device)
    text_to_image = F.cross_entropy(logits, targets)
    image_to_text = F.cross_entropy(logits.T, targets)
    return 0.5 * (text_to_image + image_to_text)


def clip_relative_denominator_loss(base_logits, synthetic_logits):
    """Additional row-denominator pressure from one synthetic image per text.

    Both inputs must already use the same detached CLIP scale. Subtracting the
    original row log-partition function makes this objective invariant to a
    common additive shift of the ordinary and synthetic logits.
    """
    if base_logits.ndim != 2:
        raise ValueError("Relative OT base logits must have shape [B, B]")
    if synthetic_logits.ndim != 1:
        raise ValueError("Relative OT synthetic logits must have shape [B]")
    if base_logits.shape[0] != synthetic_logits.shape[0]:
        raise ValueError("Relative OT base and synthetic batch sizes must match")
    base_lse = torch.logsumexp(base_logits, dim=1)
    augmented_lse = torch.logsumexp(
        torch.cat((base_logits, synthetic_logits.unsqueeze(1)), dim=1),
        dim=1,
    )
    # logaddexp(base_lse, synthetic) - base_lse is exactly
    # softplus(synthetic - base_lse), but this form avoids cancellation after a
    # large common logit shift.
    loss = F.softplus(synthetic_logits - base_lse).mean()
    return loss, base_lse, augmented_lse


def synthetic_barycentric_weights(plan, support_mask, mode="ot"):
    """Return OT or exact-uniform coefficients on one unchanged support."""
    if mode not in SYNTHETIC_WEIGHTING_MODES:
        raise ValueError(f"Unsupported synthetic weighting mode: {mode}")
    if plan.shape != support_mask.shape or support_mask.dtype != torch.bool:
        raise ValueError("plan and Boolean support_mask must have the same shape")
    if mode == "ot":
        # Preserve the historical training expression exactly.
        return plan / plan.sum(1, keepdim=True).clamp_min(1e-8)
    weights = support_mask.to(plan.dtype)
    return weights / weights.sum(1, keepdim=True).clamp_min(1)


def scheduled_ot_alpha(step, *, alpha_max, warmup_steps, ramp_steps):
    """Historical fixed warmup followed by a linear alpha ramp."""
    if step < warmup_steps or alpha_max == 0:
        return 0.0
    if ramp_steps <= 0:
        return float(alpha_max)
    progress = min(1.0, (step - warmup_steps) / ramp_steps)
    return float(alpha_max) * progress


def compute_clip_alpha_effective(
    alpha,
    coupling_entropy,
    positive_selected_gap,
    *,
    entropy_gate_enabled,
    entropy_threshold,
    gap_suppress_easy,
    gap_downweight_hard,
    hard_alpha_scale,
):
    """Apply optional entropy gating and the preserved historical gap gate."""
    if alpha == 0:
        return 0.0, 4
    if entropy_gate_enabled and coupling_entropy > entropy_threshold:
        return 0.0, 3
    if positive_selected_gap > gap_suppress_easy:
        return 0.0, 1
    if positive_selected_gap < gap_downweight_hard:
        return float(alpha) * hard_alpha_scale, 2
    return float(alpha), 0


def configure_clip_trainable_parameters(model, policy):
    """Freeze CLIP, then unfreeze projections, final blocks, and logit scale."""
    if policy != "projections_last_blocks_logit_scale":
        raise ValueError(f"Unsupported CLIP fine-tuning policy: {policy}")
    clip = model.clip_model if isinstance(model, CLIPEncoderBackend) else model
    clip.requires_grad_(False)

    modules = {
        "visual_projection": clip.visual_projection,
        "text_projection": clip.text_projection,
        "vision_last_block": clip.vision_model.encoder.layers[-1],
        "text_last_block": clip.text_model.encoder.layers[-1],
    }
    for module in modules.values():
        module.requires_grad_(True)
    clip.logit_scale.requires_grad_(True)

    prefix = "clip_model." if isinstance(model, CLIPEncoderBackend) else ""
    trainable = [
        {"name": name, "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    expected_prefixes = (
        f"{prefix}visual_projection.",
        f"{prefix}text_projection.",
        f"{prefix}vision_model.encoder.layers.",
        f"{prefix}text_model.encoder.layers.",
        f"{prefix}logit_scale",
    )
    unexpected = [
        item["name"]
        for item in trainable
        if not item["name"].startswith(expected_prefixes)
    ]
    if unexpected:
        raise AssertionError(f"Unexpected trainable CLIP parameters: {unexpected}")

    return {
        "policy": policy,
        "trainable_parameter_count": sum(item["numel"] for item in trainable),
        "trainable_tensor_count": len(trainable),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "parameters": trainable,
    }


def build_clip_optimizer(model, config):
    """Build deterministic, non-overlapping AdamW groups for the fixed policy."""
    groups = {"projection": [], "encoder": [], "logit_scale": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "visual_projection" in name or "text_projection" in name:
            groups["projection"].append(parameter)
        elif name.endswith("logit_scale"):
            groups["logit_scale"].append(parameter)
        else:
            groups["encoder"].append(parameter)
    if not all(groups.values()):
        raise AssertionError("Every CLIP optimizer parameter group must be non-empty")
    return torch.optim.AdamW(
        [
            {"params": groups["projection"], "lr": config["projection_lr"]},
            {"params": groups["encoder"], "lr": config["encoder_lr"]},
            {"params": groups["logit_scale"], "lr": config["logit_scale_lr"]},
        ],
        weight_decay=config["weight_decay"],
    )


class FreshBatchRawCosineOTCO(nn.Module):
    """Historical sparse Sinkhorn on fresh live-batch raw cosine every step."""

    def __init__(self, config):
        super().__init__()
        if config["cost_space"] != "raw_cosine":
            raise ValueError("First CLIP treatment requires raw_cosine OT cost")
        if config["solver"] != "historical_sparse_ot":
            raise ValueError("First CLIP treatment preserves historical_sparse_ot")
        if config["update_freq"] != 1:
            raise ValueError("First CLIP treatment requires update_freq=1")
        loss_type = config.get("loss_type", "historical_absolute_sigmoid")
        if loss_type not in OT_LOSS_TYPES:
            raise ValueError(f"Unsupported CLIP OT loss_type: {loss_type}")
        if (
            loss_type == "clip_relative_denominator"
            and config.get("synthetic_logit_gate_enabled", True)
        ):
            raise ValueError(
                "clip_relative_denominator requires the absolute synthetic-logit "
                "gate to be disabled"
            )
        synthetic_weighting = config.get("synthetic_weighting", "ot")
        if synthetic_weighting not in SYNTHETIC_WEIGHTING_MODES:
            raise ValueError(
                f"Unsupported synthetic weighting mode: {synthetic_weighting}"
            )
        self.config = dict(config)
        self.loss_type = loss_type
        self.synthetic_weighting = synthetic_weighting
        self.plan_build_count = 0
        self.last_plan = None

    def _make_fresh_plan(self, raw_similarity):
        batch_size = raw_similarity.shape[0]
        positives = torch.eye(
            batch_size, dtype=torch.bool, device=raw_similarity.device
        )
        with torch.no_grad():
            transport = build_audited_transport_plan(
                raw_similarity.detach().float(),
                positives,
                top_k=self.config["top_k"],
                ot_eps=self.config["ot_eps"],
                sinkhorn_iters=self.config["sinkhorn_iters"],
                solver=self.config["solver"],
            )
        self.plan_build_count += 1
        self.last_plan = transport.plan.detach().clone()
        return transport

    def forward(
        self,
        *,
        raw_similarity,
        image_features,
        text_features,
        logit_scale,
        species_ids,
        step,
    ):
        transport = self._make_fresh_plan(raw_similarity)
        plan = transport.plan
        weights = synthetic_barycentric_weights(
            plan, transport.support_mask, mode="ot"
        )
        selected = plan.argmax(1)
        row = torch.arange(plan.shape[0], device=plan.device)

        raw_float = raw_similarity.detach().float()
        positive_similarity = raw_float.diagonal()
        selected_similarity = raw_float[row, selected]
        gap = positive_similarity - selected_similarity
        negative = raw_float.masked_fill(
            torch.eye(plan.shape[0], dtype=torch.bool, device=plan.device),
            float("-inf"),
        )
        selected_rank = (
            (negative > selected_similarity.unsqueeze(1)).sum(1).float() + 1
        )

        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(1)
        support_size = transport.support_mask.sum(1).float().clamp_min(1)
        normalized_entropy = entropy / support_size.log().clamp_min(1e-12)
        peak_mass = weights.max(1).values

        active_weights = synthetic_barycentric_weights(
            plan,
            transport.support_mask,
            mode=self.synthetic_weighting,
        )
        live_weights = active_weights.to(image_features.dtype)
        synthetic_features = F.normalize(live_weights @ image_features, dim=-1)
        synthetic_similarity = (text_features * synthetic_features).sum(1)

        counterfactual_metrics = {}
        if self.synthetic_weighting == "uniform_topk":
            # The OT plan is already required for gating. Use its row weights
            # only for a detached observational comparison; this path cannot
            # affect the active uniform synthetic or its gradients.
            with torch.no_grad():
                counterfactual_ot_features = F.normalize(
                    weights.to(image_features.dtype) @ image_features.detach(),
                    dim=-1,
                )
                counterfactual_ot_similarity = (
                    text_features.detach() * counterfactual_ot_features
                ).sum(1).float()
            active_uniform_similarity = synthetic_similarity.detach().float()
            uniform_minus_ot = (
                active_uniform_similarity - counterfactual_ot_similarity
            )
        # CLIP's learned scale remains trainable through the native contrastive
        # logits, but neither OT loss may update it directly.
        ot_scale = logit_scale.detach()
        synthetic_logits = synthetic_similarity * ot_scale
        relative_metrics = {}
        if self.loss_type == "historical_absolute_sigmoid":
            gate_threshold = self.config.get("synthetic_logit_gate", -4.0)
            gate_enabled = self.config.get("synthetic_logit_gate_enabled", True)
            synthetic_gate = (
                synthetic_logits > gate_threshold
                if gate_enabled
                else torch.ones_like(synthetic_logits, dtype=torch.bool)
            )
            if synthetic_gate.any():
                ot_loss = -F.logsigmoid(-synthetic_logits[synthetic_gate]).mean()
            else:
                ot_loss = synthetic_logits.mean() * 0
        else:
            # Add the synthetic image only to the text->image denominator. Both
            # ordinary and synthetic OT logits share the same detached scale.
            ot_base_logits = raw_similarity * ot_scale
            ot_loss, base_lse, augmented_lse = clip_relative_denominator_loss(
                ot_base_logits, synthetic_logits
            )
            synthetic_gate = torch.ones_like(synthetic_logits, dtype=torch.bool)
            relative_metrics = {
                "ot_relative_loss": float(ot_loss.detach().item()),
                "base_row_logsumexp_mean": float(
                    base_lse.detach().mean().item()
                ),
                "augmented_row_logsumexp_mean": float(
                    augmented_lse.detach().mean().item()
                ),
                "synthetic_logit_minus_base_lse_mean": float(
                    (synthetic_logits.detach() - base_lse.detach()).mean().item()
                ),
            }

        contributor_mask = (
            transport.support_mask
            if self.synthetic_weighting == "uniform_topk"
            else plan > 0
        )
        hardest_real_similarity = raw_float.masked_fill(
            ~contributor_mask, float("-inf")
        ).max(1).values
        synthetic_delta = (
            synthetic_similarity.detach().float() - hardest_real_similarity
        )
        if self.synthetic_weighting == "uniform_topk":
            counterfactual_ot_delta = (
                counterfactual_ot_similarity - hardest_real_similarity
            )
            counterfactual_metrics = {
                "active_uniform_synthetic_similarity": float(
                    active_uniform_similarity.mean().item()
                ),
                "counterfactual_ot_synthetic_similarity": float(
                    counterfactual_ot_similarity.mean().item()
                ),
                "uniform_minus_ot_synthetic_similarity": float(
                    uniform_minus_ot.mean().item()
                ),
                "fraction_uniform_gt_ot_synthetic": float(
                    (uniform_minus_ot > 0).float().mean().item()
                ),
                "uniform_synthetic_vs_hardest_real_delta": float(
                    synthetic_delta.mean().item()
                ),
                "ot_synthetic_vs_hardest_real_delta": float(
                    counterfactual_ot_delta.mean().item()
                ),
                "fraction_uniform_synthetic_gt_hardest_real": float(
                    (synthetic_delta > 0).float().mean().item()
                ),
                "fraction_ot_synthetic_gt_hardest_real": float(
                    (counterfactual_ot_delta > 0).float().mean().item()
                ),
            }

        species = torch.as_tensor(species_ids, device=plan.device)
        selected_same_species = species[selected] == species
        diagonal_mask = torch.eye(
            raw_similarity.shape[0],
            dtype=torch.bool,
            device=raw_similarity.device,
        )
        positive_cosine = raw_similarity.diagonal().detach().float()
        offdiagonal_cosine = raw_similarity.detach().float()[~diagonal_mask]
        synthetic_cosine = synthetic_similarity.detach().float()
        mean_entropy = float(entropy.mean().item())
        mean_gap = float(gap.mean().item())
        alpha_scheduled = scheduled_ot_alpha(
            step,
            alpha_max=self.config["alpha_max"],
            warmup_steps=self.config["warmup_steps"],
            ramp_steps=self.config["ramp_steps"],
        )
        alpha_effective, gate_state_id = compute_clip_alpha_effective(
            alpha_scheduled,
            mean_entropy,
            mean_gap,
            entropy_gate_enabled=self.config["entropy_gate_enabled"],
            entropy_threshold=self.config["entropy_threshold"],
            gap_suppress_easy=self.config["gap_suppress_easy"],
            gap_downweight_hard=self.config["gap_downweight_hard"],
            hard_alpha_scale=self.config["hard_alpha_scale"],
        )
        mass = audit_transport_mass(transport)
        metrics = {
            "ot_loss": float(ot_loss.detach().item()),
            "ot_loss_type": self.loss_type,
            "synthetic_weighting_mode": self.synthetic_weighting,
            "alpha_scheduled": alpha_scheduled,
            "alpha_effective": alpha_effective,
            "gate_state_id": gate_state_id,
            "gate_state": GAP_GATE_STATES[gate_state_id],
            "coupling_entropy": mean_entropy,
            "normalized_coupling_entropy": float(
                normalized_entropy.mean().item()
            ),
            "positive_selected_gap": mean_gap,
            "coupling_peak_mass": float(peak_mass.mean().item()),
            "selected_rank": float(selected_rank.mean().item()),
            "same_species_selected_rate": float(
                selected_same_species.float().mean().item()
            ),
            "synthetic_similarity": float(synthetic_similarity.mean().item()),
            "positive_similarity": float(positive_similarity.mean().item()),
            "fraction_synthetic_gt_positive": float(
                (synthetic_similarity.detach().float() > positive_similarity)
                .float()
                .mean()
                .item()
            ),
            "hardest_real_contributor_similarity": float(
                hardest_real_similarity.mean().item()
            ),
            "synthetic_vs_hardest_real_delta": float(
                synthetic_delta.mean().item()
            ),
            "fraction_synthetic_gt_hardest_real": float(
                (synthetic_delta > 0).float().mean().item()
            ),
            "synthetic_gate_active_fraction": float(
                synthetic_gate.float().mean().item()
            ),
            "mean_positive_cosine": float(positive_cosine.mean().item()),
            "mean_offdiagonal_cosine": float(offdiagonal_cosine.mean().item()),
            "mean_synthetic_cosine": float(synthetic_cosine.mean().item()),
            "median_positive_cosine": float(positive_cosine.median().item()),
            "median_offdiagonal_cosine": float(
                offdiagonal_cosine.median().item()
            ),
            "transport_total_mass": mass["total_mass_returned"],
            "transport_max_row_marginal_error": mass[
                "max_row_marginal_error"
            ],
            "transport_max_column_marginal_error": mass[
                "max_column_marginal_error"
            ],
            "transport_fraction_mass_removed": mass[
                "fraction_mass_removed_by_final_mask"
            ],
            "plan_build_count": self.plan_build_count,
            **relative_metrics,
            **counterfactual_metrics,
        }
        return ot_loss, alpha_effective, metrics


class CLIPTrainingObjective(nn.Module):
    """Native CLIP baseline with an optional isolated OTCO addition."""

    def __init__(self, ot_config):
        super().__init__()
        self.ot_enabled = bool(ot_config["enabled"])
        self.otco = FreshBatchRawCosineOTCO(ot_config) if self.ot_enabled else None

    def forward(self, model_output, *, species_ids, step):
        clip_loss = native_clip_contrastive_loss(model_output.logits)
        if not self.ot_enabled:
            return CLIPLossOutput(
                total_loss=clip_loss,
                clip_loss=clip_loss,
                ot_loss=None,
                weighted_ot_loss=None,
                metrics={
                    "clip_loss": float(clip_loss.detach().item()),
                    "total_loss": float(clip_loss.detach().item()),
                    "ot_enabled": False,
                },
            )

        ot_loss, alpha_effective, ot_metrics = self.otco(
            raw_similarity=model_output.raw_similarity,
            image_features=model_output.image_features,
            text_features=model_output.text_features,
            logit_scale=model_output.logit_scale,
            species_ids=species_ids,
            step=step,
        )
        weighted_ot = ot_loss * alpha_effective
        total_loss = clip_loss if alpha_effective == 0 else clip_loss + weighted_ot
        metrics = {
            "clip_loss": float(clip_loss.detach().item()),
            "weighted_ot_loss": float(weighted_ot.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "ot_enabled": True,
            **ot_metrics,
        }
        if self.otco.loss_type == "clip_relative_denominator":
            metrics["weighted_ot_relative_loss"] = float(
                weighted_ot.detach().item()
            )
            metrics["weighted_synthetic_relative_loss"] = float(
                weighted_ot.detach().item()
            )
        return CLIPLossOutput(
            total_loss=total_loss,
            clip_loss=clip_loss,
            ot_loss=ot_loss,
            weighted_ot_loss=weighted_ot,
            metrics=metrics,
        )


def tensor_gradient_norm(gradients):
    """Global L2 norm for a sequence returned by torch.autograd.grad."""
    squares = [
        gradient.detach().float().pow(2).sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squares:
        return 0.0
    return math.sqrt(float(torch.stack(squares).sum().item()))


def parameter_gradient_norm(parameters):
    """Global L2 norm of currently accumulated parameter gradients."""
    return tensor_gradient_norm([parameter.grad for parameter in parameters])
