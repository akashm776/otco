"""Contrastive objectives used to train OTLIP.

Notation used throughout this module:

* ``B``: current batch size (number of aligned text-image pairs).
* ``N``: number of candidate images in an external cached pool.
* ``d``: shared embedding dimension, normally 512.
* ``logits``: ``[B, B]`` temperature-scaled similarities. The diagonal contains
  matching pairs and off-diagonal cells are in-batch negatives.
* ``text_emb`` and ``image_emb``: normalized live ``[B, d]`` embeddings.
* ``image_pool``: detached epoch-cached ``[N, d]`` image embeddings.
* ``plan``: ``[B, B]`` batch-local or ``[B, N]`` pool transport coupling.
* ``local_mask``: Boolean matrix identifying the top-k edges allowed in OT.
* ``alpha``: scheduled auxiliary-loss weight before quality-based gating.
* ``alpha_effective``: weight actually applied after entropy/gap gating.

Selection and Sinkhorn planning are intentionally performed without gradients.
The final loss remains differentiable through live text embeddings and, when
live image features are used, through the image encoder as well.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_alpha_effective(
    alpha,
    coupling_entropy,
    pos_selected_gap,
    entropy_threshold=3.0,
    gap_suppress_easy=0.10,
    gap_downweight_hard=-0.07,
    hard_alpha_scale=0.25,
):
    """Choose how strongly OT should influence one optimization step.

    ``coupling_entropy`` is mean row entropy after each row is normalized into
    outgoing-flow proportions. ``pos_selected_gap`` is positive cosine
    similarity minus the primary OT-selected negative similarity.

    Gap bucket IDs:
      0 = useful     (-0.05 <= gap <= gap_suppress_easy)
      1 = too_easy   (gap > gap_suppress_easy)
      2 = too_hard   (gap < gap_downweight_hard)
      3 = diffuse    (coupling_entropy > entropy_threshold)
      4 = inactive   (alpha == 0)
    """
    if alpha == 0.0:
        return 0.0, 4

    if coupling_entropy > entropy_threshold:
        return 0.0, 3

    if pos_selected_gap > gap_suppress_easy:
        return 0.0, 1

    if pos_selected_gap < gap_downweight_hard:
        return alpha * hard_alpha_scale, 2

    return alpha, 0


def _negative_rank_stats(raw_sim, selected_indices):
    """Measure one selected image's hardness for every batch text.

    ``raw_sim`` is ``[B, B]`` cosine similarity and ``selected_indices`` is the
    chosen image column for each row. Diagonal positives are excluded. Rank one
    means the selected image is the hardest in-batch negative.
    """
    bsz = raw_sim.size(0)
    mask = torch.eye(bsz, device=raw_sim.device, dtype=torch.bool)
    neg_sims = raw_sim.masked_fill(mask, float("-inf"))

    batch_idx = torch.arange(bsz, device=raw_sim.device)
    selected_sims = neg_sims[batch_idx, selected_indices]
    rank = (neg_sims > selected_sims.unsqueeze(1)).sum(dim=1).float() + 1.0
    pos_sims = raw_sim.diag()
    gap = pos_sims - selected_sims

    return {
        "mean_selected_rank": rank.mean().item(),
        "median_selected_rank": rank.median().item(),
        "avg_selected_sim": selected_sims.mean().item(),
        "pos_selected_gap": gap.mean().item(),
    }


def _pool_rank_stats(pool_sim, selected_indices, pos_sims):
    """Measure selected-negative hardness inside an external image pool.

    ``pool_sim`` is ``[B, N]`` cosine similarity, ``selected_indices`` contains
    one pool column per text, and ``pos_sims`` contains the ``B`` live positive
    similarities. No diagonal is masked because positives were excluded from
    the sampled pool upstream using image IDs.
    """
    bsz = pool_sim.size(0)
    batch_idx = torch.arange(bsz, device=pool_sim.device)
    selected_sims = pool_sim[batch_idx, selected_indices]
    rank = (pool_sim > selected_sims.unsqueeze(1)).sum(dim=1).float() + 1.0
    gap = pos_sims - selected_sims
    return {
        "mean_selected_rank": rank.mean().item(),
        "median_selected_rank": rank.median().item(),
        "avg_selected_sim": selected_sims.mean().item(),
        "pos_selected_gap": gap.mean().item(),
    }


class SigLIPLoss(nn.Module):
    """Binary matching loss over all ``B²`` image-caption pairs.

    ``logit_bias`` is a learned scalar that calibrates the imbalance between B
    positive cells and B²-B negative cells. No softmax is needed: each matrix
    cell is an independent positive/negative decision.
    """

    def __init__(self, init_bias=0.0):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, logits):
        # Convert the identity matrix from {0, 1} to {-1, +1}: matching diagonal
        # cells receive +1 and every mismatched cell receives -1.
        n = logits.shape[0]
        logits = logits + self.logit_bias
        labels = 2 * torch.eye(n, device=logits.device) - 1
        # Multiplication by labels flips negative-pair logits before logsigmoid;
        # averaging by n² gives every pair equal weight.
        loss = -torch.sum(F.logsigmoid(labels * logits)) / (n * n)
        return loss


class HardNegativeLoss(nn.Module):
    """
    SigLIP + explicit hard negative term.
    Finds hardest in-batch negative and adds extra push-away pressure.
    """

    def __init__(self, init_bias=0.0, alpha=0.5, warmup_steps=1000):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))
        self.alpha_max = alpha
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_alpha(self):
        if self.current_step < self.warmup_steps:
            return 0.0
        return self.alpha_max

    def forward(self, logits, text_emb=None, image_emb=None, temp=None):
        # The unused embedding/temp arguments keep a common interface with all
        # auxiliary loss classes called by src/main.py.
        B = logits.shape[0]

        # Base SigLIP loss
        logits_biased = logits + self.logit_bias
        labels = 2 * torch.eye(B, device=logits.device) - 1
        base_loss = -torch.sum(F.logsigmoid(labels * logits_biased)) / (B * B)

        alpha = self.get_alpha()

        if alpha > 0:
            # Hide diagonal positives, then take the largest remaining logit in
            # each row: one strict hardest negative per text.
            mask = torch.eye(B, device=logits.device, dtype=torch.bool)
            masked_logits = logits.masked_fill(mask, float("-inf"))
            hardest_neg_sims = masked_logits.max(dim=1)[0]  # [B]
            hard_loss = -F.logsigmoid(-hardest_neg_sims).mean()

            total_loss = base_loss + alpha * hard_loss
            loss_dict = {
                "base_loss": base_loss.item(),
                "hard_loss": hard_loss.item(),
                "alpha": float(alpha),
                "total_loss": total_loss.item(),
            }
        else:
            total_loss = base_loss
            loss_dict = {
                "base_loss": base_loss.item(),
                "hard_loss": 0.0,
                "alpha": 0.0,
                "total_loss": total_loss.item(),
            }

        self.current_step += 1
        return total_loss, loss_dict


class SoftmaxMixLoss(nn.Module):
    """SigLIP plus OT-derived barycentric synthetic negatives.

    Constructor variables:
        init_bias: Initial learned base-logit calibration scalar.
        alpha: Maximum synthetic-loss weight.
        warmup_steps: Baseline-only steps in fixed-warmup mode.
        top_k: Candidate edges retained for each text supply vertex.
        tau: Legacy configuration value; the current OT path uses ``ot_eps``.
        update_freq: Batch-local plan recomputation interval. Pool plans are
            always recomputed because a fresh pool is sampled each step.
        gate_sim: Minimum synthetic logit admitted to the auxiliary loss.
        ot_eps: Entropic temperature of the Gibbs transport kernel.
        sinkhorn_iters: Alternating marginal-normalization iterations.
        adaptive_warmup: Activate OT based on plan concentration instead of a
            fixed step count.
        entropy_threshold: Suppress plans whose mean row entropy is too high.
        entropy_check_freq: Plan checks between adaptive-warmup decisions.
        gap_suppress_easy: Suppress selected negatives far below positives.
        gap_downweight_hard: Reduce pressure when a negative beats its positive
            by a suspiciously large margin.
        hard_alpha_scale: Fraction of alpha retained in that too-hard regime.
    """

    def __init__(
        self,
        init_bias=0.0,
        alpha=0.5,
        warmup_steps=1000,
        top_k=32,
        tau=0.05,
        update_freq=10,
        gate_sim=-0.05,
        ot_eps=0.05,
        sinkhorn_iters=30,
        adaptive_warmup=False,
        entropy_threshold=3.0,
        entropy_check_freq=100,
        gap_suppress_easy=0.10,
        gap_downweight_hard=-0.07,
        hard_alpha_scale=0.25,
    ):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))
        self.alpha_max = alpha
        self.warmup_steps = warmup_steps
        self.top_k = top_k
        self.tau = tau
        self.update_freq = update_freq
        self.gate_sim = gate_sim
        self.ot_eps = ot_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.adaptive_warmup = adaptive_warmup
        self.entropy_threshold = entropy_threshold
        self.entropy_check_freq = entropy_check_freq
        # Conditional alpha gating thresholds (initial values — tune per dataset)
        self.gap_suppress_easy = gap_suppress_easy
        self.gap_downweight_hard = gap_downweight_hard
        self.hard_alpha_scale = hard_alpha_scale

        self.current_step = 0
        # These caches apply only to square batch-local OT. Pool-mode plans are
        # identity-specific [B, N] matrices and are freshly constructed.
        self.cached_plan = None  # [B, B] OT coupling
        self.cached_local_mask = None
        # adaptive warmup state
        self.ot_ready = (
            not adaptive_warmup
        )  # fixed warmup: always ready after warmup_steps
        self.steps_since_ready = 0
        self.last_warmup_entropy = float("nan")

    def get_alpha(self):
        """Return zero before activation, then ramp to alpha_max over 1000 steps."""
        if self.adaptive_warmup:
            if not self.ot_ready:
                return 0.0
            progress = min(1.0, self.steps_since_ready / 1000.0)
            return self.alpha_max * progress
        if self.current_step < self.warmup_steps:
            return 0.0
        progress = min(1.0, (self.current_step - self.warmup_steps) / 1000.0)
        return self.alpha_max * progress

    def forward(
        self,
        logits,
        text_emb,
        image_emb,
        temp=None,
        image_pool=None,
        precomputed_plan=None,
        precomputed_local_mask=None,
        live_feats=None,
        live_local_idx=None,
    ):
        """Compute base SigLIP plus a conditionally gated OT-Mix loss.

        Args:
            logits: Live ``[B, B]`` scaled batch similarities.
            text_emb: Normalized live ``[B, d]`` text embeddings.
            image_emb: Normalized live ``[B, d]`` positive image embeddings.
            temp: Model temperature; its reciprocal converts cosine to logits.
            image_pool: Optional detached epoch-cached ``[N, d]`` support.
            precomputed_plan: Optional ``[B, N]`` pool plan already calculated
                during live-contributor discovery, avoiding duplicate Sinkhorn.
            precomputed_local_mask: Top-k support paired with that plan.
            live_feats: Optional differentiable ``[M, d]`` re-encoded images.
            live_local_idx: ``[M]`` columns mapping live features back to pool
                plan columns.

        Pool positives are excluded by image ID in ``src/main.py``. Pool plans
        are fresh each step; only the older batch-local path caches a plan.
        """
        B = logits.shape[0]

        # Safety: invalidate batch-local cached plan when batch size changes.
        # Pool plans are never cached (recomputed every step for fresh pool samples).
        if (
            image_pool is None
            and self.cached_plan is not None
            and self.cached_plan.size(0) != B
        ):
            self.cached_plan = None
            self.cached_local_mask = None

        # Default diagnostics ensure a consistent loss_dict during warmup.
        num_gated = 0
        selected_rank = 0.0
        selected_rank_median = 0.0
        selected_sim_mean = 0.0
        pos_selected_gap = 0.0
        coupling_entropy = 0.0
        coupling_peak_mass = 0.0
        mass_retained = 1.0

        # Base SigLIP loss (always over the batch; unchanged by pool mode)
        logits_biased = logits + self.logit_bias
        labels = 2 * torch.eye(B, device=logits.device) - 1
        base_loss = -torch.sum(F.logsigmoid(labels * logits_biased)) / (B * B)

        # ``scale`` converts normalized cosine similarity to the exact logit
        # scale used by OTLIP. It is inferred later only when temp is absent.
        scale = (
            logits.new_tensor(1.0 / float(temp)) if temp is not None else None
        )

        # Adaptive warmup: check plan entropy every entropy_check_freq steps
        if self.adaptive_warmup and not self.ot_ready:
            if self.current_step % self.entropy_check_freq == 0:
                with torch.no_grad():
                    if image_pool is not None and scale is not None:
                        test_logits = (
                            text_emb @ image_pool.T * scale + self.logit_bias
                        )
                        test_plan, test_mask = self._make_plan_pool(test_logits)
                    else:
                        test_plan, test_mask = self._make_plan(
                            text_emb, image_emb, logits_biased
                        )
                    # Normalize each supply row into outgoing-flow proportions;
                    # entropy then measures whether flow is concentrated or diffuse.
                    local = test_plan * test_mask.float()
                    local = local / local.sum(dim=1, keepdim=True).clamp_min(
                        1e-8
                    )
                    entropy = (
                        -(local * local.clamp_min(1e-12).log())
                        .sum(dim=1)
                        .mean()
                        .item()
                    )
                self.last_warmup_entropy = entropy
                if entropy < self.entropy_threshold:
                    self.ot_ready = True

        alpha = self.get_alpha()
        synthetic_loss = logits.new_tensor(0.0)
        avg_synth_sim = logits.new_tensor(0.0)
        avg_synth_logit = logits.new_tensor(0.0)
        # Always reflect passed pool size, even if OT is inactive (warmup / gated off)
        pool_size_used = image_pool.size(0) if image_pool is not None else 0

        if alpha > 0:
            if scale is None:
                with torch.no_grad():
                    raw = text_emb @ image_emb.T
                    off = ~torch.eye(B, dtype=torch.bool, device=logits.device)
                    scale = (
                        logits.detach()[off] / (raw.detach()[off] + 1e-8)
                    ).median()

            if image_pool is not None:
                # ── Pool path: OT over [B, N] ──────────────────────────────────
                # N is independent of B, so this is a rectangular bipartite graph.
                N = image_pool.size(0)
                pool_size_used = N
                if precomputed_plan is not None:
                    # Live re-forward discovery already solved OT for this exact
                    # text/pool support; reuse it without changing identities.
                    plan = precomputed_plan
                    local_mask = (
                        precomputed_local_mask
                        if precomputed_local_mask is not None
                        else (plan > 1e-12)
                    )
                else:
                    with torch.no_grad():
                        pool_logits = (
                            text_emb @ image_pool.T * scale + self.logit_bias
                        )  # [B, N]
                        plan, local_mask = self._make_plan_pool(pool_logits)

                if live_feats is not None and live_local_idx is not None:
                    # Live re-forward path: synthetic built from live image features (has gradient)
                    # Keep only plan columns that were re-encoded. M is the count
                    # of unique contributors retained after the upstream cap.
                    live_plan_w = plan[:, live_local_idx]  # [B, M]
                    # Z is retained row mass; renormalizing makes selected live
                    # contributors sum to one before barycentric mixing.
                    Z = live_plan_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    live_weights = (
                        live_plan_w / Z
                    )  # [B, M] renormalized over live set
                    synthetic_neg = (
                        live_weights @ live_feats
                    )  # [B, d], grad flows through live_feats
                    mass_retained = (
                        (
                            live_plan_w.sum(dim=1)
                            / plan.sum(dim=1).clamp_min(1e-8)
                        )
                        .mean()
                        .item()
                    )
                else:
                    # Detached pool path (v1)
                    # Gradients reach current text embeddings through synth_sim,
                    # but cannot enter detached cached image embeddings.
                    row_mass = plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    row_weights = plan / row_mass
                    synthetic_neg = row_weights @ image_pool  # [B, d], detached
                    mass_retained = 1.0
                synthetic_neg = synthetic_neg / (
                    synthetic_neg.norm(dim=1, keepdim=True) + 1e-8
                )

                synth_sim = (text_emb * synthetic_neg).sum(dim=1)  # [B]
                synth_logits = scale * synth_sim

                # Gate out synthetics that are too easy to provide useful pressure.
                gate = (synth_logits > self.gate_sim).float()
                num_gated = int(gate.sum().item())
                if num_gated > 0:
                    synthetic_loss = (
                        -F.logsigmoid(-synth_logits) * gate
                    ).sum() / (gate.sum() + 1e-8)
                else:
                    synthetic_loss = synth_logits.mean() * 0.0

                avg_synth_sim = synth_sim.mean()
                avg_synth_logit = synth_logits.mean()

                with torch.no_grad():
                    # The largest plan edge is a diagnostic representative; the
                    # actual synthetic uses the complete normalized plan row.
                    pool_sim = text_emb @ image_pool.T  # [B, N] raw cosine
                    pos_sims = (text_emb * image_emb).sum(
                        dim=1
                    )  # [B] positive pair cosines
                    selected_indices = plan.argmax(dim=1)
                    rank_stats = _pool_rank_stats(
                        pool_sim, selected_indices, pos_sims
                    )
                    selected_rank = rank_stats["mean_selected_rank"]
                    selected_rank_median = rank_stats["median_selected_rank"]
                    selected_sim_mean = rank_stats["avg_selected_sim"]
                    pos_selected_gap = rank_stats["pos_selected_gap"]

                    local_mass = plan * local_mask.float()
                    local_mass = local_mass / local_mass.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-8)
                    row_entropy = -(
                        local_mass * local_mass.clamp_min(1e-12).log()
                    ).sum(dim=1)
                    coupling_entropy = row_entropy.mean().item()
                    coupling_peak_mass = (
                        local_mass.max(dim=1).values.mean().item()
                    )

            else:
                # ── Batch-local path (existing behavior) ───────────────────────
                # Unlike pool mode, update_freq can reuse a positional [B, B]
                # coupling on later batches; update_freq=1 avoids stale support.
                if (self.current_step % self.update_freq == 0) or (
                    self.cached_plan is None
                ):
                    with torch.no_grad():
                        plan, local_mask = self._make_plan(
                            text_emb, image_emb, logits_biased
                        )
                    self.cached_plan = plan.detach()
                    self.cached_local_mask = local_mask

                row_mass = self.cached_plan.sum(dim=1, keepdim=True).clamp_min(
                    1e-8
                )
                row_weights = self.cached_plan / row_mass
                synthetic_neg = row_weights @ image_emb  # [B, d]
                synthetic_neg = synthetic_neg / (
                    synthetic_neg.norm(dim=1, keepdim=True) + 1e-8
                )

                synth_sim = (text_emb * synthetic_neg).sum(dim=1)  # [B]
                synth_logits = scale * synth_sim

                gate = (synth_logits > self.gate_sim).float()
                num_gated = int(gate.sum().item())
                if num_gated > 0:
                    synthetic_loss = (
                        -F.logsigmoid(-synth_logits) * gate
                    ).sum() / (gate.sum() + 1e-8)
                else:
                    synthetic_loss = synth_logits.mean() * 0.0

                avg_synth_sim = synth_sim.mean()
                avg_synth_logit = synth_logits.mean()

                with torch.no_grad():
                    raw_sim = text_emb @ image_emb.T
                    selected_indices = self.cached_plan.argmax(dim=1)
                    rank_stats = _negative_rank_stats(raw_sim, selected_indices)
                    selected_rank = rank_stats["mean_selected_rank"]
                    selected_rank_median = rank_stats["median_selected_rank"]
                    selected_sim_mean = rank_stats["avg_selected_sim"]
                    pos_selected_gap = rank_stats["pos_selected_gap"]

                    local_mass = (
                        self.cached_plan * self.cached_local_mask.float()
                    )
                    local_mass = local_mass / local_mass.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-8)
                    row_entropy = -(
                        local_mass * (local_mass.clamp_min(1e-12)).log()
                    ).sum(dim=1)
                    coupling_entropy = row_entropy.mean().item()
                    coupling_peak_mass = (
                        local_mass.max(dim=1).values.mean().item()
                    )

        alpha_effective, gap_bucket_id = compute_alpha_effective(
            alpha,
            coupling_entropy,
            pos_selected_gap,
            entropy_threshold=self.entropy_threshold,
            gap_suppress_easy=self.gap_suppress_easy,
            gap_downweight_hard=self.gap_downweight_hard,
            hard_alpha_scale=self.hard_alpha_scale,
        )

        total_loss = base_loss + alpha_effective * synthetic_loss

        loss_dict = {
            "base_loss": base_loss.item(),
            "synthetic_loss": synthetic_loss.item(),
            "alpha_scheduled": float(alpha),
            "alpha_effective": float(alpha_effective),
            "total_loss": total_loss.item(),
            "avg_synthetic_sim": avg_synth_sim.item(),
            "avg_synthetic_logit": avg_synth_logit.item(),
            "num_gated": num_gated if alpha > 0 else 0,
            "ot_step": self.current_step,
            "selected_neg_rank_mean": selected_rank,
            "selected_neg_rank_median": selected_rank_median,
            "selected_neg_sim": selected_sim_mean,
            "pos_selected_gap": pos_selected_gap,
            "coupling_entropy": coupling_entropy,
            "coupling_peak_mass": coupling_peak_mass,
            "ot_ready": int(self.ot_ready),
            "warmup_entropy": (
                self.last_warmup_entropy
                if self.adaptive_warmup and not self.ot_ready
                else 0.0
            ),
            "ot_suppressed_entropy": float(gap_bucket_id == 3),
            "ot_suppressed_too_easy": float(gap_bucket_id == 1),
            "ot_downweighted_too_hard": float(gap_bucket_id == 2),
            "gap_bucket_id": gap_bucket_id,
            "gap_bucket_useful": float(gap_bucket_id == 0),
            "gap_bucket_too_easy": float(gap_bucket_id == 1),
            "gap_bucket_too_hard": float(gap_bucket_id == 2),
            "pool_size": pool_size_used,
            "pool_mode": "pool" if image_pool is not None else "batch",
            "mass_retained": mass_retained,
        }

        self.current_step += 1
        if self.ot_ready:
            self.steps_since_ready += 1
        return total_loss, loss_dict

    def _make_plan_pool(self, pool_logits):
        """Solve rectangular OT between B texts and N cached images.

        ``pool_logits`` contains biased scaled similarities. ``local_mask``
        retains each text's top-k low-cost edges. Uniform marginals ``a`` and
        ``b`` assign supply 1/B and demand 1/N. Sinkhorn vectors ``u`` and ``v``
        iteratively scale the kernel to satisfy those marginal constraints.
        """
        B, N = pool_logits.shape
        k = min(self.top_k, N)

        _, topk_indices = torch.topk(pool_logits, k=k, dim=1)
        local_mask = torch.zeros(
            B, N, dtype=torch.bool, device=pool_logits.device
        )
        local_mask.scatter_(1, topk_indices, True)

        # High similarity should become low cost. Subtracting from the maximum
        # makes costs nonnegative without changing candidate ordering.
        max_logit = pool_logits.max()
        cost = (max_logit - pool_logits).clamp_min(0.0)

        # The Gibbs kernel converts low costs to high affinities; ot_eps controls
        # how concentrated or diffuse that conversion is.
        kernel = torch.exp(-cost / self.ot_eps) * local_mask.float()
        kernel = kernel.clamp_min(1e-12)

        a = torch.full((B,), 1.0 / B, device=pool_logits.device)
        b = torch.full((N,), 1.0 / N, device=pool_logits.device)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.sinkhorn_iters):
            u = a / (kernel @ v + 1e-8)
            v = b / (kernel.t() @ u + 1e-8)

        # Equivalent to diag(u) @ kernel @ diag(v), without creating diagonals.
        plan = (u.unsqueeze(1) * kernel) * v.unsqueeze(0)
        plan = plan * local_mask.float()
        return plan, local_mask

    def _make_plan(self, text_emb, image_emb, logits_biased):
        """Solve square batch-local OT while excluding positive diagonal edges."""
        B = text_emb.size(0)
        k = min(self.top_k, B - 1) if B > 1 else 1

        # Matching text/image vertices cannot act as negative transport edges.
        diag_mask = torch.eye(B, device=logits_biased.device, dtype=torch.bool)

        # Top-k selection in logit space (ranking identical to cosine since scale > 0)
        masked_logits = logits_biased.masked_fill(diag_mask, float("-inf"))
        _, topk_indices = torch.topk(masked_logits, k=k, dim=1)
        local_mask = torch.zeros(
            B, B, dtype=torch.bool, device=logits_biased.device
        )
        local_mask.scatter_(1, topk_indices, True)
        local_mask = local_mask & (~diag_mask)

        # Logit-space cost: hard negatives (high logit) get low cost.
        # Shift so min cost over off-diagonal = 0.
        max_logit = masked_logits.max()
        cost = (max_logit - logits_biased).clamp_min(0.0)

        kernel = torch.exp(-cost / self.ot_eps) * local_mask.float()
        kernel = kernel.clamp_min(1e-12)

        a = torch.full((B,), 1.0 / B, device=logits_biased.device)
        b = torch.full((B,), 1.0 / B, device=logits_biased.device)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.sinkhorn_iters):
            u = a / (kernel @ v + 1e-8)
            v = b / (kernel.t() @ u + 1e-8)

        plan = (u.unsqueeze(1) * kernel) * v.unsqueeze(0)
        plan = plan * local_mask.float()
        return plan, local_mask


class OTSelectLoss(nn.Module):
    """SigLIP plus one selected top-k negative per text.

    The historical name is misleading: this class does not solve Sinkhorn OT.
    Softmax followed by argmax selects the largest top-k cosine similarity.
    """

    def __init__(
        self, init_bias=0.0, alpha=0.1, warmup_steps=1000, top_k=32, tau=0.05
    ):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))
        self.alpha_max = alpha
        self.warmup_steps = warmup_steps
        self.top_k = top_k
        self.tau = tau
        self.current_step = 0

    def get_alpha(self):
        if self.current_step < self.warmup_steps:
            return 0.0
        progress = min(1.0, (self.current_step - self.warmup_steps) / 1000.0)
        return self.alpha_max * progress

    def forward(self, logits, text_emb, image_emb, temp=None):
        B = logits.shape[0]

        logits_biased = logits + self.logit_bias
        labels = 2 * torch.eye(B, device=logits.device) - 1
        base_loss = -torch.sum(F.logsigmoid(labels * logits_biased)) / (B * B)

        alpha = self.get_alpha()

        if alpha > 0:
            raw_sim = text_emb @ image_emb.T
            mask = torch.eye(B, device=raw_sim.device, dtype=torch.bool)
            raw_sim_masked = raw_sim.masked_fill(mask, float("-inf"))

            k = min(self.top_k, B - 1)
            topk_sims, topk_indices = torch.topk(raw_sim_masked, k=k, dim=1)

            with torch.no_grad():
                weights = F.softmax(topk_sims / self.tau, dim=1)
                max_weight_idx = weights.argmax(dim=1)

            batch_range = torch.arange(B, device=topk_indices.device)
            selected_neg_indices = topk_indices[batch_range, max_weight_idx]
            selected_neg_emb = image_emb[selected_neg_indices]

            if temp is None:
                with torch.no_grad():
                    raw = text_emb @ image_emb.T
                    off = ~torch.eye(B, dtype=torch.bool, device=logits.device)
                    scale = (
                        logits.detach()[off] / (raw.detach()[off] + 1e-8)
                    ).median()
            else:
                scale = logits.new_tensor(1.0 / float(temp))

            selected_sim = (text_emb * selected_neg_emb).sum(dim=1)
            selected_logits = scale * selected_sim
            select_loss = -F.logsigmoid(-selected_logits).mean()

            total_loss = base_loss + alpha * select_loss
            rank_stats = _negative_rank_stats(raw_sim, selected_neg_indices)
            loss_dict = {
                "base_loss": base_loss.item(),
                "select_loss": select_loss.item(),
                "alpha": float(alpha),
                "total_loss": total_loss.item(),
                "avg_selected_sim": selected_sim.mean().item(),
                "selected_neg_rank_mean": rank_stats["mean_selected_rank"],
                "selected_neg_rank_median": rank_stats["median_selected_rank"],
                "pos_selected_gap": rank_stats["pos_selected_gap"],
            }
        else:
            total_loss = base_loss
            loss_dict = {
                "base_loss": base_loss.item(),
                "select_loss": 0.0,
                "alpha": 0.0,
                "total_loss": total_loss.item(),
                "avg_selected_sim": 0.0,
                "selected_neg_rank_mean": 0.0,
                "selected_neg_rank_median": 0.0,
                "pos_selected_gap": 0.0,
            }

        self.current_step += 1
        return total_loss, loss_dict


class MemoryBankLoss(nn.Module):
    """SigLIP plus hard negatives from a circular queue of older embeddings.

    ``queue_ptr`` is the next insertion location and ``queue_len`` is the valid
    entry count. Registered buffers move with the module but are not trainable.
    Stored embeddings are detached, so memory loss updates current texts but not
    examples whose embeddings were produced on previous steps.
    """

    def __init__(
        self,
        init_bias=0.0,
        alpha=0.5,
        warmup_steps=1000,
        queue_size=1024,
        top_k=32,
    ):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))
        self.alpha_max = alpha
        self.warmup_steps = warmup_steps
        self.queue_size = queue_size
        self.top_k = top_k

        self.current_step = 0

        self.register_buffer("image_queue", None)
        self.register_buffer("text_queue", None)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_len", torch.zeros(1, dtype=torch.long))

    def get_alpha(self):
        if self.current_step < self.warmup_steps:
            return 0.0
        progress = min(1.0, (self.current_step - self.warmup_steps) / 1000.0)
        return self.alpha_max * progress

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_emb, text_emb):
        """Insert B embeddings into queue_size slots, wrapping at the end."""
        B, d = image_emb.shape

        if self.image_queue is None:
            self.image_queue = torch.zeros(
                self.queue_size, d, device=image_emb.device
            )
            self.text_queue = torch.zeros(
                self.queue_size, d, device=text_emb.device
            )

        ptr = int(self.queue_ptr)

        if ptr + B > self.queue_size:
            remaining = self.queue_size - ptr
            self.image_queue[ptr:] = image_emb[:remaining]
            self.text_queue[ptr:] = text_emb[:remaining]

            overflow = B - remaining
            self.image_queue[:overflow] = image_emb[remaining:]
            self.text_queue[:overflow] = text_emb[remaining:]

            ptr = overflow
        else:
            self.image_queue[ptr : ptr + B] = image_emb
            self.text_queue[ptr : ptr + B] = text_emb
            ptr = (ptr + B) % self.queue_size

        self.queue_ptr[0] = ptr
        self.queue_len[0] = min(self.queue_size, self.queue_len[0] + B)

    def forward(self, logits, text_emb, image_emb, temp=None):
        B = logits.shape[0]

        logits_biased = logits + self.logit_bias
        labels = 2 * torch.eye(B, device=logits.device) - 1
        base_loss = -torch.sum(F.logsigmoid(labels * logits_biased)) / (B * B)

        # Detach to avoid retaining autograd graphs across optimization steps.
        self._dequeue_and_enqueue(image_emb.detach(), text_emb.detach())

        alpha = self.get_alpha()

        if alpha > 0 and self.image_queue is not None:
            filled = int(self.queue_len.item())

            if filled > B:
                queue_sim = (
                    text_emb @ self.image_queue[:filled].T
                )  # [B, filled]
                k = min(self.top_k, filled)
                topk_sims = torch.topk(queue_sim, k=k, dim=1)[0]  # [B, k]

                if temp is None:
                    with torch.no_grad():
                        raw = text_emb @ image_emb.T
                        off = ~torch.eye(
                            B, dtype=torch.bool, device=logits.device
                        )
                        scale = (
                            logits.detach()[off] / (raw.detach()[off] + 1e-8)
                        ).median()
                else:
                    scale = logits.new_tensor(1.0 / float(temp))

                hard_logits = scale * topk_sims
                memory_loss = -F.logsigmoid(-hard_logits).mean()

                total_loss = base_loss + alpha * memory_loss
                loss_dict = {
                    "base_loss": base_loss.item(),
                    "memory_loss": memory_loss.item(),
                    "alpha": float(alpha),
                    "total_loss": total_loss.item(),
                    "queue_filled": filled,
                    "avg_queue_sim": topk_sims.mean().item(),
                }
            else:
                total_loss = base_loss
                loss_dict = {
                    "base_loss": base_loss.item(),
                    "memory_loss": 0.0,
                    "alpha": 0.0,
                    "total_loss": total_loss.item(),
                    "queue_filled": filled,
                    "avg_queue_sim": 0.0,
                }
        else:
            total_loss = base_loss
            loss_dict = {
                "base_loss": base_loss.item(),
                "memory_loss": 0.0,
                "alpha": 0.0,
                "total_loss": total_loss.item(),
                "queue_filled": 0,
                "avg_queue_sim": 0.0,
            }

        self.current_step += 1
        return total_loss, loss_dict
