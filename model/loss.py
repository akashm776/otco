import torch
import torch.nn as nn
import torch.nn.functional as F


def _negative_rank_stats(raw_sim, selected_indices):
    """Rank diagnostics for chosen negatives within each batch row."""
    bsz = raw_sim.size(0)
    mask = torch.eye(bsz, device=raw_sim.device, dtype=torch.bool)
    neg_sims = raw_sim.masked_fill(mask, float('-inf'))

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


class SigLIPLoss(nn.Module):
    """Baseline SigLIP loss (no synthetic negatives)"""
    def __init__(self, init_bias=0.0):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, logits):
        n = logits.shape[0]
        logits = logits + self.logit_bias
        labels = 2 * torch.eye(n, device=logits.device) - 1
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
        B = logits.shape[0]

        # Base SigLIP loss
        logits_biased = logits + self.logit_bias
        labels = 2 * torch.eye(B, device=logits.device) - 1
        base_loss = -torch.sum(F.logsigmoid(labels * logits_biased)) / (B * B)

        alpha = self.get_alpha()

        if alpha > 0:
            mask = torch.eye(B, device=logits.device, dtype=torch.bool)
            masked_logits = logits.masked_fill(mask, float('-inf'))
            hardest_neg_sims = masked_logits.max(dim=1)[0]  # [B]
            hard_loss = -F.logsigmoid(-hardest_neg_sims).mean()

            total_loss = base_loss + alpha * hard_loss
            loss_dict = {
                'base_loss': base_loss.item(),
                'hard_loss': hard_loss.item(),
                'alpha': float(alpha),
                'total_loss': total_loss.item()
            }
        else:
            total_loss = base_loss
            loss_dict = {
                'base_loss': base_loss.item(),
                'hard_loss': 0.0,
                'alpha': 0.0,
                'total_loss': total_loss.item()
            }

        self.current_step += 1
        return total_loss, loss_dict


class SoftmaxMixLoss(nn.Module):
    """
    Paper-faithful OT-Mix:
    - local top-k support per text anchor
    - masked entropy-regularized OT (Sinkhorn) over batch
    - barycentric synthetic negatives with stop-grad through OT plan
    - periodic OT-plan updates
    """
    def __init__(self, init_bias=0.0, alpha=0.5, warmup_steps=1000,
                 top_k=32, tau=0.05, update_freq=10, gate_sim=-0.05,
                 ot_eps=0.05, sinkhorn_iters=30):
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

        self.current_step = 0
        self.cached_plan = None  # [B, B] OT coupling
        self.cached_local_mask = None

    def get_alpha(self):
        if self.current_step < self.warmup_steps:
            return 0.0
        progress = min(1.0, (self.current_step - self.warmup_steps) / 1000.0)
        return self.alpha_max * progress

    def forward(self, logits, text_emb, image_emb, temp=None):
        B = logits.shape[0]

        # Safety: if batch size changes, invalidate cached plan
        if self.cached_plan is not None and self.cached_plan.size(0) != B:
            self.cached_plan = None
            self.cached_local_mask = None

        num_gated = 0
        selected_rank = 0.0
        selected_rank_median = 0.0
        selected_sim_mean = 0.0
        pos_selected_gap = 0.0
        coupling_entropy = 0.0
        coupling_peak_mass = 0.0

        # Base SigLIP loss
        logits_biased = logits + self.logit_bias
        labels = 2 * torch.eye(B, device=logits.device) - 1
        base_loss = -torch.sum(F.logsigmoid(labels * logits_biased)) / (B * B)

        alpha = self.get_alpha()
        synthetic_loss = logits.new_tensor(0.0)
        avg_synth_sim = logits.new_tensor(0.0)
        avg_synth_logit = logits.new_tensor(0.0)

        if alpha > 0:
            # Update cached OT plan periodically (no grad through plan)
            if (self.current_step % self.update_freq == 0) or (self.cached_plan is None):
                with torch.no_grad():
                    plan, local_mask = self._make_plan(text_emb, image_emb, logits_biased)
                self.cached_plan = plan.detach()
                self.cached_local_mask = local_mask

            # Rebuild synthetic negative with gradients only to image embeddings
            row_mass = self.cached_plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
            row_weights = self.cached_plan / row_mass
            synthetic_neg = row_weights @ image_emb  # [B, d]
            synthetic_neg = synthetic_neg / (synthetic_neg.norm(dim=1, keepdim=True) + 1e-8)

            # Exact scale
            if temp is None:
                with torch.no_grad():
                    raw = text_emb @ image_emb.T
                    off = ~torch.eye(B, dtype=torch.bool, device=logits.device)
                    scale = (logits.detach()[off] / (raw.detach()[off] + 1e-8)).median()
            else:
                scale = logits.new_tensor(1.0 / float(temp))

            synth_sim = (text_emb * synthetic_neg).sum(dim=1)  # [B]
            synth_logits = scale * synth_sim

            # Gate in logit space: skip synthetics easier than gate_sim threshold
            gate = (synth_logits > self.gate_sim).float()
            num_gated = int(gate.sum().item())

            if num_gated > 0:
                synthetic_loss = (-F.logsigmoid(-synth_logits) * gate).sum() / (gate.sum() + 1e-8)
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

                local_mass = self.cached_plan * self.cached_local_mask.float()
                local_mass = local_mass / local_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
                row_entropy = -(local_mass * (local_mass.clamp_min(1e-12)).log()).sum(dim=1)
                coupling_entropy = row_entropy.mean().item()
                coupling_peak_mass = local_mass.max(dim=1).values.mean().item()

        total_loss = base_loss + alpha * synthetic_loss

        loss_dict = {
            'base_loss': base_loss.item(),
            'synthetic_loss': synthetic_loss.item(),
            'alpha': float(alpha),
            'total_loss': total_loss.item(),
            'avg_synthetic_sim': avg_synth_sim.item(),
            'avg_synthetic_logit': avg_synth_logit.item(),
            'num_gated': num_gated if alpha > 0 else 0,
            'ot_step': self.current_step,
            'selected_neg_rank_mean': selected_rank,
            'selected_neg_rank_median': selected_rank_median,
            'selected_neg_sim': selected_sim_mean,
            'pos_selected_gap': pos_selected_gap,
            'coupling_entropy': coupling_entropy,
            'coupling_peak_mass': coupling_peak_mass,
        }

        self.current_step += 1
        return total_loss, loss_dict

    def _make_plan(self, text_emb, image_emb, logits_biased):
        B = text_emb.size(0)
        k = min(self.top_k, B - 1) if B > 1 else 1

        diag_mask = torch.eye(B, device=logits_biased.device, dtype=torch.bool)

        # Top-k selection in logit space (ranking identical to cosine since scale > 0)
        masked_logits = logits_biased.masked_fill(diag_mask, float('-inf'))
        _, topk_indices = torch.topk(masked_logits, k=k, dim=1)
        local_mask = torch.zeros(B, B, dtype=torch.bool, device=logits_biased.device)
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
    """
    OT-Select: Use softmax weights to SELECT the hardest negative (not mix).
    """
    def __init__(self, init_bias=0.0, alpha=0.1, warmup_steps=1000,
                 top_k=32, tau=0.05):
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
            raw_sim_masked = raw_sim.masked_fill(mask, float('-inf'))

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
                    scale = (logits.detach()[off] / (raw.detach()[off] + 1e-8)).median()
            else:
                scale = logits.new_tensor(1.0 / float(temp))

            selected_sim = (text_emb * selected_neg_emb).sum(dim=1)
            selected_logits = scale * selected_sim
            select_loss = -F.logsigmoid(-selected_logits).mean()

            total_loss = base_loss + alpha * select_loss
            rank_stats = _negative_rank_stats(raw_sim, selected_neg_indices)
            loss_dict = {
                'base_loss': base_loss.item(),
                'select_loss': select_loss.item(),
                'alpha': float(alpha),
                'total_loss': total_loss.item(),
                'avg_selected_sim': selected_sim.mean().item(),
                'selected_neg_rank_mean': rank_stats["mean_selected_rank"],
                'selected_neg_rank_median': rank_stats["median_selected_rank"],
                'pos_selected_gap': rank_stats["pos_selected_gap"],
            }
        else:
            total_loss = base_loss
            loss_dict = {
                'base_loss': base_loss.item(),
                'select_loss': 0.0,
                'alpha': 0.0,
                'total_loss': total_loss.item(),
                'avg_selected_sim': 0.0,
                'selected_neg_rank_mean': 0.0,
                'selected_neg_rank_median': 0.0,
                'pos_selected_gap': 0.0,
            }

        self.current_step += 1
        return total_loss, loss_dict


class MemoryBankLoss(nn.Module):
    """
    Memory bank for more negative diversity.

    FIXED:
    - tracks queue_len (not confused with ptr)
    - can use exact scale from model.temp
    """
    def __init__(self, init_bias=0.0, alpha=0.5, warmup_steps=1000,
                 queue_size=1024, top_k=32):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))
        self.alpha_max = alpha
        self.warmup_steps = warmup_steps
        self.queue_size = queue_size
        self.top_k = top_k

        self.current_step = 0

        self.register_buffer('image_queue', None)
        self.register_buffer('text_queue', None)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_len', torch.zeros(1, dtype=torch.long))

    def get_alpha(self):
        if self.current_step < self.warmup_steps:
            return 0.0
        progress = min(1.0, (self.current_step - self.warmup_steps) / 1000.0)
        return self.alpha_max * progress

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_emb, text_emb):
        B, d = image_emb.shape

        if self.image_queue is None:
            self.image_queue = torch.zeros(self.queue_size, d, device=image_emb.device)
            self.text_queue = torch.zeros(self.queue_size, d, device=text_emb.device)

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
            self.image_queue[ptr:ptr + B] = image_emb
            self.text_queue[ptr:ptr + B] = text_emb
            ptr = (ptr + B) % self.queue_size

        self.queue_ptr[0] = ptr
        self.queue_len[0] = min(self.queue_size, self.queue_len[0] + B)

    def forward(self, logits, text_emb, image_emb, temp=None):
        B = logits.shape[0]

        logits_biased = logits + self.logit_bias
        labels = 2 * torch.eye(B, device=logits.device) - 1
        base_loss = -torch.sum(F.logsigmoid(labels * logits_biased)) / (B * B)

        # Update memory
        self._dequeue_and_enqueue(image_emb.detach(), text_emb.detach())

        alpha = self.get_alpha()

        if alpha > 0 and self.image_queue is not None:
            filled = int(self.queue_len.item())

            if filled > B:
                queue_sim = text_emb @ self.image_queue[:filled].T  # [B, filled]
                k = min(self.top_k, filled)
                topk_sims = torch.topk(queue_sim, k=k, dim=1)[0]     # [B, k]

                if temp is None:
                    with torch.no_grad():
                        raw = text_emb @ image_emb.T
                        off = ~torch.eye(B, dtype=torch.bool, device=logits.device)
                        scale = (logits.detach()[off] / (raw.detach()[off] + 1e-8)).median()
                else:
                    scale = logits.new_tensor(1.0 / float(temp))

                hard_logits = scale * topk_sims
                memory_loss = -F.logsigmoid(-hard_logits).mean()

                total_loss = base_loss + alpha * memory_loss
                loss_dict = {
                    'base_loss': base_loss.item(),
                    'memory_loss': memory_loss.item(),
                    'alpha': float(alpha),
                    'total_loss': total_loss.item(),
                    'queue_filled': filled,
                    'avg_queue_sim': topk_sims.mean().item()
                }
            else:
                total_loss = base_loss
                loss_dict = {
                    'base_loss': base_loss.item(),
                    'memory_loss': 0.0,
                    'alpha': 0.0,
                    'total_loss': total_loss.item(),
                    'queue_filled': filled,
                    'avg_queue_sim': 0.0
                }
        else:
            total_loss = base_loss
            loss_dict = {
                'base_loss': base_loss.item(),
                'memory_loss': 0.0,
                'alpha': 0.0,
                'total_loss': total_loss.item(),
                'queue_filled': 0,
                'avg_queue_sim': 0.0
            }

        self.current_step += 1
        return total_loss, loss_dict
