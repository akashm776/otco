import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel

from src.config_loader import load_run_config
from model.loss import HardNegativeLoss, MemoryBankLoss, OTSelectLoss, SigLIPLoss, SoftmaxMixLoss
from model.model import OTLIP
from src.data_setup import build_data_bundle
from src.utils import (
    compute_embedding_stats,
    compute_retrieval_metrics,
    evaluate_image_to_text_retrieval,
    evaluate_retrieval,
    get_device,
    log_experiment,
    print_experiment_comparison,
    sanity_check_eval,
    save_checkpoint,
)


def _image_only_collate(batch):
    """Lightweight collate for pool building — images and IDs only, no tokenization."""
    images = torch.stack([item["image"] for item in batch])
    image_ids = torch.tensor([item["image_id"] for item in batch], dtype=torch.long)
    return {"images": images, "image_id": image_ids}


@torch.no_grad()
def _build_image_pool(model, train_dataset, config, device):
    """Forward all training images at epoch start and return cached embeddings.

    Returns:
        pool_feats: [N_train, d] normalized image embeddings on device
        pool_image_ids: [N_train] integer image IDs on device
    """
    pool_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=_image_only_collate,
        num_workers=config.get("num_workers", 4),
        drop_last=False,
    )
    model.eval()
    all_feats = []
    all_ids = []
    for batch in pool_loader:
        images = batch["images"].to(device)
        proj = model.encode_images(images)
        feats = F.normalize(proj, dim=-1)
        all_feats.append(feats)
        all_ids.append(batch["image_id"].to(device))
    model.train()
    return torch.cat(all_feats, dim=0), torch.cat(all_ids, dim=0)


def _select_live_contributors(plan, chosen_idx, top_m, max_live, device):
    """Select pool contributors for live re-forward.

    Returns:
        unique_pool_idx [M]: indices into pool_feats (== dataset indices since shuffle=False)
        live_local_idx [M]:  indices into the N-candidate slice (0..N-1) for each unique contributor
    M <= max_live.
    """
    B, N = plan.shape
    actual_top_m = min(top_m, N)

    # Top-m local indices per query: [B, actual_top_m]
    top_local = plan.topk(actual_top_m, dim=1).indices

    # Map to pool_feats global indices: [B, actual_top_m]
    top_pool = chosen_idx[top_local.reshape(-1)].reshape(top_local.shape)

    flat_pool = top_pool.reshape(-1)                          # [B * actual_top_m]
    unique_pool_idx, inverse_idx = torch.unique(flat_pool, return_inverse=True)
    M = len(unique_pool_idx)

    if M > max_live:
        # Keep top-max_live by aggregate OT mass across all queries
        row_idx = torch.arange(B, device=device).repeat_interleave(actual_top_m)
        flat_mass = plan[row_idx, top_local.reshape(-1)]      # [B * actual_top_m]
        agg_mass = torch.zeros(M, device=device)
        agg_mass.scatter_add_(0, inverse_idx, flat_mass)
        _, keep_k = agg_mass.topk(max_live)
        unique_pool_idx = unique_pool_idx[keep_k]
        M = max_live

    # Find local (0..N-1) index for each unique pool entry
    eq = (chosen_idx.unsqueeze(0) == unique_pool_idx.unsqueeze(1))  # [M, N]
    live_local_idx = eq.long().argmax(dim=1)                        # [M]
    return unique_pool_idx, live_local_idx


def parse_args():
    parser = argparse.ArgumentParser(description="Train OTCLIP using YAML configuration.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to run configuration YAML file")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory (takes priority over config)")
    return parser.parse_args()


def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_model_and_optimizer(
    vision_model,
    text_model,
    device,
    unfreeze_strategy="projection_only",
    projection_lr=1e-3,
    encoder_lr=5e-5,
    weight_decay=0.01,
):
    model = OTLIP(vision_model, text_model, device=device).to(device)

    for encoder in (model.vision_model, model.text_model):
        if getattr(encoder, 'supports_gradient_checkpointing', False):
            encoder.gradient_checkpointing_enable()
            print(f"Gradient checkpointing enabled: {type(encoder).__name__}")

    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.text_model.parameters():
        param.requires_grad = False

    if unfreeze_strategy == "projection_only":
        print("Strategy: Training projection heads only")
        param_groups = [
            {"params": model.vision_proj.parameters(), "lr": projection_lr},
            {"params": model.text_proj.parameters(), "lr": projection_lr},
        ]
    elif unfreeze_strategy == "text_last_layer":
        print("Strategy: Projection heads + text encoder last layer")
        for param in model.text_model.transformer.layer[-1].parameters():
            param.requires_grad = True
        param_groups = [
            {"params": model.vision_proj.parameters(), "lr": projection_lr},
            {"params": model.text_proj.parameters(), "lr": projection_lr},
            {"params": model.text_model.transformer.layer[-1].parameters(), "lr": encoder_lr},
        ]
    elif unfreeze_strategy == "vision_last_layer":
        print("Strategy: Projection heads + vision encoder last layer")
        for param in model.vision_model.encoder.stages[-1].parameters():
            param.requires_grad = True
        param_groups = [
            {"params": model.vision_proj.parameters(), "lr": projection_lr},
            {"params": model.text_proj.parameters(), "lr": projection_lr},
            {"params": model.vision_model.encoder.stages[-1].parameters(), "lr": encoder_lr},
        ]
    elif unfreeze_strategy == "both_last_layer":
        print("Strategy: Projection heads + both encoders' last layers")
        for param in model.text_model.transformer.layer[-1].parameters():
            param.requires_grad = True
        for param in model.vision_model.encoder.stages[-1].parameters():
            param.requires_grad = True
        param_groups = [
            {"params": model.vision_proj.parameters(), "lr": projection_lr},
            {"params": model.text_proj.parameters(), "lr": projection_lr},
            {"params": model.text_model.transformer.layer[-1].parameters(), "lr": encoder_lr},
            {"params": model.vision_model.encoder.stages[-1].parameters(), "lr": encoder_lr},
        ]
    else:
        raise ValueError(f"Unknown unfreeze_strategy: {unfreeze_strategy}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs: parallelizing encoders only")
        model.vision_model = torch.nn.DataParallel(model.vision_model)
        model.text_model = torch.nn.DataParallel(model.text_model)

    return model, optimizer


def build_loss(config, device):
    print(f"\nInitializing loss: {config['loss_type']}")
    if config["loss_type"] == "baseline":
        return SigLIPLoss().to(device)
    if config["loss_type"] == "hard_negative":
        return HardNegativeLoss(
            alpha=config.get("alpha", 0.5),
            warmup_steps=config.get("warmup_steps", 1000),
        ).to(device)
    if config["loss_type"] == "softmax_mix":
        return SoftmaxMixLoss(
            alpha=config.get("alpha", 0.5),
            warmup_steps=config.get("warmup_steps", 1000),
            top_k=config.get("top_k", 32),
            tau=config.get("tau", 0.05),
            update_freq=config.get("update_freq", 10),
            gate_sim=config.get("gate_sim", -0.05),
            ot_eps=config.get("ot_eps", 0.05),
            sinkhorn_iters=config.get("sinkhorn_iters", 30),
            adaptive_warmup=config.get("adaptive_warmup", False),
            entropy_threshold=config.get("entropy_threshold", 3.0),
            entropy_check_freq=config.get("entropy_check_freq", 100),
            gap_suppress_easy=config.get("gap_suppress_easy", 0.10),
            gap_downweight_hard=config.get("gap_downweight_hard", -0.07),
            hard_alpha_scale=config.get("hard_alpha_scale", 0.25),
        ).to(device)
    if config["loss_type"] == "ot_select":
        return OTSelectLoss(
            alpha=config.get("alpha", 0.1),
            warmup_steps=config.get("warmup_steps", 1000),
            top_k=config.get("top_k", 32),
            tau=config.get("tau", 0.05),
        ).to(device)
    if config["loss_type"] == "memory_bank":
        return MemoryBankLoss(
            alpha=config.get("alpha", 0.5),
            warmup_steps=config.get("warmup_steps", 1000),
            queue_size=config.get("queue_size", 1024),
            top_k=config.get("top_k", 32),
        ).to(device)
    raise ValueError(f"Unknown loss type: {config['loss_type']}")


def main():
    args = parse_args()
    run_config = load_run_config(args.config)
    config = run_config["experiment_config"]
    run_section = run_config["run"]
    dataset_cfg = run_config["dataset"]
    experiment_name = run_config["experiment_name"]

    config["config_file"] = run_config["config_path"]
    config["experiments_file"] = run_config["experiments_path"]

    set_global_seed(config["seed"])

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'=' * 80}")
    print(f"  config_file: {run_config['config_path']}")
    print(f"  experiments_file: {run_config['experiments_path']}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  dataset_backend: {dataset_cfg.get('backend', 'local_flickr8k')}")
    print(f"{'=' * 80}\n")

    checkpoint_dir = args.checkpoint_dir or run_section.get("checkpoint_dir") or f"checkpoints/{experiment_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    root_dir = os.path.abspath(os.curdir)
    print("Building dataset and dataloaders...")
    data_bundle = build_data_bundle(
        config=config,
        root_dir=root_dir,
        dataset_cfg=dataset_cfg,
    )

    train_dataset = data_bundle.train_dataset
    train_loader = data_bundle.train_loader
    val_loader_all = data_bundle.val_loader_all
    val_loader_canonical = data_bundle.val_loader_canonical
    steps_per_epoch = data_bundle.stats["steps_per_epoch"]

    print(f"\n{'=' * 80}")
    print("DATASET VERIFICATION")
    print(f"{'=' * 80}")
    for key in sorted(data_bundle.stats.keys()):
        print(f"  {key}: {data_bundle.stats[key]}")
    print(f"{'=' * 80}\n")

    print("Initializing models...")
    vision_model = AutoModel.from_pretrained(config["model_vision"])
    text_model = AutoModel.from_pretrained(config["model_text"])
    device = get_device()

    model, optimizer = setup_model_and_optimizer(
        vision_model,
        text_model,
        device,
        unfreeze_strategy=config["unfreeze_strategy"],
        projection_lr=config.get("projection_lr", 1e-3),
        encoder_lr=config.get("encoder_lr", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )
    criterion = build_loss(config, device)

    print("\nRunning sanity check on canonical validation data...")
    sanity_check_eval(model, val_loader_canonical, device)

    best_avg_recall = 0.0
    global_step = 0
    start_epoch = 0

    latest_path = f"{checkpoint_dir}/latest.pt"
    if os.path.exists(latest_path):
        print(f"\nResuming from checkpoint: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_avg_recall = ckpt.get("best_avg_recall", 0.0)
        if hasattr(criterion, "current_step") and "criterion_step" in ckpt:
            criterion.current_step = ckpt["criterion_step"]
        if hasattr(criterion, "ot_ready") and "ot_ready" in ckpt:
            criterion.ot_ready = ckpt["ot_ready"]
        if hasattr(criterion, "steps_since_ready") and "steps_since_ready" in ckpt:
            criterion.steps_since_ready = ckpt["steps_since_ready"]
        if hasattr(criterion, "cached_plan") and "cached_plan" in ckpt and ckpt["cached_plan"] is not None:
            criterion.cached_plan = ckpt["cached_plan"].to(device)
        if hasattr(criterion, "cached_local_mask") and "cached_local_mask" in ckpt and ckpt["cached_local_mask"] is not None:
            criterion.cached_local_mask = ckpt["cached_local_mask"].to(device)
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, best R@1 {best_avg_recall:.2f}%")
        print(f"  Loss current_step restored to: {getattr(criterion, 'current_step', 'n/a')}")
        print(f"  OT ready restored to: {getattr(criterion, 'ot_ready', 'n/a')}")
        print(f"  Cached plan restored: {getattr(criterion, 'cached_plan', None) is not None}")

    pool_size = config.get("pool_size", 0)

    for epoch in range(start_epoch, config["num_epochs"]):
        if hasattr(criterion, "cached_plan"):
            criterion.cached_plan = None
            criterion.cached_local_mask = None

        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch + 1}/{config['num_epochs']}")
        print(f"{'=' * 80}")

        # Build epoch image pool for large-pool OT (Option A: epoch cache)
        pool_feats = None
        pool_image_ids = None
        if pool_size > 0:
            print(f"  Building image pool (pool_size={pool_size}, N_train={len(train_dataset)})...")
            pool_feats, pool_image_ids = _build_image_pool(model, train_dataset, config, device)
            print(f"  Pool cached: {pool_feats.shape[0]} embeddings on {device}")

        train_dataset.set_epoch(epoch)
        if data_bundle.stratified_sampler is not None:
            data_bundle.stratified_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            text_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            image_batch = batch["images"].to(device)
            batch_size = image_batch.size(0)

            # Sample pool for large-pool OT, excluding current batch positives.
            # Build eligible set FIRST (exclude batch positives), THEN sample exactly
            # pool_size from it. Invariant: image_pool.shape[0] == pool_size whenever
            # N_train - batch_size >= pool_size (always true for CUB-200).
            image_pool = None
            chosen_idx = None
            if pool_size > 0 and pool_feats is not None:
                batch_ids = batch["image_ids"].to(device)
                exclude_mask = torch.isin(pool_image_ids, batch_ids)
                available_idx = (~exclude_mask).nonzero(as_tuple=True)[0]
                n_sample = min(pool_size, len(available_idx))
                if n_sample < pool_size:
                    print(f"  [WARN] pool: only {n_sample}/{pool_size} eligible images after positive exclusion")
                perm = torch.randperm(len(available_idx), device=device)[:n_sample]
                chosen_idx = available_idx[perm]
                image_pool = pool_feats[chosen_idx]  # [n_sample, d], detached

            logits, image_features, text_features = model(image_batch, text_batch)

            # Phase III v2: live re-forward of top OT contributors.
            # Uses cached pool for candidate discovery (no_grad), then re-forwards the
            # top-m contributors per query through the live image encoder (with gradient).
            # Pool position k == dataset index k (DataLoader shuffle=False, no drop_last).
            live_feats = None
            live_local_idx_v2 = None
            precomputed_plan = None
            precomputed_local_mask = None
            num_live = 0
            if (pool_size > 0 and config.get("live_reforward", False)
                    and pool_feats is not None and image_pool is not None
                    and criterion.get_alpha() > 0):
                scale_val = 1.0 / float(model.temp.item())
                with torch.no_grad():
                    pool_logits = (text_features @ image_pool.T * scale_val
                                   + criterion.logit_bias.detach())
                    precomputed_plan, precomputed_local_mask = criterion._make_plan_pool(pool_logits)

                unique_pool_idx, live_local_idx_v2 = _select_live_contributors(
                    precomputed_plan, chosen_idx,
                    top_m=config.get("live_top_m", 2),
                    max_live=config.get("max_live_reforward", 64),
                    device=device,
                )
                num_live = len(unique_pool_idx)
                ds_indices = unique_pool_idx.cpu().tolist()
                live_imgs = torch.stack([train_dataset[i]["image"] for i in ds_indices]).to(device)
                live_proj = model.encode_images(live_imgs)
                live_feats = F.normalize(live_proj, dim=-1)  # [M, d], gradient enabled

            if config["loss_type"] == "baseline":
                loss = criterion(logits)
                loss_dict = {"base_loss": loss.item(), "total_loss": loss.item()}
            else:
                loss, loss_dict = criterion(
                    logits, text_features, image_features, temp=model.temp,
                    image_pool=image_pool,
                    precomputed_plan=precomputed_plan,
                    precomputed_local_mask=precomputed_local_mask,
                    live_feats=live_feats,
                    live_local_idx=live_local_idx_v2,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_norm"])
            optimizer.step()
            epoch_loss += loss.item()

            if global_step % 100 == 0 or batch_idx == len(train_loader) - 1:
                mean_diag = logits.diag().mean()
                mean_offdiag = (logits.sum() - logits.diag().sum()) / (batch_size**2 - batch_size)
                diag_std = logits.diag().std()
                retrieval = compute_retrieval_metrics(logits)
                emb_stats = compute_embedding_stats(image_features, text_features)

                print(f"\n{'-' * 80}")
                print(f"Step {global_step} (Epoch {epoch + 1}, Batch {batch_idx + 1}/{steps_per_epoch})")
                print(f"{'-' * 80}")
                print(f"  Total Loss: {loss_dict['total_loss']:.4f}")

                if "base_loss" in loss_dict:
                    print(f"  Base Loss: {loss_dict['base_loss']:.4f}")
                if "hard_loss" in loss_dict and loss_dict.get("hard_loss", 0) > 0:
                    print(f"  Hard Negative Loss: {loss_dict['hard_loss']:.4f}")
                if "synthetic_loss" in loss_dict:
                    print(f"\n  OT-MIX STATE")
                    print(f"    OT Step (loss counter): {loss_dict.get('ot_step', '?')}")
                    alpha_key = 'alpha_scheduled' if 'alpha_scheduled' in loss_dict else 'alpha'
                    print(f"    Alpha:                  {loss_dict[alpha_key]:.4f}")
                    ot_active = loss_dict[alpha_key] > 0
                    print(f"    OT Active:              {ot_active}")
                    if 'ot_ready' in loss_dict:
                        print(f"    OT Ready (adaptive):    {bool(loss_dict['ot_ready'])}")
                    warmup_ent = loss_dict.get('warmup_entropy', 0.0)
                    if warmup_ent and warmup_ent == warmup_ent:  # not nan, not 0
                        print(f"    Warmup Plan Entropy:    {warmup_ent:.4f}  (threshold={criterion.entropy_threshold:.2f})")
                    if ot_active:
                        print(f"    Synthetic Loss:         {loss_dict['synthetic_loss']:.6f}")
                        print(f"    Num Gated (active/B):   {loss_dict.get('num_gated', 0)}/{loss_dict.get('num_gated', 0) if loss_dict.get('num_gated', 0) > 0 else 'B'}")
                        print(f"    Avg Synthetic Sim:      {loss_dict.get('avg_synthetic_sim', 0):.4f}")
                        print(f"    Avg Synthetic Logit:    {loss_dict.get('avg_synthetic_logit', 0):.4f}")
                        if loss_dict.get('selected_neg_rank_mean', 0) > 0:
                            print(f"    Selected Neg Rank:      mean={loss_dict['selected_neg_rank_mean']:.2f}  median={loss_dict.get('selected_neg_rank_median', 0):.2f}")
                            print(f"    Pos - Selected Gap:     {loss_dict.get('pos_selected_gap', 0):.4f}")
                        if loss_dict.get('pool_size', 0) > 0:
                            print(f"    Pool Mode:              pool (N={loss_dict['pool_size']})")
                        if num_live > 0:
                            print(f"    Live Re-forward:        {num_live} contributors")
                            mr = loss_dict.get('mass_retained', 1.0)
                            print(f"    Mass Retained:          {mr:.4f}")
                        if loss_dict.get('coupling_entropy', 0) > 0:
                            print(f"    Coupling Entropy:       {loss_dict['coupling_entropy']:.4f}")
                            print(f"    Coupling Peak Mass:     {loss_dict.get('coupling_peak_mass', 0):.4f}")
                        if 'alpha_effective' in loss_dict:
                            _bucket_names = {0: "useful", 1: "too_easy", 2: "too_hard", 3: "diffuse", 4: "inactive"}
                            a_eff = loss_dict['alpha_effective']
                            a_sched = loss_dict.get('alpha_scheduled', loss_dict.get('alpha', 0))
                            bucket_id = loss_dict.get('gap_bucket_id', -1)
                            flag = ""
                            if loss_dict.get('ot_suppressed_entropy'):
                                flag = "  [suppressed: diffuse plan]"
                            elif loss_dict.get('ot_suppressed_too_easy'):
                                flag = "  [suppressed: too easy]"
                            elif loss_dict.get('ot_downweighted_too_hard'):
                                flag = "  [downweighted: too hard]"
                            print(f"    Alpha Effective:        {a_eff:.4f} (scheduled={a_sched:.4f}){flag}")
                            print(f"    Gap Bucket:             {_bucket_names.get(bucket_id, '?')}")
                if "select_loss" in loss_dict and loss_dict.get("select_loss", 0) > 0:
                    print(f"  OT-Select Loss: {loss_dict['select_loss']:.4f}")
                    print(f"  Avg Selected Sim: {loss_dict.get('avg_selected_sim', 0):.4f}")
                if "memory_loss" in loss_dict and loss_dict.get("memory_loss", 0) > 0:
                    print(f"  Memory Bank Loss: {loss_dict['memory_loss']:.4f}")
                    print(f"  Queue Filled: {loss_dict.get('queue_filled', 0)}/{config.get('queue_size', 0)}")
                    print(f"  Avg Queue Sim: {loss_dict.get('avg_queue_sim', 0):.4f}")

                with torch.no_grad():
                    raw_cos = text_features @ image_features.T
                    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=raw_cos.device)
                    raw_pos = raw_cos.diagonal()
                    raw_hardest = raw_cos.masked_fill(diag_mask, float('-inf')).max(dim=1).values
                    raw_all_neg = raw_cos[~diag_mask]
                print("\n  RAW COSINE SIMILARITY")
                print(f"    Positive pairs: {raw_pos.mean().item():.4f} ± {raw_pos.std().item():.4f}")
                print(f"    All negatives:  {raw_all_neg.mean().item():.4f}")
                print(f"    Hardest neg:    {raw_hardest.mean().item():.4f}")
                print(f"    Pos-Hard gap:   {(raw_pos - raw_hardest).mean().item():.4f}")

                print("\n  LOGIT STATISTICS")
                print(f"    Diagonal mean: {mean_diag.item():.4f} ± {diag_std.item():.4f}")
                print(f"    Off-diagonal mean: {mean_offdiag.item():.4f}")
                print(f"    Gap (pos - neg): {(mean_diag - mean_offdiag).item():.4f}")
                print("\n  BATCH RETRIEVAL (training proxy)")
                print(f"    Recall@1: {retrieval['recall_at_1']:.1f}%")
                print(f"    Mean rank: {retrieval['mean_rank']:.2f}")
                print("\n  EMBEDDING HEALTH")
                print(f"    Image variance: {emb_stats['img_variance']:.6f}")
                print(f"    Text variance: {emb_stats['txt_variance']:.6f}")

            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\n{'=' * 80}")
        print(f"VALIDATION - Epoch {epoch + 1}")
        print(f"{'=' * 80}")
        print(f"Average training loss: {avg_train_loss:.4f}")

        val_metrics_t2i_all = evaluate_retrieval(model, val_loader_all, device)
        val_metrics_i2t_all = evaluate_image_to_text_retrieval(model, val_loader_all, device)
        val_metrics_t2i_canonical = evaluate_retrieval(model, val_loader_canonical, device)
        val_metrics_i2t_canonical = evaluate_image_to_text_retrieval(model, val_loader_canonical, device)

        print(f"\n📊 ALL CAPTIONS ({val_metrics_t2i_all['num_captions']} captions, {val_metrics_t2i_all['num_images']} images)")
        print("  Text → Image:")
        print(f"    R@1:  {val_metrics_t2i_all['R@1']:.2f}%")
        print(f"    R@5:  {val_metrics_t2i_all['R@5']:.2f}%")
        print(f"    R@10: {val_metrics_t2i_all['R@10']:.2f}%")
        print("  Image → Text:")
        print(f"    R@1:  {val_metrics_i2t_all['R@1']:.2f}%")
        print(f"    R@5:  {val_metrics_i2t_all['R@5']:.2f}%")
        print(f"    R@10: {val_metrics_i2t_all['R@10']:.2f}%")

        print(f"\n📌 CANONICAL (first caption only, {val_metrics_t2i_canonical['num_captions']} samples)")
        print("  Text → Image:")
        print(f"    R@1:  {val_metrics_t2i_canonical['R@1']:.2f}%")
        print(f"    R@5:  {val_metrics_t2i_canonical['R@5']:.2f}%")
        print(f"    R@10: {val_metrics_t2i_canonical['R@10']:.2f}%")
        print("  Image → Text:")
        print(f"    R@1:  {val_metrics_i2t_canonical['R@1']:.2f}%")
        print(f"    R@5:  {val_metrics_i2t_canonical['R@5']:.2f}%")
        print(f"    R@10: {val_metrics_i2t_canonical['R@10']:.2f}%")

        avg_recall_canonical = (val_metrics_t2i_canonical["R@1"] + val_metrics_i2t_canonical["R@1"]) / 2
        print(f"\n🎯 Average Recall@1 (canonical): {avg_recall_canonical:.2f}%")

        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_train_loss,
            "best_avg_recall": best_avg_recall,
            "criterion_step": getattr(criterion, "current_step", 0),
            "ot_ready": getattr(criterion, "ot_ready", True),
            "steps_since_ready": getattr(criterion, "steps_since_ready", 0),
            "cached_plan": getattr(criterion, "cached_plan", None),
            "cached_local_mask": getattr(criterion, "cached_local_mask", None),
        }, latest_path)
        print(f"  Latest checkpoint saved (epoch {epoch + 1})")

        if avg_recall_canonical > best_avg_recall:
            best_avg_recall = avg_recall_canonical
            save_checkpoint(
                model,
                optimizer,
                global_step,
                avg_train_loss,
                f"{checkpoint_dir}/best_model.pt",
            )
            print(f"\n✓ New best model! Canonical Avg R@1 = {best_avg_recall:.2f}%")

        print(f"Current best canonical Avg R@1: {best_avg_recall:.2f}%")

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE - FINAL EVALUATION")
    print(f"{'=' * 80}")

    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_all_t2i = evaluate_retrieval(model, val_loader_all, device)
    final_all_i2t = evaluate_image_to_text_retrieval(model, val_loader_all, device)
    final_canonical_t2i = evaluate_retrieval(model, val_loader_canonical, device)
    final_canonical_i2t = evaluate_image_to_text_retrieval(model, val_loader_canonical, device)

    print("\n📊 FINAL RESULTS - ALL CAPTIONS")
    print(
        f"  Text → Image: R@1={final_all_t2i['R@1']:.2f}%, R@5={final_all_t2i['R@5']:.2f}%, "
        f"R@10={final_all_t2i['R@10']:.2f}%"
    )
    print(
        f"  Image → Text: R@1={final_all_i2t['R@1']:.2f}%, R@5={final_all_i2t['R@5']:.2f}%, "
        f"R@10={final_all_i2t['R@10']:.2f}%"
    )

    print("\n📌 FINAL RESULTS - CANONICAL (OFFICIAL METRIC)")
    print(
        f"  Text → Image: R@1={final_canonical_t2i['R@1']:.2f}%, R@5={final_canonical_t2i['R@5']:.2f}%, "
        f"R@10={final_canonical_t2i['R@10']:.2f}%"
    )
    print(
        f"  Image → Text: R@1={final_canonical_i2t['R@1']:.2f}%, R@5={final_canonical_i2t['R@5']:.2f}%, "
        f"R@10={final_canonical_i2t['R@10']:.2f}%"
    )
    print(f"  Average R@1: {(final_canonical_t2i['R@1'] + final_canonical_i2t['R@1']) / 2:.2f}%")

    final_metrics = {
        "all_captions": {
            "text_to_image": {
                "R@1": final_all_t2i["R@1"],
                "R@5": final_all_t2i["R@5"],
                "R@10": final_all_t2i["R@10"],
            },
            "image_to_text": {
                "R@1": final_all_i2t["R@1"],
                "R@5": final_all_i2t["R@5"],
                "R@10": final_all_i2t["R@10"],
            },
        },
        "canonical": {
            "text_to_image": {
                "R@1": final_canonical_t2i["R@1"],
                "R@5": final_canonical_t2i["R@5"],
                "R@10": final_canonical_t2i["R@10"],
            },
            "image_to_text": {
                "R@1": final_canonical_i2t["R@1"],
                "R@5": final_canonical_i2t["R@5"],
                "R@10": final_canonical_i2t["R@10"],
            },
            "average_recall_at_1": (final_canonical_t2i["R@1"] + final_canonical_i2t["R@1"]) / 2,
        },
    }

    log_experiment(config, final_metrics)
    print(f"\n{'=' * 80}")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"{'=' * 80}\n")

    if run_section.get("print_experiment_comparison", True):
        try:
            print_experiment_comparison()
        except Exception as exc:
            print(f"\n⚠️  Could not generate comparison (this is normal for first run): {exc}")


if __name__ == "__main__":
    main()
