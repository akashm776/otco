import os
import csv
import json
import math
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

from model.model import OTLIP
from src.config_loader import load_diagnostic_config


ROOT_DIR = Path(__file__).resolve().parents[1]


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================
def load_model_checkpoint(model, filepath, device):
    """Load model weights only for inference."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    step = checkpoint.get("step", None)
    loss = checkpoint.get("loss", None)

    if step is not None and loss is not None:
        print(f"Loaded model checkpoint from step {step}, loss: {loss:.4f}")
    elif step is not None:
        print(f"Loaded model checkpoint from step {step}")
    else:
        print("Loaded model checkpoint")

    return step, loss


# =============================================================================
# DATA LOADING
# =============================================================================
def load_flickr8k_split(root_dir, train_split=0.8):
    captions = []
    caption_list = []
    images = []
    curr = None
    icurr = None
    flag = True

    captions_file = f"{root_dir}/data/datasets/Flickr8k/captions.txt"

    print("Loading Flickr8K dataset...")
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            if flag:
                flag = False
                continue

            line = line.strip()
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue

            image, caption = parts[0], parts[1]

            if icurr is None:
                images.append(f"{root_dir}/data/datasets/Flickr8k/Images/{image}")
                icurr = image
            elif icurr != image:
                images.append(f"{root_dir}/data/datasets/Flickr8k/Images/{image}")
                icurr = image

            if len(caption_list) == 0:
                curr = image
                caption_list.append(caption)
            elif curr == image:
                caption_list.append(caption)
            else:
                captions.append(caption_list)
                caption_list = [caption]
                curr = image

        if len(caption_list) > 0:
            captions.append(caption_list)

    print(f"Loaded {len(images)} unique images with {sum(len(c) for c in captions)} total captions")

    num_images = len(images)
    train_size = int(train_split * num_images)

    train_images = images[:train_size]
    train_captions = captions[:train_size]
    val_images = images[train_size:]
    val_captions = captions[train_size:]

    print(f"Train set: {len(train_images)} unique images")
    print(f"Val set: {len(val_images)} unique images")

    return train_images, train_captions, val_images, val_captions


def flatten_caption_pairs(image_paths, captions_nested):
    rows = []
    for img_path, caps in zip(image_paths, captions_nested):
        for cap in caps:
            rows.append({
                "caption": cap,
                "gt_image_path": img_path,
            })
    return rows


def canonical_caption_pairs(image_paths, captions_nested):
    rows = []
    for img_path, caps in zip(image_paths, captions_nested):
        if len(caps) == 0:
            continue
        rows.append({
            "caption": caps[0],
            "gt_image_path": img_path,
        })
    return rows


# =============================================================================
# MODEL SETUP
# =============================================================================
def setup_model_for_test(
    vision_model_name,
    text_model_name,
    device,
):
    vision_model = AutoModel.from_pretrained(vision_model_name)
    text_model = AutoModel.from_pretrained(text_model_name)
    model = OTLIP(vision_model, text_model, device=device).to(device)
    return model


# =============================================================================
# IMAGE POOL DATASET
# =============================================================================
class FlickrImagePoolDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, path


def image_pool_collate(batch):
    images, paths = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(paths)


# =============================================================================
# ENCODING
# =============================================================================
@torch.no_grad()
def encode_caption(model, tokenizer, caption, device, max_length=77):
    text_batch = tokenizer(
        [caption],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    text_batch = {k: v.to(device) for k, v in text_batch.items()}
    z_text = model.encode_texts(text_batch)
    z_text = F.normalize(z_text, dim=-1)
    return z_text


@torch.no_grad()
def encode_image_pool(model, image_paths, processor, device, batch_size=64, num_workers=4):
    dataset = FlickrImagePoolDataset(image_paths, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=image_pool_collate,
    )

    embs = []
    out_paths = []

    for images, paths in loader:
        images = images.to(device)
        z_imgs = model.encode_images(images)
        z_imgs = F.normalize(z_imgs, dim=-1)
        embs.append(z_imgs.cpu())
        out_paths.extend(paths)

    embs = torch.cat(embs, dim=0)
    return embs, out_paths


# =============================================================================
# RETRIEVAL / RANKING
# =============================================================================
def compute_similarity_scores(z_text, image_embs):
    """
    z_text: [1, d]
    image_embs: [N, d] on CPU
    returns: sims [N], costs [N]
    """
    sims = (z_text.cpu() @ image_embs.T).squeeze(0)
    costs = 1.0 - sims
    return sims, costs


def compute_gt_rank_and_metrics(sims, image_paths, gt_image_path):
    """
    Rank is 1-based.
    """
    if gt_image_path not in image_paths:
        return {
            "gt_rank": None,
            "gt_sim": None,
            "gt_cost": None,
            "gt_in_top1": False,
            "gt_in_top5": False,
            "gt_in_top10": False,
        }

    sorted_indices = torch.argsort(sims, descending=True)
    gt_idx = image_paths.index(gt_image_path)

    positions = (sorted_indices == gt_idx).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return {
            "gt_rank": None,
            "gt_sim": None,
            "gt_cost": None,
            "gt_in_top1": False,
            "gt_in_top5": False,
            "gt_in_top10": False,
        }

    gt_rank = int(positions.item()) + 1
    gt_sim = float(sims[gt_idx].item())
    gt_cost = float((1.0 - sims[gt_idx]).item())

    return {
        "gt_rank": gt_rank,
        "gt_sim": gt_sim,
        "gt_cost": gt_cost,
        "gt_in_top1": gt_rank <= 1,
        "gt_in_top5": gt_rank <= 5,
        "gt_in_top10": gt_rank <= 10,
    }


def retrieve_nearest_negative_images(sims, image_paths, gt_image_path, top_k=5):
    """
    Return top-k nearest negatives, excluding the GT image.
    """
    rows = []
    costs = 1.0 - sims

    for idx, (path, sim, cost) in enumerate(zip(image_paths, sims.tolist(), costs.tolist())):
        if path == gt_image_path:
            continue
        rows.append((idx, path, sim, cost))

    rows.sort(key=lambda x: x[2], reverse=True)
    rows = rows[:top_k]

    indices = [r[0] for r in rows]
    paths = [r[1] for r in rows]
    sims_out = torch.tensor([r[2] for r in rows], dtype=torch.float32)
    costs_out = torch.tensor([r[3] for r in rows], dtype=torch.float32)

    return indices, paths, sims_out, costs_out


# =============================================================================
# OT BARYCENTRIC NEGATIVE
# =============================================================================
def compute_ot_barycentric_negative(z_text, negative_image_embs, epsilon=0.07):
    sims = (z_text @ negative_image_embs.T).squeeze(0)
    costs = 1.0 - sims
    weights = torch.softmax(-costs / epsilon, dim=0)
    z_tilde = torch.sum(weights.unsqueeze(1) * negative_image_embs, dim=0)
    z_tilde = F.normalize(z_tilde, dim=-1)
    return z_tilde, weights, costs, sims


# =============================================================================
# FILE / FIGURE HELPERS
# =============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_copy(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def save_diagnostic_panel(
    sample_id,
    caption,
    gt_image_path,
    gt_rank,
    gt_sim,
    negative_paths,
    negative_sims,
    negative_costs,
    negative_weights,
    panel_path,
):
    ncols = 1 + len(negative_paths)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5))

    if ncols == 1:
        axes = [axes]

    gt_img = Image.open(gt_image_path).convert("RGB")
    axes[0].imshow(gt_img)
    axes[0].set_title(f"GT\nrank={gt_rank}\nsim={gt_sim:.4f}")
    axes[0].axis("off")

    for i, neg_path in enumerate(negative_paths, start=1):
        img = Image.open(neg_path).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(
            f"Neg {i}\n"
            f"sim={negative_sims[i-1]:.4f}\n"
            f"cost={negative_costs[i-1]:.4f}\n"
            f"w={negative_weights[i-1]:.4f}"
        )
        axes[i].axis("off")

    fig.suptitle(f"[{sample_id}] {caption}", fontsize=11)
    plt.tight_layout()
    plt.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SUMMARY METRICS
# =============================================================================
def compute_summary(metadata_rows):
    if len(metadata_rows) == 0:
        return {}

    valid_rows = [r for r in metadata_rows if r["gt_rank"] is not None]
    if len(valid_rows) == 0:
        return {
            "num_samples": len(metadata_rows),
            "num_valid_samples": 0,
        }

    def mean_of(key):
        vals = [float(r[key]) for r in valid_rows if r[key] is not None and not math.isnan(float(r[key]))]
        return sum(vals) / len(vals) if len(vals) > 0 else None

    summary = {
        "num_samples": len(metadata_rows),
        "num_valid_samples": len(valid_rows),
        "recall_at_1": sum(int(r["gt_in_top1"]) for r in valid_rows) / len(valid_rows),
        "recall_at_5": sum(int(r["gt_in_top5"]) for r in valid_rows) / len(valid_rows),
        "recall_at_10": sum(int(r["gt_in_top10"]) for r in valid_rows) / len(valid_rows),
        "mean_gt_rank": mean_of("gt_rank"),
        "median_gt_rank": sorted([r["gt_rank"] for r in valid_rows])[len(valid_rows) // 2],
        "mean_text_gt_sim": mean_of("text_gt_sim"),
        "mean_text_synth_sim": mean_of("text_synth_sim"),
        "mean_gt_synth_sim": mean_of("gt_synth_sim"),
        "mean_neg1_sim": mean_of("neg_1_sim"),
        "mean_neg1_minus_gt_sim": mean_of("neg_1_minus_gt_sim"),
        "fraction_neg1_beats_gt": sum(int(r["neg_1_beats_gt"]) for r in valid_rows) / len(valid_rows),
    }

    return summary


# =============================================================================
# MAIN DIAGNOSTIC
# =============================================================================
def run_many_caption_diagnostic(
    checkpoint_path="checkpoints/baseline/best_model.pt",
    vision_model_name="microsoft/resnet-50",
    text_model_name="distilbert-base-uncased",
    train_split=0.8,
    eval_split="val",
    retrieval_pool_split="val",
    top_k=5,
    batch_size=64,
    num_workers=4,
    epsilon=0.07,
    max_length=77,
    max_captions=None,
    caption_stride=1,
    canonical_only=False,
    output_dir="outputs/ot_diagnostic",
    cache_image_embs=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = ROOT_DIR / output_dir
    panels_dir = output_dir / "panels"
    gt_dir = output_dir / "gt"
    neg_dir = output_dir / "negatives"
    ensure_dir(output_dir)
    ensure_dir(panels_dir)
    ensure_dir(gt_dir)
    ensure_dir(neg_dir)

    # -------------------------------------------------------------------------
    # Load split
    # -------------------------------------------------------------------------
    train_images, train_captions, val_images, val_captions = load_flickr8k_split(
        ROOT_DIR, train_split=train_split
    )

    if eval_split == "train":
        eval_images, eval_captions = train_images, train_captions
    elif eval_split == "val":
        eval_images, eval_captions = val_images, val_captions
    else:
        raise ValueError("eval_split must be 'train' or 'val'")

    if retrieval_pool_split == "train":
        pool_images = train_images
    elif retrieval_pool_split == "val":
        pool_images = val_images
    else:
        raise ValueError("retrieval_pool_split must be 'train' or 'val'")

    if canonical_only:
        eval_rows = canonical_caption_pairs(eval_images, eval_captions)
    else:
        eval_rows = flatten_caption_pairs(eval_images, eval_captions)

    if caption_stride > 1:
        eval_rows = eval_rows[::caption_stride]

    if max_captions is not None:
        eval_rows = eval_rows[:max_captions]

    print(f"Evaluating {len(eval_rows)} captions")
    print(f"Retrieval pool size: {len(pool_images)} images")

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    processor = AutoImageProcessor.from_pretrained(vision_model_name, use_fast=False)

    model = setup_model_for_test(
        vision_model_name=vision_model_name,
        text_model_name=text_model_name,
        device=device,
    )

    ckpt_full_path = ROOT_DIR / checkpoint_path
    load_model_checkpoint(model, ckpt_full_path, device)
    model.eval()

    # -------------------------------------------------------------------------
    # Image embedding cache
    # -------------------------------------------------------------------------
    cache_path = output_dir / f"{retrieval_pool_split}_image_embs.pt"

    if cache_image_embs and cache_path.exists():
        print(f"Loading cached image embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        image_embs = cache["image_embs"]
        image_paths = cache["image_paths"]
    else:
        print("Encoding retrieval pool images...")
        image_embs, image_paths = encode_image_pool(
            model=model,
            image_paths=pool_images,
            processor=processor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if cache_image_embs:
            torch.save({"image_embs": image_embs, "image_paths": image_paths}, cache_path)
            print(f"Saved cached image embeddings to {cache_path}")

    image_path_to_index = {p: i for i, p in enumerate(image_paths)}

    # -------------------------------------------------------------------------
    # Run diagnostic
    # -------------------------------------------------------------------------
    metadata_rows = []

    for idx, row in enumerate(eval_rows):
        caption = row["caption"]
        gt_image_path = row["gt_image_path"]

        if gt_image_path not in image_path_to_index:
            print(f"[skip {idx}] GT image not found in retrieval pool: {gt_image_path}")
            continue

        z_text = encode_caption(
            model=model,
            tokenizer=tokenizer,
            caption=caption,
            device=device,
            max_length=max_length,
        )

        sims, costs = compute_similarity_scores(z_text, image_embs)

        gt_metrics = compute_gt_rank_and_metrics(
            sims=sims,
            image_paths=image_paths,
            gt_image_path=gt_image_path,
        )

        neg_indices, neg_paths, neg_sims, neg_costs = retrieve_nearest_negative_images(
            sims=sims,
            image_paths=image_paths,
            gt_image_path=gt_image_path,
            top_k=top_k,
        )

        negative_image_embs = image_embs[neg_indices].to(device)
        z_tilde, weights, ot_costs, ot_sims = compute_ot_barycentric_negative(
            z_text=z_text,
            negative_image_embs=negative_image_embs,
            epsilon=epsilon,
        )

        gt_idx = image_path_to_index[gt_image_path]
        z_gt = image_embs[gt_idx].to(device)

        text_gt_sim = torch.dot(z_text.squeeze(0), z_gt).item()
        text_synth_sim = torch.dot(z_text.squeeze(0), z_tilde).item()
        gt_synth_sim = torch.dot(z_gt, z_tilde).item()

        sample_id = f"sample_{idx:05d}"

        # Save image copies
        gt_copy_path = gt_dir / f"{sample_id}_gt.jpg"
        safe_copy(gt_image_path, gt_copy_path)

        neg_copy_paths = []
        for j, neg_path in enumerate(neg_paths, start=1):
            dst = neg_dir / f"{sample_id}_neg_{j:02d}.jpg"
            safe_copy(neg_path, dst)
            neg_copy_paths.append(str(dst))

        # Save panel
        panel_path = panels_dir / f"{sample_id}.png"
        save_diagnostic_panel(
            sample_id=sample_id,
            caption=caption,
            gt_image_path=gt_image_path,
            gt_rank=gt_metrics["gt_rank"],
            gt_sim=gt_metrics["gt_sim"],
            negative_paths=neg_paths,
            negative_sims=neg_sims.tolist(),
            negative_costs=neg_costs.tolist(),
            negative_weights=weights.tolist(),
            panel_path=panel_path,
        )

        row_out = {
            "sample_id": sample_id,
            "caption": caption,
            "gt_image_path": gt_image_path,
            "gt_copy_path": str(gt_copy_path),
            "panel_path": str(panel_path),
            "gt_rank": gt_metrics["gt_rank"],
            "gt_in_top1": gt_metrics["gt_in_top1"],
            "gt_in_top5": gt_metrics["gt_in_top5"],
            "gt_in_top10": gt_metrics["gt_in_top10"],
            "text_gt_sim": text_gt_sim,
            "text_gt_cost": 1.0 - text_gt_sim,
            "text_synth_sim": text_synth_sim,
            "gt_synth_sim": gt_synth_sim,
            "top_k": top_k,
            "epsilon": epsilon,
        }

        for j in range(top_k):
            row_out[f"neg_{j+1}_path"] = neg_paths[j] if j < len(neg_paths) else ""
            row_out[f"neg_{j+1}_copy_path"] = neg_copy_paths[j] if j < len(neg_copy_paths) else ""
            row_out[f"neg_{j+1}_sim"] = float(neg_sims[j]) if j < len(neg_sims) else math.nan
            row_out[f"neg_{j+1}_cost"] = float(neg_costs[j]) if j < len(neg_costs) else math.nan
            row_out[f"neg_{j+1}_weight"] = float(weights[j]) if j < len(weights) else math.nan

        neg1_sim = float(neg_sims[0]) if len(neg_sims) > 0 else math.nan
        row_out["neg_1_minus_gt_sim"] = neg1_sim - text_gt_sim if not math.isnan(neg1_sim) else math.nan
        row_out["neg_1_beats_gt"] = bool(neg1_sim > text_gt_sim) if not math.isnan(neg1_sim) else False

        metadata_rows.append(row_out)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_rows):
            print(f"Processed {idx + 1}/{len(eval_rows)} captions")

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------
    csv_path = output_dir / "metadata.csv"
    json_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.json"

    if metadata_rows:
        fieldnames = list(metadata_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_rows, f, indent=2)

        summary = compute_summary(metadata_rows)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved CSV metadata to {csv_path}")
        print(f"Saved JSON metadata to {json_path}")
        print(f"Saved summary to {summary_path}")

        print("\nSummary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    else:
        print("No metadata rows were generated.")
        summary = {}

    return metadata_rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OT diagnostics using YAML configuration.")
    parser.add_argument("--config", type=str, default="configs/diagnostic.yaml", help="Path to diagnostic YAML config")
    args = parser.parse_args()

    loaded = load_diagnostic_config(args.config)
    cfg = loaded["diagnostic_config"]
    print(f"Loaded diagnostic config from: {loaded['config_path']}")

    run_many_caption_diagnostic(
        checkpoint_path=cfg["checkpoint_path"],
        vision_model_name=cfg["vision_model_name"],
        text_model_name=cfg["text_model_name"],
        train_split=cfg["train_split"],
        eval_split=cfg["eval_split"],
        retrieval_pool_split=cfg["retrieval_pool_split"],
        top_k=cfg["top_k"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        epsilon=cfg["epsilon"],
        max_length=cfg["max_length"],
        max_captions=cfg["max_captions"],
        caption_stride=cfg["caption_stride"],
        canonical_only=cfg["canonical_only"],
        output_dir=cfg["output_dir"],
        cache_image_embs=cfg["cache_image_embs"],
    )
