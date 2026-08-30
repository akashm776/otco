"""Run the frozen CLIP/CUB geometry diagnostic configured by YAML.

This entry point performs no optimization and creates no checkpoints.  It
encodes a deterministic, species-balanced holdout from the CUB training split,
builds a fresh OT plan over the complete diagnostic pool, and writes both
aggregate distributions and per-query evidence.
"""

import argparse
import csv
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from data.hf_cub200_dataset import (
    HFCUB200CanonicalCaptionDataset,
    get_cub200_class_labels,
    load_hf_cub200_splits,
)
from model.clip_backend import CLIPEncoderBackend
from src.clip_geometry_metrics import compute_geometry_diagnostics, summarize_geometry


ROOT_DIR = Path(__file__).resolve().parents[1]


class CUBCLIPDiagnosticDataset(Dataset):
    """Canonical image-caption pairs plus stable CUB metadata."""

    def __init__(self, grouped_split, species_ids, selected_indices):
        self.base = HFCUB200CanonicalCaptionDataset(
            grouped_split, image_transform=lambda image: image
        )
        self.grouped_split = grouped_split
        self.species_ids = species_ids
        self.selected_indices = list(selected_indices)

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        source_index = self.selected_indices[index]
        item = self.base[source_index]
        group = self.grouped_split.groups[source_index]
        item.update(
            {
                "diagnostic_index": index,
                "source_index": source_index,
                "image_id": source_index,
                "image_key": group.image_key,
                "species_id": self.species_ids[source_index],
            }
        )
        return item


def keep_records_as_list(records):
    """Let the CLIP processor batch PIL images in the parent process."""
    return records


def stratified_holdout_indices(labels, *, fraction, seed, max_samples=None):
    """Select a deterministic, approximately class-balanced diagnostic holdout."""
    if not 0 < fraction <= 1:
        raise ValueError("holdout_fraction must be in (0, 1]")
    by_class = {}
    for index, label in enumerate(labels):
        by_class.setdefault(str(label), []).append(index)

    rng = random.Random(seed)
    held_out_by_class = {}
    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        count = max(1, round(len(indices) * fraction))
        held_out_by_class[label] = indices[:count]

    # Round-robin limiting prevents large classes from dominating the cap.
    selected = []
    offset = 0
    while True:
        added = False
        for label in sorted(held_out_by_class):
            bucket = held_out_by_class[label]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if max_samples is not None and len(selected) >= max_samples:
                    return selected
        if not added:
            return selected
        offset += 1


def resolve_device(runtime_config):
    requested = runtime_config.get("device", "cuda")
    require_cuda = runtime_config.get("require_cuda", True)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if require_cuda and (requested != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("This configured diagnostic requires a CUDA GPU")

    device = torch.device(requested)
    device_info = {"device": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        total_vram_mib = properties.total_memory / (1024**2)
        required_vram_mib = runtime_config.get("min_vram_mib", 0)
        if total_vram_mib < required_vram_mib:
            raise RuntimeError(
                f"GPU has {total_vram_mib:.0f} MiB VRAM; "
                f"configuration requires {required_vram_mib} MiB"
            )
        device_info.update(
            {"gpu_name": properties.name, "total_vram_mib": total_vram_mib}
        )
    return device, device_info


@torch.inference_mode()
def encode_dataset(model, processor, loader, device):
    image_features = []
    text_features = []
    metadata = []

    for records in loader:
        processed = processor(
            images=[record["image"] for record in records],
            text=[record["caption"] for record in records],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        pixel_values = processed["pixel_values"].to(device)
        text_batch = {
            key: value.to(device)
            for key, value in processed.items()
            if key in ("input_ids", "attention_mask")
        }
        image_features.append(model.encode_images(pixel_values).cpu())
        text_features.append(model.encode_texts(text_batch).cpu())
        for record in records:
            metadata.append(
                {
                    key: record[key]
                    for key in (
                        "diagnostic_index",
                        "source_index",
                        "image_id",
                        "image_key",
                        "image_path",
                        "species_id",
                        "caption",
                    )
                }
            )

    return torch.cat(image_features), torch.cat(text_features), metadata


def _tensor_value(tensor, index):
    return float(tensor[index].detach().cpu().item())


def build_per_query_rows(diagnostics, metadata, gate_sim):
    rows = []
    selected = diagnostics["selected_indices"].detach().cpu().tolist()
    hardest = diagnostics["hardest_indices"].detach().cpu().tolist()
    for index, anchor in enumerate(metadata):
        selected_record = metadata[selected[index]]
        hardest_record = metadata[hardest[index]]
        synthetic_logit = _tensor_value(diagnostics["synthetic_logits"], index)
        rows.append(
            {
                "diagnostic_index": index,
                "source_index": anchor["source_index"],
                "image_key": anchor["image_key"],
                "species_id": anchor["species_id"],
                "caption": anchor["caption"],
                "selected_index": selected[index],
                "selected_image_key": selected_record["image_key"],
                "selected_species_id": selected_record["species_id"],
                "selected_same_species": bool(
                    diagnostics["same_species_selected"][index].item()
                ),
                "hardest_index": hardest[index],
                "hardest_image_key": hardest_record["image_key"],
                "hardest_species_id": hardest_record["species_id"],
                "hardest_same_species": bool(
                    diagnostics["same_species_hardest"][index].item()
                ),
                "positive_similarity": _tensor_value(
                    diagnostics["positive_similarity"], index
                ),
                "selected_similarity": _tensor_value(
                    diagnostics["selected_similarity"], index
                ),
                "positive_selected_gap": _tensor_value(
                    diagnostics["positive_selected_gap"], index
                ),
                "selected_rank": _tensor_value(diagnostics["selected_rank"], index),
                "selected_hardness_percentile": _tensor_value(
                    diagnostics["selected_hardness_percentile"], index
                ),
                "coupling_entropy": _tensor_value(
                    diagnostics["coupling_entropy"], index
                ),
                "normalized_coupling_entropy": _tensor_value(
                    diagnostics["normalized_coupling_entropy"], index
                ),
                "coupling_peak_mass": _tensor_value(
                    diagnostics["coupling_peak_mass"], index
                ),
                "hardest_negative_similarity": _tensor_value(
                    diagnostics["hardest_negative_similarity"], index
                ),
                "hardest_wrong_class_similarity": _tensor_value(
                    diagnostics["hardest_wrong_class_similarity"], index
                ),
                "wrong_class_margin": _tensor_value(
                    diagnostics["wrong_class_margin"], index
                ),
                "synthetic_similarity": _tensor_value(
                    diagnostics["synthetic_similarity"], index
                ),
                "synthetic_logit": synthetic_logit,
                "passes_historical_gate_sim": synthetic_logit > gate_sim,
                "historical_gap_entropy_bucket": diagnostics["gate_overlay"][index],
            }
        )
    return rows


def choose_representative_examples(rows, count):
    selectors = {
        "lowest_gap": ("positive_selected_gap", False),
        "highest_gap": ("positive_selected_gap", True),
        "sharpest_plan": ("coupling_entropy", False),
        "most_diffuse_plan": ("coupling_entropy", True),
        "largest_wrong_class_margin": ("wrong_class_margin", True),
        "smallest_wrong_class_margin": ("wrong_class_margin", False),
    }
    examples = {}
    for name, (field, reverse) in selectors.items():
        ordered = sorted(rows, key=lambda row: row[field], reverse=reverse)
        examples[name] = ordered[:count]
    return examples


def write_outputs(output_dir, report, rows, selected_indices):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "clip_geometry_report.json"
    rows_path = output_dir / "clip_geometry_per_query.csv"
    selection_path = output_dir / "diagnostic_holdout_indices.json"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(selected_indices, handle, indent=2)
    return report_path, rows_path, selection_path


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("experiment", {}).get("mode") != "frozen_geometry_diagnostic":
        raise ValueError("This runner only accepts mode=frozen_geometry_diagnostic")
    if config.get("dataset", {}).get("backend") != "hf_cub200":
        raise ValueError("This experiment requires dataset.backend=hf_cub200")
    if config.get("model", {}).get("backend") != "clip":
        raise ValueError("This experiment requires model.backend=clip")
    if config.get("sampling", {}).get("caption_view") != "canonical_first_caption":
        raise ValueError("This diagnostic currently requires one canonical caption per image")
    if config.get("ot", {}).get("cost_space") != "clip_scaled_logits":
        raise ValueError("OT must use clip_scaled_logits to match the existing OT-Mix path")
    if config.get("ot", {}).get("exclude_same_image_positives") is not True:
        raise ValueError("Same-image positives must be excluded from OT candidates")
    return config


def run(config):
    # Keep the heavyweight Transformers processing stack out of pure metric and
    # sampling imports (and therefore out of their unit tests).
    from transformers import AutoProcessor

    seed = config["sampling"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    device, device_info = resolve_device(config["runtime"])

    dataset_config = config["dataset"]
    train_grouped, val_grouped, split_info = load_hf_cub200_splits(
        dataset_name=dataset_config["dataset_name"],
        train_hf_split=dataset_config.get("train_split", "train"),
        val_hf_split=dataset_config.get("val_split", "test"),
    )
    source_name = config["sampling"].get("source_split", "train")
    if source_name == "train":
        grouped_split = train_grouped
    elif source_name in ("validation", "val", "test"):
        grouped_split = val_grouped
    else:
        raise ValueError(f"Unsupported diagnostic source split: {source_name}")
    species_ids = get_cub200_class_labels(grouped_split)
    selected_indices = stratified_holdout_indices(
        species_ids,
        fraction=config["sampling"].get("holdout_fraction", 0.2),
        seed=seed,
        max_samples=config["sampling"].get("max_samples"),
    )
    dataset = CUBCLIPDiagnosticDataset(
        grouped_split, species_ids, selected_indices
    )
    loader = DataLoader(
        dataset,
        batch_size=config["runtime"].get("batch_size", 64),
        shuffle=False,
        num_workers=config["runtime"].get("num_workers", 2),
        collate_fn=keep_records_as_list,
        pin_memory=device.type == "cuda",
    )

    checkpoint = config["model"]["checkpoint"]
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = CLIPEncoderBackend.from_pretrained(checkpoint).to(device).freeze()
    image_features, text_features, metadata = encode_dataset(
        model, processor, loader, device
    )
    logit_scale = float(model.get_logit_scale().detach().cpu().item())

    ot_config = config["ot"]
    historical = config.get("historical_threshold_overlay", {})
    diagnostics = compute_geometry_diagnostics(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        image_ids=[record["image_id"] for record in metadata],
        species_ids=[record["species_id"] for record in metadata],
        top_k=ot_config.get("top_k", 32),
        ot_eps=ot_config.get("ot_eps", 0.7),
        sinkhorn_iters=ot_config.get("sinkhorn_iters", 30),
        historical_thresholds=historical,
    )
    gate_sim = historical.get("gate_sim", -4.0)
    rows = build_per_query_rows(diagnostics, metadata, gate_sim)
    summary = summarize_geometry(diagnostics)
    passing_gate_sim = sum(row["passes_historical_gate_sim"] for row in rows)
    summary["historical_gate_sim_overlay"] = {
        "threshold": gate_sim,
        "count": passing_gate_sim,
        "fraction": passing_gate_sim / len(rows),
    }

    image_norms = image_features.norm(dim=1)
    text_norms = text_features.norm(dim=1)
    report = {
        "experiment": config["experiment"],
        "model": {
            "checkpoint": checkpoint,
            "frozen": True,
            "logit_scale": logit_scale,
        },
        "runtime": device_info,
        "dataset": {
            **split_info,
            "diagnostic_source_split": source_name,
            "holdout_fraction": config["sampling"].get("holdout_fraction", 0.2),
            "num_pairs": len(dataset),
            "num_species": len({record["species_id"] for record in metadata}),
            "seed": seed,
        },
        "embedding_checks": {
            "image_norm_mean": float(image_norms.mean().item()),
            "image_norm_max_abs_error": float((image_norms - 1).abs().max().item()),
            "text_norm_mean": float(text_norms.mean().item()),
            "text_norm_max_abs_error": float((text_norms - 1).abs().max().item()),
        },
        "ot": ot_config,
        "historical_threshold_overlay": {
            **historical,
            "interpretation": "observation only; no loss was gated or optimized",
        },
        "summary": summary,
        "representative_examples": choose_representative_examples(
            rows, config["output"].get("representative_examples_per_group", 5)
        ),
    }
    output_dir = ROOT_DIR / config["output"]["directory"]
    paths = write_outputs(output_dir, report, rows, selected_indices)

    print("Frozen CLIP geometry diagnostic complete")
    print(f"  model: {checkpoint}")
    print(f"  diagnostic pairs/species: {len(dataset)}/{report['dataset']['num_species']}")
    print(f"  logit scale: {logit_scale:.4f}")
    print(
        "  canonical Avg R@1: "
        f"{100 * (summary['retrieval']['text_to_image']['r_at_1'] + summary['retrieval']['image_to_text']['r_at_1']) / 2:.2f}%"
    )
    print(
        "  median entropy / normalized entropy: "
        f"{summary['distributions']['coupling_entropy']['median']:.4f} / "
        f"{summary['distributions']['normalized_coupling_entropy']['median']:.4f}"
    )
    print(
        "  median positive-selected gap / selected rank: "
        f"{summary['distributions']['positive_selected_gap']['median']:.4f} / "
        f"{summary['distributions']['selected_rank']['median']:.1f}"
    )
    for path in paths:
        print(f"  wrote: {path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hf_cub200_clip_geometry.yaml",
        help="Frozen geometry diagnostic YAML",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(load_config(arguments.config))
