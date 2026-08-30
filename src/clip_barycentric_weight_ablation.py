"""Run the frozen CLIP OT-vs-uniform barycentric-weight ablation on CUB-200."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
import yaml

from data.hf_cub200_dataset import get_cub200_class_labels, load_hf_cub200_splits
from model.clip_backend import CLIPEncoderBackend
from src.clip_barycentric_weight_metrics import (
    compute_barycentric_weight_ablation,
    summarize_barycentric_weight_ablation,
)
from src.clip_geometry_diagnostic import (
    CUBCLIPDiagnosticDataset,
    encode_dataset,
    keep_records_as_list,
    resolve_device,
)
from src.clip_geometry_v2_metrics import (
    compute_transport_variant,
    summarize_transport_variant,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_committed_holdout(path):
    """Load the fixed V2 selection instead of regenerating a new holdout."""
    resolved = _resolve_repo_path(path)
    contents = resolved.read_bytes()
    indices = json.loads(contents)
    if not isinstance(indices, list) or not indices:
        raise ValueError("Diagnostic holdout must be a non-empty JSON list")
    if len(indices) != len(set(indices)) or not all(
        isinstance(index, int) and index >= 0 for index in indices
    ):
        raise ValueError("Diagnostic holdout indices must be unique non-negative ints")
    return indices, hashlib.sha256(contents).hexdigest(), resolved


def _scalar(tensor, index):
    return float(tensor[index].detach().cpu().item())


def build_per_query_rows(diagnostics, metadata):
    hardest_indices = diagnostics["hardest_real_indices"].detach().cpu().tolist()
    rows = []
    for index, anchor in enumerate(metadata):
        rows.append(
            {
                "query_index": index,
                "source_index": anchor["source_index"],
                "image_key": anchor["image_key"],
                "species_id": anchor["species_id"],
                "positive_similarity": _scalar(
                    diagnostics["positive_similarity"], index
                ),
                "hardest_real_index": hardest_indices[index],
                "hardest_real_similarity": _scalar(
                    diagnostics["hardest_real_similarity"], index
                ),
                "ot_synthetic_similarity": _scalar(
                    diagnostics["ot_synthetic_similarity"], index
                ),
                "uniform_synthetic_similarity": _scalar(
                    diagnostics["uniform_synthetic_similarity"], index
                ),
                "ot_vs_hardest_delta": _scalar(
                    diagnostics["ot_vs_hardest_delta"], index
                ),
                "uniform_vs_hardest_delta": _scalar(
                    diagnostics["uniform_vs_hardest_delta"], index
                ),
                "ot_vs_positive_delta": _scalar(
                    diagnostics["ot_vs_positive_delta"], index
                ),
                "uniform_vs_positive_delta": _scalar(
                    diagnostics["uniform_vs_positive_delta"], index
                ),
                "ot_minus_uniform_synthetic_similarity": _scalar(
                    diagnostics["ot_minus_uniform_synthetic_similarity"], index
                ),
                "ot_harder_than_hardest_real": bool(
                    diagnostics["ot_harder_than_hardest_real"][index].item()
                ),
                "uniform_harder_than_hardest_real": bool(
                    diagnostics["uniform_harder_than_hardest_real"][index].item()
                ),
                "support_size": int(diagnostics["support_size"][index].item()),
                "ot_weight_entropy": _scalar(
                    diagnostics["ot_weight_entropy"], index
                ),
                "ot_weight_normalized_entropy": _scalar(
                    diagnostics["ot_weight_normalized_entropy"], index
                ),
                "ot_peak_weight": _scalar(diagnostics["ot_peak_weight"], index),
                "uniform_weight_entropy": _scalar(
                    diagnostics["uniform_weight_entropy"], index
                ),
                "uniform_peak_weight": _scalar(
                    diagnostics["uniform_peak_weight"], index
                ),
                "l1_distance_ot_vs_uniform_weights": _scalar(
                    diagnostics["l1_distance_ot_vs_uniform_weights"], index
                ),
                "cosine_between_ot_and_uniform_synthetic": _scalar(
                    diagnostics["cosine_between_ot_and_uniform_synthetic"], index
                ),
            }
        )
    return rows


def write_outputs(output_dir, report, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "clip_barycentric_weight_ablation_report.json"
    rows_path = output_dir / "clip_barycentric_weight_ablation_per_query.csv"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"report": report_path, "per_query": rows_path}


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("experiment", {}).get("mode") != "frozen_barycentric_weight_ablation":
        raise ValueError(
            "Runner requires mode=frozen_barycentric_weight_ablation"
        )
    if config.get("dataset", {}).get("backend") != "hf_cub200":
        raise ValueError("Runner requires dataset.backend=hf_cub200")
    if config.get("model", {}).get("backend") != "clip":
        raise ValueError("Runner requires model.backend=clip")
    if config.get("sampling", {}).get("caption_view") != "canonical_first_caption":
        raise ValueError("Runner requires canonical_first_caption sampling")
    ot = config.get("ot", {})
    expected_ot = {
        "cost_space": "raw_cosine",
        "top_k": 32,
        "ot_eps": 0.049,
        "sinkhorn_iters": 30,
        "solver": "historical_sparse_ot",
        "exclude_same_image_positives": True,
    }
    mismatches = {
        key: (ot.get(key), expected)
        for key, expected in expected_ot.items()
        if ot.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Ablation OT configuration mismatch: {mismatches}")
    return config


@torch.inference_mode()
def run(config):
    from transformers import AutoProcessor

    seed = config["sampling"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    device, device_info = resolve_device(config["runtime"])

    selected_indices, holdout_sha256, holdout_path = load_committed_holdout(
        config["sampling"]["holdout_indices_path"]
    )
    expected_count = config["sampling"]["holdout_size"]
    if len(selected_indices) != expected_count:
        raise ValueError(
            f"Committed holdout has {len(selected_indices)} entries; "
            f"expected {expected_count}"
        )

    dataset_config = config["dataset"]
    train_grouped, val_grouped, split_info = load_hf_cub200_splits(
        dataset_name=dataset_config["dataset_name"],
        train_hf_split=dataset_config.get("train_split", "train"),
        val_hf_split=dataset_config.get("val_split", "test"),
    )
    source_name = config["sampling"].get("source_split", "train")
    if source_name not in {"train", "val"}:
        raise ValueError("sampling.source_split must be 'train' or 'val'")
    grouped_split = train_grouped if source_name == "train" else val_grouped
    if max(selected_indices) >= len(grouped_split):
        raise ValueError("Committed holdout contains an out-of-range source index")
    species_ids = get_cub200_class_labels(grouped_split)
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
    image_features_cpu, text_features_cpu, metadata = encode_dataset(
        model, processor, loader, device
    )
    image_features = image_features_cpu.to(device)
    text_features = text_features_cpu.to(device)
    logit_scale = float(model.get_logit_scale().detach().cpu().item())
    image_ids = [record["image_id"] for record in metadata]
    selected_species_ids = [record["species_id"] for record in metadata]

    ot = config["ot"]
    transport = compute_transport_variant(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        image_ids=image_ids,
        species_ids=selected_species_ids,
        cost_space=ot["cost_space"],
        top_k=ot["top_k"],
        ot_eps=ot["ot_eps"],
        sinkhorn_iters=ot["sinkhorn_iters"],
        solver=ot["solver"],
    )
    comparison = config["comparison"]
    diagnostics = compute_barycentric_weight_ablation(
        image_features=image_features,
        text_features=text_features,
        plan=transport["plan"],
        support_mask=transport["support_mask"],
        positive_mask=transport["positive_mask"],
        tie_tolerance=comparison["tie_tolerance"],
    )
    summaries = summarize_barycentric_weight_ablation(
        diagnostics,
        bootstrap_seed=comparison["bootstrap_seed"],
        bootstrap_samples=comparison["bootstrap_samples"],
    )
    rows = build_per_query_rows(diagnostics, metadata)

    identical_support = bool(
        torch.equal(diagnostics["support_mask"], diagnostics["ot_support_mask"])
        and torch.equal(
            diagnostics["support_mask"], diagnostics["uniform_support_mask"]
        )
    )
    report = {
        "experiment": config["experiment"],
        "no_training_or_gradients": True,
        "model": {
            "checkpoint": checkpoint,
            "frozen": True,
            "logit_scale_observational_only": logit_scale,
        },
        "runtime": device_info,
        "dataset": {
            **split_info,
            "diagnostic_source_split": source_name,
            "sample_count": len(dataset),
            "species_count": len(set(selected_species_ids)),
            "seed": seed,
            "caption_view": config["sampling"]["caption_view"],
            "holdout_identity": {
                "path": str(holdout_path.relative_to(ROOT_DIR)),
                "sha256": holdout_sha256,
                "committed_indices_loaded_directly": True,
            },
        },
        "transport": {
            **ot,
            "positive_exclusion_semantics": (
                "All candidates sharing the paired image_id are excluded before "
                "top-k selection."
            ),
            "ot_rows_normalized_after_historical_solver": True,
            "identical_support_for_ot_and_uniform": identical_support,
            "mass_audit": transport["transport_mass_audit"],
        },
        "synthetic_equations": {
            "ot": "normalize(sum_{j in S_i} row_normalized_ot_plan_ij * v_j)",
            "uniform": "normalize((1 / |S_i|) * sum_{j in S_i} v_j)",
            "support": "S_i is the identical positive-excluded raw-cosine top-32 support for both methods.",
        },
        "ot_weighted": summaries["ot_weighted"],
        "uniform_weighted": summaries["uniform_weighted"],
        "paired_comparison": summaries["paired_comparison"],
        "existing_diagnostic_preservation": {
            "v1_and_v2_source_files_modified": False,
            "v1_and_v2_output_contracts_modified": False,
            "existing_outputs_rewritten": False,
        },
    }
    output_dir = _resolve_repo_path(config["output"]["directory"])
    paths = write_outputs(output_dir, report, rows)

    print("Frozen CLIP barycentric-weight ablation complete")
    print(f"  checkpoint / pairs: {checkpoint} / {len(dataset)}")
    print(f"  identical support: {identical_support}")
    print(
        "  P(synthetic > hardest real), OT / uniform: "
        f"{report['ot_weighted']['fraction_synthetic_harder_than_hardest_real']:.4f} / "
        f"{report['uniform_weighted']['fraction_synthetic_harder_than_hardest_real']:.4f}"
    )
    print(
        "  mean OT - uniform synthetic similarity: "
        f"{report['paired_comparison']['ot_minus_uniform_synthetic_similarity']['mean']:.6f}"
    )
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hf_cub200_clip_barycentric_weight_ablation.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(load_config(arguments.config))
