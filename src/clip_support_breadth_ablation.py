"""Run the frozen B64 CLIP OT-vs-uniform support-breadth ablation."""

import argparse
import csv
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
import yaml

from data.hf_cub200_dataset import get_cub200_class_labels, load_hf_cub200_splits
from model.clip_backend import CLIPEncoderBackend
from src.clip_barycentric_weight_ablation import (
    load_committed_holdout,
    validate_holdout_indices,
)
from src.clip_geometry_diagnostic import (
    CUBCLIPDiagnosticDataset,
    encode_dataset,
    keep_records_as_list,
    resolve_device,
)
from src.clip_geometry_v2_metrics import deterministic_batches
from src.clip_support_breadth_metrics import (
    build_support_breadth_curve,
    compute_support_breadth_batch,
    initialize_k_accumulators,
    summarize_support_breadth_k,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_config(path):
    """Load and strictly validate the single-variable frozen ablation."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("experiment", {}).get("mode") != "frozen_support_breadth_ablation":
        raise ValueError("Runner requires mode=frozen_support_breadth_ablation")
    if config.get("dataset", {}).get("backend") != "hf_cub200":
        raise ValueError("Runner requires dataset.backend=hf_cub200")
    if config.get("model", {}) != {
        "backend": "clip",
        "checkpoint": "openai/clip-vit-base-patch32",
    }:
        raise ValueError("Runner requires the frozen OpenAI CLIP ViT-B/32 backend")
    sampling = config.get("sampling", {})
    if sampling.get("caption_view") != "canonical_first_caption":
        raise ValueError("Runner requires canonical_first_caption sampling")
    if sampling.get("holdout_size") != 1024 or sampling.get("seed") != 42:
        raise ValueError("Runner requires the committed 1024-example seed-42 holdout")

    batch = config.get("batch_emulation", {})
    expected_batch = {"batch_size": 64, "num_batches": 200, "seed": 4242}
    if batch != expected_batch:
        raise ValueError(f"Batch emulation must exactly equal {expected_batch}")
    ot = config.get("ot", {})
    expected_ot = {
        "solver": "historical_sparse_ot",
        "cost_space": "raw_cosine",
        "ot_eps": 0.049,
        "sinkhorn_iters": 30,
        "exclude_same_image_positives": True,
        "k_values": [2, 4, 8, 16, 32],
    }
    if ot != expected_ot:
        raise ValueError(f"Support-breadth OT config must exactly equal {expected_ot}")
    comparison = config.get("comparison", {})
    expected_comparison = {
        "tie_tolerance": 1e-6,
        "bootstrap_seed": 42,
        "bootstrap_samples": 10000,
    }
    if comparison != expected_comparison:
        raise ValueError(f"Comparison config must exactly equal {expected_comparison}")
    return config


def _scalar(value, index):
    return float(value[index].detach().cpu().item())


def build_observation_rows(
    *, batch_index, pool_indices, metadata, top_k, result
):
    """Create stable batch-query rows for one k."""
    diagnostics = result["diagnostics"]
    eligible = result["eligible_negative_count"]
    support_fraction = result["support_fraction"]
    rows = []
    for query_index, pool_index in enumerate(pool_indices.tolist()):
        anchor = metadata[pool_index]
        ot_minus_uniform = _scalar(
            diagnostics["ot_minus_uniform_synthetic_similarity"], query_index
        )
        rows.append(
            {
                "batch_index": batch_index,
                "query_index_within_batch": query_index,
                "observation_id": f"batch_{batch_index:04d}_query_{query_index:02d}",
                "holdout_pool_index": pool_index,
                "source_index": anchor["source_index"],
                "image_key": anchor["image_key"],
                "species_id": anchor["species_id"],
                "top_k": top_k,
                "eligible_negative_count": int(eligible[query_index].item()),
                "support_fraction": _scalar(support_fraction, query_index),
                "positive_similarity": _scalar(
                    diagnostics["positive_similarity"], query_index
                ),
                "hardest_real_similarity": _scalar(
                    diagnostics["hardest_real_similarity"], query_index
                ),
                "ot_synthetic_similarity": _scalar(
                    diagnostics["ot_synthetic_similarity"], query_index
                ),
                "uniform_synthetic_similarity": _scalar(
                    diagnostics["uniform_synthetic_similarity"], query_index
                ),
                "ot_vs_hardest_delta": _scalar(
                    diagnostics["ot_vs_hardest_delta"], query_index
                ),
                "uniform_vs_hardest_delta": _scalar(
                    diagnostics["uniform_vs_hardest_delta"], query_index
                ),
                "uniform_minus_ot_similarity": -ot_minus_uniform,
                "ot_harder_than_hardest": bool(
                    diagnostics["ot_harder_than_hardest_real"][query_index].item()
                ),
                "uniform_harder_than_hardest": bool(
                    diagnostics["uniform_harder_than_hardest_real"]
                    [query_index].item()
                ),
                "uniform_harder_than_ot": bool(
                    diagnostics["uniform_harder_than_ot"][query_index].item()
                ),
                "ot_harder_than_uniform": bool(
                    diagnostics["ot_harder_than_uniform"][query_index].item()
                ),
                "ot_weight_entropy": _scalar(
                    diagnostics["ot_weight_entropy"], query_index
                ),
                "ot_normalized_entropy": _scalar(
                    diagnostics["ot_weight_normalized_entropy"], query_index
                ),
                "ot_peak_weight": _scalar(
                    diagnostics["ot_peak_weight"], query_index
                ),
                "uniform_weight_entropy": _scalar(
                    diagnostics["uniform_weight_entropy"], query_index
                ),
                "uniform_normalized_entropy": _scalar(
                    diagnostics["uniform_weight_normalized_entropy"], query_index
                ),
                "uniform_peak_weight": _scalar(
                    diagnostics["uniform_peak_weight"], query_index
                ),
                "l1_ot_vs_uniform_weights": _scalar(
                    diagnostics["l1_distance_ot_vs_uniform_weights"], query_index
                ),
                "cosine_ot_uniform_synthetic": _scalar(
                    diagnostics["cosine_between_ot_and_uniform_synthetic"],
                    query_index,
                ),
            }
        )
    return rows


def write_outputs(output_dir, report, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "clip_support_breadth_ablation_report.json"
    rows_path = output_dir / "clip_support_breadth_ablation_per_observation.csv"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"report": report_path, "per_observation": rows_path}


@torch.inference_mode()
def run(config):
    """Encode the fixed holdout once and evaluate the paired B64 k sweep."""
    from transformers import AutoProcessor

    seed = config["sampling"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    device, device_info = resolve_device(config["runtime"])
    holdout, holdout_sha256, holdout_path = load_committed_holdout(
        config["sampling"]["holdout_indices_path"]
    )

    dataset_config = config["dataset"]
    train_grouped, val_grouped, split_info = load_hf_cub200_splits(
        dataset_name=dataset_config["dataset_name"],
        train_hf_split=dataset_config.get("train_split", "train"),
        val_hf_split=dataset_config.get("val_split", "test"),
    )
    source_name = config["sampling"]["source_split"]
    grouped_split = train_grouped if source_name == "train" else val_grouped
    validate_holdout_indices(
        holdout, grouped_split, config["sampling"]["holdout_size"]
    )
    species_ids = get_cub200_class_labels(grouped_split)
    dataset = CUBCLIPDiagnosticDataset(grouped_split, species_ids, holdout)
    loader = DataLoader(
        dataset,
        batch_size=config["runtime"]["batch_size"],
        shuffle=False,
        num_workers=config["runtime"]["num_workers"],
        collate_fn=keep_records_as_list,
        pin_memory=device.type == "cuda",
    )

    checkpoint = config["model"]["checkpoint"]
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = CLIPEncoderBackend.from_pretrained(checkpoint).to(device).freeze()
    image_cpu, text_cpu, metadata = encode_dataset(
        model, processor, loader, device
    )
    image_features = image_cpu.to(device)
    text_features = text_cpu.to(device)
    image_ids = [record["image_id"] for record in metadata]
    selected_species_ids = [record["species_id"] for record in metadata]

    batch_config = config["batch_emulation"]
    batches = deterministic_batches(
        len(metadata),
        batch_config["batch_size"],
        batch_config["num_batches"],
        batch_config["seed"],
    )
    ot = config["ot"]
    comparison = config["comparison"]
    accumulators = initialize_k_accumulators(ot["k_values"])
    rows = []
    for batch_index, cpu_indices in enumerate(batches):
        indices = cpu_indices.to(device)
        batch_image_ids = [image_ids[index] for index in cpu_indices.tolist()]
        batch_species_ids = [
            selected_species_ids[index] for index in cpu_indices.tolist()
        ]
        sweep = compute_support_breadth_batch(
            image_features=image_features.index_select(0, indices),
            text_features=text_features.index_select(0, indices),
            image_ids=batch_image_ids,
            species_ids=batch_species_ids,
            k_values=ot["k_values"],
            ot_eps=ot["ot_eps"],
            sinkhorn_iters=ot["sinkhorn_iters"],
            solver=ot["solver"],
            tie_tolerance=comparison["tie_tolerance"],
        )
        for top_k, result in sweep.items():
            bucket = accumulators[top_k]
            bucket["batch_diagnostics"].append(result["diagnostics"])
            bucket["eligible_counts"].append(
                result["eligible_negative_count"].detach().cpu()
            )
            bucket["support_fractions"].append(
                result["support_fraction"].detach().cpu()
            )
            bucket["mass_audits"].append(
                result["transport"]["transport_mass_audit"]
            )
            bucket["reference_errors"].append(
                result["max_abs_ot_similarity_vs_existing_v2"]
            )
            rows.extend(
                build_observation_rows(
                    batch_index=batch_index,
                    pool_indices=cpu_indices,
                    metadata=metadata,
                    top_k=top_k,
                    result=result,
                )
            )

    per_k = {}
    for top_k, bucket in accumulators.items():
        per_k[top_k] = summarize_support_breadth_k(
            top_k=top_k,
            bootstrap_seed=comparison["bootstrap_seed"],
            bootstrap_samples=comparison["bootstrap_samples"],
            **bucket,
        )
    curve = build_support_breadth_curve(per_k)
    k32 = per_k[32]
    report = {
        "experiment": config["experiment"],
        "checkpoint": checkpoint,
        "frozen": True,
        "no_training_or_gradients": True,
        "runtime": device_info,
        "dataset": {
            **split_info,
            "diagnostic_source_split": source_name,
            "holdout_identity": {
                "path": str(holdout_path.relative_to(ROOT_DIR)),
                "sha256": holdout_sha256,
                "committed_indices_loaded_directly": True,
            },
            "sample_count": len(dataset),
            "species_count": len(set(selected_species_ids)),
            "caption_view": config["sampling"]["caption_view"],
            "model_data_seed": seed,
        },
        "batch_emulation": {
            **batch_config,
            "helper": "src.clip_geometry_v2_metrics.deterministic_batches",
            "semantics": (
                "Repeated torch.randperm(count, generator=seed) permutations; "
                "each permutation is partitioned into full B64 batches until "
                "exactly 200 batches are collected. The resulting index tensors "
                "are generated once and reused unchanged for every k."
            ),
            "same_batches_and_candidate_pools_across_all_k": True,
            "underlying_examples_may_repeat_across_batches": True,
        },
        "transport": {
            "solver": ot["solver"],
            "cost_space": ot["cost_space"],
            "ot_eps": ot["ot_eps"],
            "sinkhorn_iters": ot["sinkhorn_iters"],
            "positive_exclusion_semantics": (
                "All candidates sharing the paired image_id are excluded before "
                "raw-cosine top-k selection, via the existing V2 helper."
            ),
            "k_values": ot["k_values"],
            "fixed_across_k": ["cost_space", "ot_eps", "sinkhorn_iters", "solver"],
            "ot_and_uniform_support_identical_within_each_k": True,
        },
        "synthetic_equations": {
            "ot": "normalize(sum_{j in S_i^k} (pi_ij / sum_m pi_im) * v_j)",
            "uniform": "normalize((1 / |S_i^k|) * sum_{j in S_i^k} v_j)",
            "shared_support": (
                "S_i^k is the identical same-image-positive-excluded raw-cosine "
                "top-k support for OT and uniform weighting."
            ),
        },
        "comparison": {
            **comparison,
            "bootstrap_interpretation": (
                "Observation-level descriptive paired CIs, not iid population "
                "guarantees, because deterministic batches reuse holdout examples."
            ),
        },
        "per_k": {str(top_k): per_k[top_k] for top_k in sorted(per_k)},
        "support_breadth_curve": curve,
        "existing_v2_k32_reference": {
            "construction_helper_reused": (
                "src.clip_geometry_v2_metrics.compute_transport_variant"
            ),
            "observed_fraction_ot_synthetic_gt_hardest": k32["ot_weighted"]
            ["fraction_synthetic_harder_than_hardest_real"],
            "observed_mean_ot_synthetic_minus_hardest": k32["ot_weighted"]
            ["synthetic_minus_hardest_real"]["mean"],
            "prior_reported_context_approximate": {
                "fraction": 0.876,
                "mean_delta": 0.014,
                "used_as_pass_fail_test": False,
            },
        },
        "existing_diagnostic_preservation": {
            "v1_v2_and_full_pool_sources_modified": False,
            "existing_outputs_rewritten": False,
        },
    }
    output_dir = _resolve_repo_path(config["output"]["directory"])
    paths = write_outputs(output_dir, report, rows)

    print("Frozen CLIP support-breadth ablation complete")
    print(f"  checkpoint / holdout / batches: {checkpoint} / {len(dataset)} / {len(batches)}")
    for row in curve:
        print(
            f"  k={row['top_k']:>2}: OT={row['ot_fraction_gt_hardest']:.4f}, "
            f"uniform={row['uniform_fraction_gt_hardest']:.4f}, "
            f"mean(U-OT)={row['uniform_minus_ot_mean_similarity']:+.6f}"
        )
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hf_cub200_clip_support_breadth_ablation.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(load_config(arguments.config))
