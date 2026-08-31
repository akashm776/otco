"""Run the frozen CLIP geometry-adaptive neighborhood diagnostic."""

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
from src.clip_geometry_adaptive_metrics import (
    compute_geometry_adaptive_batch,
    summarize_geometry_adaptive,
)
from src.clip_geometry_diagnostic import (
    CUBCLIPDiagnosticDataset,
    encode_dataset,
    keep_records_as_list,
    resolve_device,
)
from src.clip_geometry_v2_metrics import deterministic_batches


ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_config(path):
    """Load and strictly validate this single geometry-selector diagnostic."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("experiment", {}).get("mode") != (
        "frozen_geometry_adaptive_neighborhood"
    ):
        raise ValueError(
            "Runner requires mode=frozen_geometry_adaptive_neighborhood"
        )
    if config.get("dataset", {}).get("backend") != "hf_cub200":
        raise ValueError("Runner requires dataset.backend=hf_cub200")
    if config.get("model", {}) != {
        "backend": "clip",
        "checkpoint": "openai/clip-vit-base-patch32",
    }:
        raise ValueError("Runner requires frozen OpenAI CLIP ViT-B/32")
    sampling = config.get("sampling", {})
    if sampling.get("holdout_size") != 1024 or sampling.get("seed") != 42:
        raise ValueError("Runner requires the committed seed-42 1024-example holdout")
    if sampling.get("caption_view") != "canonical_first_caption":
        raise ValueError("Runner requires canonical_first_caption")
    expected_batch = {"batch_size": 64, "num_batches": 200, "seed": 4242}
    if config.get("batch_emulation") != expected_batch:
        raise ValueError(f"Batch emulation must exactly equal {expected_batch}")
    expected_selector = {
        "similarity_metric": "raw_cosine",
        "allowed_k_values": [2, 4, 8, 16, 32],
        "selector": "largest_boundary_gap",
        "selector_tie_tolerance": 1e-12,
        "selector_tie_break": "smaller_k",
        "exclude_same_image_positives": True,
    }
    if config.get("selector") != expected_selector:
        raise ValueError(f"Selector config must exactly equal {expected_selector}")
    expected_comparison = {
        "primary_fixed_k": 8,
        "tie_tolerance": 1e-6,
        "bootstrap_seed": 42,
        "bootstrap_samples": 10000,
    }
    if config.get("comparison") != expected_comparison:
        raise ValueError(f"Comparison config must exactly equal {expected_comparison}")
    return config


def _scalar(tensor, index):
    return float(tensor[index].detach().cpu().item())


def build_observation_rows(
    *, batch_index, pool_indices, metadata, diagnostics
):
    """Emit one paired geometry/oracle/fixed-control row per batch query."""
    choices = diagnostics["allowed_k_values"]
    rows = []
    for query_index, pool_index in enumerate(pool_indices.tolist()):
        anchor = metadata[pool_index]
        geometry_similarity = _scalar(
            diagnostics["geometry_synthetic_similarity"], query_index
        )
        oracle_similarity = _scalar(
            diagnostics["oracle_synthetic_similarity"], query_index
        )
        fixed_similarities = {
            top_k: _scalar(
                diagnostics["fixed_by_k"][top_k]["synthetic_similarity"],
                query_index,
            )
            for top_k in choices
        }
        positive = _scalar(diagnostics["positive_similarity"], query_index)
        hardest = _scalar(diagnostics["hardest_real_similarity"], query_index)
        selected_k = int(diagnostics["geometry_selected_k"][query_index].item())
        oracle_k = int(diagnostics["oracle_selected_k"][query_index].item())
        row = {
            "batch_index": batch_index,
            "query_index_within_batch": query_index,
            "observation_id": f"batch_{batch_index:04d}_query_{query_index:02d}",
            "holdout_pool_index": pool_index,
            "source_index": anchor["source_index"],
            "image_key": anchor["image_key"],
            "species_id": anchor["species_id"],
            "eligible_negative_count": int(
                diagnostics["eligible_negative_count"][query_index].item()
            ),
        }
        for choice_index, top_k in enumerate(choices):
            row[f"gap_after_k{top_k}"] = _scalar(
                diagnostics["boundary_gaps"][:, choice_index], query_index
            )
        row.update(
            {
                "selected_boundary_gap": _scalar(
                    diagnostics["selected_boundary_gap"], query_index
                ),
                "geometry_selected_k": selected_k,
                "oracle_selected_k": oracle_k,
                "geometry_selected_support_fraction": selected_k
                / row["eligible_negative_count"],
                "positive_similarity": positive,
                "hardest_real_similarity": hardest,
                **{
                    f"uniform_k{top_k}_similarity": fixed_similarities[top_k]
                    for top_k in choices
                },
                "geometry_synthetic_similarity": geometry_similarity,
                "oracle_synthetic_similarity": oracle_similarity,
                "geometry_minus_k8_similarity": (
                    geometry_similarity - fixed_similarities[8]
                ),
                "oracle_minus_geometry_similarity": (
                    oracle_similarity - geometry_similarity
                ),
                "geometry_vs_hardest_delta": geometry_similarity - hardest,
                "oracle_vs_hardest_delta": oracle_similarity - hardest,
                "k8_vs_hardest_delta": fixed_similarities[8] - hardest,
                "geometry_harder_than_hardest": geometry_similarity > hardest,
                "oracle_harder_than_hardest": oracle_similarity > hardest,
                "k8_harder_than_hardest": fixed_similarities[8] > hardest,
            }
        )
        rows.append(row)
    return rows


def write_outputs(output_dir, report, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "clip_geometry_adaptive_neighborhood_report.json"
    rows_path = (
        output_dir / "clip_geometry_adaptive_neighborhood_per_observation.csv"
    )
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"report": report_path, "per_observation": rows_path}


@torch.inference_mode()
def run(config):
    """Encode once and run the geometry-only selector on fixed B64 pools."""
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
    image_cpu, text_cpu, metadata = encode_dataset(model, processor, loader, device)
    image_features = image_cpu.to(device)
    text_features = text_cpu.to(device)
    image_ids = [record["image_id"] for record in metadata]

    batch_config = config["batch_emulation"]
    batches = deterministic_batches(
        len(metadata),
        batch_config["batch_size"],
        batch_config["num_batches"],
        batch_config["seed"],
    )
    selector = config["selector"]
    comparison = config["comparison"]
    rows = []
    for batch_index, cpu_indices in enumerate(batches):
        indices = cpu_indices.to(device)
        diagnostics = compute_geometry_adaptive_batch(
            image_features=image_features.index_select(0, indices),
            text_features=text_features.index_select(0, indices),
            image_ids=[image_ids[index] for index in cpu_indices.tolist()],
            allowed_k_values=selector["allowed_k_values"],
            selector_tie_tolerance=selector["selector_tie_tolerance"],
            comparison_tie_tolerance=comparison["tie_tolerance"],
        )
        rows.extend(
            build_observation_rows(
                batch_index=batch_index,
                pool_indices=cpu_indices,
                metadata=metadata,
                diagnostics=diagnostics,
            )
        )

    summaries = summarize_geometry_adaptive(
        rows,
        allowed_k_values=selector["allowed_k_values"],
        tie_tolerance=comparison["tie_tolerance"],
        bootstrap_seed=comparison["bootstrap_seed"],
        bootstrap_samples=comparison["bootstrap_samples"],
    )
    eligible_values = sorted({row["eligible_negative_count"] for row in rows})
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
            "caption_view": config["sampling"]["caption_view"],
            "model_data_seed": seed,
        },
        "batch_emulation": {
            **batch_config,
            "helper": "src.clip_geometry_v2_metrics.deterministic_batches",
            "semantics": (
                "Repeated torch.randperm(count, generator=seed) permutations; "
                "full B64 partitions are collected until exactly 200 batches."
            ),
            "candidate_pools_match_support_breadth_diagnostic": True,
            "candidate_negative_count": (
                eligible_values[0] if len(eligible_values) == 1 else eligible_values
            ),
        },
        "selector": {
            "definition": (
                "k_i = argmax_{k in {2,4,8,16,32}} "
                "(s_i(rank k) - s_i(rank k+1))"
            ),
            "input_api": "eligible negative raw-cosine similarities only",
            "forbidden_inputs_absent": [
                "synthetic_similarity",
                "OT_weights_or_plan",
                "species_or_labels",
                "retrieval_metrics",
                "gradients_or_loss",
            ],
            "tie_break": "smaller k within numerical tolerance",
            **selector,
        },
        "positive_exclusion_semantics": (
            "All candidates sharing paired image_id are excluded before sorting "
            "and top-k support construction."
        ),
        "uniform_construction": {
            "support_helper": "src.clip_geometry_v2_metrics._top_k_support",
            "weight_helper": (
                "src.clip_barycentric_weight_metrics.uniform_weights_from_support"
            ),
            "barycenter_helper": (
                "src.clip_barycentric_weight_metrics."
                "construct_normalized_barycenters"
            ),
            "equation": "normalize((1 / k_i) * sum_{j in top-k_i} v_j)",
            "fixed_k8_construction_unchanged": True,
            "sinkhorn_or_ot_used": False,
        },
        "oracle": {
            "definition": (
                "evaluation-only argmax over uniform synthetic similarities for "
                "k in {2,4,8,16,32}, smaller-k tie break"
            ),
            "oracle_is_evaluation_only": True,
            "oracle_does_not_feed_geometry_selector": True,
        },
        "comparison": comparison,
        **summaries,
    }
    output_dir = _resolve_repo_path(config["output"]["directory"])
    paths = write_outputs(output_dir, report, rows)

    selected = report["geometry_gap_adaptive"][
        "selected_neighborhood_distribution"
    ]
    headroom = report["adaptive_headroom"]
    print("Frozen CLIP geometry-adaptive neighborhood diagnostic complete")
    print(f"  checkpoint / observations: {checkpoint} / {len(rows)}")
    print("  selected k distribution:", json.dumps(selected, sort_keys=True))
    print(
        "  mean geometry-k8 / oracle-geometry: "
        f"{report['geometry_vs_fixed_k8']['geometry_minus_fixed_k8_synthetic_similarity']['mean']:+.6f} / "
        f"{report['geometry_vs_oracle']['oracle_minus_geometry_synthetic_similarity']['mean']:+.6f}"
    )
    print(
        "  oracle gain recovered:",
        headroom["fraction_of_oracle_gain_recovered"],
    )
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hf_cub200_clip_geometry_adaptive_neighborhood.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(load_config(arguments.config))
