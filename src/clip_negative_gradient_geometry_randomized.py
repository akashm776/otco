"""Frozen CLIP tangent-gradient diagnostic across randomized B64 partitions."""

import argparse
import csv
import json
from pathlib import Path
import random
import shutil

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
from src.clip_negative_gradient_geometry import (
    compute_batch_gradient_geometry,
    summarize_rows,
)
from src.clip_negative_gradient_randomized_metrics import (
    annotate_species_composition,
    build_cross_partition_stability,
    evaluate_robustness_criteria,
    flattened_partition_sha256,
    make_partition_indices,
    pairwise_query_correlations,
    same_species_stratification,
    summarize_species_composition,
    validate_partition,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_FILENAME = "negative_gradient_geometry_randomized_report.json"
CSV_FILENAME = "gradient_geometry_randomized_per_query.csv"
PARTITIONS_FILENAME = "batch_partitions.json"


EXPECTED_PARTITIONS = [
    {"name": "sequential", "mode": "sequential"},
    {"name": "shuffle_seed_42", "mode": "shuffled", "seed": 42},
    {"name": "shuffle_seed_123", "mode": "shuffled", "seed": 123},
    {"name": "shuffle_seed_4242", "mode": "shuffled", "seed": 4242},
]


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_config(path):
    """Strictly validate the batch-composition-only frozen extension."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected_experiment = {
        "name": (
            "cub200_frozen_clip_vit_b32_negative_gradient_geometry_"
            "randomized_batches"
        ),
        "mode": "frozen_negative_gradient_geometry_randomized_batches",
    }
    experiment = config.get("experiment", {})
    if {key: experiment.get(key) for key in expected_experiment} != expected_experiment:
        raise ValueError(f"Experiment must identify {expected_experiment}")
    if config.get("model") != {
        "backend": "clip",
        "checkpoint": "openai/clip-vit-base-patch32",
    }:
        raise ValueError("Diagnostic requires frozen OpenAI CLIP ViT-B/32")
    expected_sampling = {
        "source_split": "train",
        "holdout_indices_path": "configs/cub200_clip_diagnostic_holdout_indices.json",
        "holdout_size": 1024,
        "seed": 42,
        "caption_view": "canonical_first_caption",
    }
    if config.get("sampling") != expected_sampling:
        raise ValueError(f"Sampling must exactly equal {expected_sampling}")
    diagnostic = config.get("diagnostic", {})
    if diagnostic.get("batch_size") != 64 or diagnostic.get("top_k") != 8:
        raise ValueError("Diagnostic requires B64 and uniform top-8")
    if diagnostic.get("tangent_tolerance") != 1e-5:
        raise ValueError("Tangent audit tolerance must be exactly 1e-5")
    if diagnostic.get("partitions") != EXPECTED_PARTITIONS:
        raise ValueError(f"Partitions must exactly equal {EXPECTED_PARTITIONS}")
    return config


def build_partition_conditions(partition_specs, count, batch_size):
    """Construct and audit all label-free deterministic partitions."""
    conditions = {}
    artifact = {
        "encoding": {
            "flattened_sha256": (
                "sha256(compact JSON encoding of flattened holdout-position list)"
            ),
            "positions_are_indices_into_committed_holdout": True,
        },
        "partitions": {},
        "flattened_sha256": {},
        "invariants": {},
    }
    for spec in partition_specs:
        seed = spec.get("seed") if spec["mode"] == "shuffled" else None
        batches = make_partition_indices(count, batch_size, seed=seed)
        invariants = validate_partition(batches, count, batch_size)
        name = spec["name"]
        conditions[name] = {
            "name": name,
            "mode": spec["mode"],
            "seed": seed,
            "batches": batches,
        }
        artifact["partitions"][name] = batches
        artifact["flattened_sha256"][name] = flattened_partition_sha256(batches)
        artifact["invariants"][name] = invariants
    return conditions, artifact


def run_feature_partition(
    *,
    condition,
    image_features,
    text_features,
    metadata,
    logit_scale,
    top_k,
    tangent_tolerance,
):
    """Evaluate one exhaustive partition using reviewed tangent-gradient math."""
    rows = []
    query_residuals = []
    image_residuals = []
    device = image_features.device
    for batch_index, positions in enumerate(condition["batches"]):
        cpu_indices = torch.tensor(positions, dtype=torch.long)
        device_indices = cpu_indices.to(device)
        batch_metadata = [metadata[index] for index in positions]
        batch_species = [record["species_id"] for record in batch_metadata]
        result = compute_batch_gradient_geometry(
            image_features.index_select(0, device_indices),
            text_features.index_select(0, device_indices),
            logit_scale,
            species_ids=batch_species,
            top_k=top_k,
        )
        query_residuals.append(
            result["tangent_audit"]["max_abs_query_gradient_dot_embedding"]
        )
        image_residuals.append(
            result["tangent_audit"]["max_abs_image_gradient_dot_embedding"]
        )
        for query_position, (holdout_position, base_row) in enumerate(
            zip(positions, result["rows"])
        ):
            record = metadata[holdout_position]
            row = annotate_species_composition(base_row, batch_species)
            row.update(
                {
                    "partition_name": condition["name"],
                    "partition_seed": condition["seed"],
                    "batch_index": batch_index,
                    "query_position_within_batch": query_position,
                    "query_index": holdout_position,
                    "source_index": record["source_index"],
                    "image_key": record["image_key"],
                }
            )
            rows.append(row)
        del result

    tangent_audit = {
        "tolerance": tangent_tolerance,
        "max_abs_query_gradient_dot_embedding": max(query_residuals),
        "max_abs_image_gradient_dot_embedding": max(image_residuals),
    }
    tangent_audit["query_pass"] = (
        tangent_audit["max_abs_query_gradient_dot_embedding"] < tangent_tolerance
    )
    tangent_audit["image_pass"] = (
        tangent_audit["max_abs_image_gradient_dot_embedding"] < tangent_tolerance
    )
    if not tangent_audit["query_pass"] or not tangent_audit["image_pass"]:
        raise AssertionError(
            f"Partition {condition['name']} failed tangent audit: {tangent_audit}"
        )
    if len(rows) != len(metadata):
        raise AssertionError("Each partition must emit exactly one row per example")
    if {row["query_index"] for row in rows} != set(range(len(metadata))):
        raise AssertionError("Each holdout position must appear once per partition")
    return rows, tangent_audit


def _partition_summary(rows, tangent_audit):
    summary = summarize_rows(rows)
    summary["tangent_gradient_audit"] = tangent_audit
    summary["species_batch_composition"] = summarize_species_composition(rows)
    return summary


def write_outputs(
    output_dir,
    *,
    report,
    rows,
    config,
    holdout_path,
    partition_artifact,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    csv_path = output_dir / CSV_FILENAME
    config_path = output_dir / "resolved_config.yaml"
    holdout_copy = output_dir / "diagnostic_holdout_indices.json"
    partitions_path = output_dir / PARTITIONS_FILENAME
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with partitions_path.open("w", encoding="utf-8") as handle:
        json.dump(partition_artifact, handle, indent=2)
    shutil.copyfile(holdout_path, holdout_copy)
    return [report_path, csv_path, config_path, holdout_copy, partitions_path]


def run(config):
    """Encode once, then rerun identical gradient math under four partitions."""
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
    grouped = (
        train_grouped
        if config["sampling"]["source_split"] == "train"
        else val_grouped
    )
    validate_holdout_indices(holdout, grouped, config["sampling"]["holdout_size"])
    species_ids = get_cub200_class_labels(grouped)
    dataset = CUBCLIPDiagnosticDataset(grouped, species_ids, holdout)
    loader = DataLoader(
        dataset,
        batch_size=config["runtime"]["encoding_batch_size"],
        shuffle=False,
        num_workers=config["runtime"]["num_workers"],
        collate_fn=keep_records_as_list,
        pin_memory=device.type == "cuda",
    )

    checkpoint = config["model"]["checkpoint"]
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = CLIPEncoderBackend.from_pretrained(checkpoint).to(device).freeze()
    image_cpu, text_cpu, metadata = encode_dataset(model, processor, loader, device)
    logit_scale = model.get_logit_scale().detach()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Frozen encoding populated model parameter gradients")

    diagnostic = config["diagnostic"]
    conditions, partition_artifact = build_partition_conditions(
        diagnostic["partitions"], len(dataset), diagnostic["batch_size"]
    )
    image_features = image_cpu.to(device)
    text_features = text_cpu.to(device)
    rows_by_partition = {}
    partition_reports = {}
    all_rows = []
    for name, condition in conditions.items():
        print(f"Starting partition: {name}", flush=True)
        rows, tangent_audit = run_feature_partition(
            condition=condition,
            image_features=image_features,
            text_features=text_features,
            metadata=metadata,
            logit_scale=logit_scale,
            top_k=diagnostic["top_k"],
            tangent_tolerance=diagnostic["tangent_tolerance"],
        )
        rows_by_partition[name] = rows
        partition_reports[name] = _partition_summary(rows, tangent_audit)
        all_rows.extend(rows)
        joint = partition_reports[name]["joint_gradient_alignment"][
            "cosine_u8_vs_hardest_real"
        ]
        print(
            f"Completed {name}: joint mean={joint['mean']:+.6f}, "
            f"fraction<0={joint['fraction_lt_0']:.4f}",
            flush=True,
        )

    if len(all_rows) != len(conditions) * len(dataset):
        raise AssertionError("Randomized diagnostic must emit exactly 4096 rows")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Embedding autograd populated model parameter gradients")
    shuffled_names = [
        spec["name"] for spec in diagnostic["partitions"] if spec["mode"] == "shuffled"
    ]
    report = {
        "experiment": config["experiment"],
        "model": {
            "checkpoint": checkpoint,
            "frozen": True,
            "all_parameter_grads_none": True,
            "logit_scale": float(logit_scale.cpu().item()),
        },
        "no_optimizer_no_parameter_updates_no_training": True,
        "runtime": device_info,
        "dataset": {
            **split_info,
            "source_split": config["sampling"]["source_split"],
            "sample_count": len(dataset),
            "caption_view": config["sampling"]["caption_view"],
            "holdout_path": str(Path(holdout_path).relative_to(ROOT_DIR)),
            "holdout_sha256": holdout_sha256,
            "same_examples_in_every_partition": True,
        },
        "scientific_control": {
            "only_variable": "partition of identical holdout positions into B64 batches",
            "labels_used_for_partitioning": False,
            "examples_resampled": False,
            "caption_selection_changed": False,
            "candidate_objectives_or_tangent_gradients_changed": False,
        },
        "partition_method": {
            "helper": (
                "src.clip_negative_gradient_randomized_metrics."
                "make_partition_indices"
            ),
            "algorithm": (
                "Sequential uses torch.arange(count). Shuffled conditions use "
                "torch.randperm(count, generator=torch.Generator('cpu').manual_seed(seed)); "
                "each order is split into 16 non-overlapping contiguous chunks of 64."
            ),
            "conditions": diagnostic["partitions"],
        },
        "candidate_and_gradient_semantics": {
            "u8": (
                "normalize((1/8) sum of exact positive-excluded raw-cosine top-8 "
                "live image embeddings); selection and weights stop-gradient"
            ),
            "hardest_real": (
                "argmax_{j != i} q_i dot v_j with detached index and live selected logit"
            ),
            "relative_objective": (
                "existing clip_relative_denominator_loss with detached CLIP scale"
            ),
            "tangent_gradients": (
                "existing reviewed unit-norm pre-normalization leaves and normalization "
                "Jacobian; unchanged from the sequential diagnostic"
            ),
        },
        "partitions": partition_reports,
        "cross_partition_stability": build_cross_partition_stability(
            partition_reports, shuffled_names
        ),
        "robustness_criteria": evaluate_robustness_criteria(
            partition_reports, shuffled_names
        ),
        "pairwise_per_query_stability": pairwise_query_correlations(
            rows_by_partition, shuffled_names
        ),
        "same_species_stratification_shuffled_only": {
            name: same_species_stratification(rows_by_partition[name])
            for name in shuffled_names
        },
        "output_cardinality": {
            "partition_count": len(conditions),
            "queries_per_partition": len(dataset),
            "per_query_csv_rows": len(all_rows),
        },
    }
    output_dir = _resolve_repo_path(config["output"]["directory"])
    paths = write_outputs(
        output_dir,
        report=report,
        rows=all_rows,
        config=config,
        holdout_path=holdout_path,
        partition_artifact=partition_artifact,
    )
    print("Randomized-batch negative-gradient geometry diagnostic complete")
    print(
        "  all shuffled partitions pass robustness criteria: "
        f"{report['robustness_criteria']['all_three_shuffled_partitions_pass_all_criteria']}"
    )
    for path in paths:
        print(f"  wrote: {path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "configs/hf_cub200_clip_negative_gradient_geometry_"
            "randomized_batches.yaml"
        ),
    )
    parser.add_argument("--output-directory", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved = load_config(arguments.config)
    if arguments.output_directory:
        resolved["output"]["directory"] = arguments.output_directory
    run(resolved)
