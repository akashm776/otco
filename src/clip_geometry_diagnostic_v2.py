"""Run scale-controlled, batch-emulated frozen CLIP diagnostics on CUB-200.

This is an additive V2 experiment. It reads the same deterministic holdout as
V1, performs no optimization, and writes to a separate output directory.
"""

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
from src.clip_geometry_diagnostic import (
    CUBCLIPDiagnosticDataset,
    encode_dataset,
    keep_records_as_list,
    resolve_device,
    stratified_holdout_indices,
)
from src.clip_geometry_v2_metrics import (
    compare_transport_variants,
    compute_transport_variant,
    compute_zero_shot_species_evaluation,
    emulate_batch_local_gates,
    summarize_batch_emulation,
    summarize_transport_variant,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def species_name_from_image_key(image_key):
    """Convert common CUB path/filename formats into a prompt-ready name."""
    key = str(image_key)
    if "/" in key:
        name = key.split("/", 1)[0]
        if "." in name and name.split(".", 1)[0].isdigit():
            name = name.split(".", 1)[1]
    else:
        name = Path(key).stem
        for _ in range(2):
            left, separator, right = name.rpartition("_")
            if separator and right.isdigit():
                name = left
            else:
                break
    return " ".join(name.replace("_", " ").split()).lower()


def build_species_name_map(grouped_split, species_ids):
    mapping = {}
    for group, species_id in zip(grouped_split.groups, species_ids):
        mapping.setdefault(species_id, species_name_from_image_key(group.image_key))
    return mapping


@torch.inference_mode()
def encode_species_prompts(model, processor, prompts, device, batch_size=64):
    features = []
    for start in range(0, len(prompts), batch_size):
        processed = processor(
            text=prompts[start : start + batch_size],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_batch = {
            key: value.to(device)
            for key, value in processed.items()
            if key in ("input_ids", "attention_mask")
        }
        features.append(model.encode_texts(text_batch))
    return torch.cat(features)


def _value(tensor, index):
    return float(tensor[index].detach().cpu().item())


def build_per_query_rows(variant_name, diagnostics, metadata, gate_sim):
    rows = []
    selected = diagnostics["selected_indices"].detach().cpu().tolist()
    hardest = diagnostics["hardest_indices"].detach().cpu().tolist()
    hardest_real_contributors = diagnostics[
        "hardest_real_contributor_indices"
    ].detach().cpu().tolist()
    overlay = diagnostics["per_query_historical_threshold_overlay"]
    for index, anchor in enumerate(metadata):
        selected_record = metadata[selected[index]]
        hardest_record = metadata[hardest[index]]
        hardest_real_contributor_record = metadata[
            hardest_real_contributors[index]
        ]
        synthetic_logit = _value(diagnostics["synthetic_logits"], index)
        rows.append(
            {
                "variant": variant_name,
                "diagnostic_index": index,
                "source_index": anchor["source_index"],
                "image_key": anchor["image_key"],
                "species_id": anchor["species_id"],
                "species_name": anchor["species_name"],
                "caption": anchor["caption"],
                "selected_index": selected[index],
                "selected_image_key": selected_record["image_key"],
                "selected_species_id": selected_record["species_id"],
                "selected_same_species": bool(
                    diagnostics["selected_same_species"][index].item()
                ),
                "hardest_index": hardest[index],
                "hardest_image_key": hardest_record["image_key"],
                "hardest_species_id": hardest_record["species_id"],
                "ot_selected_equals_strict_hardest": selected[index]
                == hardest[index],
                "positive_similarity": _value(
                    diagnostics["positive_similarity"], index
                ),
                "selected_similarity": _value(
                    diagnostics["selected_similarity"], index
                ),
                "positive_selected_gap": _value(
                    diagnostics["positive_selected_gap"], index
                ),
                "selected_rank_global_but_top_k_prefiltered": _value(
                    diagnostics["selected_rank"], index
                ),
                "coupling_entropy": _value(
                    diagnostics["coupling_entropy"], index
                ),
                "normalized_coupling_entropy": _value(
                    diagnostics["normalized_coupling_entropy"], index
                ),
                "coupling_peak_mass": _value(
                    diagnostics["coupling_peak_mass"], index
                ),
                "synthetic_similarity": _value(
                    diagnostics["synthetic_similarity"], index
                ),
                "synthetic_logit": synthetic_logit,
                "hardest_real_contributor_index": hardest_real_contributors[index],
                "hardest_real_contributor_image_key": (
                    hardest_real_contributor_record["image_key"]
                ),
                "hardest_real_contributor_species_id": (
                    hardest_real_contributor_record["species_id"]
                ),
                "hardest_real_contributor_similarity": _value(
                    diagnostics["hardest_real_contributor_similarity"], index
                ),
                "synthetic_vs_hardest_real_similarity_delta": _value(
                    diagnostics[
                        "synthetic_vs_hardest_real_similarity_delta"
                    ],
                    index,
                ),
                "synthetic_vs_hardest_real_logit_delta": _value(
                    diagnostics["synthetic_vs_hardest_real_logit_delta"],
                    index,
                ),
                "synthetic_harder_than_hardest_real_contributor": _value(
                    diagnostics[
                        "synthetic_vs_hardest_real_similarity_delta"
                    ],
                    index,
                )
                > 0,
                "synthetic_similarity_gt_positive": _value(
                    diagnostics["synthetic_similarity"], index
                )
                > _value(diagnostics["positive_similarity"], index),
                "best_same_species_rank": _value(
                    diagnostics["best_same_species_rank"], index
                ),
                "best_wrong_species_rank": _value(
                    diagnostics["best_wrong_species_rank"], index
                ),
                "wrong_species_margin": _value(
                    diagnostics["wrong_species_margin"], index
                ),
                "passes_historical_gate_sim_overlay": synthetic_logit > gate_sim,
                "per_query_historical_threshold_overlay": overlay[index],
            }
        )
    return rows


def write_outputs(output_dir, report, per_query_rows, batch_rows, indices):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_dir / "clip_geometry_v2_report.json",
        "per_query": output_dir / "clip_geometry_v2_per_query.csv",
        "batch_emulation": output_dir / "clip_geometry_v2_batch_emulation.csv",
        "holdout_indices": output_dir / "diagnostic_holdout_indices.json",
    }
    with paths["report"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with paths["per_query"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query_rows[0]))
        writer.writeheader()
        writer.writerows(per_query_rows)
    with paths["batch_emulation"].open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0]))
        writer.writeheader()
        writer.writerows(batch_rows)
    with paths["holdout_indices"].open("w", encoding="utf-8") as handle:
        json.dump(indices, handle, indent=2)
    return paths


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("experiment", {}).get("mode") != "frozen_geometry_diagnostic_v2":
        raise ValueError("V2 runner requires mode=frozen_geometry_diagnostic_v2")
    if config.get("dataset", {}).get("backend") != "hf_cub200":
        raise ValueError("V2 runner requires dataset.backend=hf_cub200")
    if config.get("model", {}).get("backend") != "clip":
        raise ValueError("V2 runner requires model.backend=clip")
    if config.get("sampling", {}).get("source_split", "train") not in {
        "train",
        "val",
    }:
        raise ValueError("sampling.source_split must be 'train' or 'val'")
    if config.get("sampling", {}).get("caption_view") != "canonical_first_caption":
        raise ValueError("V2 runner requires canonical_first_caption sampling")
    required = {
        "historical_scaled_logit_ot",
        "matched_scaled_logit_ot",
        "raw_cosine_ot",
    }
    missing = required - set(config.get("transport_variants", {}))
    if missing:
        raise ValueError(f"Missing required transport variants: {sorted(missing)}")
    return config


def run(config):
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
    grouped_split = train_grouped if source_name == "train" else val_grouped
    species_ids = get_cub200_class_labels(grouped_split)
    species_names = build_species_name_map(grouped_split, species_ids)
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
    image_features_cpu, text_features_cpu, metadata = encode_dataset(
        model, processor, loader, device
    )
    for record in metadata:
        record["species_name"] = species_names[record["species_id"]]
    image_features = image_features_cpu.to(device)
    text_features = text_features_cpu.to(device)
    logit_scale = float(model.get_logit_scale().detach().cpu().item())
    image_ids = [record["image_id"] for record in metadata]
    selected_species_ids = [record["species_id"] for record in metadata]

    thresholds = config["historical_thresholds"]
    variant_results = {}
    variant_summaries = {}
    variant_errors = {}
    per_query_rows = []
    effective_epsilons = {}
    for name, variant in config["transport_variants"].items():
        effective_epsilons[name] = (
            variant["ot_eps"] / logit_scale
            if variant["cost_space"] == "clip_scaled_logits"
            else variant["ot_eps"]
        )
        try:
            diagnostics = compute_transport_variant(
                image_features=image_features,
                text_features=text_features,
                logit_scale=logit_scale,
                image_ids=image_ids,
                species_ids=selected_species_ids,
                cost_space=variant["cost_space"],
                top_k=variant["top_k"],
                ot_eps=variant["ot_eps"],
                sinkhorn_iters=variant["sinkhorn_iters"],
                solver=variant.get("solver", "historical_sparse_ot"),
                historical_thresholds=thresholds,
            )
        except ValueError as error:
            variant_errors[name] = str(error)
            continue
        variant_results[name] = diagnostics
        variant_summaries[name] = summarize_transport_variant(diagnostics)
        per_query_rows.extend(
            build_per_query_rows(name, diagnostics, metadata, thresholds["gate_sim"])
        )

    raw = variant_results["raw_cosine_ot"]
    matched = variant_results["matched_scaled_logit_ot"]
    scale_equivalence = compare_transport_variants(raw, matched)

    batch_variants = {
        name: variant
        for name, variant in config["transport_variants"].items()
        if variant.get("include_in_batch_emulation", True)
        and name in variant_results
    }
    batch_config = config["batch_emulation"]
    batch_rows = emulate_batch_local_gates(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        image_ids=image_ids,
        species_ids=selected_species_ids,
        variants=batch_variants,
        batch_size=batch_config["batch_size"],
        num_batches=batch_config["num_batches"],
        seed=batch_config["seed"],
        scheduled_alpha=batch_config["scheduled_alpha"],
        thresholds=thresholds,
    )

    prompt_species_ids = list(dict.fromkeys(selected_species_ids))
    prompt_template = config["species_evaluation"]["prompt_template"]
    prompts = [
        prompt_template.format(species_name=species_names[species_id])
        for species_id in prompt_species_ids
    ]
    species_text_features = encode_species_prompts(
        model,
        processor,
        prompts,
        device,
        batch_size=config["species_evaluation"].get("batch_size", 64),
    )
    species_evaluation = compute_zero_shot_species_evaluation(
        image_features,
        species_text_features,
        selected_species_ids,
        prompt_species_ids,
    )

    report = {
        "experiment": config["experiment"],
        "no_training_or_gradients": True,
        "v1_preservation": {
            "existing_config": "configs/hf_cub200_clip_geometry.yaml",
            "existing_output_directory": "outputs/cub200_frozen_clip_vit_b32_geometry",
            "historical_condition": "historical_scaled_logit_ot",
        },
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
            "num_species": len(prompt_species_ids),
            "seed": seed,
            "holdout_indices_are_reused_for_all_variants": True,
        },
        "transport_variants": config["transport_variants"],
        "effective_cosine_epsilons": effective_epsilons,
        "full_pool_variant_summaries": variant_summaries,
        "variant_errors": variant_errors,
        "raw_cosine_vs_matched_scaled_logit_equivalence": scale_equivalence,
        "batch_emulation": {
            "config": batch_config,
            "actual_gate_semantics": (
                "compute_alpha_effective applied to batch mean entropy and batch "
                "mean positive-selected gap"
            ),
            "summaries": summarize_batch_emulation(batch_rows),
        },
        "species_evaluation": {
            "prompt_template": prompt_template,
            **species_evaluation,
        },
    }
    output_dir = ROOT_DIR / config["output"]["directory"]
    paths = write_outputs(
        output_dir, report, per_query_rows, batch_rows, selected_indices
    )

    print("Frozen CLIP geometry V2 diagnostic complete")
    print(f"  model/logit scale: {checkpoint} / {logit_scale:.4f}")
    print(f"  pairs/species: {len(dataset)}/{len(prompt_species_ids)}")
    print(
        "  raw-vs-matched selected agreement: "
        f"{scale_equivalence['selected_index_agreement']:.6f}"
    )
    print(
        "  zero-shot species top-1/top-5: "
        f"{species_evaluation['top_1_accuracy']:.4f} / "
        f"{species_evaluation['top_5_accuracy']:.4f}"
    )
    for name, summary in report["batch_emulation"]["summaries"].items():
        print(f"  {name} gate states: {summary['gate_state_fractions']}")
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hf_cub200_clip_geometry_v2.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(load_config(arguments.config))
