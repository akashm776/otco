"""Frozen CLIP embedding-gradient geometry for U8 and hardest-real negatives."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from data.hf_cub200_dataset import get_cub200_class_labels, load_hf_cub200_splits
from model.clip_backend import CLIPEncoderBackend
from model.clip_training import (
    clip_relative_denominator_loss,
    hardest_real_negative_indices,
    synthetic_barycentric_weights,
)
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
from src.clip_geometry_v2_metrics import _top_k_support
from src.clip_negative_gradient_geometry_metrics import (
    alignment_summary,
    correlation_pair,
    cosine_alignment,
    distribution_summary,
    gradient_footprint,
    gradient_share_on_indices,
    margin_directional_change,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_FILENAME = "negative_gradient_geometry_report.json"
CSV_FILENAME = "gradient_geometry_per_query.csv"


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_config(path):
    """Load and strictly validate the frozen one-question diagnostic."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected_experiment = {
        "name": "cub200_frozen_clip_vit_b32_negative_gradient_geometry",
        "mode": "frozen_negative_gradient_geometry",
    }
    experiment = config.get("experiment", {})
    if {key: experiment.get(key) for key in expected_experiment} != expected_experiment:
        raise ValueError(f"Experiment must identify {expected_experiment}")
    if config.get("model") != {
        "backend": "clip",
        "checkpoint": "openai/clip-vit-base-patch32",
    }:
        raise ValueError("Diagnostic requires frozen OpenAI CLIP ViT-B/32")
    sampling = config.get("sampling", {})
    required_sampling = {
        "source_split": "train",
        "holdout_indices_path": "configs/cub200_clip_diagnostic_holdout_indices.json",
        "holdout_size": 1024,
        "seed": 42,
        "caption_view": "canonical_first_caption",
    }
    if sampling != required_sampling:
        raise ValueError(f"Sampling must exactly equal {required_sampling}")
    diagnostic = config.get("diagnostic", {})
    if diagnostic != {"batch_size": 64, "top_k": 8}:
        raise ValueError("Diagnostic must use sequential B64 batches and uniform top-8")
    return config


def _scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def _joint(query_gradient, image_gradient):
    return torch.cat((query_gradient.reshape(-1), image_gradient.reshape(-1)))


def _autograd_pair(loss, queries, images):
    query_gradient, image_gradient = torch.autograd.grad(
        loss,
        (queries, images),
        retain_graph=True,
        create_graph=False,
    )
    return query_gradient, image_gradient


def compute_batch_gradient_geometry(
    image_features,
    text_features,
    logit_scale,
    *,
    species_ids=None,
    top_k=8,
):
    """Compute paired per-query embedding gradients on one frozen feature batch."""
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError("Image and text features must have equal [B, D] shapes")
    batch_size = image_features.shape[0]
    if batch_size <= top_k:
        raise ValueError("Batch must contain more candidates than top_k")
    if species_ids is not None and len(species_ids) != batch_size:
        raise ValueError("species_ids length must match batch size")

    images = F.normalize(image_features.detach(), dim=-1).clone().requires_grad_(True)
    queries = F.normalize(text_features.detach(), dim=-1).clone().requires_grad_(True)
    raw_similarity = queries @ images.T
    positive_mask = torch.eye(batch_size, dtype=torch.bool, device=images.device)

    # Selection is explicitly stop-gradient and reuses the established top-k helper.
    _, support_mask = _top_k_support(
        raw_similarity.detach(), positive_mask, top_k
    )
    dummy_plan = support_mask.to(raw_similarity.dtype)
    uniform_weights = synthetic_barycentric_weights(
        dummy_plan, support_mask, mode="uniform_topk"
    ).detach()
    u8_synthetic = F.normalize(uniform_weights @ images, dim=-1)
    hardest_indices = hardest_real_negative_indices(raw_similarity)

    detached_scale = torch.as_tensor(
        logit_scale, device=images.device, dtype=images.dtype
    ).detach()
    base_logits = raw_similarity * detached_scale
    row_indices = torch.arange(batch_size, device=images.device)
    u8_logits = (queries * u8_synthetic).sum(1) * detached_scale
    hardest_logits = base_logits[row_indices, hardest_indices]

    rows = []
    for query_index in range(batch_size):
        base_row = base_logits[query_index : query_index + 1]
        u8_loss, _, _ = clip_relative_denominator_loss(
            base_row, u8_logits[query_index : query_index + 1]
        )
        real_loss, _, _ = clip_relative_denominator_loss(
            base_row, hardest_logits[query_index : query_index + 1]
        )
        native_loss = (
            torch.logsumexp(base_logits[query_index], dim=0)
            - base_logits[query_index, query_index]
        )
        hardest_index = int(hardest_indices[query_index].item())
        margin = (
            raw_similarity[query_index, query_index]
            - raw_similarity[query_index, hardest_index]
        )

        u8_query_all, u8_image_gradient = _autograd_pair(
            u8_loss, queries, images
        )
        real_query_all, real_image_gradient = _autograd_pair(
            real_loss, queries, images
        )
        native_query_all, native_image_gradient = _autograd_pair(
            native_loss, queries, images
        )
        margin_query_all, margin_image_gradient = _autograd_pair(
            margin, queries, images
        )

        u8_query_gradient = u8_query_all[query_index]
        real_query_gradient = real_query_all[query_index]
        native_query_gradient = native_query_all[query_index]
        margin_query_gradient = margin_query_all[query_index]
        u8_joint = _joint(u8_query_gradient, u8_image_gradient)
        real_joint = _joint(real_query_gradient, real_image_gradient)
        native_joint = _joint(native_query_gradient, native_image_gradient)
        margin_joint = _joint(margin_query_gradient, margin_image_gradient)

        u8_footprint = gradient_footprint(u8_image_gradient)
        real_footprint = gradient_footprint(real_image_gradient)
        support_indices = support_mask[query_index].nonzero(as_tuple=False).flatten()
        positive_similarity = raw_similarity[query_index, query_index]
        hardest_similarity = raw_similarity[query_index, hardest_index]
        synthetic_similarity = (queries[query_index] * u8_synthetic[query_index]).sum()

        row = {
            "query_index_within_batch": query_index,
            "positive_similarity": _scalar(positive_similarity),
            "hardest_real_index": hardest_index,
            "hardest_real_similarity": _scalar(hardest_similarity),
            "u8_support_indices": json.dumps(support_indices.detach().cpu().tolist()),
            "u8_synthetic_similarity": _scalar(synthetic_similarity),
            "u8_minus_hardest_real": _scalar(
                synthetic_similarity - hardest_similarity
            ),
            "u8_minus_positive": _scalar(synthetic_similarity - positive_similarity),
            "query_gradient_cosine_u8_real": _scalar(
                cosine_alignment(u8_query_gradient, real_query_gradient)
            ),
            "query_gradient_norm_u8": _scalar(u8_query_gradient.norm()),
            "query_gradient_norm_real": _scalar(real_query_gradient.norm()),
            "query_gradient_norm_ratio_u8_real": _scalar(
                u8_query_gradient.norm() / real_query_gradient.norm().clamp_min(1e-12)
            ),
            "image_gradient_cosine_u8_real": _scalar(
                cosine_alignment(u8_image_gradient, real_image_gradient)
            ),
            "image_gradient_norm_u8": _scalar(u8_image_gradient.norm()),
            "image_gradient_norm_real": _scalar(real_image_gradient.norm()),
            "image_gradient_norm_ratio_u8_real": _scalar(
                u8_image_gradient.norm() / real_image_gradient.norm().clamp_min(1e-12)
            ),
            "image_gradient_effective_support_u8": _scalar(
                u8_footprint["effective_support"]
            ),
            "image_gradient_effective_support_real": _scalar(
                real_footprint["effective_support"]
            ),
            "image_gradient_entropy_u8": _scalar(u8_footprint["entropy"]),
            "image_gradient_entropy_real": _scalar(real_footprint["entropy"]),
            "image_gradient_normalized_entropy_u8": _scalar(
                u8_footprint["normalized_entropy"]
            ),
            "image_gradient_normalized_entropy_real": _scalar(
                real_footprint["normalized_entropy"]
            ),
            "image_gradient_largest_share_u8": _scalar(u8_footprint["largest_share"]),
            "image_gradient_largest_share_real": _scalar(real_footprint["largest_share"]),
            "image_gradient_top2_share_u8": _scalar(u8_footprint["top2_share"]),
            "image_gradient_top2_share_real": _scalar(real_footprint["top2_share"]),
            "image_gradient_top4_share_u8": _scalar(u8_footprint["top4_share"]),
            "image_gradient_top4_share_real": _scalar(real_footprint["top4_share"]),
            "image_gradient_top8_share_u8": _scalar(u8_footprint["top8_share"]),
            "image_gradient_top8_share_real": _scalar(real_footprint["top8_share"]),
            "u8_gradient_fraction_on_u8_support": _scalar(
                gradient_share_on_indices(u8_image_gradient, support_indices)
            ),
            "real_gradient_fraction_on_u8_support": _scalar(
                gradient_share_on_indices(real_image_gradient, support_indices)
            ),
            "u8_gradient_fraction_on_hardest_real": _scalar(
                gradient_share_on_indices(u8_image_gradient, [hardest_index])
            ),
            "real_gradient_fraction_on_hardest_real": _scalar(
                gradient_share_on_indices(real_image_gradient, [hardest_index])
            ),
            "joint_gradient_cosine_u8_real": _scalar(
                cosine_alignment(u8_joint, real_joint)
            ),
            "joint_gradient_norm_u8": _scalar(u8_joint.norm()),
            "joint_gradient_norm_real": _scalar(real_joint.norm()),
            "joint_gradient_norm_ratio_u8_real": _scalar(
                u8_joint.norm() / real_joint.norm().clamp_min(1e-12)
            ),
            "native_alignment_u8": _scalar(cosine_alignment(u8_joint, native_joint)),
            "native_alignment_hardest_real": _scalar(
                cosine_alignment(real_joint, native_joint)
            ),
            "margin_directional_change_u8": _scalar(
                margin_directional_change(margin_joint, u8_joint)
            ),
            "margin_directional_change_hardest_real": _scalar(
                margin_directional_change(margin_joint, real_joint)
            ),
            "u8_relative_loss": _scalar(u8_loss),
            "hardest_real_relative_loss": _scalar(real_loss),
        }
        row["u8_minus_hardest_real_native_alignment"] = (
            row["native_alignment_u8"] - row["native_alignment_hardest_real"]
        )
        row["u8_minus_real_margin_directional_change"] = (
            row["margin_directional_change_u8"]
            - row["margin_directional_change_hardest_real"]
        )
        if species_ids is not None:
            anchor_species = species_ids[query_index]
            support_species = [species_ids[index] for index in support_indices.tolist()]
            row.update(
                {
                    "species_id": anchor_species,
                    "u8_support_same_species_fraction": sum(
                        value == anchor_species for value in support_species
                    )
                    / len(support_species),
                    "u8_support_unique_species_count": len(set(support_species)),
                    "hardest_real_same_species": (
                        species_ids[hardest_index] == anchor_species
                    ),
                }
            )
        rows.append(row)
    return {
        "rows": rows,
        "support_mask": support_mask.detach(),
        "uniform_weights": uniform_weights.detach(),
        "u8_synthetic": u8_synthetic.detach(),
        "model_free_local_leaves": {"queries": queries, "images": images},
    }


def _values(rows, field):
    return [row[field] for row in rows]


def summarize_rows(rows):
    """Aggregate the complete pre-registered diagnostic report sections."""
    u8_hardness = _values(rows, "u8_minus_hardest_real")
    u8_positive = _values(rows, "u8_minus_positive")
    query_alignment = _values(rows, "query_gradient_cosine_u8_real")
    image_alignment = _values(rows, "image_gradient_cosine_u8_real")
    joint_alignment = _values(rows, "joint_gradient_cosine_u8_real")
    native_u8 = _values(rows, "native_alignment_u8")
    native_real = _values(rows, "native_alignment_hardest_real")
    margin_u8 = _values(rows, "margin_directional_change_u8")
    margin_real = _values(rows, "margin_directional_change_hardest_real")

    correlations = {}
    correlation_fields = {
        "query_gradient_alignment": query_alignment,
        "image_gradient_alignment": image_alignment,
        "joint_gradient_alignment": joint_alignment,
        "u8_native_alignment": native_u8,
        "u8_margin_directional_change": margin_u8,
    }
    for name, values in correlation_fields.items():
        correlations[f"u8_hardness_vs_{name}"] = correlation_pair(
            u8_hardness, values
        )
    if "u8_support_unique_species_count" in rows[0]:
        diversity = _values(rows, "u8_support_unique_species_count")
        for name, values in correlation_fields.items():
            correlations[f"support_species_diversity_vs_{name}"] = correlation_pair(
                diversity, values
            )

    native_delta = np.asarray(native_u8) - np.asarray(native_real)
    margin_delta = np.asarray(margin_u8) - np.asarray(margin_real)
    report = {
        "hardness_summary": {
            "positive_similarity": distribution_summary(
                _values(rows, "positive_similarity")
            ),
            "hardest_real_similarity": distribution_summary(
                _values(rows, "hardest_real_similarity")
            ),
            "uniform_top8_synthetic_similarity": distribution_summary(
                _values(rows, "u8_synthetic_similarity")
            ),
            "uniform_top8_minus_hardest_real": distribution_summary(u8_hardness),
            "fraction_uniform_top8_gt_hardest_real": float(
                (np.asarray(u8_hardness) > 0).mean()
            ),
            "uniform_top8_minus_positive": distribution_summary(u8_positive),
            "fraction_uniform_top8_gt_positive": float(
                (np.asarray(u8_positive) > 0).mean()
            ),
        },
        "query_gradient_alignment": {
            "cosine_u8_vs_hardest_real": alignment_summary(query_alignment),
            "norm_ratio_u8_over_hardest_real": distribution_summary(
                _values(rows, "query_gradient_norm_ratio_u8_real")
            ),
        },
        "image_gradient_alignment": {
            "cosine_u8_vs_hardest_real": alignment_summary(image_alignment),
            "norm_ratio_u8_over_hardest_real": distribution_summary(
                _values(rows, "image_gradient_norm_ratio_u8_real")
            ),
        },
        "joint_gradient_alignment": {
            "cosine_u8_vs_hardest_real": alignment_summary(joint_alignment),
            "norm_ratio_u8_over_hardest_real": distribution_summary(
                _values(rows, "joint_gradient_norm_ratio_u8_real")
            ),
        },
        "gradient_footprint": {
            field: distribution_summary(_values(rows, field))
            for field in (
                "image_gradient_effective_support_u8",
                "image_gradient_effective_support_real",
                "image_gradient_entropy_u8",
                "image_gradient_entropy_real",
                "image_gradient_normalized_entropy_u8",
                "image_gradient_normalized_entropy_real",
                "image_gradient_largest_share_u8",
                "image_gradient_largest_share_real",
                "image_gradient_top2_share_u8",
                "image_gradient_top2_share_real",
                "image_gradient_top4_share_u8",
                "image_gradient_top4_share_real",
                "image_gradient_top8_share_u8",
                "image_gradient_top8_share_real",
                "u8_gradient_fraction_on_u8_support",
                "real_gradient_fraction_on_u8_support",
                "u8_gradient_fraction_on_hardest_real",
                "real_gradient_fraction_on_hardest_real",
            )
        },
        "native_objective_alignment": {
            "u8": alignment_summary(native_u8),
            "hardest_real": alignment_summary(native_real),
            "u8_minus_hardest_real": distribution_summary(native_delta),
            "fraction_u8_gt_hardest_real": float((native_delta > 0).mean()),
            "fraction_hardest_real_gt_u8": float((native_delta < 0).mean()),
        },
        "margin_directional_change": {
            "u8": distribution_summary(margin_u8),
            "hardest_real": distribution_summary(margin_real),
            "u8_minus_hardest_real": distribution_summary(margin_delta),
            "fraction_u8_better": float((margin_delta > 0).mean()),
            "fraction_hardest_real_better": float((margin_delta < 0).mean()),
        },
        "relative_objective": {
            "u8": distribution_summary(_values(rows, "u8_relative_loss")),
            "hardest_real": distribution_summary(
                _values(rows, "hardest_real_relative_loss")
            ),
        },
        "correlations": correlations,
    }
    if "u8_support_unique_species_count" in rows[0]:
        report["species_observational_diagnostics"] = {
            "labels_used_for_selection_or_loss": False,
            "u8_support_same_species_fraction": distribution_summary(
                _values(rows, "u8_support_same_species_fraction")
            ),
            "u8_support_unique_species_count": distribution_summary(
                _values(rows, "u8_support_unique_species_count")
            ),
            "fraction_hardest_real_same_species": float(
                np.asarray(_values(rows, "hardest_real_same_species"), dtype=float).mean()
            ),
        }
    return report


def write_outputs(output_dir, report, rows, config, holdout_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    csv_path = output_dir / CSV_FILENAME
    config_path = output_dir / "resolved_config.yaml"
    holdout_copy = output_dir / "diagnostic_holdout_indices.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    shutil.copyfile(holdout_path, holdout_copy)
    return [report_path, csv_path, config_path, holdout_copy]


def run(config):
    """Encode the fixed holdout and compute local embedding gradients only."""
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
    grouped = train_grouped if config["sampling"]["source_split"] == "train" else val_grouped
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
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise AssertionError("Frozen diagnostic found a trainable model parameter")
    image_cpu, text_cpu, metadata = encode_dataset(model, processor, loader, device)
    logit_scale = model.get_logit_scale().detach()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Encoding populated a frozen model parameter gradient")

    batch_size = config["diagnostic"]["batch_size"]
    all_rows = []
    for batch_index, start in enumerate(range(0, len(dataset), batch_size)):
        stop = min(start + batch_size, len(dataset))
        if stop - start != batch_size:
            raise ValueError("The exact holdout must divide into complete B64 batches")
        batch_metadata = metadata[start:stop]
        result = compute_batch_gradient_geometry(
            image_cpu[start:stop].to(device),
            text_cpu[start:stop].to(device),
            logit_scale,
            species_ids=[record["species_id"] for record in batch_metadata],
            top_k=config["diagnostic"]["top_k"],
        )
        for local_index, row in enumerate(result["rows"]):
            record = batch_metadata[local_index]
            row.update(
                {
                    "query_index": start + local_index,
                    "batch_index": batch_index,
                    "holdout_pool_index": start + local_index,
                    "source_index": record["source_index"],
                    "image_key": record["image_key"],
                }
            )
            all_rows.append(row)
        del result
        print(f"Completed gradient batch {batch_index + 1}/{len(dataset) // batch_size}", flush=True)

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Embedding autograd populated model parameter gradients")
    summaries = summarize_rows(all_rows)
    holdout_bytes = Path(holdout_path).read_bytes()
    report = {
        "experiment": config["experiment"],
        "model": {
            "checkpoint": checkpoint,
            "frozen": True,
            "all_parameter_grads_none": True,
            "logit_scale": _scalar(logit_scale),
        },
        "no_optimizer_no_parameter_updates_no_training": True,
        "runtime": device_info,
        "dataset": {
            **split_info,
            "source_split": config["sampling"]["source_split"],
            "sample_count": len(dataset),
            "caption_view": config["sampling"]["caption_view"],
            "holdout_path": str(Path(holdout_path).relative_to(ROOT_DIR)),
            "holdout_sha256": hashlib.sha256(holdout_bytes).hexdigest(),
            "loaded_holdout_sha256": holdout_sha256,
        },
        "batching": {
            "batch_size": batch_size,
            "batch_count": len(dataset) // batch_size,
            "shuffle": False,
            "semantics": (
                "Sequential non-overlapping chunks of the exact committed 1024-example "
                "holdout; each example is evaluated exactly once in batch-local B64 geometry."
            ),
        },
        "candidate_construction": {
            "uniform_u8": "normalize((1/8) * sum_{j in positive-excluded raw-cosine top-8} v_j)",
            "hardest_real": "argmax_{j != i} q_i dot v_j",
            "support_helper": "src.clip_geometry_v2_metrics._top_k_support",
            "uniform_weight_helper": "model.clip_training.synthetic_barycentric_weights",
            "hardest_real_helper": "model.clip_training.hardest_real_negative_indices",
            "selection_stop_gradient": True,
            "embedding_candidate_paths_live": True,
        },
        "objectives": {
            "relative_helper": "model.clip_training.clip_relative_denominator_loss",
            "base_logits": "L = (q @ V.T) * detach(CLIP logit scale)",
            "u8": "D_i = log(sum_j exp(L_ij) + exp(l_i^U8)) - logsumexp(L_i)",
            "hardest_real": "D_i = log(sum_j exp(L_ij) + exp(L_i,j*)) - logsumexp(L_i)",
            "native_row": "C_i = -L_ii + logsumexp(L_i)",
            "alpha_or_schedule": None,
        },
        "gradient_definitions": {
            "query": "cos(grad_q_i D_U8, grad_q_i D_real)",
            "image": "cos(vec(grad_V D_U8), vec(grad_V D_real))",
            "joint": "cos([grad_q_i; vec(grad_V)]_U8, [grad_q_i; vec(grad_V)]_real)",
            "effective_image_support": "1 / sum_j p_j^2, p_j=||grad_vj||/sum_k||grad_vk||",
            "gradient_entropy": "-sum_j p_j log(p_j)",
            "native_alignment": "cos(unit auxiliary joint gradient, native row-CE joint gradient)",
            "margin_directional_change": "-grad(s_pos-s_hardest) dot unit(auxiliary joint gradient)",
        },
        **summaries,
    }
    output_dir = _resolve_repo_path(config["output"]["directory"])
    paths = write_outputs(output_dir, report, all_rows, config, holdout_path)
    print("Frozen CLIP negative-gradient geometry diagnostic complete")
    print(
        "  joint gradient cosine mean: "
        f"{report['joint_gradient_alignment']['cosine_u8_vs_hardest_real']['mean']:.6f}"
    )
    print(
        "  native alignment mean (U8 / real): "
        f"{report['native_objective_alignment']['u8']['mean']:.6f} / "
        f"{report['native_objective_alignment']['hardest_real']['mean']:.6f}"
    )
    for path in paths:
        print(f"  wrote: {path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hf_cub200_clip_negative_gradient_geometry.yaml",
    )
    parser.add_argument("--output-directory", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resolved = load_config(arguments.config)
    if arguments.output_directory:
        resolved["output"]["directory"] = arguments.output_directory
    run(resolved)
