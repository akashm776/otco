"""Controlled native-CLIP baseline versus fresh batch-local OTCO training."""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from model.clip_backend import CLIPEncoderBackend
from model.clip_training import (
    CLIPTrainingObjective,
    build_clip_optimizer,
    configure_clip_trainable_parameters,
    parameter_gradient_norm,
    tensor_gradient_norm,
)
from src.clip_training_data import ROOT_DIR, build_clip_training_data
from src.clip_training_eval import evaluate_clip


def load_training_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["experiment"]["mode"] != "controlled_clip_finetuning":
        raise ValueError("CLIP trainer requires controlled_clip_finetuning mode")
    if config["model"]["backend"] != "clip":
        raise ValueError("CLIP trainer requires model.backend=clip")
    if config["loss"]["base"] != "native_clip_symmetric_contrastive":
        raise ValueError("First CLIP experiment requires the native CLIP loss")
    if config["model"]["checkpoint"] != "openai/clip-vit-base-patch32":
        raise ValueError("First CLIP experiment requires openai/clip-vit-base-patch32")
    if config["model"]["trainable_policy"] != (
        "projections_last_blocks_logit_scale"
    ):
        raise ValueError("Unexpected CLIP trainable-parameter policy")
    if config["dataset"]["caption_sampling"] != "epoch_random":
        raise ValueError("Both controlled arms require epoch_random captions")
    if config["evaluation"]["every_epochs"] != 1:
        raise ValueError("The first controlled pair evaluates every epoch")
    if config["checkpoint"]["resume"]:
        raise ValueError("The first controlled pair must start at epoch zero")
    if config["experiment"]["seed"] != config["training"]["seed"]:
        raise ValueError("Experiment and training seeds must match")
    ot = config["ot"]
    required = {
        "cost_space": "raw_cosine",
        "solver": "historical_sparse_ot",
        "update_freq": 1,
        "top_k": 32,
        "ot_eps": 0.049,
        "sinkhorn_iters": 30,
        "entropy_gate_enabled": False,
    }
    for key, expected in required.items():
        if ot[key] != expected:
            raise ValueError(f"First CLIP experiment requires ot.{key}={expected!r}")
    expected_arm = {
        "cub200_clip_vit_b32_baseline": False,
        "cub200_clip_vit_b32_otco_rawcos_gap_gate": True,
        "cub200_clip_vit_b32_otco_relative_denominator": True,
    }
    name = config["experiment"]["name"]
    if name not in expected_arm or bool(ot["enabled"]) != expected_arm[name]:
        raise ValueError("Experiment name and OT treatment flag do not match")
    expected_loss_type = {
        "cub200_clip_vit_b32_baseline": "historical_absolute_sigmoid",
        "cub200_clip_vit_b32_otco_rawcos_gap_gate": (
            "historical_absolute_sigmoid"
        ),
        "cub200_clip_vit_b32_otco_relative_denominator": (
            "clip_relative_denominator"
        ),
    }
    if ot["loss_type"] != expected_loss_type[name]:
        raise ValueError("Experiment name and OT loss type do not match")
    expected_absolute_gate = ot["loss_type"] == "historical_absolute_sigmoid"
    if bool(ot["synthetic_logit_gate_enabled"]) != expected_absolute_gate:
        raise ValueError(
            "Absolute synthetic-logit gating must be enabled only for the "
            "historical loss mode"
        )
    if config["training"]["mixed_precision"] not in {"bf16", "none"}:
        raise ValueError("The first A100 experiment supports bf16 or none")
    return config


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(runtime):
    if runtime["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = {"device": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        total_mib = properties.total_memory / (1024**2)
        if total_mib < runtime["min_vram_mib"]:
            raise RuntimeError(
                f"GPU has {total_mib:.0f} MiB; {runtime['min_vram_mib']} required"
            )
        info.update({"gpu": properties.name, "total_vram_mib": total_mib})
    return device, info


def _autocast(device, mixed_precision):
    if device.type != "cuda" or mixed_precision == "none":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _text_batch(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("input_ids", "attention_mask")
        if key in batch
    }


def build_scheduler(optimizer, *, warmup_steps, total_steps):
    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class EpochMetricAccumulator:
    def __init__(self):
        self.values = defaultdict(list)
        self.gates = Counter()

    def add(self, metrics):
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in {
                "gate_state_id",
                "plan_build_count",
            }:
                self.values[key].append(float(value))
        if "gate_state" in metrics:
            self.gates[metrics["gate_state"]] += 1

    def summarize(self):
        result = {
            key: float(sum(values) / len(values))
            for key, values in self.values.items()
            if values
        }
        for key, values in self.values.items():
            if key.startswith("projection_gradient_norm_"):
                result[f"{key}_sample_count"] = len(values)
        total = sum(self.gates.values())
        if total:
            result["gap_gate_state_fractions"] = {
                state: count / total for state, count in sorted(self.gates.items())
            }
        return result


def _projection_parameters(model):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ("visual_projection" in name or "text_projection" in name)
    ]


def diagnostic_gradient_norms(loss_output, projection_parameters):
    base_gradients = torch.autograd.grad(
        loss_output.clip_loss,
        projection_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    result = {"projection_gradient_norm_clip": tensor_gradient_norm(base_gradients)}
    if loss_output.weighted_ot_loss is not None:
        ot_gradients = torch.autograd.grad(
            loss_output.weighted_ot_loss,
            projection_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        result["projection_gradient_norm_weighted_ot"] = tensor_gradient_norm(
            ot_gradients
        )
    return result


def train_epoch(
    model,
    objective,
    data,
    optimizer,
    scheduler,
    device,
    config,
    *,
    epoch,
    global_step,
    remaining_gradient_diagnostics,
):
    model.train()
    data.train_dataset.set_epoch(epoch)
    accumulator = EpochMetricAccumulator()
    projection_parameters = _projection_parameters(model)
    for batch in data.train_loader:
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config["training"]["mixed_precision"]):
            output = model(
                batch["pixel_values"].to(device, non_blocking=True),
                _text_batch(batch, device),
            )
            losses = objective(
                output,
                species_ids=batch["species_ids"].to(device),
                step=global_step,
            )
        metrics = dict(losses.metrics)
        if (
            remaining_gradient_diagnostics > 0
            and losses.weighted_ot_loss is not None
            and metrics["alpha_effective"] > 0
        ):
            metrics.update(diagnostic_gradient_norms(losses, projection_parameters))
            remaining_gradient_diagnostics -= 1
        losses.total_loss.backward()
        metrics["total_trainable_gradient_norm"] = parameter_gradient_norm(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config["optimizer"]["max_gradient_norm"],
        )
        optimizer.step()
        # Standard CLIP safety clamp; it is applied identically in both arms.
        model.clip_model.logit_scale.data.clamp_(max=math.log(100.0))
        scheduler.step()
        metrics["learned_logit_scale"] = float(
            model.get_logit_scale().detach().float().item()
        )
        accumulator.add(metrics)
        global_step += 1
    return global_step, remaining_gradient_diagnostics, accumulator.summarize()


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, metrics):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def consider_species_checkpoint(record, best_score=float("-inf"), best_epoch=None):
    """Return updated best-state metadata, including a valid epoch-zero win."""
    score = record["evaluation"]["species"]["top_1_accuracy"]
    if score > best_score:
        return score, record["epoch"], True
    return best_score, best_epoch, False


def run(config, *, output_directory=None, checkpoint_directory=None):
    from transformers import AutoProcessor

    seed = config["training"]["seed"]
    set_global_seed(seed)
    device, device_info = resolve_device(config["runtime"])
    checkpoint = config["model"]["checkpoint"]
    processor = AutoProcessor.from_pretrained(checkpoint)
    data = build_clip_training_data(
        config, processor, pin_memory=device.type == "cuda"
    )
    model = CLIPEncoderBackend.from_pretrained(checkpoint).to(device)
    inventory = configure_clip_trainable_parameters(
        model, config["model"]["trainable_policy"]
    )
    objective = CLIPTrainingObjective(config["ot"]).to(device)
    optimizer = build_clip_optimizer(model, config["optimizer"])
    total_steps = len(data.train_loader) * config["training"]["epochs"]
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=config["scheduler"]["warmup_steps"],
        total_steps=total_steps,
    )

    output_dir = Path(
        output_directory
        or ROOT_DIR / config["output"]["root"] / config["experiment"]["name"]
    )
    checkpoint_dir = Path(
        checkpoint_directory
        or ROOT_DIR
        / config["checkpoint"]["root"]
        / config["experiment"]["name"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_json(output_dir / "trainable_parameters.json", inventory)
    write_json(output_dir / "data_exclusion_report.json", data.exclusion_report)

    print(json.dumps({"device": device_info}, indent=2))
    print(json.dumps({"data_exclusion": data.exclusion_report}, indent=2))
    print(json.dumps({"trainable_parameters": inventory}, indent=2))
    print(f"Training batches per epoch: {len(data.train_loader)}")
    print(f"Total optimizer steps: {total_steps}")

    metrics_path = output_dir / "metrics.jsonl"
    history = []
    evaluation_options = {
        "mixed_precision": config["training"]["mixed_precision"],
        "prompt_template": config["evaluation"]["species_prompt"],
        "retrieval_chunk_size": config["evaluation"]["retrieval_chunk_size"],
    }
    epoch_zero = {
        "epoch": 0,
        "global_step": 0,
        "training": None,
        "evaluation": evaluate_clip(
            model, processor, data, device, **evaluation_options
        ),
    }
    history.append(epoch_zero)
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(epoch_zero) + "\n")
    print(json.dumps(epoch_zero, indent=2))

    best_species_top1, best_epoch, epoch_zero_is_best = (
        consider_species_checkpoint(epoch_zero)
    )
    if not epoch_zero_is_best:
        raise AssertionError("Epoch zero must initialize the best checkpoint")
    save_checkpoint(
        checkpoint_dir / "best_model.pt",
        model,
        optimizer,
        scheduler,
        config,
        0,
        epoch_zero,
    )
    global_step = 0
    remaining_gradient_diagnostics = config["diagnostics"][
        "separate_projection_gradient_steps"
    ]
    for epoch in range(1, config["training"]["epochs"] + 1):
        global_step, remaining_gradient_diagnostics, training_metrics = train_epoch(
            model,
            objective,
            data,
            optimizer,
            scheduler,
            device,
            config,
            epoch=epoch,
            global_step=global_step,
            remaining_gradient_diagnostics=remaining_gradient_diagnostics,
        )
        evaluation = evaluate_clip(
            model, processor, data, device, **evaluation_options
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "training": training_metrics,
            "evaluation": evaluation,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            config,
            epoch,
            record,
        )
        best_species_top1, best_epoch, is_new_best = consider_species_checkpoint(
            record, best_species_top1, best_epoch
        )
        if is_new_best:
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                record,
            )

    summary = {
        "experiment_name": config["experiment"]["name"],
        "best_epoch_by_species_top_1": best_epoch,
        "best_species_top_1_accuracy": best_species_top1,
        "final": history[-1],
        "epoch_zero": epoch_zero,
        "device": device_info,
        "data_exclusion": data.exclusion_report,
        "trainable_parameters": inventory,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    parser.add_argument("--checkpoint-directory")
    args = parser.parse_args()
    config = load_training_config(args.config)
    run(
        config,
        output_directory=args.output_directory,
        checkpoint_directory=args.checkpoint_directory,
    )


if __name__ == "__main__":
    main()
