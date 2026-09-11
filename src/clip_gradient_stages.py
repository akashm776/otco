"""Observe fixed held-out tangent gradients during the existing CLIP trainer.

This is a measurement experiment, not an adaptive curriculum. Candidate support
is recomputed at each state; examples, captions and batch membership stay fixed.
"""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from model.clip_training import scheduled_ot_alpha
from src import clip_train
from src.clip_barycentric_weight_ablation import load_committed_holdout
from src.clip_geometry_diagnostic import (
    CUBCLIPDiagnosticDataset, encode_dataset, keep_records_as_list,
)
from src.clip_negative_gradient_geometry_randomized import (
    EXPECTED_PARTITIONS, build_partition_conditions, run_feature_partition,
    _partition_summary,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "joint_cosine": "joint_gradient_cosine_u8_real",
    "native_alignment_u8": "native_alignment_u8",
    "native_alignment_real": "native_alignment_hardest_real",
    "margin_change_u8": "margin_directional_change_u8",
    "margin_change_real": "margin_directional_change_hardest_real",
}


def stage_steps(warmup, ramp, total):
    """Keys count completed updates; the objective sees zero-based step indices."""
    first_active = warmup + (1 if ramp > 0 else 0)
    after_full = warmup + max(ramp, 0) + 1
    if not 0 < first_active < after_full < total:
        raise ValueError("Stages must be distinct and strictly inside training")
    return {0: "pretrained", first_active: "before_auxiliary",
            after_full: "after_ramp", total: "late"}


@contextmanager
def observational_state(model):
    """Preserve global RNG streams and every module's train/eval flag.

    The observer never uses the training DataLoader or optimizer and computes
    gradients only on detached feature leaves, leaving parameter .grad intact.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def compact_metrics(rows):
    total_count = len(rows)
    rows = [r for r in rows if r.get("gradient_metrics_valid", True)]
    if not rows:
        return {**dict.fromkeys(list(METRICS) + ["fraction_negative_joint_cosine",
                "fraction_positive_u8_margin", "fraction_u8_jointly_useful"]),
                "valid_query_count": 0, "undefined_query_count": total_count}
    result = {name: float(np.mean([row[key] for row in rows]))
              for name, key in METRICS.items()}
    result["fraction_negative_joint_cosine"] = float(np.mean([
        row["joint_gradient_cosine_u8_real"] < 0 for row in rows]))
    result["fraction_positive_u8_margin"] = float(np.mean([
        row["margin_directional_change_u8"] > 0 for row in rows]))
    result["fraction_u8_jointly_useful"] = float(np.mean([
        row["margin_directional_change_u8"] > 0 and row["native_alignment_u8"] > 0
        for row in rows]))
    result.update(valid_query_count=len(rows), undefined_query_count=total_count - len(rows))
    return result


def paired_transitions(previous, current):
    """Paired descriptive changes, not independent replicates or causal effects."""
    key = lambda row: (row["partition_name"], row["query_index"])
    old, new = {key(r): r for r in previous}, {key(r): r for r in current}
    if old.keys() != new.keys() or len(old) != len(previous) or len(new) != len(current):
        raise ValueError("Stage rows must have identical unique partition/query keys")
    pairs = [(old[k], new[k]) for k in old if k[0] != "sequential"]
    if not pairs:
        raise ValueError("Missing shuffled partitions")
    for a, b in pairs:
        if (a["source_index"], a["image_key"]) != (b["source_index"], b["image_key"]):
            raise ValueError("Query identity changed across stages")
    total_count = len(pairs)
    pairs = [(a, b) for a, b in pairs if a.get("gradient_metrics_valid", True)
             and b.get("gradient_metrics_valid", True)]
    mean = lambda values: float(np.mean(values)) if values else None
    return {
        "query_partition_observations": total_count,
        "paired_valid_observations": len(pairs),
        "fraction_u8_margin_nonpositive_to_positive": mean([
            a["margin_directional_change_u8"] <= 0 < b["margin_directional_change_u8"]
            for a, b in pairs]),
        "fraction_u8_margin_positive_to_nonpositive": mean([
            a["margin_directional_change_u8"] > 0 >= b["margin_directional_change_u8"]
            for a, b in pairs]),
        "mean_native_alignment_u8_change": mean([
            b["native_alignment_u8"] - a["native_alignment_u8"] for a, b in pairs]),
    }


class GradientStageObserver:
    def __init__(self, output_dir, diagnostic, arm):
        self.output_dir = Path(output_dir)
        self.diagnostic = diagnostic
        self.arm = arm
        self.reports = []
        self.previous_rows = None
        self.metadata_hash = None

    def initialize(self, *, model, processor, data, device, config, total_steps):
        d = self.diagnostic
        if total_steps != d["expected_total_steps"]:
            raise ValueError(f"Expected {d['expected_total_steps']} training steps, got {total_steps}")
        self.stages = stage_steps(d["reference_warmup_steps"], d["reference_ramp_steps"], total_steps)
        for field in ("warmup_steps", "ramp_steps"):
            if config["ot"][field] != d[f"reference_{field}"]:
                raise ValueError("Arm schedule differs from the reference measurement schedule")
        self.config, self.processor, self.device = config, processor, device
        self.holdout, self.holdout_hash, _ = load_committed_holdout(
            config["dataset"]["diagnostic_holdout_indices"])
        if len(self.holdout) != d["holdout_size"]:
            raise ValueError("Unexpected diagnostic holdout size")
        train = data.train_dataset
        if set(train.source_indices) & set(self.holdout):
            raise ValueError("Diagnostic examples leaked into training")
        self.loader = DataLoader(
            CUBCLIPDiagnosticDataset(train.grouped_split, train.species_ids, self.holdout),
            batch_size=d["batch_size"], shuffle=False, drop_last=False,
            num_workers=d["encoding_num_workers"], collate_fn=keep_records_as_list,
            generator=torch.Generator().manual_seed(20260910),
        )
        self.conditions, self.partitions = build_partition_conditions(
            EXPECTED_PARTITIONS, len(self.holdout), d["batch_size"])
        self.output_dir.mkdir(parents=True, exist_ok=False)
        clip_train.write_json(self.output_dir / "batch_partitions.json", self.partitions)
        clip_train.write_json(self.output_dir / "diagnostic_holdout_indices.json", self.holdout)
        with (self.output_dir / "diagnostic_config.yaml").open("w") as handle:
            yaml.safe_dump(d, handle)

    def __call__(self, *, model, epoch, global_step):
        if global_step not in self.stages:
            return
        if any(r["completed_updates"] == global_step for r in self.reports):
            raise ValueError("Repeated diagnostic stage")
        with observational_state(model), torch.enable_grad():
            self.capture(model, epoch, global_step)

    def capture(self, model, epoch, global_step):
        stage = self.stages[global_step]
        print(f"[gradient stages] {self.arm}: {stage} at {global_step} completed updates", flush=True)
        image_features, text_features, metadata = encode_dataset(
            model, self.processor, self.loader, self.device)
        # Clone outside inference_mode: these features feed fresh autograd graphs.
        image_features, text_features = image_features.clone(), text_features.clone()
        metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if self.metadata_hash is not None and metadata_hash != self.metadata_hash:
            raise AssertionError("Diagnostic images, captions or ordering changed")
        self.metadata_hash = metadata_hash
        scale = float(model.get_logit_scale().detach().float().item())
        feature_hash = hashlib.sha256(image_features.numpy().tobytes()
                                     + text_features.numpy().tobytes()
                                     + json.dumps(scale).encode()).hexdigest()
        stage_dir = self.output_dir / f"{global_step:06d}_{stage}"
        stage_dir.mkdir()
        clip_train.write_json(stage_dir / "metadata.json", metadata)
        if self.diagnostic["save_features"]:
            torch.save({"image_features": image_features, "text_features": text_features,
                        "logit_scale": scale}, stage_dir / "features.pt")
        rows, summaries, compact = [], {}, {}
        images, texts = image_features.to(self.device), text_features.to(self.device)
        for name, condition in self.conditions.items():
            partition_rows, audit = run_feature_partition(
                condition=condition, image_features=images, text_features=texts,
                metadata=metadata, logit_scale=scale, top_k=self.diagnostic["top_k"],
                tangent_tolerance=self.diagnostic["tangent_tolerance"], record_undefined=True)
            rows.extend(partition_rows)
            valid = [r for r in partition_rows if r["gradient_metrics_valid"]]
            summaries[name] = {
                "total_query_count": len(partition_rows), "valid_query_count": len(valid),
                "undefined_query_count": len(partition_rows) - len(valid),
                "tangent_gradient_audit": audit,
                "valid_queries_only": _partition_summary(valid, audit) if len(valid) >= 2 else None,
            }
            compact[name] = compact_metrics(partition_rows)
        ot = self.config["ot"]
        alpha = lambda step: (scheduled_ot_alpha(step, alpha_max=ot["alpha_max"],
            warmup_steps=ot["warmup_steps"], ramp_steps=ot["ramp_steps"])
            if ot["enabled"] else 0.0)
        report = {
            "arm": self.arm, "stage": stage, "completed_updates": global_step,
            "epoch": epoch, "logit_scale": scale,
            "previous_update_scheduled_alpha": alpha(global_step - 1) if global_step else None,
            "next_update_scheduled_alpha": alpha(global_step) if stage != "late" else None,
            "holdout_sha256": self.holdout_hash, "metadata_sha256": metadata_hash,
            "features_and_scale_sha256": feature_hash,
            "partition_sha256": self.partitions["flattened_sha256"],
            "semantics": {
                "gradients": "Detached embedding tangent gradients; not encoder/optimizer gradients",
                "native_reference": "Text-to-image row CE, not full symmetric training objective",
                "margin_reference": "Positive minus current hardest-real raw cosine",
                "candidates": "Current positive-excluded top-8 uniform barycenter and hardest real",
                "selection": "Stop-gradient; embedding paths through candidate construction live",
                "diagnostic_alpha": "Unweighted directions; schedule metadata is training-only",
                "scale": "Current learned CLIP scale, detached within the diagnostic",
                "encoding": "eval mode, fp32, canonical first captions",
                "jointly_useful": "Positive native row-CE alignment AND positive real-margin change",
                "inference_limit": "Descriptive local proxies; no downstream or causal benefit established",
                "undefined_gradients": "Retained with validity=false and norms; direction summaries exclude them. Report valid denominators; changing coverage can bias stage comparisons.",
            },
            "partitions": summaries, "compact_by_partition": compact,
        }
        if self.previous_rows is not None:
            report["transition_from_previous_stage"] = paired_transitions(self.previous_rows, rows)
        clip_train.write_json(stage_dir / "report.json", report)
        with (stage_dir / "per_query.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
        self.reports.append(report)
        self.previous_rows = rows
        clip_train.write_json(self.output_dir / "stage_summary.json", self.reports)
        print(f"[gradient stages] saved {stage_dir}", flush=True)

    def finish(self):
        if {r["completed_updates"] for r in self.reports} != set(self.stages):
            raise AssertionError("Training finished without every requested diagnostic stage")


def plot_results(output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("*/diagnostics/stage_summary.json"))
    if not files:
        raise ValueError("No stage summaries found")
    panels = [
        ("joint_cosine", "U8 vs real: joint gradient cosine"),
        ("native_alignment_u8", "U8 alignment with native row CE"),
        ("native_alignment_real", "Real alignment with native row CE"),
        ("margin_change_u8", "Real-margin change: U8 direction"),
        ("margin_change_real", "Real-margin change: real direction"),
        ("fraction_u8_jointly_useful", "U8: fraction passing both local proxies"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    records = []
    reference = None
    for path in files:
        reports = json.loads(path.read_text())
        identity = [(r["completed_updates"], r["holdout_sha256"], r["metadata_sha256"],
                     r["partition_sha256"]) for r in reports]
        if reference is not None and identity != reference:
            raise ValueError("Cannot compare arms with different stages/examples/partitions")
        reference = identity
        for ax, (metric, title) in zip(axes.flat, panels):
            values = np.asarray([[v[metric] for name, v in r["compact_by_partition"].items()
                                  if name != "sequential"] for r in reports], dtype=float)
            steps = [r["completed_updates"] for r in reports]
            line, = ax.plot(steps, values.mean(axis=1), marker="o", label=reports[0]["arm"])
            ax.fill_between(steps, values.min(axis=1), values.max(axis=1),
                            color=line.get_color(), alpha=.12)
            ax.axhline(0, color="gray", linewidth=.7)
            ax.set(title=title, xlabel="Completed optimizer updates")
        for r in reports:
            for name, values in r["compact_by_partition"].items():
                records.append({"arm": r["arm"], "stage": r["stage"],
                                "completed_updates": r["completed_updates"],
                                "partition": name, **values})
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed held-out batches across fine-tuning stages\n"
                 "Valid queries only (counts in CSV). 3-shuffle mean; min–max bands, NOT confidence intervals")
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"gradient_stage_trends.{suffix}", dpi=170)
    plt.close(fig)
    with (output_dir / "gradient_stage_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/clip_gradient_stages.yaml")
    parser.add_argument("--arm", choices=["baseline", "uniform_top8", "hardest_real", "pressure_matched"])
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--checkpoint-directory")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.output_directory)
        return
    if args.arm is None or args.checkpoint_directory is None:
        parser.error("Training requires --arm and --checkpoint-directory")
    protocol = yaml.safe_load((ROOT / args.config).read_text())
    config = clip_train.load_training_config(ROOT / protocol["arms"][args.arm])
    output = Path(args.output_directory) / args.arm
    checkpoint_output = Path(args.checkpoint_directory) / args.arm
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing arm: {output}")
    if checkpoint_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoints: {checkpoint_output}")
    observer = GradientStageObserver(output / "diagnostics", protocol["diagnostic"], args.arm)
    clip_train.run(config, output_directory=output / "training",
                   checkpoint_directory=checkpoint_output,
                   observer=observer)
    observer.finish()


if __name__ == "__main__":
    main()
