"""Rebuild the CLIP documentation figures from archived JSON; no training.

Run: uv run --with matplotlib==3.10.8 python scripts/plot_clip_results.py
"""

import io
import json
import os
from pathlib import Path
import tempfile

plot_cache = tempfile.mkdtemp(prefix="otco-matplotlib-")
os.environ.setdefault("MPLCONFIGDIR", plot_cache)
os.environ.setdefault("XDG_CACHE_HOME", plot_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiment_results/clip_2026-08-30_to_09-01"
OUTPUT = ROOT / "docs/figures"
INK = "#203047"
MUTED = "#536479"
COLORS = {
    "baseline": "#203047",
    "ot_absolute": "#b3535b",
    "ot_relative_a005": "#7599c9",
    "ot_relative_a05": "#2469ad",
    "uniform_top32": "#9167a6",
    "uniform_top8": "#cf7630",
    "uniform_top8_pressure_matched": "#287f77",
    "hardest_real": "#7261aa",
}
LABELS = {
    "baseline": "Native CLIP baseline",
    "ot_absolute": "OT absolute / α=0.05",
    "ot_relative_a005": "OT relative / α=0.05",
    "ot_relative_a05": "OT relative / α=0.5",
    "uniform_top32": "Uniform top-32 / α=0.5",
    "uniform_top8": "Uniform top-8 / α=0.5",
    "uniform_top8_pressure_matched": "Uniform top-8 / α=0.134",
    "hardest_real": "Hardest real / α=0.5",
}


def load_runs():
    runs = {}
    for name in LABELS:
        path = DATA / "training" / name / "metrics.jsonl"
        runs[name] = [json.loads(line) for line in path.read_text().splitlines()]
        if [row["epoch"] for row in runs[name]] != list(range(51)):
            raise ValueError(f"Expected epochs 0–50: {path}")
    return runs


def canonical_average(row):
    retrieval = row["evaluation"]["canonical_retrieval"]
    return (retrieval["text_to_image"]["r_at_1"] + retrieval["image_to_text"]["r_at_1"]) / 2


def decorate(ax, title, xlabel, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=14)
    ax.set_xlabel(xlabel, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5eaf0", linewidth=0.8)
    ax.set_axisbelow(True)


def schedule_bands(ax):
    # Epoch endpoints are 77 steps apart. Boundaries are approximate on an
    # epoch axis; actual measured values remain the unsmoothed epoch records.
    ax.axvspan(0, 1000 / 77, color="#e8edf3", alpha=0.65, zorder=0)
    ax.axvspan(1000 / 77, 2000 / 77, color="#f3eee2", alpha=0.55, zorder=0)
    ax.set_xlim(0, 50)
    ax.set_xticks(np.arange(0, 51, 10))


def save(fig, name, footer):
    fig.text(0.02, 0.015, footer, fontsize=9, color=MUTED, va="bottom")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=170, facecolor="white")
    svg = io.StringIO()
    fig.savefig(
        svg,
        format="svg",
        facecolor="white",
        metadata={"Date": None, "Creator": "OTCO scripts/plot_clip_results.py"},
    )
    # Matplotlib emits trailing spaces in multiline path attributes. Keep the
    # generated vector files clean for Git without changing their geometry.
    (OUTPUT / f"{name}.svg").write_text(
        "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    print(f"Generated {name}.png and {name}.svg")


def training_overview(runs):
    selected = ["baseline", "ot_relative_a05", "uniform_top8", "hardest_real"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.17, top=0.73, wspace=0.28)
    fig.suptitle("CLIP fine-tuning: retrieval rises, species recognition falls", x=0.02, ha="left", y=0.98, fontsize=17, weight="bold")
    for name in selected:
        rows = runs[name]
        epochs = [r["epoch"] for r in rows]
        kwargs = {"color": COLORS[name], "label": LABELS[name], "linewidth": 2.3 if name == "baseline" else 1.8}
        if name == "baseline":
            kwargs["linestyle"] = "--"
        axes[0].plot(epochs, [canonical_average(r) for r in rows], **kwargs)
        axes[1].plot(epochs, [100 * r["evaluation"]["species"]["top_1_accuracy"] for r in rows], **kwargs)
    decorate(axes[0], "Canonical retrieval", "Epoch", "Average R@1 (%)")
    decorate(axes[1], "Fixed-prompt species recognition", "Epoch", "Species top-1 (%)")
    axes[0].set_ylim(0, 2.2)
    axes[1].set_ylim(40, 53)
    for ax in axes:
        schedule_bands(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.91), ncol=2, frameon=False, columnspacing=3)
    save(fig, "clip_training_overview", "Selected arms · seed 42 · raw epoch measurements, no smoothing\nShading: auxiliary warmup (gray, steps 0–1000) and ramp (sand, steps 1000–2000); baseline has no auxiliary.")


def final_comparison(runs):
    names = list(LABELS)
    baseline = runs["baseline"][-1]
    reference = [canonical_average(baseline), baseline["evaluation"]["species"]["top_1_accuracy"] * 100]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2), sharey=True)
    fig.subplots_adjust(left=0.30, right=0.95, bottom=0.18, top=0.80, wspace=0.22)
    fig.suptitle("Final outcomes relative to the native CLIP baseline", x=0.02, ha="left", y=0.98, fontsize=17, weight="bold")
    fig.text(0.02, 0.90, f"Epoch 50 baseline: {reference[0]:.3f}% canonical Avg R@1 / {reference[1]:.2f}% species top-1", color=MUTED)
    for i, name in enumerate(names):
        row = runs[name][-1]
        values = [canonical_average(row), row["evaluation"]["species"]["top_1_accuracy"] * 100]
        for ax, value, ref in zip(axes, values, reference):
            delta = value - ref
            ax.plot([0, delta], [i, i], color=COLORS[name], alpha=0.6, linewidth=2)
            ax.scatter(delta, i, s=55, color=COLORS[name], zorder=3)
            ax.annotate(f"{delta:+.3f}" if ax is axes[0] else f"{delta:+.2f}", (delta, i), xytext=(-8 if delta < 0 else 8, 0), textcoords="offset points", ha="right" if delta < 0 else "left", va="center", fontsize=9, color=INK)
    axes[0].set_yticks(range(len(names)), [LABELS[n] for n in names])
    axes[0].invert_yaxis()
    for ax, title in zip(axes, ["Canonical Avg R@1", "Species top-1"]):
        decorate(ax, title, "Change from baseline (percentage points)")
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlim(-0.145, 0.035)
    axes[0].set_xticks([-0.10, -0.05, 0])
    axes[1].set_xlim(-2.6, 0.50)
    axes[1].set_xticks([-2, -1, 0])
    save(fig, "clip_final_comparison", "One training seed per arm; differences are descriptive, not significance estimates.\nAxes show changes, not absolute accuracy, and use different scales. All-caption retrieval is reported separately in the tables.")


def hardness_dynamics(runs):
    selected = ["uniform_top8", "uniform_top8_pressure_matched"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.73, wspace=0.28)
    fig.suptitle("Synthetic hardness changes under auxiliary training", x=0.02, ha="left", y=0.98, fontsize=17, weight="bold")
    for name in selected:
        rows = runs[name][1:]
        epochs = [r["epoch"] for r in rows]
        style = {"color": COLORS[name], "linewidth": 2.2, "label": LABELS[name]}
        axes[0].plot(epochs, [100 * r["training"]["fraction_synthetic_gt_hardest_real"] for r in rows], **style)
        axes[1].plot(epochs, [r["training"]["alpha_effective"] for r in rows], **style)
    decorate(axes[0], "Synthetic similarity exceeds hardest-real", "Epoch", "Fraction of training queries (%)")
    decorate(axes[1], "Applied auxiliary coefficient", "Epoch", "Mean effective alpha")
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 0.55)
    for ax in axes:
        schedule_bands(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.91), ncol=2, frameon=False)
    save(fig, "clip_hardness_dynamics", "Seed 42 · unsmoothed training-batch epoch means, not fixed-holdout measurements\nGray: auxiliary warmup; sand: ramp. Alpha=0.134 matches early projection-gradient pressure only; it is not an adaptive controller.")


def gradient_geometry():
    report = json.loads((DATA / "diagnostics/gradient_randomized.json").read_text())
    stability = report["cross_partition_stability"]
    partitions = ["shuffle_seed_42", "shuffle_seed_123", "shuffle_seed_4242"]

    def values(key):
        return np.array([stability[key]["by_partition"][p] for p in partitions])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4))
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.20, top=0.73, wspace=0.43)
    fig.suptitle("Frozen gradients separate hardness from update direction", x=0.02, ha="left", y=0.98, fontsize=17, weight="bold")
    hardness = values("fraction_u8_gt_hardest_real").mean() * 100
    fig.text(0.02, 0.88, f"Uniform top-8 is harder than the hardest real negative for {hardness:.2f}% of queries (mean across shuffled partitions).", color=MUTED)
    for ax, keys, title, ylabel in [
        (axes[0], ["u8_native_alignment", "hardest_real_native_alignment"], "Alignment with native row CE", "Gradient cosine"),
        (axes[1], ["u8_margin_directional_change", "hardest_real_margin_directional_change"], "Real-negative margin response", "Predicted change / unit auxiliary step"),
    ]:
        for x, key, name in zip([0, 1], keys, ["uniform_top8", "hardest_real"]):
            y = values(key)
            ax.bar(x, y.mean(), width=0.5, color=COLORS[name], alpha=0.8)
            ax.scatter(x + np.array([-0.09, 0, 0.09]), y, color=INK, s=24, zorder=3)
            ax.annotate(f"{y.mean():+.3f}", (x, y.mean()), xytext=(0, 15 if y.mean() > 0 else -19), textcoords="offset points", ha="center", fontsize=11)
        ax.set_xticks([0, 1], ["Uniform top-8", "Hardest real"])
        decorate(ax, title, "", ylabel)
        ax.axhline(0, color=MUTED, linewidth=1)
    axes[0].set_ylim(-0.04, 0.43)
    axes[1].set_ylim(-0.35, 1.50)
    y = values("joint_gradient_cosine_mean")
    axes[2].scatter(range(3), y, color=COLORS["ot_relative_a05"], s=60, zorder=3)
    axes[2].axhline(y.mean(), color=COLORS["ot_relative_a05"], linestyle="--", linewidth=1.3)
    axes[2].axhline(0, color=MUTED, linewidth=1)
    axes[2].text(0.04, 0.12, f"Mean {y.mean():+.3f}", transform=axes[2].transAxes, color=COLORS["ot_relative_a05"])
    axes[2].set_xticks(range(3), ["42", "123", "4242"])
    axes[2].set_ylim(-0.45, 0.10)
    decorate(axes[2], "Top-8 vs. hardest-real direction", "Partition shuffle seed", "Mean joint gradient cosine")
    save(fig, "clip_gradient_geometry", "Dots are three partitions of the same 1,024 frozen examples, not independent training seeds or confidence intervals.\nTangent embedding gradients; margin = positive similarity − hardest-real similarity. These measurements do not prove a full-training mechanism.")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": MUTED, "axes.edgecolor": "#c3cdd9",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "svg.fonttype": "none", "svg.hashsalt": "otco-clip-results",
    })
    runs = load_runs()
    training_overview(runs)
    final_comparison(runs)
    hardness_dynamics(runs)
    gradient_geometry()


if __name__ == "__main__":
    main()
