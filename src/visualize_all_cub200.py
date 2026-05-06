"""Generate recall tables and diagnostic plots for every CUB-200 results file."""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── parser ────────────────────────────────────────────────────────────────────

def parse_file(path):
    text = Path(path).read_text()
    steps, epochs = [], []

    step_pattern = re.compile(
        r"Step\s+(\d+)\s+\(Epoch\s+(\d+),.*?\)"
        r".*?Total Loss:\s*([\d.]+)"
        r".*?Positive pairs:\s*([-\d.]+)\s*±\s*([\d.]+)"
        r".*?All negatives:\s*([-\d.]+)"
        r".*?Hardest neg:\s*([-\d.]+)"
        r".*?Pos-Hard gap:\s*([-\d.]+)"
        r".*?Diagonal mean:\s*([-\d.]+)\s*±\s*([\d.]+)"
        r".*?Off-diagonal mean:\s*([-\d.]+)"
        r".*?Gap \(pos - neg\):\s*([-\d.]+)"
        r".*?Recall@1:\s*([\d.]+)%"
        r".*?Mean rank:\s*([\d.]+)"
        r".*?Image variance:\s*([\d.]+)"
        r".*?Text variance:\s*([\d.]+)",
        re.DOTALL,
    )
    for m in step_pattern.finditer(text):
        steps.append({
            "step": int(m.group(1)),
            "epoch": int(m.group(2)),
            "loss": float(m.group(3)),
            "cos_pos": float(m.group(4)),
            "cos_neg_mean": float(m.group(6)),
            "cos_neg_hard": float(m.group(7)),
            "logit_gap": float(m.group(12)),
            "batch_r1": float(m.group(13)),
            "img_var": float(m.group(15)),
            "txt_var": float(m.group(16)),
        })

    val_pattern = re.compile(
        r"VALIDATION - Epoch\s+(\d+).*?"
        r"Average training loss:\s*([\d.]+).*?"
        r"ALL CAPTIONS.*?"
        r"Text → Image:.*?R@1:\s*([\d.]+)%.*?R@5:\s*([\d.]+)%.*?R@10:\s*([\d.]+)%.*?"
        r"Image → Text:.*?R@1:\s*([\d.]+)%.*?R@5:\s*([\d.]+)%.*?R@10:\s*([\d.]+)%.*?"
        r"CANONICAL.*?"
        r"Text → Image:.*?R@1:\s*([\d.]+)%.*?R@5:\s*([\d.]+)%.*?R@10:\s*([\d.]+)%.*?"
        r"Image → Text:.*?R@1:\s*([\d.]+)%.*?R@5:\s*([\d.]+)%.*?R@10:\s*([\d.]+)%.*?"
        r"Average Recall@1 \(canonical\):\s*([\d.]+)%",
        re.DOTALL,
    )
    seen = set()
    for m in val_pattern.finditer(text):
        ep = int(m.group(1))
        if ep in seen:
            continue
        seen.add(ep)
        epochs.append({
            "epoch": ep,
            "can_t2i_r1": float(m.group(9)),
            "can_t2i_r5": float(m.group(10)),
            "can_t2i_r10": float(m.group(11)),
            "can_i2t_r1": float(m.group(12)),
            "can_i2t_r5": float(m.group(13)),
            "can_i2t_r10": float(m.group(14)),
            "avg_r1": float(m.group(15)),
        })

    epochs.sort(key=lambda x: x["epoch"])
    steps.sort(key=lambda x: x["step"])
    return steps, epochs


def parse_ot_steps(path):
    """Extract OT-specific diagnostics from step blocks (optional, best-effort)."""
    text = Path(path).read_text()
    ot_pattern = re.compile(
        r"Step\s+(\d+)\s+\(Epoch\s+(\d+),.*?\)"
        r".*?Alpha:\s*([\d.]+)"
        r".*?Coupling Entropy:\s*([\d.]+)",
        re.DOTALL,
    )
    ot_steps = []
    for m in ot_pattern.finditer(text):
        ot_steps.append({
            "step": int(m.group(1)),
            "alpha": float(m.group(3)),
            "coupling_entropy": float(m.group(4)),
        })
    ot_steps.sort(key=lambda x: x["step"])
    return ot_steps


# ── table ─────────────────────────────────────────────────────────────────────

def write_table(epochs, label, out_path):
    header = (
        f"{'Ep':>3} | "
        f"{'T→I R@1':>8} {'T→I R@5':>8} {'T→I R@10':>9} | "
        f"{'I→T R@1':>8} {'I→T R@5':>8} {'I→T R@10':>9} | "
        f"{'Avg R@1':>8}"
    )
    sep = "-" * len(header)
    lines = [f"{label} — Canonical Recall (first caption only, 5794-image pool)", sep, header, sep]

    for e in [e for e in epochs if e["epoch"] % 5 == 0 or e == epochs[-1]]:
        lines.append(
            f"{e['epoch']:>3} | "
            f"{e['can_t2i_r1']:>7.2f}% {e['can_t2i_r5']:>7.2f}% {e['can_t2i_r10']:>8.2f}% | "
            f"{e['can_i2t_r1']:>7.2f}% {e['can_i2t_r5']:>7.2f}% {e['can_i2t_r10']:>8.2f}% | "
            f"{e['avg_r1']:>7.2f}%"
        )
    lines.append(sep)

    best = max(epochs, key=lambda e: e["avg_r1"])
    lines.append(
        f"BEST ep{best['epoch']:02d}: T→I {best['can_t2i_r1']:.2f}%  "
        f"I→T {best['can_i2t_r1']:.2f}%  Avg {best['avg_r1']:.2f}%"
    )

    table = "\n".join(lines)
    print(table)
    Path(out_path).write_text(table + "\n")
    print(f"\nTable saved → {out_path}")


# ── plots ─────────────────────────────────────────────────────────────────────

def make_plots(steps, epochs, ot_steps, label, out_path):
    ep_x = [e["epoch"] for e in epochs]
    st_x = [s["step"] for s in steps]
    has_ot = len(ot_steps) > 0

    nrows = 4 if not has_ot else 5
    fig = plt.figure(figsize=(16, 5 * nrows))
    fig.suptitle(f"{label} — Training Diagnostics", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.55, wspace=0.35)

    # row 0: R@1 (full width)
    ax_r1 = fig.add_subplot(gs[0, :])
    ax_r1.plot(ep_x, [e["can_t2i_r1"] for e in epochs], marker="o", ms=3, label="T→I R@1")
    ax_r1.plot(ep_x, [e["can_i2t_r1"] for e in epochs], marker="s", ms=3, label="I→T R@1")
    ax_r1.plot(ep_x, [e["avg_r1"] for e in epochs], marker="^", ms=3, lw=2, label="Avg R@1", color="black")
    best = max(epochs, key=lambda e: e["avg_r1"])
    ax_r1.axvline(best["epoch"], color="gray", lw=0.8, linestyle="--")
    ax_r1.set_title(f"Canonical R@1  (best: ep{best['epoch']} Avg={best['avg_r1']:.2f}%)")
    ax_r1.set_xlabel("Epoch"); ax_r1.set_ylabel("R@1 (%)"); ax_r1.legend(); ax_r1.grid(True, alpha=0.3)

    # row 1: R@5 / R@10
    ax_r5 = fig.add_subplot(gs[1, 0])
    ax_r5.plot(ep_x, [e["can_t2i_r5"] for e in epochs], label="T→I R@5")
    ax_r5.plot(ep_x, [e["can_i2t_r5"] for e in epochs], label="I→T R@5")
    ax_r5.set_title("Canonical R@5"); ax_r5.set_xlabel("Epoch"); ax_r5.set_ylabel("%")
    ax_r5.legend(); ax_r5.grid(True, alpha=0.3)

    ax_r10 = fig.add_subplot(gs[1, 1])
    ax_r10.plot(ep_x, [e["can_t2i_r10"] for e in epochs], label="T→I R@10")
    ax_r10.plot(ep_x, [e["can_i2t_r10"] for e in epochs], label="I→T R@10")
    ax_r10.set_title("Canonical R@10"); ax_r10.set_xlabel("Epoch"); ax_r10.set_ylabel("%")
    ax_r10.legend(); ax_r10.grid(True, alpha=0.3)

    # row 2: logit gap / cosine similarity
    ax_lg = fig.add_subplot(gs[2, 0])
    ax_lg.plot(st_x, [s["logit_gap"] for s in steps], color="steelblue")
    ax_lg.axhline(0, color="red", lw=0.8, linestyle="--")
    ax_lg.set_title("Logit Gap (pos − neg)"); ax_lg.set_xlabel("Step"); ax_lg.set_ylabel("Logit units")
    ax_lg.grid(True, alpha=0.3)

    ax_cos = fig.add_subplot(gs[2, 1])
    ax_cos.plot(st_x, [s["cos_pos"] for s in steps], label="Positive pairs", color="green")
    ax_cos.plot(st_x, [s["cos_neg_mean"] for s in steps], label="All negatives", color="gray")
    ax_cos.plot(st_x, [s["cos_neg_hard"] for s in steps], label="Hardest neg", color="red", linestyle="--")
    ax_cos.axhline(0, color="black", lw=0.5, linestyle=":")
    ax_cos.set_title("Cosine Similarity"); ax_cos.set_xlabel("Step"); ax_cos.set_ylabel("Cosine sim")
    ax_cos.legend(fontsize=8); ax_cos.grid(True, alpha=0.3)

    # row 3: batch recall+loss / embedding variance
    ax_br = fig.add_subplot(gs[3, 0])
    ax_br2 = ax_br.twinx()
    ax_br.plot(st_x, [s["batch_r1"] for s in steps], color="purple", label="Batch R@1")
    ax_br2.plot(st_x, [s["loss"] for s in steps], color="orange", linestyle="--", label="Loss")
    ax_br.set_title("Batch Recall@1 & Training Loss"); ax_br.set_xlabel("Step")
    ax_br.set_ylabel("Batch R@1 (%)", color="purple"); ax_br2.set_ylabel("Loss", color="orange")
    l1, lb1 = ax_br.get_legend_handles_labels(); l2, lb2 = ax_br2.get_legend_handles_labels()
    ax_br.legend(l1 + l2, lb1 + lb2, fontsize=8); ax_br.grid(True, alpha=0.3)

    ax_var = fig.add_subplot(gs[3, 1])
    ax_var.plot(st_x, [s["img_var"] for s in steps], label="Image variance", color="navy")
    ax_var.plot(st_x, [s["txt_var"] for s in steps], label="Text variance", color="darkorange")
    ax_var.set_title("Embedding Variance"); ax_var.set_xlabel("Step"); ax_var.set_ylabel("Variance")
    ax_var.legend(); ax_var.grid(True, alpha=0.3)

    # row 4 (OT runs only): alpha ramp / coupling entropy
    if has_ot:
        ot_x = [s["step"] for s in ot_steps]

        ax_alpha = fig.add_subplot(gs[4, 0])
        ax_alpha.plot(ot_x, [s["alpha"] for s in ot_steps], color="teal")
        ax_alpha.set_title("OT Alpha Schedule"); ax_alpha.set_xlabel("Step"); ax_alpha.set_ylabel("Alpha")
        ax_alpha.grid(True, alpha=0.3)

        ax_ent = fig.add_subplot(gs[4, 1])
        ax_ent.plot(ot_x, [s["coupling_entropy"] for s in ot_steps], color="crimson")
        ax_ent.axhline(3.0, color="gray", lw=0.8, linestyle="--", label="threshold=3.0")
        ax_ent.set_title("OT Coupling Entropy"); ax_ent.set_xlabel("Step"); ax_ent.set_ylabel("Entropy")
        ax_ent.legend(fontsize=8); ax_ent.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

RUNS = [
    ("CUB_200_baseline.txt",                    "CUB-200 Baseline",                  "cub200_baseline"),
    ("CUB_200_OT-MIX.txt",                      "CUB-200 OT-Mix Adaptive",           "cub200_ot_mix_adaptive"),
    ("CUB_200_OT-MIX_gated.txt",                "CUB-200 OT-Mix Adaptive Gated",     "cub200_ot_mix_adaptive_gated"),
    ("CUB_200_OT-MIX_cached_pool_128.txt",      "CUB-200 Cached-Pool N=128",         "cub200_ot_mix_cached_pool_128"),
    ("CUB_200_OT-MIX_cached_pool_256.txt",      "CUB-200 Cached-Pool N=256",         "cub200_ot_mix_cached_pool_256"),
    ("CUB_200_OT-MIX_cached_pool_512.txt",      "CUB-200 Cached-Pool N=512",         "cub200_ot_mix_cached_pool_512"),
    ("CUB_200_OT-MIX_live_reforward_pool_128.txt", "CUB-200 Live Re-forward N=128",  "cub200_ot_mix_live_reforward_128"),
    ("CUB_200_OT-MIX_mixed.txt",                "CUB-200 OT-Mix Mixed Batching",     "cub200_ot_mix_mixed"),
    ("CUB_200_OT-MIX_mixed_gated.txt",          "CUB-200 OT-Mix Mixed+Gated",        "cub200_ot_mix_mixed_gated"),
    ("CUB_200_OT-MIX_stratified.txt",           "CUB-200 OT-Mix Stratified",         "cub200_ot_mix_stratified"),
]

if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"

    for fname, label, stem in RUNS:
        src = results_dir / fname
        if not src.exists():
            print(f"SKIP (not found): {fname}")
            continue

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        steps, epochs = parse_file(src)
        if not epochs:
            print(f"  No validation epochs parsed — skipping.")
            continue

        ot_steps = parse_ot_steps(src)
        print(f"  Parsed {len(steps)} step snapshots, {len(epochs)} epochs, {len(ot_steps)} OT records")

        write_table(epochs, label, results_dir / f"{stem}_table.txt")
        make_plots(steps, epochs, ot_steps, label, results_dir / f"{stem}_plots.png")
