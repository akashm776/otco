"""Parse CUB-200 results file and produce a recall table + diagnostic plots."""

import re
import sys
from pathlib import Path


def parse_file(path):
    text = Path(path).read_text()
    steps, epochs = [], []

    # ── step blocks ──────────────────────────────────────────────────────────
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
            "cos_pos_std": float(m.group(5)),
            "cos_neg_mean": float(m.group(6)),
            "cos_neg_hard": float(m.group(7)),
            "cos_gap": float(m.group(8)),
            "logit_diag": float(m.group(9)),
            "logit_diag_std": float(m.group(10)),
            "logit_offdiag": float(m.group(11)),
            "logit_gap": float(m.group(12)),
            "batch_r1": float(m.group(13)),
            "mean_rank": float(m.group(14)),
            "img_var": float(m.group(15)),
            "txt_var": float(m.group(16)),
        })

    # ── epoch validation blocks ───────────────────────────────────────────────
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
            "train_loss": float(m.group(2)),
            "all_t2i_r1": float(m.group(3)),
            "all_t2i_r5": float(m.group(4)),
            "all_t2i_r10": float(m.group(5)),
            "all_i2t_r1": float(m.group(6)),
            "all_i2t_r5": float(m.group(7)),
            "all_i2t_r10": float(m.group(8)),
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


def print_table(epochs, out_path):
    header = (
        f"{'Ep':>3} | "
        f"{'T→I R@1':>8} {'T→I R@5':>8} {'T→I R@10':>9} | "
        f"{'I→T R@1':>8} {'I→T R@5':>8} {'I→T R@10':>9} | "
        f"{'Avg R@1':>8}"
    )
    # Epochs 5/10/15 from checkpoint eval script (R@5/R@10 not captured then)
    early = [
        {"epoch":  5, "can_t2i_r1": 0.07, "can_i2t_r1": 0.03, "avg_r1": 0.05},
        {"epoch": 10, "can_t2i_r1": 0.38, "can_i2t_r1": 0.45, "avg_r1": 0.41},
        {"epoch": 15, "can_t2i_r1": 0.38, "can_i2t_r1": 0.78, "avg_r1": 0.58},
    ]

    sep = "-" * len(header)
    lines = [
        "CUB-200 Baseline — Canonical Recall (first caption only, 5794-image pool)",
        sep, header, sep,
    ]
    for e in early:
        lines.append(
            f"{e['epoch']:>3} | "
            f"{e['can_t2i_r1']:>7.2f}%      —         — | "
            f"{e['can_i2t_r1']:>7.2f}%      —         — | "
            f"{e['avg_r1']:>7.2f}%"
        )
    for e in [e for e in epochs if e["epoch"] % 5 == 0]:
        lines.append(
            f"{e['epoch']:>3} | "
            f"{e['can_t2i_r1']:>7.2f}% {e['can_t2i_r5']:>7.2f}% {e['can_t2i_r10']:>8.2f}% | "
            f"{e['can_i2t_r1']:>7.2f}% {e['can_i2t_r5']:>7.2f}% {e['can_i2t_r10']:>8.2f}% | "
            f"{e['avg_r1']:>7.2f}%"
        )
    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    Path(out_path).write_text(table + "\n")
    print(f"\nTable saved → {out_path}")


def make_plots(steps, epochs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ep_x = [e["epoch"] for e in epochs]
    st_x = [s["step"] for s in steps]

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle("CUB-200 Baseline — Training Diagnostics", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── top row: recall curves ───────────────────────────────────────────────
    ax_r1 = fig.add_subplot(gs[0, :])
    ax_r1.plot(ep_x, [e["can_t2i_r1"] for e in epochs], marker="o", ms=3, label="T→I R@1")
    ax_r1.plot(ep_x, [e["can_i2t_r1"] for e in epochs], marker="s", ms=3, label="I→T R@1")
    ax_r1.plot(ep_x, [e["avg_r1"] for e in epochs], marker="^", ms=3, lw=2, label="Avg R@1", color="black")
    ax_r1.set_title("Canonical R@1 (main metric)")
    ax_r1.set_xlabel("Epoch")
    ax_r1.set_ylabel("R@1 (%)")
    ax_r1.legend()
    ax_r1.grid(True, alpha=0.3)

    # ── row 2 left: R@5 / R@10 ──────────────────────────────────────────────
    ax_r5 = fig.add_subplot(gs[1, 0])
    ax_r5.plot(ep_x, [e["can_t2i_r5"] for e in epochs], label="T→I R@5")
    ax_r5.plot(ep_x, [e["can_i2t_r5"] for e in epochs], label="I→T R@5")
    ax_r5.set_title("Canonical R@5")
    ax_r5.set_xlabel("Epoch"); ax_r5.set_ylabel("%"); ax_r5.legend(); ax_r5.grid(True, alpha=0.3)

    ax_r10 = fig.add_subplot(gs[1, 1])
    ax_r10.plot(ep_x, [e["can_t2i_r10"] for e in epochs], label="T→I R@10")
    ax_r10.plot(ep_x, [e["can_i2t_r10"] for e in epochs], label="I→T R@10")
    ax_r10.set_title("Canonical R@10")
    ax_r10.set_xlabel("Epoch"); ax_r10.set_ylabel("%"); ax_r10.legend(); ax_r10.grid(True, alpha=0.3)

    # ── row 3 left: logit gap ────────────────────────────────────────────────
    ax_lg = fig.add_subplot(gs[2, 0])
    ax_lg.plot(st_x, [s["logit_gap"] for s in steps], color="steelblue")
    ax_lg.axhline(0, color="red", lw=0.8, linestyle="--")
    ax_lg.set_title("Logit Gap (pos − neg diagonal)")
    ax_lg.set_xlabel("Step"); ax_lg.set_ylabel("Logit units"); ax_lg.grid(True, alpha=0.3)

    # ── row 3 right: cosine similarity ──────────────────────────────────────
    ax_cos = fig.add_subplot(gs[2, 1])
    ax_cos.plot(st_x, [s["cos_pos"] for s in steps], label="Positive pairs", color="green")
    ax_cos.plot(st_x, [s["cos_neg_mean"] for s in steps], label="All negatives", color="gray")
    ax_cos.plot(st_x, [s["cos_neg_hard"] for s in steps], label="Hardest neg", color="red", linestyle="--")
    ax_cos.axhline(0, color="black", lw=0.5, linestyle=":")
    ax_cos.set_title("Cosine Similarity")
    ax_cos.set_xlabel("Step"); ax_cos.set_ylabel("Cosine sim"); ax_cos.legend(fontsize=8); ax_cos.grid(True, alpha=0.3)

    # ── row 4 left: batch recall + loss ─────────────────────────────────────
    ax_br = fig.add_subplot(gs[3, 0])
    ax_br2 = ax_br.twinx()
    ax_br.plot(st_x, [s["batch_r1"] for s in steps], color="purple", label="Batch R@1")
    ax_br2.plot(st_x, [s["loss"] for s in steps], color="orange", linestyle="--", label="Loss")
    ax_br.set_title("Batch Recall@1 & Training Loss")
    ax_br.set_xlabel("Step"); ax_br.set_ylabel("Batch R@1 (%)", color="purple")
    ax_br2.set_ylabel("Loss", color="orange")
    lines1, labels1 = ax_br.get_legend_handles_labels()
    lines2, labels2 = ax_br2.get_legend_handles_labels()
    ax_br.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax_br.grid(True, alpha=0.3)

    # ── row 4 right: embedding variance ─────────────────────────────────────
    ax_var = fig.add_subplot(gs[3, 1])
    ax_var.plot(st_x, [s["img_var"] for s in steps], label="Image variance", color="navy")
    ax_var.plot(st_x, [s["txt_var"] for s in steps], label="Text variance", color="darkorange")
    ax_var.set_title("Embedding Variance")
    ax_var.set_xlabel("Step"); ax_var.set_ylabel("Variance"); ax_var.legend(); ax_var.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "experiment_logs"
    src = Path(__file__).parent.parent / "results" / "CUB_200_baseline.txt"
    if not src.exists():
        print(f"Not found: {src}"); sys.exit(1)

    steps, epochs = parse_file(src)
    print(f"Parsed {len(steps)} step snapshots, {len(epochs)} epoch validations\n")

    print_table(epochs, results_dir / "cub200_baseline_table.txt")
    make_plots(steps, epochs, results_dir / "cub200_baseline_plots.png")
