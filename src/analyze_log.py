import argparse
import re
from collections import defaultdict


STEP_RE = re.compile(r"Step\s+(\d+)\s+\(Epoch\s+(\d+),")
EPOCH_RE = re.compile(r"EPOCH\s+(\d+)/")
VAL_CANONICAL_RE = re.compile(r"Average Recall@1 \(canonical\):\s+([0-9.]+)%")


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_log(path):
    steps = []
    val_by_epoch = {}
    current_epoch = None
    current_step = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.strip()

        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        step_match = STEP_RE.search(line)
        if step_match:
            if current_step is not None:
                steps.append(current_step)
            current_step = {
                "step": int(step_match.group(1)),
                "epoch": int(step_match.group(2)),
            }
            continue

        if current_step is not None:
            if "Alpha:" in line:
                current_step["alpha"] = _to_float(line.split(":")[-1].strip())
            elif "Synthetic Negative Loss:" in line:
                current_step["synthetic_loss"] = _to_float(line.split(":")[-1].strip())
            elif "OT-Select Loss:" in line:
                current_step["select_loss"] = _to_float(line.split(":")[-1].strip())
            elif "Hard Negative Loss:" in line:
                current_step["hard_loss"] = _to_float(line.split(":")[-1].strip())
            elif "Avg Synthetic Sim:" in line:
                current_step["avg_synthetic_sim"] = _to_float(line.split(":")[-1].strip())
            elif "Avg Selected Sim:" in line:
                current_step["avg_selected_sim"] = _to_float(line.split(":")[-1].strip())
            elif "Selected Neg Rank (mean):" in line:
                current_step["selected_neg_rank_mean"] = _to_float(line.split(":")[-1].strip())
            elif "Selected Neg Rank (median):" in line:
                current_step["selected_neg_rank_median"] = _to_float(line.split(":")[-1].strip())
            elif "Pos - Selected Gap:" in line:
                current_step["pos_selected_gap"] = _to_float(line.split(":")[-1].strip())
            elif "Coupling Entropy:" in line:
                current_step["coupling_entropy"] = _to_float(line.split(":")[-1].strip())
            elif "Coupling Peak Mass:" in line:
                current_step["coupling_peak_mass"] = _to_float(line.split(":")[-1].strip())

        val_match = VAL_CANONICAL_RE.search(line)
        if val_match and current_epoch is not None:
            val_by_epoch[current_epoch] = float(val_match.group(1))

    if current_step is not None:
        steps.append(current_step)

    return steps, val_by_epoch


def summarize_by_epoch(steps, val_by_epoch):
    by_epoch = defaultdict(list)
    for row in steps:
        by_epoch[row["epoch"]].append(row)

    metrics = [
        "alpha",
        "synthetic_loss",
        "select_loss",
        "hard_loss",
        "avg_synthetic_sim",
        "avg_selected_sim",
        "selected_neg_rank_mean",
        "selected_neg_rank_median",
        "pos_selected_gap",
        "coupling_entropy",
        "coupling_peak_mass",
    ]

    summary = []
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        item = {
            "epoch": epoch,
            "num_steps_logged": len(rows),
            "val_canonical_avg_r1": val_by_epoch.get(epoch),
        }
        for m in metrics:
            vals = [r[m] for r in rows if m in r and r[m] is not None]
            item[m] = (sum(vals) / len(vals)) if vals else None
        summary.append(item)
    return summary


def print_summary(summary, warmup_steps=None):
    print(
        "epoch steps valR1 alpha selRank selGap avgSelSim avgSynthSim "
        "entropy peakMass synthLoss selectLoss hardLoss"
    )
    for s in summary:
        row = [
            f"{s['epoch']:>5}",
            f"{s['num_steps_logged']:>5}",
            f"{fmt(s['val_canonical_avg_r1'], pct=True):>6}",
            f"{fmt(s['alpha']):>5}",
            f"{fmt(s['selected_neg_rank_mean']):>7}",
            f"{fmt(s['pos_selected_gap']):>6}",
            f"{fmt(s['avg_selected_sim']):>9}",
            f"{fmt(s['avg_synthetic_sim']):>10}",
            f"{fmt(s['coupling_entropy']):>7}",
            f"{fmt(s['coupling_peak_mass']):>8}",
            f"{fmt(s['synthetic_loss']):>9}",
            f"{fmt(s['select_loss']):>10}",
            f"{fmt(s['hard_loss']):>8}",
        ]
        print(" ".join(row))

    if warmup_steps is None:
        return

    pre, post = [], []
    for s in summary:
        if s["alpha"] is None:
            continue
        if s["alpha"] <= 0:
            pre.append(s)
        else:
            post.append(s)

    print("\nphase avg_valR1 avg_selRank avg_selGap avg_entropy avg_peakMass")
    for name, chunk in [("pre-ot", pre), ("post-ot", post)]:
        if not chunk:
            print(f"{name:>7} n/a n/a n/a n/a n/a")
            continue
        print(
            f"{name:>7} "
            f"{fmt(mean(chunk, 'val_canonical_avg_r1'), pct=True)} "
            f"{fmt(mean(chunk, 'selected_neg_rank_mean'))} "
            f"{fmt(mean(chunk, 'pos_selected_gap'))} "
            f"{fmt(mean(chunk, 'coupling_entropy'))} "
            f"{fmt(mean(chunk, 'coupling_peak_mass'))}"
        )


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v:.2f}%" if pct else f"{v:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize OT training diagnostics from results logs.")
    parser.add_argument("logfile", help="Path to results txt log")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Optional warmup steps (for phase split)")
    args = parser.parse_args()

    steps, val_by_epoch = parse_log(args.logfile)
    if not steps:
        raise SystemExit("No step logs found in file.")

    summary = summarize_by_epoch(steps, val_by_epoch)
    print_summary(summary, warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
