"""Small independent arithmetic check for the research defense workbook.

Read-only, standard library only. Does not execute model/optimizer code or prove
the authenticity of recorded GPU checks. Run from any working directory.
"""

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "experiment_results/clip_usefulness_prospective_2026-09"
SEEDS = (789, 2026, 31415)
STEPS = (100, 250, 500, 750, 1001)
PARTS = ("shuffle_seed_42", "shuffle_seed_123", "shuffle_seed_4242")
ARMS = ("baseline", "uniform_top8", "hardest_real")


def read(path):
    return json.loads(path.read_text())


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def close(actual, expected, description):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), description)


def main():
    audit = read(DIRECTORY / "audit.json")
    checked = 0
    for path in DIRECTORY.rglob("*"):
        if not path.is_file() or path.name in {"audit.json", "README.md"}:
            continue
        relative = str(path.relative_to(DIRECTORY))
        expected = audit["files"][relative]
        payload = path.read_bytes()
        require(len(payload) == expected["bytes"], f"Size: {relative}")
        require(hashlib.sha256(payload).hexdigest() == expected["sha256"], f"Hash: {relative}")
        checked += 1

    states = []
    records = 0
    for seed in SEEDS:
        path = DIRECTORY / f"seed_{seed}/results/paired/paired_updates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        indexed = {(r["checkpoint_step"], r["trial"], r["arm"]): r for r in rows}
        expected_keys = {(s, t, a) for s in STEPS for t in range(16) for a in ARMS}
        require(len(rows) == 240 and set(indexed) == expected_keys, f"Branch grid: {seed}")
        means = {}
        for key, row in indexed.items():
            require(all(len(row["heldout_loss"][p]) == 16 for p in PARTS), f"Partitions: {key}")
            means[key] = fmean(fmean(row["heldout_loss"][p]) for p in PARTS)
            close(means[key], row["heldout_mean"], f"Branch mean: {seed}/{key}")
        for key, row in indexed.items():
            delta = means[key] - means[key[0], key[1], "baseline"]
            close(delta, row["incremental_heldout_loss"], f"Paired difference: {seed}/{key}")
        for step in STEPS:
            selected = [indexed[step, t, "uniform_top8"] for t in range(16)]
            effect = fmean(means[step, t, "uniform_top8"] - means[step, t, "baseline"]
                           for t in range(16))
            states.append({"training_seed": seed, "checkpoint_step": step,
                           "effect": effect,
                           "full_gradient_alignment": fmean(r["full_gradient_alignment"] for r in selected)})
        records += len(rows)

    archived_states = {(s["training_seed"], s["checkpoint_step"]): s
                       for s in read(DIRECTORY / "results/states.json")}
    require(len(archived_states) == 15, "Archived state count")
    for state in states:
        archived = archived_states[state["training_seed"], state["checkpoint_step"]]
        for field in ("effect", "full_gradient_alignment"):
            close(state[field], archived[field], f"State {field}: {state}")

    frozen = read(DIRECTORY / "results/frozen_rules.json")
    require(frozen == read(ROOT / "configs/clip_usefulness_frozen_rules.json"), "Frozen artifact equality")
    report = read(DIRECTORY / "results/prediction_report.json")
    results = {}
    for name, rule in frozen["rules"].items():
        per_seed = []
        correct = nonzero_correct = nonzero_count = 0
        tp_total = fp_total = 0
        for seed in SEEDS:
            chosen = [s for s in states if s["training_seed"] == seed]
            truth = [s["effect"] < 0 for s in chosen]
            prediction = [(s[rule["feature"]] <= rule["threshold"]) == rule["lower_is_helpful"]
                          for s in chosen]
            positives = sum(truth)
            negatives = len(truth) - positives
            require(positives > 0 and negatives > 0, f"Both outcome classes: {seed}")
            tp = sum(y and p for y, p in zip(truth, prediction))
            tn = sum(not y and not p for y, p in zip(truth, prediction))
            per_seed.append((tp / positives + tn / negatives) / 2)
            tp_total += tp
            fp_total += negatives - tn
            correct += tp + tn
            for state, y, p in zip(chosen, truth, prediction):
                if abs(state["effect"]) > 1e-6:
                    nonzero_count += 1
                    nonzero_correct += y == p
        require(correct == 13 and nonzero_correct == 12 and nonzero_count == 13, f"Accuracy: {name}")
        results[name] = fmean(per_seed)
        print(f"{name}: mean seed balanced accuracy={100*fmean(per_seed):.6f}%; "
              f"correct={correct}/15; outside band={nonzero_correct}/{nonzero_count}; "
              f"true positives={tp_total}; false positives={fp_total}")
    difference = results["full_gradient_alignment"] - results["checkpoint_step"]
    close(difference, report["primary_mean_balanced_accuracy_difference"], "Primary difference")

    training = ROOT / "experiment_results/clip_2026-08-30_to_09-01/training"
    averages = {}
    for path in sorted(training.glob("*/summary.json")):
        metrics = read(path)["final"]["evaluation"]["canonical_retrieval"]
        averages[path.parent.name] = fmean(metrics[d]["r_at_1"] for d in ("text_to_image", "image_to_text"))
    require(len(averages) == 8, "Eight original training arms")
    require(all(value < averages["baseline"] for arm, value in averages.items()
                if arm != "baseline"), "No final canonical auxiliary win")
    print(f"Primary difference: {100*difference:.6f} percentage points")
    print(f"Verified {checked} available inventory files, {records} branches, {len(states)} states.")
    print("Eight original final canonical Avg R@1 percentages:",
          json.dumps({k: round(v, 6) for k, v in averages.items()}, sort_keys=True))
    print("PASS: archive consistency and selected arithmetic only; GPU execution not rerun.")


if __name__ == "__main__":
    main()
