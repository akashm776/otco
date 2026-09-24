"""Public evidence integrity and independently recomputed descriptive arithmetic."""
import json
from pathlib import Path
from statistics import fmean

import pytest

from scripts.archive_clip_followup_evidence import STUDIES, safe_path, verify

ROOT = Path(__file__).resolve().parents[1] / "experiment_results"


@pytest.mark.parametrize("name", STUDIES)
def test_retained_hashes(name):
    verify(ROOT / name)


@pytest.mark.parametrize("name", ["/absolute", "../escape", "a/../../b", "a\\b", ""])
def test_unsafe_paths(name):
    with pytest.raises(ValueError):
        safe_path(name)


def read(path):
    return json.loads(path.read_text())


def test_diagnostic_recorded_checks_and_arithmetic():
    directory = ROOT / "clip_checkpoint_diagnostic_2026-09"
    rows = [read(p) for p in sorted(directory.glob("seed_*/step_*/result.json"))]
    assert len(rows) == 12
    assert {(r["training_seed"], r["checkpoint_step"]) for r in rows} == {
        (seed, step) for seed in (789, 2026, 31415) for step in (100, 250, 500, 750)}
    for row in rows:
        audit = row["audit"]
        assert len(set(audit["branch_start_hashes"].values())) == 1
        for key in ("exact_trial_replay", "exact_no_update_report_replay", "matched_rng_and_lrs",
                    "frozen_parameters_unchanged", "source_immutable"):
            assert audit[key] is True
        assert audit["actual_optimizer_steps"] == 152
        for arm, count in (("native", 0), ("pulse", 1), ("sustained", 50)):
            assert len(row["traces"][arm]) == 50
            assert sum(t["coefficient"] > 0 for t in row["traces"][arm]) == count
        assert row["first_reports"]["pulse"] == row["first_reports"]["sustained"]
        for arm in ("pulse", "sustained"):
            difference = row["differences"][arm]
            for reports, metric in (("first_reports", "step1_loss"), ("endpoints", "step50_loss")):
                for result in row[reports].values():
                    assert result["native_loss"] == pytest.approx(
                        fmean(fmean(values) for values in result["partition_losses"].values()), abs=1e-12)
                assert difference[metric] == pytest.approx(
                    row[reports][arm]["native_loss"] - row[reports]["native"]["native_loss"], abs=1e-12)
            assert difference["step50_pool_r1_pp"] == pytest.approx(
                row["endpoints"][arm]["pool_retrieval"]["mean_r1_percent"] -
                row["endpoints"]["native"]["pool_retrieval"]["mean_r1_percent"], abs=1e-12)
    summary = read(directory / "comparison.json")
    for arm in ("pulse", "sustained"):
        for metric, value in summary["seed_mean"][arm].items():
            assert value == pytest.approx(fmean(r["differences"][arm][metric] for r in rows), abs=1e-12)
    for name, predictions in summary["predictor_descriptives"].items():
        for outcome, arm, metric in (("immediate", "pulse", "step1_loss"),
                                     ("pulse50", "pulse", "step50_loss"),
                                     ("sustained50", "sustained", "step50_loss")):
            correct = sum(r["predictors"]["decisions"][name] ==
                          (r["differences"][arm][metric] < 0) for r in rows)
            assert correct == predictions[outcome]["correct_sign"]
    assert all((r["differences"]["sustained"]["step50_loss"] < 0) ==
               (r["checkpoint_step"] == 100) for r in rows)


@pytest.mark.parametrize("study", ["clip_gated_training_2026-09", "clip_exposure_matched_2026-09"])
def test_rollout_endpoint_metrics(study):
    directory = ROOT / study
    comparison = read(directory / "comparison.json")
    assert len(comparison["rows"]) == 12
    for row in comparison["rows"]:
        path = directory / f"seed_{row['seed']}" / row["arm"] / "training/metrics.jsonl"
        final = json.loads(path.read_text().splitlines()[-1])
        assert final["global_step"] == 1001
        evaluation = final["evaluation"]
        assert row["canonical_avg_r1_percent"] == pytest.approx(fmean(
            evaluation["canonical_retrieval"][direction]["r_at_1"]
            for direction in ("text_to_image", "image_to_text")), abs=1e-12)
        assert row["species_top1_percent"] == pytest.approx(
            100 * evaluation["species"]["top_1_accuracy"], abs=1e-12)


def test_transfer_raw_effects_and_frozen_scores():
    directory = ROOT / "clip_evaluation_transfer_2026-09"
    states = read(directory / "results/states.json")
    report = read(directory / "results/prediction_report.json")
    assert len(states) == 15
    for seed in (789, 2026, 31415):
        path = directory / f"seed_{seed}/results/transfer/paired_updates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        indexed = {(r["checkpoint_step"], r["trial"], r["arm"]): r for r in rows}
        assert len(rows) == len(indexed) == 240
        for row in rows:
            mean = fmean(fmean(row["heldout_loss"][p]) for p in
                         ("shuffle_seed_42", "shuffle_seed_123", "shuffle_seed_4242"))
            assert row["heldout_mean"] == pytest.approx(mean, abs=1e-12)
            baseline = indexed[row["checkpoint_step"], row["trial"], "baseline"]
            assert row["incremental_heldout_loss"] == pytest.approx(
                mean - baseline["heldout_mean"], abs=1e-12)
        for state in (s for s in states if s["training_seed"] == seed):
            effect = fmean(indexed[state["checkpoint_step"], trial, "uniform_top8"]
                           ["incremental_heldout_loss"] for trial in range(16))
            assert state["effect"] == pytest.approx(effect, abs=1e-12)
    for prediction in report["predictions"]:
        state = next(s for s in states if s["training_seed"] == prediction["training_seed"]
                     and s["checkpoint_step"] == prediction["checkpoint_step"])
        assert prediction["actual_helpful"] == (state["effect"] < 0)
        decision = (state["full_gradient_alignment"] > 0.030847286945687016
                    if prediction["predictor"] == "full_gradient_alignment"
                    else state["checkpoint_step"] <= 175)
        assert prediction["predicted_helpful"] == decision
    balanced = {}
    for score in report["per_seed_scores"]:
        rows = [r for r in report["predictions"] if r["training_seed"] == score["training_seed"]
                and r["predictor"] == score["predictor"]]
        correct = sum(r["actual_helpful"] == r["predicted_helpful"] for r in rows)
        assert correct == score["correct"]
        rates = [fmean(r["predicted_helpful"] == outcome for r in rows
                       if r["actual_helpful"] == outcome) for outcome in (True, False)]
        balanced[score["predictor"], score["training_seed"]] = fmean(rates)
        assert score["balanced_accuracy"] == pytest.approx(fmean(rates), abs=1e-12)
    difference = fmean(balanced["full_gradient_alignment", seed] -
                       balanced["checkpoint_step", seed] for seed in (789, 2026, 31415))
    assert report["primary_mean_balanced_accuracy_difference"] == pytest.approx(difference, abs=1e-12)
