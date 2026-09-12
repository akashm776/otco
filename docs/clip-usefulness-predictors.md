# Next analysis: predictors of synthetic usefulness

Completed and verified locally; no new training, GPU, dependencies or Drive access.

Alignment (full or projection) classified **11/15** held-out states correctly, mean seed balanced accuracy **0.764**, versus step **9/15 / 0.611**. Excluding effects within ±1e-6, both scored **9/12**. This is a screening clue, not convincing evidence of superiority. The [frozen-rule new-seed test is now completed](clip-usefulness-prospective.md). [Original report, 15 states and all 105 candidate/fold predictions](../experiment_results/clip_usefulness_predictors_2026-09/README.md) are archived with source/script hashes and a file inventory; no new fit was performed during archiving.

```bash
cd /Users/akashmittal/Projects/otco
python3 scripts/analyze_clip_usefulness_predictors.py
```

Input: newest `clip_paired_intermediate_*_complete.zip` in Downloads (or specify `--archive PATH`). Output: a new sibling folder ending `_predictor_analysis`, containing a Markdown report, state/prediction CSVs, and JSON with fitted rules and input/script hashes. Existing output folders are never overwritten; use `--output-directory PATH` for another run.

Use synthetic uniform-top-8 only, averaging the 16 fixed batches within each of fifteen seed/checkpoint states. Fit each single-feature threshold on two training seeds, then evaluate on the third; repeat for all three held-out seeds. Candidate features: full/projection gradient alignment, weighted gradient ratio, actual update difference ratio and update cosine. Compare with training-majority and checkpoint-step baselines. Report every candidate, not an automatically selected winner.

Primary labels use the sign of incremental held-out loss; a fixed ±1e-6 sensitivity flag reports accuracy excluding near-zero effects without refitting. No held-out-loss-derived predictor is used. Tests check threshold direction, fold counts and that changing held-out labels cannot alter fitted rules.

This is retrospective screening on three seeds with reused evaluation inputs, not untouched validation or a deployable curriculum. Actual-update features require trial updates. Any promising predictor needs prospective validation on new seeds/independent evaluation before a training intervention.
