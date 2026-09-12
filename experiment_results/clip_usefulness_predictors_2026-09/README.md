# Exploratory usefulness predictor screening

Leave one training seed out: fit on ten states, evaluate five states from the remaining seed.
Synthetic uniform-top-8 only. Fixed diagnostic batches are averaged within each seed/state.

| Predictor | Seed 42 correct/5 | Seed 123 correct/5 | Seed 456 correct/5 | Mean seed balanced accuracy | Correct outside ±1e-6 |
|---|---:|---:|---:|---:|---:|
| majority_baseline | 1 | 2 | 2 | 0.500 | 4/12 |
| checkpoint_step | 5 | 1 | 3 | 0.611 | 9/12 |
| full_gradient_alignment | 4 | 3 | 4 | 0.764 | 9/12 |
| projection_gradient_alignment | 4 | 3 | 4 | 0.764 | 9/12 |
| weighted_gradient_ratio | 4 | 2 | 3 | 0.653 | 8/12 |
| actual_update_difference_ratio | 4 | 2 | 3 | 0.653 | 8/12 |
| actual_update_cosine_to_native | 4 | 2 | 3 | 0.653 | 8/12 |

## Interpretation limits

- Compare every candidate with the majority and checkpoint-step baselines; do not select a winning rule as a validated curriculum.
- Threshold and direction are fitted using training seeds only. Actual held-out effects never enter predictor features.
- Near-zero sensitivity excludes |effect| ≤ 1e-6 only from the secondary score; it does not change training labels or fit.
- Only three seeds and fifteen dependent states; no significance tests or confidence intervals. The same holdout and diagnostic inputs are reused.
- Actual-update features require trial updates and resets; these are retrospective diagnostics, not free pre-update signals.
- This analysis was proposed after inspecting stage outcomes. Leave-one-seed-out is a screening check, not untouched prospective validation.
- If a feature looks promising, preregister it for new seeds/independent evaluation before a curriculum training experiment.

Outputs: `states.csv`, `predictions.csv`, `analysis.json`. Preserve these with the source ZIP.
