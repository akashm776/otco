# Policy-test robustness diagnostic

Completed offline: zero optimizer updates. Source inputs authenticated; no source files modified.

Source: `clip_policy_test_20260930T050231_622869Z`. Verified 334 manifest-listed files.

## Endpoint comparisons

Gated minus comparator at update 50; negative loss favors gating.

| Comparator | Mean loss difference | R@1 difference (pp) | Pair wins / ties / losses |
|---|---:|---:|---:|
| native | -0.000485803 | -0.056966 | 6 / 0 / 6 |
| sustained | -0.000532366 | +0.105794 | 9 / 1 / 2 |
| random_matched | -0.000034341 | +0.016276 | 6 / 1 / 5 |

## Primary timing contrast: sensitivity

| Check | Loss difference |
|---|---:|
| Reporting partition 2026092403 | +0.000019442 |
| Reporting partition 2026092404 | -0.000088124 |
| Omit seed 789 | -0.000045160 |
| Omit seed 2026 | -0.000037095 |
| Omit seed 31415 | -0.000020768 |
| Omit stream 2026093002 | -0.000143419 |
| Omit stream 2026093003 | +0.000074737 |
| Omit checkpoint 100 | +0.000004787 |
| Omit checkpoint 500 | -0.000073468 |

Paired single-batch deletion range: -0.000088975 to +0.000032356; 15/16 deletions retain a favorable sign.

Partition-A weighting zero crossing: 0.8192562986670849. The original analysis weights the partitions equally; this diagnostic does not change those weights.

Seed sign-symmetry reference: 2/8 patterns are at least as extreme in absolute mean (fraction 0.250). This is NOT a confirmatory randomized-treatment p-value or proof of no effect.

## Interpretation

Read the partition and stream effects alongside the averaged primary. A negative average does not establish robustness if these signs disagree. A timing advantage does not by itself establish improvement over native training. Keep all fixed strata; do not select favorable ones.

## Limits

- Post-hoc diagnostic of reused states, seeds and reporting images, not independent confirmation.
- Both partitions contain the same 512 images; differences reflect contrastive batch grouping.
- Batch deletion is a paired influence diagnostic, not a new image-level or partition bootstrap.
- No confidence interval is claimed from only three reused seeds. Sign symmetry is an explicit assumption, not a design guarantee.
- New reporting partitions cannot be evaluated from batch-loss summaries; final embeddings or saved branch states would be needed.
- Loss changes and retrieval changes are separate endpoints. No threshold or policy selection is performed.

Full paired values, all strata and influence checks are in `analysis.json`. No captions, images, or model weights are exported.
