# Offline policy robustness diagnostic

## Contract

Input: the complete policy-test export `clip_policy_test_20260930T050231_622869Z`,
as a directory or its downloaded ZIP. The exact backup manifest and source
bundle are pinned. The earlier partial marginal run in `failed runs` is not an
input and must not be counted as additional evidence.

Operation: authenticate every file, require the full 12-set/48-branch grid,
recompute update-50 paired losses from the original two sets of eight reporting
batch losses, and check them against the recorded pair and seed summaries.
The diagnostic never loads a model or executes any embedded source code.

Output: `analysis.json`, `REPORT.md`, and a checksummed `completion.json` in a
new directory. Source data are read-only. Existing outputs are never overwritten.
No captions, images or weights are copied. Standard library only, zero GPU or
optimizer updates, no dependency installation.

## Fixed analysis family

For gated minus native, sustained, and exposure-matched random timing:

1. Equal-weight stream-within-state, state-within-seed, seed-level means.
2. Both reporting partitions separately, including seed-level values.
3. Leave-one-seed, leave-one-stream, and leave-one-checkpoint sensitivity.
4. All eight partition × checkpoint × stream strata, with all seed values.
5. Sixteen paired reporting-batch deletions: remove one fixed batch position
   from one partition, consistently across all arms/states/streams/seeds, while
   retaining equal weight for the two partitions. These are influence checks,
   not independent replications or a new validation set.
6. Linear partition-weight sensitivity and its zero crossing, if any. Original
   weights stay 50/50; the diagnostic does not optimize them.
7. Exhaustive enumeration of the eight sign patterns of the three seed-level
   effects. The two-sided tail fraction is only a reference under independence
   and sign symmetry of seed effects under a null. It is NOT a design-based
   randomized-policy test here, and no confirmatory significance or confidence
   interval is claimed. Do not substitute the 12 correlated pairs or 16 batches
   for independent seeds.

This diagnostic was selected after inspecting the policy results; it is not
preregistered or an independent test. The two reporting partitions reuse the
same 512 images but change their contrastive negative pools. Stored batch losses
cannot reconstruct new partitions or image-level outcomes; embeddings or final
branch checkpoints would be necessary for that separate experiment.

## Run locally

```bash
.venv/bin/python scripts/analyze_clip_policy_robustness.py \
  --input /Users/akashmittal/Downloads/clip_policy_test_20260930T050231_622869Z \
  --output experiment_results/clip_policy_robustness_2026-09
```

A ZIP path also works. Input identity/checksum failures stop before an output
directory is created. No automatic reruns or threshold tuning follow a result.

## Structural update to make after analysis

Distinguish (a) stable early/later native-training contrasts, (b) improvement over
always-on pressure, and (c) a timing-specific benefit beyond dose. Report all
three even when one looks favorable. Inspect whether partition and stream signs
agree before describing the small timing contrast as robust. A negative loss
effect is not itself evidence of improved retrieval.

## Completed local result — September 30

The source directory and ZIP yield byte-identical analysis artifacts after
verification of all 334 manifest-listed files. The saved report is
`experiment_results/clip_policy_robustness_2026-09/REPORT.md`.

Question: is the small gate-versus-random timing effect stable to the available
paired sensitivity checks? Working hypothesis: a broadly useful timing signal
should not depend on a single reporting grouping or continuation stream.

Result: the primary −0.00003434 survives all three leave-one-seed-out checks,
but its sign reverses between reporting partitions and between streams.
Removing zero-based batch 5 (the sixth batch) from reporting partition
2026092404, consistently across the paired branches, changes the aggregate to
+0.00003236. Each of the three seed summaries then becomes positive. The other
15 batch-deletion checks keep the negative aggregate sign. This is influence
analysis, not evidence that this batch is invalid or should be excluded.

Structural update: the existing average timing advantage is fragile; it is not
supported as a robust gate benefit. Do not tune the threshold or reporting
weights, or discard the influential batch to improve the result.

Next question: would the fixed policy's effect retain its sign across additional
preselected reporting partitions? Existing batch summaries cannot answer that;
a separate bounded evaluation would need final branch embeddings or states.
