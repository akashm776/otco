# Paired-update replication: two new training seeds

[Project](../README.md) · [Completed seed-42 findings](clip-paired-updates.md)

**Status: completed and audited.** Run `clip_paired_seeds_20260912T050100_795138Z`, pinned source `2c4c610576c5167114bb6778cef526f0a8d64df8`, A100 40 GB. All **192 new branches** completed; seed 42 remains the historical reference. Seeds and protocol were fixed before seeing either new result.

## Findings: early benefit repeats, later harm does not reliably repeat

The uniform-top-8 synthetic auxiliary improves the pre-specified held-out score relative to a matched native-only update in **16/16 early trials in each of three seeds**. At the later state, its effect is smaller and changes sign across seeds. The original seed-42 result therefore did not establish a universal late-stage disadvantage.

| Training seed | Early mean extra loss (×10⁻⁶) | Early beneficial trials | Later mean extra loss (×10⁻⁶) | Later beneficial trials |
|---|---:|---:|---:|---:|
| 42 — previous run | −25.63 | 16/16 | +2.72 | 0/16 |
| 123 | −29.56 | 16/16 | −4.87 | 15/16 |
| 456 | −13.80 | 16/16 | +1.43 | 3/16 |

Negative means lower held-out loss **than the paired native-only step**, not an accuracy gain and not necessarily a decrease from the starting checkpoint. Counts refer to the primary average across three shuffled partitions.

![All paired inputs and per-seed means on common vertical scales.](figures/clip_paired_seed_replication.png)

Intuitively, the synthetic negative supplies a small useful nudge early across all three learning trajectories. Later, the nudge is much weaker and whether it helps depends on the state the learner has reached. This supports **a more consistent early one-step benefit**, not “always harmful later” or a universal switch-off update.

### Effect size and controls

The early synthetic effect adds about **0.88%, 0.78%, and 0.38%** to the mean native-only held-out loss reduction in seeds 42, 123, and 456 respectively. These are ratios of small loss changes, not retrieval-accuracy improvements. The earlier near-null [100-update pulse rollout](clip-early-pulse.md) still limits claims about long-term utility.

Seed 123 illustrates the distinction between relative and absolute benefit: its later native-only step **increases** held-out loss by +0.00028763 on average; native + synthetic increases it by +0.00028276. The auxiliary helps by reducing that increase, not by improving on the initial checkpoint. In seeds 42 and 456, both later branches reduce initial loss, but the synthetic branch reduces it slightly less.

The hardest-real control has a positive mean incremental loss in all six seed/stage combinations on the primary score:

| Training seed | Real: early mean extra loss (×10⁻⁶), beneficial trials | Real: later mean extra loss (×10⁻⁶), beneficial trials |
|---|---:|---:|
| 42 | +5.43; 6/16 | +1.93; 7/16 |
| 123 | +10.56; 3/16 | +8.39; 4/16 |
| 456 | +4.51; 5/16 | +4.26; 3/16 |

Its full-gradient alignment remains stronger than synthetic alignment, so raw alignment alone is not a sufficient predictor of the held-out AdamW effect. Both auxiliaries produce less native training-batch loss reduction than the native-only branch on average at every seed/state; early synthetic transfer is not simply better fitting of the optimized batch.

### Consistency, limitations and next question

- Early synthetic benefit holds for every trial in each individual partition in all three seeds. Later partition means are positive for seed 42 and negative for seed 123. Seed 456's three shuffled means are positive, while its sequential mean is slightly negative (−0.45 ×10⁻⁶). Thus the weaker later effect is also sensitive to evaluation grouping.
- Fixed diagnostic identities/captions and partitions match the original hashes exactly, with no overlap between selected training inputs and the holdout/calibration examples. The new seed configs, 13-epoch stops and original 3,850-update scheduler horizon are verified. The six seed/state feature hashes are distinct, and recorded no-update/replay, optimizer-reset and frozen-state checks all pass.
- The downloaded ZIP contains 54 files and **192 new branch records**. Its offline audit checks file integrity and recomputes recorded per-batch loss differences and summaries, including the 96 historical rows. Package versions match seed 42. Checkpoints are not in the ZIP: saved hashes match the inventories, but GPU checks are recorded runtime evidence, not independently rerun offline.
- There are **three training seeds**, not 48 independent seeds at a stage. The same holdout has been reused across experiments; these are small one-step CLIP/CUB effects, not an untouched-set confirmation, a demonstrated curriculum gain, or an OT-weighting advantage. No significance claim follows from the 16/16 batch counts alone.

**Follow-up completed:** [five checkpoints across the same three seed trajectories](clip-paired-intermediate.md) show non-monotonic, seed-dependent effects; all historical endpoints replayed exactly. The [subsequent frozen-rule new-seed test](clip-usefulness-prospective.md) modestly favors alignment on balanced accuracy but ties ordinary and non-near-zero accuracy. Do not choose a universal switch-off step from the original endpoints. Learning rates, optimizer moments and relative auxiliary pressure change with stage, so this does not isolate representation geometry.

[All results, provenance and reproducible audit](../experiment_results/clip_paired_seeds_2026-09/README.md) · [Plotting script](../scripts/plot_clip_paired_seeds.py)

## Question and design

Does the small early synthetic benefit / later disadvantage repeat across independently seeded native CLIP trajectories? The original seed-42 result was consistent across its 16 diagnostic training batches, but those batches were not independent training seeds.

The single Colab cell executes these stages sequentially:

1. Train native baseline **seed 123** through 1,001 updates (13 epochs), retaining the original 3,850-update learning-rate schedule; save model/AdamW states at 100 and 1,001.
2. Restore those states for **96 paired one-step branches**: 2 states × 16 batches × native-only / uniform-top-8 / hardest-real.
3. Repeat both stages for **seed 456**: another baseline trajectory and 96 branches.
4. Verify completion and create one combined results ZIP, with per-seed summaries and a three-seed comparison plot including the already completed seed 42.

Only native baseline trajectories are trained. The auxiliary branches each take one update and reset, never accumulating into auxiliary training rollouts. There are **192 new branch records** and two new training seeds; including the historical reference gives three training seeds, not 48 independent replicates per stage.

## Fixed controls

The [paired protocol](../configs/clip_paired_updates.yaml) and coefficients remain unchanged: synthetic **0.06005351588542947**, real **0.3753711620568176**. No recalibration or holdout-based tuning occurs. Reconstruct and exclude the original 256 calibration source indices.

Training seeds change the loader order and epoch-specific training caption sampling. Diagnostic caption sampling stays at **seed 42, epoch 2**, with the original 16 selected B64 training batches and 1,024 held-out canonical captions. Hash checks require the serialized diagnostic image/caption identities and held-out partitions to match the original experiment exactly. The fixed probes use the same inputs at both states and across seeds.

At each new seed's two checkpoints, save a held-out feature hash during training; later paired evaluation must reproduce it exactly. Preserve the existing optimizer-reset, native-update replay, frozen-parameter, source-state and loss-arithmetic controls. The new early snapshot includes its training seed; the late snapshot's saved training config must agree. The seed-42 reference is historical and identified separately in the combined summary; no new run of seed 42 is implied.

Report each seed's mean, range and beneficial-trial count. Assess whether the early/later sign pattern repeats **within each seed**, not only in a pooled mean. This is CLIP/CUB seed robustness, not architecture transfer, an untouched-holdout confirmation, or evidence that a curriculum improves long-term retrieval. Learning rates, optimizer moments and relative auxiliary strength still differ between stages.

## One-cell overnight run — no Drive storage

Use an A100 Colab runtime. Paste the entire [one-cell launcher](../colabs/clip_paired_seeds_one_cell.py) into one code cell, or run the prepared notebook cell headed **“Overnight: seeds 123 + 456 — NO DRIVE”**. Run only that cell, not the earlier Drive-based experiment cells. Its pinned [foreground runner](../colabs/run_clip_paired_seed_replication.py) installs dependencies, runs tests, executes both seeds, prints progress every 25 baseline updates and every paired branch, verifies the result counts, and requests a browser download of the combined ZIP.

**Nothing is mounted or saved to Google Drive.** Model/dataset caches, checkpoints, logs and reports use local `/content/` paths. Allow at least 15 GiB of local runtime disk space. The download excludes model checkpoints and dataset tensors, so it contains the much smaller reports, manifests, caption identities, training metrics and plots needed for analysis. The checkpoints remain only on the runtime disk.

Keep the Colab tab open, the computer awake, and browser downloads allowed. The launcher prints `LOCAL_ARCHIVE:` and its SHA256 for manual retrieval through Colab's Files panel if automatic downloading is blocked. Rerunning the one-cell launcher after successful completion re-downloads the existing ZIP if it is still on the runtime, without retraining. A handled execution error produces an `interrupted_or_failed.zip` with available reports and logs, not a false completion message. After an incomplete attempt, a fresh rerun starts fresh training; it does not resume a partial trajectory.

**Active training does not guarantee indefinite runtime availability.** Colab enforces a maximum VM lifetime and variable resource limits, and a destroyed VM loses its local files. A local ZIP or partial report cannot protect against that loss before downloading. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html). There is deliberately no external backup under the requested no-Drive policy.

[Study config](../configs/clip_paired_seed_replication.yaml) · [Baseline/checkpoint orchestration](../src/clip_paired_seed_replication.py) · [Tests](../tests/test_clip_paired_seed_replication.py)
