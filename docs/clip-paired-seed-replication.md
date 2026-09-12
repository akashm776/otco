# Paired-update replication: two new training seeds

[Project](../README.md) · [Completed seed-42 findings](clip-paired-updates.md)

**Status: prepared for a user-launched overnight Colab run; no new GPU results yet.** Seeds and protocol are fixed before seeing either new result.

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
