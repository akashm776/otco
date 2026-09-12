# Does the auxiliary improve an actual optimizer update?

[Project](../README.md) · [Early-pulse findings](clip-early-pulse.md)

**Status: implemented; GPU results pending.** This protocol is fixed before observing its results. It is a mechanism check, not a curriculum search or a new training rollout.

## Why this experiment

The early synthetic pulse gave a tiny final canonical R@1 gain (+0.017 percentage points), a species loss (−0.259 points), and no consistent overall advantage. Projection-head gradient agreement alone did not predict added benefit. AdamW momentum, per-coordinate scaling, different parameter-group learning rates, full encoder gradients, and total-gradient clipping can all separate that proxy from the actual update.

Ask: **Does adding the auxiliary reduce loss on other examples more than a native-only update from the same state?**

## Fixed design

Use the **baseline** model and optimizer checkpoints after 100 and 1,001 updates from `clip_early_pulse_20260911T233840_978774Z`. The latter is later in the short pilot, not the end of the original 3,850-update training horizon. Preserve each checkpoint's own optimizer moments, parameter-group learning rates, scheduler state, and trainable subset. Thus the state comparison includes learning-rate/momentum changes; it does not isolate representation geometry alone.

Choose 16 disjoint B64 training batches with NumPy seed 20260912, excluding both the 1,024-example diagnostic holdout and the 256 calibration examples. Fix epoch-2 caption sampling across both states so the inputs are identical. These images belong to the training pool and are not novel images to the checkpoints. Never iterate or advance the original training loader.

For every state/batch, restore the source model parameters and optimizer/scheduler state before each branch: native-only, native + uniform-top-8 auxiliary, and native + hardest-real auxiliary. Reset branch RNGs identically. Apply exactly one training-path bf16 update, total-gradient norm clipping at 1.0, AdamW, the usual logit-scale clamp, and the scheduler step. Keep the previous training-only coefficients **0.06005351588542947** and **0.3753711620568176** fixed. They are not retuned at the later checkpoint or continuously pressure-matched.

That gives **96 branches: 2 checkpoints × 16 training batches × 3 arms**. No branch inherits another branch's update.

## Primary and supporting measurements

Re-encode all 1,024 held-out images and canonical captions after every branch. Evaluate native symmetric CLIP loss using the four existing B64 partitions. The primary score is the mean over the three shuffled partitions; preserve sequential and individual-batch scores too. Encoder inference is fp32; similarity/loss arithmetic is fp64 to resolve small differences. Processed inputs are cached once in CPU memory without changing padding, captions, or batch membership.

For each treatment, report `heldout_loss_after_treatment − heldout_loss_after_native_update`. **Negative means incremental benefit.** Also report each branch's change from the pre-update checkpoint, same-training-batch loss changes under the same fp32/fp64 evaluation path, full-trainable and projection-head gradient alignment, and the actual parameter-update direction, magnitude, and difference from native (including projection/encoder/logit-scale group norms).

Report every paired input, the mean, median, range and number of beneficial trials, not just favorable examples. These 16 inputs are conditional paired probes, not 16 training seeds. The held-out examples are reused across partitions, stages and previous experiments; they are not independent replicates or an untouched confirmation set. No test-split retrieval results select this protocol or its coefficients.

## Validity and interpretation

Verify checkpoint file checksums against the Drive backup inventory, exact step-100 prefix hashes, and reproduction of the archived held-out feature hashes at both states. Re-encode without an update to check determinism. Repeat the first native update at each checkpoint and require identical parameter deltas and held-out losses. Verify optimizer reset before each branch, source-state immutability, and unchanged frozen parameters afterward. Tests compare the one-step implementation against the existing trainer.

Early-only incremental benefit would motivate a timing hypothesis, not prove long-term gains. Training-only benefit would suggest limited transfer beyond the optimized batch. No consistent incremental benefit would weaken the justification for tuning activation windows for this construction and strength. Any promising effect still needs a multi-seed training test; a one-step null does not rule out delayed or cumulative benefits.

## Run and outputs

Use an A100 Colab runtime with Google Drive mounted. The [foreground runner](../colabs/run_clip_paired_updates.py) requires a full immutable `--source-commit`. It retrieves and verifies the two saved baseline checkpoints, tests the code, prints each branch's result, backs up partial reports to `MyDrive/OTCO/paired_updates/<run-id>/`, then verifies and downloads a final ZIP. No new encoder checkpoints are produced; original inputs are never overwritten.

```bash
python -m src.clip_paired_updates \
  --source-directory /path/to/clip_early_pulse_20260911T233840_978774Z \
  --output-directory outputs/clip_paired_updates_run1
```

The source directory must contain the backed-up `results/`, `checkpoints/`, and `backup_manifest.json`, not just the downloaded results ZIP. Outputs include selected input identities/captions, protocol and checkpoint provenance, all 96 branch records with per-held-out-batch losses, reset checks, summaries, plots, and a source/environment manifest. Dataset tensors are not included in the report ZIP.

[Protocol](../configs/clip_paired_updates.yaml) · [Implementation](../src/clip_paired_updates.py) · [Tests](../tests/test_clip_paired_updates.py)
