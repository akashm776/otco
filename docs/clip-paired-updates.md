# Does the auxiliary improve an actual optimizer update?

[Project](../README.md) · [Early-pulse findings](clip-early-pulse.md)

**Status: completed and audited.** Run `clip_paired_updates_20260912T042828_516737Z`, pinned source `d22b9d7cec94d5b044b814d774911cab5e37d560`, A100 40 GB. All 96 branches completed. The protocol below was committed before observing the results. This is a mechanism check, not a curriculum search or a new training rollout.

## Findings

**The synthetic auxiliary adds a small held-out benefit at update 100 and a smaller disadvantage at update 1,001.** Each entry below compares the treatment's post-update loss with its paired native-only post-update loss; negative is beneficial. The primary score averages the three fixed shuffled partitions.

| Checkpoint | Auxiliary | Mean extra held-out loss | Range across 16 trials | Beneficial trials |
|---|---|---:|---:|---:|
| 100 | Uniform top-8 | −0.00002563 | −0.00003155 to −0.00001899 | 16/16 |
| 100 | Hardest real | +0.00000543 | −0.00002024 to +0.00003565 | 6/16 |
| 1,001 | Uniform top-8 | +0.00000272 | +0.00000042 to +0.00000581 | 0/16 |
| 1,001 | Hardest real | +0.00000193 | −0.00001832 to +0.00002610 | 7/16 |

![Actual one-step effects; both panels use the same loss scale.](figures/clip_paired_update_effects.png)

Intuitively: imagine copying the same learner three times and giving each copy the same practice batch. One practices normally; another also uses a synthetic negative; the third adds a real negative. We then quiz them on other examples. The synthetic nudge helps every paired early trial on the primary score, but slightly weakens every later trial. We reset the copies before the next batch, so these effects do not accumulate into a training trajectory.

The magnitude matters. At update 100, mean native-only held-out loss reduction is **0.00290888**, versus **0.00293451** with synthetic pressure: about **0.88% more reduction**. At update 1,001, the corresponding reductions are **0.00018800** and **0.00018528**: about **1.45% less reduction**. These are ratios of mean loss reductions, not retrieval-accuracy percentages. Both methods improve from their starting checkpoints on average; “hurts later” means worse *than the native-only update*.

### Supporting checks and interpretation

- Early synthetic benefit is present in all 16 trials in each individual partition, including sequential. Later, each partition's mean effect is positive, but shuffle-123 has 4/16 individually beneficial trials and sequential has 5/16. Thus “0/16 later” refers to the pre-specified three-shuffle average, not every constituent partition or example.
- Mean extra **training-batch** loss is positive for both auxiliaries at both stages. Synthetic values are +0.00009634 early and +0.00002107 later. The early held-out benefit is not simply a larger drop on the optimized batch; this is compatible with a regularizing effect, not proof of its mechanism.
- Native/auxiliary **full-gradient cosine on these training batches** averages +0.0541 early and +0.0227 later for synthetic, versus +0.2659/+0.2673 for hardest real. Stronger raw alignment does not guarantee a better held-out AdamW update. These training-batch values are not the previous held-out shared-head probe, whose later alignment was negative.
- Synthetic actual-update differences have norm about **2.61% early / 2.03% later** of the native update norm; update-direction cosine stays above 0.999 on average. Small directional changes can have measurable but tiny effects. Fixed coefficients yield mean weighted gradient ratios **0.1090 / 0.0602** for synthetic, versus **0.1015 / 0.1018** for real; pressure is not matched across stages.
- Both checkpoints reproduce the archived feature hashes exactly. No-update re-encoding and first-native-update replay are exact. All 96 optimizer-reset checks pass; frozen parameters and source states remain unchanged. The downloaded ZIP matches the verified Drive SHA256, and an offline audit recomputes every recorded loss difference and summary. These checks establish within-run consistency, not cross-hardware numerical robustness.

This answers the narrow question **yes, useful synthetic pressure can depend on the state at which it is applied** in this particular CLIP/CUB experiment. It does not locate the transition between updates 100 and 1,001, separate learning-rate/momentum from representation changes, or establish a general rule. It also tests a uniform synthetic construction, not the incremental value of OT weighting. One seed and a reused diagnostic holdout remain important limitations; tiny effects merit replication. The near-null [100-update pulse pilot](clip-early-pulse.md) is still the relevant longer-horizon evidence.

**Next check, prepared but not run:** [one-cell replication on baseline seeds 123 and 456](clip-paired-seed-replication.md), keeping the two states, diagnostic inputs and auxiliary coefficients fixed. Replicate the sign pattern before using a denser checkpoint scan to choose an activation window. Any selected curriculum then needs an independent multi-seed downstream evaluation.

[All 96 branch records, summaries, provenance and audit](../experiment_results/clip_paired_updates_2026-09/README.md) · [Reproduce the figure](../scripts/plot_clip_paired_updates.py)

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
