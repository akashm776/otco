# Early synthetic-negative pulse: causal pilot

[Project](../README.md) · [Readiness diagnostic](clip-warmup-readiness.md)

## Question and preregistered decision

Does briefly adding synthetic-negative pressure early in fine-tuning improve what native CLIP would learn anyway? The preceding single-seed diagnostic found weak positive projection-head alignment at update 100 (6/6 selected shuffled batches), near-zero mean alignment at 250, and negative means at 500–1,001. This motivates a candidate interval; it does not establish an optimal window or a general OTCO principle.

Compare **baseline**, **uniform-top-8 synthetic**, and **hardest-real** arms. Apply the auxiliary only for objective-step indices **100–199**: the first pulse update takes the model from 100 to 101 completed updates; the last takes it from 199 to 200. No ramp or adaptive gate. Then return to native-only training through **1,001 updates / 13 epochs**, keeping the original **3,850-update cosine schedule**, optimizer, seed 42, data exclusion, captions, and batch order.

This is a uniform-synthetic intervention, not a test of OT weighting. Native CLIP trains the logit scale; the auxiliary uses its detached value. Positive-excluded raw-cosine top-8 support is stop-gradient, uniform barycentric image paths remain live, and the auxiliary is the existing text-to-image relative-denominator increment. Hardest-real duplicates the hardest real negative in that denominator.

## Strength matching and exact controls

At the common update-100 state, use four disjoint B64 **training** batches, selected with NumPy generator seed 20260911 from the training complement. Use epoch-2 training captions, train mode, and the same bf16 path as the intervention. Never calibrate on the excluded diagnostic holdout or test set.

For each auxiliary, freeze `alpha = 0.10 * mean(native gradient norm) / mean(auxiliary gradient norm)`. Norms include **all trainable parameters**, before clipping (including native logit-scale derivatives; auxiliary scale derivative is zero). The 10% target is a conservative, predeclared pilot choice, not tuned using outcomes. This matches the ratio of mean norms on calibration batches, not every future batch, gradient direction, or AdamW update magnitude. Log realized full-parameter ratios for all 100 pulse updates; the ordinary total-gradient clip remains active.

Rerun the identical seeded baseline prefix per arm. At update 100, require exact canonical hashes of model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNGs, loader generator, epoch, and consumed-batch position. Fail before treatment if any differ. Save the common full model/optimizer/RNG snapshot. Branching uses verified replay rather than restoring a partially consumed multiprocess loader: the checkpoint does **not** serialize worker prefetch or the iterator and is not an arbitrary mid-epoch resume file. Calibration preserves model modes/RNGs and `.grad`, and never iterates the training loader.

## Outcomes, diagnostics, and limits

Primary outcome: final update-1,001 canonical average R@1 difference versus baseline. Secondary: species top-1 (watch for forgetting), retrieval directions and all-caption retrieval, immediate post-pulse differences at 200, and the full epoch curves. Report all arms and signed differences, not just the best checkpoint. The test set has already been examined in earlier experiments, so this is exploratory evidence, not an untouched confirmatory test.

At 100, 200, and 1,001, repeat fixed held-out per-query embedding and matched projection/batch probes, and evaluate retrieval/species performance. These measurements preserve RNGs and model modes. The diagnostic reports' inherited scheduled-alpha field describes the reference baseline config, **not** the applied pulse; `pulse_audit.json` is authoritative for intervention timing and coefficients. The original config remains native baseline; the separately saved `protocol.json` specifies the injected objective.

One seed cannot establish reliable improvements. Even a positive early-pulse effect does not show early timing is better than a matched later pulse; that is a subsequent timing-control experiment. A null result applies to this interval/strength/model, not all curricula. Repeat promising findings over seeds, then compare OT-generated negatives and a non-CLIP model.

Outputs: exact-prefix hashes; training-only calibration identities, captions and norms; pulse timing and realized gradient ratios; three diagnostic snapshots per arm; epoch and stage evaluations; raw performance JSON and PNG/SVG curves; source/environment manifest; complete-result ZIP. Large encoder/optimizer checkpoints remain in Colab's checkpoint directory and are not included in the result ZIP.

```bash
python -m src.clip_early_pulse \
  --output-directory outputs/clip_early_pulse_run1 \
  --checkpoint-directory checkpoints/clip_early_pulse_run1
```

[Protocol](../configs/clip_early_pulse.yaml) · [Implementation](../src/clip_early_pulse.py) · [Tests](../tests/test_clip_early_pulse.py) · [Colab runner](../colabs/run_clip_early_pulse.py)

## Persistent backups in Colab

The initial run's temporary Colab filesystem disappeared before completion could be verified. Mount Google Drive using `from google.colab import drive; drive.mount('/content/drive')` and approve the Google authorization prompt. The independent [backup worker](../colabs/backup_clip_run.py) can run alongside an already-started experiment without changing training code or RNGs:

```bash
python -u colabs/backup_clip_run.py \
  --output-directory /content/otco_outputs/RUN_ID \
  --checkpoint-directory /content/otco_checkpoints/RUN_ID
```

It waits for a real Drive mount, then checks every 60 seconds and mirrors results into `MyDrive/OTCO/early_pulse/RUN_ID/`. Each copied file is checksum-verified before publication; `backup_manifest.json` records the inventory and any errors. Incomplete JSON files are retried. Stage tensors are copied only after their completion marker. The common update-100 checkpoint and final checkpoint of each completed arm are also backed up (several GB total); actively overwritten epoch checkpoints and redundant best-model copies are excluded. This preserves completed work, but does not implement arbitrary mid-epoch resume. Drive authorization is required: a waiting worker alone is **not** an active persistent backup. A Drive mount/write failure is logged and retried independently of training.
