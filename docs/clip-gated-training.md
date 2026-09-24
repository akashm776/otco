# Frozen alignment-gated training pilot

Status (September 22): **implemented and locally validated; no GPU run started**.
The launcher is tested on the pinned base plus packaged source overlay,
including a four-arm toy rollout through the real trainer, exact prefix hashes,
gate boundary/audit checks, probe state preservation and Drive fail-closed tests.
These are fabricated offline checks, not model results. Colab notebook access
and reconnection both returned `Transport closed`; execution requires the
one-cell launcher below or restoring the Colab connection.

This continues the proposed four-arm comparison after the completed checkpoint
replay/evaluation-pool transfer. It is a training intervention, not another replay
of the 720 one-step observations. Existing downloaded results are untouched.

## Frozen protocol

- Three reused training seeds: 789, 2026, 31415. Four arms per seed: native
  baseline, always-on uniform-top-8, step-gated, alignment-gated.
- All arms start at pretrained CLIP and share 100 native-only updates. Their
  step-100 model, AdamW, scheduler, loader-generator and global RNG hashes must
  agree within a seed. No unsafe mid-epoch resume from saved checkpoints.
- Run 13 epochs / 1,001 updates, preserving the 3,850-update cosine LR horizon,
  B64, bf16, trainable layers, holdout exclusions and training data order.
- Same live-embedding, detached-support, normalized uniform-top-8 auxiliary;
  coefficient **0.06005351588542947**, no recalibration or OT-weight optimization.
- “Always on” means objective steps 100–1000, not from initialization.
- Step gate: objective step ≤175 after the common prefix, thus steps 100–175
  inclusive (**76 auxiliary updates**, including the update from state 175).
- Alignment gate: arithmetic mean of the **same 16 fixed B64 training-probe
  cosines** > **0.030847286945687016**. Caption seed 42, epoch 2; archived batch
  identity/caption SHA256 checked before training. Not a single-minibatch rule.
- Refresh at completed updates **100, 250, 500, 750** on the alignment arm's own
  current trajectory; hold each decision until the next refresh. These are new
  rollout choices, not part of the earlier predictor validation. No test/holdout
  loss, species label or retrieval score enters the gate.
- Preserve probe RNG, module modes, buffers and parameter gradients; hash-check
  model/optimizer/scheduler/RNG state around the probe. Nonfinite probes stop.

Primary readout: final (not best-epoch selected) canonical mean-direction R@1,
alignment minus native and alignment minus step, with all three seed differences
and their descriptive means in percentage points. Always-on is the third
comparator. Retain every epoch's two-direction canonical and all-caption
retrieval and species metrics. Report final species top-1 differences, active
update counts, probe seconds and elapsed arm time including evaluation/backups.

This is a **pilot on reused CUB test data and training seeds**, not untouched-test
confirmation or a significance test. Arms are update-count matched, **not
auxiliary-exposure or wall-clock matched**. A win alone would not isolate
adaptive timing from auxiliary dose. It does not establish novelty over existing
gradient-similarity auxiliary weighting; a prior-method and exposure-matched
comparison is required before that claim. No automatic threshold search,
extra arms or follow-on training is authorized by the launcher.

## Persistent execution

Use [the generated one-cell launcher](../colabs/clip_gated_training_drive_one_cell.py)
in a new A100 Colab code cell, **not Run all**, and approve the Drive mount.
It pins base commit `7e7cfea90b60415dc9561efbe97bcd383cc1580e`, verifies the source
overlay, then runs tests before training. This is twelve trajectories and a
multi-hour job, not a quick continuation of the downloaded native checkpoint.

Allow **25–30 GB free Drive quota** (30 GB recommended), plus **40 GiB free runtime
disk** for retained states, working copies and caches. The live persistent
checkpoints are estimated at about 17–18 GB; this is not a strict quota cap.
Drive-side retained versions/trash and unrelated files can consume additional
quota; the launcher never purges existing Drive data. The Drive FUSE
quota display is not trusted as a remote storage guarantee. Destination:
`MyDrive/OTCO/evaluation_transfer/clip_gated_training_<timestamp>/`.
The existing backup root is reused; run IDs distinguish this intervention.

Hybrid retention saves **24 permanent model/AdamW/scheduler/RNG checkpoints**:

- **12 final states** at update 1,001: every seed × arm.
- **3 shared prefix states** at update 100: one per seed. All arms must still
  reproduce the identical model, optimizer, scheduler and RNG prefix hashes.
- **9 alignment-arm states** at updates 250, 500 and 750: three per seed.

One additional `rolling_checkpoint.pt` is replaced at every existing checkpoint
boundary (100, 250, 500, 750, 1001) of the active trajectory. It contains seed,
arm, update, optimizer, scheduler, RNG and gate metadata; it is not an every-step
backup. The last rolling copy remains at completion, so there are **25 live .pt
files**, not 60 snapshots plus separate latest/best copies. Temporary replacement
files briefly need additional space. A failed replacement preserves the previous
verified Drive copy.

The trainer's latest/best files are written to one fixed-size local-only scratch
area reused across arms, outside the mirrored output tree. Best-epoch **metrics**
remain in the reports, but best-epoch **weights** are not retained on Drive.
All epoch metrics, probe measurements, gate decisions, configuration, source
and verification records are retained. Scientific settings and evaluations are
unchanged; intermediate replay for comparator arms is the deliberate trade-off.

Files are copied synchronously with SHA256 read-back. The output tree is the
deliverable; no second giant ZIP is made. This policy applies only to new runs;
it does not delete or migrate existing results. A failed copy stops the run.
Successfully backed-up files survive;
work since the last backup can still be lost. Checkpoints preserve gate state but
do not support automatic arbitrary mid-epoch resume (worker prefetch is absent).

Duplicate bundles already on Drive block automatic reruns. Completion verifies
all 24 permanent states plus the rolling copy, rejects extra checkpoint copies,
and reads back the entire Drive tree. Wait for
**DRIVE_SYNC_COMPLETE**, printed only after `drive.flush_and_unmount()` succeeds,
then confirm the folder in Drive before releasing the runtime. A running cell
does not guarantee Colab VM lifetime.

Regenerate the launcher with `python scripts/build_clip_gated_training_cell.py`.
