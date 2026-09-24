# Saved-checkpoint diagnostic v1

Paste all of `colabs/clip_checkpoint_diagnostic_drive_one_cell.py` into one Colab
code cell, on an A100 40 GB runtime. Approve the normal Drive mount. The cell
clones the pinned base and applies a checksummed embedded overlay: no git push
or dependency on another notebook's Python imports.

Required input is the original folder
`MyDrive/OTCO/evaluation_transfer/clip_gated_training_20260922T114312_608272Z`.
Twelve checkpoint SHA256 values are embedded from its audited backup manifest.
The source files are read-only. The newer exposure-matched run is not silently
substituted. Missing inputs, version mismatches, failed tests, or an existing
diagnostic run cause a stop, not automatic retraining/resume.

If that Drive folder was deleted, use the updated cell with the recovery archive.
`python -m scripts.build_clip_checkpoint_recovery_archive` reads the nine original
download ZIPs without changing them, verifies all 12 checkpoint pins, and creates
`output/recovery/clip_checkpoint_diagnostic_inputs.zip` (about 8.31 GB).
Upload that ZIP to `MyDrive/OTCO/evaluation_transfer/` using Drive's upload UI,
or place it in the Colab runtime at `/content/clip_checkpoint_diagnostic_inputs.zip`.
Colab cannot read the Mac's Downloads directory directly. No archive upload is
needed when the original Drive folder is still present.

The fallback verifies the archive's exact member set, provenance, sizes, and
checkpoint SHA256 pins, then extracts only these inputs to local runtime disk.
It does not recreate the original large Drive folder or rerun the old training.
Original manifests are preserved, with a separate `RESTORED_SUBSET_MANIFEST.json`
explicitly documenting that only diagnostic inputs were restored. Changed existing
local files are refused. Previously verified extraction files may be reused;
this is input recovery, not automatic resumption of an experiment.

## Scientific question and fixed budget

Does the weak rollout result reflect failure to predict immediate effects on the
gate's own trajectory, transient benefits, or a loss/retrieval mismatch?

- Seeds: 789, 2026, 31415. Saved updates: 100, 250, 500, 750.
- Step 100 is the shared native prefix. Later states are `alignment_gated`.
- Three branches/state: native for 50 updates; uniform-top8 auxiliary on only
  the first update (pulse); auxiliary on all 50 updates (sustained).
- Auxiliary coefficient remains **0.06005351588542947**, not recalibrated.
- 1,800 branch optimizer updates plus 24 exact trial updates = **1,824**.
- Original AdamW state, parameter groups, clipping, bf16 training, logit-scale
  clamp, and 3,850-update scheduler horizon are preserved.
- Model, optimizer, scheduler, and all saved global RNG streams are restored for
  each branch. Every first step must exactly reproduce its native/aux trial.
- No gate controls these branches: all fixed interventions are run regardless
  of scores. Scores are diagnostic predictions, not deployed policies.

Each state gets one precommitted 3,200-image training stream (50 B64 batches,
without replacement within the state), sampled from the original 4,970 training
images by deterministic SHA256 ranking. Every arm receives exactly these same
processed batches and captions. Captions use the source training seed and saved
epoch. These are **counterfactual continuations on a new matched stream**, not
literal resumption of the old DataLoader's partially consumed permutation.
LR and RNG evolution must agree across all branches. This limited design does
not estimate within-state uncertainty over multiple continuation streams.

## Separate data roles

The original excluded 1,024 CUB training images are deterministically split by
image into 512 meta images and 512 reporting images. Both are excluded from all
branch updates and the historical training probes. Image-key disjointness is
asserted, not merely caption/index disjointness. Both use canonical first captions.

The pool was used in historical diagnostics. Separation is prospective for this
diagnostic only; it does **not** make these images a fresh independent test set.
The experiment is exploratory and the checkpoint trajectories/seeds are reused.

Meta objective: mean native symmetric contrastive loss over eight fixed B64
batches, FP32 encoder computation and FP64 logit/reduction arithmetic. Only this
pool supplies meta gradients. Reporting uses two fixed shuffled B64 partitions
of its own 512 images and averages their losses. Partitions are sensitivity
views, not independent replicates. Original CUB test data are not evaluated.

The reporting pool also yields canonical bidirectional R@1 against 512 candidates.
This is a secondary within-pool metric, **not** the historical full 5,794-image /
57,940-caption retrieval benchmark. Different candidate counts are not comparable.

## Predictors, committed before reporting

1. First-update training native/auxiliary gradient cosine, sign threshold zero.
2. Original-state meta-gradient/auxiliary cosine, sign threshold zero.
3. AdamW displacement scores S0 and SN, sign threshold zero (negative favorable).
4. Historical 16-batch mean alignment and its previously frozen threshold
   0.030847286945687016, measured with the archived probe protocol. It is kept
   distinct from the new first-batch cosine.

Clone identical source state, run one native step and one native+auxiliary step,
and form `delta = theta_native_aux_plus - theta_native_plus` in the full ordered
trainable-parameter space. `S0 = grad_meta(theta) dot delta`;
`SN = grad_meta(theta_native_plus) dot delta`.

The displacement includes the actual clipped AdamW update and clamp. The scores
remain first-order predictions, not guarantees or differentiation through the
optimizer. Predictors are written and synchronously backed up before any report
outcome is evaluated. No report result is passed to predictor computation, no
threshold is fitted, and no reporting outcome selects a checkpoint or branch.

## Outcomes and interpretation

Primary: report native-loss difference **sustained minus native at update 50**.
Secondary: pulse minus native at update 50; immediate auxiliary minus native at
update 1; 512-image pool R@1 differences at update 50. Negative loss differences
are favorable; positive R@1 differences are favorable. Both auxiliary branches
must have identical update-one outcomes.

Report each saved state, then average four checkpoint effects within each seed,
then average the three seeds. No p-values or independent-12-state claim. Predictor
sign accuracy and effect-weighted decision regret are descriptive only. The old
absolute 1e-6 loss band is reported as sensitivity, not practical equivalence or
a fitted decision boundary. Do not select the best score from this report and
call the same images a confirmation set.

Interpretation is conditional: wrong immediate signs motivate studying the
predictor; immediate benefits lost after 50 steps motivate horizon analysis;
loss gains without pool-retrieval gains suggest an objective/metric mismatch;
near-zero effects provide a reason not to expand gate tuning in this setting.
This diagnostic alone does not prove any of those mechanisms universally.

## Artifacts and operational safety

Allow 15 GiB free runtime disk for model/data caches and 100 MB additional Drive
space for results and source provenance (no new model checkpoint archive).
For recovery from ZIP, start with at least 25 GiB free runtime disk for extracted
checkpoints plus caches. Uploading the recovery ZIP to Drive needs another 8.31 GB
of Drive space; extraction itself does not duplicate those checkpoints on Drive.
Inputs are hashed before deserialization. Source PyTorch release must match;
the launcher does not silently replace the CUDA/PyTorch installation.

Outputs go into a fresh `clip_checkpoint_diagnostic_<UTC>` folder under
`MyDrive/OTCO/evaluation_transfer`. Key files are `protocol.json`,
`data_roles.json`, `branch_plans.json`, per-state `training_batches.json`,
`predictions.json`, `result.json`, and `comparison.json`. `result.json` includes
all branch losses, update traces, source hashes and replay checks. Drive backups
occur at the design commitment, prediction commitment, and completed-state
boundaries. A failure during a state does not create a resumable checkpoint;
inspect the partial run before authorizing another execution.

Wait for **DRIVE_SYNC_COMPLETE** before releasing Colab. The stored completion
manifest precedes the notebook's final Drive flush. Local CPU tests and a pinned
bundle staging test validate implementation mechanics; an A100 run is still
required to validate the full saved-checkpoint/data path.
