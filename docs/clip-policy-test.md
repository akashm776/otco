# Frozen alignment-policy test

Status: implemented locally; GPU experiment not yet run.

## Question and evidence

Does using the frozen alignment rule to choose when to apply uniform top-8
synthetic-negative pressure improve a 50-update continuation beyond merely
reducing the number of auxiliary updates?

The completed marginal-utility export `clip_marginal_utility_20260930T021908_849917Z`
contains all 12 matched pairs, 84 audited states and 168 paired trials. All 239
manifest-listed files were checked locally. The primary history contrast is
+0.000015630572524405473: the next auxiliary update is less useful after sustained
history. All 12 pair contrasts have that sign at offsets 10, 25 and 50 under both
reporting partitions. This is evidence of history-dependent marginal utility,
not proof that gradient alignment mediates it or that a gate improves retrieval.

In the user's Drive `OTCO/evaluation_transfer`, the first partial marginal run is
stored under `failed runs`; the second marginal run is the complete record.
The six completed pairs in the partial export are byte-identical to those in
the complete export. They are NOT six additional replications. Neither folder
is modified or used as an input to this new experiment.

## Fixed contract

- Input: original checksum-pinned checkpoints at updates 100 and 500, for training
  seeds 789, 2026 and 31415. Uses the existing 12-checkpoint recovery ZIP.
- Two new matched continuation streams: 2026093002 and 2026093003. Each consists
  of 50 batches of 64 distinct training images (3,200 images per stream).
  Stream plans are checked against both prior time-course streams and the
  original diagnostic stream; new orderings do not mean new images or seeds.
- Four branches per checkpoint/stream: `gated`, `native`, `sustained`,
  `random_matched`. Restore identical model, AdamW, scheduler and RNG state.
- Budget: 3 seeds × 2 checkpoints × 2 streams × 4 branches × 50 = **2,400 updates**.
  No one-step optimizer trials, retraining, automatic retries or new checkpoints.
- Same coefficient 0.06005351588542947, native training implementation,
  bf16/gradient clipping/logit clamp and original 3,850-step scheduler horizon.
- Reporting at completed offsets 0, 10, 25, 50; no outcome-dependent early stop.

## Policy and exposure control

The gate uses the mean of the same 16 historical training-batch gradient
cosines, with threshold **0.030847286945687016**. At completed offsets **0, 10,
25**, measure its OWN current state, commit the decision, and apply the auxiliary
loss on subsequent updates while the score is strictly above the threshold.
Equality switches off; nonfinite scores abort. Reactivation is allowed.
The refresh cadence is a newly fixed design choice, not a previously validated
policy. The threshold is unchanged; selecting this rule after observing earlier
results is still exploratory policy selection.

Historical probes are training examples and may overlap continuation images.
The 512-example meta pool is reserved, disjoint, and unused. The separate
512-example reporting pool and its two fixed batch partitions are unchanged.
Neither report losses nor retrieval outcomes enter policy decisions.

Before any updates, commit one SHA256-ranked permutation of the 50 update
positions per matched set (seed 2026093004). After running the gated branch, let
K be its active-update count. `random_matched` activates the first K positions
in that precommitted ordering, then runs from the original checkpoint. It
receives only K, not gate scores or reporting outcomes. No redraw or selection
among permutations is allowed. K=0/50 yields a degenerate identical schedule,
which is retained and reported, not excluded.

This is a conditional exposure-matched timing comparator, not an independently
deployable policy: its dose comes from the gate. Total dose matches at update
50 ONLY; intermediate observations do not isolate timing from exposure. One
permutation per matched set gives a noisy timing comparison, not an estimate
over many random schedules. Shared samples, learning rates and RNG are checked
after every update; all observations/persistence callbacks must preserve state,
gradients and train/eval modes.

## Estimand and interpretation

Primary: final reporting loss **gated minus random_matched**, averaging streams
within checkpoint, checkpoints within seed, then the three seeds equally.
Negative is favorable to gating. Retain each state's raw reporting partitions,
loss, 512-image-pool retrieval, gate scores/decisions, exact actions and exposure.

Secondary comparisons: gated minus native and gated minus sustained, plus
fixed time courses and reporting-pool R@1. A timing advantage without improvement
over native does not establish an overall training benefit. Do not select the
best reporting time, checkpoint, stream, seed, threshold or permutation.

Three reused training seeds are the replication units, not 12 independent
replications. Checkpoint age and prior history remain confounded. Reporting data
have already been inspected. No independent-test, statistical-significance,
readiness-consumption mechanism, optimal policy or OT-weighting superiority
claim is justified. This tests uniform top-8 auxiliary pressure, not OT weighting.

## Run and validation

Paste ALL of `colabs/clip_policy_test_drive_one_cell.py` into one A100 Colab cell.
The cell checks out the pinned base and overlays authenticated sources; local
uncommitted files do not need to be pushed first. Leave
`clip_checkpoint_diagnostic_inputs.zip` in
`/content/drive/MyDrive/OTCO/evaluation_transfer/` (or `/content`).
Allow about 25 GiB local recovery space, 15 GiB free afterward, and 100 MB new
Drive space for output/logs. A100 40 GB or larger; exactly one CUDA device.
Torch's release must match the source run; the CUDA stack is not replaced.

Full child stdout/stderr streams to a unique file under
`OTCO/evaluation_transfer/clip_policy_test_logs/` and appears in Colab.
The run manifest records this path. Python failures also write `failure.json`
with a full traceback when storage is available. Hard runtime termination or
Drive I/O failure can still prevent the last bytes from persisting; a stale
`running` manifest is never a completion marker.

Wait for **DRIVE_SYNC_COMPLETE**, then download `clip_policy_test_<timestamp>`
and its separate log. Failure handling flushes and unmounts Drive; remount before
inspection. Existing marginal/time-course outputs, including `failed runs`,
are untouched. A prior policy-test manifest blocks another launch; no automatic
resume, deletion or restart is performed.

Validation covers exact optimizer trajectories, stochastic RNG matching,
decision-before-update/report barriers, dose matching, strict threshold and
reactivation, report-value independence, observer mutations, grid completeness,
aggregation, transcript preservation, and checksummed source packaging.

Next question after results: does any timing advantage also beat native
continuation, and does its direction repeat across all three seed summaries?
