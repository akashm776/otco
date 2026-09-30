# Post-treatment marginal auxiliary utility

Status: implemented; not run on GPU. This is an exploratory follow-up to the
September 29 continuation time course, not a newly independent confirmation.
The old experiments, protocols and launchers are unchanged.

## Question

Does the lower alignment measured after sustained synthetic pressure correspond
to a smaller benefit from the **next actual AdamW auxiliary update**?

For current history state s, define U(s,b) = J_report(AdamW_aux(s,b)) minus
J_report(AdamW_native(s,b)), using exactly the same batch b, optimizer state,
learning rate and stochastic state. Negative U means adding auxiliary pressure
helps the next reporting loss. This differs from cumulative sustained-minus-native
loss. Neither gradient score nor reporting outcome selects treatment.

## Paste into Colab

Paste the entire [one-cell launcher](../colabs/clip_marginal_utility_drive_one_cell.py)
into an **A100 40 GB or 80 GB** Colab runtime. The completed time course used
an A100 80 GB; exact state hashes must reproduce even on a different A100 capacity.
Hardware/library differences that change replay cause a stop, not relaxed checks.
The source Torch release must match (2.11.0 in the completed run); the cell does
not replace the CUDA stack. Local CPU tests do not establish GPU replay.

Reuse `clip_checkpoint_diagnostic_inputs.zip` in
`MyDrive/OTCO/evaluation_transfer/` (or `/content`), or the original gated source
folder. The existing ZIP is sufficient: no time-course ZIP upload is needed.
Its original 12 checkpoints are verified by the unchanged recovery code; this
study uses six. Reserve 25 GiB runtime space for recovery, at least 15 GiB free
after recovery, and 100 MB extra Drive space for results. Model/optimizer trial
snapshots are in-memory CPU copies; no large new checkpoints are saved.

Wait for `DRIVE_SYNC_COMPLETE`. Download the new `clip_marginal_utility_*` folder.
Failures preserve partial JSON output but are not automatically resumed. Existing
runs of this new experiment block duplicate execution; older time-course and
checkpoint-diagnostic runs do not. Do not restart a failed run blindly.

## Fixed bounded design

- Training seeds 789, 2026, 31415; source checkpoints **100 and 500** only.
  These were selected after reviewing the earlier results: early loss benefit
  versus a later checkpoint containing a short-to-long-horizon reversal.
- Replay both archived continuation streams, 2026092601/2026092602, from each
  starting checkpoint. Native and sustained histories run 50 updates each.
  This reconstructs states; it does not resume a saved time-course endpoint.
- Audit at offsets **0, 10, 25 and 50**. Offset zero is measured once per pair,
  shared by histories, not counted twice as a replicate.
- At every audited state, two fixed B64 trial batches each fork one native and
  one native-plus-uniform-top8 update. Restore the full pretrial state between
  actions, between batches, and before continuing. All trial steps are discarded.
- Trial batches are deterministically chosen using seed 2026093001 from training
  images excluded from that pair's entire continuation and historical probes.
  They are mutually disjoint and never contain meta/report images. The same
  batches/captions are reused across both histories and all times within a pair.
  They can overlap other pairs and were not necessarily unseen before the source
  checkpoint; they are training interventions, not new independent test data.
- Fixed coefficient 0.06005351588542947, original AdamW/clipping/bf16/clamp,
  original 3,850-step scheduler horizon. No fitted thresholds or adaptive gates.
- **12 history pairs; 24 histories; 1,200 continuation updates.**
  **84 audited states; 168 matched trial pairs; 336 trial updates.**
  **Total: 1,536 optimizer updates.** Gradient probes/evaluations add compute
  beyond the optimizer budget; no fixed runtime is promised.

## Predictions and reporting

Keep the historical 512-meta / 512-report split and both report partitions.
They remain image-disjoint and excluded from updates, but the pool is reused.
No independent-test or population-significance claim is justified.

Before accessing trial report outcomes, durably commit:

1. Historical 16-batch mean native/auxiliary alignment, with the original frozen
   threshold, and its separate historical meta-vs-fixed-probe score.
2. Native/auxiliary gradient cosine on **each actual trial batch**, sign boundary 0.
3. Meta-gradient/auxiliary cosine on **each actual trial batch**, sign boundary 0.

Then evaluate native and auxiliary one-update endpoints on report loss (both
partitions retained) and 512-image bidirectional R@1. These are actual optimizer
interventions, not a Taylor approximation. No new AdamW displacement classifier
is fitted. Persist each complete state audit to Drive before continuing.

Each trial action starts with the same full state and RNG. Gradient observations,
report evaluation and backup callbacks must preserve state. Trial routines restore
model parameters/buffers, optimizer, scheduler, all RNG streams, gradients and
module modes, including on exceptions. The restored prefixes must equal archived
September 29 hashes at all audit times. That comparison is enforced before trials;
the source tensor checkpoints and old results remain read-only.

The reference configuration embeds hashes from the local verified time-course
export `clip_continuation_timecourse_20260929T135631_790370Z`: all 615 manifest
entries were SHA256/size checked before reference extraction. It embeds only
provenance and state/plan hashes, not reporting outcomes or image captions.

## Prespecified analysis

Primary: at offset **25**, compute the marginal reporting-loss effect after
sustained history minus the marginal effect after native history. Average the
two trial batches within an anchor, two streams within a source checkpoint,
two checkpoints within each training seed, then the three seeds equally.

A positive primary means further auxiliary pressure is less beneficial/more
harmful after sustained history. Always report both underlying effects and both
starting checkpoints separately: a difference alone does not establish a sign
reversal. Offsets 10 and 50 are secondary; offset zero is a shared baseline.
Keep raw per-batch outcomes, reporting partitions and retrieval effects.

Frozen-rule sign accuracy and decision regret are **correlated descriptive**
summaries only. Report the inherited absolute 1e-6 band as sensitivity, not an
equivalence margin. Do not refit thresholds, choose a stopping time, or treat
168 trials as independent training seeds. Replication units remain three seeds.

Interpretation:

- Lower scores AND worse marginal utility after treated history would support
  state-dependent loss of auxiliary usefulness under these fixed probes.
- Lower scores WITHOUT worse marginal utility would weaken the readiness proxy.
- Favorable loss but unfavorable retrieval would preserve the metric mismatch.
- None of these outcomes proves alignment mediates the treatment effect,
  identifies an optimal controller, separates age from prior history, or
  establishes OT-weighting superiority. Exposure and trajectory co-vary.

## Verification

`python -m scripts.build_clip_marginal_utility_cell` builds the pinned overlay.
`python -m pytest -q tests/test_clip_marginal_utility.py` tests rollback on success
and failure, commit-before-report, exact prefix equivalence, RNG/LR matching,
disjoint plans, the bounded grid, aggregation and bundle identity with tiny models.
Only the Colab run can verify full-scale checkpoint/data/GPU replay.
