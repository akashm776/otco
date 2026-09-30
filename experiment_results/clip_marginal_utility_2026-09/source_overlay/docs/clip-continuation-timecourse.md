# Fresh-stream continuation time course

Status: implemented for a bounded Colab follow-up; **no GPU results yet**.
This extends, rather than overwrites, the September 24 saved-checkpoint diagnostic.
It tests whether its early/later contrast repeats on fresh matched continuation
streams, and measures how fixed gradient probes evolve on both branches.

## Paste into Colab

Paste the entire [self-contained one-cell launcher](../colabs/clip_continuation_timecourse_drive_one_cell.py)
into an A100 (40 GB or larger) Colab code cell. Authorize Google Drive mounting.
The runner does not launch locally and does not retrain the original trajectories.

Inputs: either the original
`MyDrive/OTCO/evaluation_transfer/clip_gated_training_20260922T114312_608272Z/`
folder, or the previously built **`clip_checkpoint_diagnostic_inputs.zip`** in
`MyDrive/OTCO/evaluation_transfer/` or `/content`. The same recovery ZIP is reused;
no new checkpoint upload format is needed. Its 12 tensor SHA256 pins are inherited
unchanged from `configs/clip_checkpoint_diagnostic.json` and checked before
deserialization. Recovery writes a local verified subset, not another large Drive
folder. Reserve approximately 25 GiB runtime space for recovery, with at least
15 GiB remaining for caches, and 100 MB additional Drive space for small outputs.
No large output checkpoints are written.

The launcher pins repository base
`b36bcbf7c2124bbcbe0cf372df3853dadcf79bd2` and embeds the new sources. It verifies
the payload, runs CPU preflight tests and requires the source PyTorch version.
It does not replace Colab's CUDA stack on a version mismatch. Fixed gradient and
report evaluations add compute beyond the optimizer-update budget; no fixed
wall-clock runtime is promised.

Results go to a new `clip_continuation_timecourse_<timestamp>` folder under
`MyDrive/OTCO/evaluation_transfer/`. Wait for **`DRIVE_SYNC_COMPLETE`**, not just
training completion. Probe/point files are copied and read-back verified during
the run. Failed or completed time-course runs block automatic duplicate execution;
inspect partial output instead of restarting. Earlier checkpoint-diagnostic runs
do not block this distinct experiment. This is not mid-branch resume.

## Frozen design and budget

- Same three training seeds: 789, 2026, 31415.
- Same 12 saved starting states: updates 100, 250, 500 and 750 per seed.
  Update 100 is the shared native prefix; later states are from alignment-gated
  trajectories. This is not a randomized comparison of training ages.
- Two fresh stream seeds, **2026092601 and 2026092602**, committed before any
  continuation updates. Each selects 3,200 distinct training images, arranged
  as 50 batches of 64. Plans must differ from the prior stream and each other.
  They may overlap in images: fresh streams do not mean new data or new training
  seeds. Within each saved state, caption policy and caption epoch are unchanged.
- Two branches per stream: **native** and **sustained uniform-top-8 auxiliary**.
  This focused replication omits the old pulse arm. Coefficient remains
  **0.06005351588542947** for every treated update, regardless of any probe.
- **24 matched pairs / 48 branches / exactly 2,400 optimizer updates**.
  Re-probes do not make trial optimizer updates. No threshold fitting, adaptive
  treatment, report-based selection, or early stopping.
- Original model, AdamW moments, scheduler and RNG restored for each branch.
  Same batch/caption sequence, stochastic stream and learning-rate schedule in
  each pair. The original scheduler horizon remains 3,850, not a new 50-step decay.
  This is a newly committed stream, not literal old-loader resumption.

## Data roles and observations

Keep the original 512-meta / 512-report image split, canonical captions and two
fixed reporting partitions. Both sets are excluded from the 4,970 training images
by index and image key, and remain disjoint from each other. They are drawn from
the previously used diagnostic holdout; this is **not independent-test validation**.

Observe at continuation updates **0, 1, 5, 10, 25 and 50**, regardless of effects.
Update zero is common to the branches and measured once per pair (with an exact
no-update reporting replay). At every subsequent time point, on each branch:

1. Recompute the historical mean of 16 full-parameter native/auxiliary gradient
   cosines on the same frozen training-probe batches and captions. Report the
   fixed historical threshold indicator as an observation only.
2. Compute meta native-loss gradient alignment with the auxiliary gradient on
   the **fixed first historical probe batch**, plus meta native loss. This is
   not the first continuation batch, nor a new batch selected at each time point.
3. Persist the probe before the corresponding report evaluation. Reporting uses
   no autograd: two-partition native loss and 512-image canonical retrieval R@1.

The training probes can overlap continuation images; they are diagnostic probes,
not an independent evaluation set. Gradient re-probes have no optimizer or report
object access. Hash checks enforce model/buffer, optimizer, scheduler and RNG
preservation; modes and accumulated gradients are checked as well. Treatment is
fixed even if every measured signal turns negative. These are gradient scores,
**not actual-AdamW one-step utility estimates**.

## Estimand and interpretation

For each seed, starting state, stream and observation time, define
`D = report_loss(sustained) - report_loss(native)`. Negative means helpful.
Preserve individual-stream values, both reporting partitions, retrieval changes,
all branch probe trajectories and treated-minus-native probe differences.

Primary: at update 50, average the two streams within each state, average the
three later states (250/500/750) within each seed, subtract that seed's early
(100) effect, then average the three seed contrasts equally. Report the early
and later effects separately: a positive contrast alone does not show early
benefit and later harm. Intermediate-time contrasts are prespecified descriptive
secondary outcomes, not opportunities to select a best stopping time.

Three training seeds are the replication units. Streams and checkpoints are
nested/correlated; 24 pairs are not 24 independent seeds. Starting-state age and
prior treatment history are intertwined. Re-probe drift can motivate hypotheses
but cannot establish that pressure “consumes readiness,” that a sign crossing is
causal, or that gradient alignment is a successful controller. Reporting-loss
and retrieval effects remain separate; no OT-weighting superiority is tested.

## Output and local verification

Each pair retains `result.json` (traces, all time points, probe/report differences,
state-preservation audit), standalone sealed probe and point files, and its
training-batch identities. Global files include the committed protocol, data
roles, branch plans, source/checkpoint provenance, `comparison.json`, completion
counts, and verified Drive manifest. Partial points survive a later failure,
but do not count as a completed experiment.

```bash
python -m scripts.build_clip_continuation_timecourse_cell
python -m pytest -q tests/test_clip_continuation_timecourse.py
```

The local tests use tiny fabricated models. Passing them validates invariants and
packaging, not the scientific outcome or Colab GPU execution.
