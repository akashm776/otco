# Checkpoint replay and evaluation-pool transfer

Status: **runtime reports completion; local artifact audit pending**. On September
14, the user supplied the completed log for
`clip_evaluation_transfer_20260914T130457_447373Z`: all three seeds, exact historical
states, old losses within 1e-8, 720 new-pool observations, and a verified 12.21 GB
archive with the 15 required checkpoints. The browser download stopped; raw
new-pool results and downloaded checkpoints have not yet been independently audited.
Do not infer new-pool effectiveness from the printed original-pool branch losses.

The previous prospective ZIP preserved results but
omitted checkpoint tensors. This one-cell job reconstructs the three trajectories,
checks them against the archived study, and measures the same paired updates on a
different pool. It now exports **actual checkpoints**, not just their hashes.

## Run everything in one cell

### Persistent Google Drive version (September 21)

Paste the entire [Drive-enabled one-cell launcher](../colabs/clip_evaluation_transfer_drive_one_cell.py)
into **one A100 Colab code cell**, run it, and authorize the Drive mount prompt.
This version was requested after the September 14 browser download was lost.
It has not yet produced a verified new run; the connector was unavailable during preparation.

Destination: `MyDrive/OTCO/evaluation_transfer/<unique_run_id>/`. Reserve at least
**30 GB free Drive storage** for the approximately 12.2 GB output tree plus its full
verified ZIP, and **50 GiB local runtime disk**. A storage increase does not extend
the Colab runtime lifetime. No browser download is required in this mode.

Training and replay stay on local disk. Synchronous SHA256-verified copies are
made after each closed 100/250/500/750 diagnostic snapshot, after each seed's
training finishes (including final `latest.pt`), at evaluation stage boundaries,
and at each completed seed. The final tree and full ZIP are read back and verified
on mounted Drive. `DRIVE_BACKUP_MANIFEST.json` records file hashes. A failed copy
raises an error instead of allowing training to continue without backups.
Already copied files and local originals are retained. A crash can still lose
work since the last successfully uploaded copy; mounted read-back alone is not
proof that all Drive writes have reached the remote service.

At the end the launcher calls Colab's `drive.flush_and_unmount()` and prints
**`DRIVE_SYNC_COMPLETE`** only after it succeeds. Wait for this message before
releasing the runtime, and confirm the run folder is visible in Drive. This
unmounts Drive in that runtime after the backup. Mount/write failures stop the job.

Existing persistent runs with the same source bundle block automatic duplicate
training. This is **backup, not arbitrary mid-epoch resume**: partial trajectories
need inspection before continuation. Seeds, frozen rules, data pools, training
math and historical replay tolerances remain unchanged.

Regenerate this version with
`python scripts/build_clip_evaluation_transfer_cell.py --drive`.

### Local-only version

Paste the entire [self-contained launcher](../colabs/clip_evaluation_transfer_one_cell.py)
into one **A100 Colab** code cell and run that cell only. No uploads, separate setup
cells, GitHub changes, or Google Drive mounts/writes are required. The cell checks
out immutable base `7e7cfea90b60415dc9561efbe97bcd383cc1580e` and applies its embedded,
SHA256-checked source overlay. It runs unit tests before training.

Allow **50 GiB free runtime disk**, plus approximately **13 GB on your computer**
for the final ZIP. The three-seed replay and added evaluation are a multi-hour job;
the previous study's roughly 74 minutes does not include the new pool evaluation.
An active cell does not guarantee that Colab keeps the VM alive.

## Fixed design

- Seeds **789, 2026, 31415**; updates **100, 250, 500, 750, 1001**.
- Identical native training, 13 epochs with the original 50-epoch LR horizon.
- Same diagnostic training batches, captions, coefficients, and reset AdamW branches.
- All 15 model/optimizer/scheduler **state digests**, old-pool feature hashes, and
  learning rates must match the original run exactly. All original branch losses
  must replay within absolute **1e-8**. A mismatch stops the job; it is not silently
  reclassified as the same trajectory. Serialized file hashes are recorded separately.
- Additional pool: **1,024 class-balanced CUB test-split examples**, canonical first
  captions, selection seed **20260913**, four fixed B64 partitions (three shuffled
  primary; sequential sensitivity). Stable image keys must be disjoint from the
  *entire* CUB training split, including its original diagnostic holdout.
- Freeze alignment threshold **0.030847286945687016**, step threshold **175**, and
  near-zero band **±1e-6**. No refitting or gate-driven training.

Both pools observe the **same 720 branches**; these are not 1,440 independent
updates. Aggregate to 15 seed/checkpoint states. Primary new-pool readout is the
mean per-seed balanced-accuracy difference (alignment minus step), with individual
seed scores, ordinary/non-near-zero accuracy, beneficial-trial counts, absolute
loss changes, partition sensitivity and old/new usefulness curves retained.

The test split has previously been used for epoch retrieval monitoring: this is
**evaluation-pool transfer, not untouched-test confirmation**. It remains CLIP/CUB
specific and does not demonstrate a training curriculum benefit.

## What gets downloaded

For the local-only retraining after the interrupted September 14 download, the
launcher retains the full verified archive but automatically requests the small
results-only ZIP first. A selector then offers individual checkpoint ZIPs. The
training, replay checks, and frozen evaluation design are unchanged.

One ZIP contains all output files, including the **15 required model + AdamW +
scheduler checkpoints**, additional saved checkpoints, original and new-pool raw
records, replay checks, frozen rules, scores, PNG/SVG plots, logs and source overlay.
`PACKING_MANIFEST.json` records every member's size and SHA256. The runner reads
the archive back and verifies every member before announcing completion.

If automatic download fails, use the printed ZIP path in Colab's Files panel.
**Keep the runtime until the download has actually finished.** A completed rerun
of the cell verifies and downloads its matching local ZIP without retraining.
An unfinished prior run blocks automatic retraining so it can be inspected first.
A handled failure exports available files under an explicitly failed status;
VM deletion cannot be caught and loses files not already downloaded.

### Recover an interrupted browser download without retraining

Paste the [recovery cell](../colabs/recover_clip_evaluation_transfer_one_cell.py)
into a **new cell in the same runtime**. It targets the completed September 14
archive, downloads a small results-only ZIP first, and provides a selector for
individual checkpoint ZIPs (roughly 0.7 GB each). The 15 required states appear
first, followed by additional best-model files. Each ZIP retains original paths
and has a subset manifest; copied members are checked against original SHA256
records and read back for verification. The original archive is not modified.

Save the results ZIP first, then request checkpoints individually, waiting for each
browser download to finish. The selector tracks requests, not successful local
downloads. A missing runtime archive stops recovery; it never starts training.

[Observer and orchestration](../src/clip_evaluation_transfer.py) ·
[Runner/export verification](../colabs/run_clip_evaluation_transfer.py) ·
[Tests](../tests/test_clip_evaluation_transfer.py) ·
[Regenerate the cell](../scripts/build_clip_evaluation_transfer_cell.py)
