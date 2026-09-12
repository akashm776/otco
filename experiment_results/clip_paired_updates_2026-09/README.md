# Paired actual-AdamW update evidence

[Findings and protocol](../../docs/clip-paired-updates.md) · [Project](../../README.md)

Completed A100 run: `clip_paired_updates_20260912T042828_516737Z`. Experiment source: `d22b9d7cec94d5b044b814d774911cab5e37d560`. Inputs are the baseline model/AdamW checkpoints after 100 and 1,001 updates from `clip_early_pulse_20260911T233840_978774Z`.

**Follow-up:** the [completed seed-123/456 replication](../clip_paired_seeds_2026-09/README.md) reproduces the early benefit but finds mixed later signs. This directory remains the unchanged seed-42 numeric reference.

- [All 96 branch records](paired_updates.jsonl): every held-out B64 loss in all four partitions, training losses, gradients, and actual update metrics. One seed, 16 paired input batches, not 16 independent training runs.
- [Summary](summary.json) and [offline audit](audit.json): recomputed differences, beneficial counts, partition sensitivity, initial losses, relative pressure, and SHA256 inventory of the original ZIP's 13 files.
- [Checkpoint provenance](checkpoint_provenance.json): checkpoint/feature hashes, actual learning rates, exact re-encoding and native replay, optimizer resets and state immutability checks recorded by the GPU runner.
- [Protocol](protocol.json), [environment/source manifest](run_manifest.json), [completion marker](completion.json), [training source indices](training_source_indices.json), and [fixed held-out partitions](heldout_partitions.json).

The original ZIP SHA256 is `8045ff1d9cae92e0685c9e6381479826b3f00ea216393ea3d7da71d0bc8957c4`. Browser download and verified Drive copy match. Drive location: `MyDrive/OTCO/paired_updates/clip_paired_updates_20260912T042828_516737Z/clip_paired_updates_20260912T042828_516737Z_complete.zip`.

The compact export omits caption text, images, logs, and the original generated figures. Checkpoints remain in the previous run's Drive backup and are never overwritten. Retained original files are byte-identical to ZIP members; `audit.json` and caption-free `training_source_indices.json` are derived. The offline audit checks stored arithmetic/integrity; it does not rerun GPU forward/backward passes.

Reproduce the audit/export into a **new** directory:

```bash
python scripts/audit_clip_paired_updates.py /path/to/clip_paired_updates_20260912T042828_516737Z_complete.zip --output-directory /path/to/new_export
python scripts/plot_clip_paired_updates.py
```

The plot uses these committed records and a common vertical scale for both checkpoints. It requires Matplotlib, but no GPU or Downloads archive.
