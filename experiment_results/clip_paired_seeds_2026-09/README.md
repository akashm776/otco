# Two-seed paired-update replication evidence

[Findings and protocol](../../docs/clip-paired-seed-replication.md) · [Project](../../README.md)

Completed A100 run: `clip_paired_seeds_20260912T050100_795138Z`. Pinned experiment source: `2c4c610576c5167114bb6778cef526f0a8d64df8`. Both new native baseline seeds (123 and 456) completed 1,001 updates, followed by 96 paired branches per seed. Seed 42 in the comparison is the [previous completed experiment](../clip_paired_updates_2026-09/README.md), not a newly run seed.

## Evidence

- [Per-seed comparison](results/per_seed_summary.json): all three seeds, both states and both auxiliaries, including means, ranges, beneficial counts and gradient/update measurements.
- All **192 new branch records**, retaining every held-out B64 loss: [seed 123](seed_123/results/paired/paired_updates.jsonl), [seed 456](seed_456/results/paired/paired_updates.jsonl).
- Checkpoint provenance and recorded exact replay/reset checks: [seed 123](seed_123/results/paired/checkpoint_provenance.json), [seed 456](seed_456/results/paired/checkpoint_provenance.json).
- [Offline audit](audit.json): file hashes, rechecked arithmetic, per-partition effects, native versus treatment absolute loss changes, fixed-input checks and reference comparison.
- [Study protocol](results/protocol.json), [environment/source manifest](results/run_manifest.json), [completion marker](results/completion.json). Each seed also retains its resolved training config, 14 epoch metric records (including epoch zero), training summary, calibration-exclusion indices and source-file inventory.

ZIP SHA256: `6a14b339686712ef1f9d8354b2200627642e79a892b681a1a6db32362f7f996e`. All 54 ZIP members passed CRC and extracted-byte comparisons. Package versions match the earlier seed-42 run. Fixed diagnostic image/caption identities and partitions match the original hashes; the six seed/stage feature hashes are distinct. The new rows and the 96 historical reference rows were recomputed offline from their recorded per-batch losses.

This caption-free export retains original JSON/JSONL/YAML files byte-for-byte, except omitted diagnostic identities and partitions that were verified identical to the existing [source indices](../clip_paired_updates_2026-09/training_source_indices.json), [partitions](../clip_paired_updates_2026-09/heldout_partitions.json), and [holdout config](../../configs/cub200_clip_diagnostic_holdout_indices.json). Raw captions, stdout and original figure files remain in the downloaded ZIP; plots are regenerated from the committed records. `audit.json` is a derived report.

**No Drive backup or checkpoint upload was made.** Model/AdamW checkpoints were on the Colab runtime only and are not in this archive. Their recorded hashes match the source inventories and runtime provenance, but the offline audit does not independently rerun GPU encoding/backpropagation or reconstruct checkpoints. If that runtime is gone, obtaining additional intermediate model states requires training again.

## Reproduce

```bash
python scripts/audit_clip_paired_seeds.py /path/to/clip_paired_seeds_20260912T050100_795138Z_complete.zip --output-directory /path/to/new_export
python scripts/plot_clip_paired_seeds.py
```

The audit requires the original ZIP, the committed seed-42 reference and PyYAML. The plot only needs the committed results and Matplotlib; it performs no training. Neither command writes to Drive.
