# Five-checkpoint development-seed evidence

[Findings and protocol](../../docs/clip-paired-intermediate.md) · [Project](../../README.md)

Completed run `clip_paired_intermediate_20260912T131926_801067Z`, pinned source `cce20072fe61be22bfcfd3b17ac8605d8bdce78b`. Seeds 42, 123 and 456 at updates 100, 250, 500, 750 and 1,001: **720 branch records**, including **432 intermediate records and 288 historical endpoint replays**. All six endpoint comparisons reproduced archived state/feature hashes and recorded losses exactly. These are the same three development trajectories, not additional independent seeds.

The curves are non-monotonic and seed-dependent. At update 500, synthetic helps seed 456 in all 16 trials but hurts seed 42 in all 16, across every partition. No universal switch-off step follows.

- [Per-seed summaries](results/per_seed_summary.json) · [Graph PNG](results/intermediate_usefulness.png) · [SVG](results/intermediate_usefulness.svg).
- Raw branch losses and gradient/update diagnostics: [seed 42](seed_42/results/paired/paired_updates.jsonl), [seed 123](seed_123/results/paired/paired_updates.jsonl), [seed 456](seed_456/results/paired/paired_updates.jsonl).
- [Audit](audit.json) · [Source/environment](results/run_manifest.json) · [Protocol](results/protocol.json) · [Completion](results/completion.json).

ZIP SHA256: `c34004cc190bf42bfe45e89fed137750b79ee0af7294ddcc306c1124f3eb4004`. All 101 original members passed CRC/byte checks, fixed-input hashes and recorded loss arithmetic were verified, and all fifteen seed/state feature hashes differ. The audit inventories the complete source ZIP, including members omitted from this export.

Original JSON/JSONL/YAML records are byte-preserved; raw captions, repeated diagnostic identity/partition files, stdout and per-seed plot duplicates are omitted. Fixed inputs match the already committed [reference partitions](../clip_paired_updates_2026-09/heldout_partitions.json), [source indices](../clip_paired_updates_2026-09/training_source_indices.json) and [holdout](../../configs/cub200_clip_diagnostic_holdout_indices.json). Combined PNG/SVG plots are preserved from the ZIP. `audit.json` and this README are derived artifacts.

No checkpoints or dataset tensors are committed. They were local-only in Colab; recorded checks are not an independent offline GPU replay. No Drive backup was made.

Reproduce the full ZIP audit with `python scripts/audit_clip_paired_intermediate.py /path/to/source.zip --output-directory /path/to/fresh_audit`. This requires the repository dependencies and original ZIP; it performs no training. [Portable export command](../../scripts/archive_clip_usefulness_evidence.py) also archives the prospective and screening studies.
