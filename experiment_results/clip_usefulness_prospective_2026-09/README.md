# Frozen-rule prospective-seed evidence

[Findings and protocol](../../docs/clip-usefulness-prospective.md) · [Project](../../README.md)

Completed run `clip_usefulness_prospective_20260912T175040_066114Z`, source pinned **before new outcomes** at `d907f67968b4ead5e06d416b2f675283cdba2438`. Seeds **789, 2026 and 31415**, five checkpoints each, **720 new branches / 15 states**. Rules were not refitted.

| Frozen rule | Mean seed balanced accuracy | Correct states | Correct excluding ±1e-6 |
|---|---:|---:|---:|
| Full-gradient alignment > 0.030847286945687016 | 90.28% | 13/15 | 12/13 |
| Training step ≤ 175 | 83.33% | 13/15 | 12/13 |

Primary difference: **+6.94 percentage points** for alignment, with per-seed differences +8.33, +25.00 and −12.50 points. Ordinary and non-near-zero accuracy are tied. Alignment catches all five helpful states but makes two false-positive predictions; step catches three and makes none. These are small, reused-holdout one-step CLIP/CUB effects, not a demonstrated curriculum gain or statistical significance claim.

- [All frozen-rule predictions and scores](results/prediction_report.json) · [15 states](results/states.json) · [Frozen development artifact](results/frozen_rules.json).
- Raw branch records: [seed 789](seed_789/results/paired/paired_updates.jsonl), [seed 2026](seed_2026/results/paired/paired_updates.jsonl), [seed 31415](seed_31415/results/paired/paired_updates.jsonl).
- [Graph PNG](results/prospective_usefulness.png) · [SVG](results/prospective_usefulness.svg).
- [Audit](audit.json) · [Source/environment](results/run_manifest.json) · [Protocol](results/protocol.json) · [Completion](results/completion.json).

ZIP SHA256: `854f27d954030775c2640a6ee87ec80e59ccacc1f9915773b5b07669762574d0`. All 100 members passed CRC/byte checks. The audit verifies frozen-rule equality, fixed diagnostic inputs, original package versions, 720 branch records, training seeds/horizon, file inventories and recorded checkpoint/replay checks. Predictions and primary scores were independently recomputed. Its file inventory covers all original ZIP members, including export omissions.

Original JSON/JSONL/YAML files are byte-preserved. Raw captions, repeated diagnostic identity/partition files, stdout and per-seed figure duplicates are omitted, matching the [development export](../clip_paired_intermediate_2026-09/README.md) policy. Combined PNG/SVG plots are preserved from the ZIP. The audit and this README are derived artifacts.

No checkpoints, dataset tensors or Drive backups are included. State/feature hashes are recorded runtime evidence, not independently reconstructed model states. Prospective refers to new **training seeds**, not a new evaluation dataset.

Reproduce the full audit with `python scripts/audit_clip_usefulness_prospective.py /path/to/source.zip --output-directory /path/to/fresh_audit`. It requires repository dependencies and the original ZIP but no GPU run. Regenerate both study graphs from committed numeric records with [the plotting script](../../scripts/plot_clip_usefulness_studies.py).
