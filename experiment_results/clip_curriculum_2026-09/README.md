# CLIP stage, readiness, and early-pulse results

Small numeric evidence files for the completed September 2026 experiments. All use CUB-200, CLIP ViT-B/32, seed 42 and the same excluded diagnostic holdout. Reused partitions and duplicated runs are not independent training seeds.

| Study | Source run | Archived evidence |
|---|---|---|
| Staged gradients, three 50-epoch arms | `clip_gradient_stages_20260911T013307_623587Z` | [Comparison CSV](staged_comparison.csv), [source manifest](staged_manifest.json) |
| Dense baseline warmup, 1,001 updates | `clip_warmup_readiness_20260911T120217_894964Z` | [Audited analysis](warmup_analysis.json) |
| Early pulse, three 1,001-update arms | `clip_early_pulse_20260911T233840_978774Z` | [Audit and stage metrics](early_pulse_audit.json), [epoch/stage performance](early_pulse_performance.json), [source manifest](early_pulse_manifest.json) |

Original downloaded ZIP SHA256 values:

- Staged: `bc682d1c4a60ef281169bcbdeb977542cff83ed0521a76b7d30a474dc469baec`.
- Warmup: `38e0afd2ab629690800154387504864c6385fa51a062fd049eec436301b18f51`.
- Early pulse: `5f5e63fc01ad8c891ed81294f12072980db1d83304270f495db2717689a6fe01`.

The larger result ZIPs retain raw per-query CSVs, saved diagnostic embeddings, pooled head inputs, full reports, and training logs. Encoder/optimizer checkpoints are separate and are not committed here. No image data or raw caption batches are included in this compact archive. The early-pulse run's baseline step-100 and step-1,001 checkpoints are backed up under `MyDrive/OTCO/early_pulse/<run-id>/checkpoints/baseline/`.

The staged study's 49,152 query observations, warmup's 24,576, and pulse's 36,864 count repeated query/partition/stage combinations, not unique examples. All original holdouts have 1,024 examples. CPU recomputation of pulse head probes used absolute tolerance 5e-5 plus relative tolerance 1e-4; maximum absolute difference across all metrics was 0.001095 (gradient norms are scale-dependent). Feature and metadata checksums were exact.

Interpretation: [staged](../../docs/clip-gradient-stages.md), [warmup](../../docs/clip-warmup-readiness.md), [early pulse](../../docs/clip-early-pulse.md). The completed [paired-update study](../../docs/clip-paired-updates.md) has a separate [96-branch evidence archive](../clip_paired_updates_2026-09/README.md).
