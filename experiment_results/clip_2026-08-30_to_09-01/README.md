# Archived CLIP evidence: August 30–September 1, 2026

[← Project overview](../../README.md) · [Experiment purposes and conclusions](../../docs/clip-experiments.md) · [Running guide](../../docs/running-experiments.md)

This collection contains eight completed CLIP fine-tuning runs and seven frozen diagnostic reports imported from the original Downloads archives. It is deliberately separate from the ignored `results/` and `outputs/` directories.

## Training artifacts

All eight runs include the original JSON summary, all 51 epoch records (0–50), and the resolved YAML configuration. The summary also contains trainable-parameter and data-exclusion information. Numeric contents are unchanged; trailing text whitespace is normalized.

| Experiment | Summary | Epoch metrics | Recorded setup |
|---|---|---|---|
| Native CLIP baseline | [summary](training/baseline/summary.json) | [epochs](training/baseline/metrics.jsonl) | [config](training/baseline/resolved_config.yaml) |
| OT + absolute sigmoid, α=0.05 | [summary](training/ot_absolute/summary.json) | [epochs](training/ot_absolute/metrics.jsonl) | [config](training/ot_absolute/resolved_config.yaml) |
| OT + relative denominator, α=0.05 | [summary](training/ot_relative_a005/summary.json) | [epochs](training/ot_relative_a005/metrics.jsonl) | [config](training/ot_relative_a005/resolved_config.yaml) |
| OT + relative denominator, α=0.5 | [summary](training/ot_relative_a05/summary.json) | [epochs](training/ot_relative_a05/metrics.jsonl) | [config](training/ot_relative_a05/resolved_config.yaml) |
| Uniform top-32, α=0.5 | [summary](training/uniform_top32/summary.json) | [epochs](training/uniform_top32/metrics.jsonl) | [config](training/uniform_top32/resolved_config.yaml) |
| Uniform top-8, α=0.5 | [summary](training/uniform_top8/summary.json) | [epochs](training/uniform_top8/metrics.jsonl) | [config](training/uniform_top8/resolved_config.yaml) |
| Uniform top-8, α=0.134 | [summary](training/uniform_top8_pressure_matched/summary.json) | [epochs](training/uniform_top8_pressure_matched/metrics.jsonl) | [config](training/uniform_top8_pressure_matched/resolved_config.yaml) |
| Hardest real, α=0.5 | [summary](training/hardest_real/summary.json) | [epochs](training/hardest_real/metrics.jsonl) | [config](training/hardest_real/resolved_config.yaml) |

Retrieval fields in training summaries are already percentages; species accuracies are fractions and must be multiplied by 100 for percentage tables. Frozen reports use their own documented units. The canonical average is `(text_to_image.r_at_1 + image_to_text.r_at_1) / 2`; it is derived, not a source JSON field.

## Frozen diagnostic artifacts

- [V1 geometry](diagnostics/geometry_v1.json)
- [V2 scale equivalence and sparse-solver audit](diagnostics/geometry_v2.json)
- [OT versus uniform barycentric weights](diagnostics/barycentric_weights.json)
- [Support breadth](diagnostics/support_breadth.json)
- [Adaptive neighborhood](diagnostics/adaptive_neighborhood.json)
- [Sequential tangent gradients](diagnostics/gradient_sequential.json)
- [Randomized tangent gradients](diagnostics/gradient_randomized.json)
- Randomized diagnostic [partitions](diagnostics/randomized_batch_partitions.json), [holdout identities](diagnostics/randomized_diagnostic_holdout_indices.json), and [resolved config](diagnostics/randomized_resolved_config.yaml)

The 1,024 holdout identities match [the committed training-exclusion list](../../configs/cub200_clip_diagnostic_holdout_indices.json). Three shuffled partitions reuse those examples, giving 3,072 shuffled observations; including the sequential condition gives 4,096 observations. These are not independent training replications.

## Provenance and scope

[manifest.json](manifest.json) records source archive/member names, source-byte SHA-256 hashes, and imported-file SHA-256 hashes. The original archives are not needed to read the evidence. No paths to a particular user's Downloads directory are required.

The duplicate baseline is included only once. The 22-byte top-8 results ZIP is empty; the completed archive supplies that run. Model checkpoints, repeated stdout, and large per-query CSVs remain outside Git; this is a report/epoch archive, not a complete checkpoint or per-query data release.

Dates refer to the collected run artifacts; some ZIP member timestamps use a different timezone from local download timestamps. The import was performed on September 10, 2026. No experiments were rerun during documentation.
