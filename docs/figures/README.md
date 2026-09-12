# CLIP result figures

[← Project overview](../../README.md) · [Detailed results](../clip-experiments.md)

Figures are generated directly from [the archived JSON reports and epoch metrics](../../experiment_results/clip_2026-08-30_to_09-01/README.md). PNG files are embedded in Markdown; SVG files retain vector graphics and editable text for export.

| Figure | What it shows | Downloads |
|---|---|---|
| Training overview | Raw canonical retrieval and species-recognition trajectories for baseline, OT alpha=0.5, uniform top-8 alpha=0.5, and hardest-real alpha=0.5 | [PNG](clip_training_overview.png) · [SVG](clip_training_overview.svg) |
| Final comparison | All eight arms' epoch-50 changes from baseline, in percentage points | [PNG](clip_final_comparison.png) · [SVG](clip_final_comparison.svg) |
| Hardness dynamics | Uniform top-8 hardness and applied alpha over training, comparing alpha=0.5 with alpha=0.134 | [PNG](clip_hardness_dynamics.png) · [SVG](clip_hardness_dynamics.svg) |
| Gradient geometry | Native-loss alignment, predicted real-negative margin change, and auxiliary-gradient disagreement across shuffled partitions | [PNG](clip_gradient_geometry.png) · [SVG](clip_gradient_geometry.svg) |
| Staged gradients | Four fixed measurement times across three 50-epoch arms | [PNG](clip_gradient_stages.png) |
| Warmup head probes | Matched batch embedding and shared-head alignment across six baseline states | [PNG](clip_warmup_projection_readiness.png) |
| Early-pulse trajectories | Raw retrieval/species curves for the 13-epoch pulse pilot | [PNG](clip_early_pulse_performance.png) |
| Early-pulse differences | Signed differences from baseline; one seed, not confidence intervals | [PNG](clip_early_pulse_differences.png) · [SVG](clip_early_pulse_differences.svg) |
| Paired actual updates | Extra held-out loss after an auxiliary update vs a matched native update; identical y-axis scales | [PNG](clip_paired_update_effects.png) · [SVG](clip_paired_update_effects.svg) |
| Paired seed replication | Early benefit across three seeds, smaller mixed later effects; all input trials and per-seed means on common scales | [PNG](clip_paired_seed_replication.png) · [SVG](clip_paired_seed_replication.svg) |
| Intermediate usefulness | Non-monotonic five-state curves across development seeds; shading is batch range, not confidence intervals | [PNG](../../experiment_results/clip_paired_intermediate_2026-09/results/intermediate_usefulness.png) · [SVG](../../experiment_results/clip_paired_intermediate_2026-09/results/intermediate_usefulness.svg) |
| Frozen-rule prospective test | Three new seed trajectories and frozen alignment/step balanced accuracy | [PNG](../../experiment_results/clip_usefulness_prospective_2026-09/results/prospective_usefulness.png) · [SVG](../../experiment_results/clip_usefulness_prospective_2026-09/results/prospective_usefulness.svg) |

## Reproduce

From the repository root:

```bash
uv run --with matplotlib==3.10.8 python scripts/plot_clip_results.py
```

[The plotting script](../../scripts/plot_clip_results.py) requires Matplotlib and NumPy. It reads only the committed artifacts, performs no training, and regenerates the eight image files in this directory. No Downloads archives or model checkpoints are required.

Regenerate the early-pulse difference plot from committed [numeric evidence](../../experiment_results/clip_curriculum_2026-09/early_pulse_performance.json) with `python scripts/plot_clip_early_pulse_differences.py`. The other new plots are imported from the verified study ZIPs; their original generators are `src.clip_gradient_stages.plot_results`, `src.clip_warmup_readiness.plot_projection`, and `src.clip_early_pulse.plot_performance`.

Regenerate paired-update effects with `python scripts/plot_clip_paired_updates.py`. It reads the committed [96 branch records](../../experiment_results/clip_paired_updates_2026-09/paired_updates.jsonl). Dots represent fixed training batches, not elapsed training or independent seeds; dashed lines are means. Loss differences are displayed in units of 10⁻⁶, not accuracy percentage points.

Regenerate the three-seed comparison with `python scripts/plot_clip_paired_seeds.py`. It combines the original seed-42 records with the [192 new branch records](../../experiment_results/clip_paired_seeds_2026-09/README.md). Diamonds are per-seed means; small dots are fixed diagnostic training batches, not independent seeds or confidence intervals. The late-stage effect is smaller and changes sign across seeds.

## Reading conventions

Regenerate the intermediate and prospective figures from committed numeric records with `python scripts/plot_clip_usefulness_studies.py --output-directory /path/to/fresh_plots`. Original PNG/SVG bytes remain in each evidence archive; regeneration writes to a new directory rather than replacing them.

- Training curves are unsmoothed epoch measurements from seed 42. They do not show uncertainty across training seeds.
- Canonical Avg R@1 averages the two retrieval directions. Species top-1 uses fixed prompts; the two metrics evaluate different capabilities.
- Gray and sand backgrounds show auxiliary warmup and ramping. Step boundaries are converted to approximate epoch coordinates using 77 steps per epoch; the baseline has no auxiliary schedule.
- The final-comparison axes show differences in percentage points, not absolute accuracy, and use different scales. Full values and all-caption retrieval are in the detailed results tables.
- Gradient dots are three shuffled partitions of the same 1,024 frozen examples. Bars average those partitions; dots are not confidence intervals or independent model repetitions.
- Frozen margin responses are first-order tangent-embedding measurements for the positive-minus-hardest-real margin, not observed full-model optimization steps.

The source-artifact hashes in the collection manifest cover imported data. Figures are derived outputs, reproducible with the script above.
