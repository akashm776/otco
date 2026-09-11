# Setup and running experiments

[← Project overview](../README.md) · [CLIP results](clip-experiments.md) · [Historical results](legacy-experiments.md)

## Environment and data

```bash
uv sync
```

Alternatively, install into an existing environment with `pip install -e .`. The project requires Python 3.12+, PyTorch 2.9.1+, and `datasets>=2.21,<3.0`. The pinned Colab runners use Transformers 4.57.3.

CUB-200 uses `dataset.backend: hf_cub200` and downloads from HuggingFace. Keep [the committed diagnostic holdout](../configs/cub200_clip_diagnostic_holdout_indices.json) excluded from CLIP fine-tuning: those 1,024 image IDs informed diagnostic choices.

For local Flickr8K, place `Images/` and `captions.txt` under `data/datasets/Flickr8k/`. The corresponding backend is `local_flickr8k`.

## Controlled CLIP fine-tuning

For the new fixed-batch diagnostic during fine-tuning, see [gradient usefulness across stages](clip-gradient-stages.md), including the A100 Colab notebook, controls, exact step boundaries, and output guide.

Run from the repository root:

```bash
uv run python -m src.clip_train --config configs/hf_cub200_clip_vit_b32_baseline.yaml
```

These are 50-epoch GPU experiments, not quick smoke tests. The checked-in configs require CUDA with at least 14,000 MiB VRAM; archived runs used an A100 40 GB with bf16. The trainable subset consists of both projection heads, the final encoder blocks, and logit scale.

Use the same command with a different config:

| Experiment | Config |
|---|---|
| Native CLIP baseline | [baseline](../configs/hf_cub200_clip_vit_b32_baseline.yaml) |
| OT absolute sigmoid | [raw-cosine gap-gate](../configs/hf_cub200_clip_vit_b32_otco_rawcos_gap_gate.yaml) |
| OT relative, alpha=0.05 | [relative denominator](../configs/hf_cub200_clip_vit_b32_otco_relative_denominator.yaml) |
| OT relative, alpha=0.5 | [relative native strength](../configs/hf_cub200_clip_vit_b32_otco_relative_native_strength.yaml) |
| Uniform top-32 | [uniform barycentric](../configs/hf_cub200_clip_vit_b32_uniform_barycentric_relative_native_strength.yaml) |
| Uniform top-8 | [uniform top-8](../configs/hf_cub200_clip_vit_b32_uniform_top8_relative_native_strength.yaml) |
| Hardest real | [hardest real](../configs/hf_cub200_clip_vit_b32_hardest_real_relative_native_strength.yaml) |
| Pressure-matched top-8 | [pressure matched](../configs/hf_cub200_clip_vit_b32_uniform_top8_relative_pressure_matched.yaml) |

[Colab runners](../colabs/) record their pinned source commit and package setup. To reproduce a particular archived result, consult its recorded `resolved_config.yaml` and the corresponding runner rather than assuming a later source revision is identical.

Training writes `summary.json`, `metrics.jsonl`, `resolved_config.yaml`, and data-exclusion/trainable-parameter reports beneath `outputs/<experiment_name>/`. Checkpoints live beneath `checkpoints/<experiment_name>/`. Both directories are ignored by Git. The trainer also accepts `--output-directory` and `--checkpoint-directory` overrides.

**Checkpoint semantics:** `best_model.pt` is selected by species top-1, while `latest.pt` is the latest training state. In the archived CLIP runs, the best species epoch is 0. Do not use that checkpoint as if it were the best retrieval epoch.

## Frozen CLIP diagnostics

These inspect a fixed pretrained CLIP checkpoint without optimizer updates. V1 through adaptive-neighborhood diagnostics do not compute training gradients; the two gradient diagnostics compute local embedding gradients without updating model parameters.

```bash
uv run python -m src.clip_geometry_diagnostic --config configs/hf_cub200_clip_geometry.yaml
uv run python -m src.clip_geometry_diagnostic_v2 --config configs/hf_cub200_clip_geometry_v2.yaml
```

The later diagnostics use the same invocation pattern:

| Module after `python -m` | Config |
|---|---|
| `src.clip_barycentric_weight_ablation` | [barycentric weights](../configs/hf_cub200_clip_barycentric_weight_ablation.yaml) |
| `src.clip_support_breadth_ablation` | [support breadth](../configs/hf_cub200_clip_support_breadth_ablation.yaml) |
| `src.clip_geometry_adaptive_neighborhood` | [adaptive neighborhood](../configs/hf_cub200_clip_geometry_adaptive_neighborhood.yaml) |
| `src.clip_negative_gradient_geometry` | [sequential gradients](../configs/hf_cub200_clip_negative_gradient_geometry.yaml) |
| `src.clip_negative_gradient_geometry_randomized` | [randomized gradients](../configs/hf_cub200_clip_negative_gradient_geometry_randomized_batches.yaml) |

### What V1 and V2 control

V1 preserves the historical scaled-logit OT condition: epsilon 0.7, top-32 support, and the historical sparse solver. At CLIP scale 100, that epsilon corresponds to about 0.007 in cosine space. Its entropy is not a scale-independent measure of semantic quality.

V2 compares that condition with scaled-logit epsilon 4.9 and raw-cosine epsilon 0.049. The latter pair should agree at scale 100. It records plan/index/entropy differences, sparse marginal errors, removed mass, species-neighborhood statistics, and synthetic hardness relative to real contributors. A separate support-preserving solver is diagnostic-only; the archived full-pool attempt reports infeasible support.

V2 also emulates B64 batches and applies the historical alpha-gating rule to batch-level mean entropy and positive-selected gap. This differs from V1's per-query threshold overlay. Neither diagnostic trains or gates a live model update.

V1 writes under `outputs/cub200_frozen_clip_vit_b32_geometry/`; V2 writes separately under `outputs/cub200_frozen_clip_vit_b32_geometry_v2/`. Keep their holdout-index files. Later diagnostic output directories are specified in their configs; preserved result reports are linked in [the CLIP analysis](clip-experiments.md#frozen-diagnostics).

### Interpretation boundaries

Top-k candidate selection already imposes a high full-pool hardness percentile. Strict-hardest agreement, support comparisons, and sparse-plan audits are needed before attributing that percentile to OT.

Frozen holdout retrieval, all-caption test retrieval, canonical test retrieval, and fixed-prompt species classification use different query/target definitions. Their numbers are not interchangeable. The [CLIP results page](clip-experiments.md#final-epoch-50-results) documents units and checkpoint selection.

## Historical ResNet-50 + DistilBERT training

This is a separate entry point and configuration system:

```bash
uv run python -m src.main --config configs/default.yaml
uv run python -m src.test --config configs/diagnostic.yaml
uv run python -m src.analyze_log
```

A run config selects an `experiment.name` from [the experiment registry](../configs/experiments.yaml); `experiment.overrides` patches fields for that run. The principal historical losses are baseline SigLIP-style, hard-negative, OT-select, OT-mix (`softmax_mix`), and memory-bank variants.

CUB configurations include [baseline](../configs/hf_cub200_baseline.yaml), [adaptive](../configs/hf_cub200_softmax_mix_adaptive.yaml), [mixed batching](../configs/hf_cub200_softmax_mix_mixed.yaml), [adaptive gated](../configs/hf_cub200_softmax_mix_adaptive_gated.yaml), and [cached pool 128](../configs/hf_cub200_softmax_mix_cached_pool_128.yaml). See [historical experiments](legacy-experiments.md) for the recorded schedules and outcomes.

Historical Colab workflows write logs/results under `experiments/` and `results/`. The latter is ignored by Git. Curated CLIP evidence is kept separately in [experiment_results](../experiment_results/clip_2026-08-30_to_09-01/README.md), so it is available without local Downloads or model checkpoints.
