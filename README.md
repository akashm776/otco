# OTCO: Optimal Transport Contrastive Learning

Research code for studying whether **Optimal Transport (OT)** can generate useful **synthetic hard negatives** for multimodal contrastive learning.

## Status

This is an **active research repository**. Several core experiments are complete, and follow-up experiments are ongoing. Results should be treated as **exploratory rather than final**.

The goal of this repo is not to assume that OT-generated negatives always help. The goal is to understand:

> **When do OT-derived synthetic negatives improve contrastive learning, and when do they hurt?**

---

## Research Question

Standard contrastive learning often relies on random in-batch negatives or simple hard-negative heuristics. Many of these negatives are too easy to provide useful training signal.

OTCO investigates whether Optimal Transport can improve this process by:

- identifying semantically close but mismatched examples
- constructing synthetic negatives through barycentric mixing
- injecting harder training signal than standard random negative sampling
- controlling when synthetic-negative pressure is useful rather than harmful

This repository focuses on **hard-negative generation**, **training dynamics**, and **the regimes where OT-based negatives are useful or harmful**.

---

## What Is Implemented

| Method | `loss_type` | Description |
|---|---|---|
| Baseline | `baseline` | SigLIP-style sigmoid contrastive loss |
| Hard Negative | `hard_negative` | SigLIP + explicit push on hardest in-batch negative |
| OT-Select | `ot_select` | SigLIP + OT-based selection of difficult negatives |
| **OT-Mix** | `softmax_mix` | SigLIP + barycentric synthetic negative via Sinkhorn OT |
| **OT-Mix Gated** | `softmax_mix` + alpha gating | OT-Mix with conditional `alpha_effective` based on plan quality and margin geometry |
| Memory Bank | `memory_bank` | SigLIP + negatives drawn from a rolling embedding queue |

**Architecture:** ResNet-50 → 512-d projection · DistilBERT → 512-d projection · L2-normalized shared space · temperature = 0.07

---

## Datasets

| Dataset | Role |
|---|---|
| Flickr8K | Early diagnostics — OT neighborhood quality and false-negative pressure |
| Flickr30K | Null result — OT adds no clear signal on generic captions |
| **CUB-200-2011** | Fine-grained bird retrieval, 200 species, Reed et al. captions |

These datasets span different retrieval regimes and negative granularities. The goal is to test when OT-generated negatives are meaningful, not to claim a universal improvement.

---

## Current Findings

Several findings are now clear:

- OT-derived synthetic negatives are **not automatically helpful**.
- Flickr30K produced a **null result**: OT-Mix matched continued baseline training but did not provide an independent gain.
- CUB-200 produced meaningful OT structure: OT-Mix often selected rank-1/rank-2 hard negatives with sharp transport plans.
- Ungated OT-Mix variants found useful local hard-negative structure, but did **not** beat the baseline in final retrieval.
- Mixed batching was the strongest ungated OT variant and finished second among ungated runs: **1.32% Avg R@1**, below the baseline's **1.38%**.
- **Mixed gated OT-Mix produced the best observed CUB-200 result so far: 1.44% Avg R@1**, above the baseline's **1.38%**.
- The main open issue is no longer whether OT can find hard negatives. It can. The issue is how to combine **candidate construction** and **loss gating** so OT pressure is applied only when useful.

This repository should be read as a **research artifact**, not a finished benchmark report. The mixed-gated result is a positive one-seed result and should be validated with additional seeds and ablations.

---

## Results

### Flickr30K — CONCLUDED NULL RESULT

> **Finding: OT-Mix adds no clear signal on Flickr30K. Random in-batch negatives are already too easy or too semantically distant for OT to find useful transport structure.**

ResNet-50 + DistilBERT, 512-d, canonical R@1 using the first caption and full validation pool:

| Experiment | Loss | Start | Epochs | Best Avg R@1 | Verdict |
|---|---|---|---|---|
| Baseline | SigLIP | scratch | 50 | 32.10% | reference |
| OT-Mix fine-tune, α=0.05 | SigLIP + OT | baseline ckpt | 30 | 32.50% at ep28 | null — matched by continued baseline |
| Continued baseline | SigLIP | baseline ckpt | 30 | 32.50% at ep17 | null — same gain, no OT |
| OT-Mix scratch, α=0.05 adaptive | SigLIP + OT | scratch | 50 | 31.25% | worse than baseline |

The apparent +0.40% from OT-Mix fine-tuning is explained by extra gradient steps. SigLIP alone, from the same checkpoint, reaches the same result 11 epochs faster. This suggests Flickr30K is not the right regime for OT-Mix with the current encoders and batching setup.

---

### CUB-200 — MAIN EXPERIMENT

Fine-grained bird retrieval: 200 species, 10 Reed et al. attribute-rich captions per image, 5794-image validation pool. Within-class negatives are visually and semantically confusable, making CUB-200 a better testbed for OT-based hard-negative generation.

All CUB runs use:

- ResNet-50 + DistilBERT
- 512-d shared embedding space
- `both_last_layer` unfreezing
- batch size 64
- 50 epochs
- seed 42 unless otherwise noted

---

## CUB-200 Summary

| Setting | Batching | OT Schedule | Final / official Avg R@1 | Best Avg R@1 | Final T→I R@1 | Final I→T R@1 | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| **OT-Mix mixed gated** | **25% stratified + 75% random** | Adaptive OT + conditional alpha | **1.44%** | **1.44% @ best checkpoint/final eval** | **1.19%** | 1.69% | Best observed run |
| Baseline | Random | None | 1.38% | 1.38% @ ep50 | 1.05% | **1.71%** | Strongest non-OT baseline |
| OT-Mix adaptive | Random | Adaptive OT, fixed α=0.05 | 1.35% best eval / 1.28% ep50 | 1.35% @ ep49 | 0.98% best eval / 0.93% ep50 | 1.71% best eval / 1.62% ep50 | Competitive, not a win |
| OT-Mix mixed batching | 25% stratified + 75% random | Adaptive OT, fixed α=0.05 | 1.32% | 1.32% @ ep50 | 1.12% | 1.52% | Best ungated OT variant |
| OT-Mix stratified | 100% stratified | More permissive OT | incomplete | incomplete | — | — | Confounded diagnostic run |

**Main conclusion:** OT-Mix can find meaningful hard negatives on CUB-200. Ungated OT-Mix does not reliably beat the baseline, but **mixed gated OT-Mix converts the local OT signal into a small positive retrieval gain**: 1.44% Avg R@1 vs. 1.38% for the baseline. This result combines two interventions:

1. **Mixed batching** supplies a candidate pool with more fine-grained within-class hard negatives.
2. **Conditional alpha gating** applies OT pressure only when the transport signal is trustworthy.

This is a promising one-seed result, not yet a general claim.

---

#### Baseline — COMPLETE

Random batching. Standard SigLIP-style baseline, no OT term.

| Ep | T→I R@1 | I→T R@1 | Avg R@1 |
|---|---:|---:|---:|
| 10 | 0.38% | 0.45% | 0.41% |
| 20 | 0.64% | 0.91% | 0.78% |
| 30 | 0.81% | 1.38% | 1.10% |
| 40 | 0.81% | 1.12% | 0.97% |
| 45 | 0.93% | 1.55% | 1.24% |
| 47 | 1.12% | 1.50% | 1.31% |
| **50** | **1.05%** | **1.71%** | **1.38%** |

> **Verdict:** Strongest non-OT baseline. The baseline is non-monotone but consolidates well late in training, finishing with 1.38% canonical Avg R@1.

---

#### OT-Mix Adaptive — COMPLETE

Random batching. Uses `gate_sim=-4.0`, `entropy_threshold=3.0`, `alpha=0.05`, and adaptive OT warmup.

| Ep | T→I R@1 | I→T R@1 | Avg R@1 | vs Baseline same ep |
|---|---:|---:|---:|---:|
| 10 | 0.35% | 0.59% | 0.47% | +0.06 |
| 20 | 0.57% | 1.12% | 0.85% | +0.07 |
| 30 | 0.81% | 1.42% | 1.11% | +0.01 |
| 40 | 0.88% | 0.95% | 0.91% | -0.06 |
| 45 | 0.79% | 1.55% | 1.17% | -0.07 |
| 47 | 0.93% | 1.55% | 1.24% | -0.07 |
| **49** | **0.98%** | **1.71%** | **1.35%** | **+0.06** |
| 50 | 0.93% | 1.62% | 1.28% | -0.10 |

> **Verdict:** Competitive, but not a clean win. OT-Mix adaptive confirms that OT can find meaningful hard-negative structure on CUB-200 and produces a slightly faster early trajectory. However, the baseline consolidates better late and finishes higher than the epoch-50 adaptive checkpoint. The best adaptive evaluation reaches 1.35% Avg R@1, still below baseline's 1.38%.

---

#### OT-Mix Mixed Batching — COMPLETE

Mixed batching uses 25% stratified samples and 75% random samples:

- 4 classes × 4 images = 16 within-class hard-negative candidates
- 48 random images from the full 200-class pool

The OT schedule is identical to OT-Mix adaptive:

- `gate_sim=-4.0`
- `entropy_threshold=3.0`
- `alpha=0.05`

Only the batching strategy differs from OT-Mix adaptive, making this a cleaner test of whether adding a small amount of within-class structure improves ungated OT-Mix.

| Ep | T→I R@1 | I→T R@1 | Avg R@1 | Note |
|---|---:|---:|---:|---|
| 10 | — | — | 0.45% | near adaptive |
| 15 | — | — | 0.68% | ahead of baseline/adaptive |
| 20 | — | — | 0.89% | ahead at same epoch |
| 21 | — | — | 0.98% | early high point |
| 22 | — | — | 0.69% | sharp dip |
| 29 | 0.10% | 1.10% | 0.60% | T→I rank-1 wobble |
| 30 | 0.72% | 1.16% | 0.94% | recovery |
| 31 | 0.95% | 1.38% | 1.16% | new high |
| 35 | 0.98% | 1.14% | 1.06% | stable recovery |
| 40 | 0.72% | 1.14% | 0.93% | dip |
| 45 | — | — | 1.11% | late recovery begins |
| 46 | — | — | 1.13% | improving |
| 47 | — | — | 1.18% | improving |
| 48 | — | — | 1.29% | near final level |
| 49 | — | — | 1.27% | slight dip |
| **50** | **1.12%** | **1.52%** | **1.32%** | best mixed checkpoint |

> **Verdict:** Best ungated OT variant, but still below baseline. Mixed batching improves access to meaningful fine-grained hard negatives and finishes stronger than ungated adaptive, but it does not beat the baseline.

---

#### OT-Mix Mixed Gated — COMPLETE

This run combines the two interventions that individually looked most promising:

- **mixed batching:** 25% stratified + 75% random
- **conditional alpha gating:** suppress/downweight OT depending on entropy and margin geometry

Batching:

- 4 classes × 4 images = 16 within-class hard-negative candidates
- 48 random images from the full 200-class pool

OT schedule:

- `gate_sim=-4.0`
- `entropy_threshold=3.0`
- scheduled `alpha=0.05`
- `alpha_effective` computed per step

Gating rule:

- suppress when the plan is diffuse: `coupling_entropy > 3.0`
- suppress when the synthetic is too easy: `pos_selected_gap > +0.10`
- downweight to 25% when the synthetic dominates the positive: `pos_selected_gap < -0.07`

| Stage | T→I R@1 | I→T R@1 | Avg R@1 | Note |
|---|---:|---:|---:|---|
| Epoch 10 | 0.38% | 0.47% | 0.42% | conservative early trajectory |
| Epoch 11 | 0.45% | 0.66% | 0.55% | improves cleanly |
| Epoch 12 | 0.52% | 0.74% | 0.63% | steady |
| Epoch 13 | 0.57% | 0.76% | 0.66% | steady |
| Epoch 14 | 0.52% | 0.83% | 0.67% | steady |
| Epoch 16 | 0.47% | 0.90% | 0.68% | steady |
| Epoch 48 / best checkpoint | — | — | **1.44%** | best observed checkpoint |
| Epoch 50 | 1.05% | 1.59% | 1.32% | lower than best checkpoint |
| Final evaluation | **1.19%** | **1.69%** | **1.44%** | official best-checkpoint eval |

> **Verdict:** Best observed CUB-200 run. Mixed gated OT-Mix is the first OTCO variant to beat the baseline on canonical Avg R@1: 1.44% vs. 1.38%. The improvement is small and should be validated across seeds, but the intermediate logs show that the gate is doing the intended thing: suppressing diffuse and too-easy OT states while preserving useful near-boundary hard negatives.

---

## Intermediate Log Analysis

The CUB-200 results should not be read only through recall. The intermediate logs show what OT is doing mechanistically.

### Useful OT regime

Across sampled/logged diagnostic steps, ungated mixed batching frequently produces the desired OT behavior:

- selected synthetic negatives are usually rank 1–3 during active useful steps
- coupling entropy is sharp, usually around 2.1–2.6
- `Pos - Selected Gap` is close to the boundary, often around -0.05 to 0.00
- synthetic loss is positive and contributes to training

Representative ungated mixed-batching examples:

| Epoch / step | Synthetic loss | Selected rank | Pos - Selected Gap | Entropy | Read |
|---|---:|---:|---:|---:|---|
| ep31 / step 2800 | 0.0643 | mean 1.67, median 1 | -0.0481 | 2.2995 | Useful hard-negative regime |
| ep40 / step 3700 | 0.1040 | mean 2.70, median 2 | -0.0255 | 2.3280 | Useful hard-negative regime |
| ep50 / step 4600 | 0.0868 | mean 1.67, median 1 | -0.0435 | 2.2182 | Useful hard-negative regime |

This confirms that OT is not random on CUB-200. It can find fine-grained, near-boundary hard negatives.

### Stale / easy OT regime

Ungated runs also show repeated stale or too-easy OT states:

- synthetic loss is zero or near-zero
- selected rank jumps to ~30+
- `Pos - Selected Gap` becomes strongly positive, often around +0.24 to +0.27
- the selected synthetic is no longer a useful hard negative

Representative examples from ungated mixed batching:

| Epoch / step | Synthetic loss | Selected rank | Pos - Selected Gap | Read |
|---|---:|---:|---:|---|
| ep31 / step 2882 | 0.0000 | mean 32.25, median 31 | +0.2489 | Too easy / stale |
| ep40 / step 3719 | 0.0000 | mean 29.25, median 28 | +0.2546 | Too easy / stale |
| ep50 / step 4649 | 0.0000 | mean 32.77 | +0.2710 | Too easy / stale |

This is the main training-dynamics issue. OT can find useful hard negatives, but fixed-alpha OT-Mix does not control when OT pressure is actually useful.

---

## Gated OT Analysis

The mixed gated run directly tests whether conditional OT pressure can preserve useful OT states while suppressing bad ones.

### Early diffuse plans are suppressed

| Epoch / step | Entropy | Selected rank | Gap | Scheduled α | Effective α | Bucket |
|---|---:|---:|---:|---:|---:|---|
| ep1 / step 92 | 3.1798 | 33.77 | +0.0002 | 0.0046 | **0.0000** | diffuse |
| ep2 / step 100 | 3.4348 | 31.69 | +0.0004 | 0.0050 | **0.0000** | diffuse |
| ep2 / step 185 | 3.2303 | 38.12 | +0.0011 | 0.0092 | **0.0000** | diffuse |
| ep3 / step 278 | 3.0487 | 28.83 | -0.0001 | 0.0139 | **0.0000** | diffuse |

These are exactly the early-training states where ungated OT would inject noisy synthetic-negative pressure.

### Useful OT is allowed

| Epoch / step | Synthetic loss | Selected rank | Gap | Entropy | Effective α | Bucket |
|---|---:|---:|---:|---:|---:|---|
| ep10 / step 900 | 0.0761 | mean 2.14, median 1 | -0.0549 | 2.7167 | **0.0450** | useful |
| ep12 / step 1100 | 0.0638 | mean 1.94, median 1 | -0.0537 | 2.6133 | **0.0500** | useful |
| ep17 / step 1500 | 0.0763 | mean 2.28, median 1 | -0.0485 | 2.5460 | **0.0500** | useful |
| ep19 / step 1700 | 0.0582 | mean 2.36, median 2 | -0.0441 | 2.4271 | **0.0500** | useful |

This is the intended regime: rank-1/rank-2 selected negatives, sharp enough plans, and `Pos - Selected Gap` near the decision boundary.

### Too-easy states are suppressed

| Epoch / step | Selected rank | Gap | Entropy | Effective α | Bucket |
|---|---:|---:|---:|---:|---|
| ep8 / step 743 | 32.72 | +0.1138 | 2.9532 | **0.0000** | too_easy |
| ep10 / step 929 | 30.41 | +0.1367 | 2.7423 | **0.0000** | too_easy |
| ep12 / step 1115 | 33.08 | +0.2055 | 2.6560 | **0.0000** | too_easy |
| ep19 / step 1766 | 30.06 | +0.2344 | 2.4640 | **0.0000** | too_easy |

This is the key improvement over ungated OT-Mix. The selected synthetic is no longer near the boundary, so OT pressure is removed.

### Parsed gated diagnostic summary

Sampled/logged OT diagnostic steps from mixed gated:

| Bucket | Count | Mean effective α | Mean selected rank | Mean `Pos - Selected Gap` | Mean entropy | Read |
|---|---:|---:|---:|---:|---:|---|
| Useful | 38 | 0.0478 | 3.57 | -0.0354 | 2.3875 | OT correctly on |
| Too easy | 33 | 0.0000 | 32.39 | +0.2445 | 2.3518 | OT correctly off |
| Diffuse | 12 | 0.0000 | 23.81 | -0.0067 | 3.2163 | OT correctly off |
| Too hard | 1 | 0.0125 | 2.25 | -0.0700 | 2.5533 | OT correctly downweighted |

**Interpretation:** Mixed gated OT-Mix combines better candidate construction with a controller. Mixed batching improves the probability that OT sees fine-grained hard negatives; gating prevents OT from contributing when the selected synthetic is diffuse, stale, or too easy.

---

## Directional Retrieval Analysis

Mixed gated improves the average primarily by improving Text → Image while keeping Image → Text close to baseline:

| Setting | Final / official T→I R@1 | Final / official I→T R@1 | Final / official Avg R@1 |
|---|---:|---:|---:|
| **OT-Mix mixed gated** | **1.19%** | 1.69% | **1.44%** |
| Baseline | 1.05% | **1.71%** | 1.38% |
| OT-Mix adaptive | 0.98% best eval / 0.93% ep50 | 1.71% best eval / 1.62% ep50 | 1.35% best eval / 1.28% ep50 |
| OT-Mix mixed | 1.12% | 1.52% | 1.32% |

This suggests conditional OT pressure may help the text-query-to-image direction, while avoiding the Image → Text degradation seen in ungated mixed batching.

---

## Stability Analysis

Across epochs 30–50 for the completed ungated runs:

| Setting | Mean Avg R@1 | Std | Min | Max | Final |
|---|---:|---:|---:|---:|---:|
| Baseline | **1.109%** | 0.134 | 0.87 | **1.38** | **1.38** |
| OT-Mix adaptive | 1.063% | 0.134 | 0.82 | 1.35 | 1.28 |
| OT-Mix mixed | 1.089% | **0.110** | 0.93 | 1.32 | 1.32 |

Mixed looked unstable earlier, but from epoch 30 onward it is actually the least variable of the ungated runs. Its limitation is not late collapse. Its limitation is that the final ceiling is still lower than baseline.

This stability summary only covers epochs 30–50; earlier mixed training was more volatile, including the epoch-29 Text → Image wobble.

For mixed gated, the best checkpoint reaches 1.44%, but the epoch-50 checkpoint falls to 1.32%. This means checkpoint selection matters. The next analysis should compare epoch-to-epoch variance and best-vs-final behavior across the gated run once a clean parsed table is available.

---

#### OT-Mix Stratified — DIAGNOSTIC / INCOMPLETE

Stratified batching: K=16 classes × 4 images = B=64. Uses `gate_sim=-4.5`, `entropy_threshold=3.5`.

| Ep | Avg R@1 | vs Baseline same ep |
|---|---:|---:|
| 10 | ~0.41% | ≈ flat |
| 16 | ~0.60% | -0.13% |
| 21 | 0.71% | -0.16% |

> **Verdict:** Inconclusive and confounded. Two things changed from adaptive: stratified batching and a more permissive OT schedule. Because both changed at once, underperformance cannot be cleanly attributed to batching alone. This run is useful for diagnostics but not for a clean design conclusion.

---

## Experiment Logs

Chronological research logs are in [`experiment_logs/`](experiment_logs/):

| Date | File | Summary |
|---|---|---|
| 2026-03-11 | [`11-3-26-logs.md`](experiment_logs/11-3-26-logs.md) | OT diagnostic on Flickr8K: negatives are semantically plausible but false-negative pressure is high, with P(neg1 > GT) = 0.65 |
| 2026-04-17 | [`17-4-26-logs.md`](experiment_logs/17-4-26-logs.md) | Plan to re-run diagnostics after better encoder convergence; hypothesis that improved geometry reduces false-negative pressure |
| 2026-04-22 | [`22-4-26-logs.md`](experiment_logs/22-4-26-logs.md) | Key finding: cosine-space OT is degenerate with SigLIP embeddings. Logit-space OT fixes this. Cosine entropy ≈ 3.33 with rank ≈ 17; logit entropy ≈ 2.0–2.5 with rank ≈ 1–2 |
| 2026-04-23 | [`23-4-26-logs.md`](experiment_logs/23-4-26-logs.md) | α=0.1 over-destabilizes a converged Flickr30K model. α=0.05 reduces the dip and peaks at 32.50% |
| 2026-04-24 | [`24-4-26-logs.md`](experiment_logs/24-4-26-logs.md) | Null result confirmed: continued baseline reaches 32.50% at epoch 17, 11 epochs before OT-Mix. Decision to move to CUB-200 |
| 2026-04-26 | [`26-4-26-logs.md`](experiment_logs/26-4-26-logs.md) | CUB-200 ungated analysis: baseline remained strongest; mixed batching was best ungated OT variant; OT found rank-1/rank-2 hard negatives but did not yet beat baseline |
| 2026-04-27 | [`27-4-26-logs.md`](experiment_logs/27-4-26-logs.md) | Mixed gated OT-Mix produced the best observed CUB-200 result so far: 1.44% Avg R@1. Mixed batching supplied fine-grained candidates; gating suppressed diffuse and too-easy OT states while preserving useful rank-1/rank-2 synthetic negatives |

---

## Technical Notes

### Why logit-space OT?

SigLIP's learned `logit_bias` compresses cosine similarities into a narrow range. Building the OT cost matrix directly in cosine space gives `exp(-C/ε)` a near-uniform Gibbs kernel, so Sinkhorn produces near-uniform plans regardless of embedding geometry.

Moving OT into logit space restores a usable dynamic range.

Empirically:

| OT space | Coupling entropy | Selected rank | Interpretation |
|---|---:|---:|---|
| Cosine space | ≈ 3.33 | ≈ 17 | near-uniform, close to chance |
| Logit space | ≈ 2.0–2.5 | ≈ 1–2 | sharp, meaningful hard-negative structure |

This is why current OT-Mix experiments use logit-space OT.

---

### OT-Mix Hyperparameters

| Parameter | Description |
|---|---|
| `top_k` | Local neighborhood size for OT support, usually 32 |
| `ot_eps` | Sinkhorn entropy regularization, calibrated for logit space, usually 0.7 |
| `sinkhorn_iters` | Sinkhorn iterations, usually 30 |
| `update_freq` | Steps between OT plan recomputation, usually 10 |
| `gate_sim` | Logit-space threshold; synthetics below this are excluded from the OT loss |
| `alpha` | Scheduled max weight of OT loss; ramps linearly over 1000 steps after `ot_ready` |
| `alpha_effective` | Actual per-step OT weight after entropy/gap gating |
| `adaptive_warmup` | Waits for coupling entropy below `entropy_threshold` before activating OT |
| `entropy_threshold` | OT activation/gating threshold; log(32) ≈ 3.47 is uniform, healthy range is roughly 2.0–2.5 |
| `gap_suppress_easy` | Suppress OT when `pos_selected_gap` is above this threshold |
| `gap_downweight_hard` | Downweight OT when `pos_selected_gap` is below this threshold |
| `hard_alpha_scale` | Scale factor for too-hard synthetic negatives, default 0.25 |

---

## Setup

```bash
uv sync
# or
pip install -e .
```

Requires Python ≥ 3.12 and PyTorch ≥ 2.9.

**CUB-200 via HuggingFace:** set `dataset.backend: hf_cub200`. The dataset downloads automatically. Requires `datasets<3.0`.

**Local Flickr8K:** place the dataset at:

```text
data/datasets/Flickr8k/
```

with:

```text
Images/
captions.txt
```

---

## Running Experiments

```bash
# Run the configured experiment
python -m src.main --config configs/default.yaml

# Evaluate a checkpoint
python -m src.test --config configs/diagnostic.yaml

# Compare logged runs
python -m src.analyze_log
```

Switch experiments by editing:

```text
configs/default.yaml → experiment.name
```

to any key in:

```text
configs/experiments.yaml
```

or use `experiment.overrides` to patch fields without modifying the registry.

**CUB-200 run configs:**

- `configs/hf_cub200_baseline.yaml`
- `configs/hf_cub200_softmax_mix_adaptive.yaml`
- `configs/hf_cub200_softmax_mix_stratified.yaml`
- `configs/hf_cub200_softmax_mix_mixed.yaml`
- `configs/hf_cub200_softmax_mix_mixed_gated.yaml`

Results are written to:

```text
experiments/exp_<timestamp>.json
results/
```

when run through the Colab scripts in `colabs/`.

---

## Citation

Paper in preparation. Older draft is at the root level.

---

## Future Research Directions

Future work should test whether the mixed gated OT result is robust and whether transport structure can be made more persistent across training.

Priority next steps:

- repeat mixed gated across additional seeds
- compare adaptive gated vs. mixed gated under identical logging
- add selected-rank-aware gating as a possible third condition
- parse epoch-to-epoch stability for the gated run
- study whether persistent transport structure can reduce stale/easy OT states

Rather than storing large banks of generated negatives, a future version may store compact transport information that captures how hard-negative relationships evolve over time.

This direction could provide a more stable way to study:

- hard-negative geometry
- curriculum effects
- plan freshness
- cross-modal alignment over training

These ideas are exploratory and not yet implemented.

---

## Current Findings

Several findings are now clear:

- OT-derived synthetic negatives are **not automatically helpful**.
- Flickr30K produced a **null result**: OT-Mix matched continued baseline training but did not provide an independent gain.
- CUB-200 produced meaningful OT structure: OT-Mix often selected rank-1/rank-2 hard negatives with sharp transport plans.
- Ungated OT-Mix variants found useful local hard-negative structure, but did **not** beat the baseline in final retrieval.
- Mixed batching was the strongest ungated OT variant and finished second among ungated runs: **1.32% Avg R@1**, below the baseline's **1.38%**.
- **Adaptive gated OT-Mix produced the best observed CUB-200 result so far: 1.44% Avg R@1**, above the baseline's **1.38%**.
- The main open issue is no longer whether OT can find hard negatives. It can. The issue is how to decide **when OT pressure should be applied**.

This repository should be read as a **research artifact**, not a finished benchmark report. The gated result is a positive one-seed result and should be validated with additional seeds and ablations.

---

## Results

### Flickr30K — CONCLUDED NULL RESULT

> **Finding: OT-Mix adds no clear signal on Flickr30K. Random in-batch negatives are already too easy or too semantically distant for OT to find useful transport structure.**

ResNet-50 + DistilBERT, 512-d, canonical R@1 using the first caption and full validation pool:

| Experiment | Loss | Start | Epochs | Best Avg R@1 | Verdict |
|---|---|---|---|---|
| Baseline | SigLIP | scratch | 50 | 32.10% | reference |
| OT-Mix fine-tune, α=0.05 | SigLIP + OT | baseline ckpt | 30 | 32.50% at ep28 | null — matched by continued baseline |
| Continued baseline | SigLIP | baseline ckpt | 30 | 32.50% at ep17 | null — same gain, no OT |
| OT-Mix scratch, α=0.05 adaptive | SigLIP + OT | scratch | 50 | 31.25% | worse than baseline |

The apparent +0.40% from OT-Mix fine-tuning is explained by extra gradient steps. SigLIP alone, from the same checkpoint, reaches the same result 11 epochs faster. This suggests Flickr30K is not the right regime for OT-Mix with the current encoders and batching setup.

---

### CUB-200 — MAIN EXPERIMENT

Fine-grained bird retrieval: 200 species, 10 Reed et al. attribute-rich captions per image, 5794-image validation pool. Within-class negatives are visually and semantically confusable, making CUB-200 a better testbed for OT-based hard-negative generation.

All CUB runs use:

- ResNet-50 + DistilBERT
- 512-d shared embedding space
- `both_last_layer` unfreezing
- batch size 64
- 50 epochs
- seed 42 unless otherwise noted

---

## CUB-200 Summary

| Setting | Batching | OT Schedule | Final / official Avg R@1 | Best Avg R@1 | Final T→I R@1 | Final I→T R@1 | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| **OT-Mix adaptive gated** | 25% stratified + 75% random | Adaptive OT + conditional alpha | **1.44%** | **1.44% @ best checkpoint/final eval** | **1.19%** | 1.69% | Best observed run |
| Baseline | Random | None | 1.38% | 1.38% @ ep50 | 1.05% | **1.71%** | Strongest non-OT baseline |
| OT-Mix adaptive | Random | Adaptive OT, α=0.05 | 1.35% best eval / 1.28% ep50 | 1.35% @ ep49 | 0.98% best eval / 0.93% ep50 | 1.71% best eval / 1.62% ep50 | Competitive, not a win |
| OT-Mix mixed batching | 25% stratified + 75% random | Same as adaptive | 1.32% | 1.32% @ ep50 | 1.12% | 1.52% | Best ungated OT variant |
| OT-Mix stratified | 100% stratified | More permissive OT | incomplete | incomplete | — | — | Confounded diagnostic run |

**Main conclusion:** OT-Mix can find meaningful hard negatives on CUB-200. Ungated OT-Mix does not reliably beat the baseline, but **adaptive gated OT-Mix converts the local OT signal into a small positive retrieval gain**: 1.44% Avg R@1 vs. 1.38% for the baseline. This is a promising one-seed result, not yet a general claim.

---

#### Baseline — COMPLETE

Random batching. Standard SigLIP-style baseline, no OT term.

| Ep | T→I R@1 | I→T R@1 | Avg R@1 |
|---|---:|---:|---:|
| 10 | 0.38% | 0.45% | 0.41% |
| 20 | 0.64% | 0.91% | 0.78% |
| 30 | 0.81% | 1.38% | 1.10% |
| 40 | 0.81% | 1.12% | 0.97% |
| 45 | 0.93% | 1.55% | 1.24% |
| 47 | 1.12% | 1.50% | 1.31% |
| **50** | **1.05%** | **1.71%** | **1.38%** |

> **Verdict:** Strongest non-OT baseline. The baseline is non-monotone but consolidates well late in training, finishing with 1.38% canonical Avg R@1.

---

#### OT-Mix Adaptive — COMPLETE

Random batching. Uses `gate_sim=-4.0`, `entropy_threshold=3.0`, `alpha=0.05`, and adaptive OT warmup.

| Ep | T→I R@1 | I→T R@1 | Avg R@1 | vs Baseline same ep |
|---|---:|---:|---:|---:|
| 10 | 0.35% | 0.59% | 0.47% | +0.06 |
| 20 | 0.57% | 1.12% | 0.85% | +0.07 |
| 30 | 0.81% | 1.42% | 1.11% | +0.01 |
| 40 | 0.88% | 0.95% | 0.91% | -0.06 |
| 45 | 0.79% | 1.55% | 1.17% | -0.07 |
| 47 | 0.93% | 1.55% | 1.24% | -0.07 |
| **49** | **0.98%** | **1.71%** | **1.35%** | **+0.06** |
| 50 | 0.93% | 1.62% | 1.28% | -0.10 |

> **Verdict:** Competitive, but not a clean win. OT-Mix adaptive confirms that OT can find meaningful hard-negative structure on CUB-200 and produces a slightly faster early trajectory. However, the baseline consolidates better late and finishes higher than the epoch-50 adaptive checkpoint. The best adaptive evaluation reaches 1.35% Avg R@1, still below baseline's 1.38%.

---

#### OT-Mix Mixed Batching — COMPLETE

Mixed batching uses 25% stratified samples and 75% random samples:

- 4 classes × 4 images = 16 within-class hard-negative candidates
- 48 random images from the full 200-class pool

The OT schedule is identical to OT-Mix adaptive:

- `gate_sim=-4.0`
- `entropy_threshold=3.0`
- `alpha=0.05`

Only the batching strategy differs from OT-Mix adaptive, making this a cleaner test of whether adding a small amount of within-class structure improves ungated OT-Mix.

| Ep | T→I R@1 | I→T R@1 | Avg R@1 | Note |
|---|---:|---:|---:|---|
| 10 | — | — | 0.45% | near adaptive |
| 15 | — | — | 0.68% | ahead of baseline/adaptive |
| 20 | — | — | 0.89% | ahead at same epoch |
| 21 | — | — | 0.98% | early high point |
| 22 | — | — | 0.69% | sharp dip |
| 29 | 0.10% | 1.10% | 0.60% | T→I rank-1 wobble |
| 30 | 0.72% | 1.16% | 0.94% | recovery |
| 31 | 0.95% | 1.38% | 1.16% | new high |
| 35 | 0.98% | 1.14% | 1.06% | stable recovery |
| 40 | 0.72% | 1.14% | 0.93% | dip |
| 45 | — | — | 1.11% | late recovery begins |
| 46 | — | — | 1.13% | improving |
| 47 | — | — | 1.18% | improving |
| 48 | — | — | 1.29% | near final level |
| 49 | — | — | 1.27% | slight dip |
| **50** | **1.12%** | **1.52%** | **1.32%** | best mixed checkpoint |

> **Verdict:** Best ungated OT variant, but still below baseline. Mixed batching improves access to meaningful fine-grained hard negatives and finishes stronger than ungated adaptive, but it does not beat the baseline.

---

#### OT-Mix Adaptive Gated — COMPLETE

Same stratified configuration as OT-Mix mixed batching adaptive:

- 25% stratified + 75% random
- `gate_sim=-4.0`
- `entropy_threshold=3.0`
- `alpha=0.05`
- seed 42

Adds per-step conditional alpha. OT loss is:

- suppressed when the plan is diffuse: `coupling_entropy > 3.0`
- suppressed when the synthetic is too easy: `pos_selected_gap > +0.10`
- downweighted to 25% when the synthetic dominates the positive: `pos_selected_gap < -0.07`

| Stage | T→I R@1 | I→T R@1 | Avg R@1 | Note |
|---|---:|---:|---:|---|
| Epoch 10 | 0.38% | 0.47% | 0.42% | slightly slower than ungated adaptive |
| Epoch 11 | 0.45% | 0.66% | 0.55% | improves cleanly |
| Epoch 12 | 0.52% | 0.74% | 0.63% | steady |
| Epoch 13 | 0.57% | 0.76% | 0.66% | steady |
| Epoch 14 | 0.52% | 0.83% | 0.67% | steady |
| Epoch 16 | 0.47% | 0.90% | 0.68% | steady |
| Epoch 48 / best checkpoint | — | — | **1.44%** | best observed checkpoint |
| Epoch 50 | 1.05% | 1.59% | 1.32% | lower than best checkpoint |
| Final evaluation | **1.19%** | **1.69%** | **1.44%** | official best-checkpoint eval |

> **Verdict:** Best observed CUB-200 run. Adaptive gated OT-Mix is the first OTCO variant to beat the baseline on canonical Avg R@1: 1.44% vs. 1.38%. The improvement is small and should be validated across seeds, but the intermediate logs show that the gate is doing the intended thing: suppressing diffuse and too-easy OT states while preserving useful near-boundary hard negatives.

---

## Intermediate Log Analysis

The CUB-200 results should not be read only through recall. The intermediate logs show what OT is doing mechanistically.

### Useful OT regime

Across sampled/logged diagnostic steps, ungated mixed batching frequently produces the desired OT behavior:

- selected synthetic negatives are usually rank 1–3 during active useful steps
- coupling entropy is sharp, usually around 2.1–2.6
- `Pos - Selected Gap` is close to the boundary, often around -0.05 to 0.00
- synthetic loss is positive and contributes to training

Representative ungated mixed-batching examples:

| Epoch / step | Synthetic loss | Selected rank | Pos - Selected Gap | Entropy | Read |
|---|---:|---:|---:|---:|---|
| ep31 / step 2800 | 0.0643 | mean 1.67, median 1 | -0.0481 | 2.2995 | Useful hard-negative regime |
| ep40 / step 3700 | 0.1040 | mean 2.70, median 2 | -0.0255 | 2.3280 | Useful hard-negative regime |
| ep50 / step 4600 | 0.0868 | mean 1.67, median 1 | -0.0435 | 2.2182 | Useful hard-negative regime |

This confirms that OT is not random on CUB-200. It can find fine-grained, near-boundary hard negatives.

### Stale / easy OT regime

Ungated runs also show repeated stale or too-easy OT states:

- synthetic loss is zero or near-zero
- selected rank jumps to ~30+
- `Pos - Selected Gap` becomes strongly positive, often around +0.24 to +0.27
- the selected synthetic is no longer a useful hard negative

Representative examples from ungated mixed batching:

| Epoch / step | Synthetic loss | Selected rank | Pos - Selected Gap | Read |
|---|---:|---:|---:|---|
| ep31 / step 2882 | 0.0000 | mean 32.25, median 31 | +0.2489 | Too easy / stale |
| ep40 / step 3719 | 0.0000 | mean 29.25, median 28 | +0.2546 | Too easy / stale |
| ep50 / step 4649 | 0.0000 | mean 32.77 | +0.2710 | Too easy / stale |

This is the main training-dynamics issue. OT can find useful hard negatives, but fixed-alpha OT-Mix does not control when OT pressure is actually useful.

---

## Gated OT Analysis

The adaptive gated run directly tests whether conditional OT pressure can preserve useful OT states while suppressing bad ones.

### Early diffuse plans are suppressed

| Epoch / step | Entropy | Selected rank | Gap | Scheduled α | Effective α | Bucket |
|---|---:|---:|---:|---:|---:|---|
| ep1 / step 92 | 3.1798 | 33.77 | +0.0002 | 0.0046 | **0.0000** | diffuse |
| ep2 / step 100 | 3.4348 | 31.69 | +0.0004 | 0.0050 | **0.0000** | diffuse |
| ep2 / step 185 | 3.2303 | 38.12 | +0.0011 | 0.0092 | **0.0000** | diffuse |
| ep3 / step 278 | 3.0487 | 28.83 | -0.0001 | 0.0139 | **0.0000** | diffuse |

These are exactly the early-training states where ungated OT would inject noisy synthetic-negative pressure.

### Useful OT is allowed

| Epoch / step | Synthetic loss | Selected rank | Gap | Entropy | Effective α | Bucket |
|---|---:|---:|---:|---:|---:|---|
| ep10 / step 900 | 0.0761 | mean 2.14, median 1 | -0.0549 | 2.7167 | **0.0450** | useful |
| ep12 / step 1100 | 0.0638 | mean 1.94, median 1 | -0.0537 | 2.6133 | **0.0500** | useful |
| ep17 / step 1500 | 0.0763 | mean 2.28, median 1 | -0.0485 | 2.5460 | **0.0500** | useful |
| ep19 / step 1700 | 0.0582 | mean 2.36, median 2 | -0.0441 | 2.4271 | **0.0500** | useful |

This is the intended regime: rank-1/rank-2 selected negatives, sharp enough plans, and `Pos - Selected Gap` near the decision boundary.

### Too-easy states are suppressed

| Epoch / step | Selected rank | Gap | Entropy | Effective α | Bucket |
|---|---:|---:|---:|---:|---|
| ep8 / step 743 | 32.72 | +0.1138 | 2.9532 | **0.0000** | too_easy |
| ep10 / step 929 | 30.41 | +0.1367 | 2.7423 | **0.0000** | too_easy |
| ep12 / step 1115 | 33.08 | +0.2055 | 2.6560 | **0.0000** | too_easy |
| ep19 / step 1766 | 30.06 | +0.2344 | 2.4640 | **0.0000** | too_easy |

This is the key improvement over ungated OT-Mix. The selected synthetic is no longer near the boundary, so OT pressure is removed.

### Parsed gated diagnostic summary

Sampled/logged OT diagnostic steps from adaptive gated:

| Bucket | Count | Mean effective α | Mean selected rank | Mean `Pos - Selected Gap` | Mean entropy | Read |
|---|---:|---:|---:|---:|---:|---|
| Useful | 38 | 0.0478 | 3.57 | -0.0354 | 2.3875 | OT correctly on |
| Too easy | 33 | 0.0000 | 32.39 | +0.2445 | 2.3518 | OT correctly off |
| Diffuse | 12 | 0.0000 | 23.81 | -0.0067 | 3.2163 | OT correctly off |
| Too hard | 1 | 0.0125 | 2.25 | -0.0700 | 2.5533 | OT correctly downweighted |

**Interpretation:** Adaptive gating turns OT-Mix from a fixed auxiliary loss into a controller. The useful signal is not just “synthetic negatives exist,” but that OT pressure is applied only when plan quality and margin geometry indicate that the synthetic negative is informative.

---

## Directional Retrieval Analysis

Adaptive gated improves the average primarily by improving Text → Image while keeping Image → Text close to baseline:

| Setting | Final / official T→I R@1 | Final / official I→T R@1 | Final / official Avg R@1 |
|---|---:|---:|---:|
| **OT-Mix adaptive gated** | **1.19%** | 1.69% | **1.44%** |
| Baseline | 1.05% | **1.71%** | 1.38% |
| OT-Mix adaptive | 0.98% best eval / 0.93% ep50 | 1.71% best eval / 1.62% ep50 | 1.35% best eval / 1.28% ep50 |
| OT-Mix mixed | 1.12% | 1.52% | 1.32% |

This suggests conditional OT pressure may help the text-query-to-image direction, while avoiding the Image → Text degradation seen in ungated mixed batching.

---

## Stability Analysis

Across epochs 30–50 for the completed ungated runs:

| Setting | Mean Avg R@1 | Std | Min | Max | Final |
|---|---:|---:|---:|---:|---:|
| Baseline | **1.109%** | 0.134 | 0.87 | **1.38** | **1.38** |
| OT-Mix adaptive | 1.063% | 0.134 | 0.82 | 1.35 | 1.28 |
| OT-Mix mixed | 1.089% | **0.110** | 0.93 | 1.32 | 1.32 |

Mixed looked unstable earlier, but from epoch 30 onward it is actually the least variable of the ungated runs. Its limitation is not late collapse. Its limitation is that the final ceiling is still lower than baseline.

This stability summary only covers epochs 30–50; earlier mixed training was more volatile, including the epoch-29 Text → Image wobble.

For adaptive gated, the best checkpoint reaches 1.44%, but the epoch-50 checkpoint falls to 1.32%. This means checkpoint selection matters. The next analysis should compare epoch-to-epoch variance and best-vs-final behavior across the gated run once a clean parsed table is available.

---

#### OT-Mix Stratified — DIAGNOSTIC / INCOMPLETE

Stratified batching: K=16 classes × 4 images = B=64. Uses `gate_sim=-4.5`, `entropy_threshold=3.5`.

| Ep | Avg R@1 | vs Baseline same ep |
|---|---:|---:|
| 10 | ~0.41% | ≈ flat |
| 16 | ~0.60% | -0.13% |
| 21 | 0.71% | -0.16% |

> **Verdict:** Inconclusive and confounded. Two things changed from adaptive: stratified batching and a more permissive OT schedule. Because both changed at once, underperformance cannot be cleanly attributed to batching alone. This run is useful for diagnostics but not for a clean design conclusion.

---

## Experiment Logs

Chronological research logs are in [`experiment_logs/`](experiment_logs/):

| Date | File | Summary |
|---|---|---|
| 2026-03-11 | [`11-3-26-logs.md`](experiment_logs/11-3-26-logs.md) | OT diagnostic on Flickr8K: negatives are semantically plausible but false-negative pressure is high, with P(neg1 > GT) = 0.65 |
| 2026-04-17 | [`17-4-26-logs.md`](experiment_logs/17-4-26-logs.md) | Plan to re-run diagnostics after better encoder convergence; hypothesis that improved geometry reduces false-negative pressure |
| 2026-04-22 | [`22-4-26-logs.md`](experiment_logs/22-4-26-logs.md) | Key finding: cosine-space OT is degenerate with SigLIP embeddings. Logit-space OT fixes this. Cosine entropy ≈ 3.33 with rank ≈ 17; logit entropy ≈ 2.0–2.5 with rank ≈ 1–2 |
| 2026-04-23 | [`23-4-26-logs.md`](experiment_logs/23-4-26-logs.md) | α=0.1 over-destabilizes a converged Flickr30K model. α=0.05 reduces the dip and peaks at 32.50% |
| 2026-04-24 | [`24-4-26-logs.md`](experiment_logs/24-4-26-logs.md) | Null result confirmed: continued baseline reaches 32.50% at epoch 17, 11 epochs before OT-Mix. Decision to move to CUB-200 |
| 2026-04-26 | [`26-4-26-logs.md`](experiment_logs/26-4-26-logs.md) | CUB-200 ungated analysis: baseline remained strongest; mixed batching was best ungated OT variant; OT found rank-1/rank-2 hard negatives but did not yet beat baseline |
| 2026-04-27 | [`27-4-26-logs.md`](experiment_logs/27-4-26-logs.md) | Adaptive gated OT-Mix produced the best observed CUB-200 result so far: 1.44% Avg R@1. Gating suppressed diffuse and too-easy OT states while preserving useful rank-1/rank-2 synthetic negatives |

---

## Technical Notes

### Why logit-space OT?

SigLIP's learned `logit_bias` compresses cosine similarities into a narrow range. Building the OT cost matrix directly in cosine space gives `exp(-C/ε)` a near-uniform Gibbs kernel, so Sinkhorn produces near-uniform plans regardless of embedding geometry.

Moving OT into logit space restores a usable dynamic range.

Empirically:

| OT space | Coupling entropy | Selected rank | Interpretation |
|---|---:|---:|---|
| Cosine space | ≈ 3.33 | ≈ 17 | near-uniform, close to chance |
| Logit space | ≈ 2.0–2.5 | ≈ 1–2 | sharp, meaningful hard-negative structure |

This is why current OT-Mix experiments use logit-space OT.

---

### OT-Mix Hyperparameters

| Parameter | Description |
|---|---|
| `top_k` | Local neighborhood size for OT support, usually 32 |
| `ot_eps` | Sinkhorn entropy regularization, calibrated for logit space, usually 0.7 |
| `sinkhorn_iters` | Sinkhorn iterations, usually 30 |
| `update_freq` | Steps between OT plan recomputation, usually 10 |
| `gate_sim` | Logit-space threshold; synthetics below this are excluded from the OT loss |
| `alpha` | Scheduled max weight of OT loss; ramps linearly over 1000 steps after `ot_ready` |
| `alpha_effective` | Actual per-step OT weight after entropy/gap gating |
| `adaptive_warmup` | Waits for coupling entropy below `entropy_threshold` before activating OT |
| `entropy_threshold` | OT activation/gating threshold; log(32) ≈ 3.47 is uniform, healthy range is roughly 2.0–2.5 |
| `gap_suppress_easy` | Suppress OT when `pos_selected_gap` is above this threshold |
| `gap_downweight_hard` | Downweight OT when `pos_selected_gap` is below this threshold |
| `hard_alpha_scale` | Scale factor for too-hard synthetic negatives, default 0.25 |

---

## Setup

```bash
uv sync
# or
pip install -e .
```

Requires Python ≥ 3.12 and PyTorch ≥ 2.9.

**CUB-200 via HuggingFace:** set `dataset.backend: hf_cub200`. The dataset downloads automatically. Requires `datasets<3.0`.

**Local Flickr8K:** place the dataset at:

```text
data/datasets/Flickr8k/
```

with:

```text
Images/
captions.txt
```

---

## Running Experiments

```bash
# Run the configured experiment
python -m src.main --config configs/default.yaml

# Evaluate a checkpoint
python -m src.test --config configs/diagnostic.yaml

# Compare logged runs
python -m src.analyze_log
```

Switch experiments by editing:

```text
configs/default.yaml → experiment.name
```

to any key in:

```text
configs/experiments.yaml
```

or use `experiment.overrides` to patch fields without modifying the registry.

**CUB-200 run configs:**

- `configs/hf_cub200_baseline.yaml`
- `configs/hf_cub200_softmax_mix_adaptive.yaml`
- `configs/hf_cub200_softmax_mix_stratified.yaml`
- `configs/hf_cub200_softmax_mix_mixed.yaml`
- `configs/hf_cub200_softmax_mix_adaptive_gated.yaml`

Results are written to:

```text
experiments/exp_<timestamp>.json
results/
```

when run through the Colab scripts in `colabs/`.

---

## Citation

Paper in preparation. Older draft is at the root level.

---

## Future Research Directions

Future work should test whether the gated OT result is robust and whether transport structure can be made more persistent across training.

Priority next steps:

- repeat adaptive gated across additional seeds
- compare adaptive gated vs. mixed gated under identical logging
- add selected-rank-aware gating as a possible third condition
- parse epoch-to-epoch stability for the gated run
- study whether persistent transport structure can reduce stale/easy OT states

Rather than storing large banks of generated negatives, a future version may store compact transport information that captures how hard-negative relationships evolve over time.

This direction could provide a more stable way to study:

- hard-negative geometry
- curriculum effects
- plan freshness
- cross-modal alignment over training

These ideas are exploratory and not yet implemented.
