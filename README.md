# OTCO: Optimal Transport Contrastive Learning

Research code for studying whether **Optimal Transport (OT)** can generate useful **synthetic hard negatives** for multimodal contrastive learning.

## Status

This is an **active research repository**. The codebase is functional and experiments are ongoing, but results should be treated as **exploratory rather than final**.

The goal of this repo is not to assume that OT-generated negatives always help. The goal is to understand:

> **When do OT-derived synthetic negatives improve contrastive learning, and when do they hurt?**

---

## Research Question

Standard contrastive learning often relies on random in-batch negatives or simple hard-negative heuristics. Many of these negatives are too easy to provide useful training signal.

OTCO investigates whether Optimal Transport can improve this process by:

- identifying semantically close but mismatched examples
- constructing synthetic negatives through barycentric mixing
- injecting harder training signal than standard random negative sampling

This repository focuses on **hard-negative generation**, **training dynamics**, and **the regimes where OT-based negatives are useful or harmful**.

---

## What Is Implemented

| Method | `loss_type` | Description |
|---|---|---|
| Baseline | `baseline` | SigLIP-style sigmoid contrastive loss |
| Hard Negative | `hard_negative` | SigLIP + explicit push on hardest in-batch negative |
| OT-Select | `ot_select` | SigLIP + OT-based selection of difficult negatives |
| **OT-Mix** | `softmax_mix` | SigLIP + barycentric synthetic negative via Sinkhorn OT **(proposed)** |
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
- Better OT structure did **not** yet translate into a final retrieval win over the baseline.
- Mixed batching was the strongest OT variant and finished second overall: **1.32% Avg R@1**, below the baseline's **1.38%**.
- The main open issue is no longer whether OT can find hard negatives. It can. The issue is how to decide **when OT pressure should be applied**.

This repository should be read as a **research artifact**, not a finished benchmark report.

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

| Setting | Batching | OT Schedule | Final Avg R@1 | Best Avg R@1 | Final T→I R@1 | Final I→T R@1 | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| **Baseline** | Random | None | **1.38%** | **1.38% @ ep50** | 1.05% | **1.71%** | Best run |
| **OT-Mix adaptive** | Random | Adaptive OT, α=0.05 | 1.28% | 1.35% @ ep49 | 0.93% | 1.62% | Competitive, not a win |
| **OT-Mix mixed batching** | 25% stratified + 75% random | Same as adaptive | 1.32% | 1.32% @ ep50 | **1.12%** | 1.52% | Best OT variant, still below baseline |
| OT-Mix stratified | 100% stratified | More permissive OT | incomplete | incomplete | — | — | Confounded diagnostic run |

**Main conclusion:** OT-Mix can find meaningful hard negatives on CUB-200, especially with mixed batching, but the current method does not yet beat the baseline in final retrieval.

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

> **Verdict:** Strongest completed run. The baseline is non-monotone but consolidates best late in training, finishing with the highest canonical Avg R@1.

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

> **Verdict:** Competitive, but not a clean win. OT-Mix adaptive confirms that OT can find meaningful hard-negative structure on CUB-200 and produces a slightly faster early trajectory. However, the baseline consolidates better late and finishes higher: 1.38% vs. OT-Mix adaptive final 1.28%, with adaptive's best checkpoint at 1.35%.

---

#### OT-Mix Mixed Batching — COMPLETE

Mixed batching uses 25% stratified samples and 75% random samples:

- 4 classes × 4 images = 16 within-class hard-negative candidates
- 48 random images from the full 200-class pool

The OT schedule is identical to OT-Mix adaptive:

- `gate_sim=-4.0`
- `entropy_threshold=3.0`
- `alpha=0.05`

Only the batching strategy differs from OT-Mix adaptive, making this a cleaner test of whether adding a small amount of within-class structure improves OT-Mix.

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

> **Verdict:** Best OT variant, but still below baseline. Mixed batching improves access to meaningful fine-grained hard negatives and finishes stronger than adaptive, but it does not beat the baseline. The final result is 1.32% Avg R@1 vs. baseline 1.38%.

---

## Intermediate Log Analysis

The CUB-200 results should not be read only through recall. The intermediate logs show what OT is doing mechanistically.

### Useful OT regime

Mixed batching often produces the desired OT behavior:

- selected synthetic negatives are usually rank 1–3 during active useful steps
- coupling entropy is sharp, usually around 2.1–2.6
- `Pos - Selected Gap` is close to the boundary, often around -0.05 to 0.00
- synthetic loss is positive and contributes to training

Representative mixed-batching examples:

| Epoch / step | Synthetic loss | Selected rank | Pos - Selected Gap | Entropy | Read |
|---|---:|---:|---:|---:|---|
| ep31 / step 2800 | 0.0643 | mean 1.67, median 1 | -0.0481 | 2.2995 | Useful hard-negative regime |
| ep40 / step 3700 | 0.1040 | mean 2.70, median 2 | -0.0255 | 2.3280 | Useful hard-negative regime |
| ep50 / step 4600 | 0.0868 | mean 1.67, median 1 | -0.0435 | 2.2182 | Useful hard-negative regime |

This confirms that OT is not random on CUB-200. It can find fine-grained, near-boundary hard negatives.

### Stale / easy OT regime

The same runs also show repeated stale or too-easy OT states:

- synthetic loss is zero or near-zero
- selected rank jumps to ~30+
- `Pos - Selected Gap` becomes strongly positive, often around +0.24 to +0.27
- the selected synthetic is no longer a useful hard negative

Representative examples:

| Epoch / step | Synthetic loss | Selected rank | Pos - Selected Gap | Read |
|---|---:|---:|---:|---|
| ep31 / step 2882 | 0.0000 | mean 32.25, median 31 | +0.2489 | Too easy / stale |
| ep40 / step 3719 | 0.0000 | mean 29.25, median 28 | +0.2546 | Too easy / stale |
| ep50 / step 4649 | 0.0000 | mean 32.77 | +0.2710 | Too easy / stale |

This is the main training-dynamics issue. OT can find useful hard negatives, but fixed-alpha OT-Mix does not yet control when OT pressure is actually useful.

### Parsed diagnostic summary

Logged OT diagnostic steps show the same pattern:

| Diagnostic from logged steps | Adaptive | Mixed | Read |
|---|---:|---:|---|
| OT diagnostic steps parsed | 96 | 141 | Mixed log has more sampled OT steps |
| Steps with positive synthetic loss | 55 / 96 | 82 / 141 | Both fire OT in about 58% of sampled OT steps |
| Good boundary steps | 42 / 96 | **74 / 141** | Mixed has more clean useful OT states |
| Too-easy zero-loss steps | 32 / 96 | **59 / 141** | Mixed also has many useless/stale states |
| Too-hard steps, gap < -0.07 | 1 / 96 | **0 / 141** | The main failure is not overly hard negatives |
| Diffuse entropy > 3.0 | 11 / 96 | **0 / 141** | Mixed avoids diffuse-plan failure late |
| Active-loss median selected rank | 2.19 | **2.16** | Both find rank-1/rank-2 negatives when OT is useful |
| Zero-loss median selected rank | 31.98 | 32.11 | Useless states are consistently rank ~30+ |

**Interpretation:** Mixed batching improves OT signal quality, but that better local hard-negative signal does not yet produce a final retrieval win.

---

## Directional Retrieval Analysis

Mixed batching finishes with the best Text → Image R@1:

| Setting | Final T→I R@1 | Final I→T R@1 | Final Avg R@1 |
|---|---:|---:|---:|
| Baseline | 1.05% | **1.71%** | **1.38%** |
| OT-Mix adaptive | 0.93% | 1.62% | 1.28% |
| OT-Mix mixed | **1.12%** | 1.52% | 1.32% |

This suggests mixed batching may help the text-query-to-image direction, but it loses enough on Image → Text that the average remains below baseline.

---

## Stability Analysis

Across epochs 30–50:

| Setting | Mean Avg R@1 | Std | Min | Max | Final |
|---|---:|---:|---:|---:|---:|
| Baseline | **1.109%** | 0.134 | 0.87 | **1.38** | **1.38** |
| OT-Mix adaptive | 1.063% | 0.134 | 0.82 | 1.35 | 1.28 |
| OT-Mix mixed | 1.089% | **0.110** | 0.93 | 1.32 | 1.32 |

Mixed looked unstable earlier, but from epoch 30 onward it is actually the least variable of the three. Its limitation is not late collapse. Its limitation is that the final ceiling is still lower than baseline.

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

#### OT-Mix Adaptive Gated — QUEUED

Same config as OT-Mix adaptive:

- `gate_sim=-4.0`
- `entropy_threshold=3.0`
- `alpha=0.05`
- seed 42

Adds per-step conditional alpha. OT loss is:

- suppressed when the plan is diffuse: `coupling_entropy > 3.0`
- suppressed when the synthetic is too easy: `pos_selected_gap > +0.10`
- downweighted to 25% when the synthetic dominates the positive: `pos_selected_gap < -0.07`

**Hypothesis:**

> OT-Mix does not mainly fail because OT cannot find hard negatives. It fails because useful and useless OT states are mixed together under a fixed alpha schedule. Gating should improve the sample efficiency of the OT term by applying OT pressure only when the selected synthetic is near the decision boundary.

**Key comparison metrics:**

| Metric | Baseline | Adaptive | Adaptive Gated |
|---|---:|---:|---:|
| Best canonical Avg R@1 | 1.38% | 1.35% | ? |
| Final canonical Avg R@1 | 1.38% | 1.28% | ? |
| Epoch-to-epoch std | 0.134 | 0.134 | ? |
| Largest validation dip | ? | ? | ? |
| Useful % | — | — | ? |
| Too easy suppressed % | — | — | ? |
| Too hard downweighted % | — | — | ? |
| Entropy suppressed % | — | — | ? |

The purpose of this run is not only to improve R@1. It is also to test whether conditional OT pressure suppresses stale/easy OT states while preserving useful rank-1/rank-2 synthetic negatives.

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
| 2026-04-26 | [`26-4-26-logs.md`](experiment_logs/26-4-26-logs.md) | CUB-200 final analysis: baseline remains strongest; mixed batching is best OT variant; OT finds rank-1/rank-2 hard negatives but does not yet beat baseline. Gating hypothesis refined from “prevent collapse” to “suppress stale/easy OT states.” |

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
| `alpha` | Max weight of OT loss; ramps linearly over 1000 steps after `ot_ready` |
| `adaptive_warmup` | Waits for coupling entropy below `entropy_threshold` before activating OT |
| `entropy_threshold` | OT activation threshold; log(32) ≈ 3.47 is uniform, healthy range is roughly 2.0–2.5 |
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

Future work will explore whether OTCO can use persistent transport structure across training instead of relying only on per-batch OT plans.

Rather than storing large banks of generated negatives, a future version may store compact transport information that captures how hard-negative relationships evolve over time.

This direction could provide a more stable way to study:

- hard-negative geometry
- curriculum effects
- plan freshness
- cross-modal alignment over training

These ideas are exploratory and not yet implemented.
