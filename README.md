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
| **CUB-200-2011** | **Active** — fine-grained bird retrieval, 200 species, Reed et al. captions |

These datasets span different retrieval regimes and negative granularities. The goal is to test when OT-generated negatives are meaningful, not to claim a universal improvement.

---

## Current Findings

Several findings are already clear:

- OT-derived synthetic negatives are **not automatically helpful**.
- Performance depends strongly on encoder quality, candidate negative pool, batching strategy, and OT activation schedule.
- Flickr30K produced a **null result**: OT-Mix matched continued baseline training but did not provide an independent gain.
- CUB-200 shows more meaningful OT structure, but the current OT-Mix variants are still unstable and do not yet cleanly beat the baseline.
- Current work is focused on understanding **why** OT helps in some regimes and fails in others.

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

### CUB-200 — ACTIVE

Fine-grained bird retrieval: 200 species, 10 Reed et al. attribute-rich captions per image, 5794-image validation pool. Within-class negatives are visually and semantically confusable, making CUB-200 a better testbed for OT-based hard-negative generation.

All CUB runs use:

- ResNet-50 + DistilBERT
- 512-d shared embedding space
- `both_last_layer` unfreezing
- batch size 64
- 50 epochs
- seed 42 unless otherwise noted

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

> **Trajectory note:** The baseline is non-monotone but finishes strong. It dips around epoch 40, then improves again during epochs 45–50. The final checkpoint is the best observed checkpoint.

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

> **Verdict: competitive, but not a clean win.** OT-Mix adaptive confirms that OT can find meaningful hard-negative structure on CUB-200 and produces a slightly faster early trajectory. However, the baseline consolidates better late and finishes slightly higher: 1.38% vs. OT-Mix adaptive best of 1.35% at epoch 49. This suggests the OT signal is real but not yet scheduled or gated well enough to produce a reliable final gain.

---

#### OT-Mix Stratified — RUNNING

Stratified batching: K=16 classes × 4 images = B=64. Uses `gate_sim=-4.5`, `entropy_threshold=3.5`.

| Ep | Avg R@1 | vs Baseline same ep |
|---|---:|---:|
| 10 | ~0.41% | ≈ flat |
| 16 | ~0.60% | -0.13% |
| 21 | 0.71% | -0.16% |

> **Verdict: inconclusive and confounded.** Two things changed from adaptive: stratified batching and a more permissive OT schedule. Because both changed at once, underperformance cannot be cleanly attributed to batching alone. This run is useful for diagnostics but not for a clean design conclusion.

---

#### OT-Mix Mixed Batching — RUNNING

Mixed batching uses 25% stratified samples and 75% random samples:

- 4 classes × 4 images = 16 within-class hard-negative candidates
- 48 random images from the full 200-class pool

The OT schedule is identical to adaptive:

- `gate_sim=-4.0`
- `entropy_threshold=3.0`
- `alpha=0.05`

Only the batching strategy differs from OT-Mix adaptive, making this a cleaner test of whether adding a small amount of within-class structure improves OT-Mix.

| Ep | T→I R@1 | I→T R@1 | Avg R@1 | Note |
|---|---:|---:|---:|---|
| 5 | — | — | 0.06% | early |
| 10 | — | — | 0.45% | near adaptive |
| 15 | — | — | 0.68% | ahead of baseline/adaptive |
| 18 | — | — | 0.81% | competitive |
| 20 | — | — | 0.89% | ahead at same epoch |
| **21** | — | — | **0.98%** | current best |
| 22 | — | — | 0.69% | sharp dip |
| 25 | 0.64% | 0.85% | 0.74% | still below best |
| 26 | 0.81% | 0.93% | 0.87% | recovery |
| 27 | 0.79% | 1.05% | 0.92% | stable |
| 28 | 0.91% | 0.91% | 0.91% | stable |
| 29 | 0.10% | 1.10% | 0.60% | T→I rank-1 wobble |
| 30 | 0.72% | 1.16% | 0.94% | recovery |

OT triggered around epoch 9, after coupling entropy dropped below 3.0. Alpha ramps over roughly 1000 steps, so full alpha is reached around epoch 20.

The mixed run is highly non-monotone. However, the epoch 29 drop does not appear to be a full embedding collapse: Image → Text R@1 remained stable, and Text → Image R@5/R@10 stayed reasonable. The failure appears to be a fragile top-1 ranking issue, especially in the Text → Image direction.

Mechanistically, mixed batching is doing what it was designed to do. During useful OT steps, selected synthetic negatives often have:

- selected rank near 1–3
- coupling entropy around 2.3–2.6
- `Pos - Selected Gap` around -0.03 to -0.04

This means OT is finding hard local negatives. The unresolved question is whether this local hard-negative signal improves global retrieval by the end of training.

**Three-way comparison: canonical Avg R@1**

| Ep | Baseline | Adaptive | Mixed |
|---|---:|---:|---:|
| 10 | 0.41% | 0.47% | 0.45% |
| 15 | 0.58% | 0.54% | **0.68%** |
| 20 | 0.78% | 0.85% | **0.89%** |
| 21 | 0.87% | — | **0.98%** |
| 30 | 1.10% | 1.11% | 0.94% |
| 50 | **1.38%** | 1.28% | ? |

> **Verdict: promising mechanistically, inconclusive empirically.** Mixed batching improves access to meaningful fine-grained negatives, but the validation trajectory remains unstable. The run is not collapsed, but it has not yet matched the late-training strength of baseline or adaptive. The decisive window is epochs 30–50, where the baseline made its strongest gains.

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

> OT-Mix instability comes less from synthetic negatives themselves and more from applying OT pressure unconditionally. Gating should reduce validation oscillation while preserving the useful hard-negative regime: entropy 2.0–2.5, selected rank 1–3, and `Pos - Selected Gap` near the decision boundary.

**Key comparison metrics:**

| Metric | Baseline | Adaptive | Adaptive Gated |
|---|---:|---:|---:|
| Best canonical Avg R@1 | 1.38% | 1.35% | ? |
| Final canonical Avg R@1 | 1.38% | 1.28% | ? |
| Epoch-to-epoch std | ? | ? | ? |
| Largest validation dip | ? | ? | ? |
| Useful % | — | — | ? |
| Too easy suppressed % | — | — | ? |
| Too hard downweighted % | — | — | ? |
| Entropy suppressed % | — | — | ? |

The purpose of this run is not only to improve R@1. It is also to test whether conditional OT pressure reduces instability.

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
| 2026-04-26 | [`26-4-26-logs.md`](experiment_logs/26-4-26-logs.md) | Gating hypothesis: OT-Mix instability may come from unconditional pressure, not bad negatives. `compute_alpha_effective()` implemented with per-step entropy/gap gating. `cub200_softmax_mix_adaptive_gated` queued |

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
