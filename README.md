# OTCO: Optimal Transport Contrastive Learning

Research code for studying whether **Optimal Transport (OT)** can generate useful **synthetic hard negatives** for multimodal contrastive learning.

## Status

This is an **active research repository**. The codebase is functional and experiments are ongoing, but results should be treated as **exploratory rather than final**.

The goal of this repo is not to assume OT-generated negatives always help. The goal is to understand:

> **When do OT-derived synthetic negatives improve contrastive learning, and when do they hurt?**

---

## Research Question

Standard contrastive learning often relies on random in-batch negatives or simple hard-negative heuristics. Many of these negatives are too easy to be useful.

OTCO investigates whether Optimal Transport can improve this process by:

- identifying semantically close but mismatched examples
- constructing synthetic negatives through barycentric mixing
- injecting harder training signal than standard random sampling

This repository is focused on **hard-negative generation**, **training dynamics**, and **the regimes in which OT-based negatives are useful**.

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
| Flickr8K | Early diagnostics — OT neighborhood quality, false-negative pressure |
| Flickr30K | Null result — OT adds no signal on generic captions (see below) |
| **CUB-200-2011** | **Active** — fine-grained bird retrieval, 200 species, Reed et al. captions |

These datasets span a range of retrieval difficulty and negative granularity, chosen to test in which regimes OT-generated negatives are useful.

---

## Current Findings

A few conclusions are already clear:

- OT-derived synthetic negatives are **not automatically helpful**.
- Performance depends strongly on encoder quality, the candidate negative pool, batching strategy, and OT activation schedule.
- Some settings produce **null results**, which is an important part of the research story.
- Current work is focused on understanding **why** OT helps in some regimes and fails in others.

This repository should be read as a **research artifact**, not a finished benchmark report.

---

## Results

### Flickr30K — CONCLUDED (null result)

> **Finding: OT-Mix adds no signal on Flickr30K. Random in-batch negatives are too easy for OT to find meaningful transport structure.**

ResNet-50 + DistilBERT, 512-d, canonical R@1 (first caption, full val pool):

| Experiment | Loss | Start | Epochs | Best Avg R@1 | Verdict |
|---|---|---|---|---|---|
| Baseline | SigLIP | scratch | 50 | 32.10% | reference |
| OT-Mix fine-tune (α=0.05) | SigLIP + OT | baseline ckpt | 30 | 32.50% (ep28) | null — matched by continued baseline |
| Continued baseline | SigLIP | baseline ckpt | 30 | 32.50% (ep17) | null — same gain, no OT |
| OT-Mix scratch (α=0.05, adaptive) | SigLIP + OT | scratch | 50 | 31.25% | null — worse than baseline |

The +0.40% from OT-Mix fine-tune is entirely explained by extra gradient steps. SigLIP alone, from the same checkpoint, reaches the same result 11 epochs faster. Flickr30K captions are generic high-level descriptions; in-batch random negatives are already semantically distant and OT finds no meaningful structure.

---

### CUB-200 — ACTIVE

Fine-grained bird retrieval: 200 species, 10 Reed et al. attribute-rich captions per image, 5794-image validation pool. Within-class negatives are visually and semantically confusable — the regime where OT should add signal.

All runs: ResNet-50 + DistilBERT, 512-d, `both_last_layer` unfreezing, batch size 64, 50 epochs.

#### Baseline — COMPLETE

| Ep | T→I R@1 | I→T R@1 | Avg R@1 |
|---|---|---|---|
| 10 | 0.38% | 0.45% | 0.41% |
| 20 | 0.64% | 0.91% | 0.78% |
| 30 | 0.81% | 1.38% | 1.10% |
| 40 | 0.81% | 1.12% | 0.97% |
| **50** | **1.05%** | **1.71%** | **1.38%** |

Note non-monotone trajectory — gains accelerate again at ep45–50.

#### OT-Mix adaptive — RUNNING (ep1–19 observed)

Random batching. `gate_sim=-4.0`, `entropy_threshold=3.0`, `alpha=0.05`, adaptive OT warmup.

| Ep | Avg R@1 | vs Baseline (same ep) |
|---|---|---|
| 10 | ~0.41% | ≈ flat |
| 16 | 0.73% | −0.05% |
| 17 | 0.68% | −0.10% |
| 18 | 0.79% | +0.01% |

> **Verdict: inconclusive.** OT plan reaches rank-1 selection (same-species negatives) by ep10, confirming OT finds meaningful structure on CUB-200. Performance roughly tracks baseline through ep18. Full 50-epoch run needed.

#### OT-Mix stratified — RUNNING (ep1–22 observed)

Stratified batching: K=16 classes × 4 images = B=64. `gate_sim=-4.5`, `entropy_threshold=3.5`.

| Ep | Avg R@1 | vs Baseline (same ep) |
|---|---|---|
| 10 | ~0.41% | ≈ flat |
| 16 | ~0.60% | −0.13% |
| 21 | 0.71% | −0.16% |

> **Verdict: inconclusive and confounded.** Two things changed from adaptive: (1) stratified batching, and (2) a more permissive OT schedule (entropy_threshold 3.0→3.5, gate_sim −4.0→−4.5). Cannot cleanly attribute underperformance to batching vs OT schedule. Full 50-epoch run in progress for data; not used as a design conclusion.

#### OT-Mix mixed batching — QUEUED

25% stratified (4 classes × 4 images = 16 within-class hard negatives) + 75% random (48 images, full 200-class diversity). OT schedule identical to adaptive (`gate_sim=-4.0`, `entropy_threshold=3.0`). **Only the batching strategy differs** — isolates batching effect cleanly.

> **Hypothesis:** mixed batching gives OT the within-class hard negatives it needs while restoring the cross-class diversity that pure stratified sacrifices.

---

## Experiment Logs

Chronological research log in [`experiment_logs/`](experiment_logs/):

| Date | File | Summary |
|---|---|---|
| 2026-03-11 | [`11-3-26-logs.md`](experiment_logs/11-3-26-logs.md) | OT diagnostic on Flickr8K: negatives are semantically plausible but false-negative pressure is high (P(neg1 > GT) = 0.65) |
| 2026-04-17 | [`17-4-26-logs.md`](experiment_logs/17-4-26-logs.md) | Plan: re-run diagnostic after better encoder convergence; hypothesis that geometry separation reduces false-negative pressure |
| 2026-04-22 | [`22-4-26-logs.md`](experiment_logs/22-4-26-logs.md) | **Key finding:** cosine-space OT is degenerate with SigLIP embeddings. Logit-space OT fixes it. Cosine entropy ≈ 3.33 (uniform, rank 17); logit entropy ≈ 2.0–2.5 (sharp, rank 1–2) |
| 2026-04-23 | [`23-4-26-logs.md`](experiment_logs/23-4-26-logs.md) | α=0.1 over-destabilizes converged Flickr30K model (−3.15% dip, never recovers). α=0.05 reduces dip to −1.45%, peaks at 32.50% |
| 2026-04-24 | [`24-4-26-logs.md`](experiment_logs/24-4-26-logs.md) | **Null result confirmed:** continued baseline reaches 32.50% at ep17 — 11 epochs before OT-Mix. Decision to move to CUB-200 |

---

## Technical Notes

### Why logit-space OT

SigLIP's learned `logit_bias` compresses cosine similarities into a very narrow range (~0.27 spread). Building the OT cost matrix in cosine space gives `exp(-C/ε)` a near-uniform Gibbs kernel — Sinkhorn produces near-uniform plans regardless of embedding geometry. Moving to logit space (~14× range) restores discriminative signal. Verified empirically: cosine-space OT, coupling entropy ≈ 3.33, selected rank ≈ 17 (chance); logit-space OT, entropy ≈ 2.0–2.5, rank ≈ 1–2.

### OT-Mix hyperparameters

| Parameter | Description |
|---|---|
| `top_k` | Local neighborhood size for OT support (32) |
| `ot_eps` | Sinkhorn entropy regularization — calibrated for logit space (0.7) |
| `sinkhorn_iters` | Sinkhorn iterations (30) |
| `update_freq` | Steps between OT plan recomputation (10) |
| `gate_sim` | Logit-space threshold; synthetics below this are excluded from the loss |
| `alpha` | Max weight of OT loss; ramps linearly over 1000 steps after `ot_ready` |
| `adaptive_warmup` | Wait for coupling entropy < `entropy_threshold` before activating OT |
| `entropy_threshold` | Trigger level — log(32) ≈ 3.47 is uniform; healthy operating range ≈ 2.0–2.5 |

---

## Setup

```bash
uv sync
# or
pip install -e .
```

Requires Python ≥ 3.12, PyTorch ≥ 2.9.

**CUB-200 (HuggingFace):** set `dataset.backend: hf_cub200` — downloads automatically. Requires `datasets<3.0`.

**Local Flickr8K:** place dataset at `data/datasets/Flickr8k/` with `Images/` and `captions.txt`.

---

## Running Experiments

```bash
# Run a named experiment
python -m src.main --config configs/default.yaml

# Evaluate a checkpoint
python -m src.test --config configs/diagnostic.yaml

# Compare all logged runs
python -m src.analyze_log
```

Switch experiments by editing `configs/default.yaml → experiment.name` to any key in `configs/experiments.yaml`, or use `experiment.overrides` to patch fields without modifying the registry.

**CUB-200 run configs** (for Colab/Kaggle):
- `configs/hf_cub200_baseline.yaml`
- `configs/hf_cub200_softmax_mix_adaptive.yaml`
- `configs/hf_cub200_softmax_mix_stratified.yaml`
- `configs/hf_cub200_softmax_mix_mixed.yaml`

Results log to `experiments/exp_<timestamp>.json` and to `results/` when run via the Colab scripts in `colabs/`.

---

## Citation

Paper in preparation.
