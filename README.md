# OTCO: Optimal Transport Contrastive Learning

Research implementation of **OT-Mix**, an optimal-transport-based method for generating structured synthetic negatives in multimodal contrastive learning.

> **Status:** Work in progress. Experiments ongoing. Paper in preparation.

---

## Overview

Standard contrastive learning treats all in-batch negatives equally. OT-Mix uses optimal transport to identify which images compete semantically for a given caption, then synthesizes a *barycentric* negative that represents aggregate semantic confusion rather than any single hard negative.

The core idea: instead of selecting the hardest in-batch negative, compute a soft transport plan over the top-k most similar images and mix them into a geometry-aware synthetic negative via barycentric projection.

```
L = L_base + α · L_OT
```

Where `L_base` is a SigLIP-style sigmoid contrastive loss and `L_OT` penalizes similarity between each text embedding and its OT-derived synthetic negative.

---

## Methods

| Method | Description |
|---|---|
| `baseline` | SigLIP sigmoid contrastive loss |
| `hard_negative` | SigLIP + explicit push on hardest in-batch negative |
| `ot_select` | SigLIP + softmax-weighted selection of hardest negative (OT-Select) |
| `softmax_mix` | SigLIP + barycentric synthetic negative via Sinkhorn OT **(proposed)** |
| `memory_bank` | SigLIP + hard negatives from a rolling queue of past embeddings |

---

## Architecture

- **Image encoder:** ResNet-50 → 512-d linear projection
- **Text encoder:** DistilBERT → 512-d linear projection
- **Shared space:** L2-normalized, cosine similarity
- **Dataset:** Flickr8K (local or HuggingFace) · Flickr30K (HuggingFace)

---

## Results

Preliminary results on Flickr8K (50 epochs, `both_last_layer` unfreezing, batch size 64):

| Method | T→I R@1 | I→T R@1 | Avg R@1 |
|---|---|---|---|
| baseline (SigLIP) | — | — | — |
| hard_negative | — | — | — |
| ot_select | — | — | — |
| **softmax_mix (OT-Mix)** | 14.07% | 17.41% | **15.74%** |
| memory_bank | — | — | — |

Full comparison table pending completion of all runs.

---

## Setup

```bash
# Install dependencies
uv sync
# or
pip install -e .
```

Requires Python ≥ 3.12, PyTorch ≥ 2.9.

**Local Flickr8K:** place dataset at `data/datasets/Flickr8k/` with `Images/` and `captions.txt`.

**HuggingFace datasets** (Flickr8K or Flickr30K): set `dataset.backend: hf_flickr8k` or `hf_flickr30k` in the run config — downloads automatically.

---

## Running Experiments

```bash
# Run a named experiment
python -m src.main --config configs/default.yaml

# Switch experiments by editing configs/default.yaml → experiment.name
# Available: baseline, hard_negative_a025, hard_negative_a05,
#            softmax_mix_k16, softmax_mix_k32, ot_select, memory_bank
```

Run on HuggingFace Flickr30K:

```bash
python -m src.main --config configs/hf_flickr30k.yaml
```

Results are logged to `experiments/exp_<timestamp>.json`. Compare all runs:

```bash
python -m src.analyze_log
```

---

## Configuration

Experiments use a two-level YAML system:

- `configs/default.yaml` — selects which experiment to run and sets dataset/runtime options
- `configs/experiments.yaml` — registry of all named experiment hyperparameter sets

Override individual fields without editing the registry:

```yaml
experiment:
  name: softmax_mix_k32
  overrides:
    num_epochs: 10
    alpha: 0.2
```

---

## Key Hyperparameters (OT-Mix)

| Parameter | Description |
|---|---|
| `top_k` | Local neighborhood size for OT support |
| `ot_eps` | Sinkhorn entropy regularization |
| `sinkhorn_iters` | Number of Sinkhorn iterations |
| `update_freq` | Steps between OT plan recomputation |
| `warmup_steps` | Steps before OT loss is activated |
| `alpha` | Max weight of OT loss term (ramps up after warmup) |

---

## Citation

Paper in preparation.
