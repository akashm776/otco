# OTCO: Optimal Transport Contrastive Learning

Research code studying **when synthetic hard negatives help contrastive learning—and when they do not**. OTCO uses optimal transport to weight nearby mismatched embeddings and mix them into synthetic negatives. Uniform mixtures and real-negative controls test which parts actually matter.

The current evidence separates **hardness, gradient direction, actual optimizer updates, and downstream performance**. Results below are exploratory, primarily single-seed CLIP experiments on CUB-200; they do not establish a general curriculum rule.

## Research progress

| Study | Status | Main finding |
|---|---|---|
| [Original CLIP ablations](docs/clip-experiments.md) | Completed | No auxiliary variant beats baseline final average R@1 in the eight 50-epoch runs |
| [Gradients across fine-tuning](docs/clip-gradient-stages.md) | Completed | The per-query synthetic margin proxy improves during native-only warmup |
| [Dense warmup / shared-head probes](docs/clip-warmup-readiness.md) | Completed | Weak positive head alignment at update 100 becomes near-zero or negative later |
| [Early-pulse intervention](docs/clip-early-pulse.md) | Completed | Tiny retrieval gain, lower species accuracy; no convincing overall benefit |
| [Paired actual-AdamW updates](docs/clip-paired-updates.md) | Implemented; GPU results pending | Test incremental held-out loss change from a single auxiliary-augmented update |

## Latest completed experiment: early pulse

Same model and optimizer state through update 100; apply synthetic or hardest-real auxiliary pressure for exactly 100 updates, then return to native CLIP training through update 1,001. This is **13 epochs on the original 50-epoch learning-rate schedule**, not the older epoch-50 comparison. Both coefficients were fixed using training-only gradient calibration.

| Arm | Final canonical average R@1 | Species top-1 |
|---|---:|---:|
| Native CLIP baseline | 1.614% | 45.823% |
| Uniform-top-8 synthetic pulse | 1.631% | 45.564% |
| Hardest-real pulse | 1.614% | 45.858% |

The synthetic gain is **+0.017 percentage points**—a net two top-1 retrieval successes across the two directions—while species accuracy falls **0.259 points**. Differences fluctuate during training. One seed and a repeatedly inspected evaluation set do not establish a reliable gain.

![Early-pulse differences from baseline; shaded interval is the intervention.](docs/figures/clip_early_pulse_differences.png)

[Experiment, controls, diagnostics and limitations](docs/clip-early-pulse.md) · [Numeric evidence and archive hashes](experiment_results/clip_curriculum_2026-09/README.md) · [Full training curves](docs/figures/clip_early_pulse_performance.png)

## Earlier evidence

The [original CLIP study](docs/clip-experiments.md) compares eight 50-epoch arms: baseline finishes at **1.968%** canonical average R@1, versus **1.907%** for uniform top-8 and **1.942%** for hardest-real at maximum α=0.5. Every arm's best species checkpoint is epoch zero. These longer-run values must not be directly ranked against the 13-epoch pulse table above.

[Earlier ResNet-50 + DistilBERT experiments](docs/legacy-experiments.md) use a different architecture and SigLIP-style objective. They are historical evidence, not a controlled architecture comparison with CLIP.

## Run and reproduce

Requires Python 3.12+; GPU experiments use A100 Colab runtimes. See [setup and commands](docs/running-experiments.md), [Colab runners](colabs/), and each study's protocol. Mount Drive before long runs; a completed notebook launcher cell does not by itself prove training has finished.

```bash
uv sync
uv run python -m src.clip_train --config configs/hf_cub200_clip_vit_b32_baseline.yaml
```

Code: [CLIP objectives](model/clip_training.py), [training and diagnostics](src/), [configs](configs/), [tests](tests/). Evidence: [original CLIP archive](experiment_results/clip_2026-08-30_to_09-01/README.md), [curriculum studies](experiment_results/clip_curriculum_2026-09/README.md), [figures](docs/figures/README.md), [chronological logs](experiment_logs/).

Paper in preparation. The [original OT proposal](https://github.com/akashm776/ot-paper) records the initial hypotheses; conclusions now follow the experiments above.
