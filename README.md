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
| [Paired actual-AdamW updates](docs/clip-paired-updates.md) | Completed | Synthetic update improves held-out loss in 16/16 early trials, 0/16 later trials; effects are small |

## Latest completed experiment: actual optimizer updates

Restore the same baseline model **and AdamW state**, take one native-only or auxiliary-augmented update, then compare loss on 1,024 held-out examples. Repeat with 16 fixed training batches at updates 100 and 1,001: **96 branches**, all completed and audited on A100.

| Auxiliary | Early: extra held-out loss | Later: extra held-out loss |
|---|---:|---:|
| Uniform-top-8 synthetic | −0.00002563; **16/16 beneficial** | +0.00000272; **0/16 beneficial** |
| Hardest real | +0.00000543; 6/16 beneficial | +0.00000193; 7/16 beneficial |

Negative means better than the native-only update, not an accuracy gain. Intuitively, the synthetic negative gives a tiny useful nudge early and a tiny counterproductive nudge later. This is evidence of a **state-dependent one-step effect**, not proof of a successful curriculum, an exact activation window, or generality beyond this CLIP/CUB run. Both updates still reduce loss from their starting checkpoints on average.

![Paired one-step effects, same vertical scale at both checkpoints. Below zero is better than native-only.](docs/figures/clip_paired_update_effects.png)

[Protocol, findings and limitations](docs/clip-paired-updates.md) · [All 96 records and audit](experiment_results/clip_paired_updates_2026-09/README.md)

## Earlier evidence

The [13-epoch early-pulse intervention](docs/clip-early-pulse.md) adds synthetic pressure for 100 updates, then switches it off. Final canonical average R@1 is **1.631% vs 1.614%** for baseline (+0.017 percentage points, a net two retrieval successes), while species accuracy falls **0.259 points**. Thus an early one-step benefit has not translated into a convincing overall training gain. [Training differences](docs/figures/clip_early_pulse_differences.png).

The [original CLIP study](docs/clip-experiments.md) compares eight 50-epoch arms: baseline finishes at **1.968%** canonical average R@1, versus **1.907%** for uniform top-8 and **1.942%** for hardest-real at maximum α=0.5. Every arm's best species checkpoint is epoch zero. These longer-run values must not be directly ranked against the 13-epoch pulse values.

[Earlier ResNet-50 + DistilBERT experiments](docs/legacy-experiments.md) use a different architecture and SigLIP-style objective. They are historical evidence, not a controlled architecture comparison with CLIP.

## Run and reproduce

Requires Python 3.12+; GPU experiments use A100 Colab runtimes. See [setup and commands](docs/running-experiments.md), [Colab runners](colabs/), and each study's protocol. Mount Drive before long runs; a completed notebook launcher cell does not by itself prove training has finished.

```bash
uv sync
uv run python -m src.clip_train --config configs/hf_cub200_clip_vit_b32_baseline.yaml
```

Code: [CLIP objectives](model/clip_training.py), [training and diagnostics](src/), [configs](configs/), [tests](tests/). Evidence: [original CLIP archive](experiment_results/clip_2026-08-30_to_09-01/README.md), [curriculum studies](experiment_results/clip_curriculum_2026-09/README.md), [figures](docs/figures/README.md), [chronological logs](experiment_logs/).

Paper in preparation. The [original OT proposal](https://github.com/akashm776/ot-paper) records the initial hypotheses; conclusions now follow the experiments above.
