# OTCO: Optimal Transport Contrastive Learning

Research code studying **when synthetic hard negatives help contrastive learning—and when they do not**. OTCO uses optimal transport to weight nearby mismatched embeddings and mix them into synthetic negatives. Uniform mixtures and real-negative controls test which parts actually matter.

The current evidence separates **hardness, gradient direction, actual optimizer updates, and downstream performance**. Staged one-step CLIP/CUB-200 studies cover six training seeds; the latest gated and exposure-matched rollouts cover three reused seeds. They do not establish a general curriculum rule or an OT-weighting advantage.

## Current research state — September 24, 2026

The [consolidated research record](docs/research-status-2026-09-24.md) brings together the completed evaluation-transfer, gated-training, exposure-matched, and saved-checkpoint studies. Small raw results, original manifests, source snapshots, and export audits are committed; large checkpoints and caption-bearing data remain outside Git.

The latest matched-branch diagnostic completed **12 states, 36 branches, and 1,824 optimizer updates**, with separate meta and reporting data. Sustained pressure improved reporting loss from all three update-100 states, but worsened it from all nine later states. The frozen gate predicted immediate effect signs in **11/12** states, versus **9/12** at the 50-update sustained endpoint. This motivates investigating horizon and starting-state dependence—not claiming a successful curriculum or a causal mechanism.

[Diagnostic results](experiment_results/clip_checkpoint_diagnostic_2026-09/README.md) · [Recheck exported evidence](scripts/archive_clip_followup_evidence.py) · [Colab diagnostic and recovery](docs/clip-checkpoint-diagnostic.md)

## Research progress

| Study | Status | Main finding |
|---|---|---|
| [Original CLIP ablations](docs/clip-experiments.md) | Completed | No auxiliary variant beats baseline final average R@1 in the eight 50-epoch runs |
| [Gradients across fine-tuning](docs/clip-gradient-stages.md) | Completed | The per-query synthetic margin proxy improves during native-only warmup |
| [Dense warmup / shared-head probes](docs/clip-warmup-readiness.md) | Completed | Weak positive head alignment at update 100 becomes near-zero or negative later |
| [Early-pulse intervention](docs/clip-early-pulse.md) | Completed | Tiny retrieval gain, lower species accuracy; no convincing overall benefit |
| [Paired actual-AdamW updates](docs/clip-paired-updates.md) | Completed, seed 42 | Synthetic update improves held-out loss in 16/16 early trials, 0/16 later trials; effects are small |
| [Two-seed replication](docs/clip-paired-seed-replication.md) | Completed | Early benefit repeats in seeds 123 and 456; later effects are smaller and change sign across seeds |
| [Intermediate checkpoints](docs/clip-paired-intermediate.md) | Completed | Five-state curves are non-monotonic and seed-dependent; no universal switch-off point |
| [Predictor screening](docs/clip-usefulness-predictors.md) | Completed | Alignment: 11/15 correct vs step: 9/15; tied at 9/12 after excluding near-zero effects |
| [Frozen-rule new-seed test](docs/clip-usefulness-prospective.md) | Completed | Alignment +6.94 points in mean seed balanced accuracy; ordinary and non-near-zero accuracy tied |
| [Replay + evaluation-pool transfer](experiment_results/clip_evaluation_transfer_2026-09/README.md) | Completed | New-pool alignment 15/15 versus timing 11/15; reused test split, not untouched-test confirmation |
| [Gated-training pilot](experiment_results/clip_gated_training_2026-09/README.md) | Completed, three seeds | Alignment minus baseline +0.00575 pp canonical R@1; unequal auxiliary exposure |
| [Exposure-matched timing control](experiment_results/clip_exposure_matched_2026-09/README.md) | Completed, three seeds | Alignment minus mean random timing +0.01151 pp; small, mixed seed effects |
| [Saved-checkpoint diagnostic](experiment_results/clip_checkpoint_diagnostic_2026-09/README.md) | Completed, 12 states | Immediate signs do not guarantee sustained benefit; no fresh-test or mechanism claim |

## Earlier milestone: frozen rules on new seeds

After mapping five checkpoints in development seeds 42, 123 and 456, freeze an alignment threshold and a timing comparator, then test **new seeds 789, 2026 and 31415** without refitting. The prospective run completed **720 paired branches / 15 seed-checkpoint states**.

| Frozen rule | Mean seed balanced accuracy | Correct states | Correct excluding ±1e-6 |
|---|---:|---:|---:|
| Full-gradient alignment | **90.28%** | 13/15 | 12/13 |
| Training step | 83.33% | 13/15 | 12/13 |

Alignment's primary advantage is **+6.94 percentage points**, but ordinary and non-near-zero accuracy are tied. It detects more helpful states at the cost of false positives. This is modest new-seed evidence, not a proven curriculum gain, statistical significance claim or independent-evaluation confirmation.

![New-seed usefulness curves and frozen-rule scores.](experiment_results/clip_usefulness_prospective_2026-09/results/prospective_usefulness.png)

The [five-checkpoint development study](docs/clip-paired-intermediate.md) shows non-monotonic, seed-dependent usefulness. The [predictor screening](docs/clip-usefulness-predictors.md) motivated this test. The original [three-seed endpoint comparison](docs/clip-paired-seed-replication.md) remains archived separately; endpoint replays are not additional independent seeds.

[Prospective findings and protocol](docs/clip-usefulness-prospective.md) · [720 new records and audit](experiment_results/clip_usefulness_prospective_2026-09/README.md) · [Intermediate evidence](experiment_results/clip_paired_intermediate_2026-09/README.md) · [Screening evidence](experiment_results/clip_usefulness_predictors_2026-09/README.md)

## Earlier evidence

The [13-epoch early-pulse intervention](docs/clip-early-pulse.md) adds synthetic pressure for 100 updates, then switches it off. Final canonical average R@1 is **1.631% vs 1.614%** for baseline (+0.017 percentage points, a net two retrieval successes), while species accuracy falls **0.259 points**. Thus an early one-step benefit has not translated into a convincing overall training gain. [Training differences](docs/figures/clip_early_pulse_differences.png).

The [original CLIP study](docs/clip-experiments.md) compares eight 50-epoch arms: baseline finishes at **1.968%** canonical average R@1, versus **1.907%** for uniform top-8 and **1.942%** for hardest-real at maximum α=0.5. Every arm's best species checkpoint is epoch zero. These longer-run values must not be directly ranked against the 13-epoch pulse values.

[Earlier ResNet-50 + DistilBERT experiments](docs/legacy-experiments.md) use a different architecture and SigLIP-style objective. They are historical evidence, not a controlled architecture comparison with CLIP.

## Run and reproduce

Requires Python 3.12+; GPU experiments use A100 Colab runtimes. See [setup and commands](docs/running-experiments.md), [Colab runners](colabs/), and each study's protocol. The [frozen-rule new-seed cell](colabs/clip_usefulness_prospective_one_cell.py) uses **no Drive writes**; newer transfer, gated, exposure-matched, and checkpoint-diagnostic launchers use verified Drive backups. Check the experiment's completion marker and final Drive flush, not just the launcher's status. Packaged protocols retain their preparation-time status; the dated consolidated record and evidence READMEs report completed outcomes.

```bash
uv sync
uv run python -m src.clip_train --config configs/hf_cub200_clip_vit_b32_baseline.yaml
```

Code: [CLIP objectives](model/clip_training.py), [training and diagnostics](src/), [configs](configs/), [tests](tests/). Evidence: [original CLIP archive](experiment_results/clip_2026-08-30_to_09-01/README.md), [curriculum studies](experiment_results/clip_curriculum_2026-09/README.md), [figures](docs/figures/README.md), [chronological logs](experiment_logs/).

Paper in preparation. The [original OT proposal](https://github.com/akashm776/ot-paper) records the initial hypotheses; conclusions now follow the experiments above.
