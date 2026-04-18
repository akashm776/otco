# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Collaboration mode

This repo operates in **pairing mode** — Claude works *with* the user, not autonomously for them. The default is the smallest well-defined next step, with reasoning visible through diffs, commands, and short summaries.

### For every task, follow this order

1. **Restate narrowly** — one or two sentences, reduced to the smallest meaningful unit
2. **Define the contract** — state Input / Operation / Output / Validation before writing code
3. **Execute minimally** — the smallest change that satisfies the contract
4. **Validate immediately** — run the narrowest relevant check (test, lint, metric, script output)
5. **Report briefly** — what changed · what was validated · exact result · open uncertainty · next step

### Stop and ask before proceeding if
- the task requires a large design decision
- multiple interpretations are equally plausible
- the requested change is expensive, long-running, or destructive
- the task would touch multiple unrelated files or systems

### Scope discipline
- Do exactly what was asked. Do not fix adjacent issues.
- If a nearby issue is noticed, record it under **"Possible follow-ups"** instead of expanding scope.
- Pause before: architecture changes, dependency additions, data schema changes, large deletions, long-running experiments.

### Research / experiment workflow
For research tasks, follow this loop: **Question → Current understanding → Gap → Executable test → Run → Result → Structural update → Next question**. Log each cycle in `research_log.md` (create only if needed).

### Reporting format
Unless asked otherwise:

```
### Summary
One short paragraph.

### Changed
- file A · what changed

### Validation
- command run
- result

### Uncertainty
- what is still unknown

### Next smallest step
- one suggested action
```

### Working files (create only when needed)
- `plans.md` — milestone list, acceptance criteria, out-of-scope items
- `implement.md` — exact commands, validation steps, required outputs
- `documentation.md` — what was done, decisions, how to reproduce, known issues
- `research_log.md` — structured experiment log entries

## Commands

**Install dependencies** (uses `uv`):
```bash
uv sync
# or
pip install -e .
```

**Run training**:
```bash
python -m src.main --config configs/default.yaml
```

**Run diagnostic/evaluation on a checkpoint**:
```bash
python -m src.test --config configs/diagnostic.yaml
```

**Analyze experiment logs**:
```bash
python -m src.analyze_log
```

## Configuration System

Experiments are configured via a two-level YAML system:

1. **Run config** (e.g. `configs/default.yaml`): Specifies which named experiment to run, the dataset backend, checkpoint directory, and optional per-run overrides.
2. **Experiments registry** (`configs/experiments.yaml`): Defines all named experiment configs (hyperparameters, loss type, model names). The run config's `experiment.name` key selects which entry to use.

To run a different experiment, edit `configs/default.yaml` → `experiment.name` to any key in `configs/experiments.yaml`, or add `experiment.overrides` to patch specific fields without modifying the registry.

**Dataset backends** (set via `dataset.backend` in the run config):
- `local_flickr8k`: reads from `data/datasets/Flickr8k/` on disk
- `hf_flickr8k`: downloads `nlphuji/flickr8k` from HuggingFace Hub

## Architecture

### Model (`model/model.py`)
`OTLIP` is a dual-encoder CLIP-like model: ResNet-50 (2048-d) + DistilBERT (768-d), both projected via linear heads to a shared 512-d L2-normalized space. Temperature is a fixed scalar (`temp=0.07`).

### Loss functions (`model/loss.py`)
All losses except `SigLIPLoss` return `(total_loss, loss_dict)`. The `loss_dict` carries per-component scalars for logging.

| Class | `loss_type` key | What it does |
|---|---|---|
| `SigLIPLoss` | `baseline` | Sigmoid contrastive loss over all B² pairs |
| `HardNegativeLoss` | `hard_negative` | SigLIP + explicit push on hardest in-batch negative |
| `OTSelectLoss` | `ot_select` | SigLIP + selects hardest negative via softmax over top-k similarities |
| `SoftmaxMixLoss` | `softmax_mix` | **Main method.** SigLIP + OT-derived barycentric synthetic negative |
| `MemoryBankLoss` | `memory_bank` | SigLIP + hard negatives drawn from a rolling queue of past embeddings |

**`SoftmaxMixLoss` in detail** (the paper's OT-Mix):
- Builds a cost matrix `C_ij = 1 - cos(z_i^T, z_j^I)`, restricted to top-k neighbors per text
- Runs Sinkhorn iterations to get transport plan Π
- Synthesizes a barycentric negative: `ẑ_i^I = Σ_j (Π_ij / row_sum) * z_j^I`
- Gradients are **stopped through Π** — OT acts as a geometry-derived augmentation
- Π is recomputed every `update_freq` steps (not every step)
- α ramps up linearly from 0 after `warmup_steps`

### Data (`data/`)
Two parallel dataset implementations: `flickr8k_dataset.py` (local) and `hf_flickr8k_dataset.py` (HuggingFace). Each exposes three dataset classes:
- `*UniqueImageDataset` — training: samples one random caption per image per epoch
- `*AllCaptionsDataset` — validation: all 5 captions per image
- `*CanonicalCaptionDataset` — validation: only first caption per image (the "official" metric)

### Training loop (`src/main.py`)
- Calls `build_data_bundle` → `setup_model_and_optimizer` → `build_loss`
- Logs every 100 steps with logit statistics, batch retrieval proxy, and embedding health metrics
- Validates with two loaders (all captions + canonical) and two directions (T→I, I→T)
- Best model saved by canonical average R@1; periodic checkpoints every 5 epochs
- Logs final metrics to `experiments/exp_<timestamp>.json` via `log_experiment`

### Experiment results (`experiments/*.json`)
Each completed run produces a timestamped JSON in `experiments/`. `src/analyze_log.py` and `src/utils.py:print_experiment_comparison` read these to compare runs side-by-side.

## Key hyperparameters in `experiments.yaml`

- `loss_type`: selects the loss class (see table above)
- `unfreeze_strategy`: `projection_only` | `text_last_layer` | `vision_last_layer` | `both_last_layer` — controls which parameters are trainable
- `warmup_steps`: steps before OT/hard-negative auxiliary loss is activated
- `top_k` / `tau` / `ot_eps` / `sinkhorn_iters` / `update_freq`: OT-specific knobs for `softmax_mix`
- `alpha`: max weight of the auxiliary loss term; ramps up linearly over 1000 steps post-warmup

## Paper context

`OTCO-5.pdf` in the repo root is the LaTeX source for the NeurIPS-style paper describing this project. The results tables in the paper are currently empty — experiments are ongoing. The paper's Algorithm 1 specifies a 2-way softmax for L_OT, but the implementation uses a sigmoid binary loss (consistent with the prose in §3.3).
