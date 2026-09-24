# Consolidated research record — September 24, 2026

This is the current interpretation of the completed local evidence, superseding
pre-run status text in the packaged protocol documents. It does not change the
executed protocols, source bundles, thresholds, or historical result files.
Preparation documents remain in the source snapshots and current Colab bundles
so the launchers retain their executed identities.

## Completed follow-ups

| Study | Training seeds | Main descriptive result | Limitation |
|---|---:|---|---|
| [Evaluation-pool transfer](../experiment_results/clip_evaluation_transfer_2026-09/README.md) | 3 reused | Alignment 15/15 signs; timing 11/15; mean seed balanced accuracy 100% vs 72.22% | Previously monitored test split; both pools observe the same 720 branches |
| [Gated training](../experiment_results/clip_gated_training_2026-09/README.md) | 3 reused | Alignment minus native final canonical R@1 +0.00575 pp | Auxiliary exposure differs: 400/400/401 alignment, 76 timing, 901 always-on |
| [Exposure-matched timing](../experiment_results/clip_exposure_matched_2026-09/README.md) | 3 reused | Alignment minus mean random timing +0.01151 pp R@1; species −0.00767 pp | Controls nested within seeds; burst structure, gradient dose and compute not matched |
| [Checkpoint diagnostic](../experiment_results/clip_checkpoint_diagnostic_2026-09/README.md) | 3 reused, 12 correlated states | Sustained 50-update reporting-loss difference +0.00049330; reporting-pool R@1 −0.12207 pp | One branch stream per state; reused diagnostic data |

The interventions use **uniform top-8 synthetic mixtures**, not optimized OT
weights. They cannot establish that optimal transport is necessary or superior.
Small positive retrieval means are not a significance claim or convincing
end-to-end curriculum improvement. Reused trajectories/seeds must not be counted
as additional independent replication.

## What the latest diagnostic changes

From each saved state, restore model, AdamW, scheduler and RNG, then compare native
50 updates, one auxiliary pulse followed by 49 native updates, and 50 sustained
auxiliary updates. Meta and reporting sets contain 512 disjoint images each;
both come from the previously used 1,024-image diagnostic holdout. Predictions
were recorded before reporting evaluation and did not control the branches.
The fresh, precommitted 50-step stream is matched across branches, not a literal
resumption of the old training loader.

Loss differences are treatment minus matched native (negative helps).

| Starting update | Mean sustained reporting-loss difference at 50 | Helpful seeds |
|---|---:|---:|
| 100 | −0.00118177 | 3/3 |
| 250 | +0.00170098 | 0/3 |
| 500 | +0.00097383 | 0/3 |
| 750 | +0.00048016 | 0/3 |

Frozen alignment and meta-gradient rules agree on all 12 actions. Each gets
11/12 immediate signs right (10/10 outside ±1e-6), but only 9/12 sustained signs.
The two AdamW-aware scores also agree on all 12 actions: 10/12 immediate and
10/12 sustained signs. This small descriptive difference does not validate a new
gate. Always-off already gets 9/12 sustained signs right. A checkpoint-100-only
rule fits these sustained outcomes, but is a post-hoc observation, not a
prospectively validated replacement rule.

The pulse's mean 50-step loss difference is +0.00000608, with mixed seed effects.
Even at update 100, where sustained loss improves in all seeds, reporting-pool
retrieval worsens in two seeds. Loss benefit and retrieval benefit are different
claims. Three sustained loss signs differ between the two reporting partitions.

The experiment did **not** repeatedly probe readiness after treatment. It cannot
establish that synthetic pressure “consumes readiness,” distinguish that mechanism
from other trajectory effects, or identify a universal switch-off time.

## Literature-review synthesis

Three local PDF reviews prompted the diagnostic. Their useful common distinction
is local gradient compatibility versus held-out effect versus persistence versus
training endpoints. Candidate-gate rankings and causal stories remain hypotheses.
Unresolved file-citation placeholders are not usable references. These working
reviews are not additional experimental evidence.

The prior key-reference review emphasized [Ren et al.](https://proceedings.mlr.press/v80/ren18a.html)
for validation-based meta-weighting, [Du et al.](https://arxiv.org/abs/1812.02224)
as essential alignment-weighting precedent, and
[Shin and Alvarez-Melis](https://arxiv.org/abs/2609.09099) for curriculum analysis.
None should be presented as proving our synthetic-negative AdamW schedule or
held-out retrieval benefit. That review checked key references, not every claim
or citation in all three PDFs. See the [defense workbook](research-defense.md)
and [study plan](research-study-plan.md).

## Next scientific step — proposed, not run

Replicate bounded branches with fresh, precommitted matched training streams and
a fixed reporting time course (for example 1, 5, 10, 25, 50 updates). Add
diagnostic-only reprobes on both native and auxiliary trajectories; do not let
reporting outcomes tune the gate. Report seed-level and partition sensitivity
and keep loss and retrieval separate. A change in probe sign alone would still
not establish a causal mechanism. This consolidation launches no GPU experiment.

## Evidence and implementation preservation

All four exports retain small raw results, original manifests and executed source
snapshots. The exporter verifies ZIP CRC and manifest SHA256/size for **475
non-weight members**, retaining 428 files (about 10 MB). Caption datasets, full
stdout and large weights are excluded. Original manifests still list omitted
files as provenance; that does not mean they are in Git or that this export
reverified large checkpoints. Downloaded originals are unchanged. Mounted-Drive
verification and remote flush are distinct.

```bash
python scripts/archive_clip_followup_evidence.py --verify-only
python scripts/verify_research_defense_evidence.py
python -m pytest -q
```

Recovery tools preserve 12 pinned input checkpoints without retraining old
trajectories or substituting later states. The completed diagnostic used bundle
`9e537567968d61813b329a668ba9e7db667463e6a63f38f686b269543b85bfdc`.
Its separate recovery ZIP is approximately 8.31 GB and intentionally outside Git.
Cloning this lightweight repository does not recover the checkpoint tensors.
