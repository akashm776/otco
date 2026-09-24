# Completed saved-checkpoint diagnostic

Run: `clip_checkpoint_diagnostic_20260924T122020_801040Z`.
Bundle: `9e537567968d61813b329a668ba9e7db667463e6a63f38f686b269543b85bfdc`.

[Completion](completion.json): 12 states × 3 matched 50-update branches = 1,800
updates, plus 24 predictor-trial updates. Separate 512-image meta/reporting sets
are drawn from a reused holdout. Predictions are committed before reporting;
branches do not adapt to them. Saved input states are read-only, with full
AdamW/scheduler restoration and replay/RNG/LR checks in every `result.json`.

| Mean treatment minus native | Pulse | Sustained |
|---|---:|---:|
| Immediate reporting loss | −0.00000401 | −0.00000401 |
| Reporting loss at 50 | +0.00000608 | +0.00049330 |
| 512-image canonical R@1 at 50, pp | −0.01628 | −0.12207 |

[Comparison](comparison.json): frozen alignment gets 11/12 immediate signs and
9/12 sustained signs right; AdamW-aware scores get 10/12 for each. Always-off also
gets 9/12 sustained signs. All three update-100 states benefit in sustained loss;
all nine later states worsen. That pattern is descriptive, not a validated new
time gate. Retrieval and loss disagree in some states. Only three trajectories
and one stream per state are tested. No post-treatment readiness time course was
measured, so “readiness consumption” remains a hypothesis.

[Protocol/recovery instructions](../../docs/clip-checkpoint-diagnostic.md) ·
[Interpretation and proposed next step](../../docs/research-status-2026-09-24.md).

`export_audit.json` records all 70 original manifest-listed files checked against
SHA256/size and 57 retained files including the manifest. Caption-bearing data
roles and training batches are intentionally omitted; their hashes remain in the
original manifest. Branch index plans, raw results, committed predictions and
executed source overlay are retained. No weights were written by this experiment.
Input checkpoint tensors are not included in Git. Run
`python scripts/archive_clip_followup_evidence.py --verify-only` to recheck the
retained files. Byte verification does not independently re-execute GPU checks
or certify the notebook's final remote Drive flush.
