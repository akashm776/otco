# Completed frozen-gate training pilot

Run: `clip_gated_training_20260922T114312_608272Z`.
Three seeds × four arms, each 1,001 updates with a common 100-update native prefix.

[Comparison](comparison.json): alignment minus baseline canonical average R@1
is +0.005753 pp; minus step-gated +0.008630 pp; minus always-on −0.011506 pp.
Alignment uses 400/400/401 auxiliary updates, step-gated 76 and always-on 901.
These tiny mixed effects do not isolate timing from exposure or demonstrate a
reliable curriculum gain. Species retention is not consistently improved.

[Completion](completion.json) · [Protocol](../../docs/clip-gated-training.md) ·
[Current interpretation](../../docs/research-status-2026-09-24.md).

`export_audit.json` records 130 verified non-weight members and 119 retained files
from nine downloaded ZIP parts. Per-arm metrics, gate audits, prefix hashes and
executed source are preserved. Checkpoints and caption batches are omitted;
their original manifest entries are provenance, not fresh tensor verification.
Run `python scripts/archive_clip_followup_evidence.py --verify-only` to recheck.
The run manifest's `complete_pending_drive_flush` is not proof of remote flush.
