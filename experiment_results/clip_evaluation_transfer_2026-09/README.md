# Completed evaluation-pool transfer

Run: `clip_evaluation_transfer_20260922T001127_444974Z` (September 22 rerun).
Three reused seeds, 15 states, 720 replayed branches observed on both old and new
pools. These are not 1,440 independent updates.

The [new-pool report](results/prediction_report.json) records alignment 15/15
correct signs versus timing 11/15 (13/13 versus 11/13 outside ±1e-6). Mean seed
balanced accuracy is 100% versus 72.22%, a +27.78 percentage-point difference.
The new pool is from a previously monitored test split: evaluation-pool transfer,
not untouched-test confirmation or a demonstrated training gain.

[Completion](results/completion.json) · [Raw states](results/states.json) ·
[Protocol](../../docs/clip-evaluation-transfer.md) ·
[Current interpretation](../../docs/research-status-2026-09-24.md).

`export_audit.json` records 133 verified non-weight members and 121 retained
files, including the original packing manifest and executed source overlay.
Weights/caption pools are omitted; checkpoint hashes are provenance, not a fresh
checkpoint audit. Run `python scripts/archive_clip_followup_evidence.py --verify-only`
from the repository root to check retained bytes. Original files are unmodified.
