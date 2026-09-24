# Completed exposure-matched timing control

Run: `clip_exposure_matched_20260923T114154_690094Z`.
Three reused seeds, alignment reference plus three random-timing controls per seed.
Auxiliary-update counts match within each seed (400, 400, 401 respectively).

[Comparison](comparison.json): alignment minus mean random canonical R@1 is
+0.011506 pp, with per-seed differences +0.005753, +0.031642, −0.002877 pp.
Mean species difference is −0.007670 pp. Nine random schedules are not nine
independent training seeds. Equal active-update counts do not match gradient
dose, burst structure or compute. No significance or robust timing advantage is
established.

[Completion](completion.json) · [Protocol](../../docs/clip-exposure-matched.md) ·
[Current interpretation](../../docs/research-status-2026-09-24.md).

`export_audit.json` records 142 verified non-weight members and 131 retained files
from nine ZIP parts. Raw metrics, gate schedules and source snapshots are kept;
weights and caption batches are excluded. Original manifests retain their hashes
but this export does not rehash tensors or certify remote Drive flush.
Run `python scripts/archive_clip_followup_evidence.py --verify-only` to recheck.
