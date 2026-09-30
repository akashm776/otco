# Partial marginal-utility run — incomplete, not extra evidence

Source: `clip_marginal_utility_20260930T000856_646786Z`, stored by the user under
Drive's `failed runs`. Original run status is `running`; backup status is
`in_progress`; there is no completion marker. These statuses are preserved.

All 144 listed source members passed ZIP CRC and pinned manifest hash/size checks.
136 files including the manifest are retained; nine caption-bearing files omitted.
This authenticates the **partial export**, not experiment completion.

`overlap_audit.json` checks that its six completed pair results exactly reproduce
six of the twelve pairs in `../clip_marginal_utility_2026-09`. The analysis uses
the complete run only: twelve matched sets across three training seeds, not 18
sets or additional independent replication. Partial point files are provenance only.

Recheck: `python -m scripts.archive_clip_september30_evidence --verify-only`.
Original downloads and manifests remain unchanged; no checkpoints are included.
