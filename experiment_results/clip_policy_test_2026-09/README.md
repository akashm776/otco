# Frozen policy test — completed September 30

Source: `clip_policy_test_20260930T050231_622869Z`.
Completed 12 matched sets, 48 branches and 2,400 updates, comparing the frozen
alignment gate with native, sustained and exposure-matched random timing.

Primary gated-minus-random reporting loss is `-0.00003434092384653938`.
Its sign reverses between the two reporting partitions and between continuation
streams. Gate-minus-native loss improves in all six early pairs and worsens in
all six later pairs. No robust policy advantage is established.

`comparison.json` recomputes exactly from retained pair results in protocol order.
All 334 source manifest members were checked; 321 files including the manifest
are retained and 14 caption-bearing files omitted. Archived actions, per-step
traces, endpoint hashes, source snapshots and original manifests are retained.

See `../clip_policy_robustness_2026-09/REPORT.md` for the subsequent CPU-only
audit: one paired reporting-batch deletion reverses the mean timing effect.
The reporting-pool replay in `docs/clip-policy-pool-test.md` is implemented but
has **not** produced results yet.

Recheck: `python -m scripts.archive_clip_september30_evidence --verify-only`.
For Colab replay, use the original full folder/ZIP: this filtered public export
omits caption-bearing identity checks and is not a substitute input archive.
