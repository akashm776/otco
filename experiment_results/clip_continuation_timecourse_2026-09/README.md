# Continuation time course — completed September 29

Source: `clip_continuation_timecourse_20260929T135631_790370Z`.
Completed 24 matched pairs, 48 branches and 2,400 optimizer updates. Reporting
and diagnostic-only re-probes at offsets 0, 1, 5, 10, 25 and 50.

At update 50, sustained-minus-native reporting loss improves in all six early
pairs and worsens in all 18 later pairs. Reporting-pool R@1 worsens in 22 pairs
and ties in two. The seed-averaged later-minus-early loss contrast is
`0.002752731624257043`. This is an early/later contrast on reused checkpoints and
report data, not evidence of a universal switching rule or a causal mechanism.

`comparison.json` recomputes exactly from the retained `seed_*/step_*/stream_*/result.json`
files using the frozen protocol. The export verifies all 615 original manifest
members, retains 590 files including the manifest, and excludes 26 caption-bearing
files. Original source snapshots and manifests are unchanged. No model checkpoints.

Recheck: `python -m scripts.archive_clip_september30_evidence --verify-only`.
See `export_audit.json` for exact file hashes. Original full ZIPs remain outside Git;
the filtered public export does not replace a complete Colab input archive.
