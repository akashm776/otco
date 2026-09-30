# Post-treatment marginal utility — completed September 30

Source: `clip_marginal_utility_20260930T021908_849917Z`.
Completed 12 matched pairs, 24 history branches, 84 anchors and 168 matched trial
pairs: 1,200 continuation updates plus 336 rolled-back trial updates = 1,536.

The primary treated-history-minus-native-history marginal reporting-loss contrast
at offset 25 is `0.000015630572524405473`. Sustained history reduces next-step
auxiliary usefulness in all 12 pairs at offsets 10, 25 and 50. This does not establish
a mediation mechanism, independent generalization, or a retrieval advantage.

The first incomplete run is retained separately in
`../clip_marginal_utility_partial_2026-09`. Its six completed pair results are
byte-identical to the corresponding results here; they are **not additional samples**.

`comparison.json` recomputes exactly from retained pair results and the frozen
protocol. All 239 original manifest members were checked; 226 files including the
manifest are retained and 14 caption-bearing files omitted. No model checkpoints.

Recheck: `python -m scripts.archive_clip_september30_evidence --verify-only`.
The filtered public export does not replace the original full input ZIP.
