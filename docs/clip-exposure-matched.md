# Equal-exposure randomized timing control

## Question and limits

Does the frozen alignment policy's update timing improve final canonical retrieval
over randomized timing with the same auxiliary-update count? This follows the
audited `clip_gated_training_20260922T114312_608272Z` pilot, which did not show a
consistent alignment-gating advantage. No threshold, loss, or coefficient is refit.

This is exploratory on reused CUB test data and seeds, not a fresh confirmatory
test. Uniform step randomization changes burst structure as well as placement.
It does not isolate the value of gradient information from every other scheduling
property. The controls are nested within three training seeds: twelve trajectories
are not twelve independent seeds. No significance, novelty, or OT-specific benefit
is claimed. Do not select the best control, checkpoint, or seed after observing results.

## Fixed protocol

- Training seeds: 789, 2026, 31415; each arm starts from pretrained CLIP.
- Per seed: online `alignment_gated`, then `random_matched_1`, `_2`, `_3`.
- Original 100-update native prefix, 1001 total updates, 3850-update LR horizon,
  B64 bf16, data exclusion, trainable parameters, and evaluation stay unchanged.
- Original auxiliary: uniform top-8 feature mix, coefficient
  0.06005351588542947. Alignment threshold 0.030847286945687016, strict greater-than,
  refreshed at completed updates 100, 250, 500, 750 on the reference's own state.
- Randomization seed: 202609230. Hash-rank orders for every seed/control are written
  to Drive before **any** training. Ranks are SHA-256 of
  `otco-dose-v1:{randomization_seed}:{training_seed}:{arm}:{objective_step}`,
  lexicographically sorted by digest (step as tie breaker), over steps 100..1000.
  No training RNG is consumed by randomization.
- After the reference finishes, K is computed **only** from its audited gate
  decisions. Each control activates the first K ranks, without replacement.
  The complete schedules and reference-decision checksum are written and synced
  before controls start. No evaluation metric is an input to this function.
- If K is zero or 901, or any generated schedules duplicate, stop rather than
  reroll selectively. Do not launch a changed experiment without review.
- Prefix hashes must match model, optimizer, scheduler, RNGs, loader-generator
  state, and metadata across arms. Every one of 1001 objective decisions is audited.
- Count and coefficient are matched, not auxiliary loss magnitude, gradients,
  integrated learning-rate-weighted exposure, burst structure, or wall time.
  Only the reference runs observational alignment probes. All arms use the same
  fixed-probe setup, which preserves the training RNG.

Primary contrast: within each seed, final alignment canonical mean-direction R@1
minus the arithmetic mean of **all three** randomized controls; then report the
descriptive mean across three seeds. Report individual controls, all per-seed
contrasts, and species top-1 as a secondary outcome. No best-epoch selection.

## Run

Copy the entire contents of `colabs/clip_exposure_matched_drive_one_cell.py` into
one fresh Google Colab A100 cell. It is self-contained: it checks out base commit
`7e7cfea90b60415dc9561efbe97bcd383cc1580e`, overlays checksummed code (including
`tests/__init__.py`), runs offline tests, then runs the study. No prior weights
or user uploads are needed. The base commit is pinned; dependency versions and
runtime details are recorded, but this is not a bitwise lock of external model/data
services or every transitive dependency. Live GPU execution is not tested locally.

Require 40 GiB runtime disk and 30 GB **additional** free Drive quota. About 17.3 GB
of checkpoint bytes are retained. Old results are never removed. Runtime, caches,
Drive versions/trash, and unrelated data add overhead. Quota failures stop the run.

Drive destination: `MyDrive/OTCO/evaluation_transfer/clip_exposure_matched_<UTC>/`.
There are 24 permanent full-state checkpoints: 12 finals, 3 shared prefixes, and
9 alignment intermediate states; plus one rolling checkpoint. Trainer best/latest
files live in one reused local-only scratch folder. No second full ZIP is generated.

Do not rerun a started study: the runner refuses an existing study manifest, even
with a changed bundle hash. There is no automatic mid-epoch resume. It streams
progress and prints `FINAL_COMPARISON` at the end. The notebook flushes/unmounts
Drive on success and attempts a clearly labeled partial flush on failure. Only
`DRIVE_SYNC_COMPLETE` denotes successful training plus flush. The stored manifest
remains `complete_pending_drive_flush`, because it was written before unmounting.

Useful small audit files: `comparison.json`, `protocol.json`,
`randomization_commitment.json`, `run_manifest.json`, `completion.json`,
`DRIVE_BACKUP_MANIFEST.json`, and each seed's `matched_schedules.json`, prefix
hashes, gate audits, and training summaries.

Build locally: `.venv/bin/python -m scripts.build_clip_exposure_matched_cell`.
