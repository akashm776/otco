# Fixed-endpoint reporting-pool test

This exploratory follow-up tests whether the apparent advantage of the frozen gate
over exposure-matched random timing survives a change in the reporting negative
pool. The completed policy run's two B64 reporting partitions give opposite signs;
removing one influential report batch also reverses the pooled sign. No batch is
dropped here, no gate is refitted, and no new independent-test claim is made.

## Frozen contract

- Original seeds 789, 2026, 31415; starting checkpoints 100 and 500; original two
  continuation streams 2026093002 and 2026093003; all four archived arms.
- Replay the recorded Boolean schedules, never re-probe or recompute decisions.
  Exactly 12 matched sets × 4 arms × 50 updates = 2,400 optimizer updates.
- Require archived model/optimizer/scheduler/RNG hashes at offsets 0, 10, 25, 50;
  check action, learning-rate and RNG traces at every update. Any mismatch stops.
- Require exact final encoded feature hashes, original two partition losses and
  full-pool retrieval before accepting additional evaluations.
- Primary: full-512 symmetric CLIP loss, gated minus random-matched at update 50.
  Average streams within starting checkpoints, checkpoints within seeds, then
  the three seeds equally. Negative values favor the gate.
- Secondary: full-pool R@1 and 32 new B64 partitions, seeds 2026093010–2026093041.
  Commit all groupings before replay. Each grouping uses all 512 report images
  once; all branches share it. Retain degenerate equal-schedule matched sets.
- Report all three controls and per-seed/per-starting-checkpoint contrasts.
  Partitions are sensitivity conditions, **not 32 independent replications** or a
  confidence interval. Full-512 and B64 loss have different negative-pool
  estimands; their absolute magnitudes are not interchangeable.
- Reporting images remain the same held-out 512, with canonical captions. The
  other 512 meta images remain reserved and unused. No report controls treatment.

## Colab inputs and execution

Paste all of `colabs/clip_policy_pool_test_drive_one_cell.py` in a fresh A100
Colab session. It clones a pinned base and verifies its embedded source overlay.
Do not paste just the runner. No files need to be committed or pushed first.

Place these in `/content/drive/MyDrive/OTCO/evaluation_transfer`:

1. `clip_checkpoint_diagnostic_inputs.zip` (original checkpoint recovery archive),
   unless the original complete training folder has already been restored.
2. The complete `clip_policy_test_20260930T050231_622869Z` folder, or its single
   downloaded ZIP. `/content` is also accepted for the policy folder/ZIP.

The complete policy archive is authenticated against its pinned backup manifest;
all file hashes and completed results are checked before training. A marginal
utility folder, a `failed runs` folder, or a multipart/partial ZIP is not a substitute.
Existing evidence is never moved, deleted or overwritten. Ambiguous ZIP inputs
and previous runs of this new experiment stop execution for inspection.

Use the archived A100 and matching recorded package versions. Exact state hashes
are the final numerical reproducibility check; a different stack or inexact replay
must not silently become a new continuation experiment. No automatic retry or resume.
Allow approximately 25 GiB recovery space, 15 GiB free afterward, and about 150 MiB
new Drive output. No model checkpoints are created.

Wait for `DRIVE_SYNC_COMPLETE`, then download the new `clip_policy_pool_test_*`
folder plus its log from `clip_policy_pool_test_logs`. A partial sync is not success.

## Reusable endpoint cache

Each of the 48 branches saves compressed numerical NPZ containing FP32 normalized
image/text embeddings and the FP64 scalar learned logit scale. Sidecars pin file
and feature hashes, dimensions and shared row order. No model weights, optimizer
weights, raw images or captions are stored in these caches. Row identity is an
ordered list of SHA256 image-key identifiers, with canonical first captions.

`src.clip_policy_pool_test.load_cache(path, metadata)` verifies and loads an NPZ
without pickle. `evaluate_encoded(encoded, conditions)` evaluates full-pool loss,
retrieval and supplied partitions on CPU without a model or replay. The runner
checks exact cache round-trip equivalence and state-preserving observation.

## Interpretation

A negative full-pool effect with consistently favorable partition directions would
support reporting-pool robustness of these fixed endpoints, not general policy
efficacy. A sign reversal or mixed partitions would keep the timing advantage
unresolved. Either outcome is informative; neither establishes a transport-specific
mechanism or independent generalization. Do not choose seeds, partitions or endpoints
after seeing the new results.
