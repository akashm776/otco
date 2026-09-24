# OTCO background study and practice plan

**September 24 update:** include the [latest evidence and open questions](research-status-2026-09-24.md)
in the history, inference, and defense sessions below.

Use with the [defense workbook](research-defense.md). This sequence assumes some ML familiarity; extend a session whenever its exit exercise is difficult. Plan for twelve sessions of roughly 60–90 minutes, with longer paper-reading blocks as needed. Completion depends on what you can explain and reproduce, not the calendar. The initial reading list is targeted background, not an exhaustive or current novelty survey.

## Start with a 20-minute diagnostic

Without notes or AI, answer:

1. What are the inputs, outputs, and trainable parameters of your current experiment?
2. Write the native and auxiliary losses and identify every stop-gradient.
3. What does OT add beyond row-wise softmax weighting?
4. What result would convince you the method helps, and has that result occurred?
5. What was the strongest evidence behind the last research decision you made?

Keep the original answers. Mark each **know / partly know / cannot yet explain**. This determines whether to spend more time on foundations, code, or inference.

## The session routine

Use approximately 10 minutes for recall, 25 for primary reading, 25 for a derivation/code trace, and 15 for an oral explanation and error log. Read the actual paper's method and assumptions; abstracts and AI summaries are only orientation. Attempt exercises before asking AI for an explanation. Revisit difficult items after two days and a week.

For each paper, make a one-page card: its question; objective; assumptions; negative construction; gradient path; experiments; limitations; and the exact relationship to OTCO. Include a page, equation or section reference after reading the PDF. Leave these locators blank until checked rather than inventing them.

## Twelve sessions

| Session | Background and local reading | Work you produce | Exit criterion |
|---|---|---|---|
| 1. Reconstruct the project | README; workbook claim boundaries; `docs/legacy-experiments.md` | A one-page history: proposal → legacy results → CLIP controls → gradients → actual updates → frozen rules | Give a two-minute explanation that distinguishes hypotheses from results |
| 2. Contrastive foundations | [CLIP](https://proceedings.mlr.press/v139/radford21a.html); `native_clip_contrastive_loss` | Derive row cross-entropy, its logit gradient, and the symmetric objective using a 3×3 example | Explain how the denominator already emphasizes hard negatives and why batch size matters |
| 3. Legacy versus current objective | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343); `model/loss.py` versus `model/clip_training.py` | A side-by-side comparison of pair labels, reductions, bias/scale, gradient paths, and encoders | Explain why an architecture change plus a loss change is not an isolated architecture comparison |
| 4. Optimal transport fundamentals | [Computational Optimal Transport](https://arxiv.org/abs/1803.00567), discrete transport and regularization material | Write primal constraints and a 3×3 feasible plan; contrast row softmax and balanced coupling | Explain what the column constraints buy, and what they do not say about semantics |
| 5. Sinkhorn and sparse feasibility | [Sinkhorn Distances](https://arxiv.org/abs/1306.0895); `build_audited_transport_plan` | Hand-run or implement a tiny Sinkhorn example; inspect row/column residuals before and after masking | Explain infeasibility, finite-iteration error, numerical floors, epsilon scaling, and why post-masking can lose mass |
| 6. Synthetic geometry and precedent | [Hard Negative Mixing](https://arxiv.org/abs/2010.01028); `synthetic_barycentric_weights` and normalization | Reproduce the (0.8, ±0.6) example; derive the normalization Jacobian; compare prior mixing construction to yours | Separate convex averaging, normalized direction, cosine hardness, and semantic validity |
| 7. Which negatives should be hard? | [Hard Negative Samples](https://arxiv.org/abs/2010.04592); [Debiased Contrastive Learning](https://arxiv.org/abs/2007.00224) | Explain two possible false-negative scenarios using bird-caption examples, clearly marked hypothetical | Explain why an image-ID mismatch need not be a semantic contradiction; propose a test rather than assume false negatives caused results |
| 8. The actual auxiliary | Workbook derivations; `clip_relative_denominator_loss`; `FreshBatchRawCosineOTCO.forward`; `src/clip_early_pulse.py` | Derive the denominator increment and both sets of logit partial derivatives; draw detach/live paths | Explain detached scale, alpha=0.5 qualification, and hardest-real duplication without notes |
| 9. From gradients to optimizer updates | [AdamW](https://arxiv.org/abs/1711.05101); [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224); `one_update` | Derive the first-order gradient-alignment argument; list the assumptions AdamW and held-out evaluation break | Distinguish per-query embedding, batch embedding, shared-head, full-parameter, and actual-update diagnostics |
| 10. Data and experimental design | [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/); [Reed et al.](https://openaccess.thecvf.com/content_cvpr_2016/html/Reed_Learning_Deep_Representations_CVPR_2016_paper.html); paired and pulse protocols | Draw data exclusions and the unit hierarchy; work one R@1 change from counts | Distinguish canonical/all-caption/species evaluation, paired control, reused holdout, and seed replication |
| 11. Selection and reproducible evidence | [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html); frozen rules and raw branch records | Predict the scores before running the companion verifier; independently calculate one seed's confusion matrices and one paired effect | Explain 720 records versus 15 states versus 3 new seeds, balanced versus ordinary accuracy, and the ±1e-6 sensitivity band |
| 12. Defense and collaborator preparation | Your completed decision ledger and paper cards | A 10-minute presentation, 20-minute questioning session, and a one-page list of unresolved claims | Answer with evidence and limits; identify what you would revise without improvising a historical rationale |

These reading choices have specific purposes. CLIP and SigLIP explain the two objective families used in this repository. Cuturi and Peyré/Cuturi supply the OT foundations. Hard Negative Mixing is essential construction-related prior work; Robinson and Chuang address negative selection and false negatives. Du et al. are essential prior work for an alignment-based auxiliary argument. AdamW explains why the real optimizer deserves direct measurement. The CUB and Reed sources establish dataset/caption context; verify your actual implementation's splits separately rather than copying a paper's evaluation convention.

## Background bridge from your previous OT work

The local thesis title and opening abstract concern warm-start algorithms for matching and optimal transport. If that material is familiar, use it to recall couplings, marginal feasibility, approximate optimization, and the difference between a mathematical problem and its solver. Do not assume this transfers automatically to neural representation learning. In this project the harder bridge is from geometric construction to gradient paths, adaptive optimization, semantic relevance, and credible experimental inference. This preparation did not audit the thesis proofs or import their guarantees into OTCO.

## Three small labs, with no GPU training required

### Lab A: hardness without learning

Compute the normalized mean of (0.8,0.6) and (0.8,-0.6). Vary one weight from zero to one. Before calculating, predict where query cosine is largest. Then explain why this experiment cannot show retrieval improvement or validate OT weights. Optional extension: examine nearly cancelling vectors and how the normalization Jacobian changes.

### Lab B: a transport plan that loses its guarantee

Choose a small cost matrix and a support excluding one column. Specify uniform marginals. First prove infeasibility. Then add tiny off-support kernel entries, scale, and mask the resulting plan. Measure its mass and marginal residuals. Explain which step makes the output incompatible with the written constrained problem. This is a toy illustration of a concern, not a measurement of a specific archived run.

### Lab C: reconstruct the headline

Read the companion [verifier](../scripts/verify_research_defense_evidence.py) before running it. Choose one seed/state and manually compute a branch's three-shuffle held-out mean, subtract the native paired branch, and average the 16 synthetic differences. Use the frozen feature threshold to predict the state label without looking at the outcome. Rebuild the seed's confusion matrix. Only then compare with the archived report and script output.

The verifier runs with `python3 scripts/verify_research_defense_evidence.py` from the repository root. It uses standard-library arithmetic and files, not the model, a GPU, or the production scoring function. Passing it supports internal consistency; the person defending the work should still be able to explain each aggregation.

## Reading priority if time is short

Before the first meeting, prioritize sessions 1, 2, 5, 8, 9, and 11 and read the abstracts/method overviews of the two most direct precedents: Hard Negative Mixing and Adapting Auxiliary Losses Using Gradient Similarity. Label the literature comparison incomplete until you have read their methods and experiments carefully. Do not claim readiness from having merely skimmed the papers.

## Your first practice task

Write 150–250 words, without AI or notes, answering:

> What does OTCO change in the native CLIP objective, why might that help, and what is the strongest result that makes you doubt the original idea?

Then write the two loss equations. Keep the first version. A useful tutoring response should diagnose conceptual gaps in your answer, ask you to repair one, and make you explain it again. A polished replacement paragraph would not establish your understanding.
