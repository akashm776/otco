# OTCO research defense workbook

**September 24 update:** read the [current research record](research-status-2026-09-24.md)
first. This workbook preserves the September 12 preparation baseline; evaluation
transfer, gated rollouts, exposure controls and the checkpoint diagnostic have
since completed. Do not describe those studies as unrun.

Prepared 2026-09-12 from the local proposal, implementation, study protocols, and archived results. Start here, then use the [study plan](research-study-plan.md). This is a first preparation audit, not certification that every historical experiment or theoretical claim has been verified.

## 1. What you are preparing to do

The objective is to own the reasoning: explain the question, derive the method, identify what each experiment isolates, reproduce key evidence, and say when a decision should be revised. You do not need to prove that every past choice was optimal. A defensible answer can be: “This was a bounded exploratory choice; this control tested its consequence; this uncertainty remains.”

Do not memorize this AI-written workbook as your defense. Write your own answers first, then use it to check them. A rationale reconstructed today is not evidence of what motivated a decision originally.

### A defensible opening, to rewrite in your own words

> I began by asking whether optimal transport could combine confusing image embeddings into useful synthetic negatives for image-text contrastive learning. The experiments led to a narrower question: when does synthetic-negative pressure help optimization? In the current CLIP/CUB setting, greater cosine hardness does not establish better learning, and the eight original training arms show no final canonical retrieval win over baseline. Paired optimizer experiments show small one-step benefits that vary with training state. A frozen alignment rule has a modest new-seed balanced-accuracy advantage over timing, while ordinary accuracy ties. This has not established an OT-weighting advantage or a successful training curriculum.

Evidence: [original CLIP experiments](clip-experiments.md), [paired protocol](clip-paired-updates.md), [prospective protocol/results](clip-usefulness-prospective.md).

## 2. Claim boundaries you must know without notes

| Claim | What the evidence permits | What it does not permit |
|---|---|---|
| Synthetic mixtures can be harder | Higher query cosine in specified frozen/batch settings; uniform mixing often achieves this | Semantic validity, helpful gradients, or better retrieval |
| OT weights improve mixing | OT and uniform are compared on matching support; no demonstrated general OT advantage | “Optimal transport is necessary” or “globally optimal negatives” |
| Longer training helps | Some secondary metrics favor treatments slightly | A consistent gain: all seven auxiliary arms lose on final canonical average R@1 in the eight-arm single-seed study |
| Early synthetic pressure can help one step | Paired early effects recur across training seeds under the fixed probes | A universal early-only rule, population significance, or cumulative training gain |
| Alignment transfers | New seeds: 90.28% versus 83.33% mean seed balanced accuracy | Decisive superiority: both rules are 13/15 correct and 12/13 outside the near-zero band |
| Evidence is reproducible | Archived arithmetic, hashes, reset/replay records support consistency | Independent re-execution of unavailable checkpoints or robustness across hardware |

The [intermediate table](clip-paired-intermediate.md) is a counterexample to a universal cutoff: at update 500 the synthetic mean effect is +22.68 × 10^-6 in seed 42 and -6.53 × 10^-6 in seed 456. Later effects can return to beneficial. Training step, representation, optimizer moments, learning rates, and relative pressure co-vary.

The [pulse pilot](clip-early-pulse.md) produces +0.017259 percentage points canonical average R@1 and -0.258887 points species accuracy. That is a net two retrieval successes across directions and 15 fewer species successes. Do not describe this as a meaningful established performance improvement.

## 3. Resolve these proposal-to-evidence discrepancies

The file [OTCO-5.pdf](../OTCO-5.pdf) is actually LaTeX text despite its extension. It records an earlier proposal, not the current implemented method. Existing source files have not been rewritten by this preparation task.

| Earlier statement or implication | Current assessment | Defensible treatment |
|---|---|---|
| Aggregation gives consistently stronger supervision than selection | Not established by current downstream results | Present as an initial hypothesis; report contrary/null evidence |
| Linear heads establish semantic validity of mixtures | Linearity permits a construction; it does not prove it corresponds to a valid semantic negative | Separate algebraic possibility from semantic interpretation |
| The transport plan and resulting synthetic negatives are detached | Current CLIP code detaches selection/weights, but keeps selected image-feature paths live through mixing and normalization | Draw the actual differentiation graph; specify which version you mean |
| Output is a balanced transport coupling | Historical solver floors unsupported entries, scales, then masks them; removing mass can violate marginals | Describe the implemented approximation and report feasibility/mass diagnostics |
| Early representations are unorganized, so auxiliary warmup is necessary | Current CLIP starts pretrained; useful one-step effects already occur at update 100 | Warmup is a hypothesis/design choice whose relevance depends on initialization |
| Negatives are treated equally by the baseline | In softmax CLIP, higher-scoring negatives already receive larger logit gradients | Explain the exact loss, not a generic slogan |
| Cached plans across batches are interchangeable with fresh plans | Batch identities and geometry can change; current CLIP computes fresh plans | Discuss cached historical experiments separately |
| The same method description covers all experiments | Legacy ResNet-50/DistilBERT with a sigmoid-style objective differs from native CLIP | Maintain separate architecture, objective, schedule, and data descriptions |

Code anchors: [CLIP losses and gradient paths](../model/clip_training.py), [historical solver](../src/clip_geometry_v2_metrics.py), [legacy record](legacy-experiments.md). Some older documents retain proposed next steps that have since completed; use the latest study and archived run rather than those future-tense passages.

## 4. Derive the current method

### A. Native contrastive objective

For a batch of B unique image-caption pairs, let unit embeddings be t_i and v_j, with logits s_ij = q t_i^T v_j and learned positive scale q. The text-to-image loss is

```text
L_T = (1/B) sum_i [-s_ii + log sum_j exp(s_ij)]
L_I = the same expression with logits transposed
L_CLIP = (L_T + L_I)/2
```

For one row, dL_i/ds_ij = softmax(s_i)_j - 1[j=i]. This already emphasizes high-score negatives. Explain why the transpose matters and why one random caption per image avoids treating repeated copies of an image as separate diagonal targets within a batch. CLIP background: [Radford et al.](https://proceedings.mlr.press/v139/radford21a.html).

### B. OT and its implementation qualification

The idealized entropic problem is

```text
min_P <P,C> + epsilon sum_ij P_ij(log P_ij - 1)
subject to P >= 0, P 1 = a, P^T 1 = b, and permitted support
```

For positive kernels on feasible support, Sinkhorn uses K = exp(-C/epsilon), P = diag(u) K diag(v), alternating u = a/(Kv) and v = b/(K^T u). A row softmax does not impose column demands; the column constraints are what create cross-query competition. Study the formulation in [Computational Optimal Transport](https://arxiv.org/abs/1803.00567) and the entropic algorithm in [Cuturi](https://arxiv.org/abs/1306.0895).

Local implementation check: `build_audited_transport_plan` uses a maximum-minus-score cost, top-k support, numerical floors and finite iterations. For an exact fixed-total-mass problem, a common cost offset changes the objective by a constant; do not assume numerical floors preserve exact invariances. The frozen V2 study reports a full-pool support with 107 unreachable columns, so uniform column marginals are infeasible. Nonempty columns alone are not sufficient to prove every sparse support feasible.

**Exercise:** construct a 3×3 support with a column containing no allowed edges and prove that a positive target marginal on that column is impossible. Explain why adding iterations cannot repair infeasibility.

### C. Mixing and normalization

For row-normalized weights w_ij, the construction is

```text
b_i = sum_j w_ij v_j
v_extra_i = b_i / ||b_i||
```

The unnormalized b_i is a convex combination; its normalized direction need not be in the original convex hull. A concrete example, derived here: query t=(1,0), negatives v_1=(0.8,0.6), v_2=(0.8,-0.6). Each has query cosine 0.8. Their uniform mean is (0.8,0); normalization gives (1,0), with cosine 1.0. OT is unnecessary for this example. It proves a geometric possibility, not that the mixture is a semantically valid negative.

For nonzero b, the normalization Jacobian is `(I - vv^T)/||b||`, where v=b/||b||. Thus normalization changes gradients; a very small mixture norm also deserves numerical inspection. In the current path, weights/support are fixed for differentiation, but contributor embeddings receive gradients. “Stop-gradient OT” does not mean the full synthetic is fixed.

### D. Relative-denominator auxiliary

Let Z_i = sum_j exp(s_ij), including the positive, and e_i be the extra logit. Then

```text
D_i = log(Z_i + exp(e_i)) - log Z_i
    = softplus(e_i - log Z_i)
L_total = L_CLIP + alpha_effective * mean_i D_i
```

This expression is invariant to a common additive shift of ordinary and extra logits. At alpha=0.5, its value gives the text-to-image half of symmetric CLIP one extra candidate, with image-to-text unchanged. Qualification: the auxiliary uses detached q, so the logit-scale gradient is not identical to that of a fully scale-trainable augmented CLIP objective.

For one D_i, treating logit coordinates independently, `dD/de = sigmoid(e-log Z)` and `dD/ds_j = -sigmoid(e-log Z)*softmax(s)_j`. The ordinary denominator remains differentiable. Therefore this auxiliary is not simply “repel the synthetic.” Trace the shared embedding paths after deriving these partial derivatives.

Hardest-real uses an image already in the denominator. Its extra contribution duplicates/reweights that real candidate; it does not add a novel image. Equal alpha does not ensure equal gradient norms, directions, or optimizer updates. Verify with [the implemented loss](../model/clip_training.py).

### E. Gradient direction versus actual usefulness

For small plain-SGD auxiliary descent, a first-order expansion gives

```text
L_native(theta - eta*g_aux) - L_native(theta)
    approximately -eta * g_native^T g_aux
```

This motivates alignment; it is not a guarantee for held-out loss, finite steps, or AdamW. Shared-parameter gradients are J^T times embedding gradients, so their cosines need not match embedding-space cosines. Per-query averages also need not match the direction of an aggregated batch objective.

AdamW uses momentum, per-coordinate second moments, parameter-group learning rates and decoupled decay. Adding an auxiliary changes the combined gradient, clipping, and new optimizer moments. Raw gradient cosine therefore does not determine the realized update. Background: [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101). Gradient-based auxiliary weighting also predates this project: [Du et al.](https://arxiv.org/abs/1812.02224).

The paired estimand is `Delta = L_holdout(theta_after_auxiliary_branch) - L_holdout(theta_after_native_branch)`. Negative means relative benefit. It does not imply either branch improved over the starting checkpoint. Resetting the complete optimizer state before each branch is essential. See [one_update and restore_branch](../src/clip_paired_updates.py).

## 5. Decision ledger: reason, control, remaining uncertainty

Labels: **documented** means supported by the linked protocol; **reconstructed** is a plausible justification to verify against your recollection/history; **open** means optimality or mechanism has not been established. These are not claims about whether a human or AI originally proposed the choice.

| Decision | Rationale/status | Alternative or challenge; evidence needed |
|---|---|---|
| CUB with fine-grained captions | Historical rationale: nearby bird descriptions provide confusable mismatches | Does instance retrieval penalize semantically acceptable matches? Compare a separate dataset and inspect caption ambiguity |
| Move to pretrained CLIP | Documented question: test an already aligned representation with its native loss | Different initialization and objective prevent a controlled comparison with legacy encoders |
| Tune projections, final blocks, scale | Fixed protocol; limited adaptation is a reconstructed practical rationale | Full fine-tuning/frozen encoders were not ruled out as superior |
| B64 and 50 epochs | Recorded settings; resource/comparability rationale needs original confirmation | Batch size changes candidate pools and loss; budget optimality is open |
| LR groups 1e-5/1e-6/1e-6; weight decay .01; clip 1 | Recorded controlled defaults, not established optima | Why these exact values? Recover provenance; do not invent a tuning study |
| Top-k and masked diagonal | Exclude paired positives; restrict local candidates | Locality is not semantic correctness; false negatives remain possible |
| Epsilon .049, 30 iterations | Recorded CLIP settings; scale controls explicitly studied | Require residual/mass evidence for solver claims; neither universal epsilon nor convergence follows from the iteration count |
| Uniform top-32 and top-8 | Isolate weighting and support width; documented controls | Top-8 was informed by exploratory geometry; hardness alone cannot validate selection |
| Fresh plans every update | Current batch identities/geometry are used | Historical cached-pool and stale-plan results require separate interpretation |
| Detach transport weights | Implemented choice; avoiding differentiated selection is a reconstructed rationale | Stability superiority over differentiable OT has not been demonstrated |
| Detached auxiliary scale | Keep direct scale training in the native objective | This is an explicit gradient design choice, not automatic equivalence of objectives |
| Relative instead of absolute auxiliary | Documented compatibility/denominator control | Asymmetric extra-image treatment remains; reverse/bidirectional synthesis is untested here |
| Warmup 1000 and ramp 1000 | Original protocol | Exact timing was not proved optimal; do not confuse auxiliary warmup with 100-step LR warmup |
| Gap gates retained, entropy gate disabled | Logged original CLIP conditions | No adaptive gate advantage: observed gate states are inactive-schedule or fully-active |
| Original pressure-match alpha .134 | Match early projection-gradient ratio to hardest-real | Local match only; differs from later full-gradient calibration |
| Pulse coefficients .0600535 / .3753712 | Four training B64 batches; 10% ratio-of-mean full-gradient target | Target is a pilot choice, not an optimum; realized pressure drifts |
| Pulse objective steps 100–199 | Readiness probes motivated an early pilot | No matched later pulse; timing superiority not isolated |
| Stop short runs at 1001 but retain 3850 LR horizon | Preserve baseline prefix trajectory | These are truncated long-schedule runs, not optimized 13-epoch runs |
| Holdout 1024, 16 diagnostic training batches, fixed captions | Exclusion and fixed inputs support paired comparisons | Sampling coverage and power not established; fixed probes constrain generalization |
| Three shuffled partitions; sequential supporting | Probe batch-composition sensitivity | Reuses the same images; not independent datasets or seeds |
| Paired model/optimizer/RNG reset | Isolate adding the auxiliary at the same complete state | Does not isolate representation from learning rate/momentum across states |
| Five measured checkpoints | Expand endpoint evidence to intermediate behavior | Cannot infer an exact continuous crossing or universal step threshold |
| Leave-one-seed-out threshold screening | Development comparison across three trajectories | Candidate search and reused holdout remain; do not present as untouched validation |
| Freeze rule; three new seeds | Prospective relative to training trajectories | No new evaluation dataset; no online gate intervention |
| Balanced accuracy primary; ±1e-6 secondary band | Declared prospective scoring choices | Equal class importance needs justification; the band is not a measured noise floor or practical-benefit threshold |
| Canonical final R@1 plus species/all-caption | Fixed comparable endpoint and complementary behavior | First-caption and image-identity conventions limit meaning; selecting best of 51 epochs is a different analysis |

Protocol sources: [CLIP](clip-experiments.md), [pulse](clip-early-pulse.md), [paired](clip-paired-updates.md), [intermediate](clip-paired-intermediate.md), [screening](clip-usefulness-predictors.md), [prospective](clip-usefulness-prospective.md). For a historical run, its archived resolved configuration outranks a subsequently modified source config.

For every unresolved row, add: original dated evidence; your explanation; strongest alternative; what observation would change your mind; status **retain / revise / unverified**. Exact defaults can be defensible controls without being scientifically optimal.

## 6. Explain the statistics and evaluation

There are six training seeds across development and prospective work, three of which are new-seed validation trajectories. The prospective study has `3 seeds × 5 states × 16 training batches × 3 branches = 720 records`. It has 15 correlated seed/checkpoint states. It does not have 720 independent training replicates. Endpoint replay and repeated partitions do not increase the number of independent trajectories.

For each seed, balanced accuracy is `(helpful-state recall + harmful-state recall)/2`; the primary metric averages seeds. Alignment catches 5/5 helpful states and incorrectly calls two harmful states helpful. Timing catches 3/5 helpful states and has no false positives. Both have 13/15 ordinary accuracy; their mean per-seed balanced accuracies differ. Keep the mean-of-seed definition: pooled balanced accuracy need not equal it. [Archived predictions](../experiment_results/clip_usefulness_prospective_2026-09/results/prediction_report.json).

The threshold `0.030847286945687016` is a fitted development boundary with reproducible serialization, not a physical constant. Timing `<=175` separates measured updates 100 and 250; update 175 was not measured. The ±1e-6 exclusion is a fixed sensitivity analysis. Neither the threshold precision nor fp64 loss arithmetic proves weak effects are robust across hardware or datasets.

Canonical retrieval uses one first caption per image; all-caption retrieval uses ten and a different relevance protocol. Species recognition uses text prompts over species. These endpoints answer different questions. For canonical N=5794, a single additional success in one direction changes average R@1 by `100/(2N) ≈ 0.00863` percentage points. B64 contrastive holdout loss and full-pool retrieval cannot be compared as though they share candidate sets.

Repeatedly inspected holdout/test outcomes can influence research choices even without backpropagation. Freezing a rule prevents new-seed refitting; it does not make the reused examples fresh. Read [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) for the selection-bias issue. No new inferential significance or confidence interval is claimed here.

## 7. The collaborator's difficult questions

Answer each aloud before looking up the evidence. Your answer should contain a claim, an artifact, and its limit.

1. What exactly is the research question now, and how did it change from the proposal?
2. Why is OT needed when uniform mixing is often equally hard or harder?
3. Is your returned plan actually feasible for the constraints written in the paper?
4. Derive a synthetic cosine larger than every constituent's without OT.
5. How do you know a nearby mismatched bird image is a valid semantic negative?
6. Which tensors are detached, and where do image/text/scale gradients flow?
7. Derive the auxiliary. Why is alpha=0.5 meaningful, and where does that equivalence stop?
8. Why does the auxiliary differentiate through real candidates outside the selected top-8?
9. What does duplicating the hardest real candidate control for, and what does it fail to match?
10. Why can favorable embedding geometry disagree with shared-head gradients and AdamW?
11. How do you know branches started with identical optimizer state and data?
12. What is your independent experimental unit? Why is 16/16 not 16 independent seeds?
13. Why do both rules get 13/15 while balanced accuracy differs?
14. Are the 10^-6 effects meaningful, numerically stable, or merely detectable within one execution path?
15. Why does early one-step benefit fail to deliver a convincing pulse gain?
16. What evidence rules out a universal switch-off point?
17. Is the holdout untouched? Exactly what is prospective about the final study?
18. What does this add beyond hard-negative mixing and gradient-based auxiliary weighting?
19. Which choices came from prior literature, measured evidence, convenience, or AI suggestions?
20. Which claim would you remove first if the collaborator challenged the evidence?

For novelty, compare against [Hard Negative Mixing](https://arxiv.org/abs/2010.01028), [Contrastive Learning with Hard Negative Samples](https://arxiv.org/abs/2010.04592), [Debiased Contrastive Learning](https://arxiv.org/abs/2007.00224), and [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224). These establish relevant precedents, not a completed novelty review. A possible contribution is the particular empirical separation of hardness, gradient spaces, actual updates, and transfer of a frozen diagnostic in this setting; establish its distinctness through full-paper comparison before claiming novelty.

## 8. Recover ownership after substantial AI assistance

For one decision per session, close AI and the notes. Write the problem, two alternatives, your prediction before inspecting results, and the observation that would falsify your preferred explanation. Then derive the relevant equation or recompute the measurement. Finally compare your account with source and history. Mark the gap you found; do not silently replace your earlier answer with a polished generated one.

Use this record repeatedly:

```text
Decision / date / run or commit:
Original rationale and evidence (or UNKNOWN):
Origin: my idea / literature / collaborator / AI suggestion / unknown:
My present explanation:
Alternative and its tradeoff:
What was actually tested:
What was not tested:
Falsifying observation:
Independent check I completed:
Current disposition: retain / revise / unverified:
```

Be factual about AI use: identify assistance with hypotheses, implementation, analysis or prose where you can substantiate it. Do not claim independent checks you have not done. “I used AI extensively and am auditing the reasoning and evidence” is more accurate than either treating AI as authority or assuming all assisted work is invalid.

## 9. Readiness and first meeting

Use a practical self-check: 0 = cannot explain; 1 = can repeat notes; 2 = can derive/explain and locate evidence; 3 = can also identify an alternative and limitation. Score the 20 questions. This is a learning rubric, not a validated assessment. Before presenting a claim confidently, reach at least 2 on its questions; any zero on objective, gradients, OT feasibility, or experimental units needs attention first.

A useful first collaborator discussion is 45 minutes: 5 on the question; 10 on the implemented objective and OT qualification; 10 on the evidence ladder; 10 on their strongest objections; 10 to agree which uncertainty deserves work. Bring this ledger, the relevant raw result, and three unresolved questions. You do not need every open scientific question answered before collaborating; you need to distinguish what you understand from what you want help resolving.

Candidate questions for that discussion, not experiments launched here: whether the narrow mechanism study is a worthwhile contribution; what independent evaluation would test the frozen rule; whether a feasible OT-vs-uniform ablation is worth its cost. A gate-driven rollout changes the trajectory and can invalidate a predictor learned only on native trajectories, so it needs its own evaluation.

## 10. Verification performed for this preparation

The companion [standard-library verifier](../scripts/verify_research_defense_evidence.py) reads committed numeric artifacts without importing training code. It checks prospective file hashes against the archived inventory, reconstructs paired loss differences from per-partition records, rebuilds the 15 state effects and frozen-rule scores, and checks the eight original final canonical averages. Run from the repository root:

```bash
python3 scripts/verify_research_defense_evidence.py
```

This verifies arithmetic and internal archive consistency. It cannot authenticate unavailable GPU states, prove preregistration timing independently, evaluate missing datasets, or validate the scientific interpretation by itself. Historical protocol-before-outcome statements above follow the stored provenance, not a fresh Git-history forensic audit.

The attempted local command `.venv/bin/python -m pytest -q tests/test_clip_usefulness_archived_results.py tests/test_clip_training.py tests/test_clip_geometry_v2.py` exited 139 before producing a test report. It is not a passing test run; the environment failure was not diagnosed in this documentation task. This first audit inspects key CLIP code and selected legacy/proposal passages; it does not certify every legacy implementation or result.
