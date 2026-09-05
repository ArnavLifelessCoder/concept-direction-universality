# Reviewer Response Plan — Resubmission

Scores on the previous submission: **2 / 4 / 4** (Reviewer 1 / YaL8 / YAbf).

Reviewer 1 is the one to design the revision around. Their objections are
methodological rather than presentational, and one of them invalidates the
paper's headline number as currently computed. Reviewers 2 and 3 both
recommended acceptance and ask mainly for breadth and citations.

Status key: **Done** (code merged, awaiting the re-run) · **Queued** (needs the
Kaggle session) · **Declined** (out of scope, with a stated reason).

---

## The one that has to be fixed first

### R1-W5 — The global direction contains the domain compared against it

> *"...four equal-magnitude mutually orthogonal directions would each have
> cosine with their normalized sum of 1/sqrt(4) = 0.5, which is close to the
> reported minimum mean honesty cosine of 0.522."*

This is correct and it is the most serious point in the reviews. Our reported
honesty minimum of **0.522 is statistically indistinguishable from complete
fragmentation**, because a domain contributes to the very direction it is
scored against. The number as published does not support the claim attached
to it, in either direction.

**Done.** `leave_one_domain_out_direction` and `lodo_cosines` in
`src/analysis/geometry_controls.py` build the global direction from the other
three domains only. `tests/test_geometry_controls.py` pins the artifact
numerically: orthogonal domains score `1/sqrt(k)` against a pooled global and
`~0` against a LODO global, for k = 2..5.

**Queued:** re-run and report LODO cosines as the primary statistic, with
pooled-global values retained only for comparability with prior work.

> **The honest possibility.** LODO can only lower these cosines. If honesty's
> early-layer LODO values land near zero, the finding strengthens into genuine
> fragmentation. If refusal's LODO values fall well below the reported
> 0.83–0.94, the universality claim weakens too. Both headline results are
> genuinely at stake, and the re-run decides them.

---

## Reviewer 1 — remaining points

### R1-W2 — No within-domain reference for cross-domain cosine

> *"A natural control would be to repeatedly split each domain into matched
> subsets, independently estimate directions, and compare the resulting
> within-domain cosine distribution against the cross-domain one."*

Exactly right, and the reviewer describes the correct protocol.

**Done.** `split_half_reliability` (200 disjoint splits per domain) gives the
within-domain noise ceiling; `matched_cross_domain_cosine` estimates
cross-domain cosines at the *same per-estimate sample size*, which matters
because cosine error grows sharply as pairs fall relative to `d_model`.
`fragmentation_index` reports the gap (primary) and a disattenuation-style
ratio (secondary, explicitly labeled heuristic).

**This is where the revision is most exposed.** `data/prompt_pairs/honesty/`
is badly unbalanced:

| domain | pairs |
|---|---|
| factual_trivia | 526 |
| politics_opinion | 179 |
| personal_advice | 121 |
| **math** | **59** |

Math carries the paper's headline claim ("near-orthogonal to all others"), and
a split-half estimate for math uses **~29 pairs to estimate a direction in
2048 dimensions**. The within-domain reliability for math may well be so low
that its apparent near-orthogonality is mostly estimation noise — which is
precisely R1's suspicion, and worse than they could have known from the paper.
Refusal has the same problem at `privacy` (29 pairs).

**Queued:** run the control and report the reliability. If math's within-domain
reliability is low, the correct response is to say so and either expand the
domain (`scripts/add_math_honesty_pairs.py`) or drop the near-orthogonality
claim. Do not report a cross-domain cosine for any domain whose within-domain
cosine is not meaningfully above zero.

### R1-W3 — Early-layer directions may not be meaningful

> *"It would be beneficial to report the unnormalized effect size, split-half
> direction reliability, permutation baselines, or other evidence that the
> extracted vector carries a stable concept signal."*

**Done.** `permutation_null` implements a sign-flip null (flip each pair's
polarity with p=0.5, recompute `||Δμ||`), which preserves pairing and
activation marginals while destroying the systematic contrast.
`direction_effect_size` reports `delta_norm`, `cohens_d`, and a
**cross-validated** projection AUC.

The cross-validation is not a detail. The first implementation scored AUC
in-sample and returned **0.86 on pure noise** at n=120, d=128, because a
difference-of-means direction fits noise whenever `d_model >= n`. Publishing
that would have manufactured exactly the artifact this reviewer is warning
about. `auc_insample` is retained only as an overfitting diagnostic and is
never reported as evidence.

**Queued:** report per-layer permutation p-values and CV AUC; mark layers that
fail as uninterpretable rather than fragmented.

### R1-W1 — The honesty contrast may not isolate honesty

> *"...the extracted directions may simply correspond to different underlying
> concepts or computations."*

Substantially correct, and not fully fixable with the current data. Our
contrast is truthful-vs-false *answers*, so in math it may track arithmetic
correctness, in trivia knowledge retrieval, and in politics/opinion hedging or
stance. Low cosine between such directions does not establish that one concept
is encoded differently.

**Done (framing).** A new paragraph in §3.1 states the limitation, names the
per-domain computations at issue, and narrows the claim to fragmentation of
*the extracted direction* rather than of an established single concept. It also
notes that refusal is less exposed, since its domains share a task and vary
only in subject matter.

**Declined (experiment).** The clean fix is an instructed honest-vs-deceptive
contrast on identical questions, holding the task fixed. That needs a new
dataset and is a paper of its own; we name it as the required follow-up rather
than gesture at it.

### R1-W4 — "Functional fragmentation" overclaims what probe transfer shows

Correct. Poor transfer is consistent with a probe preferring a strong
domain-specific feature over a weaker shared one.

**Done.** §3.5 now states both explanations, and says the test rules out only
the weakest one (identical decodable content). "Functional" is removed from the
abstract, introduction, and probing discussion. The shared-subspace possibility
is now tested rather than argued about (see R2-W2).

### R1-W6 — Steering is too limited

> *"...only 20 prompts per domain, at a single layer 18, using a single
> coefficient of 4.0 ... behavioral validation is performed only for refusal."*

Correct on every count. A refusal rate over 20 prompts moves in steps of 0.05,
so the null result was only ever established to a resolution of one or two
examples.

**Done.** `src/analysis/run_steering_sweep.py` loads the model **once** (the old
driver reloaded per setting, which is what made a sweep unaffordable) and
sweeps layer × coefficient, turning own-vs-global into a dose-response
comparison. Coefficient 0 is carried as a leaking-hook control.
`--strict-heldout` makes the previously-silent fallback to overlapping prompts
a hard error, so the acknowledged contamination cannot reach a number by
default. `--heldout-frac` splits by fraction for the unevenly sized domains.

**Done (honesty behavior).** `src/analysis/honesty_behavior.py` scores honesty
by forced-choice log-probability preference between the dataset's own true and
false answers, length-normalized. No judge model (whose own domain biases would
confound this specific comparison), continuous rather than 0.05-quantized, and
one forward pass per answer instead of generation — which is what makes
sweeping honesty affordable at all.

**Queued:** run both sweeps. Note the honesty eval is a preference between two
supplied answers, not free generation; report it as the weaker behavioral
claim it is.

### R1-W7 — Base-vs-instruct phrasing

**Done.** §5.4 now states that the two checkpoints differ by the whole of an
undisclosed post-training pipeline (SFT *and* preference optimization), that we
cannot attribute the effect to RLHF specifically, and that the defensible claim
is narrow and about this model pair. The broader question is answered by
running the comparison across all seven base/instruct pairs instead.

---

## Reviewer 2

### R2-W1 — Tan et al. (2024) not discussed

**Done.** Cited and discussed in a new Related Work paragraph. The claim that
cross-domain generalization is "usually assumed rather than measured" is
removed; the contribution is restated as the *controlled attribution* (holding
concept, source, schema, and model fixed while varying only domain) rather than
as first observation.

### R2-W2 — Levinstein & Herrmann and Bürger et al. not cited

**Done (citations).** Both added, with Levinstein & Herrmann framed as
converging evidence for the honesty result and Bürger et al. as a specific
alternative interpretation.

**Done (analysis).** `src/analysis/subspace.py` answers the reviewer's direct
question — *"Would a rank two analysis change the fragmentation conclusion?"* —
on our own cached activations, at three levels of stringency:

1. Singular spectrum of the stacked per-domain directions (rank-1 / rank-2
   mass, participation ratio).
2. Principal angles between per-domain top-2 subspaces of mean-centered
   contrast vectors.
3. **Leave-one-domain-out subspace generalization** — build a rank-k subspace
   from the other domains, measure the held-out direction's squared projection
   into it, against the `k/d` chance baseline for a random rank-k subspace.

Test 3 is the one that can actually change the conclusion. If honesty domains
share a plane, a rank-2 LODO subspace will capture a held-out direction well
above chance even where pairwise cosines are low — which would reframe the
result from "honesty fragments" to "honesty is low-rank but not rank-1", a
better paper than the one submitted.

### R2-W3 — Single 3B model

**Done (infrastructure).** `config.py` now defines 14 models on two orthogonal
axes, all as base/instruct pairs:

- **Scale ladder** (architecture fixed, capacity varies): Qwen 2.5 at 0.5B,
  1.5B, 3B, 7B. This is the axis that directly addresses the §5.5
  non-replication.
- **Family axis** (capacity ~fixed, architecture and data vary): Gemma 2 2B,
  Llama 3.2 3B, Qwen 2.5 3B.

**Queued.** Feasible on the free tier: the geometry analysis needs only forward
passes (~20 min/model at 4-bit), so breadth is cheap. Steering is the expensive
part and stays focused on the primary model.

### R2-W4 — Steering coarse

Same as R1-W6; addressed by the sweep.

---

## Reviewer 3

### R3-W1 — More sizes and families; multilingual

**Partly done.** Sizes and families are covered by the sweep above.

**Declined (multilingual).** Every domain partition, the refusal keyword
scorer, and the TruthfulQA source are English-only. Doing this properly means
translated-and-validated contrastive pairs per language, and doing it
improperly would produce a confound (translation artifacts) of exactly the kind
this paper exists to control. Named as future work.

### R3-W2 — More concepts (toxicity, hate speech)

**Declined for this revision.** `config.py` already carries a `harmlessness`
stub with no domains. Adding a third concept credibly means a fourth domain
partition with the same single-source discipline that makes honesty the clean
experiment; done hastily it would reintroduce the source confound (T1) the
design exists to eliminate. The scale and family axes buy more, per unit of
compute, against the generalizability concern both R2 and R3 raise.

### R3 — Limitations placement

**Done.** Limitations now follows Conclusion.

---

## Re-run order

Nothing in the paper's numbers survives untouched, so sequence matters.

1. **Extraction** for the scale ladder and family set, both concepts, all
   base/instruct pairs.
2. **Controls** (`run_controls_from_cache`) — LODO, reliability, permutation.
   *Read the math and privacy reliability numbers before writing any prose.*
3. **Subspace** (`run_subspace_from_cache`) — rank-2 and LODO subspace.
4. **Probes** — unchanged code, re-run for the new models.
5. **Steering sweeps** — refusal (generation) and honesty (forced choice) on
   the primary model.
6. **Rewrite Results** against whatever comes back.

Step 2 is the gate. If within-domain reliability at early layers is low, the
early-layer fragmentation claim goes away and the paper becomes a more careful
and more honest one: universality is measurable where directions are reliable,
and much of the apparent early-layer structure in this literature may be
estimation noise. That is a publishable result, and arguably a better one than
the original claim.
