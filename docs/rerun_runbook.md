# Re-run Runbook — Reviewer Revisions

Everything the resubmission needs, in dependency order, sized for the Kaggle
free tier (2×T4, ~30 GPU h/week, 9 h per session).

**Read `docs/reviewer_response.md` first** for why each step exists.

The budget shape is worth internalizing: the geometry analysis is only forward
passes, so breadth across models is cheap (~20 min/model). Steering requires
autoregressive generation and is the expensive part, so it stays on the primary
model only.

---

## Step 0 — Rebuild the gitignored data (every fresh session)

`data/prompt_pairs_promptbased/` and `data/truthfulqa/` are not committed.

```bash
python scripts/convert_truthfulqa.py
python scripts/merge_truthfulqa_into_honesty.py
python scripts/build_refusal_promptbased.py --max-per-domain 120
python scripts/validate_prompt_pairs.py --strict
```

The validator must pass before anything else runs.

---

## Step 1 — Consider expanding the two starved domains

Before spending GPU hours, note what the controls are likely to find:

| concept | domain | pairs | half for split-half |
|---|---|---:|---:|
| honesty | factual_trivia | 526 | 263 |
| honesty | politics_opinion | 179 | 89 |
| honesty | personal_advice | 121 | 60 |
| honesty | **math** | **59** | **29** |
| refusal | medical_legal | 446 | 223 |
| refusal | illegal_activity | 162 | 81 |
| refusal | violence | 91 | 45 |
| refusal | **privacy** | **29** | **14** |

Estimating a direction in 2048 dimensions from 29 pairs (math) or 14 (privacy)
is very noisy. **Math carries the paper's headline honesty claim.** If its
within-domain reliability is near zero, its "near-orthogonality" is
unfalsifiable rather than fragmented, and the claim has to go.

Expanding math is cheap and does not need a GPU:

```bash
python scripts/add_math_honesty_pairs.py --target 200
python scripts/validate_prompt_pairs.py --strict
```

Doing this *before* extraction avoids re-extracting later. It is optional only
if you are willing to report that math is under-powered.

---

## Step 2 — Extraction across the model sweep

One session per few models. Small models need no quantization; 3B+ run 4-bit.

```bash
python -m src.extraction.batch_extract \
    --model qwen-2.5-3b-instruct \
    --concepts honesty refusal \
    --max-pairs-per-domain 200 \
    --output results/activations/
```

Models, in priority order (stop wherever the budget runs out — the analysis
handles a partial sweep):

1. `qwen-2.5-3b-instruct`, `qwen-2.5-3b` — the primary pair, needed for
   everything else.
2. `qwen-2.5-1.5b{,-instruct}`, `qwen-2.5-7b{,-instruct}` — the scale ladder,
   which is what the §5.5 non-replication actually leaves open.
3. `llama-3.2-3b{,-instruct}`, `gemma-2-2b{,-it}` — the family axis.
4. `qwen-2.5-0.5b{,-instruct}` — cheapest, fills in the bottom of the ladder.

> Activations are large. Write them to Kaggle working storage and run Step 3 in
> the **same session**; do not plan to persist them between sessions.

---

## Step 3 — Controls and subspace analysis (CPU-bound, same session)

```bash
python -m src.analysis.run_replication \
    --models qwen-2.5-3b-instruct qwen-2.5-3b \
    --concepts honesty refusal \
    --activations results/activations \
    --output results/replication \
    --reliability-floor 0.2
```

Outputs `replication_<concept>_<model>.json` per model and one
`replication_summary.json`, plus a console table.

**This is the gate for the whole paper. Read it before writing any prose.**

Read in this order:

1. `within_reliability_mean` and `gated_layers` — how many layers carry a
   measurable direction at all. If most early layers are gated, the
   early-layer fragmentation claim is gone.
2. `lodo_cosine_mean` — the corrected headline. Compare against the old
   pooled-global numbers (honesty 0.522 min, refusal 0.83–0.94). LODO can only
   be lower.
3. `fragmentation_gap_mean` — cross-domain cosine against the within-domain
   ceiling at matched n. **This is the real fragmentation measure.** Near zero
   means no fragmentation detectable above estimation noise.
4. `lodo_subspace_rank2_mean` vs its chance baseline — if this is high where
   rank-1 cosine is low, the finding becomes "low-rank, not rank-1", which
   answers Reviewer 2 directly and is a better result than the original.

---

## Step 4 — Probe transfer (unchanged code, new models)

```bash
python -m src.analysis.probe_transfer \
    --model qwen-2.5-3b-instruct --concept honesty \
    --activations results/activations --output results/replication
```

---

## Step 5 — Steering sweeps (primary model only)

Always dry-run first — it prints the cell count and held-out sizes without
loading the model.

Refusal (generation; the expensive one):

```bash
python -m src.analysis.run_steering_sweep \
    --model qwen-2.5-3b-instruct --concept refusal \
    --layers 12 18 24 --coeffs -4 -2 0 2 4 8 \
    --n-heldout 60 --heldout-frac 0.3 --strict-heldout \
    --dry-run
```

Drop `--dry-run` to run. 3 layers × 6 coeffs × 4 domains × 2 conditions = 144
cells; at 60 prompts and 64 new tokens that is ~8.6k generations. Budget ~2–3 h
on T4. Cut `--layers` to `18` first if the session is tight.

Honesty (forced choice; much cheaper — no generation):

```bash
python -m src.analysis.run_steering_sweep \
    --model qwen-2.5-3b-instruct --concept honesty \
    --layers 8 16 24 --coeffs -4 -2 0 2 4 8 \
    --n-heldout 100 --heldout-frac 0.3 --strict-heldout
```

Check `summary._zero_coeff_control.max_abs_effect` in the output. It must be
~0; anything else means the steering hook is leaking across conditions and
every number in the sweep is suspect.

---

## Step 6 — Figures and paper

Regenerate figures, then rewrite Results (§5) against the actual numbers.
Sections whose prose is already written and does not depend on the re-run:
Related Work, the two new Methods subsections, the honesty construct-validity
paragraph, the base-vs-instruct scoping, and Limitations placement.

Results text still to write:

- §5.x **Controls** (`sec:results-controls`) — reliability, LODO, gap. Referenced
  from the introduction already.
- §5.x **Subspace** (`sec:subspace` results half) — the rank-2 answer.
- §5.x **Replication** (`sec:results-replication`) — the scale and family axes,
  referenced from the RLHF section already.
- Rewrite §5.3 steering against the sweep.

Those four `\label`s are referenced in the current text, so **the paper will not
compile with correct cross-references until they exist.** That is deliberate —
it prevents shipping a draft that silently omits them.

---

## If the results overturn the paper

Plausible, and fine. The strongest honest outcomes:

- **Early-layer honesty fragmentation was estimation noise.** Then the paper
  becomes a methodological correction: much of the apparent early-layer
  structure in this literature is unmeasurable at typical dataset sizes, and
  here is the protocol that shows it. Reviewer 1's objections become the
  paper's contribution.
- **Honesty is rank-2, not fragmented.** Then it confirms and extends Bürger et
  al. on new models, with a controlled domain design they did not have.
- **Refusal universality survives LODO but honesty does not.** The original
  story, now on defensible footing.

All three are publishable. What is not publishable is the current 0.522 against
a pooled global.
