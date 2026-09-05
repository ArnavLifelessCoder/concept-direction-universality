# Deliverables

Ready-to-import Kaggle notebooks for the resubmission re-run. See
`docs/rerun_runbook.md` for the full command reference and
`docs/reviewer_response.md` for why each analysis exists.

| Notebook | Session | Needs | Runtime |
|---|---|---|---|
| `session1_reliability_gate.ipynb` | 1 | GPU T4 ×2, Internet on | ~2.5–4 h |

## Before importing

The notebook clones the `reviewer-revisions` branch from GitHub. **Push it
first**, or the clone fails:

```bash
git push -u origin reviewer-revisions
```

## Importing into Kaggle

1. Kaggle → *Create* → *New Notebook* → *File* → *Import Notebook* → upload the
   `.ipynb`.
2. In the right-hand settings panel set **Accelerator: GPU T4 x2** and
   **Internet: On**. Both are required; the notebook stops early without them.
3. Run all cells top to bottom.
4. Download `replication_reports.zip` from the output pane when it finishes.

## What session 1 produces

`results/replication/replication_*.json` plus a `replication_summary.json`,
which together answer the three questions that gate the paper:

1. Which layers carry a measurable direction at all (split-half reliability +
   permutation null).
2. The leave-one-domain-out cosine — the corrected replacement for the
   pooled-global number that invalidated the previous submission's headline.
3. The gap between cross-domain cosine and the within-domain noise floor at
   matched sample size, which is the fragmentation measure that survives both
   corrections.

Unzip into `results/replication/` locally, then fill the `\PH{}` placeholders
in `paper/main.tex`:

```bash
grep -n '\\PH{' paper/main.tex
```

## Known constraint carried into this session

The `math` honesty domain has 59 pairs and **cannot be expanded with the
existing tooling** — `scripts/add_math_honesty_pairs.py` appends a fixed list of
30 pairs that are already merged, so it adds zero. Math carries the paper's
headline honesty claim, so step 4 of the notebook watches its reliability
specifically. If it falls at or below the floor, the near-orthogonality claim
has to be removed rather than hedged.
