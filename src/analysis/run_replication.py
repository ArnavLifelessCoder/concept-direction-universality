"""
run_replication.py — Controls and Subspace Analysis Across the Model Sweep
==========================================================================
Runs the reviewer-response analyses over many models in one pass, so the
paper's claims are stated per model rather than generalized from one.

For each requested model and concept this driver runs, from cached
activations:

- the controls in :mod:`~src.analysis.geometry_controls` (leave-one-domain-out
  cosine, within-domain split-half reliability, matched cross-domain cosine,
  sign-flip permutation null, cross-validated effect size), and
- the subspace tests in :mod:`~src.analysis.subspace` (spectrum, principal
  angles, leave-one-domain-out subspace generalization).

It then writes one JSON per (model, concept) plus a cross-model summary table
keyed by the two sweep axes defined in ``config.py``: the Qwen 2.5 scale ladder
and the Gemma/Llama/Qwen family set.

Reliability gating
------------------
The summary marks a layer **uninterpretable** when its within-domain
split-half reliability is at or below ``--reliability-floor``, or when the
permutation null fails to reject. Geometry at such a layer describes
estimation noise, not representation, and reporting a low cosine there as
"fragmentation" is the specific error the controls exist to prevent. Gated
layers are counted and listed rather than silently dropped.

This matters concretely for the current data. Domains are far from equal
(honesty ``math`` has 59 pairs against ``factual_trivia``'s 526; refusal
``privacy`` has 29 against ``medical_legal``'s 446), and a direction estimated
from ~29 pairs in ``d_model`` dimensions may not clear the floor at any depth.

Example
-------
::

    python -m src.analysis.run_replication \\
        --models qwen-2.5-0.5b qwen-2.5-1.5b qwen-2.5-3b qwen-2.5-7b \\
        --concepts honesty refusal \\
        --activations results/activations --output results/replication
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    BASE_INSTRUCT_PAIRS,
    CONCEPTS,
    FAMILY_SET,
    MODELS,
    PATHS,
    SCALE_LADDER,
)
from src.analysis.directions import balanced_subsample_indices
from src.analysis.geometry_controls import run_controls_from_cache
from src.analysis.subspace import run_subspace_from_cache

logger = logging.getLogger(__name__)


def analyze_model_concept(
    model_key: str,
    concept: str,
    activations_dir: Path,
    output_dir: Path,
    layers: Optional[Sequence[int]] = None,
    domains: Optional[Sequence[str]] = None,
    n_splits: int = 200,
    n_permutations: int = 500,
    balance_domains: bool = True,
    reliability_floor: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> Optional[Dict[str, object]]:
    """Run controls and subspace analysis for one model and concept.

    Args:
        model_key: Key into ``config.MODELS``.
        concept: Concept name.
        activations_dir: Cached activations root.
        output_dir: Where to write the report.
        layers: Layers to analyze; defaults to every layer of the model.
        domains: Domains; defaults to the concept's configured domains.
        n_splits / n_permutations / seed: Passed to the controls.
        balance_domains: Subsample every domain to the smallest domain's pair
            count, so the pooled global direction is not dominated by the
            largest domain (threat T4).
        reliability_floor: Within-domain split-half cosine at or below which a
            layer is marked uninterpretable.
        device: Torch device for loading activations.

    Returns:
        The report dict, or ``None`` when no cached activations were found.
    """
    cfg = MODELS[model_key]
    layers = list(layers) if layers is not None else list(range(cfg.n_layers))
    domains = list(domains) if domains is not None else list(CONCEPTS[concept].domains)

    indices_by_domain = None
    if balance_domains:
        indices_by_domain = balanced_subsample_indices(
            activations_dir, model_key, concept, domains, layers,
            seed=seed, device=device,
        ) or None

    logger.info("=== %s / %s: %d layers, %d domains ===",
                model_key, concept, len(layers), len(domains))

    controls = run_controls_from_cache(
        activations_dir, model_key, concept, domains, layers,
        n_splits=n_splits, n_permutations=n_permutations, seed=seed,
        device=device, indices_by_domain=indices_by_domain,
    )
    if not controls:
        logger.warning("No cached activations for %s/%s — skipping.", model_key, concept)
        return None

    subspace = run_subspace_from_cache(
        activations_dir, model_key, concept, domains, layers,
        ranks=(1, 2, 3), within_rank=2,
        device=device, indices_by_domain=indices_by_domain,
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_key,
        "model_name": cfg.name,
        "is_instruct": cfg.is_instruct,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "concept": concept,
        "domains": domains,
        "balance_domains": balance_domains,
        "reliability_floor": reliability_floor,
        "n_splits": n_splits,
        "n_permutations": n_permutations,
        "seed": seed,
        "controls": {str(k): v for k, v in controls.items()},
        "subspace": {str(k): v for k, v in subspace.items()},
        "summary": summarize_model_concept(controls, subspace, reliability_floor),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"replication_{concept}_{model_key}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", out)
    return report


def summarize_model_concept(
    controls: Dict[int, Dict[str, object]],
    subspace: Dict[int, Dict[str, object]],
    reliability_floor: float,
) -> Dict[str, object]:
    """Condense per-layer controls into the numbers the paper reports.

    A layer is *interpretable* when its mean within-domain split-half
    reliability exceeds ``reliability_floor``. Aggregates are computed over
    interpretable layers only, because averaging in layers where the direction
    is noise would drag the reported cosine toward zero and be read as
    fragmentation.
    """
    interpretable: List[int] = []
    gated: List[int] = []

    for layer, c in controls.items():
        within = c.get("within_mean", float("nan"))
        if isinstance(within, float) and not np.isnan(within) and within > reliability_floor:
            interpretable.append(layer)
        else:
            gated.append(layer)

    def _series(key: str, layers: Sequence[int]) -> List[float]:
        vals = []
        for l in layers:
            v = controls[l].get(key)
            if isinstance(v, (int, float)) and not np.isnan(float(v)):
                vals.append(float(v))
        return vals

    lodo = _series("lodo_mean", interpretable)
    gap = _series("gap", interpretable)
    within = _series("within_mean", interpretable)
    cross = _series("cross_domain_mean", interpretable)

    # Rank-2 leave-one-domain-out subspace generalization, over the same layers.
    lodo_sub: List[float] = []
    chance: List[float] = []
    for l in interpretable:
        entry = subspace.get(l, {}).get("lodo", {}).get("per_rank", {}).get(2)
        if entry:
            lodo_sub.append(float(entry["mean"]))
            chance.append(float(entry["chance"]))

    def _stat(vals: Sequence[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals))}

    return {
        "n_layers_total": len(controls),
        "n_layers_interpretable": len(interpretable),
        "gated_layers": sorted(gated),
        "interpretable_layers": sorted(interpretable),
        "lodo_cosine": _stat(lodo),
        "within_domain_reliability": _stat(within),
        "cross_domain_cosine_matched": _stat(cross),
        "fragmentation_gap": _stat(gap),
        "lodo_subspace_rank2": _stat(lodo_sub),
        "lodo_subspace_rank2_chance": float(np.mean(chance)) if chance else float("nan"),
    }


def build_cross_model_table(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Assemble the per-model summaries into the two sweep axes.

    Returns:
        Dict with ``rows`` (one per model/concept), ``scale_ladder``,
        ``family_set``, and ``base_vs_instruct`` (LODO cosine delta within each
        configured base/instruct pair).
    """
    rows = []
    for r in reports:
        s = r["summary"]
        rows.append({
            "model": r["model"],
            "concept": r["concept"],
            "is_instruct": r["is_instruct"],
            "d_model": r["d_model"],
            "n_layers": r["n_layers"],
            "n_interpretable": s["n_layers_interpretable"],
            "lodo_cosine_mean": s["lodo_cosine"]["mean"],
            "lodo_cosine_min": s["lodo_cosine"]["min"],
            "within_reliability_mean": s["within_domain_reliability"]["mean"],
            "fragmentation_gap_mean": s["fragmentation_gap"]["mean"],
            "lodo_subspace_rank2_mean": s["lodo_subspace_rank2"]["mean"],
        })

    def _lookup(model: str, concept: str) -> Optional[Dict[str, object]]:
        for row in rows:
            if row["model"] == model and row["concept"] == concept:
                return row
        return None

    concepts = sorted({r["concept"] for r in rows})

    base_vs_instruct = []
    for base, instruct in BASE_INSTRUCT_PAIRS:
        for concept in concepts:
            b, i = _lookup(base, concept), _lookup(instruct, concept)
            if b and i:
                base_vs_instruct.append({
                    "concept": concept,
                    "base": base,
                    "instruct": instruct,
                    "base_lodo": b["lodo_cosine_mean"],
                    "instruct_lodo": i["lodo_cosine_mean"],
                    # positive => alignment consolidates directions
                    "delta": i["lodo_cosine_mean"] - b["lodo_cosine_mean"],
                })

    return {
        "rows": rows,
        "scale_ladder": [
            _lookup(m, c) for c in concepts for m in SCALE_LADDER if _lookup(m, c)
        ],
        "family_set": [
            _lookup(m, c) for c in concepts for m in FAMILY_SET if _lookup(m, c)
        ],
        "base_vs_instruct": base_vs_instruct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run geometry controls and subspace analysis across models.",
    )
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--concepts", nargs="+", default=["honesty", "refusal"])
    parser.add_argument("--domains", nargs="*", default=None,
                        help="Override the concept's configured domains.")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--activations", default=str(PATHS.activations))
    parser.add_argument("--output", default=str(PATHS.results / "replication"))
    parser.add_argument("--n-splits", type=int, default=200)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--reliability-floor", type=float, default=0.2,
                        help="Within-domain split-half cosine at or below which "
                             "a layer is treated as uninterpretable.")
    parser.add_argument("--no-balance", action="store_true",
                        help="Disable per-domain balanced subsampling.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_dir = Path(args.output)
    reports: List[Dict[str, object]] = []

    for model_key in args.models:
        for concept in args.concepts:
            report = analyze_model_concept(
                model_key=model_key,
                concept=concept,
                activations_dir=Path(args.activations),
                output_dir=output_dir,
                layers=args.layers,
                domains=args.domains,
                n_splits=args.n_splits,
                n_permutations=args.n_permutations,
                balance_domains=not args.no_balance,
                reliability_floor=args.reliability_floor,
                seed=args.seed,
                device=args.device,
            )
            if report:
                reports.append(report)

    if not reports:
        logger.error(
            "No reports produced. Check that activations exist under %s.",
            args.activations,
        )
        return

    table = build_cross_model_table(reports)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "replication_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.now().isoformat(timespec="seconds"),
             "reliability_floor": args.reliability_floor,
             **table},
            f, indent=2,
        )
    logger.info("Wrote cross-model summary: %s", out)

    # Console table — the numbers to read first.
    print(f"\n{'model':<26} {'concept':<9} {'LODO':>7} {'within':>7} "
          f"{'gap':>7} {'rank2':>7} {'ok/L':>8}")
    print("-" * 76)
    for row in table["rows"]:
        print(
            f"{row['model']:<26} {row['concept']:<9} "
            f"{row['lodo_cosine_mean']:>7.3f} {row['within_reliability_mean']:>7.3f} "
            f"{row['fragmentation_gap_mean']:>7.3f} "
            f"{row['lodo_subspace_rank2_mean']:>7.3f} "
            f"{row['n_interpretable']:>3}/{row['n_layers']:<4}"
        )


if __name__ == "__main__":
    main()
