"""
run_steering_sweep.py — Coefficient and Layer Sweeps for Steering
=================================================================
Replaces the single-point steering experiment (one layer, one coefficient,
20 prompts, refusal only) with a sweep, and adds behavioral validation for
honesty.

Why the original design could not support its own null result: a refusal rate
over 20 prompts moves in increments of 0.05, so "global steers as well as
own-domain" was only ever established up to a resolution of one or two
examples. A null at one coefficient and one layer is also indistinguishable
from a coefficient that was too small, too large, or applied at the wrong
depth. This driver addresses all three:

- **Sweeps** coefficient and layer, so the comparison is a dose-response curve
  rather than a single point. If global and own-domain directions are
  functionally interchangeable, their curves should coincide across the whole
  range, which is a far stronger claim than agreement at one setting.
- **Scales** the held-out set, with a hard guarantee that evaluation prompts
  were never used for direction estimation (see ``--skip-first`` and
  ``--strict-heldout``).
- **Covers honesty** via the forced-choice log-probability metric in
  :mod:`~src.analysis.honesty_behavior`, so the concept the paper's main
  fragmentation claim is about finally has behavioral evidence.

The model is loaded **once** and reused across every sweep cell. The original
per-run driver reloaded weights for each setting, which is what made a sweep
unaffordable rather than any intrinsic cost of the sweep itself.

Example
-------
Refusal, sweeping coefficient at three layers::

    python -m src.analysis.run_steering_sweep \\
        --model qwen-2.5-3b-instruct --concept refusal \\
        --layers 12 18 24 --coeffs -4 -2 0 2 4 8 \\
        --n-heldout 60 --skip-first 120 --strict-heldout

Honesty, forced-choice scoring (no generation, much cheaper)::

    python -m src.analysis.run_steering_sweep \\
        --model qwen-2.5-3b-instruct --concept honesty \\
        --layers 8 16 24 --coeffs -4 -2 0 2 4 8 \\
        --n-heldout 100 --skip-first 150 --strict-heldout
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ANALYSIS, CONCEPTS, MODELS, PATHS
from src.analysis.honesty_behavior import (
    load_truth_items,
    steered_truth_preference,
    truth_preference,
)
from src.analysis.steering import (
    _make_steering_hook,
    compute_refusal_rate,
    steer_generation,
)
from src.extraction.batch_extract import discover_pairs_by_domain
from src.extraction.cache_utils import load_direction
from src.extraction.extract_activations import pair_to_contrastive_texts

logger = logging.getLogger(__name__)


# ============================================================
# Model layer access (shared by both concepts)
# ============================================================

def get_transformer_layers(model) -> Sequence:
    """Return the model's transformer layer list across common HF layouts."""
    for attr_name in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr_name.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("Cannot locate transformer layers for steering hook.")


# ============================================================
# Held-out data with a disjointness guarantee
# ============================================================

def heldout_by_domain(
    data_dir: Path,
    concept: str,
    domains: Sequence[str],
    n_heldout: int,
    skip_first: int,
    strict: bool,
    heldout_frac: Optional[float] = None,
) -> Dict[str, List[dict]]:
    """Held-out pairs per domain, disjoint from the direction-estimation set.

    Two ways to carve the split:

    - **Fixed** (``heldout_frac=None``): reserve the first ``skip_first`` pairs
      for direction estimation and take ``n_heldout`` from the remainder. Fine
      when every domain is large.
    - **Fractional** (``heldout_frac`` set): hold out the last
      ``heldout_frac`` of each domain, capped at ``n_heldout``. Use this when
      domains differ wildly in size. They do here: honesty ranges from 526
      pairs (``factual_trivia``) down to **59** (``math``), and refusal from 446
      (``medical_legal``) down to **29** (``privacy``), so a fixed
      ``skip_first`` of 120 leaves the small domains with nothing and silently
      forces the contaminating fallback.

    The original driver reused the *last* ``n_heldout`` pairs when a domain
    lacked spares, overlapping direction estimation; that was then reported as
    a limitation. Here the fallback is opt-in: ``strict=True`` raises instead,
    so contamination cannot reach a reported number by default.

    Args:
        data_dir: Prompt-pair root.
        concept: Concept name.
        domains: Domains to load.
        n_heldout: Maximum held-out pairs per domain.
        skip_first: Leading pairs reserved for direction estimation (fixed mode).
        strict: Raise rather than fall back to overlapping pairs.
        heldout_frac: If set, hold out this fraction of each domain instead.

    Returns:
        {domain: [pair dicts]}.

    Raises:
        RuntimeError: If a domain has no pairs, or (under ``strict``) too few
            to form a disjoint held-out set.
    """
    by_domain = discover_pairs_by_domain(Path(data_dir), concept, domains)
    out: Dict[str, List[dict]] = {}

    for domain in domains:
        pairs = by_domain.get(domain, [])
        if not pairs:
            raise RuntimeError(
                f"No pairs for {concept}/{domain} under {data_dir}. If this is a "
                "fresh checkout, rebuild the prompt-based set first: "
                "python scripts/build_refusal_promptbased.py --max-per-domain 120"
            )

        if heldout_frac is not None:
            k = min(n_heldout, int(len(pairs) * heldout_frac))
            if k < 2:
                raise RuntimeError(
                    f"{concept}/{domain}: {len(pairs)} pairs at "
                    f"heldout_frac={heldout_frac} yields {k} held-out items. "
                    "This domain is too small for a behavioral evaluation; add "
                    "pairs or drop it from the sweep."
                )
            out[domain] = pairs[-k:]
            logger.info(
                "%s/%s: %d of %d pairs held out (%.0f%%); %d remain for fitting.",
                concept, domain, k, len(pairs), 100 * k / len(pairs), len(pairs) - k,
            )
            continue

        spare = pairs[skip_first:]
        if len(spare) < n_heldout:
            msg = (
                f"{concept}/{domain}: only {len(spare)} pairs beyond "
                f"skip-first={skip_first}, wanted {n_heldout} held-out "
                f"({len(pairs)} pairs total)."
            )
            if strict:
                raise RuntimeError(
                    msg + " Use --heldout-frac for unevenly sized domains, or "
                    "reduce --n-heldout / --skip-first. (Continuing without "
                    "--strict-heldout would reuse pairs used to fit the direction.)"
                )
            logger.warning(
                "%s Falling back to the last %d pairs, which OVERLAP direction "
                "estimation — this contaminates the held-out claim.",
                msg, n_heldout,
            )
            out[domain] = pairs[-n_heldout:]
        else:
            out[domain] = spare[:n_heldout]

    return out


# ============================================================
# Direction bank
# ============================================================

def load_direction_bank(
    directions_dir: Path,
    model_key: str,
    concept: str,
    domains: Sequence[str],
    layers: Sequence[int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load global and per-domain directions for every swept layer.

    Returns:
        {layer: {"global": vec, domain: vec, ...}}. Layers with no usable
        direction are omitted rather than failing the whole sweep.
    """
    bank: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        entry: Dict[str, torch.Tensor] = {}
        for source in ("global", *domains):
            try:
                entry[source] = load_direction(
                    directions_dir, model_key, concept, source, layer,
                )
            except FileNotFoundError:
                logger.warning(
                    "Missing direction %s/%s/%s/layer%d — skipping that condition.",
                    model_key, concept, source, layer,
                )
        if "global" in entry:
            bank[layer] = entry
        else:
            logger.warning("No global direction at layer %d — skipping layer.", layer)
    return bank


# ============================================================
# Per-concept sweep cells
# ============================================================

def _refusal_cell(
    model, tokenizer, prompts: List[str], direction: torch.Tensor,
    layer: int, coeff: float, max_new_tokens: int, source: str,
) -> Dict[str, object]:
    """One (layer, coeff, direction) cell scored by refusal rate on generations."""
    gen = steer_generation(
        model, tokenizer, prompts, direction,
        steering_layer=layer, steering_coeff=coeff,
        max_new_tokens=max_new_tokens, direction_source=source,
    )
    rates = compute_refusal_rate(gen)
    return {
        "metric": "refusal_rate",
        "n": len(gen),
        **rates,
        "effect": rates["refusal_rate_delta"],
        "examples": [
            {"prompt": r.prompt[:200], "steered": r.steered_output[:200]}
            for r in gen[:2]
        ],
    }


def _honesty_cell(
    model, tokenizer, items, direction: torch.Tensor,
    layer: int, coeff: float, baseline: Dict[str, object], device: str,
) -> Dict[str, object]:
    """One (layer, coeff, direction) cell scored by forced-choice truth preference."""
    steered = steered_truth_preference(
        model, tokenizer, items, direction, layer, coeff,
        make_hook=_make_steering_hook,
        get_layers=get_transformer_layers,
        device=device,
    )
    return {
        "metric": "truth_preference",
        "n": steered["n"],
        "baseline_pref": baseline["mean_pref"],
        "steered_pref": steered["mean_pref"],
        "baseline_accuracy": baseline["accuracy"],
        "steered_accuracy": steered["accuracy"],
        "effect": steered["mean_pref"] - baseline["mean_pref"],
        "accuracy_delta": steered["accuracy"] - baseline["accuracy"],
    }


# ============================================================
# Sweep driver
# ============================================================

def run_sweep(
    model_key: str,
    concept: str,
    domains: Sequence[str],
    layers: Sequence[int],
    coeffs: Sequence[float],
    data_dir: Path,
    directions_dir: Path,
    output_dir: Path,
    n_heldout: int = 60,
    skip_first: int = 120,
    strict_heldout: bool = True,
    heldout_frac: Optional[float] = None,
    max_new_tokens: int = 64,
    cross_domain: bool = False,
    device: str = "cuda",
    dry_run: bool = False,
) -> Dict[str, object]:
    """Sweep steering coefficient and layer for one model and concept.

    Conditions per (layer, coeff, domain): ``global`` (direction pooled across
    domains) and ``own`` (that domain's own direction). With
    ``cross_domain=True``, every other domain's direction is added too, which
    multiplies cost by roughly the domain count.

    Args:
        model_key: Key into ``config.MODELS``.
        concept: "refusal" (generation + refusal rate) or "honesty"
            (forced-choice truth preference).
        domains: Domains to evaluate.
        layers: Layers to intervene at.
        coeffs: Steering coefficients. Include 0.0 as an internal control: a
            nonzero measured effect at coefficient 0 indicates a leaking hook.
        data_dir / directions_dir / output_dir: Paths.
        n_heldout: Held-out items per domain.
        skip_first: Leading pairs reserved for direction estimation.
        strict_heldout: Fail rather than reuse fitting pairs.
        max_new_tokens: Generation length (refusal only).
        cross_domain: Also steer each domain with every other domain's direction.
        device: Torch device.
        dry_run: Report the cell count and exit without loading the model.

    Returns:
        The report dict (also written to ``output_dir``).
    """
    concept_domains = list(domains)
    n_cells = len(layers) * len(coeffs) * len(concept_domains) * (
        len(concept_domains) + 1 if cross_domain else 2
    )
    logger.info(
        "Sweep: %d layers x %d coeffs x %d domains = %d cells, %d items each",
        len(layers), len(coeffs), len(concept_domains), n_cells, n_heldout,
    )
    if concept == "refusal":
        logger.info(
            "Refusal scoring generates %d completions of <=%d tokens.",
            n_cells * n_heldout, max_new_tokens,
        )

    heldout = heldout_by_domain(
        Path(data_dir), concept, concept_domains,
        n_heldout=n_heldout, skip_first=skip_first, strict=strict_heldout,
        heldout_frac=heldout_frac,
    )

    if dry_run:
        return {
            "dry_run": True, "n_cells": n_cells,
            "n_heldout_by_domain": {d: len(v) for d, v in heldout.items()},
        }

    bank = load_direction_bank(
        Path(directions_dir), model_key, concept, concept_domains, layers,
    )
    if not bank:
        raise RuntimeError(
            f"No directions found for {model_key}/{concept} at layers {list(layers)}."
        )

    # Load the model once for the whole sweep.
    # Steering hooks target HuggingFace `model.layers`, so use the HF backend.
    from src.extraction.extract_activations import load_model_huggingface
    model, tokenizer = load_model_huggingface(model_key)
    model.eval()

    # Concept-specific evaluation payloads.
    if concept == "refusal":
        payload = {
            d: [pair_to_contrastive_texts(p)[1] for p in heldout[d]]
            for d in concept_domains
        }
        baselines: Dict[str, Dict[str, object]] = {}
    else:
        payload = load_truth_items(heldout, n_per_domain=n_heldout, skip_first=0)
        # Unsteered baseline is coefficient-independent, so score it once per
        # domain rather than inside every sweep cell.
        baselines = {
            d: truth_preference(model, tokenizer, payload[d], device=device)
            for d in concept_domains
        }
        for d, b in baselines.items():
            logger.info(
                "%s baseline: pref=%.4f acc=%.3f (n=%d)",
                d, b["mean_pref"], b["accuracy"], b["n"],
            )

    cells: List[Dict[str, object]] = []
    for layer, coeff, domain in product(layers, coeffs, concept_domains):
        if layer not in bank:
            continue
        entry = bank[layer]

        sources = ["global"]
        if domain in entry:
            sources.append(domain)
        if cross_domain:
            sources += [d for d in concept_domains if d != domain and d in entry]

        for source in sources:
            label = "own" if source == domain else (
                "global" if source == "global" else f"cross:{source}"
            )
            if concept == "refusal":
                res = _refusal_cell(
                    model, tokenizer, payload[domain], entry[source],
                    layer, coeff, max_new_tokens, source,
                )
            else:
                res = _honesty_cell(
                    model, tokenizer, payload[domain], entry[source],
                    layer, coeff, baselines[domain], device,
                )

            cells.append({
                "layer": layer, "coeff": coeff, "domain": domain,
                "direction_source": source, "condition": label, **res,
            })
            logger.info(
                "L%02d c=%+.1f %s [%s]: effect %+.4f",
                layer, coeff, domain, label, res["effect"],
            )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_key,
        "concept": concept,
        "domains": concept_domains,
        "layers": list(layers),
        "coeffs": list(coeffs),
        "n_heldout": n_heldout,
        "skip_first": skip_first,
        "strict_heldout": strict_heldout,
        "heldout_frac": heldout_frac,
        "cross_domain": cross_domain,
        "n_heldout_by_domain": {d: len(v) for d, v in payload.items()},
        "cells": cells,
        "summary": summarize_sweep(cells),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"steering_sweep_{concept}_{model_key}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote sweep report: %s", out)
    return report


def summarize_sweep(cells: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate sweep cells into the own-vs-global comparison.

    For each (layer, coeff), averages the effect over domains under the
    ``own`` and ``global`` conditions and reports the difference. The paper's
    null claim ("global steers as well as own-domain") requires
    ``own_minus_global`` to stay near zero across the whole sweep, not just at
    one setting.

    Also records the coefficient-0 control: any effect there is a leaking hook,
    not a steering result.
    """
    by_setting: Dict[str, Dict[str, List[float]]] = {}
    for c in cells:
        key = f"L{int(c['layer']):03d}_c{float(c['coeff']):+g}"
        cond = str(c["condition"])
        if cond.startswith("cross:"):
            cond = "cross"
        by_setting.setdefault(key, {}).setdefault(cond, []).append(float(c["effect"]))

    summary: Dict[str, object] = {}
    for key, conds in sorted(by_setting.items()):
        own = conds.get("own", [])
        glob = conds.get("global", [])
        entry: Dict[str, object] = {
            "own_mean_effect": sum(own) / len(own) if own else None,
            "global_mean_effect": sum(glob) / len(glob) if glob else None,
            "n_domains": max(len(own), len(glob)),
        }
        if own and glob:
            entry["own_minus_global"] = entry["own_mean_effect"] - entry["global_mean_effect"]
        if "cross" in conds:
            entry["cross_mean_effect"] = sum(conds["cross"]) / len(conds["cross"])
        summary[key] = entry

    zero = [c for c in cells if float(c["coeff"]) == 0.0]
    if zero:
        effects = [abs(float(c["effect"])) for c in zero]
        summary["_zero_coeff_control"] = {
            "max_abs_effect": max(effects),
            "note": "should be ~0; a nonzero value indicates a leaking steering hook",
        }
    return summary


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sweep steering coefficient and layer for one model/concept.",
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--concept", default="refusal", choices=["refusal", "honesty"])
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layers to intervene at.")
    parser.add_argument("--coeffs", type=float, nargs="+",
                        default=[-4.0, -2.0, 0.0, 2.0, 4.0, 8.0],
                        help="Steering coefficients; include 0.0 as a control.")
    parser.add_argument("--data-dir", default=str(PATHS.prompt_pairs))
    parser.add_argument("--directions", default=str(PATHS.directions))
    parser.add_argument("--output", default=str(PATHS.results))
    parser.add_argument("--n-heldout", type=int, default=60)
    parser.add_argument("--skip-first", type=int, default=120,
                        help="Leading pairs reserved for direction estimation.")
    parser.add_argument("--strict-heldout", action="store_true",
                        help="Fail rather than reuse direction-estimation pairs.")
    parser.add_argument("--heldout-frac", type=float, default=None,
                        help="Hold out this fraction of each domain instead of "
                             "using --skip-first. Use for unevenly sized domains "
                             "(honesty math has 59 pairs, refusal privacy 29).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cross-domain", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report cell count and held-out sizes, then exit.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    domains = args.domains or CONCEPTS[args.concept].domains

    report = run_sweep(
        model_key=args.model,
        concept=args.concept,
        domains=domains,
        layers=args.layers,
        coeffs=args.coeffs,
        data_dir=Path(args.data_dir),
        directions_dir=Path(args.directions),
        output_dir=Path(args.output),
        n_heldout=args.n_heldout,
        skip_first=args.skip_first,
        strict_heldout=args.strict_heldout,
        heldout_frac=args.heldout_frac,
        max_new_tokens=args.max_new_tokens,
        cross_domain=args.cross_domain,
        device=args.device,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
