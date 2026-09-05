"""
steering_sweep_plot.py — Dose-Response Figure for the Steering Sweep
====================================================================
Renders the figure that replaces the single-point steering table: mean effect
against steering coefficient, one line per condition (own-domain, global, and
optionally cross-domain), one panel per intervention layer.

The comparison the figure exists to support is *curve overlap*. If own-domain
and global directions are functionally interchangeable, their curves should
coincide across the whole coefficient range at every depth; a gap that opens
at some coefficient is the per-domain advantage the paper's second hypothesis
predicted. Panels therefore share a y-axis, so depths are visually comparable
and a large effect at one layer cannot be mistaken for a large effect
everywhere.

Coefficient 0 is drawn as a reference line: every curve must pass through
approximately zero effect there, since no steering is applied. A curve that
misses it indicates a leaking hook rather than a finding, and the figure is
designed to make that visible rather than to hide it.

Usage
-----
::

    python -m src.visualization.steering_sweep_plot \\
        --report results/steering_sweep_refusal_qwen-2.5-3b-instruct.json \\
        --output paper/figures/fig_steering_sweep.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Condition -> (label, colour, marker). Colours are colourblind-safe and
# distinguishable in greyscale by marker.
CONDITION_STYLE = {
    "own": ("Own-domain direction", "#D55E00", "o"),
    "global": ("Global direction", "#0072B2", "s"),
    "cross": ("Cross-domain direction", "#999999", "^"),
}


def aggregate_cells(
    cells: Sequence[Dict[str, object]],
) -> Dict[int, Dict[str, Dict[float, Dict[str, float]]]]:
    """Average sweep cells over domains.

    Cross-domain conditions are labelled ``cross:<source>`` per source; they are
    collapsed into a single ``cross`` series, since the figure asks whether
    *any* non-own direction behaves differently, not which one.

    Returns:
        {layer: {condition: {coeff: {"mean": float, "sem": float, "n": int}}}}
    """
    buckets: Dict[int, Dict[str, Dict[float, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for c in cells:
        effect = c.get("effect")
        if effect is None or (isinstance(effect, float) and np.isnan(effect)):
            continue
        cond = str(c["condition"])
        if cond.startswith("cross:"):
            cond = "cross"
        buckets[int(c["layer"])][cond][float(c["coeff"])].append(float(effect))

    out: Dict[int, Dict[str, Dict[float, Dict[str, float]]]] = {}
    for layer, conds in buckets.items():
        out[layer] = {}
        for cond, by_coeff in conds.items():
            out[layer][cond] = {}
            for coeff, vals in by_coeff.items():
                arr = np.asarray(vals, dtype=float)
                out[layer][cond][coeff] = {
                    "mean": float(arr.mean()),
                    # SEM over domains; with one domain there is no spread to show.
                    "sem": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                    "n": int(len(arr)),
                }
    return out


def plot_sweep(
    report: Dict[str, object],
    output: Path,
    figsize: Optional[Sequence[float]] = None,
    dpi: int = 300,
) -> Path:
    """Render the dose-response figure from a sweep report.

    Args:
        report: Parsed ``steering_sweep_*.json``.
        output: Destination path; the suffix selects the format. A PNG
            companion is written alongside a PDF for slide use.
        figsize: Figure size in inches; defaults to a column-width figure
            scaled by the number of layer panels.
        dpi: Raster resolution for the PNG companion.

    Returns:
        The path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = report.get("cells", [])
    if not cells:
        raise ValueError("Report contains no cells — was the sweep actually run?")

    agg = aggregate_cells(cells)
    layers = sorted(agg)
    concept = report.get("concept", "")
    metric = str(cells[0].get("metric", "effect"))

    ylabel = {
        "refusal_rate": "$\\Delta$ refusal rate",
        "truth_preference": "$\\Delta$ truth preference (log-prob)",
    }.get(metric, "Effect")

    if figsize is None:
        figsize = (max(3.3, 2.4 * len(layers)), 2.9)

    fig, axes = plt.subplots(
        1, len(layers), figsize=figsize, sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, layer in zip(axes, layers):
        for cond in ("global", "own", "cross"):
            series = agg[layer].get(cond)
            if not series:
                continue
            label, colour, marker = CONDITION_STYLE[cond]
            coeffs = sorted(series)
            means = np.array([series[c]["mean"] for c in coeffs])
            sems = np.array([series[c]["sem"] for c in coeffs])

            ax.plot(coeffs, means, marker=marker, color=colour, label=label,
                    linewidth=1.6, markersize=4, zorder=3)
            if np.any(sems > 0):
                ax.fill_between(coeffs, means - sems, means + sems,
                                color=colour, alpha=0.18, linewidth=0, zorder=2)

        # Zero-effect and zero-coefficient references: every curve should pass
        # through their intersection.
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", zorder=1)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.set_xlabel("Steering coefficient", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(ylabel, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"Steering dose-response: {concept} ({report.get('model', '')})",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if output.suffix.lower() == ".pdf":
        fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote %s (%d layers, %d cells)", output, len(layers), len(cells))
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Plot the steering coefficient/layer sweep.",
    )
    parser.add_argument("--report", required=True,
                        help="Path to steering_sweep_<concept>_<model>.json")
    parser.add_argument("--output", required=True,
                        help="Output figure path (.pdf writes a .png alongside)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    zero = report.get("summary", {}).get("_zero_coeff_control")
    if zero and zero.get("max_abs_effect", 0.0) > 1e-6:
        logger.warning(
            "Zero-coefficient control is %.4g, not ~0 — the steering hook may be "
            "leaking across conditions, which would invalidate the whole sweep.",
            zero["max_abs_effect"],
        )

    plot_sweep(report, Path(args.output))


if __name__ == "__main__":
    main()
