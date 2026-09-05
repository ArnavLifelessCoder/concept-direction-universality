"""
subspace.py — Shared-Subspace Analysis of Concept Directions
=============================================================
Tests the alternative interpretation of low cross-domain cosine that
Burger et al. (2024, *Truth is Universal*) motivate: a concept may be carried
by a shared low-dimensional **subspace** rather than by a single shared
direction. Under that account, per-domain directions can be far apart in angle
while still lying in a common plane, and a rank-1 analysis would misreport
that shared structure as fragmentation.

Three tests, in increasing order of stringency:

1. :func:`subspace_spectrum` — stack the per-domain directions and take their
   singular values. If ``k`` domain directions are largely spanned by 2
   components, the geometry is a shared plane, not ``k`` unrelated axes.
   This is descriptive and uses all domains, so it cannot fail cleanly.

2. :func:`principal_angles` — compare the *within-domain* subspaces (top
   components of each domain's per-pair contrast vectors) between domains.
   Small principal angles mean the domains excite overlapping subspaces even
   where their mean directions diverge.

3. :func:`lodo_subspace_generalization` — the honest test, and the one to
   report. Build a rank-``k`` subspace from the other domains only, then
   measure how much of the held-out domain's direction lies inside it. This
   asks whether shared structure *predicts* an unseen domain, which
   descriptive spectra do not, and it is the subspace analogue of the
   leave-one-domain-out cosine in
   :mod:`~src.analysis.geometry_controls`.

A rank-``k`` subspace is trivially better at capturing a held-out direction
than a rank-1 one, so :func:`lodo_subspace_generalization` also reports the
chance baseline for a random ``k``-dimensional subspace in ``d`` dimensions
(expected squared projection ``k/d``). Compare against that, not against zero.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.analysis.directions import difference_of_means
from src.extraction.cache_utils import load_activations

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def _stack_unit(directions: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor]:
    """Stack a {domain: direction} dict into a unit-normalized (k, d) matrix."""
    names = sorted(directions)
    rows = []
    for n in names:
        v = directions[n].float()
        nv = torch.norm(v)
        rows.append(v / nv if nv > 0 else v)
    return names, torch.stack(rows, dim=0)


def _orthonormal_basis(mat: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-``rank`` right-singular vectors of ``mat`` as an orthonormal (rank, d) basis."""
    # full_matrices=False keeps this cheap for d_model >> n_rows.
    _, _, vh = torch.linalg.svd(mat.float(), full_matrices=False)
    return vh[:rank]


# ============================================================
# 1. Spectrum of the per-domain directions
# ============================================================

def subspace_spectrum(directions: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Singular spectrum of the stacked per-domain directions.

    Args:
        directions: {domain: (d_model,)} per-domain concept directions.

    Returns:
        Dict with:
          - ``singular_values``: descending list.
          - ``variance_explained``: fraction of squared spectral mass per
            component.
          - ``cumulative``: cumulative variance explained.
          - ``rank1`` / ``rank2``: cumulative fraction captured at rank 1 and 2.
            ``rank2`` is the number Burger et al. motivate.
          - ``participation_ratio``: ``(sum s^2)^2 / sum s^4``, an effective
            dimensionality that does not require choosing a cutoff. It runs
            from 1 (all domains share one direction) to ``k`` (mutually
            orthogonal).
          - ``n_domains``.
    """
    names, mat = _stack_unit(directions)
    if mat.shape[0] < 2:
        return {"singular_values": [], "variance_explained": [], "cumulative": [],
                "rank1": float("nan"), "rank2": float("nan"),
                "participation_ratio": float("nan"), "n_domains": mat.shape[0]}

    s = torch.linalg.svdvals(mat.float())
    s2 = (s ** 2)
    total = float(s2.sum())
    if total <= 0:
        return {"singular_values": [], "variance_explained": [], "cumulative": [],
                "rank1": float("nan"), "rank2": float("nan"),
                "participation_ratio": float("nan"), "n_domains": mat.shape[0]}

    var = (s2 / total).tolist()
    cum = np.cumsum(var).tolist()
    pr = float((s2.sum() ** 2) / (s2 ** 2).sum())

    return {
        "domains": names,
        "singular_values": s.tolist(),
        "variance_explained": var,
        "cumulative": cum,
        "rank1": float(cum[0]),
        "rank2": float(cum[1]) if len(cum) > 1 else float("nan"),
        "participation_ratio": pr,
        "n_domains": int(mat.shape[0]),
    }


# ============================================================
# 2. Principal angles between within-domain subspaces
# ============================================================

def domain_subspace(
    pos: torch.Tensor,
    neg: torch.Tensor,
    rank: int = 2,
    center: bool = True,
) -> torch.Tensor:
    """Top-``rank`` subspace of a domain's per-pair contrast vectors.

    Uses the per-pair differences ``pos - neg`` rather than raw activations, so
    the subspace describes how the concept contrast varies within the domain
    rather than where the domain's text happens to live in activation space.

    Args:
        pos / neg: (n, d_model), row-aligned.
        rank: Subspace dimension.
        center: Subtract the mean contrast first. With centering the subspace
            captures *variation around* the domain direction; without it, the
            domain direction itself dominates the first component. Centering is
            the right default when asking whether domains share structure
            beyond their mean directions.

    Returns:
        Orthonormal basis of shape (rank, d_model).
    """
    n = min(pos.shape[0], neg.shape[0])
    diff = (pos[:n].float() - neg[:n].float())
    if center:
        diff = diff - diff.mean(dim=0, keepdim=True)
    rank = int(min(rank, diff.shape[0], diff.shape[1]))
    return _orthonormal_basis(diff, rank)


def principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor) -> Dict[str, object]:
    """Principal angles between two orthonormal subspaces.

    Args:
        basis_a: (k_a, d) orthonormal rows.
        basis_b: (k_b, d) orthonormal rows.

    Returns:
        Dict with ``cosines`` (descending principal cosines), ``angles_deg``,
        ``mean_cos``, and ``min_angle_deg`` (the closest shared direction — if
        this is small, the two subspaces share at least one axis even when
        their mean directions differ).
    """
    s = torch.linalg.svdvals(basis_a.float() @ basis_b.float().T)
    cos = s.clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cos))
    return {
        "cosines": cos.tolist(),
        "angles_deg": angles.tolist(),
        "mean_cos": float(cos.mean()),
        "min_angle_deg": float(angles.min()),
    }


def pairwise_principal_angles(
    pos_by_domain: Dict[str, torch.Tensor],
    neg_by_domain: Dict[str, torch.Tensor],
    rank: int = 2,
    center: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Principal angles between every pair of domain subspaces.

    Returns:
        {"a|b": principal-angle dict}.
    """
    domains = [d for d in sorted(pos_by_domain) if d in neg_by_domain]
    bases = {
        d: domain_subspace(pos_by_domain[d], neg_by_domain[d], rank=rank, center=center)
        for d in domains
    }
    out: Dict[str, Dict[str, object]] = {}
    for i, da in enumerate(domains):
        for db in domains[i + 1 :]:
            out[f"{da}|{db}"] = principal_angles(bases[da], bases[db])
    return out


# ============================================================
# 3. Leave-one-domain-out subspace generalization
# ============================================================

def lodo_subspace_generalization(
    directions: Dict[str, torch.Tensor],
    ranks: Sequence[int] = (1, 2, 3),
    d_model: Optional[int] = None,
) -> Dict[str, object]:
    """Does a subspace built from other domains contain the held-out direction?

    For each domain and each rank ``k``: build a rank-``k`` basis from the
    *other* domains' directions, project the held-out direction onto it, and
    record the squared projection (the fraction of the held-out direction's
    energy captured, in [0, 1]).

    Because rank ``k`` can only be built from ``k_others = n_domains - 1``
    directions, ranks above that are skipped rather than silently truncated.

    Args:
        directions: {domain: (d_model,)}.
        ranks: Subspace ranks to test.
        d_model: Ambient dimension, for the chance baseline. Inferred from the
            directions when omitted.

    Returns:
        Dict with ``per_rank``: {rank: {"per_domain": {...}, "mean": float,
        "chance": float, "above_chance": float}}. ``chance`` is ``k/d``, the
        expected squared projection of a random unit vector onto a random
        rank-``k`` subspace; ``above_chance`` is ``mean - chance``.
    """
    names, mat = _stack_unit(directions)
    n_dom = mat.shape[0]
    if n_dom < 3:
        return {"per_rank": {}, "note": "needs >=3 domains for a LODO subspace"}

    d = int(d_model or mat.shape[1])
    per_rank: Dict[int, Dict[str, object]] = {}

    for k in ranks:
        k = int(k)
        if k < 1 or k > n_dom - 1:
            continue

        per_domain: Dict[str, float] = {}
        for i, held in enumerate(names):
            others = torch.cat([mat[:i], mat[i + 1 :]], dim=0)
            basis = _orthonormal_basis(others, k)          # (k, d)
            v = mat[i]                                      # unit norm
            proj_sq = float((basis @ v).pow(2).sum().clamp(0.0, 1.0))
            per_domain[held] = proj_sq

        mean = float(np.mean(list(per_domain.values())))
        chance = k / d
        per_rank[k] = {
            "per_domain": per_domain,
            "mean": mean,
            "chance": chance,
            "above_chance": mean - chance,
        }

    return {"per_rank": per_rank, "domains": names, "d_model": d}


# ============================================================
# Per-layer driver
# ============================================================

def run_subspace_analysis(
    pos_by_domain: Dict[str, torch.Tensor],
    neg_by_domain: Dict[str, torch.Tensor],
    layer: int,
    ranks: Sequence[int] = (1, 2, 3),
    within_rank: int = 2,
) -> Dict[str, object]:
    """Run all three subspace tests for one layer.

    Args:
        pos_by_domain / neg_by_domain: {domain: (n_d, d_model)}, row-aligned.
        layer: Layer index (recorded in the result).
        ranks: Ranks for the LODO subspace test.
        within_rank: Rank for the within-domain subspaces compared by
            principal angles.

    Returns:
        Dict with ``layer``, ``spectrum``, ``principal_angles``, ``lodo``.
    """
    domains = [d for d in sorted(pos_by_domain) if d in neg_by_domain]
    directions = {
        d: difference_of_means(pos_by_domain[d], neg_by_domain[d]) for d in domains
    }

    return {
        "layer": layer,
        "spectrum": subspace_spectrum(directions),
        "principal_angles": pairwise_principal_angles(
            pos_by_domain, neg_by_domain, rank=within_rank,
        ),
        "lodo": lodo_subspace_generalization(directions, ranks=ranks),
    }


def run_subspace_from_cache(
    activations_dir: Path,
    model_name: str,
    concept: str,
    domains: Sequence[str],
    layers: Sequence[int],
    ranks: Sequence[int] = (1, 2, 3),
    within_rank: int = 2,
    device: str = "cpu",
    indices_by_domain: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[int, Dict[str, object]]:
    """Load cached activations layer by layer and run the subspace analysis.

    Returns:
        {layer: result dict from :func:`run_subspace_analysis`}.
    """
    out: Dict[int, Dict[str, object]] = {}

    for layer in layers:
        pos_by_domain: Dict[str, torch.Tensor] = {}
        neg_by_domain: Dict[str, torch.Tensor] = {}

        for domain in domains:
            pos = load_activations(
                activations_dir, model_name, concept, domain,
                layers=[layer], prefix="pos_", device=device,
            )
            neg = load_activations(
                activations_dir, model_name, concept, domain,
                layers=[layer], prefix="neg_", device=device,
            )
            if layer not in pos or layer not in neg:
                continue
            p, n = pos[layer], neg[layer]
            idx = (indices_by_domain or {}).get(domain)
            if idx is not None:
                p, n = p[idx], n[idx]
            pos_by_domain[domain] = p
            neg_by_domain[domain] = n

        if len(pos_by_domain) < 2:
            continue

        res = run_subspace_analysis(
            pos_by_domain, neg_by_domain, layer,
            ranks=ranks, within_rank=within_rank,
        )
        out[layer] = res

        spec = res["spectrum"]
        lodo2 = res["lodo"].get("per_rank", {}).get(2, {})
        logger.info(
            "Layer %d | rank-1 %.3f rank-2 %.3f (PR %.2f) | LODO rank-2 %.3f (chance %.4f)",
            layer, spec.get("rank1", float("nan")), spec.get("rank2", float("nan")),
            spec.get("participation_ratio", float("nan")),
            lodo2.get("mean", float("nan")), lodo2.get("chance", float("nan")),
        )

    return out
