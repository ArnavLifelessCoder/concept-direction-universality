"""
geometry_controls.py — Controls for the Fragmentation Claim
============================================================
Statistical controls that separate *genuine* cross-domain direction
difference from estimation noise and from artifacts of how the global
direction is built.

Three distinct problems this module solves, each corresponding to a
documented threat in ``docs/experimental_design.md``:

**C1 — The global direction contains the domain it is compared against.**
``compute_global_direction`` pools every domain, so a per-domain direction is
correlated with the global direction partly because it *is* one of the
summands. With ``k`` equal-magnitude mutually orthogonal domain directions,
each still has cosine ``1/sqrt(k)`` with their normalized sum (0.5 at k=4) —
close to a "fragmented" reading of the raw number. The fix is
leave-one-domain-out (LODO): compare each domain direction against a global
direction estimated from the *other* domains only. See
:func:`leave_one_domain_out_direction`.

**C2 — Cross-domain cosine has no within-domain reference.**
A raw cross-domain cosine of 0.5 means nothing until we know what two
*independent estimates of the same domain* score. The fix is a matched
split-half protocol: split each domain into disjoint halves, estimate a
direction from each, and compare that within-domain cosine distribution
against a cross-domain distribution computed **at the same sample size**.
Cosine estimation noise is strongly n-dependent, so the sample-size match is
what makes the comparison fair. See :func:`split_half_reliability` and
:func:`matched_cross_domain_cosine`.

**C3 — Early-layer directions may be noise.**
Concepts may not be linearly encoded at all in early layers, in which case a
low cosine reflects the absence of signal rather than fragmentation of a
signal. The fix is a sign-flip permutation null that destroys the concept
contrast while preserving the pairing structure, giving a per-layer test of
whether a direction carries any contrast signal at all. See
:func:`permutation_null` and :func:`direction_effect_size`.

Core functions operate on plain tensors so they are testable without a GPU or
cached activations; ``*_from_cache`` wrappers do the disk loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.analysis.directions import difference_of_means
from src.extraction.cache_utils import load_activations

logger = logging.getLogger(__name__)


# ============================================================
# Small shared helpers
# ============================================================

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors, 0.0 if either is degenerate."""
    a = a.float()
    b = b.float()
    na, nb = torch.norm(a), torch.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((torch.dot(a, b) / (na * nb)).clamp(-1.0, 1.0))


def _disjoint_halves(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``range(n)`` into two disjoint index halves of equal size.

    When ``n`` is odd the leftover index is dropped so both halves have exactly
    ``n // 2`` rows — equal sizes keep the two direction estimates equally noisy.
    """
    perm = rng.permutation(n)
    half = n // 2
    return perm[:half], perm[half : 2 * half]


# ============================================================
# C1 — Leave-one-domain-out global direction
# ============================================================

def leave_one_domain_out_direction(
    pos_by_domain: Dict[str, torch.Tensor],
    neg_by_domain: Dict[str, torch.Tensor],
    held_out: str,
    normalize: bool = True,
) -> torch.Tensor:
    """Global direction estimated from every domain **except** ``held_out``.

    This is the honest universality test: it asks whether a direction built
    from other domains predicts the held-out domain's direction, with no
    shared data inflating the cosine.

    Args:
        pos_by_domain: {domain: (n_d, d_model)} positive-pole activations.
        neg_by_domain: {domain: (n_d, d_model)} negative-pole activations.
        held_out: Domain to exclude from the pool.
        normalize: Return a unit vector.

    Returns:
        Direction of shape (d_model,).

    Raises:
        ValueError: If no domain remains after holding one out.
    """
    others = [d for d in pos_by_domain if d != held_out and d in neg_by_domain]
    if not others:
        raise ValueError(
            f"Leave-one-domain-out needs >=2 domains; only '{held_out}' available."
        )

    pooled_pos = torch.cat([pos_by_domain[d] for d in others], dim=0)
    pooled_neg = torch.cat([neg_by_domain[d] for d in others], dim=0)
    return difference_of_means(pooled_pos, pooled_neg, normalize=normalize)


def lodo_cosines(
    pos_by_domain: Dict[str, torch.Tensor],
    neg_by_domain: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Cosine of each domain's direction to a LODO global built from the rest.

    Returns:
        {domain: cosine}. Compare against the pooled-global cosines reported by
        :func:`~src.analysis.angular_dispersion.compute_angular_dispersion`;
        the LODO values are always the more conservative estimate.
    """
    out: Dict[str, float] = {}
    for domain in pos_by_domain:
        if domain not in neg_by_domain:
            continue
        own = difference_of_means(pos_by_domain[domain], neg_by_domain[domain])
        lodo = leave_one_domain_out_direction(pos_by_domain, neg_by_domain, domain)
        out[domain] = _cos(own, lodo)
    return out


# ============================================================
# C2 — Split-half reliability and matched cross-domain cosine
# ============================================================

def split_half_reliability(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_splits: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    """Within-domain reliability: cosine between two independent half-estimates.

    Repeatedly splits the domain's pairs into two disjoint halves, computes a
    direction from each, and records their cosine. The resulting distribution
    is the **noise ceiling** for this domain at this layer: no cross-domain
    cosine can be expected to exceed it, so a cross-domain value should be read
    relative to it rather than against 1.0.

    The same row indices are used for the positive and negative sides, which
    preserves the pairing (pair *i* contributes its positive and negative to
    the same half).

    Args:
        pos: (n, d_model) positive-pole activations.
        neg: (n, d_model) negative-pole activations, row-aligned with ``pos``.
        n_splits: Number of random splits.
        seed: RNG seed.

    Returns:
        Dict with ``mean``, ``std``, ``ci_low``/``ci_high`` (2.5/97.5 pct),
        ``n_per_half``, and the raw ``cosines`` list.
    """
    n = min(pos.shape[0], neg.shape[0])
    if n < 4:
        return {
            "mean": float("nan"), "std": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "n_per_half": n // 2, "cosines": [],
        }

    rng = np.random.default_rng(seed)
    cosines: List[float] = []
    for _ in range(n_splits):
        ia, ib = _disjoint_halves(n, rng)
        da = difference_of_means(pos[ia], neg[ia])
        db = difference_of_means(pos[ib], neg[ib])
        cosines.append(_cos(da, db))

    arr = np.asarray(cosines, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n_per_half": n // 2,
        "cosines": cosines,
    }


def matched_cross_domain_cosine(
    pos_by_domain: Dict[str, torch.Tensor],
    neg_by_domain: Dict[str, torch.Tensor],
    n_per_estimate: int,
    n_splits: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    """Cross-domain cosines estimated at a *matched* sample size.

    For each unordered domain pair, draws ``n_per_estimate`` pairs from each
    domain without replacement, estimates a direction from each draw, and
    records the cosine. Using the same per-estimate sample size as
    :func:`split_half_reliability` is what makes the within- and cross-domain
    distributions comparable: both are then subject to the same estimation
    noise, so any remaining gap is attributable to domain, not to n.

    Args:
        pos_by_domain / neg_by_domain: {domain: (n_d, d_model)}.
        n_per_estimate: Rows per direction estimate (use the within-domain
            ``n_per_half`` so the comparison is matched).
        n_splits: Draws per domain pair.
        seed: RNG seed.

    Returns:
        Dict with ``per_pair`` ({"a|b": {mean, std, ci_low, ci_high}}),
        the pooled ``mean``/``std``/``ci_low``/``ci_high`` over all pairs, and
        ``n_per_estimate``.
    """
    domains = [d for d in sorted(pos_by_domain) if d in neg_by_domain]
    rng = np.random.default_rng(seed)

    per_pair: Dict[str, Dict[str, float]] = {}
    pooled: List[float] = []

    for i, da in enumerate(domains):
        for db in domains[i + 1 :]:
            na = min(pos_by_domain[da].shape[0], neg_by_domain[da].shape[0])
            nb = min(pos_by_domain[db].shape[0], neg_by_domain[db].shape[0])
            k = min(n_per_estimate, na, nb)
            if k < 2:
                continue

            cosines: List[float] = []
            for _ in range(n_splits):
                ia = rng.choice(na, size=k, replace=False)
                ib = rng.choice(nb, size=k, replace=False)
                dir_a = difference_of_means(pos_by_domain[da][ia], neg_by_domain[da][ia])
                dir_b = difference_of_means(pos_by_domain[db][ib], neg_by_domain[db][ib])
                cosines.append(_cos(dir_a, dir_b))

            arr = np.asarray(cosines, dtype=float)
            per_pair[f"{da}|{db}"] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }
            pooled.extend(cosines)

    if not pooled:
        return {"per_pair": {}, "mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_per_estimate": n_per_estimate}

    parr = np.asarray(pooled, dtype=float)
    return {
        "per_pair": per_pair,
        "mean": float(parr.mean()),
        "std": float(parr.std()),
        "ci_low": float(np.percentile(parr, 2.5)),
        "ci_high": float(np.percentile(parr, 97.5)),
        "n_per_estimate": n_per_estimate,
    }


def fragmentation_index(
    within_mean: float,
    cross_mean: float,
) -> Dict[str, float]:
    """Express a cross-domain cosine relative to the within-domain ceiling.

    Two summaries, both reported because they fail in different ways:

    - ``gap`` = within - cross. The raw, assumption-free quantity. Zero means
      two directions from different domains agree as well as two estimates of
      the same domain, i.e. no detectable fragmentation.
    - ``ratio`` = cross / within. A disattenuation-style normalization, by
      analogy with correcting a correlation for attenuation. It is a
      *heuristic*: cosines are not correlations, so the correction has no exact
      distributional justification and is unstable when ``within`` is near zero
      (it is returned as NaN below 0.05). Read it as an effect-size aid, never
      as the primary evidence.

    Returns:
        {"gap": float, "ratio": float}
    """
    gap = float(within_mean - cross_mean)
    ratio = float(cross_mean / within_mean) if within_mean > 0.05 else float("nan")
    return {"gap": gap, "ratio": ratio}


# ============================================================
# C3 — Permutation null and effect size
# ============================================================

def permutation_null(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_permutations: int = 500,
    seed: int = 42,
) -> Dict[str, object]:
    """Sign-flip null for "does this direction carry any contrast signal?".

    For each permutation, independently flips the positive/negative assignment
    of every pair with probability 0.5 and recomputes the difference-of-means
    norm. Flipping *within* pairs (rather than shuffling rows across the whole
    set) preserves the pairing and the marginal distribution of activations, so
    the only thing destroyed is the systematic direction of the contrast — which
    is exactly the quantity under test.

    The observed ``||mean(pos) - mean(neg)||`` is then compared against this
    null distribution of norms.

    Args:
        pos: (n, d_model) positive-pole activations.
        neg: (n, d_model) negative-pole activations, row-aligned.
        n_permutations: Number of sign-flip draws.
        seed: RNG seed.

    Returns:
        Dict with ``observed_norm``, ``null_mean``, ``null_std``, ``null_p95``,
        ``z`` ((observed - null_mean) / null_std), and ``p_value``, the
        one-sided proportion of null norms >= observed with add-one smoothing
        (so p is never exactly 0 and is bounded below by 1/(n_perm+1)).
    """
    n = min(pos.shape[0], neg.shape[0])
    if n < 2:
        return {"observed_norm": float("nan"), "null_mean": float("nan"),
                "null_std": float("nan"), "null_p95": float("nan"),
                "z": float("nan"), "p_value": float("nan")}

    pos_f = pos[:n].float()
    neg_f = neg[:n].float()
    observed = float(torch.norm(pos_f.mean(dim=0) - neg_f.mean(dim=0)))

    rng = np.random.default_rng(seed)
    diff = pos_f - neg_f  # (n, d_model) — per-pair contrast vectors

    null_norms = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = torch.from_numpy(
            rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        ).unsqueeze(1)
        null_norms[i] = float(torch.norm((diff * signs).mean(dim=0)))

    null_mean = float(null_norms.mean())
    null_std = float(null_norms.std())
    z = float((observed - null_mean) / null_std) if null_std > 0 else float("nan")
    p_value = float((np.sum(null_norms >= observed) + 1) / (n_permutations + 1))

    return {
        "observed_norm": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_p95": float(np.percentile(null_norms, 95)),
        "z": z,
        "p_value": p_value,
    }


def _auc(proj_pos: np.ndarray, proj_neg: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank-sum identity (exact, no thresholding)."""
    n_pos, n_neg = len(proj_pos), len(proj_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    scores = np.concatenate([proj_pos, proj_neg])
    # average ranks so ties do not bias the statistic
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    rank_sum_pos = float(ranks[:n_pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def direction_effect_size(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Separation between the two poles, in complementary forms.

    Normalizing a direction to unit length hides whether there was anything to
    normalize. These statistics restore that information:

    - ``delta_norm``: ``||mean(pos) - mean(neg)||``, the raw separation. Compare
      it against :func:`permutation_null`, not against zero.
    - ``cohens_d``: ``delta_norm`` over the pooled within-condition standard
      deviation (RMS-averaged across dimensions), a scale-free multivariate
      analogue of Cohen's d.
    - ``auc_cv``: **the statistic to report.** Cross-validated ROC AUC for
      separating the poles by projection: the direction is fit on the training
      folds and evaluated only on the held-out fold, with scores pooled across
      folds. 0.5 = no separation, 1.0 = perfect.
    - ``auc_insample``: the same quantity fit and evaluated on all rows.
      Reported only as a diagnostic. When ``d_model`` is comparable to or larger
      than ``n`` — which it always is here — a difference-of-means direction fits
      noise, and this runs far above 0.5 even for pure noise (empirically ~0.86
      at n=120, d=128). Never report it as evidence that a layer carries signal;
      the gap between it and ``auc_cv`` measures the overfitting.

    Args:
        pos / neg: (n, d_model), row-aligned.
        n_folds: Folds for the cross-validated AUC. Reduced automatically when
            there are too few pairs.
        seed: RNG seed for the fold assignment.

    Returns:
        {"delta_norm", "cohens_d", "auc_cv", "auc_insample"}
    """
    nan = float("nan")
    n = min(pos.shape[0], neg.shape[0])
    if n < 2:
        return {"delta_norm": nan, "cohens_d": nan, "auc_cv": nan, "auc_insample": nan}

    pos_f = pos[:n].float()
    neg_f = neg[:n].float()

    delta = pos_f.mean(dim=0) - neg_f.mean(dim=0)
    delta_norm = float(torch.norm(delta))

    # Pooled within-condition spread, RMS-averaged over dimensions.
    var = (pos_f.var(dim=0, unbiased=True) + neg_f.var(dim=0, unbiased=True)) / 2.0
    pooled_sd = float(torch.sqrt(var.mean()))
    cohens_d = delta_norm / pooled_sd if pooled_sd > 0 else nan

    unit = delta / delta_norm if delta_norm > 0 else delta
    auc_insample = _auc((pos_f @ unit).numpy(), (neg_f @ unit).numpy())

    # Cross-validated AUC: fit the direction on train folds only.
    k = int(min(n_folds, n))
    if k < 2:
        return {"delta_norm": delta_norm, "cohens_d": cohens_d,
                "auc_cv": nan, "auc_insample": float(auc_insample)}

    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % k
    fold_aucs: List[float] = []

    for f in range(k):
        test = folds == f
        train = ~test
        if train.sum() < 2 or test.sum() < 1:
            continue
        d_train = difference_of_means(pos_f[train], neg_f[train])
        if float(torch.norm(d_train)) == 0:
            continue
        # AUC is computed within each fold and then averaged: every fold uses a
        # different fitted direction, so raw projections carry different offsets
        # and are not comparable across folds. AUC is rank-based and unit-free,
        # so the per-fold values are.
        fold_auc = _auc(
            (pos_f[test] @ d_train).numpy(),
            (neg_f[test] @ d_train).numpy(),
        )
        if not np.isnan(fold_auc):
            fold_aucs.append(fold_auc)

    auc_cv = float(np.mean(fold_aucs)) if fold_aucs else nan

    return {
        "delta_norm": delta_norm,
        "cohens_d": cohens_d,
        "auc_cv": float(auc_cv),
        "auc_insample": float(auc_insample),
    }


# ============================================================
# Per-layer driver
# ============================================================

@dataclass
class LayerControls:
    """All control statistics for one concept at one layer."""
    layer: int
    lodo_cosine: Dict[str, float] = field(default_factory=dict)
    lodo_mean: float = float("nan")
    within_domain: Dict[str, Dict[str, float]] = field(default_factory=dict)
    within_mean: float = float("nan")
    cross_domain_mean: float = float("nan")
    cross_domain_ci: Tuple[float, float] = (float("nan"), float("nan"))
    cross_domain_per_pair: Dict[str, Dict[str, float]] = field(default_factory=dict)
    gap: float = float("nan")
    ratio: float = float("nan")
    permutation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    effect_size: Dict[str, Dict[str, float]] = field(default_factory=dict)
    min_auc_cv: float = float("nan")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_layer_controls(
    pos_by_domain: Dict[str, torch.Tensor],
    neg_by_domain: Dict[str, torch.Tensor],
    layer: int,
    n_splits: int = 200,
    n_permutations: int = 500,
    seed: int = 42,
) -> LayerControls:
    """Run every control in this module for one layer's activations.

    Args:
        pos_by_domain / neg_by_domain: {domain: (n_d, d_model)}, row-aligned
            within each domain.
        layer: Layer index (recorded in the result).
        n_splits: Splits/draws for the reliability and matched-cross estimates.
        n_permutations: Draws for the sign-flip null.
        seed: RNG seed; derived deterministically per domain.

    Returns:
        A populated :class:`LayerControls`.
    """
    res = LayerControls(layer=layer)
    domains = [d for d in sorted(pos_by_domain) if d in neg_by_domain]
    if not domains:
        return res

    # C1 — leave-one-domain-out
    if len(domains) >= 2:
        res.lodo_cosine = lodo_cosines(pos_by_domain, neg_by_domain)
        if res.lodo_cosine:
            res.lodo_mean = float(np.mean(list(res.lodo_cosine.values())))

    # C2 — within-domain reliability, then cross-domain at the matched n
    half_sizes: List[int] = []
    within_means: List[float] = []
    for di, domain in enumerate(domains):
        rel = split_half_reliability(
            pos_by_domain[domain], neg_by_domain[domain],
            n_splits=n_splits, seed=seed + di,
        )
        res.within_domain[domain] = {
            k: v for k, v in rel.items() if k != "cosines"
        }
        if not np.isnan(rel["mean"]):
            within_means.append(rel["mean"])
            half_sizes.append(int(rel["n_per_half"]))

    if within_means:
        res.within_mean = float(np.mean(within_means))

    if half_sizes and len(domains) >= 2:
        matched = matched_cross_domain_cosine(
            pos_by_domain, neg_by_domain,
            n_per_estimate=int(np.min(half_sizes)),
            n_splits=n_splits, seed=seed,
        )
        res.cross_domain_mean = matched["mean"]
        res.cross_domain_ci = (matched["ci_low"], matched["ci_high"])
        res.cross_domain_per_pair = matched["per_pair"]

        idx = fragmentation_index(res.within_mean, res.cross_domain_mean)
        res.gap, res.ratio = idx["gap"], idx["ratio"]

    # C3 — is there any signal at this layer at all?
    aucs: List[float] = []
    for di, domain in enumerate(domains):
        res.permutation[domain] = permutation_null(
            pos_by_domain[domain], neg_by_domain[domain],
            n_permutations=n_permutations, seed=seed + di,
        )
        eff = direction_effect_size(
            pos_by_domain[domain], neg_by_domain[domain], seed=seed + di,
        )
        res.effect_size[domain] = eff
        if not np.isnan(eff["auc_cv"]):
            aucs.append(eff["auc_cv"])
    if aucs:
        res.min_auc_cv = float(np.min(aucs))

    return res


def run_controls_from_cache(
    activations_dir: Path,
    model_name: str,
    concept: str,
    domains: Sequence[str],
    layers: Sequence[int],
    n_splits: int = 200,
    n_permutations: int = 500,
    seed: int = 42,
    device: str = "cpu",
    indices_by_domain: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[int, Dict[str, object]]:
    """Load cached activations layer by layer and run all controls.

    Loads one layer at a time so peak memory stays at a single layer's
    activations rather than the whole cache.

    Args:
        activations_dir: Directory of cached activation tensors.
        model_name / concept: Cache keys.
        domains: Domains to include.
        layers: Layers to analyze.
        n_splits / n_permutations / seed: Passed to :func:`run_layer_controls`.
        device: Device to load tensors onto.
        indices_by_domain: Optional balanced-subsample indices (see
            :func:`~src.analysis.directions.balanced_subsample_indices`) so the
            controls run on the same balanced sample as the main analysis.

    Returns:
        {layer: LayerControls.to_dict()}.
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
                logger.warning(
                    "Missing activations: %s/%s/%s/layer%d — skipping domain.",
                    model_name, concept, domain, layer,
                )
                continue

            p, n = pos[layer], neg[layer]
            idx = (indices_by_domain or {}).get(domain)
            if idx is not None:
                p, n = p[idx], n[idx]
            pos_by_domain[domain] = p
            neg_by_domain[domain] = n

        if not pos_by_domain:
            continue

        controls = run_layer_controls(
            pos_by_domain, neg_by_domain, layer,
            n_splits=n_splits, n_permutations=n_permutations, seed=seed,
        )
        out[layer] = controls.to_dict()

        logger.info(
            "Layer %d | LODO=%.3f | within=%.3f cross=%.3f gap=%.3f | min CV AUC=%.3f",
            layer, controls.lodo_mean, controls.within_mean,
            controls.cross_domain_mean, controls.gap, controls.min_auc_cv,
        )

    return out
