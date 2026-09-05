"""
test_geometry_controls.py — Tests for the reviewer-response controls
=====================================================================
Validates :mod:`src.analysis.geometry_controls` and
:mod:`src.analysis.subspace` against synthetic data with known ground truth.

The important cases are the ones where a *wrong* implementation would still
produce plausible-looking numbers:

- Orthogonal domain directions must score ~1/sqrt(k) against a pooled global
  direction but ~0 against a leave-one-domain-out global. This is the exact
  artifact Reviewer 1 identified, so it is pinned numerically here.
- A single shared direction plus noise must produce within-domain and
  cross-domain cosines that are close to each other; genuinely different
  directions must produce a gap. Both are checked, because an implementation
  that always reports a gap would "confirm" fragmentation everywhere.
- Pure-noise activations must fail the permutation null and score AUC ~0.5,
  which is the early-layer scenario the reviewer worried about.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.geometry_controls import (
    direction_effect_size,
    fragmentation_index,
    leave_one_domain_out_direction,
    lodo_cosines,
    matched_cross_domain_cosine,
    permutation_null,
    run_layer_controls,
    split_half_reliability,
)
from src.analysis.subspace import (
    domain_subspace,
    lodo_subspace_generalization,
    principal_angles,
    run_subspace_analysis,
    subspace_spectrum,
)

D_MODEL = 128
N_PAIRS = 120


def _make_domain(direction, n=N_PAIRS, noise=1.0, seed=0, d=D_MODEL):
    """Build (pos, neg) activations whose difference-of-means is ``direction``."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, d, generator=g) * noise
    neg = base
    pos = base + direction.unsqueeze(0)
    # Independent noise on each pole, as in real paired extraction.
    pos = pos + torch.randn(n, d, generator=g) * noise * 0.5
    neg = neg + torch.randn(n, d, generator=g) * noise * 0.5
    return pos, neg


def _orthogonal_directions(k, scale=3.0, d=D_MODEL):
    """``k`` mutually orthogonal, equal-magnitude directions."""
    eye = torch.eye(d)
    return {f"d{i}": eye[i] * scale for i in range(k)}


# ============================================================
# C1 — the pooled-global artifact and its LODO fix
# ============================================================

class TestLeaveOneDomainOut:

    def test_orthogonal_domains_hit_one_over_sqrt_k_against_pooled_global(self):
        """Reviewer 1's arithmetic: k orthogonal directions each score 1/sqrt(k).

        This is the artifact that makes a pooled-global cosine of ~0.5 at k=4
        uninterpretable, so it is pinned exactly rather than loosely.
        """
        for k in (2, 3, 4, 5):
            dirs = _orthogonal_directions(k)
            stacked = torch.stack([dirs[f"d{i}"] for i in range(k)], dim=0)
            pooled = stacked.mean(dim=0)
            pooled = pooled / torch.norm(pooled)

            for i in range(k):
                own = dirs[f"d{i}"] / torch.norm(dirs[f"d{i}"])
                cos = float(torch.dot(own, pooled))
                assert cos == pytest.approx(1.0 / np.sqrt(k), abs=1e-5), (
                    f"k={k}: expected {1/np.sqrt(k):.4f}, got {cos:.4f}"
                )

    def test_lodo_removes_the_artifact_for_orthogonal_domains(self):
        """Truly unrelated domains must score ~0 under LODO, not 1/sqrt(k)."""
        dirs = _orthogonal_directions(4, scale=6.0)
        pos_by, neg_by = {}, {}
        for i, (name, vec) in enumerate(dirs.items()):
            pos_by[name], neg_by[name] = _make_domain(vec, noise=0.5, seed=100 + i)

        cos = lodo_cosines(pos_by, neg_by)
        assert set(cos) == set(dirs)
        for name, c in cos.items():
            assert abs(c) < 0.2, f"{name}: LODO cosine {c:.3f} should be near 0"

    def test_lodo_preserves_high_cosine_for_genuinely_shared_direction(self):
        """The control must not manufacture fragmentation where none exists."""
        shared = torch.zeros(D_MODEL)
        shared[0] = 6.0
        pos_by, neg_by = {}, {}
        for i in range(4):
            pos_by[f"d{i}"], neg_by[f"d{i}"] = _make_domain(
                shared, noise=0.5, seed=200 + i,
            )

        cos = lodo_cosines(pos_by, neg_by)
        for name, c in cos.items():
            assert c > 0.9, f"{name}: shared direction should stay high, got {c:.3f}"

    def test_lodo_excludes_the_held_out_domain(self):
        dirs = _orthogonal_directions(3, scale=5.0)
        pos_by, neg_by = {}, {}
        for i, (name, vec) in enumerate(dirs.items()):
            pos_by[name], neg_by[name] = _make_domain(vec, noise=0.3, seed=300 + i)

        lodo = leave_one_domain_out_direction(pos_by, neg_by, "d0")
        own = torch.nn.functional.normalize(
            pos_by["d0"].mean(0) - neg_by["d0"].mean(0), dim=0,
        )
        assert abs(float(torch.dot(lodo, own))) < 0.25

    def test_lodo_raises_with_a_single_domain(self):
        pos, neg = _make_domain(torch.eye(D_MODEL)[0] * 3.0)
        with pytest.raises(ValueError, match="needs >=2 domains"):
            leave_one_domain_out_direction({"only": pos}, {"only": neg}, "only")


# ============================================================
# C2 — within-domain reliability vs matched cross-domain
# ============================================================

class TestSplitHalfReliability:

    def test_strong_signal_gives_high_reliability(self):
        pos, neg = _make_domain(torch.eye(D_MODEL)[0] * 8.0, noise=0.5, seed=1)
        rel = split_half_reliability(pos, neg, n_splits=50, seed=1)
        assert rel["mean"] > 0.9
        assert rel["n_per_half"] == N_PAIRS // 2
        assert rel["ci_low"] <= rel["mean"] <= rel["ci_high"]

    def test_pure_noise_gives_near_zero_reliability(self):
        """No concept signal means two half-estimates agree only by chance."""
        g = torch.Generator().manual_seed(7)
        pos = torch.randn(N_PAIRS, D_MODEL, generator=g)
        neg = torch.randn(N_PAIRS, D_MODEL, generator=g)
        rel = split_half_reliability(pos, neg, n_splits=50, seed=2)
        assert abs(rel["mean"]) < 0.25, (
            f"noise reliability {rel['mean']:.3f} should be near 0"
        )

    def test_halves_are_disjoint_and_equal_sized(self):
        from src.analysis.geometry_controls import _disjoint_halves
        rng = np.random.default_rng(0)
        for n in (10, 11, 50, 51):
            a, b = _disjoint_halves(n, rng)
            assert len(a) == len(b) == n // 2
            assert not (set(a.tolist()) & set(b.tolist()))

    def test_too_few_pairs_returns_nan_not_crash(self):
        pos, neg = _make_domain(torch.eye(D_MODEL)[0], n=3)
        rel = split_half_reliability(pos, neg, n_splits=10)
        assert np.isnan(rel["mean"])


class TestMatchedCrossDomain:

    def test_shared_direction_closes_the_gap(self):
        """Same underlying direction: cross-domain should approach within-domain."""
        shared = torch.zeros(D_MODEL)
        shared[0] = 6.0
        pos_by, neg_by = {}, {}
        for i in range(4):
            pos_by[f"d{i}"], neg_by[f"d{i}"] = _make_domain(
                shared, noise=1.0, seed=400 + i,
            )

        within = np.mean([
            split_half_reliability(pos_by[d], neg_by[d], n_splits=40, seed=5)["mean"]
            for d in pos_by
        ])
        cross = matched_cross_domain_cosine(
            pos_by, neg_by, n_per_estimate=N_PAIRS // 2, n_splits=40, seed=5,
        )
        gap = fragmentation_index(within, cross["mean"])["gap"]
        assert gap < 0.15, f"shared direction should show little gap, got {gap:.3f}"

    def test_distinct_directions_open_a_gap(self):
        """Genuinely different directions must show a gap despite matched n."""
        dirs = _orthogonal_directions(4, scale=6.0)
        pos_by, neg_by = {}, {}
        for i, (name, vec) in enumerate(dirs.items()):
            pos_by[name], neg_by[name] = _make_domain(vec, noise=1.0, seed=500 + i)

        within = np.mean([
            split_half_reliability(pos_by[d], neg_by[d], n_splits=40, seed=6)["mean"]
            for d in pos_by
        ])
        cross = matched_cross_domain_cosine(
            pos_by, neg_by, n_per_estimate=N_PAIRS // 2, n_splits=40, seed=6,
        )
        gap = fragmentation_index(within, cross["mean"])["gap"]
        assert gap > 0.5, f"orthogonal directions should show a large gap, got {gap:.3f}"

    def test_reports_every_unordered_pair(self):
        dirs = _orthogonal_directions(4, scale=4.0)
        pos_by, neg_by = {}, {}
        for i, (name, vec) in enumerate(dirs.items()):
            pos_by[name], neg_by[name] = _make_domain(vec, seed=600 + i)
        cross = matched_cross_domain_cosine(
            pos_by, neg_by, n_per_estimate=30, n_splits=10, seed=7,
        )
        assert len(cross["per_pair"]) == 6  # C(4,2)


class TestFragmentationIndex:

    def test_ratio_is_nan_when_within_is_degenerate(self):
        """A ratio against an unreliable ceiling is meaningless, not merely large."""
        idx = fragmentation_index(within_mean=0.01, cross_mean=0.005)
        assert np.isnan(idx["ratio"])
        assert idx["gap"] == pytest.approx(0.005)

    def test_equal_within_and_cross_gives_ratio_one(self):
        idx = fragmentation_index(0.8, 0.8)
        assert idx["gap"] == pytest.approx(0.0)
        assert idx["ratio"] == pytest.approx(1.0)


# ============================================================
# C3 — permutation null and effect size
# ============================================================

class TestPermutationNull:

    def test_real_signal_beats_the_null(self):
        pos, neg = _make_domain(torch.eye(D_MODEL)[0] * 8.0, noise=1.0, seed=11)
        res = permutation_null(pos, neg, n_permutations=200, seed=11)
        assert res["observed_norm"] > res["null_p95"]
        assert res["z"] > 5
        assert res["p_value"] < 0.01

    def test_pure_noise_does_not_beat_the_null(self):
        """The early-layer scenario: no signal must not look like a direction."""
        g = torch.Generator().manual_seed(12)
        pos = torch.randn(N_PAIRS, D_MODEL, generator=g)
        neg = torch.randn(N_PAIRS, D_MODEL, generator=g)
        res = permutation_null(pos, neg, n_permutations=200, seed=12)
        assert res["p_value"] > 0.05, (
            f"noise should not be significant, p={res['p_value']:.3f}"
        )

    def test_p_value_is_never_zero(self):
        pos, neg = _make_domain(torch.eye(D_MODEL)[0] * 50.0, noise=0.1, seed=13)
        res = permutation_null(pos, neg, n_permutations=100, seed=13)
        assert res["p_value"] >= 1.0 / 101


class TestEffectSize:

    def test_strong_signal_gives_high_auc(self):
        pos, neg = _make_domain(torch.eye(D_MODEL)[0] * 10.0, noise=1.0, seed=21)
        eff = direction_effect_size(pos, neg)
        assert eff["auc_cv"] > 0.95
        assert eff["cohens_d"] > 2.0
        assert eff["delta_norm"] > 0

    def test_cross_validated_auc_is_at_chance_for_pure_noise(self):
        """The early-layer scenario: no signal must score ~0.5 out of sample."""
        g = torch.Generator().manual_seed(22)
        pos = torch.randn(N_PAIRS, D_MODEL, generator=g)
        neg = torch.randn(N_PAIRS, D_MODEL, generator=g)
        eff = direction_effect_size(pos, neg)
        assert 0.35 < eff["auc_cv"] < 0.65, (
            f"CV AUC {eff['auc_cv']:.3f} should be ~0.5"
        )

    def test_in_sample_auc_is_inflated_by_overfitting_on_noise(self):
        """Pins the reason auc_insample must never be reported as evidence.

        With d_model >= n, a difference-of-means direction fits the noise, so the
        in-sample AUC is far above chance even when no concept signal exists.
        """
        g = torch.Generator().manual_seed(22)
        pos = torch.randn(N_PAIRS, D_MODEL, generator=g)
        neg = torch.randn(N_PAIRS, D_MODEL, generator=g)
        eff = direction_effect_size(pos, neg)
        assert eff["auc_insample"] > 0.75
        assert eff["auc_insample"] - eff["auc_cv"] > 0.2

    def test_auc_is_scale_invariant(self):
        """Rescaling activations must not change separability."""
        pos, neg = _make_domain(torch.eye(D_MODEL)[0] * 5.0, noise=1.0, seed=23)
        a = direction_effect_size(pos, neg)["auc_cv"]
        b = direction_effect_size(pos * 10.0, neg * 10.0)["auc_cv"]
        assert a == pytest.approx(b, abs=1e-6)


# ============================================================
# Layer driver
# ============================================================

class TestRunLayerControls:

    def test_populates_every_control(self):
        dirs = _orthogonal_directions(4, scale=6.0)
        pos_by, neg_by = {}, {}
        for i, (name, vec) in enumerate(dirs.items()):
            pos_by[name], neg_by[name] = _make_domain(vec, noise=1.0, seed=700 + i)

        res = run_layer_controls(
            pos_by, neg_by, layer=5, n_splits=20, n_permutations=50, seed=1,
        )
        assert res.layer == 5
        assert len(res.lodo_cosine) == 4
        assert len(res.within_domain) == 4
        assert len(res.permutation) == 4
        assert len(res.effect_size) == 4
        assert not np.isnan(res.gap)
        assert res.to_dict()["layer"] == 5

    def test_fragmented_and_universal_cases_are_distinguished(self):
        """End-to-end: the driver must separate the two regimes it exists to tell apart."""
        # Universal
        shared = torch.zeros(D_MODEL)
        shared[0] = 6.0
        up, un = {}, {}
        for i in range(4):
            up[f"d{i}"], un[f"d{i}"] = _make_domain(shared, noise=1.0, seed=800 + i)
        uni = run_layer_controls(up, un, 0, n_splits=20, n_permutations=30)

        # Fragmented
        dirs = _orthogonal_directions(4, scale=6.0)
        fp, fn = {}, {}
        for i, (name, vec) in enumerate(dirs.items()):
            fp[name], fn[name] = _make_domain(vec, noise=1.0, seed=900 + i)
        frag = run_layer_controls(fp, fn, 0, n_splits=20, n_permutations=30)

        assert uni.lodo_mean > 0.8 > frag.lodo_mean
        assert uni.gap < frag.gap
        # Both have real signal — the difference is geometry, not detectability.
        assert uni.min_auc_cv > 0.8 and frag.min_auc_cv > 0.8


# ============================================================
# Subspace analysis
# ============================================================

class TestSubspaceSpectrum:

    def test_identical_directions_are_rank_one(self):
        v = torch.zeros(D_MODEL)
        v[0] = 1.0
        spec = subspace_spectrum({f"d{i}": v.clone() for i in range(4)})
        assert spec["rank1"] > 0.99
        assert spec["participation_ratio"] == pytest.approx(1.0, abs=0.05)

    def test_orthogonal_directions_have_full_participation_ratio(self):
        spec = subspace_spectrum(_orthogonal_directions(4))
        assert spec["participation_ratio"] == pytest.approx(4.0, abs=0.05)
        assert spec["rank1"] == pytest.approx(0.25, abs=0.02)

    def test_planar_directions_are_captured_at_rank_two(self):
        """Four directions spread within a plane: rank-2 must capture ~all of them."""
        e0, e1 = torch.eye(D_MODEL)[0], torch.eye(D_MODEL)[1]
        dirs = {}
        for i, theta in enumerate([0.0, 0.5, 1.0, 1.5]):
            dirs[f"d{i}"] = np.cos(theta) * e0 + np.sin(theta) * e1
        spec = subspace_spectrum(dirs)
        assert spec["rank2"] > 0.99
        assert spec["participation_ratio"] < 2.5


class TestPrincipalAngles:

    def test_identical_subspaces_have_zero_angle(self):
        basis = torch.eye(D_MODEL)[:2]
        pa = principal_angles(basis, basis.clone())
        assert pa["min_angle_deg"] == pytest.approx(0.0, abs=1e-3)
        assert pa["mean_cos"] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_subspaces_have_ninety_degree_angles(self):
        a = torch.eye(D_MODEL)[:2]
        b = torch.eye(D_MODEL)[2:4]
        pa = principal_angles(a, b)
        assert pa["min_angle_deg"] == pytest.approx(90.0, abs=1e-2)

    def test_domain_subspace_is_orthonormal(self):
        pos, neg = _make_domain(torch.eye(D_MODEL)[0] * 4.0, seed=31)
        basis = domain_subspace(pos, neg, rank=3)
        gram = basis @ basis.T
        assert torch.allclose(gram, torch.eye(3), atol=1e-4)


class TestLodoSubspaceGeneralization:

    def test_planar_directions_generalize_at_rank_two(self):
        """The Burger interpretation: shared plane, divergent directions."""
        e0, e1 = torch.eye(D_MODEL)[0], torch.eye(D_MODEL)[1]
        dirs = {}
        for i, theta in enumerate([0.0, 0.7, 1.4, 2.1]):
            dirs[f"d{i}"] = np.cos(theta) * e0 + np.sin(theta) * e1

        res = lodo_subspace_generalization(dirs, ranks=(1, 2, 3))
        rank2 = res["per_rank"][2]
        assert rank2["mean"] > 0.95, (
            "a held-out direction lying in the shared plane must be captured"
        )
        assert rank2["mean"] > res["per_rank"][1]["mean"]

    def test_orthogonal_directions_do_not_generalize(self):
        """Genuine fragmentation: no subspace from others predicts the held-out one."""
        res = lodo_subspace_generalization(_orthogonal_directions(4), ranks=(1, 2, 3))
        for k, r in res["per_rank"].items():
            assert r["mean"] < 0.15, f"rank {k}: got {r['mean']:.3f}, expected ~0"

    def test_chance_baseline_scales_with_rank(self):
        res = lodo_subspace_generalization(_orthogonal_directions(4), ranks=(1, 2, 3))
        assert res["per_rank"][1]["chance"] == pytest.approx(1 / D_MODEL)
        assert res["per_rank"][2]["chance"] == pytest.approx(2 / D_MODEL)

    def test_rank_above_available_directions_is_skipped(self):
        """With 4 domains only ranks 1-3 are constructible; 4 must not appear."""
        res = lodo_subspace_generalization(_orthogonal_directions(4), ranks=(1, 2, 3, 4))
        assert 4 not in res["per_rank"]

    def test_too_few_domains_returns_note(self):
        res = lodo_subspace_generalization(_orthogonal_directions(2))
        assert res["per_rank"] == {}


class TestRunSubspaceAnalysis:

    def test_returns_all_three_tests(self):
        dirs = _orthogonal_directions(4, scale=5.0)
        pos_by, neg_by = {}, {}
        for i, (name, vec) in enumerate(dirs.items()):
            pos_by[name], neg_by[name] = _make_domain(vec, seed=1000 + i)

        res = run_subspace_analysis(pos_by, neg_by, layer=12)
        assert res["layer"] == 12
        assert "spectrum" in res and "principal_angles" in res and "lodo" in res
        assert len(res["principal_angles"]) == 6


# ============================================================
# Replication driver — reliability gating
# ============================================================

class TestReliabilityGating:
    """The gate exists so noise layers are never reported as fragmented."""

    def _controls(self, within_by_layer):
        return {
            layer: {
                "within_mean": w,
                "lodo_mean": 0.9,
                "gap": 0.05,
                "cross_domain_mean": 0.85,
            }
            for layer, w in within_by_layer.items()
        }

    def test_layers_below_floor_are_gated(self):
        from src.analysis.run_replication import summarize_model_concept
        s = summarize_model_concept(
            self._controls({0: 0.02, 1: 0.05, 2: 0.90, 3: 0.85}), {}, 0.2,
        )
        assert s["gated_layers"] == [0, 1]
        assert s["interpretable_layers"] == [2, 3]
        assert s["n_layers_interpretable"] == 2

    def test_aggregates_exclude_gated_layers(self):
        """A noise layer must not drag the reported cosine down."""
        from src.analysis.run_replication import summarize_model_concept
        controls = self._controls({0: 0.01, 1: 0.9})
        controls[0]["lodo_mean"] = 0.0   # noise layer scores ~0
        controls[1]["lodo_mean"] = 0.95
        s = summarize_model_concept(controls, {}, 0.2)
        assert s["lodo_cosine"]["mean"] == pytest.approx(0.95)

    def test_nan_reliability_is_gated(self):
        from src.analysis.run_replication import summarize_model_concept
        s = summarize_model_concept(self._controls({0: float("nan"), 1: 0.8}), {}, 0.2)
        assert s["gated_layers"] == [0]

    def test_all_layers_gated_yields_nan_not_crash(self):
        from src.analysis.run_replication import summarize_model_concept
        s = summarize_model_concept(self._controls({0: 0.01, 1: 0.02}), {}, 0.2)
        assert s["n_layers_interpretable"] == 0
        assert np.isnan(s["lodo_cosine"]["mean"])

    def test_base_instruct_delta_is_instruct_minus_base(self):
        from src.analysis.run_replication import build_cross_model_table
        def rep(model, instruct, lodo):
            return {"model": model, "concept": "honesty", "is_instruct": instruct,
                    "d_model": 896, "n_layers": 24,
                    "summary": {"n_layers_interpretable": 20,
                                "lodo_cosine": {"mean": lodo, "min": lodo, "max": lodo},
                                "within_domain_reliability": {"mean": 0.8},
                                "fragmentation_gap": {"mean": 0.1},
                                "lodo_subspace_rank2": {"mean": 0.5}}}
        table = build_cross_model_table([
            rep("qwen-2.5-0.5b", False, 0.70),
            rep("qwen-2.5-0.5b-instruct", True, 0.75),
        ])
        assert len(table["base_vs_instruct"]) == 1
        # positive delta => alignment consolidates
        assert table["base_vs_instruct"][0]["delta"] == pytest.approx(0.05)


# ============================================================
# Steering sweep aggregation
# ============================================================

class TestSweepAggregation:

    def _cells(self):
        cells = []
        for layer in (12, 18):
            for coeff in (0.0, 4.0):
                for dom in ("a", "b"):
                    for cond in ("own", "global", "cross:a", "cross:b"):
                        cells.append({
                            "layer": layer, "coeff": coeff, "domain": dom,
                            "condition": cond, "metric": "refusal_rate",
                            "effect": 0.0 if coeff == 0 else -0.2,
                        })
        return cells

    def test_cross_conditions_collapse_to_one_series(self):
        """cross:a and cross:b are one question, not two."""
        from src.visualization.steering_sweep_plot import aggregate_cells
        agg = aggregate_cells(self._cells())
        assert sorted(agg[18]) == ["cross", "global", "own"]
        # both cross sources, both domains -> 4 observations
        assert agg[18]["cross"][4.0]["n"] == 4

    def test_zero_coefficient_aggregates_to_zero(self):
        from src.visualization.steering_sweep_plot import aggregate_cells
        agg = aggregate_cells(self._cells())
        assert agg[12]["own"][0.0]["mean"] == pytest.approx(0.0)

    def test_nan_effects_are_dropped_not_propagated(self):
        """One failed cell must not blank out the whole series."""
        from src.visualization.steering_sweep_plot import aggregate_cells
        cells = self._cells()
        cells.append({"layer": 18, "coeff": 4.0, "domain": "c",
                      "condition": "own", "effect": float("nan")})
        agg = aggregate_cells(cells)
        assert not np.isnan(agg[18]["own"][4.0]["mean"])

    def test_single_observation_has_zero_sem(self):
        from src.visualization.steering_sweep_plot import aggregate_cells
        agg = aggregate_cells([{"layer": 1, "coeff": 2.0, "domain": "a",
                                "condition": "own", "effect": -0.3}])
        assert agg[1]["own"][2.0]["sem"] == 0.0
        assert agg[1]["own"][2.0]["n"] == 1
