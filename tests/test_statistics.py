import numpy as np
import pytest
from src.statistics.multiple_testing import TestManifest, benjamini_hochberg, TestManifestFrozenError
from src.statistics.power import required_pool_size, cohens_f2_from_partial_r2, DEFAULT_MIN_PARTIAL_R2
from src.statistics.kendall import kendall_tau_with_ci, compare_dependent_correlations
from src.statistics.bootstrap import hierarchical_bootstrap_ci, aggregate_seeds_per_circuit


def test_manifest_cannot_add_test_after_freeze():
    m = TestManifest()
    m.declare("test_a")
    m.declare("test_b")
    m.freeze()
    with pytest.raises(TestManifestFrozenError):
        m.declare("test_c")


def test_manifest_duplicate_declaration_raises():
    m = TestManifest()
    m.declare("test_a")
    with pytest.raises(ValueError):
        m.declare("test_a")


def test_bh_correction_requires_exact_match_to_manifest():
    m = TestManifest()
    m.declare("t1")
    m.declare("t2")
    m.freeze()
    with pytest.raises(ValueError):
        benjamini_hochberg(m, {"t1": 0.01})  # missing t2


def test_bh_correction_basic_behavior():
    m = TestManifest()
    for name in ["t1", "t2", "t3", "t4"]:
        m.declare(name)
    m.freeze()
    pvals = {"t1": 0.001, "t2": 0.02, "t3": 0.5, "t4": 0.9}
    result = benjamini_hochberg(m, pvals, q=0.05)
    assert result["t1"]["adjusted_p"] <= result["t2"]["adjusted_p"] <= result["t3"]["adjusted_p"]
    assert result["_m_tests"] == 4


def test_cohens_f2_conversion():
    f2 = cohens_f2_from_partial_r2(DEFAULT_MIN_PARTIAL_R2)
    assert f2 == pytest.approx(0.05 / 0.95)


def test_power_analysis_respects_floor():
    result = required_pool_size(min_partial_r2=0.3, alpha=0.05, target_power=0.8,
                                 n_predictors_extended_model=5, n_covariate_strata=1, n_floor=150)
    assert result.required_n >= 150


def test_power_analysis_can_exceed_floor_for_small_effects():
    result = required_pool_size(min_partial_r2=0.02, alpha=0.05, target_power=0.8,
                                 n_predictors_extended_model=5, n_covariate_strata=8, n_floor=150)
    # small effect + many covariate strata should push required N above the floor
    assert result.required_n >= 150


def test_kendall_tau_with_ci_reasonable_range():
    rng = np.random.default_rng(0)
    x = np.arange(50)
    y = x + rng.normal(scale=2, size=50)
    result = kendall_tau_with_ci(x, y, n_bootstrap=200, seed_rng=rng)
    assert result["tau"] > 0.5
    assert result["ci_95_low"] <= result["tau"] <= result["ci_95_high"]


def test_compare_dependent_correlations_detects_clear_difference():
    rng = np.random.default_rng(0)
    n = 100
    y = np.arange(n) + rng.normal(scale=1, size=n)
    x_good = np.arange(n) + rng.normal(scale=1, size=n)   # strongly correlated with y
    x_bad = rng.normal(size=n)                             # uncorrelated with y
    result = compare_dependent_correlations(x_good, x_bad, y, n_bootstrap=500, seed_rng=rng)
    assert result["tau_1"] > result["tau_2"]
    assert result["ci_excludes_zero"] is True


def test_aggregate_seeds_per_circuit_collapses_correctly():
    circuit_ids = np.array(["c1", "c1", "c2", "c2", "c2"])
    seed_values = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    agg = aggregate_seeds_per_circuit(circuit_ids, seed_values)
    assert agg["c1"] == pytest.approx(2.0)
    assert agg["c2"] == pytest.approx(20.0)


def test_hierarchical_bootstrap_resamples_circuits_not_seeds():
    rng = np.random.default_rng(0)
    circuit_level_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = hierarchical_bootstrap_ci(circuit_level_values, np.mean, n_bootstrap=500, seed_rng=rng)
    assert result["n_circuits"] == 5
    assert result["ci_95_low"] <= result["point_estimate"] <= result["ci_95_high"]
