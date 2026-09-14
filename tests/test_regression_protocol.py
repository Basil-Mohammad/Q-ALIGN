import numpy as np
import pytest
from src.statistics.regression import (
    fit_additive, fit_conjunctive_log_linear, epsilon_sensitivity_sweep, PerformanceSeries
)


def _synthetic_calibration_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    a_spec = rng.uniform(0.01, 1.0, size=n)
    a_top = rng.uniform(0.01, 1.0, size=n)
    # ground-truth relationship performance ~ a_spec^0.6 * a_top^0.4 + noise
    perf = (a_spec ** 0.6) * (a_top ** 0.4) + rng.normal(scale=0.02, size=n)
    perf = np.clip(perf, 1e-6, None)
    perf_series = PerformanceSeries(circuit_ids=np.array([f"c{i}" for i in range(n)]),
                                     values=perf, higher_is_better=True, raw_metric_name="synthetic")
    return a_spec, a_top, perf_series


def test_fit_additive_returns_nonnegative_coefficients():
    a_spec, a_top, perf = _synthetic_calibration_data()
    alpha, beta = fit_additive(a_spec, a_top, perf)
    assert alpha >= 0
    assert beta >= 0


def test_fit_conjunctive_requires_explicit_epsilon():
    a_spec, a_top, perf = _synthetic_calibration_data()
    with pytest.raises(ValueError):
        fit_conjunctive_log_linear(a_spec, a_top, perf, epsilon=None)
    with pytest.raises(ValueError):
        fit_conjunctive_log_linear(a_spec, a_top, perf, epsilon=0.0)


def test_fit_conjunctive_recovers_approximately_correct_exponents():
    a_spec, a_top, perf = _synthetic_calibration_data(n=300)
    alpha, beta, shift = fit_conjunctive_log_linear(a_spec, a_top, perf, epsilon=1e-4)
    # ground truth exponents were 0.6 and 0.4; allow generous tolerance given noise
    assert 0.3 < alpha < 0.9
    assert 0.1 < beta < 0.7


def test_epsilon_sensitivity_sweep_reports_without_selecting():
    a_spec, a_top, perf = _synthetic_calibration_data()
    grid = np.array([1e-5, 1e-4, 1e-3, 1e-2])
    results = epsilon_sensitivity_sweep(a_spec, a_top, perf, grid)
    assert len(results) == len(grid)
    for r in results:
        assert "alpha_times" in r and "beta_times" in r
        assert r["epsilon"] in grid.tolist()


def test_fit_handles_near_zero_a_spec_without_crashing():
    """Regression test for the Q1-quartile near-zero instability the
    manuscript flags (Remark 3.18a): calibration data deliberately includes
    values at exactly 0.0 (as the central-experiment pool's lowest quartile
    would), and the fit must not raise or produce NaN/inf."""
    n = 40
    rng = np.random.default_rng(1)
    a_spec = np.concatenate([np.zeros(10), rng.uniform(0.01, 1.0, size=30)])
    a_top = rng.uniform(0.01, 1.0, size=n)
    perf = rng.uniform(0.01, 1.0, size=n)
    perf_series = PerformanceSeries(circuit_ids=np.array([f"c{i}" for i in range(n)]),
                                     values=perf, higher_is_better=True, raw_metric_name="synthetic")
    alpha, beta, shift = fit_conjunctive_log_linear(a_spec, a_top, perf_series, epsilon=1e-3)
    assert np.isfinite(alpha)
    assert np.isfinite(beta)
