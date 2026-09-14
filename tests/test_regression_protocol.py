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


def test_zero_variance_predictor_raises_instead_of_returning_unstable_number():
    """Regression test for a REAL bug found in practice (Checkpoint 8,
    two different machines): when A_top is exactly constant across the
    calibration fold (e.g. saturated at 1.0 for every circuit at small
    scale), the design matrix (constant column + intercept) is singular,
    and an unregularized/NNLS-constrained least-squares fit returns an
    ARBITRARY, platform-dependent coefficient for the degenerate predictor
    -- observed values for the same nominal computation included 0, 14.3,
    and 686.9 across different runs/machines. The fix is to detect
    near-zero-variance predictors before fitting and raise, rather than
    silently return a number that happens to come out of whichever
    floating-point path the local BLAS/LAPACK/sklearn version takes.
    """
    n = 20
    rng = np.random.default_rng(2)
    a_spec_varying = rng.uniform(0.1, 0.9, n)
    a_top_constant = np.full(n, 1.0)  # exactly the real Checkpoint 8 scenario
    perf = rng.normal(size=n)
    perf_series = PerformanceSeries(circuit_ids=np.array([f"c{i}" for i in range(n)]),
                                     values=perf, higher_is_better=True, raw_metric_name="synthetic")

    with pytest.raises(ValueError, match="zero variance"):
        fit_additive(a_spec_varying, a_top_constant, perf_series)

    with pytest.raises(ValueError, match="zero variance"):
        fit_conjunctive_log_linear(a_spec_varying, a_top_constant, perf_series, epsilon=1e-3)

    # Symmetric check: a constant A_spec must also be rejected.
    a_spec_constant = np.full(n, 0.5)
    a_top_varying = rng.uniform(0.1, 0.9, n)
    with pytest.raises(ValueError, match="zero variance"):
        fit_additive(a_spec_constant, a_top_varying, perf_series)
