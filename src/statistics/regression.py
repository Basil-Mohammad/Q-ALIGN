"""
Regression protocol (manuscript Sec 6.1, Remarks 3.17-3.19).

Enforces three requirements that were identified as critical failure modes
in manuscript review and MUST hold in code, not just in prose:

1. Performance sign convention (Remark 3.18b): higher = better, always.
   MSE-based outcomes must be transformed via -log(MSE) before reaching
   ANY function in this module. This module asserts a "higher is better"
   marker is set and refuses silently-ambiguous input.
2. Calibration/evaluation fold separation (Remark 3.19): weight fitting
   (`fit_additive`, `fit_conjunctive`) and incremental-R^2 evaluation
   (`incremental_r2`) take DISJOINT data by construction -- the API shape
   makes it structurally awkward to accidentally reuse the same rows.
3. Epsilon-shift for A_x's log-linear fit is a REQUIRED, pre-registered,
   non-optional argument with no default -- callers cannot silently skip it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression

from src.alignment.aggregation import AggregationWeights, epsilon_shifted_log
from src.utils.provenance import assert_disjoint_splits, LeakageError


@dataclass(frozen=True)
class PerformanceSeries:
    """Wraps a performance array with an explicit sign-convention marker
    (Remark 3.18b). `higher_is_better` MUST be True; this dataclass exists
    so that every downstream function can assert it rather than assume it.
    """
    circuit_ids: np.ndarray
    values: np.ndarray
    higher_is_better: bool
    raw_metric_name: str  # e.g. "accuracy", "return", "neg_log_mse"

    def __post_init__(self):
        if not self.higher_is_better:
            raise ValueError(
                f"PerformanceSeries for '{self.raw_metric_name}' has higher_is_better=False. "
                f"Manuscript Remark 3.18b requires ALL performance series entering a regression "
                f"to already be transformed to a higher-is-better convention (e.g. -log(MSE)) "
                f"BEFORE constructing this object. Refusing to proceed with an untransformed series."
            )
        if len(self.circuit_ids) != len(self.values):
            raise ValueError("circuit_ids and values length mismatch.")


def transform_mse_to_higher_is_better(mse: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """The manuscript's prescribed transform: -log(MSE). `floor` guards
    against log(0) for a perfect (MSE=0) fit; fixed, not tuned.
    """
    mse = np.clip(mse, floor, None)
    return -np.log(mse)


def fit_additive(calibration_a_spec: np.ndarray, calibration_a_top: np.ndarray,
                  calibration_performance: PerformanceSeries) -> AggregationWeights:
    """Fit alpha_plus, beta_plus via OLS on the CALIBRATION fold only
    (Remark 3.17: A_+'s native fitting procedure is ordinary least squares,
    since alpha, beta enter as linear coefficients).
    """
    X = np.column_stack([calibration_a_spec, calibration_a_top])
    y = calibration_performance.values
    reg = LinearRegression(positive=True).fit(X, y)
    alpha, beta = reg.coef_
    return alpha, beta  # caller composes full AggregationWeights with gamma/delta fit separately


def fit_conjunctive_log_linear(calibration_a_spec: np.ndarray, calibration_a_top: np.ndarray,
                                calibration_performance: PerformanceSeries,
                                epsilon: float) -> tuple:
    """Fit alpha_times, beta_times via log-linear (Cobb-Douglas) OLS on the
    CALIBRATION fold only (Remark 3.17: A_x's native fitting procedure).

    epsilon is REQUIRED (no default) and must come from a frozen config
    (manuscript Remark 3.18a) -- never selected by trying several values
    and keeping the one that fits best.
    """
    if epsilon is None or epsilon <= 0:
        raise ValueError(
            "epsilon must be a fixed, pre-registered positive value (Remark 3.18a). "
            "Read it from configs/statistics/epsilon.yaml; do not pass a default here."
        )
    log_a_spec = np.array([epsilon_shifted_log(a, epsilon) for a in calibration_a_spec])
    log_a_top = np.array([epsilon_shifted_log(a, epsilon) for a in calibration_a_top])
    # performance must be strictly positive for log(); PerformanceSeries
    # values may be e.g. -log(MSE) which can be negative -- Cobb-Douglas
    # requires positivity, so we shift performance too if needed and record it.
    perf = calibration_performance.values
    perf_shift = 0.0
    if np.min(perf) <= 0:
        perf_shift = -np.min(perf) + 1e-6
    log_perf = np.log(perf + perf_shift)
    X = np.column_stack([log_a_spec, log_a_top])
    reg = LinearRegression(positive=True).fit(X, log_perf)
    alpha_times, beta_times = reg.coef_
    # Guard against degenerate zero exponents (AggregationWeights requires > 0).
    alpha_times = max(alpha_times, 1e-6)
    beta_times = max(beta_times, 1e-6)
    return alpha_times, beta_times, perf_shift


def epsilon_sensitivity_sweep(calibration_a_spec: np.ndarray, calibration_a_top: np.ndarray,
                               calibration_performance: PerformanceSeries,
                               epsilon_grid: np.ndarray) -> list:
    """Reports (does NOT select) fitted (alpha,beta) across a pre-specified
    epsilon grid (Remark 3.18a: 'report a sensitivity analysis ... at least
    an order-of-magnitude range'). This function never feeds its own output
    back into which epsilon is used downstream -- that would reintroduce
    the tuning-leakage this whole mechanism exists to prevent.
    """
    results = []
    for eps in epsilon_grid:
        alpha, beta, shift = fit_conjunctive_log_linear(
            calibration_a_spec, calibration_a_top, calibration_performance, epsilon=eps
        )
        results.append({"epsilon": float(eps), "alpha_times": alpha, "beta_times": beta, "perf_shift": shift})
    return results


def incremental_r2(evaluation_complexity_features: np.ndarray,
                    evaluation_a_scores: np.ndarray,
                    evaluation_performance: PerformanceSeries,
                    calibration_ids: np.ndarray, evaluation_ids: np.ndarray) -> dict:
    """Delta R^2 = R^2(complexity + A) - R^2(complexity only), computed
    STRICTLY on the evaluation fold (Remark 3.19). Raises LeakageError if
    calibration_ids and evaluation_ids are not disjoint -- this check is
    NOT optional and cannot be bypassed by the caller.
    """
    assert_disjoint_splits(calibration_ids.tolist(), evaluation_ids.tolist())

    y = evaluation_performance.values
    baseline = LinearRegression().fit(evaluation_complexity_features, y)
    r2_baseline = baseline.score(evaluation_complexity_features, y)

    extended_X = np.column_stack([evaluation_complexity_features, evaluation_a_scores])
    extended = LinearRegression().fit(extended_X, y)
    r2_extended = extended.score(extended_X, y)

    delta_r2 = r2_extended - r2_baseline

    # Nested-model F-test
    n = len(y)
    p_base = evaluation_complexity_features.shape[1]
    p_ext = extended_X.shape[1]
    rss_base = np.sum((y - baseline.predict(evaluation_complexity_features)) ** 2)
    rss_ext = np.sum((y - extended.predict(extended_X)) ** 2)
    df1 = p_ext - p_base
    df2 = n - p_ext - 1
    f_stat = None
    p_value = None
    if df2 > 0 and rss_ext > 0:
        f_stat = ((rss_base - rss_ext) / df1) / (rss_ext / df2)
        from scipy import stats as sp_stats
        p_value = float(1 - sp_stats.f.cdf(f_stat, df1, df2))

    return {
        "n_evaluation": n,
        "r2_baseline": float(r2_baseline),
        "r2_extended": float(r2_extended),
        "delta_r2": float(delta_r2),
        "f_statistic": float(f_stat) if f_stat is not None else None,
        "p_value_raw": p_value,
        "calibration_fold_size": len(calibration_ids),
        "evaluation_fold_size": len(evaluation_ids),
    }
