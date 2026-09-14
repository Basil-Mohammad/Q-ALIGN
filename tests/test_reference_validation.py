"""
Checkpoint 11 -- Statistical analysis validation (manuscript brief Sec 26 /
Phase 11 of the execution order).

Cross-validates our statistical implementations against TRUSTED EXTERNAL
REFERENCES (statsmodels, scipy), not just internal shape/bounds checks.
This distinction matters: the power-analysis bug (Cohen's f vs f^2,
reversed statsmodels df_num/df_denom semantics) passed every internal test
we had, because those tests only checked qualitative properties (floor
respected, larger effect -> larger N) rather than comparing against a
known-correct numeric answer. These tests close that gap for the
remaining statistical modules.
"""
import numpy as np
import pytest
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm

from src.statistics.multiple_testing import TestManifest, benjamini_hochberg
from src.statistics.regression import incremental_r2, PerformanceSeries
from src.statistics.bootstrap import hierarchical_bootstrap_ci
from src.statistics.kendall import kendall_tau_with_ci


def test_benjamini_hochberg_matches_statsmodels_exactly():
    rng = np.random.default_rng(0)
    pvals_dict = {f't{i}': float(p) for i, p in enumerate(rng.uniform(0, 1, 15))}
    pvals_dict["t_strong"] = 0.0001
    pvals_dict["t_strong2"] = 0.001

    m = TestManifest()
    for name in pvals_dict:
        m.declare(name)
    m.freeze()
    ours = benjamini_hochberg(m, pvals_dict, q=0.05)

    names = list(pvals_dict.keys())
    raw_p = [pvals_dict[n] for n in names]
    reject, adj_p, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")

    for i, name in enumerate(names):
        assert ours[name]["adjusted_p"] == pytest.approx(adj_p[i], abs=1e-9)
        assert ours[name]["significant_at_fdr"] == bool(reject[i])


def test_incremental_r2_f_test_matches_statsmodels_anova_lm():
    rng = np.random.default_rng(42)
    n = 60
    complexity = rng.normal(size=(n, 2))
    a_scores = rng.uniform(0, 1, size=n)
    true_y = 0.5 * complexity[:, 0] + 0.3 * complexity[:, 1] + 1.2 * a_scores + rng.normal(scale=0.5, size=n)

    cal_ids = np.array([f"c{i}" for i in range(30)])
    eval_ids = np.array([f"c{i}" for i in range(30, 60)])
    eval_slice = slice(30, 60)

    eval_perf = PerformanceSeries(
        circuit_ids=np.array([f"c{i}" for i in range(30, 60)]),
        values=true_y[eval_slice], higher_is_better=True, raw_metric_name="synthetic",
    )
    ours = incremental_r2(complexity[eval_slice], a_scores[eval_slice], eval_perf, cal_ids, eval_ids)

    X_base = sm.add_constant(complexity[eval_slice])
    X_ext = sm.add_constant(np.column_stack([complexity[eval_slice], a_scores[eval_slice]]))
    y_eval = true_y[eval_slice]
    m_base = sm.OLS(y_eval, X_base).fit()
    m_ext = sm.OLS(y_eval, X_ext).fit()
    comparison = anova_lm(m_base, m_ext)

    assert ours["r2_baseline"] == pytest.approx(m_base.rsquared, abs=1e-9)
    assert ours["r2_extended"] == pytest.approx(m_ext.rsquared, abs=1e-9)
    assert ours["f_statistic"] == pytest.approx(comparison["F"].iloc[1], rel=1e-6)
    assert ours["p_value_raw"] == pytest.approx(comparison["Pr(>F)"].iloc[1], rel=1e-4)


def test_hierarchical_bootstrap_ci_brackets_classical_normal_approx_ci():
    """Not an exact-match test (bootstrap and the normal approximation are
    different methods), but the bootstrap CI for the mean of a
    reasonably-sized, roughly-normal sample must be close to the classical
    CI -- a large discrepancy would indicate a real implementation bug.
    """
    rng = np.random.default_rng(1)
    data = rng.normal(loc=5.0, scale=2.0, size=200)
    result = hierarchical_bootstrap_ci(data, np.mean, n_bootstrap=20000, seed_rng=np.random.default_rng(2))
    classical_se = np.std(data, ddof=1) / np.sqrt(len(data))
    classical_low = np.mean(data) - 1.96 * classical_se
    classical_high = np.mean(data) + 1.96 * classical_se
    # Bootstrap CI bounds should be within 5% of the classical bounds for
    # this well-behaved, moderately-sized normal sample.
    assert abs(result["ci_95_low"] - classical_low) < 0.05 * abs(classical_low)
    assert abs(result["ci_95_high"] - classical_high) < 0.05 * abs(classical_high)


def test_kendall_tau_point_estimate_and_pvalue_match_scipy_exactly():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 50)
    y = 2 * x + rng.normal(scale=0.3, size=50)
    ours = kendall_tau_with_ci(x, y, n_bootstrap=500, seed_rng=np.random.default_rng(4))
    ref_tau, ref_p = sp_stats.kendalltau(x, y)
    assert ours["tau"] == pytest.approx(ref_tau, abs=1e-12)
    assert ours["p_value_raw"] == pytest.approx(ref_p, abs=1e-12)
