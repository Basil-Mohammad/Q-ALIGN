"""
Kendall's tau rank-prediction comparison (manuscript Sec 6.2 / brief Sec 23).
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
from scipy.stats import kendalltau


def kendall_tau_with_ci(x: np.ndarray, y: np.ndarray, n_bootstrap: int, seed_rng: np.random.Generator) -> Dict:
    """Point estimate + percentile bootstrap CI, resampling at the CIRCUIT
    level (the independent unit -- manuscript brief Sec 25: do not bootstrap
    individual seed-level observations if multiple seeds share a circuit;
    callers must pass one (x,y) pair per circuit, already aggregated over
    seeds, into this function).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length (one entry per circuit).")
    tau, p = kendalltau(x, y)
    n = len(x)
    boot_taus = []
    idx_all = np.arange(n)
    for _ in range(n_bootstrap):
        idx = seed_rng.choice(idx_all, size=n, replace=True)
        t, _ = kendalltau(x[idx], y[idx])
        if t == t:  # exclude NaN (can occur with degenerate resamples)
            boot_taus.append(t)
    boot_taus = np.array(boot_taus)
    ci_low, ci_high = np.percentile(boot_taus, [2.5, 97.5]) if len(boot_taus) > 0 else (np.nan, np.nan)
    return {
        "tau": float(tau),
        "p_value_raw": float(p),
        "n": n,
        "n_bootstrap": len(boot_taus),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
    }


def compare_dependent_correlations(x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
                                    n_bootstrap: int, seed_rng: np.random.Generator) -> Dict:
    """Compares tau(x1,y) vs tau(x2,y) where x1, x2 are measured on the SAME
    circuits (dependent correlations) -- manuscript Sec 6.2: 'compare
    dependent correlations rather than simply comparing two confidence
    intervals visually'. Uses a paired bootstrap of the DIFFERENCE in tau,
    which correctly accounts for the dependence between the two correlations
    (both computed against the same y on the same resampled circuits).
    """
    n = len(y)
    idx_all = np.arange(n)
    diffs = []
    for _ in range(n_bootstrap):
        idx = seed_rng.choice(idx_all, size=n, replace=True)
        t1, _ = kendalltau(x1[idx], y[idx])
        t2, _ = kendalltau(x2[idx], y[idx])
        if t1 == t1 and t2 == t2:
            diffs.append(t1 - t2)
    diffs = np.array(diffs)
    tau1, _ = kendalltau(x1, y)
    tau2, _ = kendalltau(x2, y)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5]) if len(diffs) > 0 else (np.nan, np.nan)
    return {
        "tau_1": float(tau1),
        "tau_2": float(tau2),
        "observed_diff": float(tau1 - tau2),
        "bootstrap_diff_ci_95_low": float(ci_low),
        "bootstrap_diff_ci_95_high": float(ci_high),
        "n_bootstrap": len(diffs),
        # CI excluding 0 is evidence of a genuine difference; caller should
        # not phrase this as a p-value without further derivation.
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }
