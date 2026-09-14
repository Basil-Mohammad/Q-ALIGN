"""
Sobol/ANOVA-based interaction-index estimator for the task interaction
graph G_T on tasks WITHOUT a known closed-form structure (manuscript Sec
3.6, item 2). For the periodic task family, G_T is known exactly
(src/tasks/periodic.py::PeriodicTask.exact_interaction_graph) and this
estimator is not needed; it exists for future real-data task families.

This is a SIMPLIFIED second-order Sobol index estimator (variance-based),
not a full SALib-equivalent implementation (SALib is not installed in this
sandbox). It is adequate for validating the estimator-bias/variance
methodology on synthetic tasks with known second-order interactions, but
is explicitly NOT presented as a production-grade Sobol implementation.

Threshold: PHASE0_AUDIT.md decision #3 -- 0.05 is a PLACEHOLDER requiring
pre-registration before any real-data use.
"""
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np

SOBOL_THRESHOLD_REQUIRES_PREREGISTRATION = True
PLACEHOLDER_SOBOL_THRESHOLD = 0.05


def second_order_sobol_index_estimate(f: Callable[[np.ndarray], float], d: int, i: int, j: int,
                                       n_samples: int, rng: np.random.Generator,
                                       domain_low: float = 0.0, domain_high: float = 2 * np.pi) -> float:
    """A simplified Sobol-style second-order interaction index estimate via
    the Saltelli-style variance-decomposition sampling scheme, restricted
    to the (i,j) pair of interest. Returns a value in [0,1] (clipped;
    Monte Carlo noise can otherwise push it slightly outside).
    """
    A = rng.uniform(domain_low, domain_high, size=(n_samples, d))
    B = rng.uniform(domain_low, domain_high, size=(n_samples, d))
    AB_ij = A.copy()
    AB_ij[:, [i, j]] = B[:, [i, j]]
    A_i = A.copy(); A_i[:, i] = B[:, i]
    A_j = A.copy(); A_j[:, j] = B[:, j]

    fA = np.array([f(x) for x in A])
    fB = np.array([f(x) for x in B])
    fABij = np.array([f(x) for x in AB_ij])
    fAi = np.array([f(x) for x in A_i])
    fAj = np.array([f(x) for x in A_j])

    var_y = np.var(np.concatenate([fA, fB]))
    if var_y <= 1e-12:
        return 0.0

    # Closed second-order effect approx: V_ij ~ E[fB*(fABij - fAi - fAj + fA)]
    v_ij = np.mean(fB * (fABij - fAi - fAj + fA))
    s_ij = v_ij / var_y
    return float(np.clip(s_ij, 0.0, 1.0))


def estimate_interaction_graph(f: Callable[[np.ndarray], float], d: int, n_samples: int,
                                rng: np.random.Generator, threshold: float) -> List[Tuple[int, int, float]]:
    """Screens all pairs (i,j) and keeps those with estimated second-order
    Sobol index above `threshold` (manuscript: 'thresholded at a
    pre-registered significance level'). `threshold` has NO default here --
    callers must pass PLACEHOLDER_SOBOL_THRESHOLD explicitly and are
    responsible for replacing it with a genuinely pre-registered value
    before any real-data experiment.
    """
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            s_ij = second_order_sobol_index_estimate(f, d, i, j, n_samples, rng)
            if s_ij >= threshold:
                pairs.append((i, j, s_ij))
    return pairs
