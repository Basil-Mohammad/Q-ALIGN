"""
Dynamic-programming approximation of kappa(i,j) for A_top^w at scale
(manuscript Sec 3.7: "Truncated path-sum for A_top^w").

Exact path enumeration (src/alignment/topology.py::kappa_exact) is
exponential in path length for densely connected graphs. This module
computes the same quantity via a layer-by-layer DP recursion over walks
(polynomial in n_qubits and L), and separately offers a spectral
(graph-Laplacian) approximation for very large graphs. Per the manuscript,
neither approximation is assumed bounded in [0,1] automatically -- both are
explicitly renormalized and clipped, and the renormalization's distortion
of the ranking (relative to exact, on small test graphs) must be measured
and reported by `src/experiments/estimator_validation.py`, not assumed away.
"""
from __future__ import annotations
import numpy as np
from src.alignment.topology import InteractionGraph, kappa_exact


def _adjacency_with_tau(graph: InteractionGraph) -> np.ndarray:
    n = graph.n_nodes
    A = np.zeros((n, n))
    for e, tau in graph.edges.items():
        i, j = tuple(e)
        A[i, j] = tau
        A[j, i] = tau
    return A


def kappa_dp(circuit_graph: InteractionGraph, i: int, j: int, max_len: int) -> float:
    """DP walk-generating-function approximation of kappa(i,j).

    NOTE: this counts weighted WALKS (may revisit nodes), not weighted
    SIMPLE PATHS as `kappa_exact` does -- a deliberate, documented
    relaxation to obtain polynomial cost (Sec 3.7). For graphs with limited
    entangling degree and small L this closely approximates the simple-path
    sum; the discrepancy is exactly what estimator validation must quantify.
    Returns a value renormalized into [0, 1] before being handed back.
    """
    A = _adjacency_with_tau(circuit_graph)
    n = A.shape[0]
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError("i, j out of range for circuit graph.")

    # weighted walk counts of each length ell = 1..max_len via matrix powers
    total = 0.0
    count = 0
    M = np.eye(n)
    for ell in range(1, max_len + 1):
        M = M @ A
        total += M[i, j]
        count += 1
    if count == 0:
        return 0.0
    raw = total / count  # same style of normalization as kappa_exact (divide by number of lengths considered)
    # Renormalization / clipping is MANDATORY and must be logged by the caller
    # (manuscript Sec 3.7): a weighted walk sum has no a priori upper bound
    # of exactly 1 the way a single-path product does, so we clip explicitly
    # rather than assume boundedness.
    clipped = float(np.clip(raw, 0.0, 1.0))
    return clipped


def kappa_spectral_approx(circuit_graph: InteractionGraph, i: int, j: int, max_len: int) -> float:
    """Graph-Laplacian / resistance-distance style approximation, for graphs
    too large even for the DP recursion to be worth the constant factor.
    Uses effective conductance (inverse resistance distance) as a proxy for
    connectivity strength, rescaled by the graph's own max effective
    conductance so the result lies in [0, 1] by construction (manuscript
    Sec 3.7 explicitly requires this renormalization step for this
    estimator, since -- unlike the exact/DP path-sum -- it does not inherit
    a [0,1] bound automatically).
    """
    A = _adjacency_with_tau(circuit_graph)
    n = A.shape[0]
    degrees = np.sum(A, axis=1)
    L = np.diag(degrees) - A
    # Moore-Penrose pseudoinverse of the Laplacian gives effective resistance:
    L_pinv = np.linalg.pinv(L)
    r_ij = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
    if r_ij <= 0:
        return 0.0
    conductance = 1.0 / r_ij
    # Renormalize against the maximum pairwise conductance in the graph so
    # the output is in [0,1]; this is the explicit renormalization step
    # required because this quantity has no natural [0,1] bound otherwise.
    max_conductance = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            r_ab = L_pinv[a, a] + L_pinv[b, b] - 2 * L_pinv[a, b]
            if r_ab > 0:
                max_conductance = max(max_conductance, 1.0 / r_ab)
    if max_conductance <= 0:
        return 0.0
    return float(np.clip(conductance / max_conductance, 0.0, 1.0))


def validate_dp_against_exact(circuit_graph: InteractionGraph, pairs, max_len: int) -> dict:
    """Estimator-validation helper (manuscript Sec 3.7/13 of brief):
    compares kappa_exact vs kappa_dp vs kappa_spectral_approx on a SMALL
    graph where exact enumeration is tractable, and reports absolute error,
    relative error, and rank correlation -- required before the DP or
    spectral approximation is trusted at scale.
    """
    from scipy.stats import kendalltau

    exact_vals, dp_vals, spec_vals = [], [], []
    for (i, j) in pairs:
        exact_vals.append(kappa_exact(circuit_graph, i, j, max_len))
        dp_vals.append(kappa_dp(circuit_graph, i, j, max_len))
        spec_vals.append(kappa_spectral_approx(circuit_graph, i, j, max_len))
    exact_vals = np.array(exact_vals)
    dp_vals = np.array(dp_vals)
    spec_vals = np.array(spec_vals)

    def _errs(approx):
        abs_err = np.abs(approx - exact_vals)
        rel_err = abs_err / np.clip(exact_vals, 1e-12, None)
        tau = kendalltau(exact_vals, approx).statistic if len(exact_vals) > 1 else float("nan")
        return {
            "mean_abs_error": float(np.mean(abs_err)),
            "max_abs_error": float(np.max(abs_err)),
            "mean_rel_error": float(np.mean(rel_err)),
            "kendall_tau_vs_exact": float(tau) if tau == tau else None,  # NaN check
        }

    return {
        "n_pairs": len(pairs),
        "dp_vs_exact": _errs(dp_vals),
        "spectral_vs_exact": _errs(spec_vals),
    }
