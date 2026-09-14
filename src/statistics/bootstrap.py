"""
Hierarchical bootstrap (manuscript brief Sec 24/25): resamples at the
CIRCUIT level (the true independent experimental unit), not at the level
of individual per-seed observations, to avoid pseudoreplication when
multiple training seeds share a circuit.
"""
from __future__ import annotations
from typing import Callable, Dict, List
import numpy as np


def hierarchical_bootstrap_ci(circuit_level_values: np.ndarray, statistic_fn: Callable[[np.ndarray], float],
                               n_bootstrap: int, seed_rng: np.random.Generator,
                               method: str = "percentile") -> Dict:
    """circuit_level_values: one value per circuit (already aggregated over
    that circuit's seeds by the caller -- e.g. per-circuit mean performance).
    Resampling circuits (with replacement) respects the circuit as the
    independent unit; it does NOT resample individual seeds independently
    of their parent circuit.
    """
    if method not in ("percentile",):
        raise NotImplementedError(f"Only 'percentile' bootstrap is implemented; got method='{method}'. "
                                   f"BCa is manuscript-preferred where justified but not yet implemented here "
                                   f"-- flagged as future work rather than silently substituted.")
    n = len(circuit_level_values)
    point_estimate = statistic_fn(circuit_level_values)
    boot_stats = np.empty(n_bootstrap)
    idx_all = np.arange(n)
    for b in range(n_bootstrap):
        idx = seed_rng.choice(idx_all, size=n, replace=True)
        boot_stats[b] = statistic_fn(circuit_level_values[idx])
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    return {
        "point_estimate": float(point_estimate),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n_circuits": n,
        "n_bootstrap": n_bootstrap,
        "method": method,
    }


def aggregate_seeds_per_circuit(circuit_ids: np.ndarray, seed_values: np.ndarray) -> Dict[str, float]:
    """Collapses (circuit_id, seed_value) pairs to one mean value per
    circuit -- the required aggregation step before circuit-level inference
    (manuscript brief Sec 25: '10 seeds x 150 circuits' is NOT 1500
    independent circuit observations).
    """
    out: Dict[str, List[float]] = {}
    for cid, val in zip(circuit_ids, seed_values):
        out.setdefault(cid, []).append(val)
    return {cid: float(np.mean(vals)) for cid, vals in out.items()}
