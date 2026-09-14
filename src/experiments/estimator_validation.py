"""
Checkpoint 5: estimator bias/variance validation (manuscript Sec 3.6 /
brief Sec 14). Must PASS before any estimated A_spec or A_top is used in
a downstream experiment -- this module is the gate, not a convenience.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

from src.tasks.periodic import PeriodicTask
from src.estimators.spectral_estimator import estimator_bias_variance
from src.estimators.weighted_topology import validate_dp_against_exact
from src.alignment.topology import InteractionGraph


@dataclass(frozen=True)
class EstimatorValidationCriterion:
    """Pre-registered pass/fail thresholds, defined BEFORE running the
    validation (brief Sec 14: 'Define the estimator accuracy criterion
    before downstream experiments'). Placeholder values -- must be
    finalized by the researcher before real-data use; recorded explicitly
    so changing them is a visible diff, not a silent code change.
    """
    max_relative_bias: float = 0.10
    max_rmse: float = 0.10


def validate_spectral_estimator(task: PeriodicTask, candidate_frequencies: np.ndarray,
                                 sample_sizes: list, n_repeats: int, rng: np.random.Generator,
                                 criterion: EstimatorValidationCriterion) -> Dict:
    exact_spectrum = task.exact_fourier_spectrum()
    report = estimator_bias_variance(
        exact_spectrum, task.evaluate, task.d, candidate_frequencies, sample_sizes, n_repeats, rng
    )
    largest_n = max(sample_sizes)
    at_largest_n = report[largest_n]
    passed = (
        (at_largest_n["relative_bias"] is None or abs(at_largest_n["relative_bias"]) <= criterion.max_relative_bias)
        and at_largest_n["rmse"] <= criterion.max_rmse
    )
    return {
        "per_sample_size": report,
        "criterion": {"max_relative_bias": criterion.max_relative_bias, "max_rmse": criterion.max_rmse},
        "passed_at_largest_n": bool(passed),
        "largest_n_tested": largest_n,
    }


def validate_weighted_topology_estimator(circuit_graph: InteractionGraph, pairs: list, max_len: int) -> Dict:
    return validate_dp_against_exact(circuit_graph, pairs, max_len)
