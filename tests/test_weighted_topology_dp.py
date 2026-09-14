import numpy as np
import pytest
from src.alignment.topology import circuit_interaction_graph
from src.estimators.weighted_topology import kappa_dp, kappa_spectral_approx, validate_dp_against_exact
from src.experiments.transfer import (
    freeze_calibration, apply_to_held_out, RecalibrationForbiddenError,
    select_held_out_family_by_random_draw,
)
from src.alignment.aggregation import AggregationWeights


def test_dp_approximation_bounded_0_1():
    g = circuit_interaction_graph(5, [(0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.6)])
    for (i, j) in [(0, 4), (0, 2), (1, 3)]:
        k = kappa_dp(g, i, j, max_len=3)
        assert 0.0 <= k <= 1.0


def test_spectral_approx_bounded_0_1():
    g = circuit_interaction_graph(5, [(0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.6)])
    for (i, j) in [(0, 4), (0, 2)]:
        k = kappa_spectral_approx(g, i, j, max_len=3)
        assert 0.0 <= k <= 1.0


def test_validation_report_structure():
    g = circuit_interaction_graph(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
    report = validate_dp_against_exact(g, pairs=[(0, 1), (0, 2), (0, 3)], max_len=3)
    assert "dp_vs_exact" in report
    assert "spectral_vs_exact" in report
    assert report["n_pairs"] == 3
    assert report["dp_vs_exact"]["mean_abs_error"] >= 0.0


def _weights():
    return AggregationWeights(alpha_plus=0.5, beta_plus=0.5, gamma_plus=0.0, delta_plus=0.0,
                               alpha_times=1.0, beta_times=1.0, gamma_times=0.0, delta_times=0.0)


def test_transfer_freeze_and_apply():
    frozen = freeze_calibration(_weights(), source_task_families=["periodic", "classification"])
    applied = apply_to_held_out(frozen, held_out_task_family="reinforcement_learning")
    assert applied is frozen.weights  # exact same object: no recalibration occurred


def test_transfer_forbids_recalibration_on_held_out_thats_also_source():
    frozen = freeze_calibration(_weights(), source_task_families=["periodic"])
    with pytest.raises(RecalibrationForbiddenError):
        apply_to_held_out(frozen, held_out_task_family="periodic")


def test_held_out_family_selection_is_a_random_draw_not_hand_picked():
    rng = np.random.default_rng(7)
    candidates = ["combinatorial_optimization", "physics_simulation", "graph_coloring"]
    chosen = select_held_out_family_by_random_draw(candidates, rng)
    assert chosen in candidates


def test_held_out_family_selection_requires_multiple_candidates():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        select_held_out_family_by_random_draw(["only_one"], rng)
