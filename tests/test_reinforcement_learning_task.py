import numpy as np
import pytest
from src.tasks.reinforcement_learning import make_single_pair_bandit, make_multi_pair_bandit
from src.alignment.spectral import spectral_alignment_exact
from src.alignment.topology import topological_alignment_binary, circuit_interaction_graph


def test_optimal_action_matches_sign_of_value():
    task = make_single_pair_bandit(d=2, interacting_dims=(0, 1))
    S = np.array([[1.0, 1.0], [np.pi, 0.0], [np.pi / 4, 0.0]])
    v = task.true_value(S)
    a = task.optimal_action(S)
    assert np.array_equal(a, np.where(v >= 0, 1.0, -1.0))


def test_optimal_action_achieves_zero_regret():
    task = make_single_pair_bandit(d=2, interacting_dims=(0, 1))
    rng = np.random.default_rng(0)
    S = rng.uniform(0, 2 * np.pi, size=(100, 2))
    a_star = task.optimal_action(S)
    regret = task.regret(S, a_star)
    assert np.allclose(regret, 0.0, atol=1e-9)


def test_suboptimal_action_has_positive_regret():
    task = make_single_pair_bandit(d=2, interacting_dims=(0, 1))
    rng = np.random.default_rng(0)
    S = rng.uniform(0, 2 * np.pi, size=(100, 2))
    a_star = task.optimal_action(S)
    a_wrong = -a_star  # always the WORST action
    regret = task.regret(S, a_wrong)
    v = task.true_value(S)
    # regret for the worst action should be exactly 2*|V(s)|
    assert np.allclose(regret, 2 * np.abs(v), atol=1e-9)
    assert np.all(regret >= -1e-9)


def test_regret_never_negative_random_actions():
    task = make_single_pair_bandit(d=2, interacting_dims=(0, 1))
    rng = np.random.default_rng(1)
    S = rng.uniform(0, 2 * np.pi, size=(200, 2))
    random_actions = rng.choice([-1.0, 1.0], size=200)
    regret = task.regret(S, random_actions)
    assert np.all(regret >= -1e-9)


def test_average_regret_of_optimal_policy_is_zero():
    task = make_single_pair_bandit(d=2, interacting_dims=(0, 1))
    rng = np.random.default_rng(2)
    S = rng.uniform(0, 2 * np.pi, size=(500, 2))
    a_star = task.optimal_action(S)
    assert task.average_regret(S, a_star) == pytest.approx(0.0, abs=1e-9)


def test_reward_consistency_reward_plus_regret_equals_optimal_value():
    task = make_single_pair_bandit(d=2, interacting_dims=(0, 1))
    rng = np.random.default_rng(3)
    S = rng.uniform(0, 2 * np.pi, size=(50, 2))
    actions = rng.choice([-1.0, 1.0], size=50)
    reward = task.reward(S, actions)
    regret = task.regret(S, actions)
    optimal_value = task.optimal_value(S)
    assert np.allclose(reward + regret, optimal_value, atol=1e-9)


def test_single_pair_bandit_requires_exactly_two_dims():
    with pytest.raises(ValueError):
        make_single_pair_bandit(d=3, interacting_dims=(0, 1, 2))
    with pytest.raises(ValueError):
        make_single_pair_bandit(d=2, interacting_dims=(0, 0))


def test_exact_interaction_graph_has_expected_edge():
    task = make_single_pair_bandit(d=4, interacting_dims=(1, 3))
    g = task.exact_interaction_graph()
    assert set(g.edges.keys()) == {frozenset((1, 3))}


def test_multi_pair_bandit_has_both_required_interactions():
    task = make_multi_pair_bandit(d=4, pairs=[(0, 1), (2, 3)])
    g = task.exact_interaction_graph()
    assert set(g.edges.keys()) == {frozenset((0, 1)), frozenset((2, 3))}


def test_a_spec_computable_on_bandit_value_function():
    """Integration check: the bandit's spectrum plugs into the existing,
    already-validated spectral_alignment_exact without modification."""
    task = make_single_pair_bandit(d=2, interacting_dims=(0, 1))
    spectrum = task.exact_fourier_spectrum()
    circuit_freqs = np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]])
    a_spec = spectral_alignment_exact(spectrum, circuit_freqs)
    assert a_spec == pytest.approx(1.0, abs=1e-9)


def test_a_top_computable_on_bandit_interaction_graph():
    """Integration check: the bandit's G_T plugs into the existing
    topological_alignment_binary without modification."""
    task = make_single_pair_bandit(d=3, interacting_dims=(0, 2))
    g_t = task.exact_interaction_graph()
    g_c_connected = circuit_interaction_graph(3, [(0, 2, 1.0)])
    assert topological_alignment_binary(g_t, g_c_connected, depth_L=1, encoding="angle") == pytest.approx(1.0)
