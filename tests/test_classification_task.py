import numpy as np
import pytest
from src.tasks.classification import make_phase_xor_task, make_multi_pair_xor_task
from src.alignment.spectral import spectral_alignment_exact
from src.alignment.topology import topological_alignment_binary, circuit_interaction_graph


def test_phase_xor_label_matches_hand_computed_cases():
    """Hand-verified truth table for cos(x0 - x1) >= 0."""
    task = make_phase_xor_task(d=2, interacting_dims=(0, 1))
    # x0 == x1 -> cos(0) = 1 >= 0 -> label +1
    assert task.label(np.array([[1.0, 1.0]]))[0] == 1.0
    # x0 - x1 = pi -> cos(pi) = -1 < 0 -> label -1
    assert task.label(np.array([[np.pi, 0.0]]))[0] == -1.0
    # x0 - x1 = pi/2 -> cos(pi/2) = 0 -> tie-break to +1 (by convention)
    assert task.label(np.array([[np.pi / 2, 0.0]]))[0] == 1.0
    # x0 - x1 = pi/4 -> cos(pi/4) > 0 -> label +1
    assert task.label(np.array([[np.pi / 4, 0.0]]))[0] == 1.0


def test_phase_xor_requires_exactly_two_dims():
    with pytest.raises(ValueError):
        make_phase_xor_task(d=3, interacting_dims=(0, 1, 2))
    with pytest.raises(ValueError):
        make_phase_xor_task(d=2, interacting_dims=(0, 0))


def test_phase_xor_exact_interaction_graph_has_exactly_the_expected_edge():
    task = make_phase_xor_task(d=4, interacting_dims=(1, 2))
    g = task.exact_interaction_graph()
    assert set(g.edges.keys()) == {frozenset((1, 2))}
    assert g.n_nodes == 4


def test_phase_xor_spectrum_matches_cosine_identity():
    """cos(x0 - x1) = 0.5*e^{i(x0-x1)} + 0.5*e^{-i(x0-x1)} -- verify the
    exact spectrum has exactly 2 frequencies at (+1,-1) and (-1,+1), each
    with |coefficient|^2 = 0.25 (total power 0.5, matching a unit-amplitude
    cosine's power under this Fourier convention -- same convention already
    validated in tests/test_spectral.py for the periodic task family).
    """
    task = make_phase_xor_task(d=2, interacting_dims=(0, 1))
    spectrum = task.exact_fourier_spectrum()
    freqs = spectrum.frequencies
    power = spectrum.power()
    assert len(freqs) == 2
    found = {tuple(f) for f in freqs}
    assert found == {(1.0, -1.0), (-1.0, 1.0)}
    assert np.allclose(power, 0.25)


def test_multi_pair_xor_has_both_required_interactions():
    task = make_multi_pair_xor_task(d=4, pairs=[(0, 1), (2, 3)])
    g = task.exact_interaction_graph()
    assert set(g.edges.keys()) == {frozenset((0, 1)), frozenset((2, 3))}


def test_multi_pair_xor_rejects_self_pair():
    with pytest.raises(ValueError):
        make_multi_pair_xor_task(d=4, pairs=[(0, 0)])


def test_a_spec_computable_on_classification_decision_function():
    """Integration check: the classification task's spectrum plugs into
    the existing, already-validated spectral_alignment_exact function
    without modification (proves the reuse-of-periodic-machinery design
    actually works end to end, not just in isolation).
    """
    task = make_phase_xor_task(d=2, interacting_dims=(0, 1))
    spectrum = task.exact_fourier_spectrum()
    # A circuit whose accessible frequencies include both (1,-1) and (-1,1)
    circuit_freqs = np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]])
    a_spec = spectral_alignment_exact(spectrum, circuit_freqs)
    assert a_spec == pytest.approx(1.0, abs=1e-9)

    # A circuit that cannot reach either frequency -> A_spec = 0
    circuit_freqs_miss = np.array([[2.0, 2.0], [0.0, 0.0]])
    a_spec_miss = spectral_alignment_exact(spectrum, circuit_freqs_miss)
    assert a_spec_miss == pytest.approx(0.0, abs=1e-9)


def test_a_top_computable_on_classification_interaction_graph():
    """Integration check: the classification task's G_T plugs into the
    existing topological_alignment_binary function without modification.
    """
    task = make_phase_xor_task(d=3, interacting_dims=(0, 2))
    g_t = task.exact_interaction_graph()

    # A circuit that connects qubits 0 and 2 directly -> A_top = 1
    g_c_connected = circuit_interaction_graph(3, [(0, 2, 1.0)])
    a_top = topological_alignment_binary(g_t, g_c_connected, depth_L=1, encoding="angle")
    assert a_top == pytest.approx(1.0)

    # A circuit that never connects 0 and 2 -> A_top = 0
    g_c_disconnected = circuit_interaction_graph(3, [(0, 1, 1.0)])
    a_top_zero = topological_alignment_binary(g_t, g_c_disconnected, depth_L=1, encoding="angle")
    assert a_top_zero == pytest.approx(0.0)


def test_accuracy_of_perfect_classifier_is_one():
    task = make_phase_xor_task(d=2, interacting_dims=(0, 1))
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2 * np.pi, size=(50, 2))
    true_scores = task.decision_score(X)  # a "perfect" classifier just reuses the true decision function
    acc = task.accuracy(X, true_scores)
    assert acc == pytest.approx(1.0)


def test_accuracy_of_random_classifier_is_near_half():
    task = make_phase_xor_task(d=2, interacting_dims=(0, 1))
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2 * np.pi, size=(500, 2))
    random_scores = rng.normal(size=500)  # uncorrelated with the true label
    acc = task.accuracy(X, random_scores)
    assert 0.4 < acc < 0.6  # loose bound; exact 0.5 not expected with finite noise
