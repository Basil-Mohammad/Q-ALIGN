import numpy as np
import pytest
from src.alignment.topology import (
    task_interaction_graph, circuit_interaction_graph, topological_alignment_binary,
    topological_alignment_weighted, kappa_exact, variable_to_qubit_map_is_one_to_one,
    NOT_APPLICABLE,
)


def test_amplitude_encoding_not_applicable():
    """Manuscript scope limit: A_top must return the NOT_APPLICABLE sentinel
    for amplitude encoding, never a misleading float."""
    g_t = task_interaction_graph(3, [(0, 1, 1.0)])
    g_c = circuit_interaction_graph(3, [(0, 1, 1.0)])
    result = topological_alignment_binary(g_t, g_c, depth_L=2, encoding="amplitude")
    assert result == NOT_APPLICABLE


def test_unknown_encoding_raises_rather_than_assumes():
    with pytest.raises(ValueError):
        variable_to_qubit_map_is_one_to_one("some_future_encoding_nobody_registered")


def test_binary_perfect_topological_alignment():
    """Task requires (0,1); circuit directly connects (0,1) -> A_top_bin = 1."""
    g_t = task_interaction_graph(2, [(0, 1, 1.0)])
    g_c = circuit_interaction_graph(2, [(0, 1, 1.0)])
    a_top = topological_alignment_binary(g_t, g_c, depth_L=1, encoding="angle")
    assert a_top == pytest.approx(1.0)


def test_binary_zero_topological_alignment():
    """Task requires (0,2); circuit is a disconnected pair (0,1) and isolated node 2."""
    g_t = task_interaction_graph(3, [(0, 2, 1.0)])
    g_c = circuit_interaction_graph(3, [(0, 1, 1.0)])  # no path to 2
    a_top = topological_alignment_binary(g_t, g_c, depth_L=3, encoding="angle")
    assert a_top == pytest.approx(0.0)


def test_binary_partial_topological_alignment():
    """Two required interactions, only one reachable -> A_top_bin = 0.5."""
    g_t = task_interaction_graph(3, [(0, 1, 1.0), (0, 2, 1.0)])
    g_c = circuit_interaction_graph(3, [(0, 1, 1.0)])  # only (0,1) connected
    a_top = topological_alignment_binary(g_t, g_c, depth_L=1, encoding="angle")
    assert a_top == pytest.approx(0.5)


def test_binary_bounds_always_0_1():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 5
        edges = [(int(a), int(b), 1.0) for a, b in
                 rng.choice(list((i, j) for i in range(n) for j in range(i + 1, n)),
                            size=min(3, n * (n - 1) // 2), replace=False)]
        g_t = task_interaction_graph(n, edges)
        c_edges = [(int(a), int(b), 1.0) for a, b in
                   rng.choice(list((i, j) for i in range(n) for j in range(i + 1, n)),
                              size=min(3, n * (n - 1) // 2), replace=False)]
        g_c = circuit_interaction_graph(n, c_edges)
        a_top = topological_alignment_binary(g_t, g_c, depth_L=2, encoding="angle")
        assert 0.0 <= a_top <= 1.0


def test_empty_task_graph_raises():
    g_t = task_interaction_graph(2, [])
    g_c = circuit_interaction_graph(2, [(0, 1, 1.0)])
    with pytest.raises(ValueError):
        topological_alignment_binary(g_t, g_c, depth_L=1, encoding="angle")


def test_kappa_exact_bounds():
    g_c = circuit_interaction_graph(3, [(0, 1, 0.8), (1, 2, 0.6)])
    k = kappa_exact(g_c, 0, 2, max_len=2)
    assert 0.0 <= k <= 1.0


def test_weighted_topological_alignment_matches_manual_computation():
    g_t = task_interaction_graph(3, [(0, 1, 2.0), (1, 2, 1.0)])
    g_c = circuit_interaction_graph(3, [(0, 1, 1.0), (1, 2, 0.5)])

    def fixed_kappa(circuit_graph, i, j, depth_L):
        # direct edge tau if present, else 0 (single-hop only, for a hand-checkable test)
        return circuit_graph.edges.get(frozenset((i, j)), 0.0)

    a_top_w = topological_alignment_weighted(g_t, g_c, depth_L=1, encoding="angle", kappa_fn=fixed_kappa)
    # manual: (2.0*1.0 + 1.0*0.5) / (2.0+1.0) = 2.5/3.0
    assert a_top_w == pytest.approx(2.5 / 3.0)


def test_weighted_topology_rejects_out_of_bounds_kappa():
    g_t = task_interaction_graph(2, [(0, 1, 1.0)])
    g_c = circuit_interaction_graph(2, [(0, 1, 1.0)])

    def bad_kappa(circuit_graph, i, j, depth_L):
        return 1.5  # invalid: approximation failed to renormalize

    with pytest.raises(AssertionError):
        topological_alignment_weighted(g_t, g_c, depth_L=1, encoding="angle", kappa_fn=bad_kappa)


def test_conjecture_not_asserted_as_theorem():
    """Contract test (manuscript brief Sec 48): no function in this module
    may claim the topological bound (Conjecture 3.13) as proven."""
    import src.alignment.topology as topo_module
    import inspect
    src_text = inspect.getsource(topo_module)
    assert "conjecture" not in src_text.lower() or "proven" not in src_text.lower()
    assert "theorem" not in src_text.lower()  # this module implements Def 3.6-3.8 only, not any theorem
