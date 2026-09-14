import numpy as np
import pytest
from src.circuits.encodings import build_single_hit_angle_encoding, build_multi_layer_angle_encoding
from src.circuits.entanglers import linear_chain, circular_chain, all_to_all, random_fixed_degree
from src.circuits.complexity import CircuitComplexity, complexity_matches
from src.utils.reproducibility import SeedRegistry, require_min_seeds, FIXED_SEED_LIST


def test_single_hit_accessible_frequencies_basic():
    """One qubit, one variable, multiplier 1 -> Omega_C = {-1, 0, 1}."""
    circ = build_single_hit_angle_encoding(n_qubits=1, n_vars=1, var_per_qubit=[0], multipliers=[1])
    freqs = circ.accessible_frequencies()
    flat = sorted(freqs.ravel().tolist())
    assert flat == [-1.0, 0.0, 1.0]


def test_multi_layer_extends_frequency_range():
    """Same variable re-uploaded with multiplier 1 in two layers ->
    reachable frequencies include {-2,-1,0,1,2} (signed sums of {1,1})."""
    circ = build_multi_layer_angle_encoding(
        n_qubits=1, n_vars=1,
        layers_spec=[([0], [1]), ([0], [1])]
    )
    freqs = sorted(circ.accessible_frequencies().ravel().tolist())
    assert freqs == [-2.0, -1.0, 0.0, 1.0, 2.0]


def test_frequency_multiplier_extends_spectrum():
    """Multiplier k=3 on a single hit -> Omega_C = {-3, 0, 3}."""
    circ = build_single_hit_angle_encoding(n_qubits=1, n_vars=1, var_per_qubit=[0], multipliers=[3])
    freqs = sorted(circ.accessible_frequencies().ravel().tolist())
    assert freqs == [-3.0, 0.0, 3.0]


def test_unused_variable_only_zero_frequency():
    circ = build_single_hit_angle_encoding(n_qubits=1, n_vars=2, var_per_qubit=[0], multipliers=[1])
    freqs = circ.accessible_frequencies()  # shape (K, 2)
    # variable 1 (unused) should only ever take frequency 0
    assert set(freqs[:, 1].tolist()) == {0.0}


def test_entangling_layouts_have_expected_edge_counts():
    assert len(linear_chain(4)) == 3
    assert len(circular_chain(4)) == 4
    assert len(all_to_all(4)) == 6


def test_random_fixed_degree_deterministic_given_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    e1 = random_fixed_degree(6, degree=2, rng=rng1)
    e2 = random_fixed_degree(6, degree=2, rng=rng2)
    assert e1 == e2  # same seed -> identical layout


def test_complexity_matching_exact_and_tolerant():
    a = CircuitComplexity(n_qubits=4, n_parameters=24, depth=2, n_two_qubit_gates=3)
    b = CircuitComplexity(n_qubits=4, n_parameters=24, depth=2, n_two_qubit_gates=6)
    assert complexity_matches(a, b)  # gate count not required to match
    c = CircuitComplexity(n_qubits=5, n_parameters=24, depth=2, n_two_qubit_gates=3)
    assert not complexity_matches(a, c)  # n_qubits must match exactly


def test_seed_registry_independent_streams():
    reg = SeedRegistry(master_seed=123)
    s1 = reg.stream("training", 0)
    s2 = reg.stream("initialization", 0)
    v1 = s1.normal(size=5)
    v2 = s2.normal(size=5)
    assert not np.allclose(v1, v2)  # different purposes -> different streams


def test_seed_registry_reproducible_given_same_master_seed():
    reg_a = SeedRegistry(master_seed=999)
    reg_b = SeedRegistry(master_seed=999)
    va = reg_a.stream("training", 3).normal(size=10)
    vb = reg_b.stream("training", 3).normal(size=10)
    assert np.allclose(va, vb)


def test_seed_registry_unknown_purpose_raises():
    reg = SeedRegistry(master_seed=1)
    with pytest.raises(ValueError):
        reg.stream("not_a_registered_purpose", 0)


def test_minimum_seeds_enforced():
    with pytest.raises(ValueError):
        require_min_seeds([1, 2, 3], minimum=10)
    require_min_seeds(FIXED_SEED_LIST[:10], minimum=10)  # should not raise


def test_fixed_seed_list_has_at_least_10():
    assert len(FIXED_SEED_LIST) >= 10
