import numpy as np
from src.circuits.encodings import build_multi_layer_angle_encoding
from src.circuits.entanglers import linear_chain, circular_chain
from src.circuits.generators import QAlignCircuit
from src.circuits.serialization import circuit_to_dict, circuit_from_dict, round_trip_is_exact, save_pool, load_pool


def _make_test_circuit(cid="test1"):
    enc = build_multi_layer_angle_encoding(4, 4, [([0, 1, 2, 3], [1, 2, 1, 2]), ([0, 1, 2, 3], [2, 1, 2, 1])])
    return QAlignCircuit(encoding=enc, entangling_layers=[linear_chain(4), circular_chain(4)], circuit_id=cid)


def test_round_trip_preserves_accessible_frequencies():
    circ = _make_test_circuit()
    assert round_trip_is_exact(circ)


def test_round_trip_preserves_circuit_id_and_structure():
    circ = _make_test_circuit("my_special_id")
    restored = circuit_from_dict(circuit_to_dict(circ))
    assert restored.circuit_id == "my_special_id"
    assert restored.n_qubits == circ.n_qubits
    assert restored.complexity() == circ.complexity()
    assert restored.entangling_layers == circ.entangling_layers


def test_save_and_load_pool_round_trip(tmp_path):
    pool = [_make_test_circuit(f"c{i}") for i in range(5)]
    path = str(tmp_path / "pool.json")
    save_pool(pool, path)
    loaded = load_pool(path)
    assert len(loaded) == 5
    for orig, restored in zip(pool, loaded):
        assert orig.circuit_id == restored.circuit_id
        assert np.array_equal(orig.encoding.accessible_frequencies(), restored.encoding.accessible_frequencies())
        assert orig.complexity() == restored.complexity()
