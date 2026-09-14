"""
Circuit serialization (manuscript brief file list: src/circuits/serialization.py).

Required for resumable long-run execution (brief Sec 33): a circuit pool
must be generated ONCE, persisted to disk, and reconstructed IDENTICALLY
on every resume -- re-running the RNG-based generator on resume would risk
silently producing a DIFFERENT pool if the process was interrupted mid-
generation, defeating the entire point of checkpointing.
"""
from __future__ import annotations
import json
from typing import List

from src.circuits.encodings import AngleEncodingCircuit, EncodingLayer
from src.circuits.generators import QAlignCircuit


def circuit_to_dict(circuit: QAlignCircuit) -> dict:
    return {
        "circuit_id": circuit.circuit_id,
        "n_qubits": circuit.encoding.n_qubits,
        "n_vars": circuit.encoding.n_vars,
        "layers": [
            {"qubit_variable_map": layer.qubit_variable_map, "frequency_multipliers": layer.frequency_multipliers}
            for layer in circuit.encoding.layers
        ],
        "entangling_layers": [
            [[int(i), int(j), float(tau)] for (i, j, tau) in layer]
            for layer in circuit.entangling_layers
        ],
    }


def circuit_from_dict(d: dict) -> QAlignCircuit:
    layers = [
        EncodingLayer(qubit_variable_map=layer["qubit_variable_map"],
                       frequency_multipliers=layer["frequency_multipliers"])
        for layer in d["layers"]
    ]
    encoding = AngleEncodingCircuit(n_qubits=d["n_qubits"], n_vars=d["n_vars"], layers=layers)
    entangling_layers = [
        [(int(i), int(j), float(tau)) for (i, j, tau) in layer]
        for layer in d["entangling_layers"]
    ]
    return QAlignCircuit(encoding=encoding, entangling_layers=entangling_layers, circuit_id=d["circuit_id"])


def save_pool(pool: List[QAlignCircuit], path: str) -> None:
    with open(path, "w") as f:
        json.dump([circuit_to_dict(c) for c in pool], f, indent=2)


def load_pool(path: str) -> List[QAlignCircuit]:
    with open(path) as f:
        data = json.load(f)
    return [circuit_from_dict(d) for d in data]


def round_trip_is_exact(circuit: QAlignCircuit) -> bool:
    """Sanity check: serializing then deserializing must reproduce an
    identical circuit specification (same accessible frequencies, same
    complexity) -- used as a self-check at campaign startup, not just
    assumed correct."""
    restored = circuit_from_dict(circuit_to_dict(circuit))
    import numpy as np
    same_freqs = np.array_equal(circuit.encoding.accessible_frequencies(),
                                 restored.encoding.accessible_frequencies())
    same_complexity = circuit.complexity() == restored.complexity()
    return same_freqs and same_complexity
