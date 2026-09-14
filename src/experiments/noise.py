"""
Noise-robustness protocol (manuscript Sec 5.5): fixed, pre-registered
depolarizing-noise grid, applied identically across the central-experiment
pool (or a pre-registered sub-sample, per the fallback-priority order in
Sec 5.6/5.7).

STATUS: grid schema + application mechanism implemented; NOT executed at
scale (requires the full central-experiment pool, which is not generated
in this delivery -- see PHASE0_AUDIT.md).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pennylane as qml

from src.circuits.generators import QAlignCircuit


@dataclass(frozen=True)
class NoiseGrid:
    """Must be fixed and time-stamped BEFORE any circuit is evaluated under
    it (manuscript Sec 5.5 pre-registration-parity requirement).
    """
    per_gate_depolarizing_rates: List[float]
    n_seeds_per_level: int

    def __post_init__(self):
        for p in self.per_gate_depolarizing_rates:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Depolarizing rate must be in [0,1], got {p}")


# Placeholder pre-registered grid -- must be finalized (and then frozen,
# never edited post-hoc) before any real noise-robustness experiment.
DEFAULT_NOISE_GRID = NoiseGrid(
    per_gate_depolarizing_rates=[0.0, 0.001, 0.005, 0.01, 0.02],
    n_seeds_per_level=10,
)


def noisy_qnode(circuit: QAlignCircuit, depolarizing_rate: float):
    """Builds a PennyLane mixed-state simulation with depolarizing noise
    inserted after each two-qubit gate (a first-order, Markovian noise
    model -- manuscript Discussion Sec explicitly does not claim this
    generalizes to real hardware without a measured simulation-to-hardware
    gap).
    """
    dev = qml.device("default.mixed", wires=circuit.n_qubits)

    @qml.qnode(dev)
    def node(x: np.ndarray, theta: np.ndarray):
        for l, enc_layer in enumerate(circuit.encoding.layers):
            for q in range(circuit.n_qubits):
                var = enc_layer.qubit_variable_map[q]
                if var is not None:
                    mult = enc_layer.frequency_multipliers[q]
                    qml.RX(mult * x[var], wires=q)
                a, b, c = theta[l, q]
                qml.Rot(a, b, c, wires=q)
            for (i, j, _tau) in circuit.entangling_layers[l]:
                qml.CNOT(wires=[i, j])
                if depolarizing_rate > 0:
                    qml.DepolarizingChannel(depolarizing_rate, wires=i)
                    qml.DepolarizingChannel(depolarizing_rate, wires=j)
        return qml.expval(qml.PauliZ(0))

    return node
