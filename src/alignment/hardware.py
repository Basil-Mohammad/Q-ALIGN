"""
A_hw(C, H): hardware-compatibility extension (manuscript Sec 3.5, modular
extension, NOT part of the theoretically-grounded core).

Implements a first-order depolarizing-noise fidelity decay estimate, the
specific example the manuscript names for A_hw. This is a coarse,
first-order approximation (ignores crosstalk, non-Markovian effects,
readout error, per the manuscript's own Discussion Sec on hardware
realism) -- it is not claimed to predict real-device behavior, only to
provide a simulated-noise-aware ranking signal, per manuscript Sec 5.5/6
("bound the gap between simulated and real noise conditions", not "assume
simulated results generalize to hardware").
"""
from __future__ import annotations


def depolarizing_fidelity_estimate(n_two_qubit_gates: int, per_gate_error_rate: float) -> float:
    """First-order estimate: A_hw = (1 - p)^G, where G is the two-qubit gate
    count and p is the per-gate depolarizing error rate. This is a crude
    upper-bound-style estimate (ignores single-qubit gate error, readout
    error, and idle decoherence) -- explicitly a simplification, not a
    hardware-calibrated model.
    """
    if not (0.0 <= per_gate_error_rate <= 1.0):
        raise ValueError(f"per_gate_error_rate must be in [0,1], got {per_gate_error_rate}")
    if n_two_qubit_gates < 0:
        raise ValueError("n_two_qubit_gates must be >= 0")
    fidelity = (1.0 - per_gate_error_rate) ** n_two_qubit_gates
    return float(fidelity)
