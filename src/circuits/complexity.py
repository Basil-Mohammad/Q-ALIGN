"""
Conventional complexity proxies (n_q, P, L, G) used as (a) the baseline
regressors Q-ALIGN must beat incrementally, and (b) the "matching" criteria
for the central experiment's circuit pool (manuscript Sec 5.2: hold n_q, P,
L, G approximately fixed while varying encoding/entangling structure).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitComplexity:
    n_qubits: int
    n_parameters: int
    depth: int
    n_two_qubit_gates: int

    def as_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_parameters": self.n_parameters,
            "depth": self.depth,
            "n_two_qubit_gates": self.n_two_qubit_gates,
        }


def complexity_matches(a: CircuitComplexity, b: CircuitComplexity,
                        param_tol: int = 0, depth_tol: int = 0) -> bool:
    """Exact match on n_qubits; tolerance-bounded match on P, L (manuscript
    says 'approximately fixed'; we require the tolerance to be an explicit,
    pre-registered integer, not an implicit 'close enough').
    Gate count G is NOT required to match exactly in this check (the
    manuscript's own central-experiment table matches n_q, P, L only,
    listing G as a separate proxy in the baseline regression) -- flagged
    here rather than silently assumed.
    """
    return (
        a.n_qubits == b.n_qubits
        and abs(a.n_parameters - b.n_parameters) <= param_tol
        and abs(a.depth - b.depth) <= depth_tol
    )
