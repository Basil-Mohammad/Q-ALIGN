"""
Entangling-layout generators for the pre-enumerated generator space
(manuscript PHASE0_AUDIT.md decision #5: circuits are sampled from a finite,
pre-registered set of layouts, not from an unconstrained distribution).
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def linear_chain(n_qubits: int, tau: float = 1.0) -> List[Tuple[int, int, float]]:
    return [(i, i + 1, tau) for i in range(n_qubits - 1)]


def circular_chain(n_qubits: int, tau: float = 1.0) -> List[Tuple[int, int, float]]:
    edges = linear_chain(n_qubits, tau)
    if n_qubits > 2:
        edges.append((n_qubits - 1, 0, tau))
    return edges


def all_to_all(n_qubits: int, tau: float = 1.0) -> List[Tuple[int, int, float]]:
    return [(i, j, tau) for i in range(n_qubits) for j in range(i + 1, n_qubits)]


def random_fixed_degree(n_qubits: int, degree: int, rng: np.random.Generator, tau: float = 1.0
                         ) -> List[Tuple[int, int, float]]:
    """A random graph with (approximately) fixed degree per node, built via
    the configuration-model-style stub-matching approach, then de-duplicated.
    Deterministic given `rng` (caller must supply a seeded generator from
    src.utils.reproducibility.SeedRegistry, never global numpy random state).
    """
    if degree >= n_qubits:
        return all_to_all(n_qubits, tau)
    stubs = np.repeat(np.arange(n_qubits), degree)
    rng.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs) - 1, 2):
        a, b = stubs[i], stubs[i + 1]
        if a != b:
            edges.add(frozenset((int(a), int(b))))
    return [(*tuple(e), tau) for e in edges]


LAYOUT_SPACE = {
    "linear_chain": linear_chain,
    "circular_chain": circular_chain,
    "all_to_all": all_to_all,
}  # random_fixed_degree handled separately since it needs an rng + degree param
