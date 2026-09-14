"""
A_sym(T, C): symmetry-alignment extension (manuscript Sec 3.5, modular
extension, NOT part of the theoretically-grounded core alpha/beta terms).

This is deliberately a minimal, honestly-scoped implementation: the
manuscript treats A_sym as an ablation-only extension term, never central
to the headline claims. We implement exactly one concrete symmetry check
(permutation invariance under a specified subgroup of qubit permutations)
because it is the only symmetry type both well-defined and cheaply
checkable for a gate-level circuit description without additional
manuscript-level specification. Extending to other symmetry types
(translation, reflection, task-specific group symmetries) is explicitly
OUT OF SCOPE of this delivery and is flagged as such rather than guessed.
"""
from __future__ import annotations
from typing import List, Tuple
import itertools
import numpy as np


def permutation_symmetry_alignment(circuit_gate_layout: List[Tuple[int, int]],
                                    task_is_permutation_invariant: bool,
                                    candidate_permutations: List[Tuple[int, ...]]) -> float:
    """A_sym via permutation invariance only.

    Returns 1.0 if the task is claimed permutation-invariant AND the
    circuit's two-qubit gate layout (as an edge set) is invariant under
    every permutation in `candidate_permutations` (i.e. the circuit is
    equivariant with respect to that symmetry group).
    Returns 0.0 if the task is not claimed permutation-invariant (no bonus
    for irrelevant symmetry) or if the circuit is not invariant.

    This is a binary 0/1 proxy, not a graded score -- graded symmetry
    alignment would require a manuscript-level definition (not given) of
    "how much" of a symmetry is respected, which we do not invent here.
    """
    if not task_is_permutation_invariant:
        return 0.0
    edge_set = frozenset(frozenset(e) for e in circuit_gate_layout)
    for perm in candidate_permutations:
        permuted = frozenset(frozenset((perm[i], perm[j])) for (i, j) in circuit_gate_layout)
        if permuted != edge_set:
            return 0.0
    return 1.0
