"""
Topological alignment A_top(T, C): binary and weighted variants.

Implements Definitions 3.6-3.8 and the weighted extension (Remark,
manuscript Sec 3.4) from the Q-ALIGN manuscript.

Scope limitation (manuscript Remark 3.10 / abstract "Scope" sentence):
A_top requires a well-defined (near-)one-to-one variable-to-qubit
correspondence (e.g. angle encoding). It is NOT informative under amplitude
encoding. This module NEVER silently returns a numeric score for an
inapplicable encoding; it returns the sentinel string "NOT_APPLICABLE".
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple, Union
import numpy as np
import networkx as nx  # declared dependency; see requirements.txt


NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class InteractionGraph:
    """Undirected graph over variable/qubit indices 0..d-1 with optional
    edge weights (task-side Sobol/ANOVA interaction strengths, s_ij)."""
    n_nodes: int
    edges: Dict[FrozenSet[int], float] = field(default_factory=dict)  # {frozenset({i,j}): weight}

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.n_nodes))
        for e, w in self.edges.items():
            i, j = tuple(e)
            g.add_edge(i, j, weight=w)
        return g


def task_interaction_graph(n_vars: int, interacting_pairs: List[Tuple[int, int, float]]) -> InteractionGraph:
    """Build G_T (Definition 3.6) from a list of (i, j, strength) triples.

    `strength` is the task-side interaction weight s_ij (e.g. a Sobol
    second-order index). For the binary A_top^bin, only edge presence
    (strength > 0) matters; for A_top^w, strength enters the weighted sum.
    """
    edges = {}
    for i, j, s in interacting_pairs:
        if i == j:
            raise ValueError("Self-interactions are not edges in G_T.")
        if s < 0:
            raise ValueError(f"Interaction strength must be >= 0, got {s} for ({i},{j}).")
        edges[frozenset((i, j))] = float(s)
    return InteractionGraph(n_nodes=n_vars, edges=edges)


def circuit_interaction_graph(n_qubits: int, two_qubit_gates: List[Tuple[int, int, float]]) -> InteractionGraph:
    """Build G_C (Definition 3.7) from the circuit's two-qubit gate layout.

    `two_qubit_gates` is a list of (qubit_i, qubit_j, tau_ij) triples, where
    tau_ij in [0,1] is the entangling capability of that specific gate
    (Sim et al. 2019 concurrence-based measure, or 1.0 if only binary
    connectivity -- not entangling strength -- is being modeled).
    """
    edges: Dict[FrozenSet[int], float] = {}
    for i, j, tau in two_qubit_gates:
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"tau_ij must be in [0,1], got {tau}")
        key = frozenset((i, j))
        # multiple layers may connect the same pair; keep the strongest single-gate tau
        # (path-sum aggregation across layers/paths is handled separately in kappa()).
        edges[key] = max(edges.get(key, 0.0), tau)
    return InteractionGraph(n_nodes=n_qubits, edges=edges)


def variable_to_qubit_map_is_one_to_one(encoding: str) -> bool:
    """Encoding-scope guard (manuscript Remark on Definition 3.7 / abstract Scope).

    Returns True only for encodings with a (near-)one-to-one variable<->qubit
    correspondence (e.g. 'angle'). Returns False for 'amplitude' encoding,
    where all variables share a single register by construction and A_top
    is not informative.
    """
    one_to_one_encodings = {"angle", "iqp", "reuploading_single_qubit_per_var"}
    not_applicable_encodings = {"amplitude"}
    if encoding in not_applicable_encodings:
        return False
    if encoding in one_to_one_encodings:
        return True
    raise ValueError(
        f"Unknown encoding '{encoding}'. Refusing to silently assume applicability; "
        f"register it explicitly in one_to_one_encodings or not_applicable_encodings."
    )


def light_cone(circuit_graph: InteractionGraph, source: int, max_hops: int) -> Set[int]:
    """N_L(i): qubits reachable from `source` within `max_hops` layers of G_C."""
    g = circuit_graph.to_networkx()
    lengths = nx.single_source_shortest_path_length(g, source, cutoff=max_hops)
    return set(lengths.keys()) - {source}


def topological_alignment_binary(task_graph: InteractionGraph, circuit_graph: InteractionGraph,
                                  depth_L: int, encoding: str) -> Union[float, str]:
    """Definition 3.8: A_top^bin(T, C).

    Returns the sentinel NOT_APPLICABLE (never a misleading float) if the
    encoding does not admit a one-to-one variable-to-qubit correspondence.
    """
    if not variable_to_qubit_map_is_one_to_one(encoding):
        return NOT_APPLICABLE
    if not task_graph.edges:
        raise ValueError("Empty task interaction graph: A_top is undefined (0/0) for a task with no required interactions.")
    reachable = 0
    for e in task_graph.edges:
        i, j = tuple(e)
        cone_i = light_cone(circuit_graph, i, depth_L)
        if j in cone_i:
            reachable += 1
    a_top = reachable / len(task_graph.edges)
    if not (0.0 <= a_top <= 1.0):
        raise AssertionError(f"A_top_bin out of [0,1]: {a_top}")
    return a_top


def _enumerate_paths_exact(circuit_graph: InteractionGraph, i: int, j: int, max_len: int) -> List[List[int]]:
    """Exact enumeration of all simple paths of length <= max_len between i and j.
    ONLY for small validation circuits (manuscript Sec 3.7): exponential in
    the worst case. Used as ground truth for testing the DP approximation,
    never in the main pipeline for n_qubits beyond a small validation size.
    """
    g = circuit_graph.to_networkx()
    paths = []
    for path in nx.all_simple_paths(g, i, j, cutoff=max_len):
        if len(path) - 1 <= max_len:
            paths.append(path)
    return paths


def kappa_exact(circuit_graph: InteractionGraph, i: int, j: int, max_len: int) -> float:
    """Exact kappa(i,j): sum over all simple paths p:i->j with |p|<=L of the
    product of edge entangling capabilities tau_uv along p, normalized to
    [0,1] by dividing by the theoretical maximum (number of counted paths,
    each contributing at most 1). Ground truth for small circuits only.
    """
    paths = _enumerate_paths_exact(circuit_graph, i, j, max_len)
    if not paths:
        return 0.0
    total = 0.0
    for path in paths:
        prod = 1.0
        for u, v in zip(path[:-1], path[1:]):
            tau = circuit_graph.edges.get(frozenset((u, v)), 0.0)
            prod *= tau
        total += prod
    # Normalize by path count so kappa in [0,1] by construction (each term in [0,1]).
    return float(np.clip(total / len(paths), 0.0, 1.0))


def topological_alignment_weighted(task_graph: InteractionGraph, circuit_graph: InteractionGraph,
                                    depth_L: int, encoding: str,
                                    kappa_fn=None) -> Union[float, str]:
    """Weighted A_top^w(T, C) (manuscript Remark on Def 3.8).

    kappa_fn(circuit_graph, i, j, depth_L) -> float in [0,1] is injected so
    callers can swap the exact path-sum (small circuits) for the DP
    approximation or spectral/resistance-distance approximation
    (`src/estimators/weighted_topology.py`) at scale, per manuscript Sec 3.7.
    Any approximation MUST renormalize its output into [0,1] before it
    reaches this function (Sec 3.7); this function asserts that bound
    rather than silently clipping it without logging (Sec 10 of brief).
    """
    if not variable_to_qubit_map_is_one_to_one(encoding):
        return NOT_APPLICABLE
    if kappa_fn is None:
        kappa_fn = kappa_exact
    if not task_graph.edges:
        raise ValueError("Empty task interaction graph: A_top^w is undefined (0/0).")
    num = 0.0
    denom = 0.0
    for e, s_ij in task_graph.edges.items():
        i, j = tuple(e)
        k = kappa_fn(circuit_graph, i, j, depth_L)
        if not (-1e-9 <= k <= 1 + 1e-9):
            raise AssertionError(
                f"kappa({i},{j}) = {k} outside [0,1]: approximation must renormalize before calling this function."
            )
        k = float(min(max(k, 0.0), 1.0))
        num += s_ij * k
        denom += s_ij
    if denom == 0:
        raise ValueError("Sum of task-side interaction strengths is zero: A_top^w undefined (0/0).")
    return float(np.clip(num / denom, 0.0, 1.0))
