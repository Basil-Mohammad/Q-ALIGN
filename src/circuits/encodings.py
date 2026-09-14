"""
Data-encoding generators and accessible frequency spectrum Omega_C
(manuscript Definition 3.2), for angle-encoded, data-re-uploading PQCs
(Schuld, Sweke & Meyer 2021, PRA 103:032430 -- the manuscript's cited
mechanism, see references.bib).

Only ANGLE encoding is implemented (one Pauli-rotation generator per
qubit per re-uploading layer, integer frequency multiplier). Amplitude
encoding is explicitly NOT implemented here, consistent with the
manuscript's own scope limitation on A_top (it would also require a
fundamentally different Omega_C construction not specified in the
manuscript) -- attempting to build it would be silently inventing
un-derived manuscript content.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class EncodingLayer:
    """One data re-uploading layer: for each qubit, which variable index
    it encodes (or None if untouched this layer) and the integer frequency
    multiplier (Schuld et al. 2021: repeating an encoding gate k times, or
    scaling the encoded angle by integer k, extends the accessible
    frequency set to include k*omega_0).
    """
    qubit_variable_map: List[int | None]   # length n_qubits; None = not encoded this layer
    frequency_multipliers: List[int]        # length n_qubits; ignored where map is None


@dataclass(frozen=True)
class AngleEncodingCircuit:
    n_qubits: int
    n_vars: int
    layers: List[EncodingLayer]

    def accessible_frequencies(self) -> np.ndarray:
        """Omega_C (Definition 3.2): the set of all d-dimensional frequency
        vectors reachable as signed integer combinations of the per-variable
        multipliers used anywhere across layers for that variable (Schuld
        et al. 2021: the accessible spectrum is fixed combinatorially by the
        encoding structure, independent of trainable parameters theta).
        """
        # For each variable, collect the SET of base frequencies used across
        # all layers that encode it.
        per_var_freqs: List[set] = [set() for _ in range(self.n_vars)]
        for layer in self.layers:
            for q, var in enumerate(layer.qubit_variable_map):
                if var is not None:
                    per_var_freqs[var].add(layer.frequency_multipliers[q])

        # Omega_C = all sums of signed combinations across variables:
        # for a single re-uploading structure with L "hits" per variable,
        # accessible frequencies for that variable are all integers in
        # [-sum(freqs), sum(freqs)] reachable as a signed sum of the
        # multiset of per-hit frequencies (Schuld et al. 2021, Sec III).
        # We enumerate this exactly (fine for the small qubit counts / few
        # layers used in this repo's synthetic pilot).
        def reachable_1d(freq_multiset: List[int]) -> set:
            reachable = {0}
            for f in freq_multiset:
                new_reachable = set()
                for r in reachable:
                    new_reachable.add(r + f)
                    new_reachable.add(r - f)
                reachable |= new_reachable
            return reachable

        # Reconstruct the full per-hit multiset (not just the set) per variable:
        per_var_multiset: List[List[int]] = [[] for _ in range(self.n_vars)]
        for layer in self.layers:
            for q, var in enumerate(layer.qubit_variable_map):
                if var is not None:
                    per_var_multiset[var].append(layer.frequency_multipliers[q])

        per_var_reachable = [sorted(reachable_1d(per_var_multiset[v])) if per_var_multiset[v] else [0]
                             for v in range(self.n_vars)]

        # Full Omega_C is the Cartesian product across variables (each
        # dimension's frequency chosen independently) -- consistent with a
        # product-structure multivariate re-uploading encoding.
        grids = np.meshgrid(*per_var_reachable, indexing="ij")
        omega = np.stack([g.ravel() for g in grids], axis=-1).astype(float)
        return omega


def build_single_hit_angle_encoding(n_qubits: int, n_vars: int,
                                     var_per_qubit: List[int],
                                     multipliers: List[int]) -> AngleEncodingCircuit:
    """One re-uploading layer, one variable per qubit (the common case used
    in the manuscript's central-experiment circuit family, Sec 5.1).
    `var_per_qubit[q]` gives the variable index encoded by qubit q.
    """
    if len(var_per_qubit) != n_qubits or len(multipliers) != n_qubits:
        raise ValueError("var_per_qubit and multipliers must have length n_qubits")
    layer = EncodingLayer(qubit_variable_map=list(var_per_qubit), frequency_multipliers=list(multipliers))
    return AngleEncodingCircuit(n_qubits=n_qubits, n_vars=n_vars, layers=[layer])


def build_multi_layer_angle_encoding(n_qubits: int, n_vars: int,
                                      layers_spec: List[Tuple[List[int | None], List[int]]]) -> AngleEncodingCircuit:
    """Multiple re-uploading layers (manuscript Remark on Definition 3.7:
    "multi-layer data re-uploading may route the same variable through
    several qubits across layers"). layers_spec: list of
    (qubit_variable_map, frequency_multipliers) per layer.
    """
    layers = [EncodingLayer(qubit_variable_map=m, frequency_multipliers=f) for m, f in layers_spec]
    return AngleEncodingCircuit(n_qubits=n_qubits, n_vars=n_vars, layers=layers)
