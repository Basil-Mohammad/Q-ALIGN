"""
Periodic/Fourier-heavy synthetic task family (manuscript Sec 5.1, item 1).

Target functions here have a KNOWN, closed-form Fourier decomposition by
construction, which makes A_spec exactly computable (the "gold standard"
required before trusting any sampled estimator, manuscript Sec 3.6/11 of
brief). The task's variable-interaction graph G_T is also known exactly
from which variables co-occur in the same trigonometric term (manuscript
Sec "UNSPECIFIED" decision #2 in PHASE0_AUDIT.md).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np

from src.alignment.spectral import FourierSpectrum
from src.alignment.topology import task_interaction_graph, InteractionGraph


@dataclass(frozen=True)
class PeriodicTask:
    """f_T(x) = sum_k c_k * cos(omega_k . x + phi_k), x in [0, 2*pi)^d.

    terms: list of (omega_vector, amplitude, phase)
    """
    d: int
    terms: List[Tuple[Tuple[float, ...], float, float]]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """X: shape (n_samples, d). Returns shape (n_samples,)."""
        X = np.atleast_2d(X)
        out = np.zeros(X.shape[0])
        for omega, amp, phase in self.terms:
            omega = np.array(omega)
            out += amp * np.cos(X @ omega + phase)
        return out

    def exact_fourier_spectrum(self) -> FourierSpectrum:
        """Closed-form spectrum: cos(w.x + phi) = 0.5*e^{i phi}e^{i w.x} +
        0.5*e^{-i phi}e^{-i w.x}, so each term contributes frequencies at
        +omega and -omega with coefficient magnitude amp/2 each.
        """
        freqs = []
        coeffs = []
        for omega, amp, phase in self.terms:
            omega = np.array(omega, dtype=float)
            freqs.append(omega)
            coeffs.append(0.5 * amp * np.exp(1j * phase))
            freqs.append(-omega)
            coeffs.append(0.5 * amp * np.exp(-1j * phase))
        return FourierSpectrum(frequencies=np.array(freqs), coefficients=np.array(coeffs))

    def exact_interaction_graph(self) -> InteractionGraph:
        """G_T: an edge (i,j) exists iff variables i and j both appear with
        nonzero coefficient in the SAME term's omega vector (i.e. they
        jointly determine that term's value, satisfying the manuscript's
        'joint dependency' criterion for Definition 3.6 exactly, without
        needing a Sobol estimate, since the closed form is known).
        Edge strength = the term's amplitude squared (a natural, exact
        analogue of a Sobol interaction index for this closed-form task).
        """
        pairs = {}
        for omega, amp, _ in self.terms:
            active_vars = [k for k, w in enumerate(omega) if w != 0]
            for a_idx in range(len(active_vars)):
                for b_idx in range(a_idx + 1, len(active_vars)):
                    i, j = active_vars[a_idx], active_vars[b_idx]
                    key = tuple(sorted((i, j)))
                    pairs[key] = pairs.get(key, 0.0) + amp ** 2
        triples = [(i, j, s) for (i, j), s in pairs.items()]
        if not triples:
            # No multi-variable term: task has no required interactions.
            # Caller must handle this (A_top is undefined for such a task).
            return InteractionGraph(n_nodes=self.d, edges={})
        return task_interaction_graph(self.d, triples)


def make_single_frequency_task(d: int, active_dims: Tuple[int, ...], omega_value: float = 1.0,
                                amplitude: float = 1.0, phase: float = 0.0) -> PeriodicTask:
    """Convenience constructor: one term with a single shared frequency
    across `active_dims`, zero elsewhere -- e.g. for d=3, active_dims=(0,1)
    gives f(x) = amplitude * cos(omega_value*(x0+x1) + phase), an explicit
    joint (2-way interacting) task.
    """
    omega = tuple(omega_value if k in active_dims else 0.0 for k in range(d))
    return PeriodicTask(d=d, terms=[(omega, amplitude, phase)])


def make_multi_term_task(d: int, terms_spec: List[Tuple[Tuple[int, ...], float, float, float]]) -> PeriodicTask:
    """terms_spec: list of (active_dims, omega_value, amplitude, phase)."""
    terms = []
    for active_dims, omega_value, amplitude, phase in terms_spec:
        omega = tuple(omega_value if k in active_dims else 0.0 for k in range(d))
        terms.append((omega, amplitude, phase))
    return PeriodicTask(d=d, terms=terms)
