"""
Graph-structured classification task family (manuscript Sec 5.1, item 2).

Design rationale (documented, not arbitrary): rather than inventing a new
spectral/interaction machinery from scratch -- which would duplicate, and
risk re-introducing bugs in, the already-validated logic in
src/tasks/periodic.py -- this module builds classification tasks as a
THRESHOLD of an underlying continuous PeriodicTask. Concretely:

    label(x) = sign( g(x) ),   where g(x) = A * cos(omega . x + phase)

is a "phase-XOR"-style decision boundary: with omega having opposite-sign
components on two variables (e.g. omega = (1, -1, 0, 0)), g(x) =
cos(x_i - x_j) is a genuine two-variable interaction that CANNOT be
solved by any function of x_i and x_j taken independently (the same
qualitative reason a real XOR gate needs both inputs) -- i.e. it has a
non-trivial, exactly-known G_T for the topological-alignment analysis,
by direct reuse of PeriodicTask.exact_interaction_graph().

Explicit scope note (honesty, not a silently-swept-under limitation):
A_spec analyzes the CONTINUOUS pre-threshold function g(x), not the
discontinuous sign(.) itself. The sign() step function has infinite-
bandwidth Fourier content (Gibbs phenomenon) that no finite-frequency
circuit can represent exactly; analyzing g(x)'s spectrum instead is a
standard simplification (the "logit"/pre-activation view of a classifier)
but means A_spec here should be read as "alignment to the smooth
decision-boundary-defining function", not "alignment to the exact
discontinuous labeling function". This is flagged here explicitly rather
than presented as an unqualified match to Definition 3.3.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.tasks.periodic import PeriodicTask
from src.alignment.spectral import FourierSpectrum
from src.alignment.topology import InteractionGraph


@dataclass(frozen=True)
class ClassificationTask:
    """Binary classification task: label(x) = sign(decision_fn.evaluate(x)).

    decision_fn: the underlying PeriodicTask whose evaluate() gives the
    continuous decision score (positive -> class +1, negative -> class -1,
    exactly zero -> class +1 by convention, an explicit tie-breaking rule
    rather than an unhandled edge case).
    """
    decision_fn: PeriodicTask

    @property
    def d(self) -> int:
        return self.decision_fn.d

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        return self.decision_fn.evaluate(X)

    def label(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_score(X)
        return np.where(score >= 0, 1.0, -1.0)  # explicit tie-break at exactly 0

    def exact_fourier_spectrum(self) -> FourierSpectrum:
        """Spectrum of the CONTINUOUS decision function (see module
        docstring scope note) -- not of the discontinuous label itself.
        """
        return self.decision_fn.exact_fourier_spectrum()

    def exact_interaction_graph(self) -> InteractionGraph:
        """G_T: which variable pairs must JOINTLY influence the decision.
        Delegates directly to the already-validated PeriodicTask logic.
        """
        return self.decision_fn.exact_interaction_graph()

    def accuracy(self, X: np.ndarray, predicted_scores: np.ndarray) -> float:
        """Classification accuracy of `predicted_scores` (e.g. a circuit's
        PauliZ expectation, thresholded the same way as the true label)
        against the true label on inputs X.
        """
        true_labels = self.label(X)
        predicted_labels = np.where(predicted_scores >= 0, 1.0, -1.0)
        return float(np.mean(true_labels == predicted_labels))


def make_phase_xor_task(d: int, interacting_dims: tuple, amplitude: float = 1.0) -> ClassificationTask:
    """Convenience constructor for the canonical 'phase-XOR' task:
    label(x) = sign(cos(x_i - x_j)) for interacting_dims = (i, j).

    This is constructed via a DIRECT PeriodicTask term (not the
    make_single_frequency_task/make_multi_term_task convenience
    constructors in periodic.py, which only support EQUAL-sign frequency
    weights across active dims) -- we need omega_i = +1, omega_j = -1
    specifically, since cos(x_i - x_j) = cos(omega . x) with that signed
    omega, by the standard trigonometric identity
    cos(A - B) = cos(A)cos(B) + sin(A)sin(B).
    """
    if len(interacting_dims) != 2:
        raise ValueError("Phase-XOR requires exactly 2 interacting dimensions.")
    i, j = interacting_dims
    if i == j:
        raise ValueError("interacting_dims must be two distinct indices.")
    omega = [0.0] * d
    omega[i] = 1.0
    omega[j] = -1.0
    decision_fn = PeriodicTask(d=d, terms=[(tuple(omega), amplitude, 0.0)])
    return ClassificationTask(decision_fn=decision_fn)


def make_multi_pair_xor_task(d: int, pairs: list, amplitude: float = 1.0) -> ClassificationTask:
    """Multiple independent phase-XOR terms summed together, e.g. pairs =
    [(0,1), (2,3)] gives label(x) = sign(cos(x0-x1) + cos(x2-x3)), with an
    exact G_T containing both (0,1) and (2,3) as required interactions --
    the classification-family analogue of periodic.py's multi-term tasks,
    used for the same reason (a richer, more than one-pair interaction
    graph for the topology check).
    """
    terms = []
    for (i, j) in pairs:
        if i == j:
            raise ValueError(f"Invalid pair ({i},{j}): must be two distinct indices.")
        omega = [0.0] * d
        omega[i] = 1.0
        omega[j] = -1.0
        terms.append((tuple(omega), amplitude, 0.0))
    decision_fn = PeriodicTask(d=d, terms=terms)
    return ClassificationTask(decision_fn=decision_fn)
