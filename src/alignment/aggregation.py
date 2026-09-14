"""
The three Q-ALIGN aggregation forms (manuscript Definitions 3.14, 3.16):

  A_+   -- additive (Definition 3.14). Logically inconsistent with the
           necessary-condition structure of Theorem 3.5 / Conjecture 3.13
           (Remark 3.15) -- retained ONLY as one of three ablation forms,
           never presented as the sole or default aggregation.
  A_x   -- conjunctive / weighted geometric mean (Definition 3.16,
           CORRECTED penalty form: (1 - gamma*(1 - A_sym)), not
           (1 + gamma*A_sym), which was an earlier draft error that broke
           the [0,1] bound -- see manuscript Remark 3.16-note).
  A_min -- conservative / minimum (manuscript Definition 3.16 variant).

None of the three is asserted to be "the" Q-ALIGN score. Every experiment
module reports all three (manuscript Sec 19 of brief / Remark 3.17-18).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AggregationWeights:
    """Shared weight container. alpha, beta play DIFFERENT mathematical
    roles in A_+ (linear coefficients) vs A_x (relative geometric-mean
    exponents) -- manuscript Remark 3.17 explicitly warns against assuming
    numeric identity is "fair" across forms. This dataclass therefore
    stores them as separate named tuples per form; `fit_matched_protocol`
    in src/statistics/regression.py is responsible for populating both
    under a matched-but-native calibration protocol, never by copying one
    into the other silently.
    """
    alpha_plus: float
    beta_plus: float
    gamma_plus: float
    delta_plus: float
    alpha_times: float
    beta_times: float
    gamma_times: float
    delta_times: float

    def __post_init__(self):
        for name in ("gamma_plus", "delta_plus", "gamma_times", "delta_times"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1] for the corrected bounded penalty form, got {v}")
        if self.alpha_times <= 0 or self.beta_times <= 0:
            raise ValueError("alpha_times, beta_times must be > 0 (they are geometric-mean exponents).")


def additive(a_spec: float, a_top, a_sym: float, a_hw: float, w: AggregationWeights) -> float:
    """Definition 3.14: A_+ = alpha*A_spec + beta*A_top + gamma*A_sym + delta*A_hw.

    NOTE (Remark 3.15): this form can report a moderate-to-high score even
    when one of A_spec, A_top is near zero, despite each being individually
    a necessary condition for good performance (Theorem 3.5 / Conjecture
    3.13). This is a KNOWN, DOCUMENTED limitation of A_+, not a bug -- the
    function still computes exactly what Definition 3.14 specifies.
    """
    if a_top == "NOT_APPLICABLE":
        raise ValueError("A_top is NOT_APPLICABLE for this encoding; A_+ cannot be computed (manuscript scope limit).")
    for name, v in [("a_spec", a_spec), ("a_top", a_top), ("a_sym", a_sym), ("a_hw", a_hw)]:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")
    return w.alpha_plus * a_spec + w.beta_plus * a_top + w.gamma_plus * a_sym + w.delta_plus * a_hw


def conjunctive(a_spec: float, a_top, a_sym: float, a_hw: float, w: AggregationWeights) -> float:
    """Definition 3.16 (CORRECTED form):

    A_x = [A_spec^alpha * A_top^beta]^(1/(alpha+beta))
          * (1 - gamma*(1 - A_sym))
          * (1 - delta*(1 - A_hw))

    Both penalty factors are of the form (1 - w*(1 - term)), which lies in
    [1-w, 1] and is <= 1 whenever term < 1 -- this is the bug-fixed version;
    an earlier draft used (1 + gamma*A_sym), which is >= 1 always and breaks
    both the "penalty" framing and the [0,1] bound (documented in the
    manuscript's own Remark 3.16-note as a corrected error, not hidden).

    With alpha,beta > 0 and gamma,delta in [0,1], A_x in [0,1] by
    construction: a product of a geometric mean of two [0,1] quantities
    with two further [0,1] factors.
    """
    if a_top == "NOT_APPLICABLE":
        raise ValueError("A_top is NOT_APPLICABLE for this encoding; A_x cannot be computed (manuscript scope limit).")
    for name, v in [("a_spec", a_spec), ("a_top", a_top), ("a_sym", a_sym), ("a_hw", a_hw)]:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")
    geo_mean = (a_spec ** w.alpha_times) * (a_top ** w.beta_times)
    geo_mean = geo_mean ** (1.0 / (w.alpha_times + w.beta_times))
    penalty_sym = 1.0 - w.gamma_times * (1.0 - a_sym)
    penalty_hw = 1.0 - w.delta_times * (1.0 - a_hw)
    result = geo_mean * penalty_sym * penalty_hw
    if not (-1e-9 <= result <= 1 + 1e-9):
        raise AssertionError(f"A_x out of [0,1]: {result} (this should be impossible given the input contracts above)")
    return float(np.clip(result, 0.0, 1.0))


def conservative(a_spec: float, a_top, a_sym: float, a_hw: float, w: AggregationWeights) -> float:
    """Definition 3.16 (A_min variant): min(A_spec, A_top) with the same
    corrected multiplicative penalty factors as A_x. Uses the SAME
    gamma_times/delta_times as A_x (both are "conjunctive-family" forms;
    only alpha_times/beta_times are not needed here since min() requires no
    relative weighting between A_spec and A_top).
    """
    if a_top == "NOT_APPLICABLE":
        raise ValueError("A_top is NOT_APPLICABLE for this encoding; A_min cannot be computed (manuscript scope limit).")
    for name, v in [("a_spec", a_spec), ("a_top", a_top), ("a_sym", a_sym), ("a_hw", a_hw)]:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")
    core = min(a_spec, a_top)
    penalty_sym = 1.0 - w.gamma_times * (1.0 - a_sym)
    penalty_hw = 1.0 - w.delta_times * (1.0 - a_hw)
    result = core * penalty_sym * penalty_hw
    if not (-1e-9 <= result <= 1 + 1e-9):
        raise AssertionError(f"A_min out of [0,1]: {result}")
    return float(np.clip(result, 0.0, 1.0))


def epsilon_shifted_log(x: float, epsilon: float) -> float:
    """Pre-registered epsilon-shift for the A_x Cobb-Douglas log-linear fit
    (manuscript Remark 3.18a). `epsilon` MUST be fixed before fitting (read
    from a frozen config, see src/statistics/regression.py), never tuned to
    improve the fit -- this function does not accept a search range, only a
    single fixed value, by design, to make silent epsilon-tuning impossible
    at the call site.
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"x must be in [0,1], got {x}")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 (log(0) is undefined).")
    return float(np.log(x + epsilon))
