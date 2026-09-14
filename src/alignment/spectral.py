"""
Spectral alignment A_spec(T, C).

Implements Definition 3.3 and the diagnostic use of Theorem 3.5 from:
  "Q-ALIGN: A Task-Circuit Spectral-Topological Alignment Framework..."

A_spec(T, C) = sum_{omega in Omega_C} |f_T_hat(omega)|^2 / sum_{omega in Omega_T} |f_T_hat(omega)|^2

Theorem 3.5 (spectral lower bound) is implemented ONLY as a reported diagnostic
quantity (1 - A_spec), NEVER as a claim about trainability or optimization
success (Remark, manuscript Sec 3.3): a circuit can have A_spec = 1 and still
fail to train (barren plateaus, finite-sample noise).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class FourierSpectrum:
    """A discrete Fourier spectrum: frequencies (1D or tuple-valued) and
    their complex coefficients, as would be produced by an exact synthetic
    task or by a sampled/estimated DFT."""
    frequencies: np.ndarray      # shape (K,) or (K, d)
    coefficients: np.ndarray     # shape (K,), complex

    def power(self) -> np.ndarray:
        return np.abs(self.coefficients) ** 2

    def total_power(self) -> float:
        return float(np.sum(self.power()))


def _match_frequencies(task_freqs: np.ndarray, circuit_freqs: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """Boolean mask over task_freqs indicating which are inside circuit_freqs.
    Frequencies are compared up to floating point tolerance `atol`.
    Supports 1D frequency arrays (scalar frequencies) and 2D (d-dimensional).
    """
    task_freqs = np.atleast_1d(task_freqs)
    circuit_freqs = np.atleast_1d(circuit_freqs)
    if task_freqs.ndim == 1:
        # broadcast compare
        diffs = np.abs(task_freqs[:, None] - circuit_freqs[None, :])
        return np.any(diffs <= atol, axis=1)
    else:
        mask = np.zeros(len(task_freqs), dtype=bool)
        for i, tf in enumerate(task_freqs):
            mask[i] = np.any(np.all(np.abs(circuit_freqs - tf) <= atol, axis=-1))
        return mask


def spectral_alignment_exact(task_spectrum: FourierSpectrum, circuit_frequencies: np.ndarray,
                              atol: float = 1e-9) -> float:
    """Definition 3.3, computed EXACTLY from a known closed-form task spectrum.

    This is the "gold standard" path, only valid for synthetic tasks whose
    Fourier decomposition is known analytically (manuscript Sec 3.6/5.1,
    periodic/Fourier-heavy task family). Do NOT use this path for tasks
    whose spectrum must itself be estimated -- use
    `spectral_alignment_estimated` instead and report estimator bias/variance
    per Sec 3.6 before any downstream use (Sec 3.6: "Do not use estimated
    A_spec in downstream experiments until estimator validation passes").
    """
    power = task_spectrum.power()
    total = float(np.sum(power))
    if total <= 0:
        raise ValueError("Task spectrum has zero total power; A_spec is undefined (0/0).")
    mask = _match_frequencies(task_spectrum.frequencies, circuit_frequencies, atol=atol)
    covered = float(np.sum(power[mask]))
    a_spec = covered / total
    # Numerical noise can push this a hair outside [0, 1]; log, do not silently clip.
    if not (-1e-9 <= a_spec <= 1 + 1e-9):
        raise AssertionError(f"A_spec out of [0,1] before clipping: {a_spec}")
    return float(np.clip(a_spec, 0.0, 1.0))


def spectral_alignment_estimated(sampled_task_spectrum: FourierSpectrum,
                                  circuit_frequencies: np.ndarray,
                                  atol: float = 1e-9) -> float:
    """Definition 3.3, computed from a SAMPLED/estimated task spectrum
    (Sec 3.6, "sampled spectral coverage" estimator). Numerically identical
    computation to the exact path, but semantically distinct: callers MUST
    treat the output as an estimate with associated bias/variance (see
    `src/estimators/spectral_estimator.py` and
    `src/experiments/estimator_validation.py`), not as ground truth.
    """
    return spectral_alignment_exact(sampled_task_spectrum, circuit_frequencies, atol=atol)


def spectral_lower_bound_diagnostic(a_spec: float) -> float:
    """U_spec(T, C) = 1 - A_spec(T, C) (Theorem 3.5).

    IMPORTANT (manuscript Remark, Sec 3.3): this is a lower bound on
    irreducible APPROXIMATION error only. It is NOT a trainability
    guarantee and NOT an optimization-error bound. Code that consumes this
    value must not interpret it as "probability of successful training."
    """
    if not (0.0 <= a_spec <= 1.0):
        raise ValueError(f"a_spec must be in [0,1], got {a_spec}")
    return 1.0 - a_spec
