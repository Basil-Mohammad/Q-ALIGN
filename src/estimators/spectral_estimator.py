"""
Sampled spectral-coverage estimator for A_spec (manuscript Sec 3.6, item 1).

Estimates f_T_hat(omega) at a finite set of sampled points via a
randomized DFT, for tasks WITHOUT a known closed-form spectrum. For the
periodic task family (src/tasks/periodic.py) the exact spectrum is known,
so this estimator is validated AGAINST that exact spectrum
(src/experiments/estimator_validation.py) before ever being trusted on a
task where no exact answer exists (manuscript Sec 3.6: "Do not use
estimated A_spec in downstream experiments until estimator validation
passes").
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from src.alignment.spectral import FourierSpectrum


def sampled_dft_spectrum(f: Callable[[np.ndarray], float], d: int, candidate_frequencies: np.ndarray,
                          n_samples: int, rng: np.random.Generator, domain_low: float = 0.0,
                          domain_high: float = 2 * np.pi) -> FourierSpectrum:
    """Monte Carlo estimate of the Fourier coefficients of `f` at each
    frequency in `candidate_frequencies`, via importance-sampled points
    drawn uniformly from [domain_low, domain_high)^d (manuscript Sec 3.6
    "randomized discrete Fourier transform on D-sampled inputs").

    This is a BIASED, FINITE-SAMPLE estimator; its bias/variance as a
    function of n_samples must be characterized empirically (see
    estimator_validation.py) rather than assumed negligible.

    IMPLEMENTATION NOTE (bug found and fixed during pilot execution,
    documented rather than silently patched): `f` is called ONCE on the
    full batch `X` (shape (n_samples, d)), not once per row. Calling it
    per-row via `np.array([f(x) for x in X])` silently produced a
    shape-(n_samples, 1) array (because callables like
    `PeriodicTask.evaluate` internally do `np.atleast_2d` on a single row
    and return shape (1,)), which then broadcast against the shape-
    (n_samples,) phase array into an (n_samples, n_samples) outer product
    instead of the intended elementwise product -- corrupting every
    coefficient estimate (and causing an out-of-memory crash at n=32000).
    Vectorizing the call to `f` both fixes this and is the numerically
    correct way to compute this estimator.
    """
    X = rng.uniform(domain_low, domain_high, size=(n_samples, d))
    y = np.asarray(f(X)).reshape(-1)
    if y.shape[0] != n_samples:
        raise ValueError(
            f"f(X) must return one value per row of X (shape ({n_samples},)); got shape {y.shape}. "
            f"This is exactly the shape-mismatch failure mode documented above -- refusing to proceed "
            f"silently with a shape that would corrupt the estimate."
        )
    candidate_frequencies = np.atleast_2d(candidate_frequencies)
    phase = X @ candidate_frequencies.T  # shape (n_samples, n_candidate_freqs)
    coeffs = np.mean(y[:, None] * np.exp(-1j * phase), axis=0)  # shape (n_candidate_freqs,)
    return FourierSpectrum(frequencies=candidate_frequencies.astype(float), coefficients=coeffs)


def estimator_bias_variance(exact_spectrum: FourierSpectrum, f: Callable[[np.ndarray], float], d: int,
                             candidate_frequencies: np.ndarray, sample_sizes: list,
                             n_repeats: int, rng: np.random.Generator) -> dict:
    """For each n in sample_sizes, repeat the estimator n_repeats times and
    report bias and variance of the ESTIMATED total power ratio (A_spec
    numerator/denominator components) against the exact spectrum (manuscript
    Sec 3.6 / brief Sec 14: 'Perform sample-size sweeps ... Do not select
    the sample size after seeing which one gives the strongest downstream
    result').
    """
    exact_power = exact_spectrum.power()
    exact_total = float(np.sum(exact_power))
    report = {}
    for n in sample_sizes:
        estimates = []
        for _ in range(n_repeats):
            est_spectrum = sampled_dft_spectrum(f, d, candidate_frequencies, n, rng)
            est_total = float(np.sum(est_spectrum.power()))
            estimates.append(est_total)
        estimates = np.array(estimates)
        bias = float(np.mean(estimates) - exact_total)
        variance = float(np.var(estimates))
        rmse = float(np.sqrt(bias ** 2 + variance))
        report[int(n)] = {
            "bias": bias,
            "variance": variance,
            "rmse": rmse,
            "relative_bias": bias / exact_total if exact_total != 0 else None,
        }
    return report
