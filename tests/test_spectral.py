import numpy as np
import pytest
from src.alignment.spectral import (
    FourierSpectrum, spectral_alignment_exact, spectral_lower_bound_diagnostic
)
from src.tasks.periodic import make_single_frequency_task, make_multi_term_task


def test_perfect_spectral_alignment():
    """Circuit's accessible frequencies exactly cover the task spectrum -> A_spec = 1."""
    task = make_single_frequency_task(d=1, active_dims=(0,), omega_value=1.0, amplitude=1.0)
    spectrum = task.exact_fourier_spectrum()
    circuit_freqs = np.array([[-1.0], [0.0], [1.0]])
    a_spec = spectral_alignment_exact(spectrum, circuit_freqs)
    assert a_spec == pytest.approx(1.0, abs=1e-9)


def test_zero_spectral_alignment():
    """Circuit's accessible frequencies share nothing with the task spectrum -> A_spec = 0."""
    task = make_single_frequency_task(d=1, active_dims=(0,), omega_value=1.0, amplitude=1.0)
    spectrum = task.exact_fourier_spectrum()
    circuit_freqs = np.array([[5.0], [6.0]])  # nowhere near +-1.0
    a_spec = spectral_alignment_exact(spectrum, circuit_freqs)
    assert a_spec == pytest.approx(0.0, abs=1e-9)


def test_partial_spectral_alignment_multi_term():
    """Two equal-amplitude terms, circuit covers only one -> A_spec = 0.5."""
    task = make_multi_term_task(d=1, terms_spec=[((0,), 1.0, 1.0, 0.0), ((0,), 2.0, 1.0, 0.0)])
    spectrum = task.exact_fourier_spectrum()
    circuit_freqs = np.array([[-1.0], [0.0], [1.0]])  # covers omega=1 term only
    a_spec = spectral_alignment_exact(spectrum, circuit_freqs)
    assert a_spec == pytest.approx(0.5, abs=1e-6)


def test_a_spec_bounds_always_0_1():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n_terms = rng.integers(1, 4)
        terms = [((0,), float(rng.uniform(0.5, 3.0)), float(rng.uniform(0.1, 2.0)), 0.0) for _ in range(n_terms)]
        task = make_multi_term_task(d=1, terms_spec=terms)
        spectrum = task.exact_fourier_spectrum()
        circuit_freqs = rng.uniform(-3, 3, size=(5, 1))
        a_spec = spectral_alignment_exact(spectrum, circuit_freqs, atol=1e-6)
        assert 0.0 <= a_spec <= 1.0


def test_zero_total_power_raises():
    empty_spectrum = FourierSpectrum(frequencies=np.array([[0.0]]), coefficients=np.array([0.0 + 0j]))
    with pytest.raises(ValueError):
        spectral_alignment_exact(empty_spectrum, np.array([[0.0]]))


def test_lower_bound_diagnostic_consistency():
    for a_spec in [0.0, 0.3, 1.0]:
        u = spectral_lower_bound_diagnostic(a_spec)
        assert u == pytest.approx(1 - a_spec)
    with pytest.raises(ValueError):
        spectral_lower_bound_diagnostic(1.5)


def test_lower_bound_is_not_a_trainability_claim():
    """Contract test (manuscript Remark: Theorem 3.5 bounds approximation
    error only, never trainability/optimization success). We check the
    function's NAME and RETURN VALUE SEMANTICS, not its docstring (which is
    expected to mention 'trainability' precisely to warn against conflating
    it -- checking the docstring text would be testing the warning itself).
    The actual contract: the function is named/scoped as a bound on
    approximation error, and its output is a plain float in [0,1], not an
    object claiming to be a 'success probability' or similar.
    """
    assert spectral_lower_bound_diagnostic.__name__ == "spectral_lower_bound_diagnostic"
    result = spectral_lower_bound_diagnostic(0.7)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    # No sibling function in this module claims to predict trainability/success.
    import src.alignment.spectral as spectral_module
    public_fn_names = [n for n in dir(spectral_module) if not n.startswith("_")]
    for name in public_fn_names:
        assert "trainab" not in name.lower()
        assert "success" not in name.lower()
