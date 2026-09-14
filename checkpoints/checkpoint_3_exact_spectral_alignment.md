# Checkpoint 3 -- Exact Spectral Alignment Verification

**Status: PASS**

Re-derived from a fresh run of `tests/test_spectral.py` at report-generation time.

## What is verified
- A_spec = 1.0 exactly when the circuit's accessible frequencies fully cover the task spectrum (perfect-alignment edge case).
- A_spec = 0.0 exactly when there is no frequency overlap (zero-alignment edge case).
- A_spec = 0.5 exactly for a hand-constructed two-equal-power-term task where the circuit covers only one term (partial-alignment case, verified against a hand-computed answer, not just a bounds check).
- A_spec in [0,1] holds under randomized fuzzing (20 random task/circuit combinations).
- Zero-total-power tasks correctly raise an error (0/0 is undefined) rather than returning a silent NaN or 0.
- The spectral lower bound diagnostic (Theorem 3.5) is contractually distinct from any trainability/success claim (verified by inspecting the module's public API, not just its docstring).

## Corroborating evidence from Checkpoint 5
Checkpoint 5's estimator-vs-exact comparison at n=5000 samples (the largest tested) passed the pre-registered accuracy criterion, corroborating that the EXACT spectral computation this checkpoint verifies is a trustworthy gold standard for that estimator validation.


## Raw pytest output (tail)
```
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- /home/basil/Q-ALIGN/venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/basil/Q-ALIGN
configfile: pytest.ini
collecting ... collected 7 items

tests/test_spectral.py::test_perfect_spectral_alignment PASSED           [ 14%]
tests/test_spectral.py::test_zero_spectral_alignment PASSED              [ 28%]
tests/test_spectral.py::test_partial_spectral_alignment_multi_term PASSED [ 42%]
tests/test_spectral.py::test_a_spec_bounds_always_0_1 PASSED             [ 57%]
tests/test_spectral.py::test_zero_total_power_raises PASSED              [ 71%]
tests/test_spectral.py::test_lower_bound_diagnostic_consistency PASSED   [ 85%]
tests/test_spectral.py::test_lower_bound_is_not_a_trainability_claim PASSED [100%]

============================== 7 passed in 0.45s ===============================
```

## Decision
Exact spectral alignment computation is verified correct, including against hand-computed (not just bounds-checked) reference cases. This gate has been open since early in this session and is the foundation every other checkpoint's A_spec values rest on.
