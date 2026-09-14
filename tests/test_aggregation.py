import numpy as np
import pytest
from src.alignment.aggregation import (
    additive, conjunctive, conservative, AggregationWeights, epsilon_shifted_log
)


def default_weights(**overrides):
    base = dict(alpha_plus=0.4, beta_plus=0.4, gamma_plus=0.1, delta_plus=0.1,
                alpha_times=1.0, beta_times=1.0, gamma_times=0.1, delta_times=0.1)
    base.update(overrides)
    return AggregationWeights(**base)


def test_all_three_forms_bounded_0_1_random_inputs():
    rng = np.random.default_rng(2)
    w = default_weights()
    for _ in range(200):
        a_spec, a_top, a_sym, a_hw = rng.uniform(0, 1, size=4)
        for fn in (additive, conjunctive, conservative):
            val = fn(a_spec, a_top, a_sym, a_hw, w)
            assert -1e-9 <= val <= 1 + 1e-9, f"{fn.__name__} out of bounds: {val}"


def test_additive_can_overstate_imbalanced_circuit():
    """Regression test for the documented Remark-3.15 failure mode: A_+ with
    alpha >> beta can report a moderate score even when A_top ~ 0."""
    w = default_weights(alpha_plus=0.8, beta_plus=0.2, gamma_plus=0.0, delta_plus=0.0)
    a_plus = additive(a_spec=0.9, a_top=0.05, a_sym=0.0, a_hw=1.0, w=w)
    assert a_plus > 0.7  # A_+ is misleadingly high despite near-zero A_top


def test_conjunctive_and_min_correctly_punish_imbalance():
    """A_x and A_min must both collapse toward 0 when either core term is
    near 0, unlike A_+ (this is the entire point of Definition 3.16)."""
    w = default_weights(gamma_times=0.0, delta_times=0.0)
    a_x = conjunctive(a_spec=0.9, a_top=0.05, a_sym=0.0, a_hw=1.0, w=w)
    a_min = conservative(a_spec=0.9, a_top=0.05, a_sym=0.0, a_hw=1.0, w=w)
    assert a_x < 0.3
    assert a_min == pytest.approx(0.05)


def test_symmetry_penalty_is_bounded_not_boosting():
    """Regression test for the FIXED bug: (1 - gamma*(1-A_sym)) must never
    exceed 1, unlike the earlier erroneous (1 + gamma*A_sym) draft form."""
    w = default_weights(gamma_times=0.5, delta_times=0.0)
    # a_sym = 1.0 (perfect symmetry) -> penalty factor should be exactly 1.0 (no boost, no penalty)
    a_x_full_sym = conjunctive(a_spec=0.5, a_top=0.5, a_sym=1.0, a_hw=1.0, w=w)
    a_x_no_sym_extension = conjunctive(a_spec=0.5, a_top=0.5, a_sym=1.0, a_hw=1.0,
                                        w=default_weights(gamma_times=0.0, delta_times=0.0))
    assert a_x_full_sym == pytest.approx(a_x_no_sym_extension, abs=1e-9)
    # a_sym = 0.0 (no symmetry) -> penalty factor = (1 - gamma) < 1, a genuine penalty
    a_x_zero_sym = conjunctive(a_spec=0.5, a_top=0.5, a_sym=0.0, a_hw=1.0, w=w)
    assert a_x_zero_sym < a_x_full_sym
    assert a_x_zero_sym >= 0.0


def test_conjunctive_never_exceeds_1_even_at_extremes():
    w = default_weights(gamma_times=1.0, delta_times=1.0)
    val = conjunctive(a_spec=1.0, a_top=1.0, a_sym=1.0, a_hw=1.0, w=w)
    assert val == pytest.approx(1.0, abs=1e-9)
    assert val <= 1.0 + 1e-9


def test_conservative_equals_min_at_gamma_delta_zero():
    w = default_weights(gamma_times=0.0, delta_times=0.0)
    val = conservative(a_spec=0.3, a_top=0.8, a_sym=0.0, a_hw=0.0, w=w)
    assert val == pytest.approx(0.3)


def test_not_applicable_a_top_raises_for_all_forms():
    w = default_weights()
    for fn in (additive, conjunctive, conservative):
        with pytest.raises(ValueError):
            fn(a_spec=0.5, a_top="NOT_APPLICABLE", a_sym=0.0, a_hw=1.0, w=w)


def test_out_of_range_inputs_rejected_not_silently_clipped():
    w = default_weights()
    with pytest.raises(ValueError):
        additive(a_spec=1.5, a_top=0.5, a_sym=0.0, a_hw=1.0, w=w)
    with pytest.raises(ValueError):
        conjunctive(a_spec=-0.1, a_top=0.5, a_sym=0.0, a_hw=1.0, w=w)


def test_weights_gamma_delta_out_of_0_1_rejected():
    with pytest.raises(ValueError):
        default_weights(gamma_times=1.5)
    with pytest.raises(ValueError):
        default_weights(delta_plus=-0.2)


def test_alpha_beta_times_must_be_positive():
    with pytest.raises(ValueError):
        default_weights(alpha_times=0.0)


def test_epsilon_shifted_log_near_zero_stability():
    """Regression test for the identified Q1-quartile log(0) instability
    (Remark 3.18a): near-zero A_spec/A_top must not raise or return -inf/NaN
    when epsilon is applied."""
    val = epsilon_shifted_log(0.0, epsilon=1e-3)
    assert np.isfinite(val)
    val2 = epsilon_shifted_log(1e-12, epsilon=1e-3)  # estimator-noise-level near-zero
    assert np.isfinite(val2)


def test_epsilon_shifted_log_rejects_non_positive_epsilon():
    with pytest.raises(ValueError):
        epsilon_shifted_log(0.5, epsilon=0.0)
    with pytest.raises(ValueError):
        epsilon_shifted_log(0.5, epsilon=-0.1)


def test_epsilon_shifted_log_rejects_out_of_range_x():
    with pytest.raises(ValueError):
        epsilon_shifted_log(1.5, epsilon=1e-3)
