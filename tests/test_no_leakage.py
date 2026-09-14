import numpy as np
import pytest
from src.utils.provenance import assert_disjoint_splits, LeakageError
from src.statistics.regression import (
    incremental_r2, PerformanceSeries, transform_mse_to_higher_is_better
)


def test_disjoint_splits_pass():
    assert_disjoint_splits(["c1", "c2", "c3"], ["c4", "c5"])  # should not raise


def test_overlapping_splits_raise_leakage_error():
    with pytest.raises(LeakageError):
        assert_disjoint_splits(["c1", "c2", "c3"], ["c3", "c5"])


def test_empty_calibration_raises():
    with pytest.raises(ValueError):
        assert_disjoint_splits([], ["c1"])


def test_incremental_r2_raises_on_overlapping_ids():
    rng = np.random.default_rng(0)
    n = 20
    complexity = rng.normal(size=(n, 2))
    a_scores = rng.uniform(0, 1, size=n)
    perf = PerformanceSeries(circuit_ids=np.array([f"c{i}" for i in range(n)]),
                              values=rng.normal(size=n), higher_is_better=True,
                              raw_metric_name="synthetic")
    calibration_ids = np.array([f"c{i}" for i in range(10)])
    evaluation_ids = np.array([f"c{i}" for i in range(5, 20)])  # overlaps c5..c9 with calibration!
    with pytest.raises(LeakageError):
        incremental_r2(complexity, a_scores, perf, calibration_ids, evaluation_ids)


def test_incremental_r2_succeeds_on_disjoint_ids():
    rng = np.random.default_rng(0)
    n = 20
    complexity = rng.normal(size=(n, 2))
    a_scores = rng.uniform(0, 1, size=n)
    perf = PerformanceSeries(circuit_ids=np.array([f"c{i}" for i in range(n)]),
                              values=rng.normal(size=n), higher_is_better=True,
                              raw_metric_name="synthetic")
    calibration_ids = np.array([f"c{i}" for i in range(10)])
    evaluation_ids = np.array([f"c{i}" for i in range(10, 20)])
    result = incremental_r2(complexity, a_scores, perf, calibration_ids, evaluation_ids)
    assert result["evaluation_fold_size"] == 10
    assert result["calibration_fold_size"] == 10
    assert "delta_r2" in result


def test_performance_series_rejects_lower_is_better():
    with pytest.raises(ValueError):
        PerformanceSeries(circuit_ids=np.array(["c1"]), values=np.array([0.5]),
                           higher_is_better=False, raw_metric_name="mse_raw")


def test_mse_transform_produces_higher_is_better_series():
    mse_values = np.array([0.1, 0.5, 0.01])
    transformed = transform_mse_to_higher_is_better(mse_values)
    # lower MSE (0.01) should map to a HIGHER transformed value
    assert transformed[2] > transformed[0] > transformed[1]
    # must be constructible as a valid higher-is-better PerformanceSeries
    perf = PerformanceSeries(circuit_ids=np.array(["a", "b", "c"]), values=transformed,
                              higher_is_better=True, raw_metric_name="neg_log_mse")
    assert perf.raw_metric_name == "neg_log_mse"


def test_mse_transform_handles_zero_mse_without_inf():
    mse_values = np.array([0.0, 1.0])
    transformed = transform_mse_to_higher_is_better(mse_values)
    assert np.all(np.isfinite(transformed))
