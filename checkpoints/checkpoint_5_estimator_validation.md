# Checkpoint 5 -- Estimator Bias/Variance Validation

**Status: PASS**

- Pre-registered criterion: max relative bias <= 0.1, max RMSE <= 0.1 (defined BEFORE running this sweep)
- Sample sizes tested: [100, 250, 500, 1000, 2500, 5000]
- Repeats per size: 30

## Spectral estimator (A_spec) bias/variance by sample size

| n_samples | bias | variance | RMSE | relative bias |
|---|---|---|---|---|
| 100 | 2.1597 | 0.149726 | 2.1941 | 3.4555 |
| 250 | 0.8551 | 0.016561 | 0.8647 | 1.3681 |
| 500 | 0.4308 | 0.007519 | 0.4394 | 0.6892 |
| 1000 | 0.2176 | 0.002124 | 0.2224 | 0.3481 |
| 2500 | 0.0904 | 0.000842 | 0.0949 | 0.1446 |
| 5000 | 0.0456 | 0.000320 | 0.0490 | 0.0730 |

**Result at largest tested n=5000: PASSES the pre-registered criterion.**


## Weighted-topology DP/spectral approximation vs. exact (small graph, 5 qubits)

- Pairs tested: 10
- DP approximation: mean abs error = 0.1141, Kendall's tau vs exact = 0.8666666666666666
- Spectral approximation: mean abs error = 0.1317, Kendall's tau vs exact = 0.8666666666666666

## Decision

Spectral estimator meets the pre-registered accuracy criterion at the largest tested sample size. **This gate is now open**: estimated A_spec may be used in downstream experiments AT OR ABOVE this validated sample size, for tasks with similar spectral sparsity to this gold-standard task -- NOT unconditionally for arbitrary tasks.
