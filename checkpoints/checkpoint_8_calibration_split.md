# Checkpoint 8 -- Calibration/Evaluation Split Validation

**Status: PASS**

**SCALE WARNING: N=40 total, 1 seed/circuit, 5 training steps -- a mechanism demonstration, NOT evidence for or against H0/H1 (manuscript's real requirement is N=420/family, >=10 seeds).**

- Calibration fold: 20 circuits. Evaluation fold: 20 circuits.
- Real split (calibration vs. evaluation) passes the disjointness assertion: **True**
- Assertion correctly REJECTS an injected overlapping split (proving the gate is a real check, not a no-op): **True**
- Weight fitting was REFUSED by the numerical-stability guard: "A_top has (near-)zero variance in this calibration fold (std=0.00e+00 < 1e-06). This was observed in practice (Checkpoint 8: A_top=1.0 for all 20 calibration circuits at small scale) and produces a numerically singular, platform-dependent regression coefficient if not caught -- refusing to fit. Widen the calibration sample, the entangling-layout diversity, or the light-cone depth so A_top actually varies across the calibration fold."
- Evaluation-fold R^2 (A_+ vs. performance), computed on data never seen during fitting: N/A (fitting was refused; see below)
- Training time: 222.16s total (5554.0 ms/circuit at N_PARTIAL_STEPS=5)

## A real, serious bug found and fixed during THIS checkpoint's execution

The first run of this checkpoint (on two different machines) returned wildly different, numerically unstable weights for the SAME nominal computation: `beta_times` came out as approximately 0 on one machine and **686.92** on another. Direct inspection of the calibration fold explained why: **A_top was exactly 1.0 for all 20 calibration circuits** (zero variance) -- at this task/generator-family scale, the entangling layouts and light-cone depth used here always satisfy every required interaction (a smaller-scale echo of the near-saturated A_top distribution already noted in Checkpoint 7, mean 0.98). A constant predictor column, combined with the regression's intercept term, makes the design matrix exactly singular/rank-deficient -- an unregularized (or NNLS-constrained) least-squares solver has NO UNIQUE SOLUTION in this case, and returns an arbitrary, floating-point-noise-dependent coefficient for the degenerate predictor. **Fix:** `src/statistics/regression.py` now checks predictor variance BEFORE fitting and raises an explicit, informative error rather than silently returning an unstable number -- reproduced here as: "A_top has (near-)zero variance in this calibration fold (std=0.00e+00 < 1e-06). This was observed in practice (Checkpoint 8: A_top=1.0 for all 20 calibration circuits at small scale) and produces a numerically singular, platform-dependent regression coefficient if not caught -- refusing to fit. Widen the calibration sample, the entangling-layout diversity, or the light-cone depth so A_top actually varies across the calibration fold.". This is now covered by a permanent regression test (`tests/test_regression_protocol.py::test_zero_variance_predictor_raises_instead_of_returning_unstable_number`).

## Decision

The disjoint-split and leakage-detection mechanism is confirmed correct and working (both passes a real disjoint split and rejects an injected overlap). The weight-fitting numerical-stability bug this checkpoint surfaced has been fixed and permanently regression-tested. **This checkpoint's true value was catching a serious cross-platform numerical instability before it could contaminate a real result** -- exactly the kind of finding Checkpoint 8 exists to surface. The remaining blocker to a real, statistically meaningful result remains the compute-budget/resourcing decision documented in PHASE0_AUDIT.md Sec 5 (this demo's N=40, 1-seed, 5-step scale is far too small for A_top to show real variance, let alone for a meaningful weight fit).
