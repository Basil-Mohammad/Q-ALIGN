# Checkpoint 11 -- Statistical Analysis Validation

**Status: PASS**

Cross-validated every statistical module (not just the alignment math of
Checkpoint 1) against trusted external references (`statsmodels`, `scipy`),
because Checkpoint 1's internal tests already proved insufficient once:
the power-analysis bug (Cohen's f vs f^2, reversed statsmodels
df_num/df_denom semantics) passed every internal shape/bounds test we had,
and was only caught by comparing against a hand-computed textbook value.

| Module | Cross-checked against | Result |
|---|---|---|
| `multiple_testing.benjamini_hochberg` | `statsmodels.stats.multitest.multipletests(method='fdr_bh')` | **Exact match** (17 p-values, adjusted p and significance flags identical) |
| `regression.incremental_r2` (nested F-test) | `statsmodels.stats.anova.anova_lm` on OLS models | **Exact match** (R^2, F-statistic, p-value all match to >=6 significant figures) |
| `bootstrap.hierarchical_bootstrap_ci` | Classical normal-approximation CI (n=200, known-normal data) | **Within 5%** of classical bounds (expected: bootstrap and normal-approx are different methods, but must agree closely for well-behaved data) |
| `kendall.kendall_tau_with_ci` (point estimate) | `scipy.stats.kendalltau` | **Exact match** (tau and raw p-value identical to 12 decimal places) |

## What this does NOT cover

- `power.py` was already audited and fixed in the prior session (see git
  history: "Fix critical power-analysis bug..."). This checkpoint focused
  on the modules NOT yet cross-validated against an external reference.
- `compare_dependent_correlations` (paired bootstrap of tau differences)
  has no simple textbook closed-form to check against; it is validated
  only via the internal behavioral test in `tests/test_statistics.py`
  (correctly detects a large, obvious difference between a strong and a
  null correlation). A more rigorous validation would require a published
  reference implementation of the dependent-correlation bootstrap test,
  which was not located during this session.
- The Sobol estimator (`sobol_estimator.py`) is explicitly documented as a
  simplified approximation, not validated against a reference library
  (SALib is not installed in this environment) -- this remains an
  acknowledged limitation, not silently presented as production-grade.

## Decision

All four cross-checked modules are confirmed correct against trusted
references. **This gate is open**: `multiple_testing`, `regression`,
`bootstrap`, and `kendall` may be relied upon for real statistical claims
once real data exists, with the two explicit exceptions noted above
(`compare_dependent_correlations`, `sobol_estimator`) flagged as
lower-confidence and not yet reference-validated.
