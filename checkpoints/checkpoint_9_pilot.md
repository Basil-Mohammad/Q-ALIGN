# Checkpoint 9 -- Central Experiment PILOT

**Status: PASS (mechanism verified) -- LABEL: PILOT, NOT EVIDENCE FOR H0/H1**

- Pool size: 12 circuits (3 qubits, periodic task family)
- Seeds per circuit: 3 (pilot only; final requirement is >=10, brief Sec 7)
- Quartile sizes (A_min-stratified): {'1': 10, '2': 0, '3': 0, '4': 2}
- **Note on quartile imbalance:** at N=12 (pilot scale), `numpy.percentile`-based quartile edges produced a degenerate split (10/0/0/2) rather than an even 3/3/3/3 -- because the pilot pool's A_min distribution is bimodal (most circuits fail to match either task term, a few match both) at this tiny sample size. This is EXPECTED and is itself a real, informative pilot finding: it demonstrates concretely why the manuscript's N>=150 floor (not N=12) is necessary for meaningful quartile stratification, rather than a code defect. The mechanism (stratification logic itself) is verified correct; the sample size is not, by design, sufficient for balanced strata.
- Mean |A_spec exact - A_spec estimated (n=500)| across pool: 0.0403
- **Estimator bug found and fixed during this pilot run:** `sampled_dft_spectrum` originally called the task function per-row via a list comprehension, which silently produced a shape-(n,1) array (since `PeriodicTask.evaluate` internally reshapes single rows), corrupting the Monte Carlo estimate via an unintended (n,n) broadcast and crashing with an out-of-memory error at n=32000. Fixed by vectorizing the call (`f(X)` once on the full batch); verified post-fix that estimator error shrinks monotonically with sample size (0.058 -> 0.002 abs error from n=500 to n=100,000 on a held-out partial-match test circuit), confirming genuine convergent estimator behavior rather than a residual bug. This is reported here rather than silently corrected, per the project's own integrity requirements.
- Cost ratio (alignment estimation / avg partial training, PILOT SCALE): 0.0002
- Config hash: 75969a0cfbe1dcc14b656537740a40887b9582baaaeb60ed4c369c2cfe33a61e

## Decision

Mechanism (pool generation -> alignment computation -> A_min stratification -> illustrative sampling -> partial training -> cost timing) runs end-to-end without error at pilot scale. **Next allowed step:** full central experiment requires an explicit resourcing decision (PHASE0_AUDIT.md Sec 5) before proceeding to N>=150 per task family -- this pilot does NOT authorize skipping that decision.
