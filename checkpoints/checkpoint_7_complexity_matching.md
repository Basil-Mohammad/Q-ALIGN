# Checkpoint 7 -- Complexity-Matching Validation at Scale

**Status: PASS (complexity-matching) / OPEN FINDING (stratification method)**

- Pool size: 300 circuits (4 qubits), alignment-only (no training)
- All circuits exactly match target complexity (n_q=4, P=12, depth=1): **True**
- Quartile sizes (A_min-stratified, N=300): {'1': 261, '2': 0, '3': 0, '4': 39}
- Quartile balance (max/min nonzero ratio): 6.69
- Pool generation: 0.053s total (0.18 ms/circuit)
- Alignment computation: 0.176s total (0.59 ms/circuit)

## Comparison to Checkpoint 9 pilot (N=12)

The pilot's quartile split was {'1': 10, '2': 0, '3': 0, '4': 2}, flagged there as an expected small-N artifact. At N=300, the split is {'1': 261, '2': 0, '3': 0, '4': 39} -- **the imbalance persists and is NOT a small-N artifact.** Root cause identified by inspecting the realized A_spec distribution directly: A_spec takes essentially only 3 distinct values across the pool (approximately 0.0 for ~82% of circuits, plus two partial-match values for the remainder), because the pre-registered multiplier choice set {1,2,3} (PHASE0_AUDIT.md decision #5) rarely produces an exact frequency match against this task's specific required frequencies. Since A_min = min(A_spec, A_top) and A_top is comparatively easy to satisfy (mean 0.785), A_min inherits A_spec's near-degenerate, heavily-right-skewed discrete distribution. **Percentile-based quartile stratification cannot produce a balanced split when ~82% of the population shares one exact value** -- this is a correct, expected consequence of `numpy.percentile` given this input distribution, not a bug in the stratification code itself (`stratify_into_quartiles` was already unit-tested in Checkpoint 1 and behaves correctly on well-behaved inputs).

## Decision

Complexity-matching itself is confirmed correct (all circuits exactly match the target). **However, the quartile-stratification approach is NOT yet validated as fit for purpose at the real campaign scale**, given the discrete/degenerate A_spec distribution this generator space and task combination produces. This is a genuine methodological finding, not a pass/fail code gate, and requires an explicit researcher decision before the full N=420/family campaign, among:

1. **Enrich the multiplier choice set** (e.g. {1,2,...,7} instead of {1,2,3}) to make more exact frequency matches reachable, reducing the mass concentrated at A_spec=0 -- a change to the pre-registered generator space (PHASE0_AUDIT.md decision #5), not a stratification-code fix.
2. **Replace percentile-based quartiles with fixed, pre-registered A-value bins** (e.g. [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]) rather than data-driven percentiles -- this does not fix the underlying skew but at least makes bin boundaries independent of the realized sample, and may still produce a near-empty bin if the true distribution is this skewed.
3. **Reconsider whether quartile stratification is the right analysis lens at all** for an alignment measure this discrete, versus e.g. treating A_min as a small number of natural clusters/levels rather than forcing a continuous-style quartile split.

We do not select among these here -- consistent with brief Sec 51 ('STOP and flag it. Do not silently fix it'), this is reported as an open finding for the researcher to resolve before scaling the central experiment, not resolved unilaterally in this checkpoint.
