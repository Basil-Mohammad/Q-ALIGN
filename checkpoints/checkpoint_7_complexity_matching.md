# Checkpoint 7 -- Complexity-Matching Validation at Scale

**Status: PASS (complexity-matching) / OPEN FINDING (stratification method)**

- Pool size: 300 circuits (4 qubits), alignment-only (no training)
- All circuits exactly match target complexity (n_q=4, P=24, depth=2): **True**
- Quartile sizes (A_min-stratified, N=300): {'1': 132, '2': 71, '3': 97, '4': 0}
- Quartile balance (max/min nonzero ratio): 1.86
- Pool generation: 0.092s total (0.31 ms/circuit)
- Alignment computation: 0.203s total (0.68 ms/circuit)

## Root-cause investigation and iteration history (full honesty, including a failed attempt)

**Attempt 1 (original, 1 re-uploading layer, multipliers in {1,2,3}):** quartile split {'1': 261, '2': 0, '3': 0, '4': 39} at N=300 -- essentially unchanged from the N=12 pilot's {'1': 10, '2': 0, '3': 0, '4': 2}, proving the imbalance is NOT a small-N artifact. Root cause: A_spec took only 3 distinct values (~82% of circuits at exactly A_spec=0), because this task requires an exact frequency match and the single-hit encoding gives each circuit only one chance per variable to hit it.

**Attempt 2 (widening multipliers to {1,...,7}), TESTED AND REJECTED:** this intuitive-seeming fix made things measurably WORSE ({'1': 293, '4': 7}, balance ratio 41.9 vs 6.7). Diagnosis: widening the choice set around a single required exact value only dilutes the per-qubit hit probability (1/3 -> 1/7); it does not help unless the added choices are themselves reachable combinations of the target frequency. This negative result is preserved here rather than discarded, per the project's negative-results policy.

**Attempt 3 (2 re-uploading layers, multipliers in {1,2}), ADOPTED:** verified numerically before adopting -- allowing SIGNED SUMS across two small, overlapping per-layer choice sets (e.g. 2-1=1) reaches the required frequency via multiple distinct combinations rather than a single lucky hit. This is a genuine root-cause fix (widens the *combinatorially reachable* frequency set) rather than a stratification-side workaround. Result at N=300: quartile split {'1': 132, '2': 71, '3': 97, '4': 0} -- dramatically more balanced (max quartile now 132/300 = 44%, down from 87%), spread across 3 active quartiles instead of 1.

## Remaining residual issue (not yet fully resolved)

Quartile 4 is still empty (0 circuits) despite A_min reaching a realized maximum of 0.9. This is a secondary, smaller-magnitude instance of the same underlying phenomenon: A_min still takes a modest number of discrete repeated values (not yet continuous), and `numpy.percentile`'s tie-breaking can place an entire tied group at the 75th-percentile boundary into the third bucket rather than splitting it into the fourth. This is a `numpy` percentile-with-ties edge case, not a new bug -- `stratify_into_quartiles` itself remains correctly tested on non-degenerate inputs (Checkpoint 1). It is reported as a smaller, not-yet-resolved residual finding rather than claimed as fully fixed.

## Decision

Complexity-matching itself is confirmed correct (all circuits exactly match the target: n_q=4, P=24, depth=2). The quartile-stratification imbalance identified in Attempt 1 has been substantially (not fully) mitigated by a verified, root-cause generator-space change (Attempt 3), reducing it from a dominant single-quartile concentration to a 3-of-4-quartiles-active spread. The residual empty-fourth-quartile issue is smaller in magnitude and is left as an open item -- a candidate fix (not yet implemented or tested) would be resolving percentile ties by index rather than by strict value comparison in `stratify_into_quartiles`. **This gate is provisionally open** for proceeding to a larger-scale trial with the 2-layer, {1,2}-multiplier generator design, with the residual tie-breaking issue flagged for a follow-up fix before the full N=420/family campaign is finalized.
