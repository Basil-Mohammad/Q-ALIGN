# Checkpoint 15 -- Computational Cost Analysis (Consolidated)

**Status: PARTIAL -- consolidates REAL timing data already recorded across prior checkpoints; does NOT run new experiments, and does NOT cover the full N=420/family scale.**

## Measured timings (real, from this session's actual runs)

| Source | Operation | Value | Unit |
|---|---|---|---|
| Checkpoint 7 | Alignment computation (no training) | 0.6751 | ms/circuit |
| Checkpoint 7 | Pool generation (no training) | 0.3083 | ms/circuit |
| Checkpoint 8 | Partial training (5 steps, 4 qubits, 2 layers) | 5.5540 | s/circuit |
| Checkpoint 9 (pilot) | Alignment estimation (total, 12 circuits) | 0.0468 | s total |
| Checkpoint 9 (pilot) | Partial training (total, 12 circuits x 3 seeds) | 40.2282 | s total |

## Interpretation
No task family in this session showed alignment estimation costing MORE than partial training (all measured ratios were << 1, i.e. alignment estimation is orders of magnitude cheaper than even a few steps of training) -- consistent with the manuscript's efficiency claim (Sec 5.6) at this small scale, though this has NOT been measured at the manuscript's real N=420/family, >=10-seed, convergence-appropriate scale, where training cost per circuit would be far higher than the 5-10 step 'partial training' proxy used throughout this session's mechanism checks.


## What this does NOT cover (see PHASE0_AUDIT.md Sec 5 for the full estimate)
- Training to actual convergence (this session used 5-15 step 'partial training' throughout, not the manuscript's pre-registered convergence-appropriate protocol).
- The full noise-sweep multiplier (5 noise levels x 10 seeds applied to every circuit).
- Scaling to 8-12 qubit circuits (this session used 2-5 qubits throughout for interactive-session feasibility).
- The aggregate total-campaign budget at N=420/family x 3 families (estimated, not measured, in PHASE0_AUDIT.md Sec 5: ~64,260 simulator runs for the central experiment alone).

## Decision
Real per-circuit cost ratios measured in this session are consistent with the manuscript's efficiency claim at small scale, but this is explicitly NOT a substitute for measuring the ratio at the real campaign's scale and training depth. The compute-budget decision in PHASE0_AUDIT.md Sec 5 remains the governing document for full-campaign feasibility.
