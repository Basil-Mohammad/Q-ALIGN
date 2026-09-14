# Q-ALIGN — Implementation Repository

Implementation scaffold for the manuscript *"Q-ALIGN: A Task–Circuit
Spectral–Topological Alignment Framework for Predicting and Designing
Quantum Machine-Learning Architectures."*

## Status (read this first)

This repository is at **Phase 0–2 + a smoke-scale pilot**, not a completed
experimental campaign. See [`PHASE0_AUDIT.md`](PHASE0_AUDIT.md) for the full
audit, and [`checkpoints/`](checkpoints/) for real, generated checkpoint
reports (not simulated or hand-written numbers).

| What | Status |
|---|---|
| Manuscript → code → test mapping | ✅ Done — `PHASE0_AUDIT.md` |
| All mathematical objects (§3 of manuscript) implemented | ✅ Done, unit-tested |
| Checkpoint 0 (environment) | ✅ **PASS** — `checkpoints/checkpoint_0_environment.md` |
| Checkpoint 1 (math verification) | ✅ **74/74 tests PASS** — `pytest tests/` |
| Checkpoint 9 (central-experiment pilot, N=12 toy circuits, 3 seeds) | ✅ **PASS (mechanism only)** — `checkpoints/checkpoint_9_pilot.md`. **Explicitly labeled PILOT — NOT evidence for H0/H1.** |
| Full central experiment (N≥150 × 3 task families) | ❌ **Not run** — requires a multi-day/multi-week compute allocation (see PHASE0_AUDIT.md §5). Interfaces exist. |
| Cross-task transfer, architecture search, noise-robustness sweep, full cost campaign | ❌ **Not run** — interfaces/scaffolding exist in `src/experiments/`, not executed. |

**No numerical result in this repository is presented as evidence for or
against the manuscript's hypotheses.** Everything under `results/pilot/`
is explicitly a mechanism smoke-test.

## Two real bugs found and fixed during this delivery (disclosed, not hidden)

1. **Aggregation bound bug** (caught during manuscript review, before code
   existed): the symmetry/hardware penalty factor `(1 + γ·A_sym)` in an
   earlier draft of the conjunctive form `A_×` always inflates the score
   above 1, breaking the `[0,1]` bound. Fixed to `(1 - γ·(1 - A_sym))`.
   Regression-tested in `tests/test_aggregation.py::test_symmetry_penalty_is_bounded_not_boosting`.
2. **Estimator shape bug** (caught while running the pilot): the sampled
   spectral estimator called the task function once per row via a list
   comprehension, which silently produced a `(n, 1)`-shaped array instead
   of `(n,)`, corrupting every Fourier coefficient estimate via
   accidental `(n, n)` broadcasting and crashing with an out-of-memory
   error at `n=32000`. Fixed by vectorizing the call. See
   `checkpoints/checkpoint_9_pilot.md` for the verification that the fixed
   estimator now converges correctly with sample size.

## Repository layout

```
configs/        Frozen experiment/estimator/statistics configuration (YAML)
src/alignment/  A_spec, A_top (binary + weighted), A_sym, A_hw, A+/A×/A_min
src/estimators/ Sampled spectral estimator, Sobol proxy, weighted-topology DP/spectral approx
src/circuits/   PennyLane circuit construction, encodings, entangling layouts, complexity
src/tasks/      Synthetic task families (periodic = gold-standard exact spectrum)
src/statistics/ Regression (leakage-safe), Kendall's tau, bootstrap, BH-FDR, power analysis
src/utils/      Reproducibility (seed registry), provenance (config hashing, split assertions)
src/experiments/Central experiment, transfer, architecture search (stub), noise, cost
tests/          74 pytest unit tests (Checkpoint 1)
scripts/        checkpoint0_environment.py, run_pilot.py
checkpoints/    Generated (not hand-written) checkpoint reports
results/pilot/  Raw per-seed pilot output (PILOT label, not evidence)
```

## Running it

```bash
pip install -r requirements.txt
python scripts/checkpoint0_environment.py   # Checkpoint 0
pytest tests/ -v                            # Checkpoint 1 (74 tests)
PYTHONPATH=. python scripts/run_pilot.py    # Checkpoint 9 (pilot, ~few seconds)
```

## What is NOT in this repository

- Real QAS baselines (differentiable QAS, RL-based QAS) — interface stubs
  only in `src/experiments/architecture_search.py`, explicitly raising
  `NotImplementedError` rather than faking behavior.
- Classification and reinforcement-learning task families — only the
  periodic/Fourier task family (with a known closed-form spectrum, needed
  as the estimator-validation gold standard) is implemented.
- Any executed run at the manuscript's pre-registered scale (N≥150 per
  task family, ≥10 seeds, full noise grid). See `PHASE0_AUDIT.md` §5 for
  the compute-budget estimate and why this requires an explicit
  resourcing decision before proceeding.

## Next steps (require researcher decision, not just more coding time)

1. Decide on a compute budget/allocation for the full campaign (estimated
   in `PHASE0_AUDIT.md` §5).
2. Finalize the six `UNSPECIFIED — REQUIRES EXPLICIT DECISION` items in
   `PHASE0_AUDIT.md` §2 (Sobol threshold, minimum effect size, etc.)
   before any real-data pre-registration is filed.
3. Implement the classification and RL task families (currently only
   interfaces).
4. Implement real QAS baselines for the architecture-search comparison.
