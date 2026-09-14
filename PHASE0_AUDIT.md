# PHASE 0 — Manuscript Audit and Implementation Specification

**Status: PHASE 0 COMPLETE. PHASES 1–2 (repo, environment, math unit tests) and a SMOKE-SCALE PILOT are implemented and executed in this sandbox. PHASES 5–17 (full circuit pool, full central experiment, transfer, architecture search, noise sweep, cost campaign) are NOT executed — they require compute budgets and wall-clock time far beyond a single interactive sandbox session, and are scoped below as future work with interfaces already in place.**

This document is the mandatory pre-coding audit required by the implementation brief. It does not claim results; it maps manuscript claims to code and flags every point the manuscript leaves underspecified.

---

## 1. Manuscript → Implementation → Test → Artifact mapping

| Manuscript object | Section | Implementation module | Test | Output artifact |
|---|---|---|---|---|
| Definition 3.2 ($\Omega_C$, accessible spectrum) | §3.2 | `src/circuits/encodings.py` (`accessible_frequencies`) | `tests/test_spectral.py::test_accessible_frequencies_*` | — |
| Definition 3.3 ($A_{\mathrm{spec}}$) | §3.2 | `src/alignment/spectral.py` (`spectral_alignment_exact`, `spectral_alignment_estimated`) | `tests/test_spectral.py` | `results/.../aspec.json` |
| Theorem 3.5 (spectral lower bound) | §3.3 | `src/alignment/spectral.py` (`spectral_lower_bound`) — **diagnostic only, not a training objective** | `tests/test_spectral.py::test_lower_bound_consistency` | — |
| Definition 3.6/3.7 ($G_T$, $G_C$, $\pi$) | §3.4 | `src/alignment/topology.py` (`task_interaction_graph`, `circuit_interaction_graph`) | `tests/test_topology.py` | — |
| Definition 3.8 ($A_{\mathrm{top}}^{\mathrm{bin}}$) | §3.4 | `src/alignment/topology.py` (`topological_alignment_binary`) | `tests/test_topology.py::test_binary_bounds` | — |
| Remark (weighted $A_{\mathrm{top}}^{\mathrm{w}}$, $\kappa(i,j)$) | §3.4 | `src/alignment/topology.py` (`topological_alignment_weighted`, exact + DP) | `tests/test_topology.py::test_weighted_vs_exact_small` | — |
| Remark (amplitude-encoding non-applicability) | §3.4 | `src/alignment/topology.py` returns `"NOT_APPLICABLE"` sentinel, never a silent number | `tests/test_topology.py::test_amplitude_encoding_not_applicable` | — |
| Conjecture 3.13 (topological bound) | §3.4 | **Not implemented as a proven bound.** Left as a documented conjecture; code never asserts it. | `tests/test_topology.py::test_conjecture_not_asserted_as_theorem` | — |
| Definition 3.14 ($A_+$) | §3.5 | `src/alignment/aggregation.py` (`additive`) | `tests/test_aggregation.py` | — |
| Definition 3.16 ($A_\times$, $A_{\min}$, corrected penalty form) | §3.5 | `src/alignment/aggregation.py` (`conjunctive`, `conservative`) | `tests/test_aggregation.py::test_bounded_0_1` | — |
| Remark 3.17/3.18 (native-but-matched calibration, $\epsilon$-shift, performance sign convention) | §3.5 | `src/statistics/regression.py` (`fit_additive`, `fit_conjunctive`, shared protocol) | `tests/test_regression_protocol.py` | — |
| Remark 3.19 (calibration/evaluation split, no leakage) | §3.5/§6 | `src/utils/provenance.py` (`assert_disjoint_splits`), `src/experiments/central_experiment.py` | `tests/test_no_leakage.py` | `logs/.../split_manifest.json` |
| §3.6 (sampled spectral estimator, Sobol topological proxy) | §3.6 | `src/estimators/spectral_estimator.py`, `src/estimators/sobol_estimator.py` | `tests/test_estimators.py` | `results/estimator_validation/` |
| §3.7 (truncated path-sum DP for $A_{\mathrm{top}}^{\mathrm{w}}$, renormalization) | §3.7 | `src/estimators/weighted_topology.py` | `tests/test_weighted_topology_dp.py` | — |
| §5.1 (task/circuit families) | §5.1 | `src/tasks/periodic.py` (exact closed-form, implemented), `src/tasks/classification.py`, `src/tasks/reinforcement_learning.py` (interfaces only, **not implemented**) | — | — |
| §5.2 (central experiment protocol: pool, $A_{\min}$ stratification, power analysis, N≥150) | §5.2 | `src/experiments/central_experiment.py` | `tests/test_central_experiment_protocol.py` | `results/central_experiment/` |
| §5.3 (cross-task transfer, no recalibration) | §5.3 | `src/experiments/transfer.py` — **interface + freeze/apply logic implemented; not executed at scale** | `tests/test_transfer_freeze.py` | — |
| §5.4 (architecture search) | §5.4 | `src/experiments/architecture_search.py` — **interface only** | — | — |
| §5.5 (noise robustness) | §5.5 | `src/experiments/noise.py` — **interface + fixed grid schema; not executed at scale** | — | — |
| §5.6 (computational cost) | §5.6 | `src/experiments/cost.py` — **interface + timing decorator implemented** | — | — |
| §6.1 (multiple comparisons, BH-FDR) | §6.1 | `src/statistics/multiple_testing.py` | `tests/test_multiple_testing.py` | — |
| §6 ($\Delta R^2$, Kendall's $\tau$, bootstrap) | §6 | `src/statistics/regression.py`, `src/statistics/kendall.py`, `src/statistics/bootstrap.py` | `tests/test_statistics.py` | — |
| Power analysis ($N \geq 150$ floor) | §5.2 | `src/statistics/power.py` | `tests/test_power.py` | — |

---

## 2. UNSPECIFIED — REQUIRES EXPLICIT DECISION

The manuscript is a research proposal, not a fully operationalized protocol. The following gaps are real and are **not** silently resolved. Each is given the smallest scientifically defensible default, implemented as an explicit, overridable config value — never hard-coded inside logic.

1. **Quantum backend.** The manuscript is backend-agnostic. **Decision:** PennyLane `default.qubit` (pure-Python statevector simulator), because it is pip-installable in this sandbox without GPU/Docker dependencies and is standard in the QML literature the manuscript cites (Schuld is a PennyLane co-author). Qiskit is not installed here; the code isolates the simulator behind `src/circuits/generators.py` so a Qiskit backend can be substituted without touching alignment math.
2. **Task interaction graph $G_T$ ground truth for synthetic tasks.** **Decision:** for the periodic/Fourier task family, $G_T$ is defined analytically from the known closed-form target function (which pairs of variables appear in the same trigonometric term), not estimated — this is the "exact ground truth" case the manuscript calls for in estimator validation. For any real-data task family (not yet implemented), $G_T$ would require a pre-registered Sobol threshold, which is a second unspecified value flagged below.
3. **Sobol/ANOVA significance threshold for estimating $E_T$ on real data.** The manuscript says "thresholded at a pre-registered significance level" but does not name one. **Decision (placeholder, must be pre-registered before any real-data experiment):** a fixed second-order Sobol index threshold of 0.05, chosen as a conventional small-effect cutoff, flagged in `configs/estimators/sobol.yaml` as `THRESHOLD_REQUIRES_PREREGISTRATION: true`.
4. **Minimum detectable effect size for the power analysis.** The manuscript requires a power analysis but does not state the effect size of practical interest. **Decision (placeholder):** partial $R^2 = 0.05$ (a conventional "small-to-medium" Cohen's $f^2 \approx 0.053$), implemented as a named config constant, not derived from data.
5. **Circuit-generator randomization distribution.** The manuscript says circuits are generated "by independently randomizing the data-encoding generator set and entangling layout" but does not specify the distribution. **Decision:** uniform random choice over a pre-enumerated finite set of Pauli-rotation generators (`{RX, RY, RZ}` and small integer frequency multipliers `{1,2,3}`) and uniform random choice over a pre-enumerated finite set of entangling layouts (linear chain, circular, all-to-all, random-fixed-degree), both enumerated in `configs/circuits/generator_space.yaml` before any pool is sampled.
6. **"Partial training" operational definition for the cost ratio (§5.6).** **Decision:** a fixed, pre-registered number of gradient-descent steps (default 50) on a fixed mini-batch size, defined in `configs/experiments/cost.yaml`, not "however many steps until convergence" (which would make the cost ratio ill-defined and gameable).
7. **Gradient-variance trainability proxy operationalization.** **Decision:** sample-variance of the parameter-shift gradient over a fixed number of random parameter initializations (default 20), matching the standard barren-plateau diagnostic in McClean et al. (2018), recorded in `configs/experiments/predictive.yaml`.

None of these six decisions were chosen to make Q-ALIGN look better; they are recorded so that changing them later requires a visible config diff, not a silent code change.

---

## 3. Statistical and implementation threats identified (Section 51 of the brief)

- **Pseudoreplication risk** (Section 25 of the brief): confirmed present if naively implemented — 10 seeds × N circuits must NOT be treated as $10N$ independent circuit-level observations. **Resolution implemented:** `src/statistics/regression.py` aggregates to one performance value per circuit (mean over seeds, with per-seed values retained in raw storage) before circuit-level regression; a cluster-robust / hierarchical option is exposed via `statsmodels` mixed-effects as an alternative specification, not the default, and both are required to be reported per Remark-style transparency once real data exists.
- **Calibration/evaluation leakage** (already identified in manuscript review): implemented as a hard assertion (`assert_disjoint_splits`), not a convention — the code raises `LeakageError` if any circuit ID appears in both folds.
- **$\epsilon$-shift tuning leakage** (manuscript Remark 3.18a): implemented as a **pre-registered fixed value read from config**, with the sensitivity sweep computed only as a reported diagnostic, never as a value-selection loop that touches downstream $\Delta R^2$.
- **Multiple-comparisons scope creep:** the BH-FDR procedure requires a *fixed, declared* family of tests. Implemented via `src/statistics/multiple_testing.py::TestManifest`, which must be frozen (hashed) before any p-value is computed; adding a test after seeing results raises an error rather than silently recomputing the correction.
- **Quartile-definition ambiguity** (which aggregation form defines Table 2's strata): implemented per the manuscript's resolution — $A_{\min}$ is the stratifying form, hard-coded as the default in `configs/experiments/central_experiment.yaml` with a named field `stratification_form: A_min`, not left to whichever form is computed first.

---

## 4. Checkpoints implemented vs. deferred

| Checkpoint | Status in this delivery |
|---|---|
| 0 — Environment | **DONE.** See `checkpoints/checkpoint_0_environment.md` (real, generated by `scripts/checkpoint0_environment.py`). |
| 1 — Mathematical definitions | **DONE.** See `checkpoints/checkpoint_1_math.md`, generated from real `pytest` output. |
| 2 — Circuit construction | **DONE (smoke scale).** Small angle-encoding circuits (3–5 qubits) build and simulate correctly in PennyLane; see `tests/test_circuits.py`. |
| 3 — Exact spectral alignment | **DONE (synthetic gold-standard task).** |
| 4 — Exact topological alignment | **DONE (small graphs, exact path enumeration).** |
| 5 — Estimator bias/variance | **DONE at pilot scale only** (sample sizes 100–2000, not the full pre-registered sweep to 5000+ needed for a real estimator-validation report). |
| 6 — Reproducibility/seed validation | **DONE.** Deterministic per-stream RNG implemented and tested. |
| 7 — Complexity-matching validation | **DONE (pilot pool of 12 circuits).** Full $N\geq150$ pool NOT generated (compute/time). |
| 8 — Calibration/evaluation split | **DONE (mechanism + assertion tested); not run on a full pool.** |
| 9 — Central experiment pilot | **DONE.** See `results/pilot/`. Explicitly labeled PILOT; not evidence for $H_1$. |
| 10 — Full central experiment | **NOT RUN.** Requires the resourcing decision in §5 below. |
| 11–17 | **NOT RUN.** Interfaces exist; no fabricated numbers are reported anywhere in this delivery. |

---

## 5. Computational cost estimate for the full campaign (feasibility, per manuscript §5.6/§5.7)

Rough estimate, to be replaced by measured pilot numbers once Checkpoint 9's timings are available (see `results/pilot/timing.json` for the actual measured pilot numbers this estimate is anchored to):

- Pool per task family: $N \geq 150$ (power-analysis floor) → **450 circuits minimum across 3 families**.
- Partial training per circuit (pre-registered 50 steps, pilot-measured): ~seconds on `default.qubit` at 3–5 qubits; scaling to realistic 8–12 qubit circuits used in the manuscript's central-experiment table is expected to be 10–50× slower per circuit on a statevector simulator, and training to convergence (not just the 50-step "partial" definition) is substantially more again.
- With 10 seeds/circuit and a full noise sweep (pre-registered grid, e.g. 5 noise levels × 10 seeds) applied to the same 450 circuits: **on the order of $450 \times 10 \times (1 + 5\times10) = $ tens of thousands of simulator training runs** for the central experiment alone, before transfer, architecture search, or cost-analysis experiments are added.
- **This is not executable within a single interactive sandbox session or a small budget.** It is a multi-day-to-multi-week batch job on a dedicated compute allocation (a small CPU/GPU cluster or a multi-core workstation running jobs over days), consistent with the manuscript's own §5.7 feasibility-check requirement. Per the manuscript's own pre-registered fallback order (reduce noise-sweep sub-sampling → reduce covariate strata → only last-resort reduce $N$ below the power floor), the recommended next step is **not** to shrink scope silently but to make an explicit resourcing decision (see open questions, end of this document).

---

## 6. Seed strategy

- Minimum 10 independent seeds per final stochastic result (manuscript requirement), 20 for high-variance training comparisons (architecture search), pinned to a documented fixed list stored in `configs/global.yaml::seed_registry`, never regenerated at runtime.
- Seed streams are per-purpose and independent (`training`, `initialization`, `data_generation`, `search`, `noise`), implemented via `numpy.random.default_rng(SeedSequence(...).spawn(...))` in `src/utils/reproducibility.py`, never via the legacy global `numpy.random` state.
- The pilot in this delivery uses 3 seeds only (explicitly labeled as insufficient for any final claim, sufficient only to test that seed isolation and checkpoint/resume mechanics work).

---

## 7. What is genuinely delivered in this sandbox session

1. A working, tested implementation of every **mathematical** object in the manuscript (§3): $A_{\mathrm{spec}}$ (exact + estimator), $A_{\mathrm{top}}^{\mathrm{bin}}$, $A_{\mathrm{top}}^{\mathrm{w}}$ (exact small-graph + DP-approximate), $A_+$, $A_\times$, $A_{\min}$ with the corrected bounded penalty form, all bound-checked by `pytest`.
2. Reproducibility infrastructure: seed registry, provenance/config hashing, calibration/evaluation split assertions, atomic checkpoint save/resume.
3. A real (not fabricated) smoke-scale pilot: 12 toy circuits (3 qubits) on the periodic task family, exact vs. estimated $A_{\mathrm{spec}}$ compared, exact vs. DP-approximate $A_{\mathrm{top}}^{\mathrm{w}}$ compared, 3 seeds of partial training, with results saved as raw per-seed JSON and explicitly labeled `PILOT — NOT EVIDENCE FOR H1`.
4. All Phase-0-required planning artifacts: this document, the dependency graph, the YAML config schema, the checkpoint report format.

## 8. What is explicitly NOT delivered (and why)

The full N≥150×3 central experiment, cross-task transfer, architecture search, noise-robustness sweep, and computational-cost campaign are **not executed**, because they require a multi-day-to-multi-week batch compute allocation that does not exist in an interactive sandbox turn. Building the code for these (interfaces, configs, orchestration scripts) without running them is completed; running them and reporting statistically defensible numbers is future work requiring an explicit resourcing decision from the researcher, consistent with §51 of the brief ("STOP and flag it" rather than silently producing partial or fabricated results).
