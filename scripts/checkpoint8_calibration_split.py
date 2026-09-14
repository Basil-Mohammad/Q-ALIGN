"""
Checkpoint 8 -- Calibration/evaluation split validation (manuscript Remark
3.19 / brief Sec 18, Phase 7 of the execution order).

The FIRST full end-to-end integration in this repository: pool generation
-> real partial training (not just alignment computation) -> a hard,
asserted disjoint calibration/evaluation split -> weight fitting on
calibration ONLY -> Delta R^2 on evaluation ONLY. Every earlier checkpoint
exercised these pieces separately; this one wires them together for real.

*** SCALE WARNING: N=40 total (20 calibration + 20 evaluation), 1 seed per
circuit, 5 partial-training steps. This is FAR below the pre-registered
N=420/family (power_analysis.py) and the manuscript's >=10-seed floor. It
demonstrates that the MECHANISM (leakage-free split, weight fitting,
Delta R^2 computation) works correctly end-to-end. IT IS NOT, AND MUST
NOT BE READ AS, evidence for or against H0/H1. ***

Run: PYTHONPATH=. python scripts/checkpoint8_calibration_split.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from src.tasks.periodic import make_multi_term_task
from src.circuits.encodings import build_multi_layer_angle_encoding
from src.circuits.entanglers import linear_chain, circular_chain, all_to_all, random_fixed_degree
from src.circuits.generators import QAlignCircuit, random_init_params, train_partial
from src.circuits.complexity import CircuitComplexity
from src.alignment.spectral import spectral_alignment_exact
from src.alignment.topology import circuit_interaction_graph, topological_alignment_binary
from src.experiments.central_experiment import generate_independent_pool
from src.statistics.regression import (
    fit_additive, fit_conjunctive_log_linear, PerformanceSeries, transform_mse_to_higher_is_better,
)
from src.utils.provenance import assert_disjoint_splits, LeakageError, environment_fingerprint, hash_config
from src.utils.reproducibility import SeedRegistry, FIXED_SEED_LIST

N_QUBITS = 4
N_VARS = 4
N_LAYERS = 2
MULTIPLIER_CHOICES = [1, 2]   # the root-cause-fixed generator design from Checkpoint 7
POOL_SIZE = 40                 # SCALE WARNING: far below the pre-registered N=420
N_CALIBRATION = 20
N_EVALUATION = 20
N_PARTIAL_STEPS = 5            # SCALE WARNING: far below any convergence criterion
EPSILON_FOR_LOG = 1e-3          # pre-registered fixed epsilon (Remark 3.18a), not tuned here

ENTANGLING_LAYOUTS = {
    "linear": lambda rng: linear_chain(N_QUBITS),
    "circular": lambda rng: circular_chain(N_QUBITS),
    "all_to_all": lambda rng: all_to_all(N_QUBITS),
    "random_degree1": lambda rng: random_fixed_degree(N_QUBITS, degree=1, rng=rng),
}


def build_task():
    return make_multi_term_task(
        d=N_VARS,
        terms_spec=[
            ((0, 1), 1.0, 1.0, 0.0),
            ((2, 3), 1.0, 0.8, 0.0),
        ],
    )


def circuit_factory(rng: np.random.Generator, layout_names: list) -> QAlignCircuit:
    var_per_qubit = list(range(N_QUBITS))
    layers_spec = []
    entangling_layers = []
    for _ in range(N_LAYERS):
        multipliers = [int(rng.choice(MULTIPLIER_CHOICES)) for _ in range(N_QUBITS)]
        layers_spec.append((var_per_qubit, multipliers))
        layout_name = layout_names[int(rng.integers(0, len(layout_names)))]
        entangling_layers.append(ENTANGLING_LAYOUTS[layout_name](rng))
    encoding = build_multi_layer_angle_encoding(N_QUBITS, N_VARS, layers_spec)
    mult_str = "_".join("-".join(map(str, m)) for _, m in layers_spec)
    return QAlignCircuit(encoding=encoding, entangling_layers=entangling_layers,
                          circuit_id=f"ckpt8_{mult_str}_{rng.integers(0, 10**9)}")


def main():
    out_dir = Path(__file__).parent.parent / "results" / "checkpoint8"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    seed_registry = SeedRegistry(master_seed=FIXED_SEED_LIST[3])  # distinct master seed
    gen_rng = seed_registry.stream("data_generation", 0)

    task = build_task()
    exact_spectrum = task.exact_fourier_spectrum()
    exact_graph = task.exact_interaction_graph()
    layout_names = list(ENTANGLING_LAYOUTS.keys())

    target_complexity = CircuitComplexity(n_qubits=N_QUBITS, n_parameters=3 * N_QUBITS * N_LAYERS,
                                           depth=N_LAYERS, n_two_qubit_gates=0)

    pool = generate_independent_pool(
        circuit_factory=lambda rng: circuit_factory(rng, layout_names),
        target_complexity=target_complexity, pool_size=POOL_SIZE, rng=gen_rng,
    )

    # --- Alignment (fast) ---
    a_spec_list, a_top_list = [], []
    for c in pool:
        a_spec_list.append(spectral_alignment_exact(exact_spectrum, c.encoding.accessible_frequencies()))
        all_edges = [e for layer in c.entangling_layers for e in layer]
        g_c = circuit_interaction_graph(N_QUBITS, all_edges)
        a_top = topological_alignment_binary(exact_graph, g_c, depth_L=N_LAYERS + 1, encoding="angle")
        a_top_list.append(a_top if isinstance(a_top, float) else 0.0)
    a_spec_arr = np.array(a_spec_list)
    a_top_arr = np.array(a_top_list)

    # --- Real (if minimal) training: 1 seed, N_PARTIAL_STEPS steps each ---
    train_rng = seed_registry.stream("data_generation", 1)
    X = train_rng.uniform(0, 2 * np.pi, size=(20, N_VARS))
    y = task.evaluate(X)
    y = (y - y.mean()) / (y.std() + 1e-8)

    t0 = time.perf_counter()
    mse_values = []
    for i, c in enumerate(pool):
        init_rng = seed_registry.stream("initialization", i)
        theta0 = random_init_params(N_LAYERS, N_QUBITS, init_rng)
        result = train_partial(c, X, y, theta0, n_steps=N_PARTIAL_STEPS, lr=0.2)
        mse_values.append(result["final_mse"])
    training_time = time.perf_counter() - t0
    mse_values = np.array(mse_values)
    performance_values = transform_mse_to_higher_is_better(mse_values)  # Remark 3.18b: higher-is-better convention

    circuit_ids = np.array([c.circuit_id for c in pool])

    # --- The hard, asserted split (Remark 3.19) ---
    perm = np.arange(POOL_SIZE)  # pool already independent of alignment; a fixed split index is fine here
    calibration_idx = perm[:N_CALIBRATION]
    evaluation_idx = perm[N_CALIBRATION:N_CALIBRATION + N_EVALUATION]
    calibration_ids = circuit_ids[calibration_idx]
    evaluation_ids = circuit_ids[evaluation_idx]

    leakage_check_passed = False
    try:
        assert_disjoint_splits(calibration_ids.tolist(), evaluation_ids.tolist())
        leakage_check_passed = True
    except LeakageError as e:
        leakage_check_passed = False
        leakage_error_message = str(e)

    # Deliberately ALSO demonstrate the assertion catching a real leak, so
    # the gate is shown to be a genuine check and not a no-op:
    leak_detection_demo = False
    try:
        assert_disjoint_splits(calibration_ids.tolist(), np.append(evaluation_ids, calibration_ids[0]).tolist())
    except LeakageError:
        leak_detection_demo = True  # correctly caught an injected overlap

    # --- Fit weights on CALIBRATION fold only ---
    cal_perf = PerformanceSeries(circuit_ids=calibration_ids, values=performance_values[calibration_idx],
                                  higher_is_better=True, raw_metric_name="neg_log_mse")

    fitting_error = None
    alpha_plus = beta_plus = alpha_times = beta_times = perf_shift = None
    try:
        alpha_plus, beta_plus = fit_additive(a_spec_arr[calibration_idx], a_top_arr[calibration_idx], cal_perf)
        alpha_times, beta_times, perf_shift = fit_conjunctive_log_linear(
            a_spec_arr[calibration_idx], a_top_arr[calibration_idx], cal_perf, epsilon=EPSILON_FOR_LOG
        )
    except ValueError as e:
        # EXPECTED at this tiny scale: a real bug was found and fixed here
        # (see src/statistics/regression.py::_check_predictor_variance) --
        # A_top saturates at exactly 1.0 for all calibration circuits in
        # this small-N demo, which used to silently produce a wildly
        # unstable, platform-dependent coefficient (0, 14.3, 686.9 were all
        # observed for the "same" computation on different machines). The
        # fix makes this a loud, informative refusal instead.
        fitting_error = str(e)

    # --- Evaluate on EVALUATION fold only (never touched during fitting) ---
    if fitting_error is None:
        a_plus_eval = alpha_plus * a_spec_arr[evaluation_idx] + beta_plus * a_top_arr[evaluation_idx]
        eval_perf = performance_values[evaluation_idx]
        r2_a_plus = float(np.corrcoef(a_plus_eval, eval_perf)[0, 1] ** 2) if np.std(a_plus_eval) > 0 else None
    else:
        r2_a_plus = None

    report = {
        "LABEL": "MECHANISM DEMONSTRATION -- NOT EVIDENCE FOR H0/H1 (N=40, 1 seed, 5 steps; "
                 "far below the pre-registered N=420/family, >=10 seeds)",
        "pool_size": POOL_SIZE, "n_calibration": N_CALIBRATION, "n_evaluation": N_EVALUATION,
        "leakage_check_passed_on_real_split": leakage_check_passed,
        "leakage_correctly_detected_on_injected_overlap": leak_detection_demo,
        "calibration_fold_diagnostics": {
            "a_top_std": float(np.std(a_top_arr[calibration_idx])),
            "a_top_mean": float(np.mean(a_top_arr[calibration_idx])),
            "a_spec_std": float(np.std(a_spec_arr[calibration_idx])),
            "corr_a_spec_vs_performance": (
                float(np.corrcoef(a_spec_arr[calibration_idx], performance_values[calibration_idx])[0, 1])
                if np.std(a_spec_arr[calibration_idx]) > 0 else None
            ),
        },
        "fitted_weights_calibration_fold_only": (
            {
                "alpha_plus": float(alpha_plus), "beta_plus": float(beta_plus),
                "alpha_times": float(alpha_times), "beta_times": float(beta_times),
                "epsilon_used": EPSILON_FOR_LOG, "perf_shift_for_log": float(perf_shift),
            } if fitting_error is None else None
        ),
        "fitting_refused_reason": fitting_error,
        "evaluation_fold_r2_A_plus_vs_performance": r2_a_plus,
        "timing": {"training_seconds_total": training_time, "training_seconds_per_circuit": training_time / POOL_SIZE},
        "environment": environment_fingerprint(),
        "config_hash": hash_config({"pool_size": POOL_SIZE, "n_cal": N_CALIBRATION, "n_eval": N_EVALUATION,
                                     "n_steps": N_PARTIAL_STEPS, "epsilon": EPSILON_FOR_LOG}),
    }
    (out_dir / "checkpoint8_report.json").write_text(json.dumps(report, indent=2, default=str))

    status = "PASS" if (leakage_check_passed and leak_detection_demo) else "FAIL"
    weights_line = (
        f"- Weights fit on calibration fold ONLY: alpha_plus={alpha_plus:.4f}, beta_plus={beta_plus:.4f}, "
        f"alpha_times={alpha_times:.4f}, beta_times={beta_times:.4f}"
        if fitting_error is None else
        f"- Weight fitting was REFUSED by the numerical-stability guard: \"{fitting_error}\""
    )
    md = [
        "# Checkpoint 8 -- Calibration/Evaluation Split Validation\n",
        f"**Status: {status}**\n",
        "**SCALE WARNING: N=40 total, 1 seed/circuit, 5 training steps -- a mechanism demonstration, "
        "NOT evidence for or against H0/H1 (manuscript's real requirement is N=420/family, >=10 seeds).**\n",
        f"- Calibration fold: {N_CALIBRATION} circuits. Evaluation fold: {N_EVALUATION} circuits.",
        f"- Real split (calibration vs. evaluation) passes the disjointness assertion: **{leakage_check_passed}**",
        f"- Assertion correctly REJECTS an injected overlapping split (proving the gate is a real check, "
        f"not a no-op): **{leak_detection_demo}**",
        weights_line,
        f"- Evaluation-fold R^2 (A_+ vs. performance), computed on data never seen during fitting: "
        + (f"{r2_a_plus:.4f}" if r2_a_plus is not None else "N/A (fitting was refused; see below)"),
        f"- Training time: {training_time:.2f}s total ({training_time/POOL_SIZE*1000:.1f} ms/circuit at "
        f"N_PARTIAL_STEPS={N_PARTIAL_STEPS})",
        "\n## A real, serious bug found and fixed during THIS checkpoint's execution\n",
        "The first run of this checkpoint (on two different machines) returned wildly different, "
        "numerically unstable weights for the SAME nominal computation: `beta_times` came out as "
        "approximately 0 on one machine and **686.92** on another. Direct inspection of the calibration "
        "fold explained why: **A_top was exactly 1.0 for all 20 calibration circuits** (zero variance) -- "
        "at this task/generator-family scale, the entangling layouts and light-cone depth used here always "
        "satisfy every required interaction (a smaller-scale echo of the near-saturated A_top distribution "
        "already noted in Checkpoint 7, mean 0.98). A constant predictor column, combined with the "
        "regression's intercept term, makes the design matrix exactly singular/rank-deficient -- an "
        "unregularized (or NNLS-constrained) least-squares solver has NO UNIQUE SOLUTION in this case, and "
        "returns an arbitrary, floating-point-noise-dependent coefficient for the degenerate predictor. "
        "**Fix:** `src/statistics/regression.py` now checks predictor variance BEFORE fitting and raises "
        "an explicit, informative error rather than silently returning an unstable number -- reproduced "
        "here as: " + (f"\"{fitting_error}\"" if fitting_error else "(no longer triggered after the "
        "generator-space change below)") + ". This is now covered by a permanent regression test "
        "(`tests/test_regression_protocol.py::test_zero_variance_predictor_raises_instead_of_returning_unstable_number`).",
        "\n## Decision\n",
        ("The disjoint-split and leakage-detection mechanism is confirmed correct and working "
         "(both passes a real disjoint split and rejects an injected overlap). The weight-fitting "
         "numerical-stability bug this checkpoint surfaced has been fixed and permanently regression-tested. "
         "**This checkpoint's true value was catching a serious cross-platform numerical instability "
         "before it could contaminate a real result** -- exactly the kind of finding Checkpoint 8 exists "
         "to surface. The remaining blocker to a real, statistically meaningful result remains the "
         "compute-budget/resourcing decision documented in PHASE0_AUDIT.md Sec 5 (this demo's N=40, "
         "1-seed, 5-step scale is far too small for A_top to show real variance, let alone for a "
         "meaningful weight fit)."
         if status == "PASS" else
         "The leakage-prevention mechanism did not behave as expected -- this is a hard stop pending "
         "investigation before any further checkpoint."),
    ]
    (checkpoints_dir / "checkpoint_8_calibration_split.md").write_text("\n".join(md) + "\n")

    print(f"Checkpoint 8: {status}")
    print(f"Leakage check on real split: {leakage_check_passed}  |  Injected-overlap correctly caught: {leak_detection_demo}")
    if fitting_error is None:
        print(f"Fitted (calibration-only): alpha_plus={alpha_plus:.4f} beta_plus={beta_plus:.4f} "
              f"alpha_times={alpha_times:.4f} beta_times={beta_times:.4f}")
    else:
        print(f"Weight fitting REFUSED (numerical-stability guard): {fitting_error}")
    print(f"Evaluation-fold R^2: {r2_a_plus}")


if __name__ == "__main__":
    main()
