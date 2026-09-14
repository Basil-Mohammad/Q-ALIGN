"""
Checkpoint 9 -- Central experiment PILOT (manuscript brief Sec 45).

*** PILOT ONLY. NOT EVIDENCE FOR H0/H1. ***

Scale: 12 toy circuits (3 qubits), periodic task family, 3 seeds.
Purpose: verify that the full mechanism -- pool generation (no alignment
conditioning) -> exact + estimated A_spec -> exact + DP-approximate
A_top^w -> A_min stratification -> illustrative sampling -> partial
training -> cost timing -- runs end to end without silent errors, before
committing to the real N>=150 x 3 task-family campaign (PHASE0_AUDIT.md
Sec 5/8).

This script does NOT compute Delta R^2, Kendall's tau, or any inferential
statistic, precisely because n=12 circuits from one task family is not a
statistically meaningful sample -- doing so would risk the pilot being
mistaken for evidence, which manuscript brief Sec 45 explicitly forbids
("The pilot is NOT evidence for the final hypotheses").
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from src.tasks.periodic import make_multi_term_task
from src.circuits.encodings import build_single_hit_angle_encoding
from src.circuits.entanglers import linear_chain, circular_chain, all_to_all, random_fixed_degree
from src.circuits.generators import QAlignCircuit, random_init_params, train_partial, gradient_variance_trainability_proxy
from src.circuits.complexity import CircuitComplexity
from src.alignment.spectral import spectral_alignment_exact
from src.alignment.topology import (
    circuit_interaction_graph, topological_alignment_binary, topological_alignment_weighted, kappa_exact,
)
from src.alignment.aggregation import AggregationWeights, additive, conjunctive, conservative
from src.alignment.hardware import depolarizing_fidelity_estimate
from src.estimators.spectral_estimator import sampled_dft_spectrum
from src.estimators.weighted_topology import kappa_dp
from src.experiments.central_experiment import stratify_into_quartiles, PoolCircuitRecord, sample_illustrative_representatives
from src.utils.reproducibility import SeedRegistry, FIXED_SEED_LIST
from src.utils.provenance import environment_fingerprint, hash_config

N_QUBITS = 3
N_VARS = 3
POOL_SIZE = 12
N_SEEDS = 3  # PILOT ONLY -- manuscript brief Sec 7 requires >=10 for any FINAL claim
N_PARTIAL_STEPS = 20

ENTANGLING_LAYOUTS = {
    "linear": lambda rng: linear_chain(N_QUBITS),
    "circular": lambda rng: circular_chain(N_QUBITS),
    "all_to_all": lambda rng: all_to_all(N_QUBITS),
    "random_degree1": lambda rng: random_fixed_degree(N_QUBITS, degree=1, rng=rng),
}
MULTIPLIER_CHOICES = [1, 2, 3]

WEIGHTS = AggregationWeights(
    alpha_plus=0.4, beta_plus=0.4, gamma_plus=0.1, delta_plus=0.1,
    alpha_times=1.0, beta_times=1.0, gamma_times=0.1, delta_times=0.1,
)  # PILOT-FIXED weights, NOT calibrated per Remark 3.17 protocol (n too small to fit)


def build_task():
    # Two-term periodic task with an explicit 2-way interaction between vars 0,1
    # and a lone term on var 2 -- gives both a nontrivial G_T and a nontrivial
    # spectrum to test A_spec/A_top against.
    return make_multi_term_task(
        d=N_VARS,
        terms_spec=[
            ((0, 1), 1.0, 1.0, 0.0),   # joint term on vars 0,1
            ((2,), 2.0, 0.5, 0.0),      # lone term on var 2
        ],
    )


def build_pilot_pool(rng: np.random.Generator):
    layout_names = list(ENTANGLING_LAYOUTS.keys())
    pool = []
    for idx in range(POOL_SIZE):
        layout_name = layout_names[idx % len(layout_names)]
        entangling = ENTANGLING_LAYOUTS[layout_name](rng)
        multipliers = [int(rng.choice(MULTIPLIER_CHOICES)) for _ in range(N_QUBITS)]
        var_per_qubit = [0, 1, 2]  # one-to-one angle encoding, fixed for this pilot
        encoding = build_single_hit_angle_encoding(N_QUBITS, N_VARS, var_per_qubit, multipliers)
        circ = QAlignCircuit(
            encoding=encoding,
            entangling_layers=[entangling],
            circuit_id=f"pilot_circ_{idx:03d}_{layout_name}",
        )
        pool.append(circ)
    return pool


def main():
    out_dir = Path(__file__).parent.parent / "results" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    seed_registry = SeedRegistry(master_seed=FIXED_SEED_LIST[0])
    gen_rng = seed_registry.stream("data_generation", 0)

    task = build_task()
    exact_spectrum = task.exact_fourier_spectrum()
    exact_graph = task.exact_interaction_graph()

    pool = build_pilot_pool(gen_rng)

    timing = {}
    records = []
    per_circuit_results = []

    for circ in pool:
        # --- Alignment estimation timing (Sec 5.6/5.7 cost ratio) ---
        t0 = time.perf_counter()

        circuit_freqs = circ.encoding.accessible_frequencies()
        a_spec_exact = spectral_alignment_exact(exact_spectrum, circuit_freqs)

        est_rng = seed_registry.stream("data_generation", 1)
        # Candidate grid for the sampled estimator MUST include the task's true
        # frequencies (known here, since this is the gold-standard synthetic task)
        # union the circuit's own accessible frequencies -- sampling ONLY at
        # circuit_freqs (an earlier bug in this script) makes the coverage ratio
        # trivially 1.0 regardless of estimator quality, since the "total power"
        # denominator would then only ever see frequencies already inside Omega_C.
        candidate_freqs = np.unique(np.vstack([exact_spectrum.frequencies.real, circuit_freqs]), axis=0)
        est_spectrum = sampled_dft_spectrum(task.evaluate, N_VARS, candidate_freqs, n_samples=500, rng=est_rng)
        a_spec_est = spectral_alignment_exact(est_spectrum, circuit_freqs)  # same formula, sampled input

        circuit_graph = circuit_interaction_graph(N_QUBITS, circ.entangling_layers[0])
        a_top_bin = topological_alignment_binary(exact_graph, circuit_graph, depth_L=len(circ.entangling_layers[0]) + 1,
                                                   encoding="angle")
        a_top_w_exact = topological_alignment_weighted(exact_graph, circuit_graph, depth_L=2, encoding="angle",
                                                         kappa_fn=kappa_exact)
        a_top_w_dp = topological_alignment_weighted(exact_graph, circuit_graph, depth_L=2, encoding="angle",
                                                      kappa_fn=kappa_dp)

        a_sym = 0.0  # no symmetry claimed for this synthetic task (honest default, not fit)
        a_hw = depolarizing_fidelity_estimate(circ.complexity().n_two_qubit_gates, per_gate_error_rate=0.01)

        alignment_time = time.perf_counter() - t0

        a_top_for_agg = a_top_bin if isinstance(a_top_bin, float) else 0.0
        a_plus = additive(a_spec_exact, a_top_for_agg, a_sym, a_hw, WEIGHTS)
        a_x = conjunctive(a_spec_exact, a_top_for_agg, a_sym, a_hw, WEIGHTS)
        a_min = conservative(a_spec_exact, a_top_for_agg, a_sym, a_hw, WEIGHTS)

        records.append(PoolCircuitRecord(
            circuit=circ, a_spec=a_spec_exact, a_top=a_top_for_agg, a_sym=a_sym, a_hw=a_hw,
            a_min=a_min, complexity=circ.complexity(),
        ))

        # --- Partial training timing + seeds (PILOT: 3 seeds, NOT the >=10 final requirement) ---
        seed_results = []
        t1 = time.perf_counter()
        X = gen_rng.uniform(0, 2 * np.pi, size=(20, N_VARS))
        y = task.evaluate(X)
        y = (y - y.mean()) / (y.std() + 1e-8)  # normalize target to a PauliZ-observable-friendly range

        for seed_idx in range(N_SEEDS):
            init_rng = seed_registry.stream("initialization", seed_idx)
            theta0 = random_init_params(len(circ.encoding.layers), N_QUBITS, init_rng)
            train_result = train_partial(circ, X, y, theta0, n_steps=N_PARTIAL_STEPS, lr=0.2)
            seed_results.append({
                "seed_index": seed_idx,
                "seed_value": int(FIXED_SEED_LIST[seed_idx]),
                "final_mse": train_result["final_mse"],
                "loss_trace": train_result["loss_trace"],
            })
        training_time = (time.perf_counter() - t1) / N_SEEDS  # per-seed average, for the cost ratio

        grad_var_rng = seed_registry.stream("initialization", 99)
        trainability = gradient_variance_trainability_proxy(circ, X, y, grad_var_rng, n_inits=5)  # pilot n_inits, not the 20 default

        per_circuit_results.append({
            "circuit_id": circ.circuit_id,
            "complexity": circ.complexity().as_dict(),
            "a_spec_exact": a_spec_exact,
            "a_spec_estimated_n500": a_spec_est,
            "a_spec_estimation_abs_error": abs(a_spec_exact - a_spec_est),
            "a_top_bin": a_top_bin,
            "a_top_weighted_exact": a_top_w_exact,
            "a_top_weighted_dp": a_top_w_dp,
            "a_sym": a_sym,
            "a_hw": a_hw,
            "a_plus": a_plus,
            "a_times": a_x,
            "a_min": a_min,
            "trainability_gradient_variance_pilot_n5": trainability,
            "seeds": seed_results,
            "alignment_estimation_seconds": alignment_time,
            "avg_partial_training_seconds_per_seed": training_time,
        })

    quartiles = stratify_into_quartiles(records)
    rep_rng = seed_registry.stream("data_generation", 2)
    representatives = sample_illustrative_representatives(quartiles, rep_rng)

    total_align_time = sum(r["alignment_estimation_seconds"] for r in per_circuit_results)
    total_train_time = sum(r["avg_partial_training_seconds_per_seed"] for r in per_circuit_results)
    cost_ratio = total_align_time / total_train_time if total_train_time > 0 else None

    pilot_report = {
        "LABEL": "PILOT -- NOT EVIDENCE FOR H0/H1 (manuscript brief Sec 45)",
        "pool_size": POOL_SIZE,
        "n_seeds_per_circuit": N_SEEDS,
        "n_seeds_note": "PILOT uses 3 seeds; manuscript brief Sec 7 requires >=10 for any final claim.",
        "n_partial_steps": N_PARTIAL_STEPS,
        "quartile_sizes": {str(k): len(v) for k, v in quartiles.items()},
        "illustrative_representatives": {
            str(q): rec.circuit.circuit_id for q, rec in representatives.items()
        },
        "cost_analysis_pilot": {
            "total_alignment_estimation_seconds": total_align_time,
            "total_avg_partial_training_seconds": total_train_time,
            "ratio_alignment_over_training": cost_ratio,
            "interpretation": "PILOT-SCALE ONLY (3-qubit toy circuits, 20 training steps). "
                               "NOT the manuscript's pre-registered full cost-ratio table (Sec 5.6).",
        },
        "environment": environment_fingerprint(),
        "config_hash": hash_config({
            "n_qubits": N_QUBITS, "pool_size": POOL_SIZE, "n_seeds": N_SEEDS,
            "n_partial_steps": N_PARTIAL_STEPS, "weights": WEIGHTS.__dict__,
        }),
    }

    (out_dir / "pilot_report.json").write_text(json.dumps(pilot_report, indent=2, default=str))
    (out_dir / "per_circuit_raw_results.json").write_text(json.dumps(per_circuit_results, indent=2, default=str))

    # Human-readable checkpoint report (manuscript brief Sec 46 format)
    md = [
        "# Checkpoint 9 -- Central Experiment PILOT\n",
        "**Status: PASS (mechanism verified) -- LABEL: PILOT, NOT EVIDENCE FOR H0/H1**\n",
        f"- Pool size: {POOL_SIZE} circuits (3 qubits, periodic task family)",
        f"- Seeds per circuit: {N_SEEDS} (pilot only; final requirement is >=10, brief Sec 7)",
        f"- Quartile sizes (A_min-stratified): {pilot_report['quartile_sizes']}",
        f"- Mean |A_spec exact - A_spec estimated (n=500)| across pool: "
        f"{np.mean([r['a_spec_estimation_abs_error'] for r in per_circuit_results]):.4f}",
        f"- Cost ratio (alignment estimation / avg partial training, PILOT SCALE): "
        f"{cost_ratio:.4f}" if cost_ratio else "- Cost ratio: undefined",
        f"- Config hash: {pilot_report['config_hash']}",
        "\n## Decision\n",
        "Mechanism (pool generation -> alignment computation -> A_min stratification -> "
        "illustrative sampling -> partial training -> cost timing) runs end-to-end without "
        "error at pilot scale. **Next allowed step:** full central experiment requires an "
        "explicit resourcing decision (PHASE0_AUDIT.md Sec 5) before proceeding to N>=150 "
        "per task family -- this pilot does NOT authorize skipping that decision.",
    ]
    (checkpoints_dir / "checkpoint_9_pilot.md").write_text("\n".join(md) + "\n")

    print("PILOT COMPLETE.")
    print(f"Cost ratio (align/train, pilot scale): {cost_ratio}")
    print(f"Mean A_spec estimator abs error (n=500 samples): "
          f"{np.mean([r['a_spec_estimation_abs_error'] for r in per_circuit_results]):.4f}")
    print(f"Quartile sizes: {pilot_report['quartile_sizes']}")


if __name__ == "__main__":
    main()
