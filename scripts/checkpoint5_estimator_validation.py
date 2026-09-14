"""
Checkpoint 5 -- Estimator bias/variance validation (manuscript Sec 3.6 /
brief Sec 14).

Runs the FULL pre-registered sample-size sweep (100, 250, 500, 1000, 2500,
5000) against the periodic task's KNOWN closed-form spectrum (gold
standard), for both the spectral estimator (A_spec) and the weighted-
topology DP/spectral approximations (A_top^w).

Pass/fail criteria (EstimatorValidationCriterion) are defined BEFORE this
script is run (brief Sec 14: "Define the estimator accuracy criterion
before downstream experiments. Do not select the sample size after seeing
which one gives the strongest downstream result.") -- they live in
src/experiments/estimator_validation.py, not tuned here.

Run: PYTHONPATH=. python scripts/checkpoint5_estimator_validation.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from src.tasks.periodic import make_multi_term_task
from src.alignment.topology import circuit_interaction_graph
from src.experiments.estimator_validation import (
    validate_spectral_estimator, validate_weighted_topology_estimator, EstimatorValidationCriterion
)
from src.utils.reproducibility import SeedRegistry, FIXED_SEED_LIST
from src.utils.provenance import environment_fingerprint, hash_config

# Pre-registered sample-size sweep (brief Sec 14, exact values given).
SAMPLE_SIZES = [100, 250, 500, 1000, 2500, 5000]
N_REPEATS = 30  # independent repeats per sample size, for bias/variance estimation

# Pre-registered pass/fail criterion, defined BEFORE running (brief Sec 14).
CRITERION = EstimatorValidationCriterion(max_relative_bias=0.10, max_rmse=0.10)


def build_gold_standard_task():
    # Same two-term task used in the pilot, for continuity/comparability.
    return make_multi_term_task(
        d=3,
        terms_spec=[
            ((0, 1), 1.0, 1.0, 0.0),
            ((2,), 2.0, 0.5, 0.0),
        ],
    )


def main():
    out_dir = Path(__file__).parent.parent / "results" / "checkpoint5"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    seed_registry = SeedRegistry(master_seed=FIXED_SEED_LIST[1])  # distinct master seed from the pilot
    rng = seed_registry.stream("data_generation", 10)

    task = build_gold_standard_task()

    # Candidate frequency grid: task's true frequencies plus a surrounding
    # integer grid (what a real circuit search would plausibly reach),
    # fixed BEFORE running the sweep.
    exact_freqs = task.exact_fourier_spectrum().frequencies.real
    grid_1d = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    grid = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")).reshape(3, -1).T
    candidate_frequencies = np.unique(np.vstack([exact_freqs, grid]), axis=0)

    print(f"Running spectral estimator validation: {len(candidate_frequencies)} candidate frequencies, "
          f"sample sizes {SAMPLE_SIZES}, {N_REPEATS} repeats each...")
    spectral_report = validate_spectral_estimator(
        task, candidate_frequencies, SAMPLE_SIZES, N_REPEATS, rng, CRITERION
    )

    # Weighted-topology DP/spectral-approx validation on a small graph
    # (exact enumeration is only tractable at small scale -- brief Sec 13).
    small_graph = circuit_interaction_graph(5, [(0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.6), (0, 4, 0.3)])
    all_pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    topology_report = validate_weighted_topology_estimator(small_graph, all_pairs, max_len=3)

    full_report = {
        "checkpoint": 5,
        "criterion": {"max_relative_bias": CRITERION.max_relative_bias, "max_rmse": CRITERION.max_rmse},
        "sample_sizes_tested": SAMPLE_SIZES,
        "n_repeats_per_size": N_REPEATS,
        "spectral_estimator_validation": spectral_report,
        "weighted_topology_estimator_validation": topology_report,
        "environment": environment_fingerprint(),
        "config_hash": hash_config({
            "sample_sizes": SAMPLE_SIZES, "n_repeats": N_REPEATS,
            "criterion": {"max_relative_bias": CRITERION.max_relative_bias, "max_rmse": CRITERION.max_rmse},
        }),
    }
    (out_dir / "checkpoint5_report.json").write_text(json.dumps(full_report, indent=2, default=str))

    status = "PASS" if spectral_report["passed_at_largest_n"] else "FAIL"

    md = [
        "# Checkpoint 5 -- Estimator Bias/Variance Validation\n",
        f"**Status: {status}**\n",
        f"- Pre-registered criterion: max relative bias <= {CRITERION.max_relative_bias}, "
        f"max RMSE <= {CRITERION.max_rmse} (defined BEFORE running this sweep)",
        f"- Sample sizes tested: {SAMPLE_SIZES}",
        f"- Repeats per size: {N_REPEATS}",
        "\n## Spectral estimator (A_spec) bias/variance by sample size\n",
        "| n_samples | bias | variance | RMSE | relative bias |",
        "|---|---|---|---|---|",
    ]
    for n in SAMPLE_SIZES:
        r = spectral_report["per_sample_size"][n]
        md.append(f"| {n} | {r['bias']:.4f} | {r['variance']:.6f} | {r['rmse']:.4f} | "
                   f"{r['relative_bias']:.4f} |" if r['relative_bias'] is not None else
                   f"| {n} | {r['bias']:.4f} | {r['variance']:.6f} | {r['rmse']:.4f} | N/A |")
    md.append(f"\n**Result at largest tested n={spectral_report['largest_n_tested']}: "
              f"{'PASSES' if spectral_report['passed_at_largest_n'] else 'FAILS'} the pre-registered criterion.**\n")

    md.append("\n## Weighted-topology DP/spectral approximation vs. exact (small graph, 5 qubits)\n")
    md.append(f"- Pairs tested: {topology_report['n_pairs']}")
    md.append(f"- DP approximation: mean abs error = {topology_report['dp_vs_exact']['mean_abs_error']:.4f}, "
              f"Kendall's tau vs exact = {topology_report['dp_vs_exact']['kendall_tau_vs_exact']}")
    md.append(f"- Spectral approximation: mean abs error = {topology_report['spectral_vs_exact']['mean_abs_error']:.4f}, "
              f"Kendall's tau vs exact = {topology_report['spectral_vs_exact']['kendall_tau_vs_exact']}")

    md.append(f"\n## Decision\n")
    if status == "PASS":
        md.append("Spectral estimator meets the pre-registered accuracy criterion at the largest tested "
                   "sample size. **This gate is now open**: estimated A_spec may be used in downstream "
                   "experiments AT OR ABOVE this validated sample size, for tasks with similar spectral "
                   "sparsity to this gold-standard task -- NOT unconditionally for arbitrary tasks.")
    else:
        md.append("Spectral estimator does NOT meet the pre-registered criterion even at the largest "
                   "tested sample size. **This gate remains CLOSED**: estimated A_spec must NOT be used "
                   "in any downstream experiment until either (a) a larger sample size is validated, or "
                   "(b) a better estimator is implemented. This is a legitimate stopping condition, not "
                   "something to route around.")

    (checkpoints_dir / "checkpoint_5_estimator_validation.md").write_text("\n".join(md) + "\n")

    print(f"\nCheckpoint 5: {status}")
    print(json.dumps({n: spectral_report["per_sample_size"][n] for n in SAMPLE_SIZES}, indent=2))


if __name__ == "__main__":
    main()
