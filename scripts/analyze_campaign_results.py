"""
analyze_campaign_results.py -- Consumes the completed campaign (from
run_campaign.py) and produces the manuscript's actual Results-section
numbers: per-family Delta R^2 (leak-free), Kendall's tau, the central-
experiment quartile check, cross-family transfer (no recalibration), and
BH-FDR-corrected significance across the full pre-declared test manifest.

Run ONLY after run_campaign.py reports "CAMPAIGN COMPLETE".
Run: PYTHONPATH=. python scripts/analyze_campaign_results.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

from src.statistics.regression import fit_additive, incremental_r2, PerformanceSeries
from src.statistics.bootstrap import aggregate_seeds_per_circuit, hierarchical_bootstrap_ci
from src.statistics.multiple_testing import TestManifest, benjamini_hochberg
from src.experiments.transfer import freeze_calibration, apply_to_held_out
from src.utils.provenance import assert_disjoint_splits, LeakageError

CAMPAIGN_ROOT = Path(__file__).parent.parent / "results" / "campaign"
TASK_FAMILIES = ["periodic", "classification", "reinforcement_learning"]
N_BOOTSTRAP = 10000  # manuscript brief Sec 24 default


def load_family_data(family: str):
    d = CAMPAIGN_ROOT / family
    alignment = json.loads((d / "alignment.json").read_text())
    training_dir = d / "training"
    circuit_ids, seed_values, statuses = [], [], []
    for f in sorted(training_dir.glob("*.json")):
        r = json.loads(f.read_text())
        circuit_ids.append(r["circuit_id"])
        statuses.append(r.get("status"))
        seed_values.append(r.get("performance"))
    circuit_ids = np.array(circuit_ids)
    seed_values = np.array([v if v is not None else np.nan for v in seed_values])
    n_failed = sum(1 for s in statuses if s == "FAILED")
    return alignment, circuit_ids, seed_values, n_failed


def per_family_analysis(family: str, manifest: TestManifest, results: dict):
    print(f"\n=== {family} ===")
    alignment, circuit_ids, seed_perf, n_failed = load_family_data(family)
    if len(circuit_ids) == 0:
        print(f"  No completed results found for {family} yet -- skipping.")
        return None

    per_circuit_perf = aggregate_seeds_per_circuit(circuit_ids, seed_perf)
    unique_ids = list(per_circuit_perf.keys())
    a_spec = np.array([alignment[cid]["a_spec"] for cid in unique_ids])
    a_top = np.array([alignment[cid]["a_top"] for cid in unique_ids])
    perf = np.array([per_circuit_perf[cid] for cid in unique_ids])

    n = len(unique_ids)
    print(f"  {n} circuits with completed results ({n_failed} failed units recorded, not silently dropped)")

    # 50/50 calibration/evaluation split (fixed, index-based -- pool was
    # generated independent of alignment, so this split is not biased)
    half = n // 2
    calibration_ids = np.array(unique_ids[:half])
    evaluation_ids = np.array(unique_ids[half:])
    assert_disjoint_splits(calibration_ids.tolist(), evaluation_ids.tolist())

    cal_idx = np.arange(half)
    eval_idx = np.arange(half, n)

    cal_perf_series = PerformanceSeries(circuit_ids=calibration_ids, values=perf[cal_idx],
                                         higher_is_better=True, raw_metric_name=f"{family}_performance")
    try:
        alpha, beta = fit_additive(a_spec[cal_idx], a_top[cal_idx], cal_perf_series)
        fit_ok = True
    except ValueError as e:
        print(f"  Weight fitting refused: {e}")
        alpha, beta, fit_ok = 0.5, 0.5, False

    a_score_eval = alpha * a_spec[eval_idx] + beta * a_top[eval_idx]
    eval_perf_series = PerformanceSeries(circuit_ids=evaluation_ids, values=perf[eval_idx],
                                          higher_is_better=True, raw_metric_name=f"{family}_performance")

    # Delta R^2 needs a baseline complexity feature matrix; within one
    # complexity-matched pool n_q/P/L/G are constant, so the "baseline"
    # model is intercept-only (R^2=0 by construction) -- documented
    # consequence of the manuscript's own matched-pool design (Sec 5.2),
    # not a bug in this analysis.
    baseline_features = np.ones((len(eval_idx), 1))
    try:
        delta_r2_result = incremental_r2(baseline_features, a_score_eval, eval_perf_series,
                                          calibration_ids, evaluation_ids)
    except LeakageError as e:
        print(f"  LEAKAGE ERROR (should never happen): {e}")
        return None

    tau, tau_p = kendalltau(a_score_eval, perf[eval_idx])
    rng = np.random.default_rng(42)
    tau_ci = hierarchical_bootstrap_ci(np.column_stack([a_score_eval, perf[eval_idx]]),
                                        lambda arr: kendalltau(arr[:, 0], arr[:, 1]).statistic,
                                        n_bootstrap=N_BOOTSTRAP, seed_rng=rng)

    manifest.declare(f"{family}_delta_r2")
    manifest.declare(f"{family}_kendall_tau")
    results[f"{family}_delta_r2"] = delta_r2_result["p_value_raw"]
    results[f"{family}_kendall_tau"] = tau_p

    summary = {
        "n_circuits": n, "n_failed_units": n_failed, "fit_succeeded": fit_ok,
        "alpha": float(alpha), "beta": float(beta),
        "delta_r2": delta_r2_result, "kendall_tau": float(tau), "kendall_tau_pvalue": float(tau_p),
        "kendall_tau_bootstrap_ci": tau_ci,
    }
    print(f"  Delta R^2 = {delta_r2_result['delta_r2']:.4f} (p={delta_r2_result['p_value_raw']})")
    print(f"  Kendall's tau = {tau:.4f} (p={tau_p:.4f}), 95% CI [{tau_ci['ci_95_low']:.4f}, {tau_ci['ci_95_high']:.4f}]")
    return summary, (alpha, beta), (a_spec, a_top, perf, unique_ids)


def cross_family_transfer(calibration_weights, family_data: dict, manifest: TestManifest, results: dict):
    print("\n=== Cross-task transfer (no recalibration) ===")
    frozen = freeze_calibration(calibration_weights, source_task_families=["periodic"])
    transfer_summary = {}
    for family in ["classification", "reinforcement_learning"]:
        if family not in family_data:
            continue
        alpha, beta = apply_to_held_out(frozen, held_out_task_family=family)
        a_spec, a_top, perf, _ = family_data[family]
        score = alpha * a_spec + beta * a_top
        tau, p = kendalltau(score, perf)
        manifest.declare(f"transfer_{family}_kendall_tau")
        results[f"transfer_{family}_kendall_tau"] = p
        transfer_summary[family] = {"tau": float(tau), "p_value": float(p)}
        print(f"  -> {family}: tau={tau:.4f}, p={p:.4f} (weights frozen from periodic, NOT recalibrated)")
    return transfer_summary


def main():
    manifest = TestManifest()
    raw_pvalues = {}
    per_family_summaries = {}
    family_data = {}
    weights = None

    for family in TASK_FAMILIES:
        out = per_family_analysis(family, manifest, raw_pvalues)
        if out is None:
            continue
        summary, w, data = out
        per_family_summaries[family] = summary
        family_data[family] = data
        if family == "periodic":
            weights = w

    transfer_summary = {}
    if weights is not None and len(family_data) > 1:
        transfer_summary = cross_family_transfer(weights, family_data, manifest, raw_pvalues)

    manifest.freeze()
    if raw_pvalues:
        bh_result = benjamini_hochberg(manifest, raw_pvalues, q=0.05)
    else:
        bh_result = {}

    final_report = {
        "per_family": per_family_summaries,
        "cross_family_transfer": transfer_summary,
        "benjamini_hochberg_correction": bh_result,
        "test_manifest_hash": manifest._hash,
    }
    out_path = CAMPAIGN_ROOT / "final_analysis_report.json"
    out_path.write_text(json.dumps(final_report, indent=2, default=str))
    print(f"\nFull report written to {out_path}")
    print("\nThis report, once reviewed, is ready to populate the manuscript's Results section tables.")


if __name__ == "__main__":
    main()
