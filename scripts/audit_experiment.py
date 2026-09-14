"""
Automated scientific audit (manuscript brief Sec 39 / Phase 17).

Verifies the CURRENT STATE of the repository against the checkpoint
reports and raw results actually on disk -- it does not trust narrative
claims in markdown files, it re-derives PASS/FAIL from the underlying
JSON artifacts wherever possible.

Run: PYTHONPATH=. python scripts/audit_experiment.py
Exit code: 0 if PASS, 1 if FAIL.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def check(condition: bool, message: str, diagnostics: List[Dict]) -> bool:
    diagnostics.append({"check": message, "passed": bool(condition)})
    return condition


def audit_checkpoint_0(diagnostics: List[Dict]) -> bool:
    path = RESULTS_DIR.parent / "checkpoints" / "checkpoint_0_environment.json"
    if not path.exists():
        return check(False, "Checkpoint 0 JSON report exists", diagnostics)
    data = json.loads(path.read_text())
    missing = data.get("missing_packages", ["<field not found>"])
    return check(len(missing) == 0, f"Checkpoint 0: no missing packages (found missing={missing})", diagnostics)


def audit_checkpoint_5(diagnostics: List[Dict]) -> bool:
    path = RESULTS_DIR / "checkpoint5" / "checkpoint5_report.json"
    if not path.exists():
        return check(False, "Checkpoint 5 JSON report exists", diagnostics)
    data = json.loads(path.read_text())
    spectral = data.get("spectral_estimator_validation", {})
    ok1 = check(spectral.get("passed_at_largest_n") is True,
                "Checkpoint 5: spectral estimator PASSES pre-registered criterion at largest tested n",
                diagnostics)
    # Verify bias actually decreases monotonically with n (re-derive, don't trust the label alone)
    per_n = spectral.get("per_sample_size", {})
    ns = sorted(int(k) for k in per_n.keys())
    biases = [abs(per_n[str(n)]["bias"]) for n in ns]
    monotone = all(biases[i] >= biases[i + 1] - 1e-9 for i in range(len(biases) - 1))
    ok2 = check(monotone, "Checkpoint 5: |bias| decreases monotonically with sample size (re-derived from raw data)",
                diagnostics)
    return ok1 and ok2


def audit_checkpoint_7(diagnostics: List[Dict]) -> bool:
    path = RESULTS_DIR / "checkpoint7" / "checkpoint7_report.json"
    if not path.exists():
        return check(False, "Checkpoint 7 JSON report exists", diagnostics)
    data = json.loads(path.read_text())
    ok1 = check(data.get("all_circuits_complexity_matched") is True,
                "Checkpoint 7: all pooled circuits exactly match target complexity", diagnostics)
    quartiles = data.get("quartile_sizes_A_min", {})
    total = sum(quartiles.values())
    ok2 = check(total == data.get("pool_size"),
                f"Checkpoint 7: quartile counts sum to pool_size ({total} == {data.get('pool_size')})",
                diagnostics)
    n_nonempty = sum(1 for v in quartiles.values() if v > 0)
    ok3 = check(n_nonempty >= 3,
                f"Checkpoint 7: at least 3 of 4 quartiles are non-empty (found {n_nonempty}) "
                f"-- the post-fix generator design, not the original degenerate one",
                diagnostics)
    return ok1 and ok2 and ok3


def audit_checkpoint_8(diagnostics: List[Dict]) -> bool:
    path = RESULTS_DIR / "checkpoint8" / "checkpoint8_report.json"
    if not path.exists():
        return check(False, "Checkpoint 8 JSON report exists", diagnostics)
    data = json.loads(path.read_text())
    ok1 = check(data.get("leakage_check_passed_on_real_split") is True,
                "Checkpoint 8: real calibration/evaluation split is disjoint", diagnostics)
    ok2 = check(data.get("leakage_correctly_detected_on_injected_overlap") is True,
                "Checkpoint 8: leakage assertion correctly rejects an injected overlap "
                "(proves the gate is not a no-op)", diagnostics)
    # If fitting was refused, the refusal reason must be present and non-empty (no silent None-with-no-explanation).
    if data.get("fitted_weights_calibration_fold_only") is None:
        ok3 = check(bool(data.get("fitting_refused_reason")),
                    "Checkpoint 8: weight fitting was refused, and a non-empty reason is recorded "
                    "(not a silent None)", diagnostics)
    else:
        weights = data["fitted_weights_calibration_fold_only"]
        # If weights WERE fit, they must be finite and not absurdly large (regression guard against
        # the exact 686.92-style instability bug found in this project's own history).
        vals = [weights.get("alpha_plus"), weights.get("beta_plus"),
                weights.get("alpha_times"), weights.get("beta_times")]
        finite = all(v is not None and abs(v) < 100 for v in vals)
        ok3 = check(finite, f"Checkpoint 8: fitted weights are finite and within a sane magnitude "
                             f"(<100 in absolute value): {vals}", diagnostics)
    return ok1 and ok2 and ok3


def audit_checkpoint_9(diagnostics: List[Dict]) -> bool:
    path = RESULTS_DIR / "pilot" / "pilot_report.json"
    if not path.exists():
        return check(False, "Checkpoint 9 (pilot) JSON report exists", diagnostics)
    data = json.loads(path.read_text())
    ok1 = check("PILOT" in data.get("LABEL", ""),
                "Checkpoint 9: pilot report is explicitly labeled PILOT (never presented as evidence)",
                diagnostics)
    ok2 = check(data.get("n_seeds_per_circuit", 0) < 10,
                "Checkpoint 9: pilot seed count is (as documented) below the >=10 final-claim floor "
                "-- confirms this cannot be mistaken for a final result", diagnostics)
    return ok1 and ok2


def audit_reference_validation_tests_declared() -> Tuple[bool, str]:
    """Cannot re-run pytest from inside this script reliably in all
    environments, so this checks that the reference-validation test FILE
    exists and contains the four expected test function names -- a weak
    but real static check that the Checkpoint 11 tests have not been
    silently deleted after being reported as passing.
    """
    path = REPO_ROOT / "tests" / "test_reference_validation.py"
    if not path.exists():
        return False, "tests/test_reference_validation.py does not exist"
    text = path.read_text()
    required = [
        "test_benjamini_hochberg_matches_statsmodels_exactly",
        "test_incremental_r2_f_test_matches_statsmodels_anova_lm",
        "test_hierarchical_bootstrap_ci_brackets_classical_normal_approx_ci",
        "test_kendall_tau_point_estimate_and_pvalue_match_scipy_exactly",
    ]
    missing = [r for r in required if r not in text]
    if missing:
        return False, f"Missing expected reference-validation tests: {missing}"
    return True, "All 4 reference-validation tests present in file"


def audit_no_fabricated_final_claims(diagnostics: List[Dict]) -> bool:
    """Scans all checkpoint markdown reports for language that would
    constitute an unqualified final scientific claim (e.g. asserting H1 is
    supported) without an accompanying scale/label disclaimer. This is a
    coarse heuristic, not a proof, but it catches the most obvious failure
    mode: a report that forgets to say PILOT / MECHANISM DEMONSTRATION /
    OPEN FINDING where the underlying run was explicitly small-scale.
    """
    problems = []
    for md_file in CHECKPOINTS_DIR.glob("*.md"):
        text = md_file.read_text()
        if ("N=12" in text or "N=40" in text or "N=300" in text) and "PILOT" not in text and \
           "MECHANISM" not in text and "OPEN FINDING" not in text and "demonstration" not in text.lower():
            problems.append(md_file.name)
    return check(len(problems) == 0,
                 f"No checkpoint report with a small, non-final N lacks an explicit scale disclaimer "
                 f"(problem files: {problems})", diagnostics)


def main():
    diagnostics: List[Dict] = []
    results = {
        "checkpoint_0": audit_checkpoint_0(diagnostics),
        "checkpoint_5": audit_checkpoint_5(diagnostics),
        "checkpoint_7": audit_checkpoint_7(diagnostics),
        "checkpoint_8": audit_checkpoint_8(diagnostics),
        "checkpoint_9": audit_checkpoint_9(diagnostics),
    }
    ref_ok, ref_msg = audit_reference_validation_tests_declared()
    diagnostics.append({"check": f"Checkpoint 11: {ref_msg}", "passed": ref_ok})
    results["checkpoint_11_static_check"] = ref_ok

    results["no_fabricated_final_claims"] = audit_no_fabricated_final_claims(diagnostics)

    overall_pass = all(results.values())

    report = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "per_checkpoint_results": results,
        "diagnostics": diagnostics,
        "note": "This audit re-derives PASS/FAIL from raw JSON artifacts where possible "
                "(e.g. re-checks bias monotonicity in Checkpoint 5's raw numbers) rather than "
                "trusting narrative claims in markdown reports alone.",
    }

    out_dir = REPO_ROOT / "results" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2))

    print(f"{'='*60}")
    print(f"AUDIT RESULT: {report['overall_status']}")
    print(f"{'='*60}")
    for d in diagnostics:
        mark = "PASS" if d["passed"] else "FAIL"
        print(f"[{mark}] {d['check']}")
    print(f"{'='*60}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
