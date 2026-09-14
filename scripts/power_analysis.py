"""
Real power analysis (manuscript Sec 5.2 "Power analysis" paragraph /
brief Sec 16), using the six confirmed defaults from PHASE0_AUDIT.md Sec 2.

Computes the actual required N per task family for the central experiment,
given:
  - min_partial_r2 = DEFAULT_MIN_PARTIAL_R2 (0.05, confirmed)
  - the extended regression model: outcome ~ n_q + P + L + G + A
    (5 predictors)
  - n_covariate_strata: a 2x2 binary split on (expressibility, entangling
    capability), i.e. 4 strata -- the manuscript's own documented fallback
    if the full quartile-based stratification proves infeasible (Sec 5.2),
    adopted here directly as the planning assumption rather than assuming
    the more expensive quartile x quartile stratification by default.
  - alpha: a CONSERVATIVE planning value, not the actual BH-FDR-corrected
    threshold (which can only be computed after the real test manifest and
    realized p-values exist). We use alpha/m with m=3 (one per aggregation
    form -- A_+, A_x, A_min -- since all three will be tested on the same
    outcome) as a Bonferroni-style planning approximation, which is
    deliberately more conservative than what BH-FDR will likely require in
    practice; this makes the required-N estimate a SAFE UPPER BOUND for
    planning purposes, not an exact final figure.

Run: PYTHONPATH=. python scripts/power_analysis.py
"""
from __future__ import annotations
import json
from pathlib import Path

from src.statistics.power import required_pool_size, DEFAULT_MIN_PARTIAL_R2
from src.utils.provenance import hash_config

N_PREDICTORS_EXTENDED_MODEL = 5   # n_q, P, L, G, A
N_COVARIATE_STRATA = 4            # 2x2 binary split: expressibility x entangling capability
TARGET_POWER = 0.80
PLANNING_ALPHA = 0.05 / 3         # Bonferroni-style conservative planning value (see docstring)
N_FLOOR = 150                     # manuscript-specified floor
N_TASK_FAMILIES = 3               # periodic, classification, RL


def main():
    result = required_pool_size(
        min_partial_r2=DEFAULT_MIN_PARTIAL_R2,
        alpha=PLANNING_ALPHA,
        target_power=TARGET_POWER,
        n_predictors_extended_model=N_PREDICTORS_EXTENDED_MODEL,
        n_covariate_strata=N_COVARIATE_STRATA,
        n_floor=N_FLOOR,
    )

    report = {
        "inputs": {
            "min_partial_r2": DEFAULT_MIN_PARTIAL_R2,
            "planning_alpha": PLANNING_ALPHA,
            "target_power": TARGET_POWER,
            "n_predictors_extended_model": N_PREDICTORS_EXTENDED_MODEL,
            "n_covariate_strata": N_COVARIATE_STRATA,
            "n_floor": N_FLOOR,
        },
        "result": {
            "required_n_per_task_family": result.required_n,
            "achieved_power": result.achieved_power,
            "effect_size_f2": result.effect_size_f2,
            "floor_applied": result.floor_applied,
        },
        "total_across_task_families": result.required_n * N_TASK_FAMILIES,
        "note": "planning_alpha is a CONSERVATIVE Bonferroni-style approximation, "
                "not the final BH-FDR threshold (which requires the realized test "
                "manifest). Treat required_n as a safe upper bound for planning.",
        "config_hash": hash_config({
            "min_partial_r2": DEFAULT_MIN_PARTIAL_R2, "alpha": PLANNING_ALPHA,
            "target_power": TARGET_POWER, "n_predictors": N_PREDICTORS_EXTENDED_MODEL,
            "n_strata": N_COVARIATE_STRATA, "n_floor": N_FLOOR,
        }),
    }

    out_dir = Path(__file__).parent.parent / "results" / "power_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "power_analysis_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\n=> Required N per task family: {result.required_n}")
    print(f"=> Total circuits needed across {N_TASK_FAMILIES} task families: {result.required_n * N_TASK_FAMILIES}")
    print(f"=> Floor (150) applied instead of computed requirement: {result.floor_applied}")


if __name__ == "__main__":
    main()
