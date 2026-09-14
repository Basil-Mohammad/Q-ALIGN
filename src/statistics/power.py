"""
Power analysis for sizing the central-experiment pool N (manuscript Sec
5.2 "Power analysis" paragraph): computes N required for 80% power to
detect a pre-specified minimum effect size for the covariate-conditional
regression coefficient on A, at the BH-FDR-corrected significance level,
given the planned number of covariate strata.

Uses the standard F-test power formula for a single added predictor in
multiple regression (Cohen 1988), implemented via statsmodels' noncentral-F
machinery so the computation is a recognized, checkable statistical method
rather than an ad hoc approximation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from statsmodels.stats.power import FTestPower


# PHASE0_AUDIT.md decision #4: placeholder minimum detectable effect,
# must be replaced by a researcher-specified value before any real
# pre-registration is filed. Exposed as a named constant, not buried
# inline, so changing it is a visible diff.
DEFAULT_MIN_PARTIAL_R2 = 0.05


def cohens_f2_from_partial_r2(partial_r2: float) -> float:
    if not (0 < partial_r2 < 1):
        raise ValueError("partial_r2 must be in (0,1)")
    return partial_r2 / (1 - partial_r2)


@dataclass(frozen=True)
class PowerAnalysisResult:
    required_n: int
    achieved_power: float
    effect_size_f2: float
    alpha_used: float
    n_predictors_extended_model: int
    n_covariate_strata: int
    floor_applied: bool


def required_pool_size(min_partial_r2: float, alpha: float, target_power: float,
                        n_predictors_extended_model: int, n_covariate_strata: int,
                        n_floor: int = 150) -> PowerAnalysisResult:
    """Computes required total N (manuscript floor: N >= 150 unless the
    power analysis requires more -- Sec 5.2). `alpha` should be the
    BH-FDR-corrected per-test threshold implied by the frozen test manifest
    (src/statistics/multiple_testing.py), not the raw 0.05, per Remark 6.2.

    We inflate the naive per-cell requirement by n_covariate_strata,
    reflecting that the covariate-CONDITIONAL analysis (manuscript Sec 5.2,
    'Controlling for expressibility and entangling capability') splits the
    pool across strata; the regression must be adequately powered WITHIN
    strata, not only in aggregate.
    """
    f2 = cohens_f2_from_partial_r2(min_partial_r2)
    ftp = FTestPower()
    # Search for minimal per-stratum n giving >= target_power.
    n_per_stratum = 5
    achieved = 0.0
    while achieved < target_power and n_per_stratum < 100000:
        df_num = 1  # one added predictor (A) beyond the baseline model
        df_denom = n_per_stratum - n_predictors_extended_model - 1
        if df_denom <= 0:
            n_per_stratum += 5
            continue
        achieved = ftp.power(effect_size=f2, df_num=df_num, df_denom=df_denom, alpha=alpha)
        if achieved < target_power:
            n_per_stratum += 5
    total_required = n_per_stratum * n_covariate_strata
    floor_applied = total_required < n_floor
    final_n = max(total_required, n_floor)
    return PowerAnalysisResult(
        required_n=int(final_n),
        achieved_power=float(achieved),
        effect_size_f2=float(f2),
        alpha_used=float(alpha),
        n_predictors_extended_model=n_predictors_extended_model,
        n_covariate_strata=n_covariate_strata,
        floor_applied=floor_applied,
    )
