"""
Cross-task transfer protocol (manuscript Sec 5.3): freeze calibration on
source task families, apply WITHOUT recalibration to a held-out family.

STATUS: interface + freeze/apply mechanism implemented and unit-tested
(tests/test_transfer_freeze.py). NOT executed at the manuscript's
pre-registered scale (three real task families with N>=150 pools each) --
see PHASE0_AUDIT.md Sec 5/8. Calling `run_transfer_stage` on real data
before that campaign exists will raise NotImplementedError for the task
families that are not yet built (classification, reinforcement_learning).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from src.alignment.aggregation import AggregationWeights


class RecalibrationForbiddenError(Exception):
    """Raised if code attempts to refit weights on the held-out target."""


@dataclass
class FrozenCalibration:
    weights: AggregationWeights
    source_task_families: List[str]
    frozen: bool = field(default=True, init=False)

    def apply(self) -> AggregationWeights:
        return self.weights


def freeze_calibration(weights: AggregationWeights, source_task_families: List[str]) -> FrozenCalibration:
    if len(source_task_families) == 0:
        raise ValueError("Must specify at least one source task family.")
    return FrozenCalibration(weights=weights, source_task_families=source_task_families)


def apply_to_held_out(frozen: FrozenCalibration, held_out_task_family: str) -> AggregationWeights:
    """The manuscript's central falsifiability mechanism (Sec 5.3): this
    function can ONLY read weights from a FrozenCalibration; there is no
    code path here that fits anything on `held_out_task_family` data. If a
    caller wants different weights for the held-out family, they must call
    `freeze_calibration` again explicitly with that family added to
    `source_task_families` -- which would defeat the point and is exactly
    the behavior this separation is designed to make visible in a diff.
    """
    if held_out_task_family in frozen.source_task_families:
        raise RecalibrationForbiddenError(
            f"'{held_out_task_family}' is already a source family used for calibration; "
            f"it cannot also be the held-out transfer target (Sec 5.3 requires a genuinely "
            f"unseen family)."
        )
    return frozen.apply()


def select_held_out_family_by_random_draw(candidate_families: List[str], rng: np.random.Generator) -> str:
    """Sec 5.3 pre-registration parity requirement: the held-out family for
    stage 3 must be selected by an independent random draw from a
    pre-declared candidate list, documented BEFORE any estimator is built
    (not chosen because it is known to work well or known to be hard).
    """
    if len(candidate_families) < 2:
        raise ValueError("Need at least 2 candidate held-out families for a meaningful random draw.")
    idx = rng.integers(0, len(candidate_families))
    return candidate_families[idx]
