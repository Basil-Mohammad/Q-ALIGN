"""
Computational-cost analysis (manuscript Sec 5.6/5.7): measures wall-clock
cost of alignment estimation vs. partial training, and the total-campaign
feasibility budget.

STATUS: timing decorator and ratio-computation implemented and used in the
pilot (results/pilot/timing.json contains REAL measured numbers from the
smoke-scale pilot circuits in this delivery). The full per-task-family,
per-circuit-family cost table (manuscript Sec 5.6) requires the full
central-experiment pool and is NOT produced here.
"""
from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


@contextmanager
def timed(label: str, store: Dict[str, float]):
    start = time.perf_counter()
    yield
    store[label] = time.perf_counter() - start


@dataclass
class CostReport:
    alignment_estimation_seconds: float
    partial_training_seconds: float

    @property
    def ratio_alignment_over_training(self) -> float:
        if self.partial_training_seconds <= 0:
            raise ValueError("partial_training_seconds must be > 0 to compute a ratio.")
        return self.alignment_estimation_seconds / self.partial_training_seconds

    def as_dict(self) -> dict:
        return {
            "alignment_estimation_seconds": self.alignment_estimation_seconds,
            "partial_training_seconds": self.partial_training_seconds,
            "ratio_alignment_over_training": self.ratio_alignment_over_training,
            "interpretation": (
                "ratio < 1 means alignment estimation is cheaper than one partial training run "
                "(supports the manuscript's efficiency claim for this task/circuit family); "
                "ratio >= 1 must be reported as a boundary condition, not hidden (Sec 5.6)."
            ),
        }
