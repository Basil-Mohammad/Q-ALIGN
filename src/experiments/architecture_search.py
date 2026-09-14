"""
Architecture-search experiment (manuscript Sec 5.4): Q-ALIGN-guided search
vs. random sampling, QAS baselines, depth-based/parameter-count-based
selection, exhaustive-training reference.

STATUS: INTERFACE ONLY. Not executed in this delivery. Implementing real
QAS baselines (differentiable QAS -- Zhang et al. 2022; RL-based QAS --
Ostaszewski et al. 2021) is a substantial project in its own right and is
explicitly out of scope for a single sandbox session; stubbing them with
fake behavior would violate the "do not fabricate results" rule more
severely than simply not implementing them yet. The budget-matching
contract (manuscript Sec 5.4 parity requirement) is implemented as a
schema so that when real baselines are added, they are structurally
forced to share a budget.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
import numpy as np


@dataclass(frozen=True)
class SearchBudget:
    """Pre-registered, matched-across-methods budget (Sec 5.4 parity
    requirement). Every search method function must accept exactly this
    object and MUST NOT exceed total_evaluations.
    """
    total_evaluations: int
    n_seeds: int  # manuscript brief Sec 29: >=10, prefer 20+ if high variance


class SearchMethod:
    """Abstract base. Concrete methods (random_search, qalign_guided_search,
    qas_differentiable, qas_reinforcement_learning, depth_based,
    param_count_based, exhaustive_reference) are NOT implemented here.
    """
    name: str = "abstract"

    def run(self, budget: SearchBudget, rng: np.random.Generator):
        raise NotImplementedError(
            f"{self.__class__.__name__} is a Phase-5.4 interface stub. "
            f"Implementing and running real architecture-search baselines "
            f"is future work (see PHASE0_AUDIT.md Sec 5/8); no fabricated "
            f"performance-per-evaluation numbers are produced by this class."
        )


class QAlignGuidedSearch(SearchMethod):
    name = "qalign_guided"


class RandomSearch(SearchMethod):
    name = "random"


class DepthBasedSelection(SearchMethod):
    name = "depth_based"


class ParamCountBasedSelection(SearchMethod):
    name = "param_count_based"
