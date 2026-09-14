"""
Central experiment protocol (manuscript Sec 5.2): independent pool
generation (no conditioning on A), A_min-based quartile stratification,
whole-pool regression as primary evidence.

This module implements the MECHANISM exactly as specified. It is exercised
at PILOT scale (src/experiments/pilot.py / scripts/run_pilot.py) with a
small pool (~12 circuits) to prove the mechanism works; it is NOT run at
the pre-registered N>=150-per-family scale in this delivery (see
PHASE0_AUDIT.md Sec 5/8 for the resourcing reason).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
import numpy as np

from src.circuits.generators import QAlignCircuit
from src.circuits.complexity import CircuitComplexity, complexity_matches
from src.alignment.aggregation import AggregationWeights, conservative


@dataclass
class PoolCircuitRecord:
    circuit: QAlignCircuit
    a_spec: float
    a_top: float
    a_sym: float
    a_hw: float
    a_min: float
    complexity: CircuitComplexity


def generate_independent_pool(circuit_factory: Callable[[np.random.Generator], QAlignCircuit],
                               target_complexity: CircuitComplexity, pool_size: int,
                               rng: np.random.Generator, param_tol: int = 0, depth_tol: int = 0,
                               max_attempts_multiplier: int = 20) -> List[QAlignCircuit]:
    """Generates `pool_size` circuits via `circuit_factory`, WITHOUT
    computing or conditioning on any alignment score during generation
    (manuscript Sec 5.2, step 1: 'without reference to their eventual A
    score at generation time'). Circuits not matching `target_complexity`
    within tolerance are discarded (but the discard reason is complexity
    mismatch, never alignment) and generation retries up to a bounded
    number of attempts.
    """
    pool = []
    attempts = 0
    max_attempts = pool_size * max_attempts_multiplier
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        candidate = circuit_factory(rng)
        if complexity_matches(candidate.complexity(), target_complexity, param_tol, depth_tol):
            pool.append(candidate)
    if len(pool) < pool_size:
        raise RuntimeError(
            f"Only generated {len(pool)}/{pool_size} complexity-matched circuits in "
            f"{max_attempts} attempts. Widen the generator space or complexity tolerance "
            f"(manuscript-consistent action) rather than accepting a smaller, undocumented N."
        )
    return pool


def compute_pool_alignment(pool: List[QAlignCircuit], a_spec_fn, a_top_fn, a_sym_fn, a_hw_fn,
                            weights: AggregationWeights) -> List[PoolCircuitRecord]:
    """Computes A_spec, A_top, A_sym, A_hw and the STRATIFYING form (A_min)
    for every circuit AFTER pool generation (manuscript Sec 5.2, step 2:
    'compute A for every circuit in the pool after generation').
    """
    records = []
    for c in pool:
        a_spec = a_spec_fn(c)
        a_top = a_top_fn(c)
        a_sym = a_sym_fn(c)
        a_hw = a_hw_fn(c)
        a_min = conservative(a_spec, a_top, a_sym, a_hw, weights)
        records.append(PoolCircuitRecord(circuit=c, a_spec=a_spec, a_top=a_top, a_sym=a_sym,
                                          a_hw=a_hw, a_min=a_min, complexity=c.complexity()))
    return records


def stratify_into_quartiles(records: List[PoolCircuitRecord]) -> dict:
    """Stratifies by A_min (manuscript-specified stratifying form, Sec 5.2:
    'Which aggregation form defines the quartiles'). Returns a dict
    {quartile_index (1=lowest .. 4=highest): [records]}.
    """
    if len(records) < 4:
        raise ValueError("Need at least 4 circuits to form quartiles.")
    sorted_records = sorted(records, key=lambda r: r.a_min)
    quartile_edges = np.percentile([r.a_min for r in sorted_records], [25, 50, 75])
    quartiles = {1: [], 2: [], 3: [], 4: []}
    for r in sorted_records:
        if r.a_min <= quartile_edges[0]:
            quartiles[1].append(r)
        elif r.a_min <= quartile_edges[1]:
            quartiles[2].append(r)
        elif r.a_min <= quartile_edges[2]:
            quartiles[3].append(r)
        else:
            quartiles[4].append(r)
    return quartiles


def sample_illustrative_representatives(quartiles: dict, rng: np.random.Generator) -> dict:
    """Draws ONE circuit at random from each non-empty quartile for the
    illustrative Table 2, per manuscript Sec 5.2 (never hand-picked).
    """
    reps = {}
    for q, records in quartiles.items():
        if not records:
            continue
        idx = rng.integers(0, len(records))
        reps[q] = records[idx]
    return reps
