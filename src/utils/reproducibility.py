"""
Deterministic, per-purpose RNG streams (manuscript brief Sec 6/7/25).

Rule: never use the legacy global `numpy.random` state. Every stochastic
component (circuit generation, initialization, training, search, noise)
draws from its OWN independent stream, derived from a single fixed
top-level seed via numpy.random.SeedSequence.spawn -- so streams are
reproducible AND statistically independent of each other (unlike, e.g.,
seeding each with `base_seed + offset`, which does not guarantee
independence for all PRNG algorithms).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import zlib
import numpy as np

PURPOSES = ("training", "initialization", "data_generation", "search", "noise", "bootstrap")


def _stable_purpose_code(purpose: str) -> int:
    """A deterministic integer derived from `purpose`, stable ACROSS
    process invocations (unlike Python's built-in `hash()`, which is
    randomized per-process by default via PYTHONHASHSEED for security
    reasons). Using `hash()` here was a real reproducibility bug found
    during testing: the same master_seed produced DIFFERENT streams on
    separate runs of the same script, exactly defeating the purpose of
    this module. zlib.crc32 over the UTF-8 bytes is fast, has no such
    randomization, and is identical on every platform/process/Python
    version.
    """
    return zlib.crc32(purpose.encode("utf-8"))

# Fixed, documented seed list (manuscript Sec 7: "prefer a fixed documented
# seed list", "do not randomly regenerate seeds between runs"). This is the
# ENTIRE authoritative seed list for this repository; any experiment must
# draw only from indices declared in its own frozen config.
FIXED_SEED_LIST: List[int] = [
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,  # seeds 0-9  (min 10, default)
    1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,  # seeds 10-19 (high-variance experiments)
    1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,  # seeds 20-29 (particularly noisy comparisons)
]


@dataclass
class SeedRegistry:
    """Spawns independent RNG streams per (purpose, index) pair from a
    single master SeedSequence, so re-running with the same top-level seed
    reproduces every downstream stream bit-for-bit, and different purposes
    never accidentally share entropy.
    """
    master_seed: int
    _cache: Dict[str, np.random.Generator] = field(default_factory=dict, repr=False)

    def stream(self, purpose: str, index: int) -> np.random.Generator:
        if purpose not in PURPOSES:
            raise ValueError(f"Unknown purpose '{purpose}'. Register it explicitly in PURPOSES.")
        key = f"{purpose}:{index}"
        if key not in self._cache:
            # Distinct spawn key per (purpose, index) guarantees independence.
            # Uses a STABLE hash of `purpose` (see _stable_purpose_code), not
            # Python's randomized built-in hash() -- see module docstring.
            ss = np.random.SeedSequence([self.master_seed, _stable_purpose_code(purpose), index])
            self._cache[key] = np.random.default_rng(ss)
        return self._cache[key]


def require_min_seeds(seed_list: List[int], minimum: int = 10) -> None:
    if len(seed_list) < minimum:
        raise ValueError(
            f"Only {len(seed_list)} seeds provided; manuscript brief Sec 7 requires a minimum of "
            f"{minimum} independent seeds for any final stochastic experiment. "
            f"This is a hard requirement, not a suggestion -- refusing to proceed."
        )