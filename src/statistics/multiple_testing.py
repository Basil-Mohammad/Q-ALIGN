"""
Benjamini-Hochberg FDR correction over a FROZEN, pre-declared test family
(manuscript Remark 6.2 / brief Sec 26).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from src.utils.provenance import hash_config


class TestManifestFrozenError(Exception):
    pass


@dataclass
class TestManifest:
    """A test family must be declared and hashed BEFORE any p-value in it
    is computed. Adding a test after the manifest is frozen raises an
    error -- this is the mechanism that prevents post-hoc test-family
    inflation (Remark 6.2: 'pre-specifying the full set of tests ... before
    data collection').
    """
    declared_tests: List[str] = field(default_factory=list)
    _frozen: bool = False
    _hash: str = ""

    def declare(self, test_name: str) -> None:
        if self._frozen:
            raise TestManifestFrozenError(
                f"Cannot declare new test '{test_name}': manifest is frozen "
                f"(hash={self._hash}). Adding tests after freezing is exactly "
                f"the post-hoc test-family inflation Remark 6.2 forbids."
            )
        if test_name in self.declared_tests:
            raise ValueError(f"Test '{test_name}' already declared.")
        self.declared_tests.append(test_name)

    def freeze(self) -> str:
        self._frozen = True
        self._hash = hash_config({"tests": sorted(self.declared_tests)})
        return self._hash

    def is_frozen(self) -> bool:
        return self._frozen


def benjamini_hochberg(manifest: TestManifest, raw_p_values: Dict[str, float], q: float = 0.05) -> Dict[str, dict]:
    """Standard BH-FDR step-up procedure. `raw_p_values` keys MUST exactly
    match `manifest.declared_tests` (else raises) -- prevents silently
    correcting a different test set than the one that was frozen.
    """
    if not manifest.is_frozen():
        raise TestManifestFrozenError("Manifest must be frozen before applying correction.")
    declared = set(manifest.declared_tests)
    provided = set(raw_p_values.keys())
    if declared != provided:
        missing = declared - provided
        extra = provided - declared
        raise ValueError(f"p-value set does not match frozen manifest. Missing: {missing}. Extra: {extra}.")

    names = list(raw_p_values.keys())
    pvals = np.array([raw_p_values[n] for n in names])
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh_thresh = (np.arange(1, m + 1) / m) * q

    # Standard BH: find largest k such that p_(k) <= (k/m)*q; reject all <= that k.
    below = ranked <= bh_thresh
    if np.any(below):
        k_max = np.max(np.where(below)[0])
        cutoff_p = ranked[k_max]
    else:
        cutoff_p = -1.0  # nothing significant

    # Adjusted p-values (Benjamini-Hochberg-Yekutieli monotone adjustment)
    adjusted = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        prev = min(prev, val)
        adjusted[i] = prev
    adjusted = np.clip(adjusted, 0, 1)

    result = {}
    for rank_i, orig_i in enumerate(order):
        name = names[orig_i]
        result[name] = {
            "raw_p": float(pvals[orig_i]),
            "adjusted_p": float(adjusted[rank_i]),
            "significant_at_fdr": bool(pvals[orig_i] <= cutoff_p) if cutoff_p >= 0 else False,
        }
    result["_manifest_hash"] = manifest._hash
    result["_q"] = q
    result["_m_tests"] = m
    return result
