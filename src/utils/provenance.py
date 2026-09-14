"""
Provenance and leakage-prevention utilities (manuscript Remark 3.19 / brief
Sec 18: "Add automated assertions proving intersection(calibration_ids,
evaluation_ids) == empty set").
"""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set


class LeakageError(Exception):
    """Raised when calibration and evaluation sets are not disjoint."""


def assert_disjoint_splits(calibration_ids: Iterable[str], evaluation_ids: Iterable[str]) -> None:
    cal: Set[str] = set(calibration_ids)
    ev: Set[str] = set(evaluation_ids)
    overlap = cal & ev
    if overlap:
        raise LeakageError(
            f"Calibration/evaluation leakage detected: {len(overlap)} circuit IDs appear in both "
            f"folds (manuscript Remark 3.19 forbids this). Offending IDs (up to 10 shown): "
            f"{sorted(overlap)[:10]}"
        )
    if not cal:
        raise ValueError("Calibration set is empty.")
    if not ev:
        raise ValueError("Evaluation set is empty.")


def hash_config(config: Dict[str, Any]) -> str:
    """SHA-256 hash of a canonicalized (sorted-key) JSON representation of a
    config dict (manuscript brief Sec 34: freeze YAML config, hash it,
    never overwrite on change -- create a new experiment ID instead).
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit_hash(repo_dir: str = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "UNAVAILABLE"


def environment_fingerprint() -> Dict[str, Any]:
    """Everything required by manuscript brief Sec 6 (global reproducibility
    requirements) that is machine-discoverable at runtime.
    """
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": git_commit_hash(),
    }
    for pkg in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "pennylane", "networkx", "statsmodels", "pytest"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "NOT_INSTALLED"
    return info


@dataclass(frozen=True)
class ExperimentRecord:
    """One immutable provenance record per experiment run (manuscript brief
    Sec 6 / 34). Never mutate after creation; a config change must produce a
    new ExperimentRecord with a new experiment_id.
    """
    experiment_id: str
    config_hash: str
    environment: Dict[str, Any]
    seeds_used: Dict[str, list]

    def to_json(self) -> str:
        return json.dumps(
            {
                "experiment_id": self.experiment_id,
                "config_hash": self.config_hash,
                "environment": self.environment,
                "seeds_used": self.seeds_used,
            },
            indent=2,
            sort_keys=True,
        )
