"""
Checkpoint 0: environment verification (manuscript brief Sec 9).
Run: python scripts/checkpoint0_environment.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.provenance import environment_fingerprint

def main():
    env = environment_fingerprint()
    missing = [k for k, v in env.items() if v == "NOT_INSTALLED"]
    status = "PASS" if not missing else "FAIL"
    report = {
        "checkpoint": 0,
        "status": status,
        "environment": env,
        "missing_packages": missing,
    }
    out_dir = Path(__file__).parent.parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "checkpoint_0_environment.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Checkpoint 0 -- Environment Verification\n",
        f"**Status: {status}**\n",
        "## Environment fingerprint\n",
    ]
    for k, v in env.items():
        md.append(f"- `{k}`: {v}")
    if missing:
        md.append("\n## Missing packages (FAIL if any listed)\n")
        for m in missing:
            md.append(f"- {m}")
    (out_dir / "checkpoint_0_environment.md").write_text("\n".join(md) + "\n")
    print(f"Checkpoint 0: {status}")
    print(json.dumps(env, indent=2))
    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
