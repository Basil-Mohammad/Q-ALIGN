"""
run_campaign.py -- The REAL central-experiment campaign runner (manuscript
Sec 5.2, brief Sec 27/33). Designed to run UNATTENDED for hours to days,
survive interruption (power loss, sleep, terminal close), and resume
exactly where it left off.

*** THIS IS THE FIRST NON-MECHANISM-DEMONSTRATION SCRIPT IN THIS REPOSITORY.
Its output, once complete, is intended to be real evidence for/against H1
-- not a scale-disclaimed mechanism check. ***

CONFIGURATION (fixed here, matching the researcher's chosen scale):
  N_PER_FAMILY = 150   (manuscript floor; NOT the power-analysis-corrected
                          420 -- an explicit, documented scope reduction
                          for time-feasibility, see PHASE0_AUDIT.md)
  N_SEEDS = 10          (the manuscript's absolute minimum, brief Sec 7)
  N_TRAINING_STEPS = 100 (a fixed, pre-registered "partial training"
                          definition -- see PHASE0_AUDIT.md decision #6)
  N_QUBITS = 4, N_LAYERS = 2 (matching the most-validated circuit family
                          from Checkpoints 7/8/12/13/14 in this session)

ESTIMATED WALL-CLOCK TIME: ~2.8 days on the hardware timing measured in
Checkpoint 12 of this session (~4.33 sec per circuit-seed at 8 steps,
scaled to 100 steps: ~54 sec/unit x 150 circuits x 10 seeds x 3 families
= ~675,000 sec =~ 7.8 days -- WAIT, see the script's own live ETA printout
at startup for the actual, freshly-measured estimate on YOUR hardware,
which is what should be trusted, not this docstring's back-of-envelope
number computed on a different machine.

USAGE:
  python scripts/run_campaign.py
  (safe to Ctrl+C at any time; re-running the same command resumes)

  To check progress without running:
  python scripts/run_campaign.py --status
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from src.tasks.periodic import make_multi_term_task
from src.tasks.classification import make_phase_xor_task
from src.tasks.reinforcement_learning import make_single_pair_bandit
from src.circuits.encodings import build_multi_layer_angle_encoding
from src.circuits.entanglers import linear_chain, circular_chain, all_to_all
from src.circuits.generators import QAlignCircuit, random_init_params, train_partial
from src.circuits.complexity import CircuitComplexity
from src.circuits.serialization import save_pool, load_pool, round_trip_is_exact
from src.alignment.spectral import spectral_alignment_exact
from src.alignment.topology import circuit_interaction_graph, topological_alignment_binary
from src.experiments.central_experiment import generate_independent_pool
from src.statistics.regression import transform_mse_to_higher_is_better
from src.utils.reproducibility import SeedRegistry, FIXED_SEED_LIST, require_min_seeds
from src.utils.provenance import environment_fingerprint, hash_config

# ============================== CONFIGURATION ==============================
N_QUBITS = 4
N_VARS = 4
N_LAYERS = 2
MULTIPLIER_CHOICES = [1, 2]
N_PER_FAMILY = 150
N_SEEDS = 10
N_TRAINING_STEPS = 100
LEARNING_RATE = 0.3

CAMPAIGN_ROOT = Path(__file__).parent.parent / "results" / "campaign"
ENTANGLING_LAYOUTS = {
    "linear": lambda: linear_chain(N_QUBITS),
    "circular": lambda: circular_chain(N_QUBITS),
    "all_to_all": lambda: all_to_all(N_QUBITS),
}

TASK_FAMILIES = ["periodic", "classification", "reinforcement_learning"]


def get_task(family: str):
    if family == "periodic":
        return make_multi_term_task(d=N_VARS, terms_spec=[((0, 1), 1.0, 1.0, 0.0), ((2, 3), 1.0, 0.8, 0.0)])
    if family == "classification":
        return make_phase_xor_task(d=N_VARS, interacting_dims=(0, 1))
    if family == "reinforcement_learning":
        return make_single_pair_bandit(d=N_VARS, interacting_dims=(0, 1))
    raise ValueError(f"Unknown family: {family}")


def circuit_factory(rng: np.random.Generator, layout_names: list) -> QAlignCircuit:
    var_per_qubit = list(range(N_QUBITS))
    layers_spec, entangling_layers = [], []
    for _ in range(N_LAYERS):
        multipliers = [int(rng.choice(MULTIPLIER_CHOICES)) for _ in range(N_QUBITS)]
        layers_spec.append((var_per_qubit, multipliers))
        layout_name = layout_names[int(rng.integers(0, len(layout_names)))]
        entangling_layers.append(ENTANGLING_LAYOUTS[layout_name]())
    encoding = build_multi_layer_angle_encoding(N_QUBITS, N_VARS, layers_spec)
    return QAlignCircuit(encoding=encoding, entangling_layers=entangling_layers,
                          circuit_id=f"campaign_{rng.integers(0, 10**12)}")


def family_dir(family: str) -> Path:
    d = CAMPAIGN_ROOT / family
    d.mkdir(parents=True, exist_ok=True)
    (d / "training").mkdir(exist_ok=True)
    return d


def ensure_pool(family: str, seed_registry: SeedRegistry, family_offset: int) -> list:
    """Generates the pool ONCE and persists it. On resume, loads the
    EXISTING pool from disk rather than regenerating it -- regenerating
    could silently produce a different pool if a prior run was interrupted
    mid-generation, which would corrupt any partially-completed training
    results that reference the old pool's circuit IDs.
    """
    d = family_dir(family)
    pool_path = d / "pool.json"
    if pool_path.exists():
        pool = load_pool(str(pool_path))
        print(f"[{family}] Loaded existing pool ({len(pool)} circuits) from disk.")
        return pool

    task = get_task(family)
    gen_rng = seed_registry.stream("data_generation", family_offset)
    layout_names = list(ENTANGLING_LAYOUTS.keys())
    target_complexity = CircuitComplexity(n_qubits=N_QUBITS, n_parameters=3 * N_QUBITS * N_LAYERS,
                                           depth=N_LAYERS, n_two_qubit_gates=0)
    pool = generate_independent_pool(
        circuit_factory=lambda rng: circuit_factory(rng, layout_names),
        target_complexity=target_complexity, pool_size=N_PER_FAMILY, rng=gen_rng,
    )
    for c in pool:
        assert round_trip_is_exact(c), f"Serialization self-check failed for {c.circuit_id}"
    save_pool(pool, str(pool_path))
    print(f"[{family}] Generated and saved NEW pool ({len(pool)} circuits).")
    return pool


def ensure_alignment(family: str, pool: list) -> dict:
    """Computes A_spec/A_top ONCE per circuit and persists it (cheap, but
    still checkpointed for consistency and to avoid recomputation on
    every resume of a long-running process)."""
    d = family_dir(family)
    align_path = d / "alignment.json"
    if align_path.exists():
        return json.loads(align_path.read_text())

    task = get_task(family)
    exact_spectrum = task.exact_fourier_spectrum()
    exact_graph = task.exact_interaction_graph()
    result = {}
    for c in pool:
        a_spec = spectral_alignment_exact(exact_spectrum, c.encoding.accessible_frequencies())
        all_edges = [e for layer in c.entangling_layers for e in layer]
        g_c = circuit_interaction_graph(N_QUBITS, all_edges)
        a_top_raw = topological_alignment_binary(exact_graph, g_c, depth_L=N_LAYERS + 1, encoding="angle")
        a_top = a_top_raw if isinstance(a_top_raw, float) else 0.0
        result[c.circuit_id] = {"a_spec": a_spec, "a_top": a_top}
    align_path.write_text(json.dumps(result, indent=2))
    print(f"[{family}] Computed and saved alignment for {len(pool)} circuits.")
    return result


def get_training_data(family: str, seed_registry: SeedRegistry, family_offset: int):
    task = get_task(family)
    train_rng = seed_registry.stream("data_generation", family_offset + 500)
    X = train_rng.uniform(0, 2 * np.pi, size=(30, N_VARS))
    if family == "periodic":
        y = task.evaluate(X); y = (y - y.mean()) / (y.std() + 1e-8)
    elif family == "classification":
        y = task.label(X)
    else:
        y = task.true_value(X)
    return task, X, y


def train_one_unit(family: str, task, X, y, circuit: QAlignCircuit, seed_idx: int,
                    seed_registry: SeedRegistry, family_offset: int) -> dict:
    # CRITICAL: never use Python's built-in hash() on strings here -- it is
    # randomized per-process by default (PYTHONHASHSEED), which is EXACTLY
    # the bug found and fixed in src/utils/reproducibility.py earlier this
    # session. Use the same stable hash instead.
    import zlib
    circuit_hash = zlib.crc32(circuit.circuit_id.encode("utf-8")) % (10**9)  # large modulus: negligible
    # collision probability for 150 circuits (birthday bound ~150^2/(2*1e9) ~ 1e-5), unlike an earlier
    # draft of this line which used a 100,000 modulus and had a non-negligible (~10%) collision risk.
    init_rng = seed_registry.stream("initialization", family_offset * (10**9) + circuit_hash * 100 + seed_idx)
    theta0 = random_init_params(N_LAYERS, N_QUBITS, init_rng)
    t0 = time.perf_counter()
    result = train_partial(circuit, X, y, theta0, n_steps=N_TRAINING_STEPS, lr=LEARNING_RATE)
    elapsed = time.perf_counter() - t0

    theta_final = np.array(result["final_theta"])
    if family == "periodic":
        performance = float(transform_mse_to_higher_is_better(np.array([result["final_mse"]]))[0])
    elif family == "classification":
        preds = circuit.predict(X, theta_final)
        performance = task.accuracy(X, preds)
    else:  # reinforcement_learning
        preds = circuit.predict(X, theta_final)
        actions = np.where(preds >= 0, 1.0, -1.0)
        performance = -task.average_regret(X, actions)  # higher-is-better

    return {
        "circuit_id": circuit.circuit_id,
        "seed_index": seed_idx,
        "seed_value": int(FIXED_SEED_LIST[seed_idx % len(FIXED_SEED_LIST)]),
        "final_mse": result["final_mse"],
        "performance": performance,
        "loss_trace_final_10": result["loss_trace"][-10:],
        "training_seconds": elapsed,
        "n_steps": N_TRAINING_STEPS,
        "status": "COMPLETED",
    }


def run_family(family: str, seed_registry: SeedRegistry, family_offset: int, status_only: bool = False):
    d = family_dir(family)
    training_dir = d / "training"

    pool = ensure_pool(family, seed_registry, family_offset)
    ensure_alignment(family, pool)

    total_units = len(pool) * N_SEEDS
    done_units = sum(1 for _ in training_dir.glob("*.json"))

    if status_only:
        print(f"[{family}] {done_units}/{total_units} units complete ({100*done_units/total_units:.1f}%)")
        return done_units, total_units

    task, X, y = get_training_data(family, seed_registry, family_offset)

    run_start = time.time()
    units_this_session = 0
    for c in pool:
        for seed_idx in range(N_SEEDS):
            result_path = training_dir / f"{c.circuit_id}__seed{seed_idx}.json"
            if result_path.exists():
                continue  # RESUME: already completed, skip

            try:
                result = train_one_unit(family, task, X, y, c, seed_idx, seed_registry, family_offset)
            except Exception as e:
                # Record FAILURES explicitly rather than silently skipping or crashing
                # the whole campaign (brief Sec 36: a failed run must be distinguishable
                # from a missing run).
                result = {
                    "circuit_id": c.circuit_id, "seed_index": seed_idx,
                    "status": "FAILED", "error": str(e),
                }
            result_path.write_text(json.dumps(result, indent=2, default=str))

            units_this_session += 1
            done_units += 1
            if units_this_session % 5 == 0 or done_units == total_units:
                elapsed = time.time() - run_start
                rate = units_this_session / elapsed if elapsed > 0 else 0
                remaining = total_units - done_units
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                eta_str = str(timedelta(seconds=int(eta_seconds))) if rate > 0 else "unknown"
                print(f"[{family}] {done_units}/{total_units} ({100*done_units/total_units:.1f}%) | "
                      f"rate: {rate:.3f} units/sec | ETA (this family): {eta_str} | "
                      f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return done_units, total_units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Report progress without running anything.")
    args = parser.parse_args()

    require_min_seeds(FIXED_SEED_LIST, minimum=N_SEEDS)
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)

    config = {
        "n_qubits": N_QUBITS, "n_layers": N_LAYERS, "n_per_family": N_PER_FAMILY,
        "n_seeds": N_SEEDS, "n_training_steps": N_TRAINING_STEPS, "learning_rate": LEARNING_RATE,
    }
    config_path = CAMPAIGN_ROOT / "campaign_config.json"
    config_hash = hash_config(config)
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing.get("config_hash") != config_hash:
            raise RuntimeError(
                f"Campaign config on disk (hash={existing.get('config_hash')}) does not match the "
                f"current script's config (hash={config_hash}). Per brief Sec 34, a config change must "
                f"start a NEW experiment, never silently resume with a different config. Either restore "
                f"the original config values or move/delete '{CAMPAIGN_ROOT}' to start fresh."
            )
    else:
        config_path.write_text(json.dumps({**config, "config_hash": config_hash,
                                             "started_at": datetime.now().isoformat(),
                                             "environment": environment_fingerprint()}, indent=2))
        print(f"Started NEW campaign. Config hash: {config_hash}")

    seed_registry = SeedRegistry(master_seed=FIXED_SEED_LIST[10])

    print("=" * 70)
    print(f"Q-ALIGN CAMPAIGN {'STATUS CHECK' if args.status else 'RUN'}")
    print(f"Config: N={N_PER_FAMILY}/family, seeds={N_SEEDS}, steps={N_TRAINING_STEPS}, "
          f"qubits={N_QUBITS}, layers={N_LAYERS}")
    print("=" * 70)

    campaign_start = time.time()
    grand_done, grand_total = 0, 0
    for i, family in enumerate(TASK_FAMILIES):
        done, total = run_family(family, seed_registry, family_offset=i * 10000, status_only=args.status)
        grand_done += done
        grand_total += total

    print("=" * 70)
    print(f"OVERALL: {grand_done}/{grand_total} units complete ({100*grand_done/grand_total:.1f}%)")
    if not args.status and grand_done == grand_total:
        print("CAMPAIGN COMPLETE. Run scripts/analyze_campaign_results.py next.")
    elapsed_total = time.time() - campaign_start
    print(f"This session ran for {timedelta(seconds=int(elapsed_total))}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
