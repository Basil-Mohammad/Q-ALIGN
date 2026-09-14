# Data Dictionary — `results/pilot/`

## `pilot_report.json`
| Field | Meaning |
|---|---|
| `pool_size` | Number of circuits generated (pilot: 12) |
| `n_seeds_per_circuit` | Training seeds per circuit (pilot: 3; final requirement ≥10) |
| `quartile_sizes` | Count of circuits per A_min quartile (1=lowest..4=highest) |
| `illustrative_representatives` | circuit_id sampled at random per non-empty quartile |
| `cost_analysis_pilot.ratio_alignment_over_training` | alignment-estimation wall-clock / avg. partial-training wall-clock, PILOT SCALE ONLY |
| `environment` | Full environment fingerprint (package versions, git commit, platform) |
| `config_hash` | SHA-256 of the frozen pilot configuration |

## `per_circuit_raw_results.json` (one entry per circuit)
| Field | Meaning |
|---|---|
| `circuit_id` | Unique ID, format `pilot_circ_{index}_{entangling_layout}` |
| `complexity` | `{n_qubits, n_parameters, depth, n_two_qubit_gates}` |
| `a_spec_exact` | A_spec computed from the task's known closed-form spectrum |
| `a_spec_estimated_n500` | A_spec via sampled DFT estimator, 500 samples |
| `a_top_bin` | Binary topological alignment (Definition 3.8) |
| `a_top_weighted_exact` / `a_top_weighted_dp` | Weighted A_top via exact path enumeration vs. DP approximation |
| `a_plus` / `a_times` / `a_min` | The three aggregation forms, PILOT-FIXED weights (not calibrated — n too small) |
| `trainability_gradient_variance_pilot_n5` | Gradient-variance trainability proxy, 5 random inits (pilot; final default is 20) |
| `seeds` | List of `{seed_index, seed_value, final_mse, loss_trace}` — raw per-seed results, never only an average |
| `alignment_estimation_seconds` / `avg_partial_training_seconds_per_seed` | Timing, for the cost-ratio analysis |
