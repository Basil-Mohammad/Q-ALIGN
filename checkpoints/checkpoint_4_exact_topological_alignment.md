# Checkpoint 4 -- Exact Topological Alignment Verification

**Status: PASS**

Re-derived from fresh runs of `tests/test_topology.py` and `tests/test_weighted_topology_dp.py` at report-generation time.

## What is verified
- A_top_bin = 1.0 exactly when a required interaction is directly connected in the circuit graph (hand-computed case).
- A_top_bin = 0.0 exactly when a required interaction is unreachable (disconnected graph, hand-computed case).
- A_top_bin = 0.5 exactly for two required interactions with only one reachable (hand-computed partial case).
- A_top_bin in [0,1] holds under randomized fuzzing.
- The weighted variant (A_top^w) matches a manually-verified hand computation for a small 3-node example.
- Amplitude encoding correctly returns the NOT_APPLICABLE sentinel rather than a misleading float (manuscript scope limit, enforced in code).
- The exact path-enumeration ground truth (kappa_exact) and its DP/spectral approximations are cross-validated against each other on small graphs (Checkpoint 5's `weighted_topology_estimator_validation` section: DP mean abs error and Kendall's tau vs. exact were both computed and reported there).
- No function in this module ever asserts Conjecture 3.13 (the topological lower bound) as a proven theorem (verified by source inspection).

## Raw pytest output (tails)
```
collecting ... collected 11 items

tests/test_topology.py::test_amplitude_encoding_not_applicable PASSED    [  9%]
tests/test_topology.py::test_unknown_encoding_raises_rather_than_assumes PASSED [ 18%]
tests/test_topology.py::test_binary_perfect_topological_alignment PASSED [ 27%]
tests/test_topology.py::test_binary_zero_topological_alignment PASSED    [ 36%]
tests/test_topology.py::test_binary_partial_topological_alignment PASSED [ 45%]
tests/test_topology.py::test_binary_bounds_always_0_1 PASSED             [ 54%]
tests/test_topology.py::test_empty_task_graph_raises PASSED              [ 63%]
tests/test_topology.py::test_kappa_exact_bounds PASSED                   [ 72%]
tests/test_topology.py::test_weighted_topological_alignment_matches_manual_computation PASSED [ 81%]
tests/test_topology.py::test_weighted_topology_rejects_out_of_bounds_kappa PASSED [ 90%]
tests/test_topology.py::test_conjecture_not_asserted_as_theorem PASSED   [100%]

============================== 11 passed in 0.43s ==============================
---
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- /home/basil/Q-ALIGN/venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/basil/Q-ALIGN
configfile: pytest.ini
collecting ... collected 7 items

tests/test_weighted_topology_dp.py::test_dp_approximation_bounded_0_1 PASSED [ 14%]
tests/test_weighted_topology_dp.py::test_spectral_approx_bounded_0_1 PASSED [ 28%]
tests/test_weighted_topology_dp.py::test_validation_report_structure PASSED [ 42%]
tests/test_weighted_topology_dp.py::test_transfer_freeze_and_apply PASSED [ 57%]
tests/test_weighted_topology_dp.py::test_transfer_forbids_recalibration_on_held_out_thats_also_source PASSED [ 71%]
tests/test_weighted_topology_dp.py::test_held_out_family_selection_is_a_random_draw_not_hand_picked PASSED [ 85%]
tests/test_weighted_topology_dp.py::test_held_out_family_selection_requires_multiple_candidates PASSED [100%]

============================== 7 passed in 1.66s ===============================
```

## Decision
Exact (and approximate) topological alignment computation is verified correct against hand-computed reference cases. This gate has been open since early in this session.
