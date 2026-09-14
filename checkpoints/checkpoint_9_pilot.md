# Checkpoint 9 -- Central Experiment PILOT

**Status: PASS (mechanism verified) -- LABEL: PILOT, NOT EVIDENCE FOR H0/H1**

- Pool size: 12 circuits (3 qubits, periodic task family)
- Seeds per circuit: 3 (pilot only; final requirement is >=10, brief Sec 7)
- Quartile sizes (A_min-stratified): {'1': 10, '2': 0, '3': 0, '4': 2}
- Mean |A_spec exact - A_spec estimated (n=500)| across pool: 0.0473
- Cost ratio (alignment estimation / avg partial training, PILOT SCALE): 0.0012
- Config hash: 128abfc83bd1716098988cd4822231ea8ae6e5200a0ebaac72649e0d46bfe039

## Decision

Mechanism (pool generation -> alignment computation -> A_min stratification -> illustrative sampling -> partial training -> cost timing) runs end-to-end without error at pilot scale. **Next allowed step:** full central experiment requires an explicit resourcing decision (PHASE0_AUDIT.md Sec 5) before proceeding to N>=150 per task family -- this pilot does NOT authorize skipping that decision.
