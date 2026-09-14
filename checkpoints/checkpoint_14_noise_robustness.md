# Checkpoint 14 -- Noise Robustness (Mechanism Validation)

**Status: PASS**

**SCALE WARNING: N=12 circuits, 1 seed, 10 training steps -- a mechanism validation, NOT a statistically powered result (manuscript's real protocol requires the full N=420/family pool, >=10 seeds, Sec 5.5/5.7).**

- Noise levels tested (pre-registered grid): [0.0, 0.001, 0.005, 0.01, 0.02]
- Mean classification accuracy by noise level: p=0.0: 0.617, p=0.001: 0.617, p=0.005: 0.617, p=0.01: 0.619, p=0.02: 0.622
- Mean |prediction| by noise level (sensitive shrinkage diagnostic): p=0.0: 0.268, p=0.001: 0.267, p=0.005: 0.261, p=0.01: 0.253, p=0.02: 0.239
- Accuracy non-increasing within 5% tolerance: **True**
- Prediction magnitude shrinks monotonically toward zero (the real mechanism-correctness check): **True**
- Kendall's tau, A_hw predicted vs. observed accuracy drop: -0.105

## A real methodological finding surfaced by this checkpoint

Classification **accuracy was essentially flat across the entire noise grid, including at noise levels far beyond the pre-registered maximum (tested informally up to p=0.5)** -- this is NOT a mechanism failure. Depolarizing noise shrinks expectation values toward the maximally-mixed-state value of 0; since classification accuracy depends only on the SIGN of the prediction, accuracy is insensitive to this shrinkage unless it is large enough to cross zero. The mean-|prediction| diagnostic added to this checkpoint confirms the noise mechanism itself works exactly as expected (monotonic shrinkage toward zero, verified above). **This is a genuine, reportable finding for the manuscript's own noise-robustness protocol (Sec 5.5): classification accuracy alone may be a poor outcome metric for detecting noise sensitivity in a classification-family task, and a magnitude- or confidence-based metric (e.g., mean |prediction|, or a margin-based score) should be considered as a supplementary noise-robustness outcome alongside raw accuracy** -- not a code bug, but a measurement-design consideration this mechanism check surfaced before the full campaign.

## Decision

The noise-simulation mechanism (`noisy_qnode`, depolarizing channel insertion) is confirmed to work correctly: it produces the theoretically expected monotonic shrinkage of predictions toward zero. **This checkpoint's main value was surfacing the accuracy-insensitivity finding above** before it could confuse interpretation of the full noise-robustness campaign. Recommendation for the full campaign (Sec 5.5): report mean-|prediction| or a margin-based metric alongside accuracy for classification-family noise-robustness results.
