### 1. Anomalies and missing diagnostics
- **No obvious noise-dominated effect:** `gain_mean = 0.03499458361150165` comfortably exceeds `seed_noise = 0.0010182919981824856`; both listed gains are positive and closely agree. However, `seed_sd = 0.0007066882225594238` differs from `seed_noise`; their definitions need clarification.
- **Non-monotonic trajectory:** gaps peak at step `4851` (`0.0451`, `0.0441`) and fall to `0.035` at `6930`. The final benefit is therefore smaller than the intermediate peak.
- **No observed sign conflict:** `gain_tpp30 = 0.046268982280819326` and the gate gains are positive. The JSON does not explicitly label the gate comparison as TPP120.
- **Optimum/bootstrap diagnostics absent:** `edge_extension = []` does not establish interior optima. No grid surfaces, convexity checks, bootstrap failure counts, or divergence records are supplied.
- **Speed-ups need provenance:** `speedup_untuned = 1.5183786483006478` and fitted-optimum speed-ups (`1.7723485372515568`, `1.8068602527727673`) lack reference conditions and uncertainty. They are not demonstrated wall-clock improvements.
- **Tuning substantially narrows the apparent advantage:** fitted optimal BPBs (`1.2747535298792914`, `1.2732431682165364`) differ by less than `tuned_gain_null_bpb = 0.002`.

### 2. Threats and cheapest resolutions
- **Limited seed evidence / unclear noise estimator:** audit noise definitions first; cheapest extra training is a matched constant/scaled seed pair.
- **Untuned advantage may reflect hyperparameter mismatch:** run matched validation evaluations at both fitted optima; add a paired training replicate if these optima have not actually been trained.
- **Unknown edge or non-convex optimum:** inspect existing grid results first; extend only the implicated boundary or sample the questionable basin if needed.
- **Bootstrap reliability unknown:** report failure counts and resample existing results; no training required initially.
- **Speed-up extrapolation:** audit reference curves and fit uncertainty first; validate at the predicted equal-quality budget if unsupported.
- **Schedule-mechanism generalization:** cosine WD-ratio uncertainty includes `1.0`; WSD entries are hypotheses, not outcomes. A matched WSD constant/scaled pair would test transfer.

### 3. Decision consistency
**Yes.** The reported gain clearly supports gate-1 PASS, while the small fitted-optimum difference and missing tuned validation support keeping the tuned comparison pending.