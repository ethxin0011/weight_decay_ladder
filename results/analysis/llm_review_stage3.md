### 1. Anomalies and checks
- **Noise versus effect:** Gate 1’s gain, `0.03499458361150165`, comfortably exceeds seed noise, `0.0010182919981824856`; both paired gains are positive. No obvious seed-noise anomaly there.
- **Much smaller tuned effect:** Tuned gain is `0.002583278761996155`, with CI `[0.0015270611895128144, 0.0036394963344794956]`. It is positive, but the interval crosses the preregistered tuned-null threshold, `0.002`.
- **Schedule anomaly:** WSD’s WD ratio, `0.28702679510801843`, has CI `[0.19135119673867895, 0.43054019266202764]`, consistent with **none** of the stated hypotheses. “Closest: H_none” is not evidence of consistency. Cosine supports both H_none and H_int, not uniquely H_int.
- **Speed-ups:** Untuned `1.5183786483006478` versus tuned `1.0332329547797672` is consistent with most benefit disappearing after tuning. The fitted-optimum speed-ups have an unspecified reference; they cannot be directly reconciled with the tuned speed-up.
- **Unverifiable checks:** `edge_extension: []` does not establish interior or convex optima. No bootstrap diagnostics, divergence logs, or explicitly labeled TPP120 comparison are supplied. TPP30’s reported gain is positive; opposite-sign behavior is not demonstrated.

### 2. Threats and cheapest resolution
- **Tuned effect near the null threshold; limited seed coverage:** Add a paired tuned-seed replication at the selected settings.
- **Possible boundary/local-fit optimum:** Inspect existing grid losses and fitted residuals first; if boundary-seeking, add a neighboring grid point beyond that boundary.
- **Bootstrap reliability unknown:** Report resampling unit, failed-fit frequency, and sensitivity using existing data; no training run initially needed.
- **WSD interpretation unsupported:** Verify ratio orientation and normalization first; if correct, replicate the selected WSD comparison.
- **Speed-up interpretation/model dependence:** Publish reference definitions and inversion calculations; a matched-target run is needed only if claiming directly observed speed-up.
- **TPP transfer and divergence checks missing:** Supply labeled ladder endpoints and run-status logs; add a paired comparison only where absent.

### 3. Decision consistency
**Yes:** PASS and the reported survival fraction are numerically consistent, but “genuine” should not imply that the tuned CI excludes the preregistered null threshold.