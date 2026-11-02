### 1. Anomalies and diagnostic gaps

- **Untuned effect exceeds seed noise:** mean gain **0.034995 bpb** versus noise **0.001018**; seed gains agree closely. After tuning, gain falls to **0.002583**, with CI **[0.001055, 0.004111]** spanning the preregistered **0.002** null threshold.
- **Fit–validation discrepancy:** fitted optima (**1.274754**, constant; **1.273243**, scaled) suggest a smaller advantage than the measured tuned gap. This warrants checking fit uncertainty, not declaring failure.
- **Schedule-dependent anomaly:** WSD’s WD ratio **0.287027**, CI **[0.191351, 0.430540]**, is inconsistent with every listed hypothesis. “Closest: H_none” does **not** mean supported. Cosine supports both H_none and H_int, so it does not distinguish them.
- **Scaling extrapolation misses:** L6 predicted multipliers **0.438104 / 0.948090** exceed fitted **0.331174 / 0.631673**; regrets are **0.001396 / 0.002022**.
- **Speed-ups need qualification:** **1.518379** untuned shrinks to **1.033233** tuned; these cannot be interpreted as measured runtime acceleration without their derivation.
- **Unestablished diagnostics:** no grid surfaces, bootstrap failure counts, or divergence records are supplied. Empty `edge_extension` does not establish interior, convex optima. TPP30 gain is positive (**0.046269**), but TPP120 is not explicitly labeled. Interim gaps peak and then decline; seed “1” initially has **−0.0003**.

### 2. Stage-specific validity threats and cheapest checks

- **Marginal tuned conclusion / seed coverage:** add a paired tuned constant-versus-scaled seed, keeping selected hyperparameters fixed.
- **Tuning-selection bias:** disclose whether validation was independent of selection; if not, use that fresh paired seed as held-out confirmation.
- **Unverified optimum geometry:** inspect existing grids first; extend only a boundary optimum or locally bracket an ambiguous minimum.
- **Bootstrap reliability:** audit resampling units and failed fits; rerun bootstrap on existing data before training.
- **WSD mechanism mismatch:** inspect decay implementation and fit diagnostics; if unresolved, validate a neighboring WD setting.
- **L6 transfer:** inspect existing local sweeps and uncertainty before adding runs.
- **Speed-up, divergence, TPP labeling:** audit calculations and logs; no training required initially.

### 3. Decision consistency

**Yes:** PASS and O1 match the reported point estimates, but the tuned CI does not establish gain above **0.002**.