### 1. Anomalies and missing diagnostics

- **Seed noise versus effect:** No apparent anomaly. Mean gain is **0.03499458361150165**, versus reported noise **0.0010182919981824856**; both seed gains are positive and nearly identical. However, `seed_sd` is **0.0007066882225594238**, so the two noise summaries need their definitions clarified.
- **Trajectory:** Gains peak around step **4851** (**0.0451**, **0.0441**) and decline to **0.035** at **6930**. This flags endpoint dependence, not instability or divergence.
- **Non-convex/edge optima:** Not assessable without tuning-grid results. `H_end = 10.0` is a hypothesis value, not evidence of an edge optimum.
- **Bootstrap failures / divergence:** Neither bootstrap diagnostics nor run-status logs are supplied. No failure is demonstrated, but absence cannot be verified.
- **Speed-up:** **1.4425162600178956** is unsupported by timing, compute accounting, or the loss-to-compute conversion used. Treat it as unvalidated.
- **TPP signs:** `gain_tpp30 = 0.046268982280819326` and the main gains are positive. There is no explicitly labeled TPP120 result, so cross-TPP consistency is not fully auditable.

### 2. Stage-specific threats and cheapest checks

- **Untuned baseline disadvantage:** This establishes an untuned comparison, not superiority after tuning. **Cheapest resolving experiment:** the planned matched local tuning comparison; a single arbitrary run cannot settle this.
- **Decay amount versus schedule shape:** Scaling decay changes effective regularization as well as its timing. **Cheapest extra run:** constant decay matched to Muon-SW’s integrated effective decay, with other settings fixed.
- **Limited seed evidence:** Close supplied gains do not establish robustness across training randomness. **Cheapest extra experiment:** another matched-seed pair.
- **Endpoint dependence:** The declining late gap threatens extrapolation beyond this budget. **Cheapest extra experiment:** continue a matched pair beyond the existing endpoint.
- **Unverified speed-up and diagnostics:** **Cheapest check:** audit existing timing, checkpoint, divergence, and bootstrap records before spending compute.
- **Unclear ladder coverage:** **Cheapest check:** label the existing configurations; if TPP120 is absent, add a matched comparison there.

### 3. Decision consistency

**Yes.** The positive mean gain exceeds the reported **0.003** threshold and is large relative to reported noise, consistent with **PASS** and proceeding to Stage 2—not yet a tuned-superiority claim.