# weight_decay_ladder

Code and results for the paper:

**Tune the Baseline First: Most of Muon's Scaled Weight Decay Gain Is a Tuning Gap, but Its Small Real Advantage Grows With Model Size**

## Summary

Scaled weight decay for Muon (Muon-SW) is reported to train language models more efficiently than constant decoupled weight decay. Those comparisons usually reuse one learning-rate (LR) and weight-decay (WD) setting for both methods. This repository re-runs the comparison with **both methods tuned**. It follows the MAI-Base-2 protocol: LR and WD are tuned jointly through their product, on a **source-scaled mixture-of-experts (MoE) scaling ladder**.

Main findings (held-out bits per byte, BPB; lower is better):

1. **Most of the gain is a tuning gap.** At the shared setting, Muon-SW wins by 0.035 BPB (about a 1.52x compute saving). After both methods are tuned, the gap is 0.0026 BPB (95% CI [0.0018, 0.0034]), about 1.03x. Roughly 90% of the advantage disappears.
2. **The small advantage that remains is real.** All four paired seeds favour Muon-SW at the tuned settings.
3. **That advantage grows with model size.** From 1.9M to 15M active parameters, the share of the advantage that survives tuning rises from about 10% to 13% to 21%.
4. **The best weight decay moves with scale.** The optimal LR·WD product grows for both methods, faster for Muon-SW (ratio 1.40 to 1.91).

The whole study (81 runs) ran on **one NVIDIA T4 GPU** in Azure ML. A self-resubmitting job chain kept it running through Spot-VM evictions.

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | Single source of truth: paths, Azure settings, data sources, ladder architecture, MAI hyperparameter laws, LR schedules, decay rules, preregistered thresholds. |
| `data_prep.py` | Downloads the three datasets, trains the tokenizer and builds the token pool. |
| `datastream.py` | Source-scaled, step-deterministic data stream used by training. |
| `model.py` | MAI-Base-2-style MoE language model (scaled down). |
| `optim.py` | Muon (Newton–Schulz) for 2D/3D weight matrices plus AdamW for the rest, with constant or scaled weight decay. |
| `train.py` | Trains one ladder run; resumable and pre-emption safe. |
| `plan.py` | Builds the run list for each stage, estimates cost and time, and keeps the spend ledger. |
| `azure_jobs.py` | Azure ML plumbing: clusters, environment, data asset, submission, status, result sync, cancel. |
| `chain.py` | Self-healing GPU job chain: the GPU job is its own watcher, so no compute instance needs to stay on. |
| `analysis.py` | Gates, quadratic fits, hypothesis tests, scale trend, figures and `report.md`. |
| `tests/smoke_test.py` | Unit and end-to-end checks to run before any paid job. |
| `env_conda.yml` / `env_conda_v2.yml` | Job environments (v2 adds `azure-ai-ml` / `azure-identity` for `chain.py`). |
| `orchestrate.ipynb` | Notebook that runs the whole pipeline step by step. |

## What each step does

### Step 0 – Configure (`config.py`)
- Edit the `AZURE` dict for your own subscription, workspace and cluster names. Set `PRICES` and `BUDGET_USD` too.
- **Data sources** (`SOURCES`):

  | Source | Dataset | Share | Repeats |
  |---|---|---|---|
  | web | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | 60% | 1 |
  | code | `codeparrot/codeparrot-clean` | 25% | 4 |
  | math | `open-web-math/open-web-math` | 15% | 4 |

- **Ladder:** each rung is named by its layer count `L` (3, 4, 6, 9), which fixes hidden size, heads and MoE layout. Every rung trains for 120 tokens per active parameter (TPP).
- **MAI laws:** `ln LR* = -1.51 + 0.175 ln TPP - 0.573 ln steps` and `LR*·WD* = 17.04 / steps`. The grid uses multipliers on these values (`lr_mult`, `prod_mult`).
- **Decay rules:**
  - Constant decay: `θ ← θ(1 − η_t·λ)`.
  - Muon-SW: decay scaled by `s(t) = η_t/η_max`, i.e. `θ ← θ(1 − η_t²·λ/η_max)`.
- **Preregistered decisions** (`PREREG`):
  - Gate-1 minimum gain: 0.003 BPB.
  - Edge tolerance for extending the grid.
  - Tuned-gap null threshold: 0.002 BPB.
- **Secrets** are never stored in code. They are read from environment variables or `~/.wdladder_secrets`:
  - `HF_TOKEN` (optional; for rate limits only).

Check the settings:
```bash
python config.py      # prints the ladder, MAI hyperparameters and decay integrals
```

### Step 1 – Smoke tests (`tests/smoke_test.py`)
Run these on the CPU before spending money. They check that:
1. The config laws reproduce MAI's published numbers, and run names are unique.
2. Newton–Schulz returns near-orthogonal matrices (2D and batched 3D experts).
3. The MoE forward/backward pass is dropless, the loss is finite and every parameter gets a gradient.
4. The decay rules are applied exactly: constant vs scaled.
5. The data stream is deterministic and source scaling repeats data the intended number of times.
6. With `--resume`: a run that is killed and resumed matches an uninterrupted run exactly, and a finished run is skipped.

```bash
python tests/smoke_test.py
python tests/smoke_test.py --resume     # needs the smoke pool from step 2
```

### Step 2 – Data (`data_prep.py`, `datastream.py`)
All stages can be resumed after a crash.

| Stage | What it does |
|---|---|
| `fetch` | Streams the three Hugging Face datasets into JSONL shards and drops documents under 200 characters. |
| `tokenizer` | Trains an 8,192-token byte-level BPE (EOS id 0) on a share-weighted sample. |
| `tokenize` | Encodes and deduplicates documents, shuffles them per source, holds out 3,000 eval documents per source, and writes `pool/*.bin`. |
| `verify` | Checks the pool covers every planned run (source scaling needs enough unique tokens). |

```bash
python data_prep.py --smoke all      # tiny pool for tests (~3 min)
python data_prep.py all              # full pool (~1.3B tokens)
python data_prep.py verify           # if LOW: re-fetch with --scale 1.2, delete pool/, re-tokenize
python datastream.py --selftest
```

`datastream.py` implements **source scaling**. A run of T tokens takes `share × T / repeats` unique tokens from each source and loops over them `repeats` times. Every rung therefore sees each source the same number of times. Each batch depends only on (seed, step), so a resumed run gets exactly the same data.

### Step 3 – Model and optimizer (`model.py`, `optim.py`)
- **Model:** follows the MAI-Base-2 layout at small scale:
  - MoE in every layer: 1 full-width shared expert plus top-k routed experts working in a down-projected latent, with dropless routing.
  - Local sliding-window attention, with a global layer every 6 layers and the last layer.
  - GQA, QK-norm, RoPE, RMSNorm, untied embeddings.
  - Loss = cross-entropy + load-balance auxiliary loss + router z-loss.
- **Optimizer:** Muon updates the 2D and 3D expert weights; AdamW updates embeddings, head, router and norms. Weight decay follows the constant or scaled rule from `config.py`.

### Step 4 – Train one run (`train.py`)
```bash
python train.py --run-name L3_t120_cos_scaled_lr1_p1_s0 --L 3 --tpp 120 --sched cos \
  --method scaled --lr-mult 1 --prod-mult 1 --seed 0 --threads 4 \
  --pool /path/to/pool --runs-dir /path/to/runs
```
- **Checkpoints:** saved every 15 minutes and on SIGTERM (eviction). The script resumes automatically and skips runs that are already finished.
- **Exit codes:**
  - `0`: done or diverged.
  - `75`: interrupted; resubmit to continue.
  - `1`: error.
- **Outputs** in `runs/<run>/`:
  - `summary.json`: the final BPB is at `final.bpb`.
  - `progress.json`, `evals.jsonl`, `log.jsonl`, `config.json`.
- **Useful flags:**
  - `--smoke`, `--steps N`: short test runs.
  - `--stop-at-step N`: simulates a pre-emption at step N.
  - `--benchmark N`: measures throughput only and writes `bench.json`.

### Step 5 – Plan, budget and Azure plumbing (`plan.py`, `azure_jobs.py`)
- `plan.py` defines each stage's runs, removes duplicates, estimates GPU/CPU hours and cost from measured throughput, and records spending.
  ```bash
  python plan.py --budget     # cost table per stage (after benchmarks: "measured = True")
  ```
- `azure_jobs.py`:
  - Registers the token pool as a data asset and builds the job environment.
  - Submits runs: it refuses to resubmit runs that are done or active, and refuses batches that would go over `BUDGET_USD`.
  - Syncs result files to the local machine and cancels a stage on request.
  ```bash
  python azure_jobs.py status --stages 1 2
  python azure_jobs.py sync
  python azure_jobs.py cancel --stage 2      # kill switch; checkpoints are kept
  ```

### Step 6 – Self-healing GPU chain (`chain.py`)
On a low-priority (Spot) cluster, an evicted job is not requeued automatically. `chain.py` handles this so that no compute instance has to stay on:
- One GPU job works through a queue of runs one after another.
- When it starts, it submits a **standby successor** under your identity (`UserIdentityConfiguration`). The successor waits in the queue at no cost.
- **If the worker is evicted:** the successor starts, resumes every run from its checkpoint and queues its own successor.
- **If the queue finishes:** the worker cancels its successor.
- **Safety stops:**
  - A heartbeat lease stops two workers from running the same queue.
  - A run that crashes twice is skipped.
  - Four generations in a row without progress stop the chain.
- The queue and state are stored on the datastore at `wdladder/runs/_chains/<chain>/`.

Use it from the notebook:
```python
import chain
chain.ensure_env_v2()                      # once: job environment with azure-ai-ml
chain.add(runs, cid="gpu")                 # append runs; starts a worker if none is active
chain.checkin([1], l6_tpp=120)             # CPU runs -> CPU cluster, GPU runs -> chain
chain.status("gpu")                        # chain state, generations, per-run state
chain.stop("gpu")                          # kill switch (checkpoints kept)
```

### Step 7 – Experiment stages
| Stage | Runs | What it tests |
|---|---|---|
| Benchmarks | 6 | CPU vs T4 throughput per rung. The T4 was about 29x faster, so all training was moved to the GPU. |
| 1 – L3 gate | 6 | Does Muon-SW beat constant decay at the MAI setting by more than seed noise? (PASS: 0.035 BPB) |
| 2 – L3 grid | 39 | 2D sweep over LR × LR·WD with edge extensions, fitting each method's optimum. |
| 3 – L3 confirm + WSD | ~24 | Retrains both methods at the fitted optima with four paired seeds, and runs the WSD schedule check. |
| 4 – L4 | 10 | LR·WD sweep for each method at the second rung. |
| 5 – L6 | 8 | LR·WD sweep for each method at the third rung (about 23 GPU-hours per run). |
| 6 – L9 | – | Planned but not run (compute limit). |

### Step 8 – Analysis (`analysis.py`)
```bash
python analysis.py --stage 3          # gates, fits, tuned gap, survival, hypotheses -> report/report.md
python analysis.py --stage 5          # adds the L4/L6 scale trend and the L6 transfer check
```
`analysis.py`:
- Applies the preregistered gates.
- Fits quadratic optima for each method.
- Computes the untuned gap, tuned gap (with bootstrap CI) and surviving share.
- Tests the three hypotheses for the ratio of optima: no change (1), matched integrated decay (about 1.36 for cosine) and endpoint matching (10).
- Writes the figures and `report.md`.

Before running the analysis, move the smoke-test and benchmark run folders out of `runs/` so they aren't counted as experiment runs.

## Results

| Rung | Active params | Untuned gap | Tuned gap | Surviving share | Optimum ratio (SW / const) |
|---|---|---|---|---|---|
| L3 | 1.9M | 0.0350 | 0.0036 | 10.3% | 1.40 |
| L4 | 3.9M | 0.0311 | 0.0040 | 13.0% | 1.63 |
| L6 | 15.0M | 0.0241 | 0.0051 | 21.3% | 1.91 |

L3 full 2D fit with four paired seeds: tuned gap 0.00261 BPB (95% CI [0.00183, 0.00338]), surviving share 7.4%.

**Limitations:**
- L4 and L6 use one seed each.
- Some fitted optima at L4 and L6 lie at the edge of the sweep.
- The WSD weight-decay ratio (0.29) matches none of the three hypotheses.
- The largest model has 15M active parameters.

## Requirements

- Python 3.10, PyTorch 2.4 (CUDA 11.8 for the T4, or CPU), `numpy<2`, `scipy`, `pandas`, `pyarrow`.
- Data preparation: `datasets`, `tokenizers`.
- Azure: `azure-ai-ml`, `azure-identity`, `azure-storage-blob`.

