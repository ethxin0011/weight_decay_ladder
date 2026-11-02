"""
plan.py - run lists per stage, cost model from measured benchmarks, spend ledger (torch-free)
==============================================================================================
  python plan.py --budget            # full projected spend by stage (uses runs/bench_* if present)
  python plan.py --stage 2           # list runs of stage 2 with est. hours / $
Stages (each later stage is only built after the previous gate passes):
  1  L3 TPP120 cos: const vs scaled at MAI hparams, seeds 0,1 (+TPP30 seed 0)          -> GATE 1
  2  L3 grid in MAI coords (ln LR, ln LR*WD), 3 x 5 per method + const TPP60/240 refs     -> 2-D fits
  3  L3 confirm at fitted optima (seeds 1,2) + WSD 1-D product sweep                       -> GATE 2
  4  L4 (GPU) 1-D product sweep at MAI LR, 5 per method                                    -> ratio vs scale
  5  L6 (GPU) 4 points per method around the predicted optimum (TPP 60 default)            -> transfer test
  6  optional L9 TPP30: const@MAI vs scaled@predicted optimum
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import config as C

GRID_LR = [0.5, 1.0, 2.0]
GRID_P = {"const": [0.25, 0.5, 1.0, 2.0, 4.0], "scaled": [0.5, 1.0, 2.0, 4.0, 8.0]}
L6_OFFSETS = [0.354, 0.707, 1.414, 2.83]


def fmt(x: float) -> str:
    return f"{x:.3g}"


def run_name(L, tpp, sched, method, lr_mult, prod_mult, seed):
    return f"L{L}_t{int(tpp)}_{sched}_{method}_lr{fmt(lr_mult)}_p{fmt(prod_mult)}_s{seed}"


def spec(stage, L, tpp, sched, method, lr_mult=1.0, prod_mult=1.0, seed=0, device=None):
    return dict(name=run_name(L, tpp, sched, method, lr_mult, prod_mult, seed), stage=stage, L=L, tpp=tpp,
                sched=sched, method=method, lr_mult=float(fmt(lr_mult)), prod_mult=float(fmt(prod_mult)),
                seed=seed, device=device or C.PLACEMENT.get(L, "gpu"))


def tokens_of(s) -> int:
    h = C.run_hparams(s["L"], s["tpp"])
    return h["steps"] * C.RUNG_DEFAULTS[s["L"]]["batch_tokens"]


# ----------------------------------------------------------------- stages
def stage1():
    r = [spec(1, 3, 120, "cos", m, 1, 1, s) for m in ("const", "scaled") for s in (0, 1)]
    r += [spec(1, 3, 30, "cos", m, 1, 1, 0) for m in ("const", "scaled")]
    return r


def stage2(extra=None):
    r = [spec(2, 3, 120, "cos", m, lr, p, 0) for m in ("const", "scaled") for lr in GRID_LR for p in GRID_P[m]]
    r += [spec(2, 3, t, "cos", "const", 1, 1, 0) for t in (60, 240)]
    for e in extra or []:                      # edge extensions from analysis (report/edge_ext.json)
        r.append(spec(2, 3, 120, "cos", e["method"], e["lr_mult"], e["prod_mult"], 0))
    done1 = {x["name"] for x in stage1()}                  # centre points reused from stage 1
    return [x for x in dedup(r) if x["name"] not in done1]


def stage3(fits):
    r = []
    for m in ("const", "scaled"):
        o = fits["L3_cos_2d"][m]["opt"]
        r += [spec(3, 3, 120, "cos", m, o["lr_mult"], o["prod_mult"], s) for s in (1, 2)]
        r += [spec(3, 3, 120, "wsd", m, 1, p, 0) for p in GRID_P[m]]
    return dedup(r)


def stage4():
    return [spec(4, 4, 120, "cos", m, 1, p, 0) for m in ("const", "scaled") for p in GRID_P[m]]


def stage5(fits, tpp=60):
    r = []
    for m in ("const", "scaled"):
        pstar = fits.get("pred_L6", {}).get(m, {}).get("prod_mult", 1.0 if m == "const" else 2.0)
        r += [spec(5, 6, tpp, "cos", m, 1, pstar * k, 0) for k in L6_OFFSETS]
    return r


def stage6(fits):
    ps = fits.get("pred_L9", {}).get("scaled", {}).get("prod_mult", 2.0)
    return [spec(6, 9, 30, "cos", "const", 1, 1, 0), spec(6, 9, 30, "cos", "scaled", 1, ps, 0)]


def dedup(runs):
    seen, out = set(), []
    for r in runs:
        if r["name"] not in seen:
            seen.add(r["name"])
            out.append(r)
    return out


def all_planned(fits=None, l6_tpp=120):
    fits = fits or {"L3_cos_2d": {m: {"opt": {"lr_mult": 1, "prod_mult": 1}} for m in ("const", "scaled")}}
    return stage1() + stage2() + stage3(fits) + stage4() + stage5(fits, l6_tpp) + stage6(fits)


def pool_requirements() -> dict:
    """max unique tokens per source over every planned run (incl. L6 at TPP120 and L9) + window slack."""
    need = {s: 0 for s in C.SOURCES}
    for r in all_planned():
        T = tokens_of(r) * (C.SEQ_LEN + 1) / C.SEQ_LEN
        for s, sp in C.SOURCES.items():
            need[s] = max(need[s], int(T * sp["share"] / sp["repeats"]))
    return need


# ----------------------------------------------------------------- cost model
DEFAULT_TPS = {("cpu", 3): 6000, ("cpu", 4): 3500, ("gpu", 3): 45000, ("gpu", 4): 35000,
               ("gpu", 6): 15000, ("gpu", 9): 6000}      # placeholders until benchmarks exist


def load_bench(runs_dir=None) -> dict:
    runs_dir = runs_dir or C.P["runs"]
    tps = {}
    for f in glob.glob(os.path.join(runs_dir, "bench_*", "bench.json")):
        b = json.load(open(f))
        dev = "gpu" if b["device"] == "cuda" else "cpu"
        tps[(dev, b["L"])] = max(tps.get((dev, b["L"]), 0), b["tok_per_s"])
    return tps


def estimate(r, tps=None) -> dict:
    tps = tps or {}
    key = (r["device"], r["L"])
    rate = tps.get(key) or DEFAULT_TPS.get(key, 3000)
    hours = tokens_of(r) / rate / 3600 * 1.06          # +6% periodic eval
    if r["device"] == "cpu":
        usd = hours * C.PRICES["cpu_node"] / C.CPU_RUNS_PER_NODE
    else:
        usd = hours * C.PRICES["gpu_node"]
    return dict(hours=hours, usd=usd * C.PRICES["overhead"], measured=key in tps)


# ----------------------------------------------------------------- ledger
def _ledger_path():
    return os.path.join(C.P["root"], "ledger.json")


def ledger_load(path=None):
    path = path or _ledger_path()
    return json.load(open(path)) if os.path.exists(path) else {"submissions": [], "instance_hours": 0.0}


def ledger_save(lg, path=None):
    path = path or _ledger_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(lg, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)


def actual_spend(runs_dir=None, instance_hours=0.0) -> dict:
    """$ from wall-clock actually used (progress.json of every run) + instance hours you log."""
    runs_dir = runs_dir or C.P["runs"]
    cpu = gpu = 0.0
    for f in glob.glob(os.path.join(runs_dir, "*", "progress.json")):
        p = json.load(open(f))
        h = p.get("wall_seconds", 0) / 3600
        if p.get("device") == "cuda":
            gpu += h * C.PRICES["gpu_node"]
        else:
            cpu += h * C.PRICES["cpu_node"] / C.CPU_RUNS_PER_NODE
    inst = instance_hours * C.PRICES["instance"]
    tot = (cpu + gpu) * C.PRICES["overhead"] + inst
    return dict(cpu=cpu, gpu=gpu, instance=inst, total=tot)


def budget_table(tps=None, l6_tpp=60):
    rows, cum = [], 0.0
    fits = None
    stages = [("1 L3 gate", stage1()), ("2 L3 grid", stage2()),
              ("3 L3 confirm+WSD", stage3({"L3_cos_2d": {m: {"opt": {"lr_mult": 1.3, "prod_mult": 1.7}}
                                                          for m in ("const", "scaled")}})),
              ("4 L4 GPU", stage4()), (f"5 L6 GPU tpp{l6_tpp}", stage5({}, l6_tpp)), ("6 L9 GPU (opt)", stage6({}))]
    for nm, runs in stages:
        e = [estimate(r, tps) for r in runs]
        usd = sum(x["usd"] for x in e)
        cum += usd
        rows.append((nm, len(runs), sum(x["hours"] for x in e), usd, cum, all(x["measured"] for x in e)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--l6-tpp", type=int, default=60)
    a = ap.parse_args()
    tps = load_bench()
    print("throughput (tok/s):", {f"{k[0]}-L{k[1]}": round(v) for k, v in (tps or DEFAULT_TPS).items()},
          "(measured)" if tps else "(DEFAULT placeholders - run benchmarks!)")
    print("pool requirement (M unique tokens):", {k: round(v / 1e6) for k, v in pool_requirements().items()})
    if a.budget or not a.stage:
        print(f"\n{'stage':22s} {'runs':>4} {'run-hours':>9} {'$ stage':>8} {'$ cum':>7} measured")
        for nm, n, h, u, c, m in budget_table(tps, a.l6_tpp):
            print(f"{nm:22s} {n:>4} {h:>9.1f} {u:>8.2f} {c:>7.2f} {m}")
        print(f"budget ${C.BUDGET_USD:.0f}; add compute-instance hours x ${C.PRICES['instance']}/h (stop it when idle)")
    if a.stage:
        fits = json.load(open(os.path.join(C.P["report"], "fits.json"))) if a.stage in (3, 5, 6) else None
        runs = {1: stage1, 2: stage2, 4: stage4}.get(a.stage)
        runs = runs() if runs else {3: lambda: stage3(fits), 5: lambda: stage5(fits, a.l6_tpp),
                                    6: lambda: stage6(fits)}[a.stage]()
        for r in runs:
            e = estimate(r, tps)
            print(f"{r['name']:45s} {r['device']} {e['hours']:6.1f} h ${e['usd']:.2f}")
