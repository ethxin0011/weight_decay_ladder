"""
analysis.py - gates, fits, hypothesis tests, figures, report (torch-free; runs on the instance)
===============================================================================================
  python azure_jobs.py sync            # first: pull small result files
  python analysis.py --stage 1         # gate 1 (+ interim gap trajectories while runs are in flight)
  python analysis.py --stage 2         # 2-D quadratic fits in MAI coords, edge check -> edge_ext.json
  python analysis.py --stage 3         # tuned gap (confirm runs), WSD, K-constancy, hypothesis test
  python analysis.py --stage 4|5|6     # scale transfer + predictions for the next rung
  python analysis.py --selftest        # synthetic data: checks the fitting code recovers a known optimum
Outputs in $WDL_DATA/report: results.json, fits.json, edge_ext.json, report.md, fig_*.png
Decisions are PRE-REGISTERED in config.PREREG; LLM review (llm_review.py) never changes them.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

import config as C

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------- loading
def load_runs(runs_dir=None):
    runs_dir = runs_dir or C.P["runs"]
    rows = []
    for f in glob.glob(os.path.join(runs_dir, "*", "config.json")):
        d = os.path.dirname(f)
        cfg = json.load(open(f))
        sp = os.path.join(d, "summary.json")
        s = json.load(open(sp)) if os.path.exists(sp) else {}
        fin = s.get("final") or {}
        rows.append(dict(name=os.path.basename(d), L=cfg["L"], tpp=float(cfg["tpp"]), sched=cfg["sched"],
                         method=cfg["method"], lr_mult=float(cfg["lr_mult"]), prod_mult=float(cfg["prod_mult"]),
                         seed=int(cfg["seed"]), steps=cfg["steps"], lr=cfg["lr"], wd=cfg["wd"],
                         tokens=cfg["steps"] * cfg["batch_tokens"], flops_tok=cfg["flops_per_token"],
                         active=cfg["params"]["active_nonembed"], status=s.get("status", "running"),
                         bpb=fin.get("bpb", np.nan), **{f"bpb_{k}": fin.get(k, {}).get("bpb", np.nan)
                                                         for k in C.SOURCES}, dir=d))
    return rows


def sel(rows, **kw):
    out = []
    for r in rows:
        if r["status"] != "done":
            continue
        if all((abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v) for k, v in kw.items()):
            out.append(r)
    return out


def one(rows, **kw):
    x = sel(rows, **kw)
    return x[0]["bpb"] if x else np.nan


def evals(d):
    p = os.path.join(d, "evals.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ----------------------------------------------------------------- compute-equivalent speed-up
def data_law(points):
    """points [(tokens, bpb)] of the const baseline at MAI hparams -> callable bpb->tokens."""
    pts = sorted(points)
    D = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    if len(pts) >= 4:
        from scipy.optimize import curve_fit
        try:
            f = lambda d, E, A, a: E + A * (d / 1e8) ** (-a)
            (E, A, a), _ = curve_fit(f, D, y, p0=[y.min() * 0.8, y.max() - y.min() * 0.8, 0.3], maxfev=20000)
            if A > 0 and a > 0 and E < y.min():
                return (lambda b: 1e8 * ((b - E) / A) ** (-1 / a) if b > E else np.inf), dict(E=E, A=A, alpha=a)
        except Exception:
            pass
    k = np.polyfit(np.log(D), np.log(y), 1)            # log-log interpolation / mild extrapolation
    return (lambda b: math.exp((math.log(b) - k[1]) / k[0])), dict(loglog=k.tolist())


def speedup(rows, bpb, L=3, sched="cos"):
    base = [(r["tokens"], r["bpb"]) for r in sel(rows, L=L, sched=sched, method="const", lr_mult=1.0,
                                                  prod_mult=1.0, seed=0)]
    if len(base) < 2:
        return np.nan, {}
    inv, law = data_law(base)
    ref = sel(rows, L=L, sched=sched, method="const", lr_mult=1.0, prod_mult=1.0, tpp=120.0)
    if not ref:
        return np.nan, law
    return inv(bpb) / ref[0]["tokens"], law


# ----------------------------------------------------------------- gate 1
def gate1(rows):
    P = C.PREREG
    c = [one(rows, L=3, tpp=120.0, sched="cos", method="const", lr_mult=1.0, prod_mult=1.0, seed=s) for s in (0, 1)]
    s = [one(rows, L=3, tpp=120.0, sched="cos", method="scaled", lr_mult=1.0, prod_mult=1.0, seed=k) for k in (0, 1)]
    if np.isnan(c + s).any():
        return dict(decision="WAIT", reason="stage-1 runs not finished", const=c, scaled=s)
    gains = [c[i] - s[i] for i in range(2)]
    noise = max(abs(c[0] - c[1]), abs(s[0] - s[1]))
    thr = max(P["gate1_noise_mult"] * noise, P["gate1_min_gain_bpb"])
    g = float(np.mean(gains))
    g30 = one(rows, L=3, tpp=30.0, method="const", seed=0, lr_mult=1.0, prod_mult=1.0) - \
        one(rows, L=3, tpp=30.0, method="scaled", seed=0, lr_mult=1.0, prod_mult=1.0)
    sp, _ = speedup(rows, float(np.mean(s)))
    if g > thr and min(gains) > 0:
        dec = "PASS"
    elif g > 0 and min(gains) > 0:
        dec = "WEAK"         # positive but inside noise: stage 2 still informative (tuned-vs-untuned), cheaper plan
    else:
        dec = "FAIL"
    return dict(decision=dec, gain_mean=g, gains=gains, seed_noise=noise, threshold=thr, gain_tpp30=g30,
                const=c, scaled=s, speedup_untuned=sp,
                note="PASS -> stage 2; WEAK -> stage 2 grid only (skip GPU until gate 2); FAIL -> stop, "
                     "write the negative replication note (Muon-SW gain absent on MAI-style MoE ladder)")


def interim_gap(rows):
    """paired BPB gap (const - scaled) at matched eval steps for in-flight stage-1 runs."""
    out = {}
    for seed in (0, 1):
        cs = [r for r in rows if r["L"] == 3 and r["tpp"] == 120 and r["sched"] == "cos" and r["seed"] == seed
              and r["lr_mult"] == 1 and r["prod_mult"] == 1]
        e = {r["method"]: {x["step"]: x["bpb"] for x in evals(r["dir"])} for r in cs}
        if "const" in e and "scaled" in e:
            out[seed] = {k: round(e["const"][k] - e["scaled"][k], 4) for k in sorted(set(e["const"]) & set(e["scaled"]))}
    return out


# ----------------------------------------------------------------- quadratic fits
def _design2(x, y):
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)


def _opt2(b):
    H = np.array([[2 * b[3], b[4]], [b[4], 2 * b[5]]])
    if np.linalg.eigvalsh(H).min() <= 0:
        return None
    xy = np.linalg.solve(H, -b[1:3])
    return xy, float((_design2(xy[:1], xy[1:]) @ b)[0])


def fit2d(pts, n_boot=2000):
    """pts: [(ln lr_mult, ln prod_mult, bpb)] -> optimum in MAI multipliers with bootstrap CI."""
    a = np.array(pts, float)
    X, z = _design2(a[:, 0], a[:, 1]), a[:, 2]
    b, *_ = np.linalg.lstsq(X, z, rcond=None)
    o = _opt2(b)
    res = z - X @ b
    boots = []
    for _ in range(n_boot):                       # residual bootstrap
        bb, *_ = np.linalg.lstsq(X, X @ b + RNG.choice(res, len(res)), rcond=None)
        ob = _opt2(bb)
        if ob is not None:
            boots.append([ob[0][0], ob[0][1], ob[1]])
    boots = np.array(boots) if boots else np.zeros((0, 3))
    ci = lambda k: [float(np.exp(np.percentile(boots[:, k], 2.5))), float(np.exp(np.percentile(boots[:, k], 97.5)))] \
        if len(boots) > 50 else [np.nan, np.nan]
    out = dict(coef=b.tolist(), rmse=float(np.sqrt(np.mean(res ** 2))), n=len(pts), convex=o is not None,
               boot_ok=len(boots) / n_boot)
    if o is not None:
        out["opt"] = dict(lr_mult=float(np.exp(o[0][0])), prod_mult=float(np.exp(o[0][1])), bpb=o[1],
                          lr_ci=ci(0), prod_ci=ci(1),
                          bpb_ci=[float(np.percentile(boots[:, 2], 2.5)), float(np.percentile(boots[:, 2], 97.5))]
                          if len(boots) > 50 else [np.nan, np.nan])
        # ridge geometry (MAI coords): principal axes of the Hessian
        H = np.array([[2 * b[3], b[4]], [b[4], 2 * b[5]]])
        w, v = np.linalg.eigh(H)
        out["ridge"] = dict(angle_deg=float(np.degrees(np.arctan2(v[1, 0], v[0, 0]))),
                            width_flat=float(math.sqrt(2 * 0.01 / w[0])), width_steep=float(math.sqrt(2 * 0.01 / w[1])))
    return out


def fit1d(pts, n_boot=2000):
    """pts: [(ln prod_mult, bpb)] -> optimum prod_mult with bootstrap CI."""
    a = np.array(pts, float)
    if len(a) < 3:
        return dict(n=len(a), convex=False)
    X = np.stack([np.ones(len(a)), a[:, 0], a[:, 0] ** 2], 1)
    b, *_ = np.linalg.lstsq(X, a[:, 1], rcond=None)
    res = a[:, 1] - X @ b
    def opt(bb):
        return None if bb[2] <= 0 else -bb[1] / (2 * bb[2])
    o = opt(b)
    boots = [opt(np.linalg.lstsq(X, X @ b + RNG.choice(res, len(res)), rcond=None)[0]) for _ in range(n_boot)]
    boots = np.array([x for x in boots if x is not None])
    out = dict(n=len(a), convex=o is not None, coef=b.tolist())
    if o is not None:
        out.update(prod_mult=float(np.exp(o)), bpb=float(b[0] + b[1] * o + b[2] * o * o),
                   prod_ci=[float(np.exp(np.percentile(boots, 2.5))), float(np.exp(np.percentile(boots, 97.5)))]
                   if len(boots) > 50 else [np.nan, np.nan], edge=bool(o < a[:, 0].min() or o > a[:, 0].max()))
    return out


def edge_check(fit, lrs, ps, method):
    """optimum within tolerance of / beyond the grid edge -> propose one extra row/column (5 or 3 runs)."""
    if "opt" not in fit:
        return [dict(method=method, lr_mult=l, prod_mult=max(ps) * 2) for l in lrs]   # non-convex: widen
    tol = C.PREREG["edge_tolerance"]
    x, y = math.log(fit["opt"]["lr_mult"]), math.log(fit["opt"]["prod_mult"])
    lx, ly = np.log(lrs), np.log(ps)
    sx, sy = math.log(2), math.log(2)
    ext = []
    if y > ly.max() - tol * sy:
        ext += [dict(method=method, lr_mult=l, prod_mult=max(ps) * 2) for l in lrs]
    if y < ly.min() + tol * sy:
        ext += [dict(method=method, lr_mult=l, prod_mult=min(ps) / 2) for l in lrs]
    if x > lx.max() - tol * sx:
        ext += [dict(method=method, lr_mult=max(lrs) * 2, prod_mult=p) for p in ps]
    if x < lx.min() + tol * sx:
        ext += [dict(method=method, lr_mult=min(lrs) / 2, prod_mult=p) for p in ps]
    return ext


def wd_ratio(o_s, o_c):
    """lambda*_scaled / lambda*_const from optima in MAI multipliers (wd = prod/lr)."""
    return (o_s["prod_mult"] / o_s["lr_mult"]) / (o_c["prod_mult"] / o_c["lr_mult"])


def k_rel(prod_mult, sched, method, steps=2000):
    """integrated decay at the optimum relative to MAI's law: prod_mult * mean_t(f_t s_t).
    If the 'integrated decay' law holds, this is the same number for every schedule and method."""
    return prod_mult * C.decay_integral(steps, sched, method)


# ----------------------------------------------------------------- stage analyses
def _grid_pts(rows, L, sched, method, tpp=120.0, seed=0):
    acc = {}
    for r in sel(rows, L=L, sched=sched, method=method, tpp=tpp, seed=seed):
        acc.setdefault((r["lr_mult"], r["prod_mult"]), []).append(r["bpb"])
    return [(math.log(k[0]), math.log(k[1]), float(np.mean(v))) for k, v in acc.items()]


def seed_sd(rows):
    """sd of a single run's BPB from all (config, seed) replicates available."""
    groups = {}
    for r in sel(rows):
        groups.setdefault((r["L"], r["tpp"], r["sched"], r["method"], r["lr_mult"], r["prod_mult"]), []).append(r["bpb"])
    devs = [np.var(v, ddof=1) for v in groups.values() if len(v) >= 2]
    return float(np.sqrt(np.mean(devs))) if devs else np.nan


def slice_opt_lr1(fit):
    b = fit["coef"]
    return math.exp(-b[2] / (2 * b[5])) if b[5] > 0 else np.nan


def analyse(rows, upto):
    R, F = {}, {}
    os.makedirs(C.P["report"], exist_ok=True)
    fp = os.path.join(C.P["report"], "fits.json")
    if os.path.exists(fp):
        F = json.load(open(fp))
    R["gate1"] = gate1(rows)
    R["interim_gap"] = interim_gap(rows)
    R["seed_sd"] = seed_sd(rows)
    R["hypotheses"] = {s: C.hypothesis_ratios(s) for s in ("cos", "wsd")}
    if upto >= 2:
        F["L3_cos_2d"], ext = {}, []
        for m in ("const", "scaled"):
            pts = _grid_pts(rows, 3, "cos", m)
            if len(pts) >= 8:
                F["L3_cos_2d"][m] = fit2d(pts)
                lrs = sorted({round(math.exp(p[0]), 4) for p in pts})
                ps = sorted({round(math.exp(p[1]), 4) for p in pts})
                ext += edge_check(F["L3_cos_2d"][m], lrs, ps, m)
        R["edge_extension"] = ext
        json.dump(ext, open(os.path.join(C.P["report"], "edge_ext.json"), "w"), indent=1)
        if all("opt" in F["L3_cos_2d"].get(m, {}) for m in ("const", "scaled")):
            oc, os_ = F["L3_cos_2d"]["const"]["opt"], F["L3_cos_2d"]["scaled"]["opt"]
            R["wd_ratio_cos"] = wd_ratio(os_, oc)
            R["wd_ratio_cos_ci"] = [(os_["prod_ci"][0] / os_["lr_ci"][1]) / (oc["prod_ci"][1] / oc["lr_ci"][0]),
                                    (os_["prod_ci"][1] / os_["lr_ci"][0]) / (oc["prod_ci"][0] / oc["lr_ci"][1])]
            R["K_rel_cos"] = {m: k_rel(F["L3_cos_2d"][m]["opt"]["prod_mult"], "cos", m) for m in ("const", "scaled")}
            R["optimal_bpb_fit"] = {m: F["L3_cos_2d"][m]["opt"]["bpb"] for m in ("const", "scaled")}
        sp = {}
        for m in ("const", "scaled"):
            if "opt" in F["L3_cos_2d"].get(m, {}):
                sp[m] = speedup(rows, F["L3_cos_2d"][m]["opt"]["bpb"])[0]
        R["speedup_at_fitted_opt"] = sp
        R["data_law"] = speedup(rows, 1.0)[1]
    if upto >= 3:
        sd = R["seed_sd"] if not np.isnan(R["seed_sd"]) else 0.002
        conf = {}
        for m in ("const", "scaled"):
            o = F.get("L3_cos_2d", {}).get(m, {}).get("opt")
            if o:
                import plan
                lm, pm = float(plan.fmt(o["lr_mult"])), float(plan.fmt(o["prod_mult"]))
                conf[m] = [r["bpb"] for r in sel(rows, L=3, tpp=120.0, sched="cos", method=m, lr_mult=lm, prod_mult=pm)
                           if r["seed"] in (1, 2)]
        if all(len(conf.get(m, [])) >= 1 for m in ("const", "scaled")):
            gap = float(np.mean(conf["const"]) - np.mean(conf["scaled"]))
            se = sd * math.sqrt(1 / len(conf["const"]) + 1 / len(conf["scaled"]))
            R["tuned_gap"] = dict(gap=gap, ci=[gap - 1.96 * se, gap + 1.96 * se], const=conf["const"], scaled=conf["scaled"])
            ug = R["gate1"].get("gain_mean", np.nan)
            R["survival_fraction"] = gap / ug if ug and ug > 0 else np.nan
            R["speedup_tuned"] = speedup(rows, float(np.mean(conf["scaled"])))[0] / \
                speedup(rows, float(np.mean(conf["const"])))[0]
        F["L3_wsd_1d"] = {m: fit1d([(math.log(r["prod_mult"]), r["bpb"]) for r in
                                    sel(rows, L=3, tpp=120.0, sched="wsd", method=m, lr_mult=1.0, seed=0)])
                          for m in ("const", "scaled")}
        w = F["L3_wsd_1d"]
        if all(w[m].get("convex") for m in w):
            R["wd_ratio_wsd"] = w["scaled"]["prod_mult"] / w["const"]["prod_mult"]
            R["K_rel_wsd"] = {m: k_rel(w[m]["prod_mult"], "wsd", m) for m in w}
        R["hypothesis_verdict"] = verdict(R)
    if upto >= 4:
        F["L4_cos_1d"] = {m: fit1d([(math.log(r["prod_mult"]), r["bpb"]) for r in
                                    sel(rows, L=4, tpp=120.0, sched="cos", method=m, lr_mult=1.0, seed=0)])
                          for m in ("const", "scaled")}
        F["pred_L6"], F["pred_L9"] = {}, {}
        a3, a4 = C.param_counts(C.rung_config(3))["active_nonembed"], C.param_counts(C.rung_config(4))["active_nonembed"]
        for m in ("const", "scaled"):
            p3 = slice_opt_lr1(F["L3_cos_2d"][m]) if m in F.get("L3_cos_2d", {}) else np.nan
            p4 = F["L4_cos_1d"][m].get("prod_mult", np.nan)
            for L, key in ((6, "pred_L6"), (9, "pred_L9")):
                aL = C.param_counts(C.rung_config(L))["active_nonembed"]
                if np.isfinite(p3) and np.isfinite(p4):
                    slope = (math.log(p4) - math.log(p3)) / (math.log(a4) - math.log(a3))
                    pred = math.exp(math.log(p4) + slope * (math.log(aL) - math.log(a4)))
                    pred = float(np.clip(pred, p4 / 4, p4 * 4))
                else:
                    pred = p4 if np.isfinite(p4) else (p3 if np.isfinite(p3) else 1.0)
                F[key][m] = dict(prod_mult=pred, from_L3=p3, from_L4=p4)
        R["scale_trend"] = {m: dict(L3=F["pred_L6"][m]["from_L3"], L4=F["pred_L6"][m]["from_L4"],
                                    L6_pred=F["pred_L6"][m]["prod_mult"]) for m in ("const", "scaled")}
    if upto >= 5:
        F["L6_1d"], R["L6"] = {}, {}
        for m in ("const", "scaled"):
            pts = [(math.log(r["prod_mult"]), r["bpb"], r["tpp"]) for r in sel(rows, L=6, sched="cos", method=m, seed=0)]
            if pts:
                f = fit1d([p[:2] for p in pts])
                F["L6_1d"][m] = f
                best = min(p[1] for p in pts)
                pp = F["pred_L6"][m]["prod_mult"]
                at_pred = (f["coef"][0] + f["coef"][1] * math.log(pp) + f["coef"][2] * math.log(pp) ** 2) if f.get("coef") else np.nan
                R["L6"][m] = dict(best=best, at_pred=at_pred, regret=at_pred - min(best, f.get("bpb", best)),
                                  fitted=f.get("prod_mult"), pred=pp)
        if all(m in R["L6"] for m in ("const", "scaled")):
            R["L6"]["tuned_gap"] = min(R["L6"]["const"]["best"], R["L6"]["const"]["at_pred"]) - \
                min(R["L6"]["scaled"]["best"], R["L6"]["scaled"]["at_pred"])
    if upto >= 6:
        a = sel(rows, L=9, method="const")
        b = sel(rows, L=9, method="scaled")
        if a and b:
            R["L9"] = dict(const=a[0]["bpb"], scaled=b[0]["bpb"], gap=a[0]["bpb"] - b[0]["bpb"])
    R["outcome"] = outcome(R)
    os.makedirs(C.P["report"], exist_ok=True)
    json.dump(F, open(fp, "w"), indent=1, default=float)
    return R, F


def verdict(R):
    """which hypothesis about lambda*_scaled/lambda*_const is consistent with the fitted ratio + CI."""
    out = {}
    for s in ("cos", "wsd"):
        r = R.get(f"wd_ratio_{s}")
        if r is None or not np.isfinite(r):
            continue
        ci = R.get(f"wd_ratio_{s}_ci", [r / 1.5, r * 1.5])
        h = R["hypotheses"][s]
        out[s] = dict(measured=r, ci=ci, consistent=[k for k in ("H_none", "H_int", "H_end") if ci[0] <= h[k] <= ci[1]],
                      closest=min(("H_none", "H_int", "H_end"), key=lambda k: abs(math.log(h[k]) - math.log(r))))
    return out


def outcome(R):
    g = R["gate1"]
    if g["decision"] == "FAIL":
        return "O3 non-replication: Muon-SW shows no gain over const decay at MAI hparams on the MoE ladder"
    if g["decision"] == "WAIT":
        return "pending"
    t = R.get("tuned_gap")
    if not t:
        return "O0 untuned gain confirmed (gate 1 %s); tuned comparison pending" % g["decision"]
    if t["ci"][0] > 0 and t["gap"] > C.PREREG["tuned_gain_null_bpb"]:
        return "O1 genuine: Muon-SW keeps %.0f%% of its gain after both methods are tuned" % (100 * R["survival_fraction"])
    if t["ci"][1] < 0:
        return "O4 reversal: tuned const decay beats tuned Muon-SW"
    return "O2 calibration artifact: gain disappears once decay is re-tuned (MAI LR*WD law + fitted multiplier suffices)"


# ----------------------------------------------------------------- figures + report
def figures(rows, F, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = C.P["report"]
    if F.get("L3_cos_2d"):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, m in zip(axs, ("const", "scaled")):
            f = F["L3_cos_2d"].get(m)
            pts = np.array(_grid_pts(rows, 3, "cos", m))
            if f is None or not len(pts):
                continue
            xs = np.linspace(pts[:, 0].min() - .3, pts[:, 0].max() + .3, 60)
            ys = np.linspace(pts[:, 1].min() - .3, pts[:, 1].max() + .3, 60)
            X, Y = np.meshgrid(xs, ys)
            Z = (_design2(X.ravel(), Y.ravel()) @ np.array(f["coef"])).reshape(X.shape)
            cs = ax.contour(X, Y, Z, 15)
            ax.clabel(cs, fontsize=6)
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="viridis", edgecolor="k")
            if "opt" in f:
                ax.plot(math.log(f["opt"]["lr_mult"]), math.log(f["opt"]["prod_mult"]), "r*", ms=14)
            ax.set(title=f"L3 {m}: BPB in MAI coords", xlabel="ln(LR / LR*_MAI)", ylabel="ln(LR*WD / (17.04/steps))")
            plt.colorbar(sc, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(rep, "fig_grid2d.png"), dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, ls in (("L4_cos_1d", "-"), ("L6_1d", ":"), ("L3_wsd_1d", "--")):
        L = {"L4_cos_1d": 4, "L6_1d": 6, "L3_wsd_1d": 3}[key]
        sch = "wsd" if "wsd" in key else "cos"
        for m, col in (("const", "C0"), ("scaled", "C3")):
            pts = sorted((r["prod_mult"], r["bpb"]) for r in sel(rows, L=L, sched=sch, method=m, lr_mult=1.0, seed=0))
            if pts:
                ax.plot(*zip(*pts), ls, marker="o", color=col, label=f"{key} {m}")
    ax.set(xscale="log", xlabel="LR*WD multiplier (x MAI law)", ylabel="held-out BPB", title="1-D decay sweeps")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(rep, "fig_sweeps1d.png"), dpi=150)
    plt.close(fig)
    if R.get("interim_gap"):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for sd, g in R["interim_gap"].items():
            if g:
                ax.plot(list(map(int, g)), list(g.values()), marker=".", label=f"seed {sd}")
        ax.axhline(0, color="k", lw=.5)
        ax.set(xlabel="step", ylabel="BPB(const) - BPB(scaled)", title="Stage 1 paired gap during training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(rep, "fig_gap_trajectory.png"), dpi=150)
        plt.close(fig)


def report(R, F, stage):
    def j(x):
        return json.dumps(x, indent=1, default=lambda v: float(v) if isinstance(v, (np.floating, float)) else str(v))
    lines = [f"# WDLadder report (through stage {stage})", "", f"**Outcome:** {R['outcome']}", "",
             "## Gate 1 (untuned, MAI hparams, L3 TPP120 cosine)", "```", j(R["gate1"]), "```"]
    for k in ("edge_extension", "wd_ratio_cos", "wd_ratio_cos_ci", "K_rel_cos", "speedup_at_fitted_opt", "tuned_gap",
              "survival_fraction", "speedup_tuned", "wd_ratio_wsd", "K_rel_wsd", "hypothesis_verdict", "scale_trend",
              "L6", "L9", "hypotheses", "seed_sd"):
        if k in R:
            lines += [f"## {k}", "```", j(R[k]), "```"]
    lines += ["## fits.json (MAI multipliers: lr = mult x LR*_MAI, LR*WD = mult x 17.04/steps)", "```", j(F), "```"]
    open(os.path.join(C.P["report"], "report.md"), "w").write("\n".join(lines))
    json.dump(R, open(os.path.join(C.P["report"], "results.json"), "w"), indent=1, default=float)


def run(stage):
    os.makedirs(C.P["report"], exist_ok=True)
    rows = load_runs()
    R, F = analyse(rows, stage)
    figures(rows, F, R)
    report(R, F, stage)
    print(f"runs loaded: {len(rows)} ({sum(r['status']=='done' for r in rows)} done)")
    print("GATE 1:", R["gate1"]["decision"], {k: R["gate1"].get(k) for k in ("gain_mean", "threshold", "speedup_untuned")})
    for k in ("edge_extension", "wd_ratio_cos", "tuned_gap", "survival_fraction", "hypothesis_verdict", "scale_trend", "L6", "L9"):
        if k in R:
            print(k, ":", json.dumps(R[k], default=float)[:400])
    print("OUTCOME:", R["outcome"])
    print("report ->", os.path.join(C.P["report"], "report.md"))
    return R


# ----------------------------------------------------------------- self test (synthetic ladder)
def _selftest():
    import tempfile
    import plan
    td = tempfile.mkdtemp()
    C.P.update(C.data_paths(td))
    rs = np.random.default_rng(1)
    true = {"const": (0.0, 0.0), "scaled": (0.1, math.log(1.36))}   # scaled optimum needs 1.36x more decay
    allr = plan.stage1() + plan.stage2() + plan.stage4()
    allr += [plan.spec(3, 3, 120, "cos", m, 1, p, s) for m in ("const", "scaled") for p in (1.0, 1.36) for s in (1, 2)]
    for r in allr:
        x, y = math.log(r["lr_mult"]), math.log(r["prod_mult"])
        x0, y0 = true[r["method"]]
        base = 1.10 - 0.02 * math.log(r["tpp"] / 120) - 0.03 * (r["L"] - 3)
        bpb = base + 0.02 * (x - x0) ** 2 + 0.01 * (x - x0) * (y - y0) + 0.008 * (y - y0) ** 2 \
            - (0.004 if r["method"] == "scaled" else 0) + rs.normal(0, 0.0008)
        h = C.run_hparams(r["L"], r["tpp"], r["lr_mult"], r["prod_mult"])
        d = os.path.join(C.P["runs"], r["name"])
        os.makedirs(d, exist_ok=True)
        cfg = dict(L=r["L"], tpp=r["tpp"], sched=r["sched"], method=r["method"], lr_mult=r["lr_mult"],
                   prod_mult=r["prod_mult"], seed=r["seed"], batch_tokens=C.RUNG_DEFAULTS[r["L"]]["batch_tokens"],
                   flops_per_token=1.0, params=C.param_counts(C.rung_config(r["L"])), **h)
        json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
        json.dump(dict(status="done", final=dict(bpb=bpb)), open(os.path.join(d, "summary.json"), "w"))
    rows = load_runs()
    R, F = analyse(rows, 2)
    o = F["L3_cos_2d"]["scaled"]["opt"]
    assert abs(math.log(o["prod_mult"]) - math.log(1.36)) < 0.25, o
    assert R["gate1"]["decision"] in ("PASS", "WEAK"), R["gate1"]
    print("wd ratio", R["wd_ratio_cos"], "hyp", C.hypothesis_ratios("cos"))
    json.dump([], open(os.path.join(C.P["report"], "edge_ext.json"), "w"))
    import plan as PL
    for m in ("const", "scaled"):                         # fake confirm runs at the fitted optimum
        o = F["L3_cos_2d"][m]["opt"]
        lm, pm = float(PL.fmt(o["lr_mult"])), float(PL.fmt(o["prod_mult"]))
        for sd in (1, 2):
            d = os.path.join(C.P["runs"], PL.run_name(3, 120, "cos", m, lm, pm, sd))
            os.makedirs(d, exist_ok=True)
            h = C.run_hparams(3, 120, lm, pm)
            cfg = dict(L=3, tpp=120.0, sched="cos", method=m, lr_mult=lm, prod_mult=pm, seed=sd, batch_tokens=32768,
                       flops_per_token=1.0, params=C.param_counts(C.rung_config(3)), **h)
            json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
            json.dump(dict(status="done", final=dict(bpb=1.1 - (0.004 if m == "scaled" else 0) + rs.normal(0, 8e-4))),
                      open(os.path.join(d, "summary.json"), "w"))
    rows = load_runs()
    R, F = analyse(rows, 4)
    assert "tuned_gap" in R, "stage-3 path not exercised"
    figures(rows, F, R)
    report(R, F, 4)
    print("selftest OK:", R["outcome"], "| report at", C.P["report"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    _selftest() if a.selftest else run(a.stage)
