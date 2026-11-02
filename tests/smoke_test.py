"""
tests/smoke_test.py - run BEFORE any paid job (on the compute instance, ~5-10 min, CPU only)
=============================================================================================
  cd wdladder && python tests/smoke_test.py            # unit tests (no data needed)
  python tests/smoke_test.py --resume                  # + end-to-end train/resume test on the smoke pool
Checks
  1 config laws reproduce MAI's published numbers; scaled-decay integrals; plan names unique
  2 Newton-Schulz output is ~orthogonal (2-D and batched 3-D experts)
  3 MoE forward/backward: dropless (every token routed to top_k experts), finite loss, grads everywhere
  4 decay semantics: const => p*(1-eta_t*wd); scaled => p*(1-eta_t^2/eta_max*wd)
  5 datastream determinism + source-scaling repeats
  6 (--resume) train 40 steps straight vs 20 + kill + resume 20: identical final loss (CPU)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config as C  # noqa: E402


def t_config():
    h = C.mai_hparams(30, 1734)
    assert abs(h["lr"] - 5.6e-3) / 5.6e-3 < 0.02 and abs(h["wd"] - 1.755) < 0.03, h
    r = C.hypothesis_ratios("cos")["r"]
    assert 0.70 < r < 0.76, r
    import plan
    raw = [x["name"] for x in plan.all_planned()]
    print("cross-stage duplicates (expected, deduped at submit):", sorted({n for n in raw if raw.count(n) > 1}))
    names = [x["name"] for x in plan.dedup(plan.all_planned())]
    assert len(names) == len(set(names))
    print("1 config/plan OK", {k: round(v, 4) for k, v in h.items()})


def t_ns():
    import optim
    G = torch.randn(64, 32)
    O = optim.newton_schulz(G)
    s = torch.linalg.svdvals(O)
    assert 0.5 < s.min() and s.max() < 1.3, s
    G3 = torch.randn(4, 16, 48)
    s3 = torch.linalg.svdvals(optim.newton_schulz(G3))
    assert 0.5 < s3.min() and s3.max() < 1.3
    print("2 Newton-Schulz OK", round(float(s.min()), 3), round(float(s.max()), 3))


def t_moe():
    import model
    c = C.smoke_config()
    m = model.MoELM(c)
    x = torch.randint(1, c.vocab, (4, c.seq_len + 1))
    ce, aux, z, load = m(x[:, :-1], x[:, 1:])
    assert torch.isfinite(ce) and abs(ce.item() - math.log(c.vocab)) < 1.0, ce
    assert int(load.sum()) == c.n_layer * 4 * c.seq_len * c.top_k, "dropless routing violated"
    (ce + 0.01 * aux + 1e-3 * z).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    print("3 MoE OK ce", round(ce.item(), 3), "aux", round(float(aux), 3))


def t_decay():
    import model
    import optim
    m = model.MoELM(C.smoke_config())
    opt = optim.Combined(m, lr=0.01, wd=0.5)
    p = opt.muon.param_groups[0]["params"][0]
    for method, total in (("const", 100), ("scaled", 100)):
        step = 80
        f = C.lr_factor(step, total, "cos")
        s = C.decay_scale(step, total, "cos", method)
        opt.set(0.01 * f, 0.5 * s)
        before = p.detach().clone()
        p.grad = torch.zeros_like(p)
        opt.muon.state.clear()
        opt.muon.step()           # zero grad -> NS(0)=0 -> pure decay
        ratio = (p / before).mean().item()
        want = 1 - (0.01 * f) * 0.5 * (f if method == "scaled" else 1)
        assert abs(ratio - want) < 1e-6, (method, ratio, want)
    print("4 decay semantics OK")


def t_stream():
    import datastream
    datastream._selftest()
    print("5 datastream OK")


def t_resume():
    pool = C.data_paths(C.SMOKE_ROOT)["pool"]
    if not os.path.exists(os.path.join(pool, "meta.json")):
        raise SystemExit("smoke pool missing: run  python data_prep.py --smoke all  first")
    td = tempfile.mkdtemp()
    base = [sys.executable, "train.py", "--smoke", "--steps", "40", "--pool", pool, "--runs-dir", td,
            "--threads", "4", "--method", "scaled", "--ckpt-minutes", "999"]
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(base + ["--run-name", "straight"], check=True, cwd=cwd)
    r = subprocess.run(base + ["--run-name", "resumed", "--stop-at-step", "20"], cwd=cwd)
    assert r.returncode == 75, f"expected exit 75 on simulated pre-emption, got {r.returncode}"
    assert os.path.exists(os.path.join(td, "resumed", "ckpt.pt"))
    subprocess.run(base + ["--run-name", "resumed"], check=True, cwd=cwd)
    a = json.load(open(os.path.join(td, "straight", "summary.json")))
    b = json.load(open(os.path.join(td, "resumed", "summary.json")))
    assert b["segments"] == 2, b
    da, db = a["final"]["bpb"], b["final"]["bpb"]
    assert abs(da - db) < 1e-4, (da, db)
    r = subprocess.run(base + ["--run-name", "resumed"], cwd=cwd, capture_output=True, text=True)
    assert "[skip]" in r.stdout, "finished run was not skipped"
    shutil.rmtree(td)
    print(f"6 resume OK: straight {da:.5f} == resumed {db:.5f}; finished run skipped")


if __name__ == "__main__":
    torch.manual_seed(0)
    t_config()
    t_ns()
    t_moe()
    t_decay()
    t_stream()
    if "--resume" in sys.argv:
        t_resume()
    print("ALL SMOKE TESTS PASSED")
