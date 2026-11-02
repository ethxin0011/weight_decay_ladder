"""
chain.py (v2) - self-healing GPU job chain: the GPU job IS its own watcher (no compute instance needed)
v2: successor jobs reuse the parent's registered code snapshot (fixes "code snapshot was modified
    in blob storage"); fallback = fresh code copy with a unique marker; chain.reset(cid) added.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

import config as C

ACTIVE = {"NotStarted", "Queued", "Preparing", "Starting", "Provisioning", "Running", "Finalizing",
          "CancelRequested"}
STALL_GENS = 4
MAX_FAILS = 2
LEASE_S = 240
GPU_THREADS = str(getattr(C, "GPU_THREADS", 4))
WS_FALLBACK = dict(sub="", rg="", ws="")

CONDA_V2 = """name: wdladder
channels: [conda-forge]
dependencies:
  - python=3.10
  - pip
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cu118
    - torch==2.4.1
    - numpy<2
    - scipy
    - pandas
    - pyarrow
    - azure-ai-ml>=1.15
    - azure-identity
"""


# ================================================================ shared: job definition
def build_job(cid, pool_asset, code):
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.entities import UserIdentityConfiguration
    A = C.AZURE
    return command(
        code=code,
        command=(f"python chain.py work --chain {cid} --pool ${{{{inputs.pool}}}} "
                 f"--runs ${{{{outputs.runs}}}} --pool-asset {pool_asset}"),
        environment=f"{A['environment_name']}@latest",
        compute=A["gpu_cluster"],
        inputs=dict(pool=Input(type="uri_folder", path=pool_asset, mode="download")),
        outputs=dict(runs=Output(type="uri_folder", mode="rw_mount",
                                 path=f"azureml://datastores/{A['datastore']}/paths/{A['remote_prefix']}/runs/")),
        experiment_name=A["experiment_name"], display_name=f"chain-{cid}",
        tags=dict(chain=cid, stage="chain", device="gpu"),
        identity=UserIdentityConfiguration(),
        environment_variables=dict(OMP_NUM_THREADS=GPU_THREADS, PYTHONUNBUFFERED="1"),
    )


def chain_jobs(ml, cid, active_only=True):
    out = []
    for j in ml.jobs.list():
        if getattr(j, "experiment_name", None) != C.AZURE["experiment_name"]:
            continue
        if (j.tags or {}).get("chain") != cid:
            continue
        if active_only and j.status not in ACTIVE:
            continue
        out.append(j)
    return out


# ================================================================ worker (runs inside the GPU job)
class Worker:
    def __init__(self, a):
        self.a = a
        self.cdir = os.path.join(a.runs, "_chains", a.chain)
        os.makedirs(self.cdir, exist_ok=True)
        self.me = os.environ.get("AZUREML_RUN_ID") or f"local-{os.getpid()}"
        self.lock = threading.Lock()
        self.term = False
        self.child = None
        self._ml = None

    def _p(self, fn):
        return os.path.join(self.cdir, fn)

    def _load(self, fn, default):
        try:
            return json.load(open(self._p(fn)))
        except Exception:
            return default

    def update(self, fn):
        with self.lock:
            st = self._load("state.json", {})
            fn(st)
            with open(self._p("state.json"), "w") as f:
                json.dump(st, f, indent=1)
            return st

    def queue(self):
        return self._load("queue.json", [])

    def run_done(self, r):
        d = os.path.join(self.a.runs, r["name"])
        if r["name"].startswith("bench_") and os.path.exists(os.path.join(d, "bench.json")):
            return True
        try:
            return json.load(open(os.path.join(d, "summary.json"))).get("status") in ("done", "diverged")
        except Exception:
            return False

    def progress(self, q):
        tot = 0
        for r in q:
            if self.run_done(r):
                tot += 10 ** 7
                continue
            try:
                tot += int(json.load(open(os.path.join(self.a.runs, r["name"], "progress.json"))).get("step", 0))
            except Exception:
                pass
        return tot

    def todo(self, q, st):
        fails = st.get("fails", {})
        return [r for r in q if not self.run_done(r) and fails.get(r["name"], 0) < MAX_FAILS]

    def ml(self):
        if self._ml is None:
            from azure.ai.ml import MLClient
            from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
            e = os.environ
            self._ml = MLClient(AzureMLOnBehalfOfCredential(),
                                e.get("AZUREML_ARM_SUBSCRIPTION", WS_FALLBACK["sub"]),
                                e.get("AZUREML_ARM_RESOURCEGROUP", WS_FALLBACK["rg"]),
                                e.get("AZUREML_ARM_WORKSPACE_NAME", WS_FALLBACK["ws"]))
        return self._ml

    def _code_sources(self):
        """1) reuse this job's registered code snapshot (no re-upload);
           2) fallback: fresh copy of the code with a unique marker -> brand-new snapshot."""
        try:
            cid = self.ml().jobs.get(self.me).code
            if cid:
                yield cid
        except Exception as ex:
            print("[chain] could not read own code id:", ex, flush=True)
        import shutil
        import tempfile
        d = os.path.join(tempfile.mkdtemp(prefix="wdl_code_"), "wdladder")
        shutil.copytree(os.path.dirname(os.path.abspath(__file__)), d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "outputs", "logs", "user_logs",
                                                      "*.pt", "wdladder_data*", ".ipynb_checkpoints"))
        open(os.path.join(d, ".snapshot_id"), "w").write(f"{self.me} {time.time()}")
        yield d

    def submit_successor(self):
        if self.a.local_test:
            return "local-successor"
        for k, code in enumerate(list(self._code_sources()) * 2):
            try:
                j = self.ml().jobs.create_or_update(build_job(self.a.chain, self.a.pool_asset, code))
                print(f"[chain] successor {j.name} submitted (code={str(code)[:90]})", flush=True)
                return j.name
            except Exception as ex:
                print(f"[chain] successor submit failed (try {k+1}, code={str(code)[:90]}): {ex}", flush=True)
                time.sleep(20)
        return None

    def cancel(self, name):
        if not name or self.a.local_test:
            return
        try:
            self.ml().jobs.begin_cancel(name)
            print(f"[chain] cancelled standby successor {name}", flush=True)
        except Exception as ex:
            print(f"[chain] could not cancel successor {name} ({ex}); it will start, find nothing to do "
                  f"and exit within minutes", flush=True)

    def on_term(self, *_):
        self.term = True
        print("[chain] SIGTERM (eviction/cancel) -> checkpointing child", flush=True)
        if self.child and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)

    def heartbeat(self):
        while True:
            time.sleep(60)
            try:
                self.update(lambda s: s.update(heartbeat=time.time()) if s.get("owner") == self.me else None)
            except Exception as ex:
                print("[chain] heartbeat write failed:", ex, flush=True)

    def cmd(self, r):
        if r.get("cmd"):
            c = r["cmd"]
            for k, v in (("${{inputs.pool}}", self.a.pool), ("${{outputs.runs}}", self.a.runs)):
                c = c.replace(k, v)
            c = shlex.split(c)
            if c and c[0] in ("python", "python3"):
                c[0] = sys.executable
            return c
        c = [sys.executable, "train.py", "--run-name", r["name"], "--L", str(r["L"]), "--tpp", str(r["tpp"]),
             "--sched", r["sched"], "--method", r["method"], "--lr-mult", str(r["lr_mult"]),
             "--prod-mult", str(r["prod_mult"]), "--seed", str(r["seed"]), "--threads", GPU_THREADS,
             "--pool", self.a.pool, "--runs-dir", self.a.runs]
        return c + shlex.split(r.get("extra", "") or "")

    def run(self):
        signal.signal(signal.SIGTERM, self.on_term)
        signal.signal(signal.SIGINT, self.on_term)
        st = self._load("state.json", {})
        if st.get("status") in ("stopped", "stalled"):
            print(f"[chain] chain {self.a.chain} is {st['status']} -> exit (use chain.add() to restart)")
            return 0
        for _ in range(20):
            st = self._load("state.json", {})
            if st.get("owner") and st.get("owner") != self.me and time.time() - st.get("heartbeat", 0) < LEASE_S:
                print(f"[chain] previous worker {st['owner']} still alive; waiting", flush=True)
                time.sleep(30)
                continue
            break
        else:
            print("[chain] another worker holds the lease -> exit")
            return 0

        q = self.queue()
        prog = self.progress(q)
        gens = st.get("gens", [])
        same = 0
        for g in reversed(gens):
            if g.get("start_progress") == prog:
                same += 1
            else:
                break
        if same >= STALL_GENS:
            self.update(lambda s: s.update(status="stalled", stalled_at=time.ctime()))
            print(f"[chain] STALLED: {same} generations without progress -> stopping chain.", flush=True)
            return 1
        if not self.todo(q, st):
            self.update(lambda s: s.update(status="finished", finished_at=time.ctime()))
            print("[chain] nothing to do -> finished")
            return 0

        def claim(s):
            s.setdefault("gens", []).append(dict(gen=len(s.get("gens", [])) + 1, job=self.me,
                                                 start_progress=prog, start=time.ctime()))
            s.update(owner=self.me, heartbeat=time.time(), status="running")
        self.update(claim)
        threading.Thread(target=self.heartbeat, daemon=True).start()
        succ = self.submit_successor()
        self.update(lambda s: s.update(successor=succ))
        print(f"[chain] gen {len(gens)+1} job {self.me}; standby successor {succ}; queue {len(q)} runs", flush=True)

        while not self.term:
            st = self._load("state.json", {})
            if st.get("status") == "stopped":
                print("[chain] stop requested")
                break
            todo = self.todo(self.queue(), st)
            if not todo:
                break
            r = todo[0]
            print(f"[chain] >>> {r['name']} ({len(todo)} left)", flush=True)
            self.child = subprocess.Popen(self.cmd(r))
            rc = self.child.wait()
            self.child = None
            print(f"[chain] <<< {r['name']} exit {rc}", flush=True)
            if self.term:
                return 143
            if rc == 75:
                if st.get("test_evict_once"):
                    self.update(lambda s: s.update(test_evict_once=False))
                    print("[chain] TEST: simulating eviction (hard exit, successor must resume)", flush=True)
                    os._exit(1)
                continue
            if rc != 0:
                def fail(s, n=r["name"]):
                    s.setdefault("fails", {})[n] = s.get("fails", {}).get(n, 0) + 1
                self.update(fail)

        st = self._load("state.json", {})
        left = self.todo(self.queue(), st)
        if st.get("status") == "stopped" or not left:
            self.update(lambda s: s.update(status=s.get("status") if s.get("status") == "stopped" else "finished",
                                           finished_at=time.ctime(), owner=None))
            self.cancel(succ)
        failed = [n for n, k in st.get("fails", {}).items() if k >= MAX_FAILS]
        print(f"[chain] done. skipped after {MAX_FAILS} crashes: {failed or 'none'}", flush=True)
        return 0


# ================================================================ instance-side API
def _blob(cid, fn):
    return f"{C.AZURE['remote_prefix']}/runs/_chains/{cid}/{fn}"


def _read(cid, fn, default):
    import azure_jobs as AJ
    try:
        return json.loads(AJ.container().get_blob_client(_blob(cid, fn)).download_blob().readall())
    except Exception:
        return default


def _write(cid, fn, obj):
    import azure_jobs as AJ
    AJ.container().upload_blob(_blob(cid, fn), json.dumps(obj, indent=1), overwrite=True)


def ensure_env_v2():
    import azure_jobs as AJ
    from azure.ai.ml.entities import Environment
    ml = AJ.client()
    name = C.AZURE["environment_name"]
    try:
        e = ml.environments.get(name, version="2")
        print("environment exists:", e.name, e.version)
        return e
    except Exception:
        pass
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_conda_v2.yml")
    open(fn, "w").write(CONDA_V2)
    e = ml.environments.create_or_update(Environment(
        name=name, version="2", conda_file=fn,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"))
    print("environment created:", e.name, e.version, "(image builds on first job, ~10-20 min)")
    return e


_HELP = None


def _train_flags():
    global _HELP
    if _HELP is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _HELP = subprocess.run([sys.executable, "train.py", "--help"], cwd=here, capture_output=True,
                               text=True).stdout
    return _HELP


def _entry(r, extra=""):
    import inspect
    import azure_jobs as AJ
    e = {k: r[k] for k in ("name", "L", "tpp", "sched", "method", "lr_mult", "prod_mult", "seed") if k in r}
    extra = r.get("extra", extra) or extra
    cmd = None
    try:
        g = dict(r, device="gpu")
        kw = {}
        for p in inspect.signature(AJ._train_cmd).parameters:
            if p in ("extra", "extra_args"):
                kw[p] = extra
            elif p == "threads":
                kw[p] = int(GPU_THREADS)
            elif p == "device":
                kw[p] = "gpu"
        cmd = AJ._train_cmd(g, **kw)
        if isinstance(cmd, (list, tuple)):
            cmd = " ".join(map(str, cmd))
        if extra and "extra" not in kw and "extra_args" not in kw and extra not in cmd:
            cmd += " " + extra
    except Exception as ex:
        print(f"[chain] _train_cmd unavailable ({ex}); using built-in command")
    if cmd:
        e["cmd"] = cmd
    else:
        e["extra"] = extra
    h = _train_flags()
    toks = shlex.split(cmd) if cmd else Worker.cmd(type("W", (), {"a": argparse.Namespace(pool="P", runs="R")})(), e)
    bad = [t for t in toks if t.startswith("--") and t.split("=")[0] not in h]
    if bad:
        raise SystemExit(f"[chain] train.py does not know flags {bad}; command: {toks}")
    return e


def add(runs, cid="gpu", pool_asset="azureml:wdladder-pool:1", extra="", test_evict=False):
    import azure_jobs as AJ
    q = _read(cid, "queue.json", [])
    have = {r["name"] for r in q}
    new = [_entry(r, extra) for r in runs if r["name"] not in have]
    q += new
    _write(cid, "queue.json", q)
    st = _read(cid, "state.json", {})
    st.update(status="running", pool_asset=pool_asset)
    if test_evict:
        st["test_evict_once"] = True
    _write(cid, "state.json", st)
    print(f"[chain {cid}] +{len(new)} runs (queue {len(q)})")
    ml = AJ.client()
    act = chain_jobs(ml, cid)
    if act:
        print(f"[chain {cid}] active: {[(j.name, j.status) for j in act]} (new runs are picked up automatically)")
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(here, ".snapshot_id"), "w").write(f"{cid} {time.time()}")
    j = ml.jobs.create_or_update(build_job(cid, pool_asset, here))
    print(f"[chain {cid}] started worker job {j.name}")
    return j.name


def _remote_state(name):
    import azure_jobs as AJ
    if name.startswith("bench_") and AJ.read_remote_json(name, "bench.json"):
        return "done"
    s = AJ.read_remote_json(name, "summary.json")
    if s:
        return s.get("status", "done")
    p = AJ.read_remote_json(name, "progress.json")
    return f"partial@{p.get('step')}" if p else "waiting"


def status(cid="gpu", runs=True):
    import azure_jobs as AJ
    st = _read(cid, "state.json", {})
    q = _read(cid, "queue.json", [])
    hb = st.get("heartbeat")
    print(f"[chain {cid}] status={st.get('status')} owner={st.get('owner')} "
          f"heartbeat={'%.0f s ago' % (time.time()-hb) if hb else '-'} successor={st.get('successor')}")
    for g in st.get("gens", [])[-6:]:
        print("   gen", g)
    if st.get("fails"):
        print("   crash counts:", st["fails"])
    act = chain_jobs(AJ.client(), cid)
    print("   active jobs:", [(j.name, j.status) for j in act] or "none")
    out = {}
    if runs:
        for r in q:
            out[r["name"]] = _remote_state(r["name"])
            print(f"   {r['name']:45s} {out[r['name']]}")
    return dict(state=st, runs=out, active=[j.name for j in act])


def stop(cid="gpu"):
    import azure_jobs as AJ
    st = _read(cid, "state.json", {})
    st["status"] = "stopped"
    _write(cid, "state.json", st)
    ml = AJ.client()
    for j in chain_jobs(ml, cid):
        ml.jobs.begin_cancel(j.name)
        print("cancelled", j.name)
    print(f"[chain {cid}] stopped (checkpoints kept; chain.add(...) restarts it)")


def reset(cid):
    """cancel chain jobs and clear its queue/state (run checkpoints are kept)."""
    import azure_jobs as AJ
    ml = AJ.client()
    for j in chain_jobs(ml, cid):
        ml.jobs.begin_cancel(j.name)
        print("cancelled", j.name)
    _write(cid, "state.json", {})
    _write(cid, "queue.json", [])
    print(f"[chain {cid}] reset")


def checkin(stages, l6_tpp=120, cid="gpu", pool_asset="azureml:wdladder-pool:1"):
    import azure_jobs as AJ
    AJ.sync()
    runs = AJ.stage_runs(stages, l6_tpp)
    cpu = [r for r in runs if r["device"] == "cpu"]
    gpu = [r for r in runs if r["device"] == "gpu"]
    if cpu:
        AJ.submit(cpu, pool_asset=pool_asset)
    if gpu:
        add(gpu, cid=cid, pool_asset=pool_asset)
    status(cid, runs=bool(gpu))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["work", "status", "stop"])
    ap.add_argument("--chain", default="gpu")
    ap.add_argument("--pool", default="")
    ap.add_argument("--runs", default="")
    ap.add_argument("--pool-asset", default="azureml:wdladder-pool:1")
    ap.add_argument("--local-test", action="store_true")
    a = ap.parse_args()
    if a.cmd == "work":
        sys.exit(Worker(a).run())
    elif a.cmd == "status":
        status(a.chain)
    else:
        stop(a.chain)
