"""
azure_jobs.py - Azure ML plumbing: clusters, environment, pool asset, budget-checked submission,
                pre-emption supervisor, result sync, kill switch
================================================================================================
Notebook uses these functions; CLI for long-running supervision:
  python azure_jobs.py supervise --stages 1 --interval 20     # resubmits pre-empted runs until done
  python azure_jobs.py status --stages 1 2                     # done / active / missing per run
  python azure_jobs.py sync                                    # copy small result files to $WDL_DATA/runs
  python azure_jobs.py cancel --stage 2                        # KILL SWITCH: cancel all jobs of a stage
Never submits a run that is done/diverged or already queued/running; refuses to exceed BUDGET_USD.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import config as C
import plan

ACTIVE = {"NotStarted", "Queued", "Preparing", "Starting", "Provisioning", "Running", "Finalizing",
          "CancelRequested"}
RESULT_FILES = ("summary.json", "progress.json", "config.json", "evals.jsonl", "log.jsonl",
                "data_report.json", "bench.json")
_ML = {}


# ----------------------------------------------------------------- clients
def client():
    if "ml" not in _ML:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        _ML["ml"] = MLClient(cred, C.AZURE["subscription_id"], C.AZURE["resource_group"], C.AZURE["workspace"])
    return _ML["ml"]


def container():
    """azure-storage-blob ContainerClient for the workspace datastore (state checks + sync)."""
    if "cc" not in _ML:
        from azure.storage.blob import BlobServiceClient
        ds = client().datastores.get(C.AZURE["datastore"], include_secrets=True)
        url = f"https://{ds.account_name}.blob.{getattr(ds, 'endpoint', 'core.windows.net')}"
        cr = getattr(ds, "credentials", None)
        key = getattr(cr, "account_key", None) or getattr(cr, "sas_token", None)
        if not key:
            from azure.identity import DefaultAzureCredential
            key = DefaultAzureCredential()
        _ML["cc"] = BlobServiceClient(url, credential=key).get_container_client(ds.container_name)
    return _ML["cc"]


def remote(name, fn=""):
    return f"{C.AZURE['remote_prefix']}/runs/{name}/{fn}".rstrip("/")


def read_remote_json(name, fn):
    try:
        return json.loads(container().get_blob_client(remote(name, fn)).download_blob().readall())
    except Exception:
        return None


# ----------------------------------------------------------------- infrastructure

def ensure_cpu_cluster():
    """READ-ONLY: uses the existing CPU cluster from config (patched by patch_step2.py). Never creates nodes."""
    c = client().compute.get(C.AZURE["cpu_cluster"])
    print("cpu cluster:", c.name, c.size, "tier", getattr(c, "tier", "?"), "min", c.min_instances,
          "max", c.max_instances, "idle_s", getattr(c, "idle_time_before_scale_down", "?"))
    if c.min_instances:
        print("WARNING: min_instances > 0 bills idle nodes 24/7 - set min nodes to 0 in the studio")
    if (c.size or "").lower() != C.AZURE["cpu_cluster_size"].lower():
        print("WARNING: cluster size differs from config - re-run patch_step2.py")
    return c


def check_gpu_cluster():
    c = client().compute.get(C.AZURE["gpu_cluster"])
    print("gpu cluster:", c.name, c.size, "tier", c.tier, "min", c.min_instances, "max", c.max_instances,
          "idle_s", c.idle_time_before_scale_down)
    if c.min_instances:
        print("WARNING: min_instances > 0 bills the T4 24/7 - set it to 0")
    return c


CONDA = """name: wdladder
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
"""


def ensure_env(version="1"):
    from azure.ai.ml.entities import Environment
    ml = client()
    try:
        e = ml.environments.get(C.AZURE["environment_name"], version=version)
        print("environment exists:", e.name, e.version)
        return e
    except Exception:
        pass
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_conda.yml")
    open(fn, "w").write(CONDA)
    e = Environment(name=C.AZURE["environment_name"], version=version, conda_file=fn,
                    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest")
    return ml.environments.create_or_update(e)


def upload_pool(pool_dir, name="wdladder-pool", version=None):
    """Registers the token pool as a data asset (uploaded once; jobs download it to local disk)."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data
    meta = json.load(open(os.path.join(pool_dir, "meta.json")))
    version = version or ("smoke" if meta.get("smoke") else "1")
    ml = client()
    try:
        d = ml.data.get(name, version=version)
        print("pool asset exists:", d.name, d.version)
        return d
    except Exception:
        pass
    return ml.data.create_or_update(Data(name=name, version=version, path=pool_dir, type=AssetTypes.URI_FOLDER,
                                         description=f"WDLadder token pool {meta['tokens']}"))


# ----------------------------------------------------------------- state
def active_runs() -> dict:
    """run name -> job name, for jobs of this experiment that are queued/running."""
    out = {}
    for j in client().jobs.list():
        if getattr(j, "experiment_name", None) != C.AZURE["experiment_name"] or j.status not in ACTIVE:
            continue
        for r in (j.tags or {}).get("runs", "").split(","):
            if r:
                out[r] = j.name
    return out


def run_state(name) -> str:
    s = read_remote_json(name, "summary.json")
    if s:
        return s.get("status", "done")
    if name.startswith("bench_") and read_remote_json(name, "bench.json"):
        return "done"
    p = read_remote_json(name, "progress.json")
    return f"partial@{p.get('step')}" if p else "missing"


def status(runs, verbose=True):
    act = active_runs()
    st = {}
    for r in runs:
        s = run_state(r["name"])
        if s not in ("done", "diverged") and r["name"] in act:
            s = "active:" + s
        st[r["name"]] = s
        if verbose:
            print(f"{r['name']:45s} {r['device']} {s}")
    return st


def sync(local_runs=None, prefix=None):
    """copy small result files (not checkpoints) from the datastore to $WDL_DATA/runs."""
    local_runs = local_runs or C.P["runs"]
    cc, n = container(), 0
    base = f"{C.AZURE['remote_prefix']}/runs/"
    for b in cc.list_blobs(name_starts_with=base + (prefix or "")):
        rel = b.name[len(base):]
        if rel.split("/")[-1] not in RESULT_FILES:
            continue
        dst = os.path.join(local_runs, rel)
        if os.path.exists(dst) and os.path.getsize(dst) == b.size and rel.endswith("summary.json"):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(cc.download_blob(b.name).readall())
        n += 1
    print(f"synced {n} files -> {local_runs}")


# ----------------------------------------------------------------- submission
def _train_cmd(r, threads, pool="${{inputs.pool}}", runs="${{outputs.runs}}", extra=""):
    return (f"python train.py --run-name {r['name']} --L {r['L']} --tpp {r['tpp']} --sched {r['sched']} "
            f"--method {r['method']} --lr-mult {r['lr_mult']} --prod-mult {r['prod_mult']} --seed {r['seed']} "
            f"--threads {threads} --pool {pool} --runs-dir {runs} {extra}").strip()


def _packed(cmds, names):
    """runs share one node; each writes outputs/<run>.log (uploaded by Azure ML); exit = OR of statuses."""
    lines = ["mkdir -p outputs", "rc=0"]
    for i, (c, n) in enumerate(zip(cmds, names)):
        lines.append(f"({c}) > outputs/{n}.log 2>&1 & p{i}=$!")
    for i in range(len(cmds)):
        lines.append(f"wait $p{i} || rc=$?")
    lines += ["tail -n 5 outputs/*.log", "exit $rc"]
    return "bash -c '" + "; ".join(lines).replace("'", "'\\''") + "'"


def committed_usd(lg, exclude=()):
    """estimated $ of the LAST submission of each run that is not finished (conservative)."""
    last = {}
    for x in lg["submissions"]:
        last[x["run"]] = x
    tot = 0.0
    for r, x in last.items():
        if r in exclude or x.get("oneoff"):
            continue
        if run_state(r) not in ("done", "diverged"):
            tot += x["usd"]
    return tot


def submit(runs, pool_asset="azureml:wdladder-pool:1", force=False, dry=False, extra_args="", tag_stage=None,
           usd_override=None):
    """usd_override: fixed $ per run for one-off jobs (smoke/benchmark) instead of the full-run estimate."""
    from azure.ai.ml import Input, Output, command
    ml = client()
    st = status(runs, verbose=False)
    todo = [r for r in runs if st[r["name"]] not in ("done", "diverged") and not st[r["name"]].startswith("active")]
    if not todo:
        print("nothing to submit (all done or active)")
        return []
    tps = plan.load_bench()
    est = {r["name"]: (dict(usd=usd_override, hours=0) if usd_override is not None else plan.estimate(r, tps))
           for r in todo}
    lg = plan.ledger_load()
    spent = plan.actual_spend(instance_hours=lg.get("instance_hours", 0.0))["total"]
    new = sum(e["usd"] for e in est.values())
    commit = committed_usd(lg, exclude=set(est))
    print(f"[budget] spent ${spent:.2f} + in-flight ${commit:.2f} + this batch ${new:.2f} "
          f"= ${spent+commit+new:.2f} / ${C.BUDGET_USD:.0f}")
    if spent + commit + new > C.BUDGET_USD and not force:
        raise SystemExit("[budget] REFUSED: would exceed BUDGET_USD (pass force=True only if you mean it)")
    groups = []
    cpu = [r for r in todo if r["device"] == "cpu"]
    for i in range(0, len(cpu), C.CPU_RUNS_PER_NODE):
        groups.append(("cpu", cpu[i:i + C.CPU_RUNS_PER_NODE]))
    groups += [("gpu", [r]) for r in todo if r["device"] == "gpu"]
    jobs = []
    for dev, g in groups:
        thr = C.CPU_THREADS_PER_RUN if dev == "cpu" else C.GPU_THREADS
        cmd = _packed([_train_cmd(r, thr, extra=extra_args) for r in g], [r["name"] for r in g])
        if dry:
            print("[dry]", dev, [r["name"] for r in g])
            continue
        job = command(
            code=os.path.dirname(os.path.abspath(__file__)), command=cmd,
            environment=f"{C.AZURE['environment_name']}@latest",
            compute=C.AZURE["cpu_cluster"] if dev == "cpu" else C.AZURE["gpu_cluster"],
            inputs=dict(pool=Input(type="uri_folder", path=pool_asset, mode="download")),
            outputs=dict(runs=Output(type="uri_folder", mode="rw_mount",
                                     path=f"azureml://datastores/{C.AZURE['datastore']}/paths/"
                                          f"{C.AZURE['remote_prefix']}/runs/")),
            experiment_name=C.AZURE["experiment_name"], display_name=g[0]["name"][:60],
            tags=dict(runs=",".join(r["name"] for r in g), stage=str(tag_stage or g[0]["stage"]), device=dev),
            environment_variables=dict(OMP_NUM_THREADS=str(thr), PYTHONUNBUFFERED="1"),
        )
        j = ml.jobs.create_or_update(job)
        jobs.append(j.name)
        for r in g:
            lg["submissions"].append(dict(run=r["name"], job=j.name, usd=round(est[r["name"]]["usd"], 3),
                                          time=time.ctime(), stage=r["stage"],
                                          oneoff=usd_override is not None))
        print(f"[submit] {j.name} {dev} {[r['name'] for r in g]}")
    if not dry:
        plan.ledger_save(lg)
    return jobs


def cancel(stage):
    ml, n = client(), 0
    for j in ml.jobs.list():
        if getattr(j, "experiment_name", None) == C.AZURE["experiment_name"] and j.status in ACTIVE \
                and (j.tags or {}).get("stage") == str(stage):
            ml.jobs.begin_cancel(j.name)
            n += 1
    print(f"cancelled {n} jobs of stage {stage} (checkpoints stay; resubmitting resumes them)")


def stage_runs(stages, l6_tpp=60):
    out = []
    fits_p = os.path.join(C.P["report"], "fits.json")
    fits = json.load(open(fits_p)) if os.path.exists(fits_p) else {}
    ext_p = os.path.join(C.P["report"], "edge_ext.json")
    ext = json.load(open(ext_p)) if os.path.exists(ext_p) else []
    for s in stages:
        out += {1: plan.stage1, 2: lambda: plan.stage2(ext), 3: lambda: plan.stage3(fits), 4: plan.stage4,
                5: lambda: plan.stage5(fits, l6_tpp), 6: lambda: plan.stage6(fits)}[s]()
    return plan.dedup(out)


def supervise(stages, interval_min=20, l6_tpp=60, max_hours=400, once=False):
    """pre-emption watchdog: every interval, resubmit runs that are neither done nor active."""
    t0 = time.time()
    while (time.time() - t0) / 3600 < max_hours:
        sync()                                    # budget check uses synced progress.json wall-times
        runs = stage_runs(stages, l6_tpp)
        st = status(runs, verbose=False)
        left = [r for r in runs if st[r["name"]] not in ("done", "diverged")]
        print(time.strftime("%H:%M"), {k: sum(1 for v in st.values() if v.startswith(k))
                                       for k in ("done", "diverged", "active", "partial", "missing")}, flush=True)
        if not left:
            print("all runs finished")
            return True
        submit(left)
        if once:
            return False
        time.sleep(interval_min * 60)
    return False


# ----------------------------------------------------------------- local (compute instance) runs
def launch_local(r, threads=8, runs_dir=None, pool=None, extra=""):
    """run on the compute instance itself (smoke tests, or spare cores while the instance is on anyway)."""
    runs_dir, pool = runs_dir or C.P["runs"], pool or C.P["pool"]
    os.makedirs(runs_dir, exist_ok=True)
    cmd = _train_cmd(r, threads, pool=pool, runs=runs_dir, extra=extra)
    log = open(os.path.join(runs_dir, r["name"] + ".local.log"), "a")
    return subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                            cwd=os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["supervise", "status", "sync", "cancel"])
    ap.add_argument("--stages", type=int, nargs="*", default=[1])
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--interval", type=float, default=20)
    ap.add_argument("--l6-tpp", type=int, default=60)
    ap.add_argument("--once", action="store_true", help="one check-in (resubmit missing/pre-empted runs) then exit")
    a = ap.parse_args()
    if a.cmd == "supervise":
        supervise(a.stages, a.interval, a.l6_tpp, once=a.once)
    elif a.cmd == "status":
        status(stage_runs(a.stages, a.l6_tpp))
    elif a.cmd == "sync":
        sync()
    elif a.cmd == "cancel":
        cancel(a.stage)
