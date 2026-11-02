"""
config.py - single source of truth for WDLadder (torch-free; import anywhere)
=============================================================================
Paper: scaled weight decay (Muon-SW) on an MAI-style MoE scaling ladder, tuned
under MAI's LR x WD law.  Question: how much of the reported Muon-SW speedup
survives when BOTH methods are tuned, and what law gives the optimal decay?

STEP-BY-STEP
  python config.py            # prints the ladder, MAI hyper-parameters, decay integrals
  Edit only:  AZURE / FOUNDRY (already filled), PRICES (match your billing tier),
              BUDGET_USD, PLACEMENT.  Never put secrets in this file.
Secrets are read from env vars or ~/.wdladder_secrets (written by the notebook).
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

# ----------------------------------------------------------------- paths
NOTEBOOK_ROOT = "/home/azureuser/cloudfiles/code/Users/estherxin0011"
PROJECT_DIR = os.environ.get("WDL_PROJECT", os.path.join(NOTEBOOK_ROOT, "wdladder"))
DATA_ROOT = os.environ.get("WDL_DATA", "/home/azureuser/wdladder_data")   # local SSD, not cloudfiles
SMOKE_ROOT = DATA_ROOT + "_smoke"


def data_paths(root: str = DATA_ROOT) -> dict:
    return dict(root=root, raw=os.path.join(root, "raw"), pool=os.path.join(root, "pool"),
                runs=os.path.join(root, "runs"), report=os.path.join(root, "report"),
                tokenizer=os.path.join(root, "tokenizer.json"))


P = data_paths()

# ----------------------------------------------------------------- Azure (from your settings)
AZURE = dict(
    subscription_id="7a8513b6-ada2-4ad1-aee2-687fa5663c82",
    resource_group="AIModels",
    workspace="RLVR",
    compute_instance="cpurlvr",            # Standard_E20ads_v5 (20 vCPU, 160 GB): data prep + analysis
    gpu_cluster="rlvr2",                   # Standard_NC4as_T4_v3 (1x T4, 4 vCPU): L4/L6/L9
    cpu_cluster="cpu-rlvr",                # created by azure_jobs.ensure_cpu_cluster: L3 runs
    cpu_cluster_size="Standard_F16s_v2",   # 16 vCPU AVX-512, 32 GB; 2 runs x 8 threads per node
    cpu_cluster_max_nodes=6,
    cpu_low_priority=True,
    supervisor_instance="wdl-mini",        # optional cheap instance (Standard_D2as_v5) for the supervisor
    environment_name="wdladder-env",
    experiment_name="wdladder",
    datastore="workspaceblobstore",
    remote_prefix="wdladder",              # datastore folder: wdladder/runs/<run>/
)

FOUNDRY = dict(
    endpoint="https://llmmodelsperf.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment_main="gpt-6.1-sol",         # accuracy-critical: gate reviews, results drafting
    deployment_fast="",                    # optional GPT-5-series deployment name for cheap log triage
)

# ----------------------------------------------------------------- budget (USD)
# East US Linux list prices, Oct 2026. SET gpu_node TO YOUR TIER (low priority T4 is ~5x cheaper).
PRICES = dict(cpu_node=0.135,      # F16s_v2 low priority (dedicated: 0.677)
              gpu_node=0.526,      # NC4as_T4_v3 pay-as-you-go (conservative default)
              instance=1.31,       # E20ads_v5 compute instance (stop it when idle!)
              overhead=1.15)       # node start-up + work lost to pre-emption (<=15 min per event)
BUDGET_USD = 140.0
CPU_RUNS_PER_NODE = 2
CPU_THREADS_PER_RUN = 8
GPU_THREADS = 4
PLACEMENT = {3: "cpu", 4: "gpu", 6: "gpu", 9: "gpu"}   # plan.py may move L3 to GPU if cheaper and over budget

# ----------------------------------------------------------------- data (MAI-style source scaling)
SOURCES = {
    "web":  dict(hf="HuggingFaceFW/fineweb-edu", config="sample-10BT", field="text", share=0.60, repeats=1,
                 bytes_per_token=4.4),
    "code": dict(hf="codeparrot/codeparrot-clean", config=None, field="content", share=0.25, repeats=4,
                 bytes_per_token=3.4),
    "math": dict(hf="open-web-math/open-web-math", config=None, field="text", share=0.15, repeats=4,
                 bytes_per_token=3.2),
}
SOURCE_NAMES = list(SOURCES)
VOCAB = 8192            # own byte-level BPE; a 151k Qwen3 vocab would make the LM head 20x the model
SEQ_LEN = 512
DOC_CAP = 4096          # tokens per document (longer docs truncated)
EVAL_DOCS = 3000        # held-out docs per source (never trained on)
EOS = "<|eos|>"
SEED = 1234


# ----------------------------------------------------------------- ladder (MAI rungs, scaled down)
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    head_dim: int = 64
    n_expert: int = 32
    top_k: int = 2
    latent: int = 0          # routed experts work in a shared down-projected latent (MAI-Base-2 style)
    d_ff_expert: int = 0
    d_ff_shared: int = 0     # 1 shared expert, full width (not down/up projected), as MAI-Base-2
    window: int = 128        # local attention window; global attention every `global_every` layers
    global_every: int = 6
    vocab: int = VOCAB
    seq_len: int = SEQ_LEN
    dropout: float = 0.0
    aux_coef: float = 0.01
    z_coef: float = 1e-3
    rope_base: float = 10000.0

    def is_global(self, i: int) -> bool:
        return (i + 1) % self.global_every == 0 or i == self.n_layer - 1

    def to_dict(self):
        return asdict(self)


def mai_dropout(active_params: float) -> float:
    """MAI: dropout = sigmoid(0.526 (ln N_active - 27.2)); ~0 at our scales, kept for fidelity."""
    return 1.0 / (1.0 + math.exp(-0.526 * (math.log(active_params) - 27.2)))


def rung_config(L: int) -> ModelConfig:
    d = 64 * max(1, round(256 / 3 * L / 64))            # MAI: hidden = 256/3 * L
    h = d // 64
    kv = max(1, h // 4)
    while h % kv:
        kv -= 1
    cfg = ModelConfig(n_layer=L, d_model=d, n_head=h, n_kv_head=kv, latent=d // 2, d_ff_expert=d, d_ff_shared=d)
    cfg.dropout = round(mai_dropout(param_counts(cfg)["active_nonembed"]), 5)
    return cfg


def smoke_config() -> ModelConfig:
    return ModelConfig(n_layer=2, d_model=64, n_head=2, n_kv_head=1, head_dim=32, n_expert=8, top_k=2,
                       latent=32, d_ff_expert=64, d_ff_shared=64, window=32, seq_len=128)


def param_counts(c: ModelConfig) -> dict:
    d, hd = c.d_model, c.head_dim
    attn = d * c.n_head * hd * 2 + d * c.n_kv_head * hd * 2
    shared = 3 * d * c.d_ff_shared
    latent = 2 * d * c.latent
    expert = 3 * c.latent * c.d_ff_expert
    router = d * c.n_expert
    act_layer = attn + shared + latent + router + c.top_k * expert
    tot_layer = attn + shared + latent + router + c.n_expert * expert
    emb = c.vocab * d
    return dict(active_nonembed=c.n_layer * act_layer, total_nonembed=c.n_layer * tot_layer,
                embed=emb, head=emb, total=c.n_layer * tot_layer + 2 * emb)


def flops_per_token(c: ModelConfig) -> float:
    pc = param_counts(c)
    attn = sum(12 * c.d_model * (c.seq_len / 2 if c.is_global(i) else c.window) for i in range(c.n_layer))
    return 6 * (pc["active_nonembed"] + pc["head"]) + attn


RUNG_DEFAULTS = {   # batch tokens per optimizer step; micro-batch (sequences) per device
    3: dict(batch_tokens=32768, micro=dict(cpu=16, gpu=64)),
    4: dict(batch_tokens=32768, micro=dict(cpu=16, gpu=64)),
    6: dict(batch_tokens=65536, micro=dict(cpu=8, gpu=32)),
    9: dict(batch_tokens=65536, micro=dict(cpu=4, gpu=16)),
    12: dict(batch_tokens=131072, micro=dict(cpu=4, gpu=8)),
}


# ----------------------------------------------------------------- MAI hyper-parameter laws
def mai_hparams(tpp: float, steps: int) -> dict:
    """MAI-Base-2 published laws (natural log):
         ln LR* = -1.51 + 0.175 ln TPP - 0.573 ln steps
         LR* x WD* = 17.04 / steps        (decoupled AdamW-style: p *= 1 - lr*wd)"""
    lr = math.exp(-1.51 + 0.175 * math.log(tpp) - 0.573 * math.log(steps))
    prod = 17.04 / steps
    return dict(lr=lr, wd=prod / lr, prod=prod)


def run_steps(L: int, tpp: float) -> int:
    pc = param_counts(rung_config(L))
    return max(50, int(round(tpp * pc["active_nonembed"] / RUNG_DEFAULTS[L]["batch_tokens"])))


def run_hparams(L: int, tpp: float, lr_mult: float = 1.0, prod_mult: float = 1.0) -> dict:
    """Grid lives in MAI's coordinates (ln LR, ln LR*WD): lr = lr0*lr_mult, lr*wd = prod0*prod_mult."""
    steps = run_steps(L, tpp)
    h = mai_hparams(tpp, steps)
    lr = h["lr"] * lr_mult
    prod = h["prod"] * prod_mult
    return dict(steps=steps, lr=lr, wd=prod / lr, prod=prod, lr0=h["lr"], wd0=h["wd"], prod0=h["prod"])


# ----------------------------------------------------------------- schedules
SCHED = dict(warmup_frac=0.02, final_frac=0.10, wsd_decay_frac=0.20)


def lr_factor(step: int, total: int, sched: str = "cos") -> float:
    """f(t) in [final_frac, 1]; eta_t = lr * f(t)."""
    w = max(1, int(SCHED["warmup_frac"] * total))
    if step < w:
        return (step + 1) / w
    lo = SCHED["final_frac"]
    if sched == "cos":
        x = (step - w) / max(1, total - w)
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, x)))
    if sched == "wsd":
        d0 = int(total * (1 - SCHED["wsd_decay_frac"]))
        if step < d0:
            return 1.0
        x = (step - d0) / max(1, total - d0)
        return 1.0 - (1 - lo) * min(1.0, x)
    raise ValueError(sched)


def decay_scale(step: int, total: int, sched: str, method: str) -> float:
    """Multiplier s_t on weight decay. const: s=1 (p *= 1 - eta_t*wd).
    scaled (Muon-SW, Apte 2607.23777): s = eta_t/eta_max  (p *= 1 - eta_t^2/eta_max * wd)."""
    if method == "const":
        return 1.0
    if method == "scaled":
        return lr_factor(step, total, sched)
    raise ValueError(method)


def decay_integral(total: int, sched: str, method: str) -> float:
    """sum_t f(t) s(t) / total : integrated decay per unit (lr*wd*steps)."""
    return sum(lr_factor(t, total, sched) * decay_scale(t, total, sched, method) for t in range(total)) / total


def hypothesis_ratios(sched: str, total: int = 2000) -> dict:
    """Predicted lambda*_scaled / lambda*_const under three competing hypotheses."""
    r = decay_integral(total, sched, "scaled") / decay_integral(total, sched, "const")
    return dict(H_none=1.0, H_int=1.0 / r, H_end=1.0 / SCHED["final_frac"], r=r)


# ----------------------------------------------------------------- training defaults
TRAIN = dict(
    ckpt_minutes=15.0,          # checkpoint cadence (pre-emption safety; plus on SIGTERM)
    eval_every_frac=0.10,       # held-out BPB every 10% of training
    eval_windows=256,           # windows per source per periodic eval
    final_eval_windows=1024,    # windows per source for the final eval
    grad_clip=1.0,
    adam_betas=(0.9, 0.95),
    adam_lr_mult=1.0,           # AdamW lr for embeddings/head/router = lr * mult  (MAI: AdamW off-matrix)
    muon_momentum=0.95,
    muon_ns_steps=5,
    diverge_after_frac=0.30,    # loss > initial loss after 30% of training => diverged (stop, save money)
)

# ----------------------------------------------------------------- pre-registered decisions
PREREG = dict(
    gate1_min_gain_bpb=0.003,   # Muon-SW must beat const at MAI hparams by > max(this, 2 x seed noise)
    gate1_noise_mult=2.0,
    edge_tolerance=0.25,        # fitted optimum within 0.25 grid-steps of an edge => extend grid
    tuned_gain_null_bpb=0.002,  # tuned gap below this (and CI covers 0) => "calibration artifact"
)


# ----------------------------------------------------------------- secrets
SECRETS_FILE = os.path.expanduser("~/.wdladder_secrets")


def secret(name: str, required: bool = True) -> str | None:
    """Env var first, then ~/.wdladder_secrets (KEY=VALUE lines, chmod 600)."""
    v = os.environ.get(name)
    if v:
        return v
    if os.path.exists(SECRETS_FILE):
        for line in open(SECRETS_FILE):
            if "=" in line:
                k, val = line.strip().split("=", 1)
                if k == name and val:
                    return val
    if required:
        raise RuntimeError(f"secret {name} missing: set env var or run notebook cell 0")
    return None


if __name__ == "__main__":
    print(f"{'L':>3} {'d':>5} {'H/KV':>6} {'active':>10} {'total':>11} {'steps@120':>9} {'tok@120':>8} "
          f"{'LR*':>9} {'WD*':>7} {'GFLOP/tok':>9}")
    for L in (3, 4, 6, 9):
        c = rung_config(L)
        pc = param_counts(c)
        h = run_hparams(L, 120)
        print(f"{L:>3} {c.d_model:>5} {c.n_head:>3}/{c.n_kv_head:<2} {pc['active_nonembed']/1e6:>9.2f}M "
              f"{pc['total']/1e6:>10.1f}M {h['steps']:>9} {120*pc['active_nonembed']/1e6:>7.0f}M "
              f"{h['lr']:>9.2e} {h['wd']:>7.3f} {flops_per_token(c)/1e9:>9.3f}")
    for s in ("cos", "wsd"):
        print(s, {k: round(v, 3) for k, v in hypothesis_ratios(s).items()})

# >>> WDL CPU CLUSTER OVERRIDE >>>
AZURE["cpu_cluster"] = "cpu-rlvr"
AZURE["cpu_cluster_size"] = "Standard_E16s_v3"
AZURE["cpu_low_priority"] = False
AZURE["cpu_cluster_max_nodes"] = 6
PRICES["cpu_node"] = 1.008        # $/node-hour for Standard_E16s_v3 (dedicated)
CPU_RUNS_PER_NODE = 2               # E16s_v3: 16 vCPU / 128 GB -> 2 runs x 8 threads
CPU_THREADS_PER_RUN = 8
# <<< WDL CPU CLUSTER OVERRIDE <<<
