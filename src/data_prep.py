"""
data_prep.py - fetch -> tokenizer -> tokenize -> pool -> verify  (CPU, runs on compute instance cpurlvr)
=====================================================================================================
Resumable at every stage (safe to re-run after a crash / instance restart).

  python data_prep.py --smoke all       # ~3 min, tiny pool in DATA_ROOT_smoke (for smoke tests)
  python data_prep.py all               # full pool sized from plan.py (~1.3B tokens, ~1-2.5 h on 20 vCPU)
  python data_prep.py verify            # prints token counts vs plan requirement, BPB byte stats

Stages
  fetch     stream HF datasets -> raw/<src>/shard_00000.jsonl  (tmp+rename; resume skips finished shards)
  tokenizer train byte-level BPE (vocab 8192, EOS id 0) on a share-weighted sample
  tokenize  encode shards (Rust multithreaded) -> per-source shuffled docs -> pool/*.bin
  verify    check pool >= requirement of every planned run (source scaling needs unique prefixes)
Requires HF_TOKEN (secret) only for rate limits; datasets are public.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

import config as C

DOCS_PER_SHARD = 20000


def requirements(smoke: bool, scale: float) -> dict:
    """Unique tokens needed per source (max over planned runs) + 8% margin + eval docs."""
    if smoke:
        return {s: 3_000_000 if s == "web" else 600_000 for s in C.SOURCES}
    import plan
    need = plan.pool_requirements()
    return {s: int(v * 1.08 * scale) for s, v in need.items()}


def _paths(smoke):
    return C.data_paths(C.SMOKE_ROOT if smoke else C.DATA_ROOT)


# ----------------------------------------------------------------- fetch
def fetch(smoke: bool, scale: float):
    from datasets import load_dataset
    tok = C.secret("HF_TOKEN", required=False)
    P = _paths(smoke)
    req = requirements(smoke, scale)
    for s, spec in C.SOURCES.items():
        d = os.path.join(P["raw"], s)
        os.makedirs(d, exist_ok=True)
        target_bytes = req[s] * spec["bytes_per_token"] * 1.10 + C.EVAL_DOCS * 4000
        done = sorted(glob.glob(os.path.join(d, "shard_*.jsonl")))
        have = sum(os.path.getsize(f) for f in done)
        print(f"[fetch] {s}: target {target_bytes/1e9:.2f} GB text, have {have/1e9:.2f} GB in {len(done)} shards")
        if have >= target_bytes:
            continue
        for attempt in range(5):
            try:
                ds = load_dataset(spec["hf"], spec["config"], split="train", streaming=True, token=tok)
                ds = ds.skip(len(done) * DOCS_PER_SHARD)
                buf, idx = [], len(done)
                for ex in ds:
                    t = ex.get(spec["field"]) or ""
                    if len(t) < 200:
                        continue
                    buf.append(t)
                    if len(buf) == DOCS_PER_SHARD:
                        fn = os.path.join(d, f"shard_{idx:05d}.jsonl")
                        with open(fn + ".tmp", "w") as f:
                            for x in buf:
                                f.write(json.dumps({"t": x}) + "\n")
                        os.replace(fn + ".tmp", fn)
                        have += os.path.getsize(fn)
                        idx, buf = idx + 1, []
                        if idx % 10 == 0:
                            print(f"  {s} shard {idx} {have/1e9:.2f}/{target_bytes/1e9:.2f} GB", flush=True)
                        if have >= target_bytes:
                            break
                break
            except Exception as e:  # network hiccup: resume from finished shards
                print(f"  fetch error ({e}); retry {attempt+1}/5 in 30 s")
                time.sleep(30)
                done = sorted(glob.glob(os.path.join(d, "shard_*.jsonl")))
        # note: skip() counts filtered docs slightly differently -> tiny overlap is possible; dedup in tokenize


def _iter_texts(src, P, limit=None):
    n = 0
    for fn in sorted(glob.glob(os.path.join(P["raw"], src, "shard_*.jsonl"))):
        for line in open(fn):
            yield json.loads(line)["t"]
            n += 1
            if limit and n >= limit:
                return


# ----------------------------------------------------------------- tokenizer
def train_tokenizer(smoke: bool):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    P = _paths(smoke)
    if os.path.exists(P["tokenizer"]):
        print("[tokenizer] exists:", P["tokenizer"])
        return
    budget = 2_000_000 if smoke else 300_000_000          # characters of training text
    def sample():
        for s, spec in C.SOURCES.items():
            cap, used = budget * spec["share"], 0
            for t in _iter_texts(s, P):
                yield t[:20000]
                used += min(len(t), 20000)
                if used > cap:
                    break
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=C.VOCAB, special_tokens=[C.EOS], min_frequency=2,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t0 = time.time()
    tk.train_from_iterator(sample(), trainer=tr)
    assert tk.token_to_id(C.EOS) == 0
    tk.save(P["tokenizer"])
    print(f"[tokenizer] vocab {tk.get_vocab_size()} trained in {time.time()-t0:.0f}s")


def token_bytes(tk) -> np.ndarray:
    """UTF-8 byte length of each token (ByteLevel: one printable char == one byte). EOS counts 0 bytes."""
    vocab = tk.get_vocab()
    tb = np.zeros(max(vocab.values()) + 1, dtype=np.int32)
    for tok, i in vocab.items():
        tb[i] = 0 if tok == C.EOS else len(tok)
    return tb


# ----------------------------------------------------------------- tokenize -> pool
def tokenize(smoke: bool, scale: float):
    from tokenizers import Tokenizer
    import datastream
    P = _paths(smoke)
    if os.path.exists(os.path.join(P["pool"], "meta.json")):
        print("[tokenize] pool exists:", P["pool"], "(delete it to rebuild)")
        return
    tk = Tokenizer.from_file(P["tokenizer"])
    req = requirements(smoke, scale)
    docs = {}
    for s in C.SOURCES:
        t0, out, ntok, seen = time.time(), [], 0, set()
        it = _iter_texts(s, P)
        while True:
            chunk = [t for _, t in zip(range(4096), it)]
            if not chunk:
                break
            for t, enc in zip(chunk, tk.encode_batch(chunk)):
                h = hash(t[:256])
                if h in seen:
                    continue
                seen.add(h)
                ids = enc.ids[: C.DOC_CAP - 1] + [0]
                out.append(np.asarray(ids, dtype=np.uint16))
                ntok += len(ids)
            if ntok > req[s] + C.EVAL_DOCS * 1500:
                break
        rng = np.random.default_rng([C.SEED, len(s)])
        order = rng.permutation(len(out))
        docs[s] = [out[i] for i in order]
        print(f"[tokenize] {s}: {len(out)} docs {ntok/1e6:.1f}M tokens ({time.time()-t0:.0f}s)", flush=True)
    eval_docs = 200 if smoke else C.EVAL_DOCS
    datastream.write_pool(P["pool"], docs, token_bytes(tk), eval_docs,
                          extra_meta=dict(tokenizer=P["tokenizer"], smoke=smoke, created=time.ctime()))
    print("[tokenize] pool written:", P["pool"])


def verify(smoke: bool, scale: float) -> bool:
    import datastream
    P = _paths(smoke)
    pool = datastream.Pool(P["pool"])
    req = requirements(smoke, scale)
    ok = True
    for s in pool.meta["sources"]:
        have = pool.n_tokens(s)
        flag = "OK " if have >= req[s] / 1.08 else "LOW"
        ok &= flag == "OK "
        ev = pool.n_tokens(s, "eval")
        o = pool.off[s, "eval"]
        bpt = pool.token_bytes[np.asarray(pool.tok[s, "eval"][: min(len(pool.tok[s, "eval"]), 2_000_000)])].mean()
        print(f"[verify] {flag} {s:5s} train {have/1e6:8.1f}M (need {req[s]/1.08/1e6:8.1f}M)  eval {ev/1e6:.2f}M "
              f"tokens / {len(o)-1} docs, {bpt:.2f} bytes/token")
    print("[verify]", "PASS" if ok else "FAIL - re-run fetch with --scale 1.2 then delete pool/ and tokenize")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "tokenizer", "tokenize", "verify", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply pool size (e.g. 1.2 if verify is LOW)")
    a = ap.parse_args()
    if a.stage in ("fetch", "all"):
        fetch(a.smoke, a.scale)
    if a.stage in ("tokenizer", "all"):
        train_tokenizer(a.smoke)
    if a.stage in ("tokenize", "all"):
        tokenize(a.smoke, a.scale)
    if a.stage in ("verify", "all"):
        raise SystemExit(0 if verify(a.smoke, a.scale) else 1)
