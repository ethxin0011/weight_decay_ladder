"""
datastream.py - source-scaled, step-deterministic data stream (torch-free)
==========================================================================
Implements MAI's *source scaling*: a run of T tokens takes, per source s,
    unique_s = share_s * T / repeats_s   tokens (a prefix of a pre-shuffled pool)
and loops over it repeats_s times (fresh document permutation each repeat).
So every rung sees each source the same number of times; unique-data/params is
constant at fixed TPP.

Determinism: batch(step) depends only on (run seed, step) -> exact resume after
pre-emption without saving data-loader state.

Pool layout (written by data_prep.py):
  pool/<src>_train.bin  uint16 tokens (docs end with EOS, already shuffled)
  pool/<src>_train_off.npy int64 doc offsets (n_docs+1)
  pool/<src>_eval.bin / _eval_off.npy   held-out docs
  pool/token_bytes.npy  int32 UTF-8 bytes per token id (EOS=0)    pool/meta.json

Self test:  python datastream.py --selftest
"""
from __future__ import annotations

import json
import os

import numpy as np

import config as C


class Pool:
    def __init__(self, pool_dir: str):
        self.dir = pool_dir
        self.meta = json.load(open(os.path.join(pool_dir, "meta.json")))
        self.token_bytes = np.load(os.path.join(pool_dir, "token_bytes.npy"))
        self.tok, self.off = {}, {}
        for s in self.meta["sources"]:
            for split in ("train", "eval"):
                self.tok[s, split] = np.memmap(os.path.join(pool_dir, f"{s}_{split}.bin"), dtype=np.uint16, mode="r")
                self.off[s, split] = np.load(os.path.join(pool_dir, f"{s}_{split}_off.npy"))

    def n_tokens(self, s, split="train") -> int:
        return int(self.off[s, split][-1])


def write_pool(pool_dir: str, docs: dict, token_bytes: np.ndarray, eval_docs: int, extra_meta=None):
    """docs[src] = list of 1-D uint16 arrays (each already ends with EOS), already shuffled."""
    os.makedirs(pool_dir, exist_ok=True)
    meta = dict(sources=list(docs), vocab=int(len(token_bytes)), tokens={}, docs={})
    for s, ds in docs.items():
        for split, part in (("eval", ds[:eval_docs]), ("train", ds[eval_docs:])):
            lens = np.array([len(d) for d in part], dtype=np.int64)
            off = np.zeros(len(part) + 1, dtype=np.int64)
            np.cumsum(lens, out=off[1:])
            mm = np.memmap(os.path.join(pool_dir, f"{s}_{split}.bin.tmp"), dtype=np.uint16, mode="w+",
                           shape=(max(1, int(off[-1])),))
            for d, o in zip(part, off[:-1]):
                mm[o:o + len(d)] = d
            mm.flush()
            del mm
            os.replace(os.path.join(pool_dir, f"{s}_{split}.bin.tmp"), os.path.join(pool_dir, f"{s}_{split}.bin"))
            np.save(os.path.join(pool_dir, f"{s}_{split}_off.npy"), off)
            meta["tokens"][f"{s}_{split}"] = int(off[-1])
            meta["docs"][f"{s}_{split}"] = len(part)
    np.save(os.path.join(pool_dir, "token_bytes.npy"), token_bytes.astype(np.int32))
    meta.update(extra_meta or {})
    json.dump(meta, open(os.path.join(pool_dir, "meta.json"), "w"), indent=1)


def _stable_hash(s: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % (2 ** 31)


class _SourceLoop:
    """Virtual concatenation of the unique prefix of one source, repeated with per-repeat doc permutations."""

    def __init__(self, pool: Pool, src: str, unique_tokens: int, seq: int, seed: int):
        off = pool.off[src, "train"]
        n_doc = int(np.searchsorted(off, unique_tokens, side="left"))
        n_doc = max(1, min(n_doc, len(off) - 1))
        self.tok, self.seq = pool.tok[src, "train"], seq
        self.starts, self.lens = off[:n_doc], np.diff(off[:n_doc + 1])
        self.unique = int(self.lens.sum())
        self.win_per_rep = max(1, self.unique // (seq + 1))
        self.seed, self.src = seed * 7919 + _stable_hash(src), src
        self._cache = {}

    def _perm(self, rep: int):
        # stable seed (python's hash() is salted per process and would break resume determinism)
        if rep not in self._cache:
            if len(self._cache) > 2:
                self._cache.clear()
            rng = np.random.default_rng([self.seed, rep])
            p = rng.permutation(len(self.lens)) if rep > 0 else np.arange(len(self.lens))
            cum = np.zeros(len(p) + 1, dtype=np.int64)
            np.cumsum(self.lens[p], out=cum[1:])
            self._cache[rep] = (p, cum)
        return self._cache[rep]

    def window(self, j: int) -> np.ndarray:
        rep, k = divmod(j, self.win_per_rep)
        p, cum = self._perm(rep)
        pos, need = k * (self.seq + 1), self.seq + 1
        out = np.empty(need, dtype=np.int64)
        i = int(np.searchsorted(cum, pos, side="right") - 1)
        filled = 0
        while filled < need:
            d = p[i % len(p)]
            a = pos - cum[i] if filled == 0 else 0
            take = min(need - filled, int(self.lens[d] - a))
            s0 = int(self.starts[d] + a)
            out[filled:filled + take] = self.tok[s0:s0 + take]
            filled += take
            i += 1
        return out


class Stream:
    """batch(step) -> int64 array [batch_seqs, seq+1]. Fully determined by (seed, steps, batch, pool)."""

    def __init__(self, pool: Pool, steps: int, batch_seqs: int, seq: int, seed: int = 0,
                 sources: dict | None = None, allow_short: bool = False):
        sources = sources or C.SOURCES
        self.steps, self.B, self.seq = steps, batch_seqs, seq
        n_win = steps * batch_seqs
        names = [s for s in sources if s in pool.meta["sources"]]
        shares = np.array([sources[s]["share"] for s in names], dtype=float)
        shares /= shares.sum()
        counts = np.floor(shares * n_win).astype(np.int64)
        counts[0] += n_win - counts.sum()
        self.loops, self.report = [], {}
        for s, n in zip(names, counts):
            reps = sources[s]["repeats"]
            need = int(np.ceil(n * (seq + 1) / reps))
            have = pool.n_tokens(s)
            if need > have and not allow_short:
                raise ValueError(f"pool too small for {s}: need {need/1e6:.1f}M unique tokens, have "
                                 f"{have/1e6:.1f}M (run data_prep with a larger plan or pass allow_short)")
            lp = _SourceLoop(pool, s, min(need, have), seq, seed)
            self.loops.append(lp)
            self.report[s] = dict(windows=int(n), unique_tokens=lp.unique,
                                  effective_repeats=round(n * (seq + 1) / max(1, lp.unique), 3))
        rng = np.random.default_rng([seed, 99])
        lab = np.repeat(np.arange(len(names), dtype=np.int8), counts)
        rng.shuffle(lab)
        self.lab = lab
        self.local = np.zeros(n_win, dtype=np.int64)
        for k in range(len(names)):
            m = lab == k
            self.local[m] = np.arange(int(m.sum()))
        self.names = names

    def batch(self, step: int) -> np.ndarray:
        a = (step % self.steps) * self.B
        return np.stack([self.loops[self.lab[g]].window(int(self.local[g])) for g in range(a, a + self.B)])


def eval_windows(pool: Pool, src: str, n: int, seq: int) -> np.ndarray:
    """First n non-overlapping windows of the held-out docs (fixed across all runs)."""
    tok = pool.tok[src, "eval"]
    n = min(n, (len(tok) - 1) // (seq + 1))
    return np.asarray(tok[: n * (seq + 1)], dtype=np.int64).reshape(n, seq + 1)


def _selftest():
    import tempfile
    rng = np.random.default_rng(0)
    eos = 0
    docs = {}
    for s in C.SOURCES:
        ds = []
        for _ in range(400):
            d = rng.integers(1, 500, size=int(rng.integers(20, 300))).astype(np.uint16)
            ds.append(np.append(d, np.uint16(eos)))
        docs[s] = ds
    tb = np.full(500, 3, dtype=np.int32)
    tb[0] = 0
    with tempfile.TemporaryDirectory() as td:
        write_pool(td, docs, tb, eval_docs=20)
        pool = Pool(td)
        st = Stream(pool, steps=40, batch_seqs=8, seq=64, seed=3, allow_short=True)
        b1 = st.batch(17)
        st2 = Stream(Pool(td), steps=40, batch_seqs=8, seq=64, seed=3, allow_short=True)
        assert np.array_equal(b1, st2.batch(17)), "not deterministic"
        assert b1.shape == (8, 65)
        assert not np.array_equal(st.batch(0), st.batch(1))
        # source scaling: repeats of code/math == 4 when pool is large enough
        st3 = Stream(pool, steps=60, batch_seqs=8, seq=64, seed=1)
        for s, r in st3.report.items():
            assert abs(r["effective_repeats"] - C.SOURCES[s]["repeats"]) < 0.35 * C.SOURCES[s]["repeats"], (s, r)
        ev = eval_windows(pool, "web", 5, 64)
        assert ev.shape == (5, 65)
        print("datastream selftest OK", st3.report)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
