"""
model.py - MAI-Base-2-style MoE language model (scaled down), CPU + T4 friendly
===============================================================================
Mirrors MAI-Base-2 layout choices that matter for the optimizer study:
  * MoE in EVERY layer: 1 shared expert (full width, not projected) + top-k of E routed experts
  * routed experts live in a down-projected latent (MAI "MoE down projection"), dropless routing
  * local sliding-window attention with a global layer every 6 layers (+ last layer), GQA, QK-norm, RoPE
  * untied embeddings, RMSNorm pre-norm, dropout from MAI's law (~0 at these scales)
Loss = CE + aux_coef * switch load-balance + z_coef * router z-loss. Logits in fp32.
No custom kernels: runs identically on CPU (fp32) and T4 (fp16 autocast; T4 has no bf16).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_cache(seq, hd, base, device):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(seq, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):            # x [B,H,T,hd], half-split convention
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[: x.shape[-2]].to(x.dtype), sin[: x.shape[-2]].to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, c, is_global):
        super().__init__()
        self.c, self.is_global = c, is_global
        hd = c.head_dim
        self.wq = nn.Linear(c.d_model, c.n_head * hd, bias=False)
        self.wk = nn.Linear(c.d_model, c.n_kv_head * hd, bias=False)
        self.wv = nn.Linear(c.d_model, c.n_kv_head * hd, bias=False)
        self.wo = nn.Linear(c.n_head * hd, c.d_model, bias=False)
        self.qn, self.kn = RMSNorm(hd), RMSNorm(hd)
        self.wo.is_out_proj = True

    def forward(self, x, cos, sin, local_mask):
        B, T, _ = x.shape
        c, hd = self.c, self.c.head_dim
        q = self.qn(self.wq(x).view(B, T, c.n_head, hd)).transpose(1, 2)
        k = self.kn(self.wk(x).view(B, T, c.n_kv_head, hd)).transpose(1, 2)
        v = self.wv(x).view(B, T, c.n_kv_head, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = c.n_head // c.n_kv_head
        if rep > 1:
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        p = c.dropout if self.training else 0.0
        if self.is_global or c.window >= T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=local_mask[:T, :T], dropout_p=p)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class MoE(nn.Module):
    """shared SwiGLU expert + dropless top-k routed SwiGLU experts in a d/2 latent."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        d, E, lt, f = c.d_model, c.n_expert, c.latent, c.d_ff_expert
        self.router = nn.Linear(d, E, bias=False)
        self.sh_gate = nn.Linear(d, c.d_ff_shared, bias=False)
        self.sh_up = nn.Linear(d, c.d_ff_shared, bias=False)
        self.sh_down = nn.Linear(c.d_ff_shared, d, bias=False)
        self.down = nn.Linear(d, lt, bias=False)          # MAI-style MoE down projection (shared by experts)
        self.up = nn.Linear(lt, d, bias=False)
        self.w_gate = nn.Parameter(torch.empty(E, lt, f))  # 3-D expert weights (Muon batches them)
        self.w_up = nn.Parameter(torch.empty(E, lt, f))
        self.w_down = nn.Parameter(torch.empty(E, f, lt))
        self.sh_down.is_out_proj = True
        self.up.is_out_proj = True
        self.w_down.is_out_proj = True

    def forward(self, x):
        B, T, d = x.shape
        c = self.c
        xf = x.reshape(-1, d)
        shared = self.sh_down(F.silu(self.sh_gate(xf)) * self.sh_up(xf))
        logits = self.router(xf).float()
        probs = logits.softmax(-1)
        topv, topi = probs.topk(c.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        # aux losses (Switch load balance + router z-loss)
        frac = F.one_hot(topi, c.n_expert).float().sum(1).mean(0) / c.top_k
        aux = c.n_expert * (frac * probs.mean(0)).sum()
        z = torch.logsumexp(logits, -1).pow(2).mean()
        # dropless dispatch: sort token-slots by expert
        h = self.down(xf)
        flat_e = topi.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        tok_idx = order // c.top_k
        counts = torch.bincount(flat_e, minlength=c.n_expert).tolist()
        hs = h[tok_idx]
        outs, a = [], 0
        for e, n in enumerate(counts):
            if n:
                he = hs[a:a + n]
                outs.append((F.silu(he @ self.w_gate[e]) * (he @ self.w_up[e])) @ self.w_down[e])
            a += n
        y = torch.cat(outs, 0) * topv.reshape(-1)[order].unsqueeze(-1).to(h.dtype)
        routed = torch.zeros_like(h).index_add_(0, tok_idx, y)
        out = shared + self.up(routed)
        load = torch.tensor(counts, dtype=torch.float32)
        return out.view(B, T, d), aux, z, load


class Block(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.attn = Attention(c, c.is_global(i))
        self.moe = MoE(c)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, mask):
        x = x + self.drop(self.attn(self.n1(x), cos, sin, mask))
        m, aux, z, load = self.moe(self.n2(x))
        return x + self.drop(m), aux, z, load


class MoELM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList([Block(c, i) for i in range(c.n_layer)])
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)    # untied (MAI-Base-2)
        cos, sin = rope_cache(c.seq_len, c.head_dim, c.rope_base, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        i = torch.arange(c.seq_len)
        mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < c.window)
        self.register_buffer("local_mask", mask, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        out_std = 0.02 / math.sqrt(2 * self.c.n_layer)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=out_std if getattr(m, "is_out_proj", False) else 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, MoE):
                for p in (m.w_gate, m.w_up):
                    nn.init.normal_(p, std=0.02)
                nn.init.normal_(m.w_down, std=out_std)

    def forward(self, idx, targets=None, reduction="mean"):
        x = self.emb(idx)
        aux = z = 0.0
        loads = []
        for b in self.blocks:
            x, a, zz, ld = b(x, self.cos, self.sin, self.local_mask)
            aux, z = aux + a, z + zz
            loads.append(ld)
        logits = self.head(self.norm(x)).float()
        if targets is None:
            return logits
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)
        n = len(self.blocks)
        return ce, aux / n, z / n, torch.stack(loads)
