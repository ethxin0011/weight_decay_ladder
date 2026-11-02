"""
optim.py - Muon (MAI-Base-2: non-embedding 2-D params) + AdamW (embeddings/head/norms/router)
==============================================================================================
Weight decay is DECOUPLED and set every step by train.py:
    group["weight_decay"] = wd * s_t       s_t = 1 (const)  or  eta_t/eta_max (Muon-SW, scaled)
    p <- p * (1 - lr_t * weight_decay)     => const: 1 - eta_t*wd ; scaled: 1 - eta_t^2/eta_max*wd
Muon update RMS-matched to AdamW (0.2*sqrt(max(m,n)), Moonlight convention) so MAI's AdamW-derived
LR/WD laws are a meaningful starting point. 3-D expert weights [E, m, n] are orthogonalised in one
batched Newton-Schulz call. NS runs in fp32 (T4 has no bf16; fp16 NS is unstable).
"""
from __future__ import annotations

import torch

NS_COEF = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    X = G.float()
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.transpose(-2, -1)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = NS_COEF
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        X = a * X + (b * A + c * A @ A) @ X
    if tr:
        X = X.transpose(-2, -1)
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, mom, wd = g["lr"], g["momentum"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p, dtype=torch.float32)
                buf = st["buf"]
                grad = p.grad.float()
                buf.mul_(mom).add_(grad)
                u = grad.add(buf, alpha=mom) if g["nesterov"] else buf
                o = newton_schulz(u, g["ns_steps"])
                scale = 0.2 * max(p.shape[-2], p.shape[-1]) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(o.to(p.dtype), alpha=-lr * scale)


def param_groups(model):
    """returns (muon_params, adam_decay, adam_nodecay) with names, for logging + checks."""
    muon, ad, an = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("emb.") or n.startswith("head."):
            ad.append((n, p))
        elif p.ndim >= 2 and "router" not in n:
            muon.append((n, p))
        else:
            an.append((n, p))          # norms, router: AdamW without decay
    return muon, ad, an


class Combined:
    """Holds Muon + AdamW; train.py sets lr / weight_decay on every group each step."""

    def __init__(self, model, lr, wd, adam_lr_mult=1.0, betas=(0.9, 0.95), momentum=0.95, ns_steps=5):
        mu, ad, an = param_groups(model)
        self.muon = Muon([p for _, p in mu], lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=wd)
        self.adam = torch.optim.AdamW([dict(params=[p for _, p in ad], weight_decay=wd, decays=True),
                                       dict(params=[p for _, p in an], weight_decay=0.0, decays=False)],
                                      lr=lr * adam_lr_mult, betas=betas, eps=1e-8)
        self.muon.param_groups[0]["decays"] = True
        self.adam_lr_mult = adam_lr_mult
        self.optimizers = [self.muon, self.adam]
        self.n_params = dict(muon=sum(p.numel() for _, p in mu), adam_decay=sum(p.numel() for _, p in ad),
                             adam_nodecay=sum(p.numel() for _, p in an))

    def set(self, lr_t: float, wd_t: float):
        """lr_t = eta_t (scheduled); wd_t = wd * s_t (method-dependent decay scale)."""
        for g in self.muon.param_groups:
            g["lr"], g["weight_decay"] = lr_t, wd_t
        for g in self.adam.param_groups:
            g["lr"] = lr_t * self.adam_lr_mult
            g["weight_decay"] = wd_t if g["decays"] else 0.0

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad(set_to_none=True)

    def state_dict(self):
        return dict(muon=self.muon.state_dict(), adam=self.adam.state_dict())

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd["muon"])
        self.adam.load_state_dict(sd["adam"])
