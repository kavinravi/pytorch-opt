"""PSGD with a Kronecker-factored (affine) gradient-whitening preconditioner
(Xi-Lin Li; update math follows psgd_torch's update_precond_affine_math_ for
real tensors).

Each parameter (reshaped (shape[0], -1); 1D as a column) carries triangular
factors Q_L, Q_R with P = (Q_L^T Q_L) (x) (Q_R^T Q_R) fitted to whiten
gradients: at the criterion's optimum E[(P_L G P_R)(.)^T] is proportional to
the identity, i.e. P approximates (E[vec g vec g^T])^(-1/2) in Kronecker form.
Whitening pairs are (v, g) with v ~ N(0, I) from an internal seeded generator.
Sides larger than `max_preconditioner_dim` fall back to diagonal factors (a
mixed dense/diagonal pair is simplified to diagonal/diagonal).

Deviations from the reference, both for determinism: Q balancing runs every
100 updates instead of with probability 0.01, and preconditioner updates gate
on a seeded generator when update_probability < 1.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from ._common import DiagnosticsMixin, StepTimer

_TINY = 1.2e-38


def _norm_lower_bound(A: torch.Tensor) -> torch.Tensor:
    """Cheap tight lower bound on the spectral norm (Li's norm_lower_bound)."""
    max_abs = A.abs().max()
    if float(max_abs) == 0.0:
        return max_abs
    A = A / max_abs
    aa = A * A
    value0, i = torch.max(aa.sum(dim=0), 0)
    value1, j = torch.max(aa.sum(dim=1), 0)
    if float(value0) > float(value1):
        x = A[:, i] @ A
        return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A.T)
    x = A @ A[j]
    return max_abs * torch.linalg.vector_norm(A.T @ (x / torch.linalg.vector_norm(x)))


def _update_dense_dense_(Ql, Qr, v, G, step):
    """Whitening pair (v, G), both factors upper-triangular (real case)."""
    A = Ql @ G @ Qr.T
    Bh = torch.linalg.solve_triangular(
        Ql.T, torch.linalg.solve_triangular(Qr, v, upper=True, left=False), upper=False)
    AAh, BhB = A @ A.T, Bh @ Bh.T
    AhA, BBh = A.T @ A, Bh.T @ Bh
    grad1 = torch.triu(AAh - BhB)
    grad2 = torch.triu(AhA - BBh)
    step1 = step / (_norm_lower_bound(AAh + BhB) + _TINY)   # '2nd' normalizer
    step2 = step / (_norm_lower_bound(AhA + BBh) + _TINY)
    Ql.sub_(step1 * grad1 @ Ql)
    Qr.sub_(step2 * grad2 @ Qr)


def _update_diag_diag_(ql, qr, v, G, step):
    A = ql[:, None] * G * qr
    Bh = v / qr / ql[:, None]
    AAc1, BBc1 = (A * A).sum(dim=1), (Bh * Bh).sum(dim=1)
    AAc2, BBc2 = (A * A).sum(dim=0), (Bh * Bh).sum(dim=0)
    grad1 = AAc1 - BBc1
    grad2 = AAc2 - BBc2
    step1 = step / ((AAc1 + BBc1).max() + _TINY)
    step2 = step / ((AAc2 + BBc2).max() + _TINY)
    ql.sub_(step1 * grad1 * ql)
    qr.sub_(step2 * grad2 * qr)


def _precond_grad(Ql, Qr, G):
    if Ql.ndim == 2:
        return torch.linalg.multi_dot([Ql.T, Ql, G, Qr.T, Qr])
    return (Ql * Ql)[:, None] * G * (Qr * Qr)


class PSGD(Optimizer, DiagnosticsMixin):
    def __init__(self, params, lr: float = 0.01, precond_lr: float = 0.1,
                 momentum: float = 0.9, init_scale: float | None = None,
                 max_preconditioner_dim: int = 1024,
                 update_probability: float = 1.0, grad_clip_max_norm: float | None = None,
                 weight_decay: float = 0.0, seed: int = 0):
        defaults = dict(lr=lr, precond_lr=precond_lr, momentum=momentum,
                        init_scale=init_scale,
                        max_preconditioner_dim=max_preconditioner_dim,
                        update_probability=update_probability,
                        grad_clip_max_norm=grad_clip_max_norm,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)
        self._updates = 0

    @classmethod
    def state_layout(cls) -> dict:
        return {"Ql": "replicable", "Qr": "replicable",
                "momentum_buffer": "shardable"}

    def state_dict(self):
        sd = super().state_dict()
        sd["psgd"] = {"gen_state": self._gen.get_state(), "updates": self._updates}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        extra = sd.pop("psgd", None)
        super().load_state_dict(sd)
        if extra is not None:
            self._gen.set_state(extra["gen_state"])
            self._updates = int(extra["updates"])

    def _init_factors(self, st, g2, group):
        m, n = g2.shape
        scale = group["init_scale"]
        if scale is None:
            scale = float((g2.pow(2).mean() + _TINY) ** (-1.0 / 4.0))
        dense_l = m <= group["max_preconditioner_dim"]
        dense_r = n <= group["max_preconditioner_dim"]
        if not (dense_l and dense_r):     # simplify mixed pairs to diag/diag
            dense_l = dense_r = False
        if dense_l:
            st["Ql"] = scale * torch.eye(m, device=g2.device, dtype=g2.dtype)
            st["Qr"] = scale * torch.eye(n, device=g2.device, dtype=g2.dtype)
        else:
            st["Ql"] = scale * torch.ones(m, device=g2.device, dtype=g2.dtype)
            st["Qr"] = scale * torch.ones(n, device=g2.device, dtype=g2.dtype)

    def _update_factors(self, st, g2, group):
        v = torch.randn(g2.shape, generator=self._gen, dtype=g2.dtype).to(g2.device)
        if st["Ql"].ndim == 2:
            _update_dense_dense_(st["Ql"], st["Qr"], v, g2, group["precond_lr"])
        else:
            _update_diag_diag_(st["Ql"], st["Qr"], v, g2, group["precond_lr"])
        self._updates += 1
        if self._updates % 100 == 0:      # deterministic balancing
            rho = (st["Ql"].abs().max() / st["Qr"].abs().max()).sqrt()
            st["Ql"].div_(rho)
            st["Qr"].mul_(rho)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        n_diag = 0
        with StepTimer() as t_all:
            curvature_ms = 0.0
            updates: list[tuple[torch.Tensor, torch.Tensor]] = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g2 = p.grad.reshape(p.shape[0], -1) if p.ndim >= 2 else p.grad.reshape(-1, 1)
                    st = self.state[p]
                    if "Ql" not in st:
                        self._init_factors(st, g2, group)
                    if st["Ql"].ndim == 1:
                        n_diag += 1
                    do_update = group["update_probability"] >= 1.0 or bool(
                        torch.rand((), generator=self._gen) < group["update_probability"])
                    if do_update:
                        with StepTimer() as t_c:
                            self._update_factors(st, g2, group)
                        curvature_ms += t_c.ms
                    if group["momentum"]:
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(g2)
                        st["momentum_buffer"].mul_(group["momentum"]).add_(
                            g2, alpha=1 - group["momentum"])
                        m = st["momentum_buffer"]
                    else:
                        m = g2
                    updates.append((p, _precond_grad(st["Ql"], st["Qr"], m).reshape(p.shape)))
            if self.param_groups[0]["grad_clip_max_norm"] is not None:
                total = torch.sqrt(sum(u.pow(2).sum() for _, u in updates))
                cap = self.param_groups[0]["grad_clip_max_norm"]
                if float(total) > cap:
                    updates = [(p, u * (cap / (float(total) + _TINY))) for p, u in updates]
            for p, u in updates:
                group = next(g for g in self.param_groups if any(q is p for q in g["params"]))
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])
        self._diag = {"n_diag_params": n_diag, "q_updates": self._updates,
                      "step_ms": t_all.ms, "curvature_ms": curvature_ms}
        return loss
