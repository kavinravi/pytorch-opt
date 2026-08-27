"""SOAP: Adam run in Shampoo's Kronecker-factor eigenbasis (Vyas et al.).

For each 2D+ parameter (reshaped (out, -1)) maintain Shampoo factors L, R and
their eigenbases Q_L, Q_R (refreshed every `precondition_frequency` steps).
The first moment lives in the original space (re-rotated on use); the second
moment lives in the rotated space. With identity rotations SOAP is exactly
Adam -- that identity is the analytic test. 1D/0D and oversize parameters get
plain Adam.

On basis refresh the rotated second moment is transported into the new basis
through the squared basis-change matrices (the covariance-diagonal transform;
exact under permutations/sign flips, where the reference implementation's
eigenvalue-sort reordering is a special case).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from .. import ops
from ._common import DiagnosticsMixin, StepTimer


class SOAP(Optimizer, DiagnosticsMixin):
    def __init__(self, params, lr: float = 3e-3, betas: tuple = (0.95, 0.95),
                 shampoo_beta: float = 0.95, eps: float = 1e-8,
                 precondition_frequency: int = 10,
                 max_preconditioner_dim: int = 1024, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                        precondition_frequency=precondition_frequency,
                        max_preconditioner_dim=max_preconditioner_dim,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @classmethod
    def state_layout(cls) -> dict:
        return {"L": "replicable", "R": "replicable", "Q_L": "replicable",
                "Q_R": "replicable", "step": "shardable", "stale": "replicable",
                "exp_avg": "shardable", "exp_avg_sq": "shardable"}

    @staticmethod
    def _adam_update(p, g, st, group, transform=None, back=None):
        """torch.optim.Adam formula; moments optionally kept/used in a rotated
        space via transform/back callables."""
        b1, b2 = group["betas"]
        st["step"] += 1
        t = st["step"]
        st["exp_avg"].lerp_(g, 1 - b1)                     # original space
        g_rot = transform(g) if transform is not None else g
        st["exp_avg_sq"].mul_(b2).addcmul_(g_rot, g_rot, value=1 - b2)
        m_rot = transform(st["exp_avg"]) if transform is not None else st["exp_avg"]
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        denom = (st["exp_avg_sq"].sqrt() / bc2 ** 0.5).add_(group["eps"])
        upd_rot = m_rot / denom
        upd = back(upd_rot) if back is not None else upd_rot
        if group["weight_decay"]:
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
        p.add_(upd, alpha=-group["lr"] / bc1)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        curvature_ms = 0.0
        stales = []
        n_adam = 0
        with StepTimer() as t_all:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    matrix_path = p.ndim >= 2
                    if matrix_path:
                        g2 = g.reshape(g.shape[0], -1)
                        if max(g2.shape) > group["max_preconditioner_dim"]:
                            matrix_path = False
                    if not matrix_path:
                        n_adam += 1
                        if "exp_avg" not in st:
                            st["exp_avg"] = torch.zeros_like(p)
                            st["exp_avg_sq"] = torch.zeros_like(p)
                            st["step"] = 0
                        self._adam_update(p, g, st, group)
                        continue
                    m, n = g2.shape
                    if "L" not in st:
                        st["L"] = torch.zeros(m, m, device=p.device, dtype=p.dtype)
                        st["R"] = torch.zeros(n, n, device=p.device, dtype=p.dtype)
                        st["exp_avg"] = torch.zeros_like(g2)
                        st["exp_avg_sq"] = torch.zeros_like(g2)
                        st["step"] = 0
                        st["stale"] = 0
                    ops.kron_factor_update_(st["L"], st["R"], g2, beta2=group["shampoo_beta"])
                    if st["step"] % group["precondition_frequency"] == 0:
                        with StepTimer() as t_c:
                            Q_L_new = torch.linalg.eigh(st["L"].double())[1].to(p.dtype)
                            Q_R_new = torch.linalg.eigh(st["R"].double())[1].to(p.dtype)
                            if "Q_L" in st:
                                # transport the rotated second moment into the
                                # new basis: v is the diagonal of a covariance
                                # in the old basis, so it maps through the
                                # squared basis-change matrices (exact for
                                # permutations/sign flips).
                                C_L = st["Q_L"].T @ Q_L_new
                                C_R = st["Q_R"].T @ Q_R_new
                                st["exp_avg_sq"] = (C_L.pow(2).T
                                                    @ st["exp_avg_sq"]
                                                    @ C_R.pow(2))
                            st["Q_L"], st["Q_R"] = Q_L_new, Q_R_new
                        curvature_ms += t_c.ms
                        st["stale"] = 0
                    else:
                        st["stale"] += 1
                    stales.append(st["stale"])
                    Q_L, Q_R = st["Q_L"], st["Q_R"]
                    self._adam_update(
                        p.view(m, n), g2, st, group,
                        transform=lambda x: Q_L.T @ x @ Q_R,
                        back=lambda x: Q_L @ x @ Q_R.T)
        self._diag = {"stale_steps": max(stales) if stales else 0,
                      "n_adam_params": n_adam, "step_ms": t_all.ms,
                      "curvature_ms": curvature_ms}
        return loss
