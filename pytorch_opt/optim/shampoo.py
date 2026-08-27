"""Shampoo: Kronecker-factored full-matrix preconditioning (Gupta et al.,
with the practical schedule/grafting machinery of Anil et al.).

Parameters with ndim >= 2 are reshaped to (shape[0], -1) and preconditioned as
L^(-1/4) G R^(-1/4); roots are recomputed every `precondition_frequency` steps
and reused (stale) in between. 1D/0D parameters, and any matrix with a side
larger than `max_preconditioner_dim`, use a diagonal AdaGrad-style fallback.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from .. import ops
from ._common import DiagnosticsMixin, StepTimer

_GRAFT = ("none", "sgd", "adagrad")


class Shampoo(Optimizer, DiagnosticsMixin):
    def __init__(self, params, lr: float = 0.03, beta2: float = 0.999, eps: float = 1e-8,
                 precondition_frequency: int = 20, start_step: int = 0,
                 max_preconditioner_dim: int = 1024, graft: str = "none",
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 root_method: str = "auto", root_dtype: torch.dtype = torch.float64,
                 diag_eps: float = 1e-10, track_cond: bool = True):
        if graft not in _GRAFT:
            raise ValueError(f"graft must be one of {_GRAFT}, got {graft!r}")
        defaults = dict(lr=lr, beta2=beta2, eps=eps,
                        precondition_frequency=precondition_frequency,
                        start_step=start_step, max_preconditioner_dim=max_preconditioner_dim,
                        graft=graft, momentum=momentum, weight_decay=weight_decay,
                        root_method=root_method, root_dtype=root_dtype,
                        diag_eps=diag_eps, track_cond=track_cond)
        super().__init__(params, defaults)

    @classmethod
    def state_layout(cls) -> dict:
        return {"L": "replicable", "R": "replicable", "L_root": "replicable",
                "R_root": "replicable", "step": "replicable", "stale": "replicable",
                "factor_updates": "replicable",
                "diag_acc": "shardable", "graft_acc": "shardable",
                "momentum_buffer": "shardable"}

    def _diag_direction(self, st, g, group):
        if "diag_acc" not in st:
            st["diag_acc"] = torch.zeros_like(g)
        acc = st["diag_acc"]
        if group["beta2"] == 1.0:
            acc.addcmul_(g, g)
        else:
            acc.mul_(group["beta2"]).addcmul_(g, g, value=1 - group["beta2"])
        return g / (acc.sqrt() + group["diag_eps"])

    @staticmethod
    def _cond(F: torch.Tensor, eps: float) -> float:
        ev = torch.linalg.eigvalsh(F.double())
        lo = float(ev.min()) + eps
        hi = float(ev.max()) + eps
        return hi / max(lo, 1e-300)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        curvature_ms = 0.0
        conds = []
        stales = []
        graft_ratios = []
        n_diag = 0
        with StepTimer() as t_all:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["stale"] = 0
                    matrix_path = p.ndim >= 2
                    if matrix_path:
                        g2 = g.reshape(g.shape[0], -1)
                        if max(g2.shape) > group["max_preconditioner_dim"]:
                            matrix_path = False
                    if matrix_path:
                        m, n = g2.shape
                        if "L" not in st:
                            st["L"] = torch.zeros(m, m, device=p.device, dtype=p.dtype)
                            st["R"] = torch.zeros(n, n, device=p.device, dtype=p.dtype)
                        ops.kron_factor_update_(st["L"], st["R"], g2, beta2=group["beta2"])
                        st["factor_updates"] = st.get("factor_updates", 0) + 1
                        refresh = (st["step"] >= group["start_step"]
                                   and st["step"] % group["precondition_frequency"] == 0)
                        if refresh:
                            # bias-correct EMA factors (Adam-style); sum mode
                            # (beta2 == 1) needs no correction
                            bc = 1.0
                            if group["beta2"] < 1.0:
                                bc = 1.0 - group["beta2"] ** st["factor_updates"]
                            with StepTimer() as t_c:
                                st["L_root"] = ops.inverse_matrix_root(
                                    st["L"] / bc, 4, damping=group["eps"],
                                    method=group["root_method"], root_dtype=group["root_dtype"])
                                st["R_root"] = ops.inverse_matrix_root(
                                    st["R"] / bc, 4, damping=group["eps"],
                                    method=group["root_method"], root_dtype=group["root_dtype"])
                            curvature_ms += t_c.ms
                            st["stale"] = 0
                            if group["track_cond"] and m <= 512 and n <= 512:
                                conds.append((self._cond(st["L"], group["eps"]),
                                              self._cond(st["R"], group["eps"])))
                        else:
                            st["stale"] += 1
                        if "L_root" in st:
                            P = ops.precond_apply_two_sided(st["L_root"], g2, st["R_root"])
                        else:  # bootstrap before first refresh (start_step > 0)
                            P = self._diag_direction(st, g2, group)
                        if group["graft"] != "none":
                            if group["graft"] == "sgd":
                                target = g2.norm()
                            else:  # adagrad
                                if "graft_acc" not in st:
                                    st["graft_acc"] = torch.zeros_like(g2)
                                st["graft_acc"].addcmul_(g2, g2)
                                target = (g2 / (st["graft_acc"].sqrt() + group["diag_eps"])).norm()
                            ratio = float(target / (P.norm() + 1e-30))
                            P = P * ratio
                            graft_ratios.append(ratio)
                        upd = P.reshape(p.shape)
                        stales.append(st["stale"])
                    else:
                        n_diag += 1
                        upd = self._diag_direction(st, g, group)
                    if group["momentum"]:
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(upd)
                        st["momentum_buffer"].mul_(group["momentum"]).add_(upd)
                        upd = st["momentum_buffer"]
                    if group["weight_decay"]:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    p.add_(upd, alpha=-group["lr"])
                    st["step"] += 1
        self._diag = {
            "stale_steps": max(stales) if stales else 0,
            "cond_L": conds[0][0] if conds else None,
            "cond_R": conds[0][1] if conds else None,
            "graft_ratio": graft_ratios[0] if graft_ratios else None,
            "n_diag_params": n_diag,
            "step_ms": t_all.ms,
            "curvature_ms": curvature_ms,
        }
        return loss
