"""Muon: momentum orthogonalized by Newton-Schulz iteration (Jordan et al.).

Routing (per the design spec): orthogonalized momentum applies to parameters
with ndim >= 2 (conv kernels flattened to (out, -1)); everything else falls
back to an internal AdamW. Override per param group with ``use_preconditioner`` (or legacy ``use_muon``).
Embeddings and output heads belong in an AdamW group (``use_muon=False``) --
routing by ndim cannot detect them.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from .. import ops
from ._common import DiagnosticsMixin, StepTimer, adamw_update, matrix_path, validate_groups, validate_checkpoint_groups

_LR_ADJUST = ("spectral", "match_rms_adam", "none")


class Muon(Optimizer, DiagnosticsMixin):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0,
                 lr_adjust: str = "spectral", adamw_lr: float = 3e-4,
                 adamw_betas: tuple = (0.9, 0.95), adamw_eps: float = 1e-8,
                 adamw_wd: float = 0.0):
        if lr_adjust not in _LR_ADJUST:
            raise ValueError(f"lr_adjust must be one of {_LR_ADJUST}, got {lr_adjust!r}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, lr_adjust=lr_adjust, adamw_lr=adamw_lr,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps, adamw_wd=adamw_wd,
                        use_muon=None, use_preconditioner=None)
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["use_muon"] is not None:
                if (group["use_preconditioner"] is not None
                        and group["use_preconditioner"] != group["use_muon"]):
                    raise ValueError("use_muon conflicts with use_preconditioner")
                # Keep legacy use_muon=False using its adamw_* options.
                if group["use_preconditioner"] is None and not group["use_muon"]:
                    group["lr"] = group["adamw_lr"]
                    group["weight_decay"] = group["adamw_wd"]
                group["use_preconditioner"] = group["use_muon"]
        validate_groups(self.param_groups)

    @classmethod
    def state_layout(cls) -> dict:
        return {"momentum_buffer": "shardable", "exp_avg": "shardable",
                "exp_avg_sq": "shardable", "step": "shardable"}

    @staticmethod
    def _scale(shape, lr_adjust: str) -> float:
        m, n = shape[-2], shape[-1]
        if lr_adjust == "spectral":
            return max(1.0, m / n) ** 0.5
        if lr_adjust == "match_rms_adam":
            return 0.2 * max(m, n) ** 0.5
        return 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with StepTimer() as t:
            n_muon = n_adamw = 0
            sq_sum = 0.0
            n_el = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    routed = matrix_path(p, group)
                    st = self.state[p]
                    if routed:
                        n_muon += 1
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(p)
                        buf = st["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(g)
                        u = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                        u2 = u if u.ndim == 2 else u.reshape(u.shape[0], -1)
                        O = ops.newton_schulz_orthogonalize(u2, steps=group["ns_steps"])
                        scale = self._scale(O.shape, group["lr_adjust"])
                        if group["weight_decay"]:
                            p.mul_(1.0 - group["lr"] * group["weight_decay"])
                        upd = O.reshape(p.shape)
                        eff = group["lr"] * scale
                        p.add_(upd, alpha=-eff)
                        sq_sum += float(upd.pow(2).sum()) * eff * eff
                        n_el += upd.numel()
                    else:
                        n_adamw += 1
                        upd, eff = adamw_update(p, g, st, group)
                        sq_sum += float(upd.pow(2).sum()) * eff * eff
                        n_el += upd.numel()
        self._diag = {"update_rms": (sq_sum / max(n_el, 1)) ** 0.5,
                      "n_muon_params": n_muon, "n_adamw_params": n_adamw,
                      "step_ms": t.ms}
        return loss

    def load_state_dict(self, state_dict):
        validate_checkpoint_groups(self.param_groups, state_dict["param_groups"])
        return super().load_state_dict(state_dict)
