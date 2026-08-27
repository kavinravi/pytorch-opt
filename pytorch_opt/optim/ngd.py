"""Exact natural gradient descent via the dense (true) Fisher.

Builds the full Fisher matrix column-by-column from Fisher-vector products
(GGN identity for 'mse'/'ce'), then solves (F + damping I) d = g. O(n^2)
memory, O(n^3) solve -- small models only; intended as the exact oracle the
Kronecker approximations are compared against.

Closure protocol (same as TrustNCG's ggn mode): closure returns
``(loss, outputs)`` with the graph attached, no backward() inside.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from ..curvature.paramvec import ParamVector
from ..curvature.products import make_fisher_vp
from ._common import DiagnosticsMixin, StepTimer

_CLOSURE_MSG = ("NGD.step() requires a closure returning (loss, outputs) with the "
                "autograd graph attached (no backward() inside)")


class NGD(Optimizer, DiagnosticsMixin):
    requires_closure = True
    curvature = "ggn"   # tells harnesses the closure must return (loss, outputs)

    def __init__(self, params, lr: float = 1.0, damping: float = 1e-4,
                 loss_type: str = "mse", momentum: float = 0.0,
                 max_params: int = 5000):
        if loss_type not in ("mse", "ce"):
            raise ValueError(f"loss_type must be 'mse' or 'ce', got {loss_type!r}")
        defaults = dict(lr=lr, damping=damping, loss_type=loss_type,
                        momentum=momentum, max_params=max_params)
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("NGD supports exactly one param group")
        self._params = self.param_groups[0]["params"]
        self._pv = ParamVector(self._params)
        if self._pv.numel > self.param_groups[0]["max_params"]:
            raise ValueError(
                f"NGD builds a dense {self._pv.numel}x{self._pv.numel} Fisher; "
                f"refusing above max_params={self.param_groups[0]['max_params']}")
        self._momentum_buf: torch.Tensor | None = None

    @classmethod
    def state_layout(cls) -> dict:
        return {"momentum_buffer": "shardable"}

    def state_dict(self):
        sd = super().state_dict()
        sd["ngd"] = {"momentum_buffer": self._momentum_buf}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        extra = sd.pop("ngd", None)
        super().load_state_dict(sd)
        if extra is not None and extra["momentum_buffer"] is not None:
            self._momentum_buf = extra["momentum_buffer"].clone()

    def step(self, closure=None):
        if closure is None:
            raise TypeError(_CLOSURE_MSG)
        group = self.param_groups[0]
        with StepTimer() as t_all:
            with torch.enable_grad():
                out = closure()
            if not (isinstance(out, tuple) and len(out) == 2):
                raise TypeError(_CLOSURE_MSG)
            loss, outputs = out
            grads = torch.autograd.grad(loss, self._params, retain_graph=True)
            g = torch.cat([t.reshape(-1) for t in grads])
            n = g.numel()
            with StepTimer() as t_c:
                fvp = make_fisher_vp(outputs, group["loss_type"], self._params)
                I = torch.eye(n, device=g.device, dtype=g.dtype)
                F = torch.stack([fvp(I[i]) for i in range(n)], dim=1)
                F = 0.5 * (F + F.T) + group["damping"] * I
                d = torch.linalg.solve(F, g)
            if group["momentum"]:
                if self._momentum_buf is None:
                    self._momentum_buf = torch.zeros_like(d)
                self._momentum_buf.mul_(group["momentum"]).add_(d)
                d = self._momentum_buf
            self._pv.add_(d, alpha=-group["lr"])
        self._diag = {"fisher_cond": float(torch.linalg.cond(F)),
                      "nat_grad_norm": float(d.norm()),
                      "step_ms": t_all.ms, "curvature_ms": t_c.ms}
        return loss.detach()
