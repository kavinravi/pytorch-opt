"""Trust-region Newton-CG (Steihaug-Toint), closure-based.

API asymmetry (by design, see spec): ``step(closure)`` REQUIRES a closure that
re-evaluates the model and returns the loss WITH its autograd graph and does
NOT call backward(). For ``curvature="ggn"`` the closure must return
``(loss, outputs)`` so the Gauss-Newton product can be built from the model
outputs. Trust-region methods evaluate the objective more than once per step;
there is no way around the closure.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from ..curvature.paramvec import ParamVector
from ..curvature.products import ce_hess_mvp, make_ggn_vp, make_hvp, mse_hess_mvp
from ..curvature.steihaug import steihaug_cg
from ._common import DiagnosticsMixin, StepTimer

_CLOSURE_MSG = (
    "TrustNCG.step() requires a closure that recomputes the loss and returns "
    "it WITH the autograd graph (do not call backward() inside it). For "
    "curvature='ggn', return (loss, outputs)."
)


class TrustNCG(Optimizer, DiagnosticsMixin):
    requires_closure = True

    def __init__(self, params, delta0: float = 1.0, delta_max: float = 1e3,
                 eta: float = 0.15, shrink_threshold: float = 0.25,
                 expand_threshold: float = 0.75, curvature: str = "hessian",
                 ggn_loss: str = "mse", damping: float = 0.0,
                 max_cg_iter: int = 250, cg_tol: float | None = None):
        if curvature not in ("hessian", "ggn"):
            raise ValueError(f"curvature must be 'hessian' or 'ggn', got {curvature!r}")
        if ggn_loss not in ("mse", "ce"):
            raise ValueError(f"ggn_loss must be 'mse' or 'ce', got {ggn_loss!r}")
        defaults = dict(delta0=delta0, delta_max=delta_max, eta=eta,
                        shrink_threshold=shrink_threshold, expand_threshold=expand_threshold,
                        curvature=curvature, ggn_loss=ggn_loss, damping=damping,
                        max_cg_iter=max_cg_iter, cg_tol=cg_tol)
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("TrustNCG supports exactly one param group")
        self._params = self.param_groups[0]["params"]
        self._pv = ParamVector(self._params)
        self._delta = float(delta0)

    # expose for _testing.run_steps protocol detection
    @property
    def curvature(self) -> str:
        return self.param_groups[0]["curvature"]

    @classmethod
    def state_layout(cls) -> dict:
        return {"delta": "replicable"}

    def state_dict(self):
        sd = super().state_dict()
        sd["trust"] = {"delta": self._delta}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        trust = sd.pop("trust", None)
        super().load_state_dict(sd)
        if trust is not None:
            self._delta = float(trust["delta"])

    def step(self, closure=None):
        if closure is None:
            raise TypeError(_CLOSURE_MSG)
        group = self.param_groups[0]
        with StepTimer() as t_all:
            with torch.enable_grad():
                out = closure()
            if isinstance(out, tuple):
                loss, outputs = out
            else:
                loss, outputs = out, None

            if group["curvature"] == "ggn":
                if outputs is None:
                    raise TypeError(_CLOSURE_MSG)
                hess = (mse_hess_mvp(outputs) if group["ggn_loss"] == "mse"
                        else ce_hess_mvp(outputs))
                mvp0 = make_ggn_vp(outputs, hess, self._params)
                grads = torch.autograd.grad(loss, self._params, retain_graph=True)
                g = torch.cat([t.reshape(-1) for t in grads])
            else:
                mvp0, g = make_hvp(loss, self._params)

            damping = group["damping"]
            mvp = (lambda v: mvp0(v) + damping * v) if damping else mvp0

            if float(g.norm()) < 1e-12:
                self._diag = {"delta": self._delta, "rho": None, "cg_iters": 0,
                              "cg_reason": "zero_grad", "accepted": False,
                              "pred_reduction": 0.0, "step_ms": t_all.ms}
                return loss.detach()

            s, info = steihaug_cg(mvp, g, self._delta, tol=group["cg_tol"],
                                  max_iter=group["max_cg_iter"])
            pred = float(g @ s + 0.5 * (s @ mvp(s)))   # model reduction m(s) (< 0 expected)
            theta0 = self._pv.gather()
            self._pv.add_(s)
            with torch.no_grad():
                out2 = closure()
                new_loss = float(out2[0] if isinstance(out2, tuple) else out2)

            loss_val = float(loss.detach())
            rho = (loss_val - new_loss) / (-pred) if pred < 0 else float("-inf")
            accepted = rho > group["eta"]
            if not accepted:
                self._pv.assign_(theta0)   # exact revert
            snorm = float(s.norm())
            if rho < group["shrink_threshold"]:
                self._delta *= 0.25
            elif rho > group["expand_threshold"] and snorm >= 0.99 * self._delta:
                self._delta = min(2.0 * self._delta, group["delta_max"])
        self._diag = {"delta": self._delta, "rho": rho, "cg_iters": info["iters"],
                      "cg_reason": info["reason"], "accepted": accepted,
                      "pred_reduction": pred, "step_ms": t_all.ms}
        return loss.detach()
