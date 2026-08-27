"""Hessian-free optimization (Martens 2010): damped CG-Newton with
Levenberg-Marquardt damping adaptation, CG warm starts, and backtracking over
CG iterates.

Closure protocol: like TrustNCG -- closure returns loss with graph (no
backward()); for curvature="ggn" (default, PSD) it returns (loss, outputs).
Solves (B + damping*I) s = -g by CG (warm-started from 0.95 * previous s),
records iterates at exponentially spaced checkpoints, walks them backwards to
the best actual objective value, then adapts damping from the reduction ratio
with Martens' constants (x1.5 grow / x2/3 shrink).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from ..curvature.damping import lm_update
from ..curvature.paramvec import ParamVector
from ..curvature.products import ce_hess_mvp, make_ggn_vp, make_hvp, mse_hess_mvp
from ._common import DiagnosticsMixin, StepTimer

_CLOSURE_MSG = ("HessianFree.step() requires a closure returning the loss with its "
                "graph (curvature='hessian') or (loss, outputs) (curvature='ggn'); "
                "no backward() inside")


def _cg_with_checkpoints(mvp, b, x0, tol, max_iter):
    """CG on A x = b from x0; returns (checkpoints [(iter, x)], final iters)."""
    x = x0.clone()
    r = b - mvp(x)
    d = r.clone()
    rr = r @ r
    checkpoints = []
    next_ckpt = 1
    k = 0
    for k in range(1, max_iter + 1):
        Ad = mvp(d)
        dAd = d @ Ad
        if float(dAd) <= 0:
            break
        alpha = rr / dAd
        x = x + alpha * d
        r = r - alpha * Ad
        rr_new = r @ r
        if k == next_ckpt:
            checkpoints.append((k, x.clone()))
            next_ckpt *= 2
        if float(rr_new.sqrt()) < tol:
            break
        d = r + (rr_new / rr) * d
        rr = rr_new
    if not checkpoints or checkpoints[-1][0] != k:
        checkpoints.append((k, x.clone()))
    return checkpoints, k


class HessianFree(Optimizer, DiagnosticsMixin):
    requires_closure = True

    def __init__(self, params, lr: float = 1.0, curvature: str = "ggn",
                 ggn_loss: str = "mse", damping: float = 1e-1,
                 max_cg_iter: int = 100, cg_tol: float | None = None,
                 warm_start_decay: float = 0.95, backtrack: bool = True):
        if curvature not in ("hessian", "ggn"):
            raise ValueError(f"curvature must be 'hessian' or 'ggn', got {curvature!r}")
        defaults = dict(lr=lr, curvature=curvature, ggn_loss=ggn_loss,
                        max_cg_iter=max_cg_iter, cg_tol=cg_tol,
                        warm_start_decay=warm_start_decay, backtrack=backtrack)
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("HessianFree supports exactly one param group")
        self._params = self.param_groups[0]["params"]
        self._pv = ParamVector(self._params)
        self._damping = float(damping)
        self._prev_s: torch.Tensor | None = None

    @property
    def curvature(self) -> str:
        return self.param_groups[0]["curvature"]

    @classmethod
    def state_layout(cls) -> dict:
        return {"damping": "replicable", "prev_s": "replicable"}

    def state_dict(self):
        sd = super().state_dict()
        sd["hf"] = {"damping": self._damping, "prev_s": self._prev_s}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        extra = sd.pop("hf", None)
        super().load_state_dict(sd)
        if extra is not None:
            self._damping = float(extra["damping"])
            ps = extra["prev_s"]
            self._prev_s = ps.clone() if ps is not None else None

    def _eval_loss(self, closure) -> float:
        with torch.no_grad():
            out = closure()
        return float(out[0] if isinstance(out, tuple) else out)

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
            lam = self._damping
            mvp = lambda v: mvp0(v) + lam * v

            gnorm = g.norm()
            if float(gnorm) < 1e-12:
                self._diag = {"damping": lam, "rho": None, "cg_iters": 0,
                              "backtracked_to": None, "accepted": False,
                              "step_ms": t_all.ms}
                return loss.detach()
            tol = group["cg_tol"]
            if tol is None:
                tol = float(torch.minimum(torch.tensor(0.5, device=g.device),
                                          gnorm.sqrt()) * gnorm)
            x0 = (group["warm_start_decay"] * self._prev_s
                  if self._prev_s is not None else torch.zeros_like(g))
            ckpts, iters = _cg_with_checkpoints(mvp, -g, x0, tol, group["max_cg_iter"])
            # model values must be computed BEFORE any parameter mutation --
            # the mvp closures backprop through the original graph, which
            # in-place parameter updates invalidate.
            qs = {k: float(g @ s + 0.5 * (s @ mvp(s))) for k, s in ckpts}

            f0 = float(loss.detach())
            theta0 = self._pv.gather()
            best = None
            if group["backtrack"]:
                for k, s in reversed(ckpts):     # latest first; walk back while worse
                    self._pv.assign_(theta0)
                    self._pv.add_(s, alpha=group["lr"])
                    f_new = self._eval_loss(closure)
                    if best is None or f_new < best[2]:
                        best = (k, s, f_new)
                    if f_new <= f0:
                        break
            else:
                k, s = ckpts[-1]
                self._pv.assign_(theta0)
                self._pv.add_(s, alpha=group["lr"])
                best = (k, s, self._eval_loss(closure))
            k_sel, s_sel, f_new = best
            q = qs[k_sel]                                       # damped model value
            accepted = f_new < f0
            if not accepted:
                self._pv.assign_(theta0)
            else:
                self._pv.assign_(theta0)
                self._pv.add_(s_sel, alpha=group["lr"])
            rho = (f_new - f0) / q if q < 0 else float("-inf")
            self._damping = lm_update(self._damping, rho, factor=2.0 / 3.0)
            self._prev_s = s_sel.detach().clone()
        self._diag = {"damping": self._damping, "rho": rho, "cg_iters": iters,
                      "backtracked_to": k_sel, "accepted": accepted,
                      "step_ms": t_all.ms}
        return loss.detach()
