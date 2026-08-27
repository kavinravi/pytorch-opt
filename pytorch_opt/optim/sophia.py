"""Sophia (Liu et al.): sign-magnitude-clipped diagonal second-order steps.

update: p -= lr * clip(m / max(gamma*h, eps), -1, 1), with m an EMA of grads
and h an EMA of a diagonal Hessian estimate refreshed every `estimate_freq`
steps: estimator="hutchinson" (Sophia-H, z*Hz) or "gnb" (Sophia-G,
Gauss-Newton-Bartlett: B * grad(CE(outputs, sampled labels))^2).

Closure protocol: closure returns loss with graph (hutchinson) or
(loss, outputs) (gnb; outputs are logits). No backward() inside.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from ._common import DiagnosticsMixin, StepTimer

_ESTIMATORS = ("hutchinson", "gnb")
_CLOSURE_MSG = ("Sophia.step() requires a closure returning the loss with its graph "
                "(estimator='hutchinson') or (loss, outputs) (estimator='gnb'); "
                "no backward() inside")


class Sophia(Optimizer, DiagnosticsMixin):
    requires_closure = True

    def __init__(self, params, lr: float = 1e-2, betas: tuple = (0.96, 0.99),
                 gamma: float = 0.05, eps: float = 1e-12, estimator: str = "hutchinson",
                 estimate_freq: int = 10, weight_decay: float = 0.0, seed: int = 0):
        if estimator not in _ESTIMATORS:
            raise ValueError(f"estimator must be one of {_ESTIMATORS}, got {estimator!r}")
        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps,
                        estimator=estimator, estimate_freq=estimate_freq,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.estimator = estimator
        # harness hint: gnb needs (loss, outputs) closures, like ggn modes
        self.curvature = "ggn" if estimator == "gnb" else "hessian"
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)
        self._steps = 0

    @classmethod
    def state_layout(cls) -> dict:
        return {"exp_avg": "shardable", "hess_diag": "shardable"}

    def state_dict(self):
        sd = super().state_dict()
        sd["sophia"] = {"gen_state": self._gen.get_state(), "steps": self._steps}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        extra = sd.pop("sophia", None)
        super().load_state_dict(sd)
        if extra is not None:
            self._gen.set_state(extra["gen_state"])
            self._steps = int(extra["steps"])

    def _hutchinson_diag(self, params, grads):
        zs = []
        for p in params:
            z = torch.randint(0, 2, p.shape, generator=self._gen, dtype=p.dtype)
            zs.append((2.0 * z - 1.0).to(p.device))
        dot = sum((g * z).sum() for g, z in zip(grads, zs))
        hzs = torch.autograd.grad(dot, params, retain_graph=False)
        return [(z * hz).detach() for z, hz in zip(zs, hzs)]

    def _gnb_diag(self, params, outputs):
        if outputs is None or outputs.ndim != 2:
            raise TypeError(_CLOSURE_MSG)
        B = outputs.shape[0]
        probs = torch.softmax(outputs.detach(), dim=-1)
        yhat = torch.multinomial(probs.cpu(), 1, generator=self._gen).squeeze(1).to(outputs.device)
        loss_s = F.cross_entropy(outputs, yhat)
        gs = torch.autograd.grad(loss_s, params, retain_graph=False, allow_unused=True)
        return [(B * g.pow(2)).detach() if g is not None else torch.zeros_like(p)
                for p, g in zip(params, gs)]

    def step(self, closure=None):
        if closure is None:
            raise TypeError(_CLOSURE_MSG)
        group0 = self.param_groups[0]
        refresh = self._steps % group0["estimate_freq"] == 0
        with StepTimer() as t_all:
            with torch.enable_grad():
                out = closure()
            if isinstance(out, tuple):
                loss, outputs = out
            else:
                loss, outputs = out, None
            params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
            need_graph = refresh and self.estimator == "hutchinson"
            retain = need_graph or (refresh and self.estimator == "gnb")
            grads = torch.autograd.grad(loss, params, create_graph=need_graph,
                                        retain_graph=retain)
            curvature_ms = 0.0
            if refresh:
                with StepTimer() as t_c:
                    if self.estimator == "hutchinson":
                        diags = self._hutchinson_diag(params, grads)
                    else:
                        diags = self._gnb_diag(params, outputs)
                curvature_ms = t_c.ms
                b2 = group0["betas"][1]
                for p, d in zip(params, diags):
                    st = self.state[p]
                    if "hess_diag" not in st:
                        st["hess_diag"] = d.clone()
                    else:
                        st["hess_diag"].mul_(b2).add_(d, alpha=1 - b2)
            clipped = 0
            total = 0
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    b1 = group["betas"][0]
                    for p in group["params"]:
                        if not p.requires_grad:
                            continue
                        g = grads[i].detach()
                        st = self.state[p]
                        if "exp_avg" not in st:
                            st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                        h = st.get("hess_diag", torch.zeros_like(p))
                        denom = (group["gamma"] * h).clamp_min(group["eps"])
                        ratio = st["exp_avg"] / denom
                        clipped += int((ratio.abs() > 1).sum())
                        total += ratio.numel()
                        upd = ratio.clamp(-1.0, 1.0)
                        if group["weight_decay"]:
                            p.mul_(1.0 - group["lr"] * group["weight_decay"])
                        p.add_(upd, alpha=-group["lr"])
                        i += 1
            self._steps += 1
        self._diag = {"clip_fraction": clipped / max(total, 1),
                      "estimate_refreshed": refresh,
                      "step_ms": t_all.ms, "curvature_ms": curvature_ms}
        return loss.detach()
