"""AdaHessian (Yao et al.): Adam-style steps with a Hutchinson estimate of the
Hessian diagonal as the second moment.

Closure protocol (like TrustNCG): ``step(closure)`` with a closure returning
the loss WITH its graph, no backward() -- the Hessian diagonal needs a
double backward. Rademacher probes come from an internal seeded generator so
trajectories are deterministic and survive state_dict round-trips.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from ._common import DiagnosticsMixin, StepTimer

_CLOSURE_MSG = ("AdaHessian.step() requires a closure returning the loss with its "
                "autograd graph (no backward() inside)")


class AdaHessian(Optimizer, DiagnosticsMixin):
    requires_closure = True

    def __init__(self, params, lr: float = 0.1, betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 hessian_power: float = 1.0, update_freq: int = 1,
                 n_samples: int = 1, seed: int = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        hessian_power=hessian_power, update_freq=update_freq,
                        n_samples=n_samples)
        super().__init__(params, defaults)
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)
        self._steps = 0

    @classmethod
    def state_layout(cls) -> dict:
        return {"exp_avg": "shardable", "exp_avg_sq": "shardable",
                "hess_diag": "shardable", "t": "shardable"}

    def state_dict(self):
        sd = super().state_dict()
        sd["adahessian"] = {"gen_state": self._gen.get_state(), "steps": self._steps}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        extra = sd.pop("adahessian", None)
        super().load_state_dict(sd)
        if extra is not None:
            self._gen.set_state(extra["gen_state"])
            self._steps = int(extra["steps"])

    def _rademacher_like(self, p: torch.Tensor) -> torch.Tensor:
        z = torch.randint(0, 2, p.shape, generator=self._gen, dtype=p.dtype)
        return (2.0 * z - 1.0).to(p.device)

    def _hutchinson(self, params, grads, n_samples: int):
        D = [torch.zeros_like(p) for p in params]
        for i in range(n_samples):
            zs = [self._rademacher_like(p) for p in params]
            dot = sum((g * z).sum() for g, z in zip(grads, zs))
            hzs = torch.autograd.grad(dot, params, retain_graph=(i < n_samples - 1))
            for d, z, hz in zip(D, zs, hzs):
                d.add_(z * hz, alpha=1.0 / n_samples)
        # spatial averaging for conv kernels (paper's variance reduction)
        return [d.abs().mean(dim=[2, 3], keepdim=True) if d.ndim == 4 else d.abs()
                for d in D]

    def step(self, closure=None):
        if closure is None:
            raise TypeError(_CLOSURE_MSG)
        with StepTimer() as t_all:
            group0 = self.param_groups[0]
            refresh = self._steps % group0["update_freq"] == 0
            with torch.enable_grad():
                loss = closure()
            params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=refresh)
            curvature_ms = 0.0
            if refresh:
                with StepTimer() as t_c:
                    diags = self._hutchinson(params, grads, group0["n_samples"])
                curvature_ms = t_c.ms
                for p, d in zip(params, diags):
                    self.state[p]["hess_diag"] = d.detach()
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    b1, b2 = group["betas"]
                    k = group["hessian_power"]
                    for p in group["params"]:
                        if not p.requires_grad:
                            continue
                        g = grads[i].detach()
                        st = self.state[p]
                        if "exp_avg" not in st:
                            st["exp_avg"] = torch.zeros_like(p)
                            st["exp_avg_sq"] = torch.zeros_like(p)
                            st["t"] = 0
                        st["t"] += 1
                        st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                        d = st["hess_diag"]
                        st["exp_avg_sq"].mul_(b2).addcmul_(d, d, value=1 - b2)
                        bc1 = 1 - b1 ** st["t"]
                        bc2 = 1 - b2 ** st["t"]
                        denom = (st["exp_avg_sq"] / bc2).pow(k / 2).add_(group["eps"])
                        if group["weight_decay"]:
                            p.mul_(1.0 - group["lr"] * group["weight_decay"])
                        p.addcdiv_(st["exp_avg"] / bc1, denom.expand_as(p), value=-group["lr"])
                        i += 1
            self._steps += 1
        self._diag = {"hutchinson_refreshed": refresh, "step_ms": t_all.ms,
                      "curvature_ms": curvature_ms}
        return loss.detach()
