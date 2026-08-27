"""K-FAC natural-gradient optimizer (Martens & Grosse).

API asymmetry (by design, see spec): KFAC takes the ``model``, not a parameter
iterable -- it must install hooks on tracked modules (Linear, Conv2d) and key
its curvature state by module path. Parameters of untracked modules get plain
SGD-with-momentum at ``sgd_lr`` (default: ``lr``).

Curvature modes: ``fisher_mode="empirical"`` accumulates factors from the real
training backward on a ``stats_every`` cadence; ``"sampled"`` accumulates only
inside ``update_curvature(outputs)``, which draws labels from the model's
predictive distribution (true Fisher). The natural-gradient update for a
module with weight gradient V (out x in, bias column appended when present) is
``(G + gamma_G I)^-1 V (A + gamma_A I)^-1`` with pi-split factored damping.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from ..curvature.damping import kfac_factored_damping
from ..curvature.kron import KronTracker
from ._common import DiagnosticsMixin, StepTimer

_FISHER_MODES = ("empirical", "sampled")


class KFAC(Optimizer, DiagnosticsMixin):
    def __init__(self, model: nn.Module, lr: float = 0.01, damping: float = 1e-3,
                 ema_decay: float | None = 0.95, momentum: float = 0.9,
                 stats_every: int = 1, inv_every: int = 10,
                 fisher_mode: str = "empirical", weight_decay: float = 0.0,
                 max_grad_norm: float | None = None, sgd_lr: float | None = None):
        if fisher_mode not in _FISHER_MODES:
            raise ValueError(f"fisher_mode must be one of {_FISHER_MODES}, got {fisher_mode!r}")
        self.model = model
        self.fisher_mode = fisher_mode
        self.tracker = KronTracker(model, ema_decay=ema_decay)
        self._name_to_module = {name: m for m, name in self.tracker.tracked.items()}
        kfac_params, kfac_ids = [], set()
        for m in self.tracker.tracked:
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    kfac_params.append(p)
                    kfac_ids.add(id(p))
        other = [p for p in model.parameters() if p.requires_grad and id(p) not in kfac_ids]
        groups = [{"params": kfac_params, "kfac": True}]
        if other:
            groups.append({"params": other, "kfac": False})
        defaults = dict(lr=lr, damping=damping, momentum=momentum,
                        stats_every=stats_every, inv_every=inv_every,
                        weight_decay=weight_decay, max_grad_norm=max_grad_norm,
                        sgd_lr=sgd_lr if sgd_lr is not None else lr)
        super().__init__(groups, defaults)
        self._steps = 0
        self._inv: dict[str, dict] = {}
        self._inv_stale = 0
        self._sync_tracking()

    def _sync_tracking(self) -> None:
        g = self.param_groups[0]
        self.tracker.enabled = (self.fisher_mode == "empirical"
                                and self._steps % g["stats_every"] == 0)

    @classmethod
    def state_layout(cls) -> dict:
        return {"A": "replicable", "G": "replicable", "iA": "replicable",
                "iG": "replicable", "steps": "replicable", "inv_stale": "replicable",
                "momentum_buffer": "shardable"}

    # ------------------------------------------------------------- curvature

    def update_curvature(self, outputs: torch.Tensor, kind: str = "categorical",
                         generator: torch.Generator | None = None) -> None:
        if self.fisher_mode != "sampled":
            raise RuntimeError("update_curvature() is only used with fisher_mode='sampled'; "
                               "empirical mode accumulates from the training backward")
        if self._steps % self.param_groups[0]["stats_every"] != 0:
            return
        self.tracker.sampled_backward(outputs, kind=kind, generator=generator)

    @staticmethod
    def _damped_inverse(F: torch.Tensor, gamma: float):
        ev, V = torch.linalg.eigh(F.double())
        inv = (V @ torch.diag(1.0 / (ev + gamma)) @ V.T).to(F.dtype)
        cond = float((ev.max() + gamma) / max(float(ev.min()) + gamma, 1e-300))
        return inv, cond

    def _refresh_inverses(self) -> list[tuple[float, float]]:
        g = self.param_groups[0]
        conds = []
        for name, f in self.tracker.factors.items():
            gamma_A, gamma_G = kfac_factored_damping(f["A"], f["G"], g["damping"])
            iA, cA = self._damped_inverse(f["A"], gamma_A)
            iG, cG = self._damped_inverse(f["G"], gamma_G)
            self._inv[name] = {"iA": iA, "iG": iG, "cond_A": cA, "cond_G": cG}
            conds.append((cA, cG))
        return conds

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        g0 = self.param_groups[0]
        curvature_ms = 0.0
        conds = []
        with StepTimer() as t_all:
            if self.tracker.factors and self._steps % g0["inv_every"] == 0:
                with StepTimer() as t_c:
                    conds = self._refresh_inverses()
                curvature_ms = t_c.ms
                self._inv_stale = 0
            elif self._inv:
                self._inv_stale += 1

            # ---- natural gradients for tracked modules
            nat: list[tuple[torch.Tensor, torch.Tensor]] = []  # (param, nat_grad)
            for name, m in self._name_to_module.items():
                inv = self._inv.get(name)
                w = m.weight
                if w.grad is None:
                    continue
                gW = w.grad.reshape(w.shape[0], -1)
                has_bias = m.bias is not None and m.bias.grad is not None
                V = torch.cat([gW, m.bias.grad.unsqueeze(1)], dim=1) if has_bias else gW
                if inv is not None:
                    V = inv["iG"] @ V @ inv["iA"]
                if has_bias:
                    nat.append((w, V[:, :-1].reshape(w.shape)))
                    nat.append((m.bias, V[:, -1]))
                else:
                    nat.append((w, V.reshape(w.shape)))

            nat_norm = torch.sqrt(sum(d.pow(2).sum() for _, d in nat)) if nat else torch.tensor(0.0)
            scale = 1.0
            if g0["max_grad_norm"] is not None and float(nat_norm) > g0["max_grad_norm"]:
                scale = g0["max_grad_norm"] / (float(nat_norm) + 1e-12)

            for p, d in nat:
                st = self.state[p]
                if g0["momentum"]:
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p)
                    st["momentum_buffer"].mul_(g0["momentum"]).add_(d, alpha=scale)
                    upd = st["momentum_buffer"]
                else:
                    upd = d * scale
                if g0["weight_decay"]:
                    p.mul_(1.0 - g0["lr"] * g0["weight_decay"])
                p.add_(upd, alpha=-g0["lr"])

            # ---- untracked params: SGD with momentum
            for group in self.param_groups[1:]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if group["momentum"]:
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(p)
                        st["momentum_buffer"].mul_(group["momentum"]).add_(p.grad)
                        upd = st["momentum_buffer"]
                    else:
                        upd = p.grad
                    if group["weight_decay"]:
                        p.mul_(1.0 - group["sgd_lr"] * group["weight_decay"])
                    p.add_(upd, alpha=-group["sgd_lr"])

            self._steps += 1
            self._sync_tracking()
        self._diag = {
            "damping": g0["damping"],
            "mean_cond_A": sum(c[0] for c in conds) / len(conds) if conds else None,
            "mean_cond_G": sum(c[1] for c in conds) / len(conds) if conds else None,
            "inv_stale_steps": self._inv_stale,
            "nat_grad_norm": float(nat_norm),
            "step_ms": t_all.ms,
            "curvature_ms": curvature_ms,
        }
        return loss

    # ------------------------------------------------------------ state dict

    def state_dict(self):
        sd = super().state_dict()
        sd["kfac"] = {
            "steps": self._steps,
            "inv_stale": self._inv_stale,
            "factors": {n: {k: v for k, v in f.items()} for n, f in self.tracker.factors.items()},
            "inv": {n: {k: v for k, v in d.items()} for n, d in self._inv.items()},
            "counts": dict(self.tracker._counts),
        }
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        extra = sd.pop("kfac", None)
        super().load_state_dict(sd)
        if extra is not None:
            self._steps = int(extra["steps"])
            self._inv_stale = int(extra["inv_stale"])
            self.tracker.factors = {
                n: {k: v.clone() for k, v in f.items()} for n, f in extra["factors"].items()
            }
            self._inv = {n: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}
                         for n, d in extra["inv"].items()}
            self.tracker._counts = dict(extra["counts"])
            self._sync_tracking()
