"""EKFAC (George et al.): K-FAC's Kronecker eigenbasis with an optimally
rescaled diagonal.

Instead of inverting the damped factors, keep their eigenbases Q_A, Q_G
(refreshed every `inv_every` steps) and divide the rotated gradient by
per-component second moments s (EMA of the rotated minibatch gradient squares,
updated every step -- the paper's amortized estimate of the optimal diagonal):

    update = Q_G [ (Q_G^T V Q_A) / (s + damping) ] Q_A^T

With s frozen to the outer product of the factor eigenvalues and vanishing
damping, EKFAC reduces exactly to K-FAC (that identity is the analytic test).
"""

from __future__ import annotations

import torch

from .kfac import KFAC


class EKFAC(KFAC):
    def __init__(self, model, *args, scale_decay: float = 0.95, **kwargs):
        self._scales: dict[str, torch.Tensor] = {}
        self._scale_decay = scale_decay
        super().__init__(model, *args, **kwargs)

    def _refresh_inverses(self):
        g = self.param_groups[0]
        conds = []
        for name, f in self.tracker.factors.items():
            evA, QA = torch.linalg.eigh(f["A"].double())
            evG, QG = torch.linalg.eigh(f["G"].double())
            dt = f["A"].dtype
            lam = g["damping"]
            cA = float((evA.max() + lam) / max(float(evA.min()) + lam, 1e-300))
            cG = float((evG.max() + lam) / max(float(evG.min()) + lam, 1e-300))
            self._inv[name] = {"QA": QA.to(dt), "QG": QG.to(dt),
                               "evA": evA.to(dt), "evG": evG.to(dt),
                               "cond_A": cA, "cond_G": cG}
            conds.append((cA, cG))
        return conds

    def _precondition(self, name: str, V: torch.Tensor) -> torch.Tensor:
        inv = self._inv.get(name)
        if inv is None:
            return V
        lam = self.param_groups[0]["damping"]
        Vt = inv["QG"].T @ V @ inv["QA"]
        s = self._scales.get(name)
        if s is None:
            s = Vt.pow(2).detach().clone()
            self._scales[name] = s
        else:
            s.mul_(self._scale_decay).add_(Vt.pow(2), alpha=1 - self._scale_decay)
        return inv["QG"] @ (Vt / (s + lam)) @ inv["QA"].T

    def state_dict(self):
        sd = super().state_dict()
        sd["ekfac"] = {"scales": {n: v for n, v in self._scales.items()}}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        extra = sd.pop("ekfac", None)
        super().load_state_dict(sd)
        if extra is not None:
            self._scales = {n: v.clone() for n, v in extra["scales"].items()}
