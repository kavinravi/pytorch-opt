"""K-FAC natural-gradient optimizer (Martens & Grosse).

API asymmetry (by design, see spec): KFAC takes the ``model``, not a parameter
iterable -- it must install hooks on tracked modules (Linear, Conv2d) and key
its curvature state by module path. Unselected parameters use AdamW. Pass
``params=matrix_param_groups(...)`` to share an explicit split with the other
optimizers. Without groups, supported untied module weights/biases use K-FAC.

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
from ._common import DiagnosticsMixin, StepTimer, adamw_update, validate_groups, validate_checkpoint_groups

_FISHER_MODES = ("empirical", "sampled")


class KFAC(Optimizer, DiagnosticsMixin):
    def __init__(self, model: nn.Module, lr: float = 0.01, damping: float = 1e-3,
                 ema_decay: float | None = 0.95, momentum: float = 0.9,
                 stats_every: int = 1, inv_every: int = 10,
                 fisher_mode: str = "empirical", weight_decay: float = 0.0,
                 max_grad_norm: float | None = None, *, params=None,
                 adamw_lr: float = 3e-4, adamw_betas: tuple = (0.9, 0.95),
                 adamw_eps: float = 1e-8, adamw_wd: float = 0.0):
        if fisher_mode not in _FISHER_MODES:
            raise ValueError(f"fisher_mode must be one of {_FISHER_MODES}, got {fisher_mode!r}")
        if stats_every < 1 or inv_every < 1:
            raise ValueError("stats_every and inv_every must be positive")
        self.model = model
        self.fisher_mode = fisher_mode
        owners = {}
        for module in model.modules():
            for p in module.parameters(recurse=False):
                owners[id(p)] = owners.get(id(p), 0) + 1
        supported = {name: m for name, m in model.named_modules()
                     if isinstance(m, (nn.Linear, nn.Conv2d))
                     and (not isinstance(m, nn.Conv2d) or m.groups == 1)
                     and m.weight.requires_grad and owners[id(m.weight)] == 1}
        if params is None:
            primary = [p for m in supported.values()
                       for p in m.parameters(recurse=False)
                       if p.requires_grad and owners[id(p)] == 1]
            selected_ids = {id(p) for p in primary}
            params = [{"params": primary, "use_preconditioner": True}]
            # Preserve standard exclusions for the automatic fallback as well.
            for decay in (False, True):
                other = [p for p in model.parameters() if p.requires_grad
                         and id(p) not in selected_ids
                         and (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)) == decay]
                if other:
                    params.append(dict(params=other, use_preconditioner=False,
                                       lr=adamw_lr, weight_decay=adamw_wd if decay else 0.0))
        else:
            params = list(params)
            if any(not isinstance(g, dict) or not isinstance(g.get("use_preconditioner"), bool)
                   for g in params):
                raise ValueError("KFAC params must be groups with explicit use_preconditioner")
        # Keep curvature settings in the first primary group. Per-group learning
        # rates, momentum and weight decay are applied below to each parameter.
        params = sorted(params, key=lambda g: not g["use_preconditioner"])
        defaults = dict(lr=lr, damping=damping, momentum=momentum,
                        stats_every=stats_every, inv_every=inv_every,
                        weight_decay=weight_decay, max_grad_norm=max_grad_norm,
                        adamw_lr=adamw_lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)
        super().__init__(params, defaults)
        validate_groups(self.param_groups, matrices=False)
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        assigned = {id(p) for g in self.param_groups for p in g["params"]}
        if assigned != trainable:
            raise ValueError("KFAC groups must cover every trainable model parameter exactly once")
        selected_ids = {id(p) for g in self.param_groups if g["use_preconditioner"] for p in g["params"]}
        self._name_to_module = {n: m for n, m in supported.items() if id(m.weight) in selected_ids}
        bias_modules = {n for n, m in self._name_to_module.items()
                        if m.bias is not None and id(m.bias) in selected_ids
                        and owners[id(m.bias)] == 1}
        covered = {id(m.weight) for m in self._name_to_module.values()}
        covered.update(id(self._name_to_module[n].bias) for n in bias_modules)
        if covered != selected_ids:
            raise ValueError("KFAC selection must contain supported, untied module weights and optional biases")
        self.tracker = KronTracker(model, ema_decay=ema_decay,
                                   module_names=self._name_to_module, bias_modules=bias_modules)
        self._steps = 0
        self._inv: dict[str, dict] = {}
        self._last_counts = {}
        self._inv_stale = 0
        self._sync_tracking()

    def _sync_tracking(self) -> None:
        collect = self._steps % self.param_groups[0]["stats_every"] == 0
        self.tracker.capture = collect
        self.tracker.enabled = self.fisher_mode == "empirical" and collect

    def set_grad_scale(self, scale: float) -> None:
        """Set before backward: 1 / accumulation_steps, or AMP scale / steps.

        This corrects curvature only. Unscale ordinary parameter gradients before
        calling step, as with any optimizer. Sampled Fisher ignores this scale.
        """
        self.tracker.set_grad_scale(scale)

    @classmethod
    def state_layout(cls) -> dict:
        return {"A": "replicable", "G": "replicable", "iA": "replicable",
                "iG": "replicable", "steps": "replicable", "inv_stale": "replicable",
                "momentum_buffer": "shardable", "step": "shardable",
                "exp_avg": "shardable", "exp_avg_sq": "shardable"}

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

    def _precondition(self, name: str, V: torch.Tensor) -> torch.Tensor:
        """Apply the (approximate) inverse Fisher to a module gradient matrix.
        Subclasses (EKFAC) override this."""
        inv = self._inv.get(name)
        if inv is None:
            raise RuntimeError(f"Missing K-FAC inverse for {name!r}")
        with torch.autocast(device_type=V.device.type, enabled=False):
            return (inv["iG"] @ V.to(inv["iG"].dtype) @ inv["iA"]).to(V.dtype)

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
        # Validate every selected layer before changing any parameters. A fused
        # or functional call can produce weight gradients without firing hooks.
        for name, module in self._name_to_module.items():
            if module.weight.grad is None and not (
                self.tracker.includes_bias(module) and module.bias.grad is not None
            ):
                continue
            fresh = self.tracker._counts.get(name, 0) > self._last_counts.get(name, 0)
            if name not in self.tracker.factors or (
                self._steps % g0["stats_every"] == 0 and not fresh
            ):
                raise RuntimeError(
                    f"No K-FAC curvature collected for {name!r}. Check update_curvature "
                    "and module hooks; fused/functional projections bypass hooks. "
                    "For Mamba-2, use_mem_eff_path=False exposes out_proj."
                )
        groups = {id(p): group for group in self.param_groups for p in group["params"]}
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
                w = m.weight
                has_bias = self.tracker.includes_bias(m)
                if w.grad is None and not (has_bias and m.bias.grad is not None):
                    continue
                gW = (w.grad if w.grad is not None else torch.zeros_like(w)).reshape(w.shape[0], -1)
                if has_bias:
                    gb = m.bias.grad if m.bias.grad is not None else torch.zeros_like(m.bias)
                    V = torch.cat([gW, gb.unsqueeze(1)], dim=1)
                else:
                    V = gW
                V = self._precondition(name, V)
                if w.grad is not None:
                    nat.append((w, (V[:, :-1] if has_bias else V).reshape(w.shape)))
                if has_bias and m.bias.grad is not None:
                    nat.append((m.bias, V[:, -1]))

            nat_norm = torch.sqrt(sum(d.pow(2).sum() for _, d in nat)) if nat else torch.tensor(0.0)
            scale = 1.0
            if g0["max_grad_norm"] is not None and float(nat_norm) > g0["max_grad_norm"]:
                scale = g0["max_grad_norm"] / (float(nat_norm) + 1e-12)

            for p, d in nat:
                group = groups[id(p)]
                st = self.state[p]
                if group["momentum"]:
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p)
                    st["momentum_buffer"].mul_(group["momentum"]).add_(d, alpha=scale)
                    upd = st["momentum_buffer"]
                else:
                    upd = d * scale
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(upd, alpha=-group["lr"])

            n_adamw = 0
            for group in self.param_groups:
                if group["use_preconditioner"]:
                    continue
                for p in group["params"]:
                    if p.grad is not None:
                        adamw_update(p, p.grad, self.state[p], group)
                        n_adamw += 1
            self._last_counts = dict(self.tracker._counts)
            self._steps += 1
            self._sync_tracking()
        self._diag = {
            "n_adamw_params": n_adamw,
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
            "last_counts": dict(self._last_counts),
            "modules": list(self._name_to_module),
            "bias_modules": sorted(self.tracker.bias_modules),
            "fisher_mode": self.fisher_mode,
            "ema_decay": self.tracker.ema_decay,
            "grad_scale": self.tracker.grad_scale,
        }
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        extra = sd.pop("kfac", None)
        if extra is None:
            raise ValueError("Checkpoint is missing K-FAC curvature state")
        if "modules" in extra:
            if (extra["modules"] != list(self._name_to_module)
                    or extra["bias_modules"] != sorted(self.tracker.bias_modules)):
                raise ValueError("Checkpoint K-FAC module selection differs from this optimizer")
        validate_checkpoint_groups(self.param_groups, sd["param_groups"])
        super().load_state_dict(sd)
        self._steps = int(extra["steps"])
        self._inv_stale = int(extra["inv_stale"])
        self.fisher_mode = extra.get("fisher_mode", self.fisher_mode)
        self.tracker.ema_decay = extra.get("ema_decay", self.tracker.ema_decay)
        self.tracker.set_grad_scale(extra.get("grad_scale", 1.0))
        self.tracker.factors = self._restore_curvature(extra["factors"])
        self._inv = self._restore_curvature(extra["inv"])
        self.tracker._counts = dict(extra["counts"])
        self._last_counts = dict(extra.get("last_counts", extra["counts"]))
        self._sync_tracking()

    def _restore_curvature(self, values):
        restored = {}
        for name, tensors in values.items():
            weight = self._name_to_module[name].weight
            dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
            restored[name] = {k: v.to(device=weight.device, dtype=dtype).clone()
                              if torch.is_tensor(v) else v for k, v in tensors.items()}
        return restored
