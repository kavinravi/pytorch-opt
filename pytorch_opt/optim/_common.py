"""Shared plumbing for pytorch-opt optimizers: diagnostics + step timing."""

from __future__ import annotations

import time

import torch


def matrix_path(p, group):
    """Resolve explicit routing. A selected matrix never silently falls back."""
    selected = group.get("use_preconditioner")
    if selected is not None and not isinstance(selected, bool):
        raise ValueError("use_preconditioner must be True, False, or None")
    selected = p.ndim >= 2 if selected is None else selected
    if selected and p.ndim < 2:
        raise ValueError("Only matrix parameters can use the preconditioner")
    if selected and p.numel() == 0:
        raise ValueError("Selected matrices must be nonempty")
    if selected and max(p.shape[0], p.numel() // p.shape[0]) > group.get("max_preconditioner_dim", float("inf")):
        raise ValueError(
            f"Selected matrix {tuple(p.shape)} exceeds max_preconditioner_dim; "
            "increase the limit or explicitly assign use_preconditioner=False"
        )
    return selected


def validate_groups(groups, *, matrices=True):
    seen = set()
    for group in groups:
        b1, b2 = group["adamw_betas"]
        if not (0 <= b1 < 1 and 0 <= b2 < 1):
            raise ValueError("adamw_betas must be in [0, 1)")
        if min(group["lr"], group["weight_decay"], group["adamw_lr"],
               group["adamw_wd"], group["adamw_eps"]) < 0:
            raise ValueError("Learning rates, weight decay, and epsilon must be nonnegative")
        for p in group["params"]:
            if id(p) in seen:
                raise ValueError("A parameter may appear in only one optimizer group")
            seen.add(id(p))
            if matrices:
                matrix_path(p, group)


def validate_checkpoint_groups(current, saved):
    """Prevent resuming moments under a different parameter assignment."""
    for group, old in zip(current, saved):
        if "use_preconditioner" not in old:
            raise ValueError("Checkpoint predates explicit AdamW routing; start fresh optimizer state")
        if group.get("use_preconditioner") != old["use_preconditioner"]:
            raise ValueError("Checkpoint optimizer routing differs from current parameter groups")
        if group.get("param_names") != old.get("param_names"):
            raise ValueError("Checkpoint parameter names/order differ from current groups")


def adamw_update(p, grad, state, group):
    """Shared AdamW fallback, using Muon's existing update and state format.

    Explicit fallback groups use ordinary lr/weight_decay so PyTorch schedulers
    update both arms. Automatic mixed groups retain the adamw_* options.
    """
    if grad.is_sparse:
        raise RuntimeError("AdamW fallback requires dense gradients")
    explicit = group.get("use_preconditioner") is False
    lr = group["lr"] if explicit else group["adamw_lr"]
    wd = group["weight_decay"] if explicit else group["adamw_wd"]
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] = 0
    state["step"] += 1
    b1, b2 = group["adamw_betas"]
    state["exp_avg"].lerp_(grad, 1 - b1)
    state["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
    bc1, bc2 = 1 - b1 ** state["step"], 1 - b2 ** state["step"]
    denom = (state["exp_avg_sq"].sqrt() / bc2 ** 0.5).add_(group["adamw_eps"])
    update = (state["exp_avg"] / bc1) / denom
    p.mul_(1 - lr * wd).add_(update, alpha=-lr)
    return update, lr


class StepTimer:
    """Context manager measuring wall-clock milliseconds."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._final = None
        return self

    def __exit__(self, *exc):
        self._final = (time.perf_counter() - self.t0) * 1e3
        return False

    @property
    def ms(self) -> float:
        if self._final is not None:
            return self._final
        return (time.perf_counter() - self.t0) * 1e3


class DiagnosticsMixin:
    """Optimizers populate self._diag each step; read via .diagnostics."""

    @property
    def diagnostics(self) -> dict:
        return dict(getattr(self, "_diag", {}))
