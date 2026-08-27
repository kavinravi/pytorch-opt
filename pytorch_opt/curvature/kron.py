"""Kronecker-factored curvature statistics (K-FAC style) via module hooks.

Assumes mean-over-batch loss reduction (project convention). For nn.Linear the
factors are A = E[a_aug a_aug^T] over rows and G = E[ghat ghat^T] with
ghat = B * dL/dout (per-example output gradients). For nn.Conv2d, activations
are unfolded into patches and spatial locations are folded into the batch (KFC
convention); the 1x1-spatial-output case reduces exactly to the Linear case.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


class KronTracker:
    def __init__(self, model: nn.Module, ema_decay: float | None = 0.95,
                 modules: tuple = (nn.Linear, nn.Conv2d)):
        self.model = model
        self.ema_decay = ema_decay  # None -> running mean
        self.enabled = False
        self.factors: dict[str, dict[str, torch.Tensor]] = {}
        self._module_names: dict[nn.Module, str] = {}
        self._inputs: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, modules):
                self._module_names[mod] = name
                self._handles.append(mod.register_forward_hook(self._fwd_hook))

    @property
    def tracked(self) -> dict[nn.Module, str]:
        return dict(self._module_names)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextmanager
    def track(self):
        prev = self.enabled
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = prev

    # ------------------------------------------------------------------ hooks

    def _fwd_hook(self, module, inputs, output):
        # Capture unconditionally while training: sampled_backward() may run
        # against a forward that happened before tracking was enabled, and may
        # reuse the same forward repeatedly. A detached reference is cheap.
        # The gradient hook rides on the output tensor (fires on every backward
        # through this forward's graph; gated on self.enabled at fire time).
        # Skip eval-mode and no-grad forwards (e.g. validation passes):
        # there is no backward coming, and grad hooks cannot attach.
        if not (module.training and output.requires_grad):
            return
        self._inputs[self._module_names[module]] = inputs[0].detach()
        output.register_hook(lambda grad, m=module: self._on_grad_out(m, grad))

    def _on_grad_out(self, module, grad_output):
        if not (self.enabled and module.training):
            return
        name = self._module_names[module]
        a = self._inputs.get(name)
        if a is None:
            return
        A_b, G_b = self._batch_factors(module, a, grad_output.detach())
        self._update(name, "A", A_b)
        self._update(name, "G", G_b)
        self._counts[name] = self._counts.get(name, 0) + 1

    def _update(self, name: str, key: str, val: torch.Tensor) -> None:
        d = self.factors.setdefault(name, {})
        if key not in d:
            d[key] = val.clone()
        elif self.ema_decay is None:
            cnt = self._counts.get(name, 0)  # completed update pairs so far
            d[key].mul_(cnt / (cnt + 1)).add_(val, alpha=1.0 / (cnt + 1))
        else:
            d[key].mul_(self.ema_decay).add_(val, alpha=1.0 - self.ema_decay)

    def _batch_factors(self, module, a, g):
        if isinstance(module, nn.Linear):
            a2 = a.reshape(-1, a.shape[-1])
            g2 = g.reshape(-1, g.shape[-1])
            n = a2.shape[0]
            ghat = g2 * n
            if module.bias is not None:
                a2 = torch.cat([a2, a2.new_ones(n, 1)], dim=1)
            return a2.T @ a2 / n, ghat.T @ ghat / n
        # Conv2d
        B = a.shape[0]
        patches = F.unfold(a, module.kernel_size, dilation=module.dilation,
                           padding=module.padding, stride=module.stride)  # (B, C*k*k, L)
        L = patches.shape[-1]
        at = patches.permute(0, 2, 1).reshape(B * L, -1)
        if module.bias is not None:
            at = torch.cat([at, at.new_ones(B * L, 1)], dim=1)
        gt = g.reshape(B, g.shape[1], -1).permute(0, 2, 1).reshape(B * L, -1)
        ghat = gt * B
        return at.T @ at / (B * L), ghat.T @ ghat / (B * L)

    # -------------------------------------------------------- sampled Fisher

    def sampled_backward(self, outputs: torch.Tensor, kind: str = "categorical",
                         generator: torch.Generator | None = None,
                         retain_graph: bool = True) -> None:
        """Extra backward with labels sampled from the model's predictive
        distribution (true-Fisher statistics). Never touches .grad."""
        params = [p for m in self._module_names for p in m.parameters() if p.requires_grad]
        if kind == "categorical":
            probs = torch.softmax(outputs.detach(), dim=-1)
            if generator is not None:
                t = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1).to(outputs.device)
            else:
                t = torch.multinomial(probs, 1).squeeze(1)
            loss = F.cross_entropy(outputs, t)
        elif kind == "gaussian":
            if generator is not None:
                noise = torch.randn(outputs.shape, generator=generator,
                                    dtype=outputs.dtype).to(outputs.device)
            else:
                noise = torch.randn_like(outputs)
            t = outputs.detach() + noise
            loss = 0.5 * (outputs - t).pow(2).sum() / outputs.shape[0]
        else:
            raise ValueError(f"kind must be 'categorical' or 'gaussian', got {kind!r}")
        with self.track():
            torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
