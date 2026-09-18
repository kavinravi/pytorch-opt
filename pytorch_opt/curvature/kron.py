"""Kronecker-factored curvature statistics (K-FAC style) via module hooks.

Assumes mean-over-batch loss reduction (project convention). For nn.Linear the
factors are A = E[a_aug a_aug^T] over rows and G = E[ghat ghat^T] with
ghat = B * dL/dout (per-example output gradients). For nn.Conv2d, activations
are unfolded into patches and spatial locations are folded into the batch (KFC
convention); the 1x1-spatial-output case reduces exactly to the Linear case.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F
from torch import nn


class KronTracker:
    def __init__(self, model: nn.Module, ema_decay: float | None = 0.95,
                 modules: tuple = (nn.Linear, nn.Conv2d), *,
                 module_names=None, bias_modules=None):
        self.model = model
        self.ema_decay = ema_decay  # None -> running mean
        self.enabled = False
        self.capture = True
        self.grad_scale = 1.0
        self.bias_modules = None if bias_modules is None else set(bias_modules)
        self.factors: dict[str, dict[str, torch.Tensor]] = {}
        self._module_names: dict[nn.Module, str] = {}
        self._counts: dict[str, int] = {}
        self._handles = []
        selected = None if module_names is None else set(module_names)
        for name, mod in model.named_modules():
            if isinstance(mod, modules) and (selected is None or name in selected):
                if isinstance(mod, nn.Conv2d) and mod.groups != 1:
                    raise ValueError("KronTracker does not support grouped Conv2d")
                self._module_names[mod] = name
                self._handles.append(mod.register_forward_hook(self._fwd_hook))

    @property
    def tracked(self) -> dict[nn.Module, str]:
        return dict(self._module_names)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def set_grad_scale(self, scale: float) -> None:
        """Scale applied to the mean loss, e.g. GradScaler scale / accumulation steps."""
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Gradient scale must be positive and finite")
        self.grad_scale = float(scale)

    def includes_bias(self, module):
        return (module.bias is not None and module.bias.requires_grad
                and (self.bias_modules is None
                     or self._module_names[module] in self.bias_modules))

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
        if not (self.capture and module.training and output.requires_grad):
            return
        # Bind each invocation's input to its output. Reused layers and multiple
        # forwards must not read the last invocation's activation. The graph owns
        # this reference, so completed backwards release the saved activation.
        a = inputs[0].detach()
        output.register_hook(lambda grad, m=module, a=a: self._on_grad_out(m, a, grad))

    def _on_grad_out(self, module, a, grad_output):
        if not (self.enabled and module.training):
            return
        name = self._module_names[module]
        # Hooks may run inside autocast. Keep both factors in at least FP32.
        dtype = torch.float64 if module.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=a.device.type, enabled=False):
            A_b, G_b = self._batch_factors(
                module, a.to(dtype), grad_output.detach().to(dtype) / self.grad_scale)
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
            if self.includes_bias(module):
                a2 = torch.cat([a2, a2.new_ones(n, 1)], dim=1)
            return a2.T @ a2 / n, ghat.T @ ghat / n
        # Conv2d
        B = a.shape[0]
        patches = F.unfold(a, module.kernel_size, dilation=module.dilation,
                           padding=module.padding, stride=module.stride)  # (B, C*k*k, L)
        L = patches.shape[-1]
        at = patches.permute(0, 2, 1).reshape(B * L, -1)
        if self.includes_bias(module):
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
        params = list({id(p): p for m in self._module_names
                       for p in m.parameters(recurse=False) if p.requires_grad}.values())
        if not params:
            return
        if kind == "categorical":
            logits = outputs.reshape(-1, outputs.shape[-1])
            logits = logits.double() if logits.dtype == torch.float64 else logits.float()
            probs = torch.softmax(logits.detach(), dim=-1)
            if generator is not None:
                t = torch.multinomial(probs.to(generator.device), 1, generator=generator).squeeze(1).to(outputs.device)
            else:
                t = torch.multinomial(probs, 1).squeeze(1)
            loss = F.cross_entropy(logits, t)
        elif kind == "gaussian":
            if generator is not None:
                noise = torch.randn(outputs.shape, generator=generator, device=generator.device,
                                    dtype=outputs.dtype).to(outputs.device)
            else:
                noise = torch.randn_like(outputs)
            t = outputs.detach() + noise
            loss = 0.5 * (outputs - t).pow(2).sum() / (outputs.numel() // outputs.shape[-1])
        else:
            raise ValueError(f"kind must be 'categorical' or 'gaussian', got {kind!r}")
        # This auxiliary loss is neither divided for accumulation nor AMP-scaled.
        previous_scale = self.grad_scale
        self.grad_scale = 1.0
        try:
            with self.track():
                torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
        finally:
            self.grad_scale = previous_scale
