"""Tiny models, seeded data, and a protocol-aware training loop for tests."""

from __future__ import annotations

import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self, d_in: int = 8, h: int = 16, d_out: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.Tanh(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    """1x8x8 input -> logits over 3 classes."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)              # -> 4x6x6
        self.conv2 = nn.Conv2d(4, 4, 3, stride=2)    # -> 4x2x2
        self.head = nn.Linear(16, k)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.head(x.flatten(1))


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_regression(n: int = 64, d: int = 8, *, seed: int = 0, device="cpu"):
    g = _gen(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g) / d**0.5
    y = X @ w + 0.05 * torch.randn(n, 1, generator=g)
    return X.to(device), y.to(device)


def make_classification(n: int = 64, d: int = 8, k: int = 3, *, seed: int = 0, device="cpu"):
    g = _gen(seed)
    X = torch.randn(n, d, generator=g)
    W = torch.randn(d, k, generator=g)
    y = (X @ W + 0.5 * torch.randn(n, k, generator=g)).argmax(dim=1)
    return X.to(device), y.to(device)


def mse_half(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The project-wide MSE convention: 0.5 * sum of squares / batch."""
    return 0.5 * (pred - target).pow(2).sum() / pred.shape[0]


def run_steps(model, opt, X, y, loss_fn, n: int) -> list[float]:
    """Drive any pytorch-opt optimizer for n full-batch steps; returns losses.

    Handles the three step protocols: closure-based (TrustNCG et al., which
    set ``requires_closure = True``), sampled-Fisher (KFAC-style with
    ``fisher_mode == "sampled"``), and plain backward+step.
    """
    losses: list[float] = []
    for _ in range(n):
        if getattr(opt, "requires_closure", False):
            def closure():
                out = model(X)
                loss = loss_fn(out, y)
                if getattr(opt, "curvature", "hessian") == "ggn":
                    return loss, out
                return loss

            loss = opt.step(closure)
            losses.append(float(loss))
        else:
            opt.zero_grad(set_to_none=True)
            out = model(X)
            loss = loss_fn(out, y)
            if getattr(opt, "fisher_mode", None) == "sampled":
                opt.update_curvature(out)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return losses
