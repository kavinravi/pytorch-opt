"""Spectrum diagnostics."""

from __future__ import annotations

import torch

from .curvature.lanczos import lanczos_eigs
from .curvature.products import make_hvp


def hessian_eigs(closure, params, k: int = 6, iters: int | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Top-k Hessian eigenvalue estimates at the current parameters.

    `closure` recomputes the loss with its graph (no backward), like TrustNCG's.
    """
    params = list(params)
    with torch.enable_grad():
        loss = closure()
    hvp_fn, g = make_hvp(loss, params)
    return lanczos_eigs(hvp_fn, g.numel(), k=k, iters=iters, device=g.device,
                        dtype=g.dtype, generator=generator)
