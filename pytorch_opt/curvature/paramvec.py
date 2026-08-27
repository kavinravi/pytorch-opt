"""Flatten/unflatten a parameter list to a single working vector."""

from __future__ import annotations

import torch


class ParamVector:
    """View a list of tensors as one flat vector for vector-space algorithms (CG, etc.).

    The flat vector lives on the first parameter's device; chunks are cast back
    to each parameter's own device on scatter, so mixed-device parameter lists
    are supported. Mixed dtypes are not.
    """

    def __init__(self, params):
        self.params = list(params)
        if not self.params:
            raise ValueError("ParamVector needs at least one parameter")
        dtype = self.params[0].dtype
        for p in self.params:
            if p.dtype != dtype:
                raise TypeError(
                    f"ParamVector requires a single dtype, got {p.dtype} and {dtype}"
                )
        self.dtype = dtype
        self.device = self.params[0].device
        self._shapes = [p.shape for p in self.params]
        self._numels = [p.numel() for p in self.params]
        self.numel = sum(self._numels)

    def _split(self, flat: torch.Tensor):
        if flat.shape != (self.numel,):
            raise ValueError(f"expected flat vector of shape ({self.numel},), got {tuple(flat.shape)}")
        return torch.split(flat, self._numels)

    def gather(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1).to(self.device) for p in self.params])

    def gather_grads(self, fill_none: float = 0.0) -> torch.Tensor:
        chunks = []
        for p in self.params:
            if p.grad is None:
                chunks.append(torch.full((p.numel(),), fill_none, dtype=self.dtype, device=self.device))
            else:
                chunks.append(p.grad.detach().reshape(-1).to(self.device))
        return torch.cat(chunks)

    @torch.no_grad()
    def add_(self, flat: torch.Tensor, alpha: float = 1.0) -> None:
        for p, chunk in zip(self.params, self._split(flat)):
            p.add_(chunk.to(p.device).view_as(p), alpha=alpha)

    @torch.no_grad()
    def assign_(self, flat: torch.Tensor) -> None:
        for p, chunk in zip(self.params, self._split(flat)):
            p.copy_(chunk.to(p.device).view_as(p))

    def unflatten(self, flat: torch.Tensor):
        return [c.view(s) for c, s in zip(self._split(flat), self._shapes)]
