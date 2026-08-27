"""Lanczos estimation of extremal eigenvalues of an implicit symmetric operator."""

from __future__ import annotations

import torch


def lanczos_eigs(mvp, dim: int, k: int = 6, iters: int | None = None, *,
                 device="cpu", dtype: torch.dtype = torch.float32,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Top-k eigenvalue estimates (descending) of the operator behind mvp.

    Full reorthogonalization (fine for the small iteration counts we use).
    """
    if iters is None:
        iters = min(dim, max(2 * k, 20))
    iters = min(iters, dim)
    q = torch.randn(dim, generator=generator, dtype=dtype).to(device)
    q = q / q.norm()
    Q: list[torch.Tensor] = []
    alphas: list[float] = []
    betas: list[float] = []
    for j in range(iters):
        w = mvp(q)
        alpha = q @ w
        w = w - alpha * q
        for qi in Q:            # full reorthogonalization (twice for stability)
            w = w - (qi @ w) * qi
        for qi in Q:
            w = w - (qi @ w) * qi
        Q.append(q)
        alphas.append(float(alpha))
        beta = w.norm()
        if float(beta) < 1e-10 or j == iters - 1:
            break
        betas.append(float(beta))
        q = w / beta
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if betas:
        off = torch.tensor(betas, dtype=torch.float64)
        T += torch.diag(off, 1) + torch.diag(off, -1)
    evals = torch.linalg.eigvalsh(T).flip(0)
    return evals[:k].to(dtype)
