"""Steihaug-Toint truncated CG for the trust-region subproblem."""

from __future__ import annotations

import torch


def _boundary_tau(s: torch.Tensor, d: torch.Tensor, radius: float) -> torch.Tensor:
    """Positive root of ||s + tau*d|| = radius."""
    sd = s @ d
    dd = d @ d
    ss = s @ s
    disc = sd * sd + dd * (radius * radius - ss)
    return (-sd + disc.clamp_min(0.0).sqrt()) / dd


def steihaug_cg(mvp, g: torch.Tensor, radius: float, tol: float | None = None,
                max_iter: int = 250):
    """Approximately minimize m(s) = g.s + 0.5 s.H s subject to ||s|| <= radius.

    mvp: callable v -> H @ v (flat tensors). Returns (s, info) with
    info["reason"] in {"converged", "boundary", "neg_curvature", "max_iter", "zero_grad"}.
    """
    g = g.detach()
    gnorm = g.norm()
    if float(gnorm) < 1e-12:
        return torch.zeros_like(g), {"iters": 0, "reason": "zero_grad"}
    if tol is None:
        tol = float(torch.minimum(torch.tensor(0.5, device=g.device), gnorm.sqrt()) * gnorm)

    s = torch.zeros_like(g)
    r = g.clone()
    d = -g
    rr = r @ r
    for k in range(1, max_iter + 1):
        Hd = mvp(d)
        dHd = d @ Hd
        if float(dHd) <= 0.0:
            tau = _boundary_tau(s, d, radius)
            return s + tau * d, {"iters": k, "reason": "neg_curvature"}
        alpha = rr / dHd
        s_next = s + alpha * d
        if float(s_next.norm()) >= radius:
            tau = _boundary_tau(s, d, radius)
            return s + tau * d, {"iters": k, "reason": "boundary"}
        s = s_next
        r = r + alpha * Hd
        rr_new = r @ r
        if float(rr_new.sqrt()) < tol:
            return s, {"iters": k, "reason": "converged"}
        d = -r + (rr_new / rr) * d
        rr = rr_new
    return s, {"iters": max_iter, "reason": "max_iter"}
