"""Damping policies shared across optimizers."""

from __future__ import annotations

import math

import torch


def kfac_factored_damping(A: torch.Tensor, G: torch.Tensor, damping: float):
    """K-FAC pi-split of Tikhonov damping across the two Kronecker factors.

    pi = sqrt((tr(A)/dim A) / (tr(G)/dim G)); gamma_A = pi*sqrt(damping),
    gamma_G = sqrt(damping)/pi.
    """
    tr_a = float(A.diagonal(dim1=-2, dim2=-1).sum())
    tr_g = float(G.diagonal(dim1=-2, dim2=-1).sum())
    nu_a = tr_a / A.shape[-1]
    nu_g = tr_g / G.shape[-1]
    if not (math.isfinite(nu_a) and math.isfinite(nu_g)) or nu_a <= 0 or nu_g <= 0:
        pi = 1.0
    else:
        pi = math.sqrt(nu_a / nu_g)
    sq = math.sqrt(damping)
    return pi * sq, sq / pi


def lm_update(damping: float, rho: float, lo: float = 0.25, hi: float = 0.75,
              factor: float = 0.9, min_damping: float = 1e-8,
              max_damping: float = 1e8) -> float:
    """Levenberg-Marquardt style damping adaptation from a reduction ratio."""
    if rho > hi:
        damping = damping * factor
    elif rho < lo:
        damping = damping / factor
    return min(max(damping, min_damping), max_damping)
