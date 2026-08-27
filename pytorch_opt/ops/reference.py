"""Pure-torch ground-truth implementations of the pytorch-opt ops layer.

Every native op is validated against this module; when in doubt, this file wins.
"""

from __future__ import annotations

import torch

_VALID_ROOT_METHODS = ("auto", "eigh", "newton")


def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.mT)


def _matrix_power_int(A: torch.Tensor, p: int) -> torch.Tensor:
    return torch.linalg.matrix_power(A, p)


def _coupled_newton(M: torch.Tensor, p: int, max_iter: int, tol: float) -> torch.Tensor:
    """Coupled Schur-Newton iteration for M^(-1/p), M symmetric PD (Anil et al. style)."""
    n = M.shape[-1]
    I = torch.eye(n, dtype=M.dtype, device=M.device).expand_as(M)
    alpha = -1.0 / p
    fro = torch.linalg.matrix_norm(M, ord="fro")
    z = (1.0 + p) / (2.0 * fro.clamp_min(torch.finfo(M.dtype).tiny))
    z = z.reshape(z.shape + (1, 1))
    X = I * z.pow(-alpha)          # z^(1/p) * I
    Mk = M * z
    prev_err = None
    prev_X = X
    for _ in range(max_iter):
        Mi = (1.0 - alpha) * I + alpha * Mk
        X = X @ Mi
        Mk = _matrix_power_int(Mi, p) @ Mk
        err = (Mk - I).abs().amax()
        if prev_err is not None and err > prev_err * 1.2:
            X = prev_X  # diverging: keep last good iterate
            break
        prev_X, prev_err = X, err
        if err < tol:
            break
    return X


def inverse_matrix_root(
    A: torch.Tensor,
    p: int,
    damping: float = 0.0,
    method: str = "auto",
    root_dtype: torch.dtype = torch.float64,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> torch.Tensor:
    """(A + damping*I)^(-1/p) for symmetric PSD A, batched over leading dims."""
    if int(p) != p or p < 1:
        raise ValueError(f"p must be a positive integer, got {p}")
    if method not in _VALID_ROOT_METHODS:
        raise ValueError(f"method must be one of {_VALID_ROOT_METHODS}, got {method!r}")
    orig_dtype = A.dtype
    M = _sym(A.to(root_dtype))
    if damping:
        n = M.shape[-1]
        M = M + damping * torch.eye(n, dtype=M.dtype, device=M.device).expand_as(M)
    if method == "newton":
        root = _coupled_newton(M, int(p), max_iter, tol)
    else:  # auto -> eigh in the reference backend
        evals, evecs = torch.linalg.eigh(M)
        evals = evals.clamp_min(1e-30)
        root = evecs @ torch.diag_embed(evals.pow(-1.0 / p)) @ evecs.mT
    return _sym(root).to(orig_dtype)


def newton_schulz_orthogonalize(
    G: torch.Tensor,
    steps: int = 5,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> torch.Tensor:
    """Quintic Newton-Schulz approximation of the orthogonal polar factor of G.

    Singular values of the output land in a band around 1 (by design of the
    coefficients), not exactly at 1. Batched over leading dims.
    """
    if G.ndim < 2:
        raise ValueError("newton_schulz_orthogonalize expects a matrix (ndim >= 2)")
    a, b, c = coeffs
    work_dtype = torch.float64 if G.dtype == torch.float64 else torch.float32
    X = G.to(work_dtype)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT
    X = X / (torch.linalg.matrix_norm(X, ord="fro", keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


def kron_factor_update_(
    L: torch.Tensor, R: torch.Tensor, G: torch.Tensor, beta2: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place Kronecker factor accumulation: L += G G^T, R += G^T G (or EMA)."""
    if G.ndim != 2:
        raise ValueError(f"G must be 2D, got shape {tuple(G.shape)}")
    GGt = G @ G.mT
    GtG = G.mT @ G
    if beta2 == 1.0:
        L.add_(GGt)
        R.add_(GtG)
    else:
        L.mul_(beta2).add_(GGt, alpha=1.0 - beta2)
        R.mul_(beta2).add_(GtG, alpha=1.0 - beta2)
    return L, R


def precond_apply_two_sided(
    L_root: torch.Tensor, G: torch.Tensor, R_root: torch.Tensor
) -> torch.Tensor:
    """L_root @ G @ R_root."""
    return L_root @ G @ R_root
