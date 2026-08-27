"""pytorch-opt ops layer: backend-dispatched shared numerics.

Backends: "reference" (pure torch, ground truth), "native" (compiled
C++/ATen), "auto" (native when loaded, reference otherwise).
"""

from __future__ import annotations

import os
import warnings

import torch

from . import reference

_VALID_BACKENDS = ("auto", "native", "reference")
_BACKEND = "auto"
_NATIVE = None  # module returned by torch.utils.cpp_extension.load


def get_backend() -> str:
    return _BACKEND


def native_available() -> bool:
    return _NATIVE is not None


def set_backend(name: str) -> None:
    global _BACKEND
    if name not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {name!r}")
    if name == "native" and _NATIVE is None:
        raise RuntimeError("native backend not loaded; call pytorch_opt.ops.load_native() first")
    _BACKEND = name


def load_native(verbose: bool = False) -> bool:
    """Compile/load the native extension. Returns True on success; never raises."""
    global _NATIVE
    if _NATIVE is not None:
        return True
    try:
        from . import native_build

        _NATIVE = native_build.load(verbose=verbose)
        return True
    except Exception as e:  # degrade, never break import/usability
        warnings.warn(f"pytorch-opt: native backend unavailable ({type(e).__name__}: {e}); "
                      f"using reference implementations", RuntimeWarning, stacklevel=2)
        return False


def _use_native() -> bool:
    return _BACKEND == "native" or (_BACKEND == "auto" and _NATIVE is not None)


def inverse_matrix_root(A, p, damping=0.0, method="auto", root_dtype=torch.float64,
                        max_iter=100, tol=1e-10):
    if _use_native():
        return _NATIVE.inverse_matrix_root(A, int(p), float(damping), str(method),
                                           root_dtype == torch.float64, int(max_iter), float(tol))
    return reference.inverse_matrix_root(A, p, damping=damping, method=method,
                                         root_dtype=root_dtype, max_iter=max_iter, tol=tol)


def newton_schulz_orthogonalize(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    if _use_native():
        a, b, c = coeffs
        return _NATIVE.newton_schulz_orthogonalize(G, int(steps), float(a), float(b), float(c), float(eps))
    return reference.newton_schulz_orthogonalize(G, steps=steps, coeffs=coeffs, eps=eps)


def kron_factor_update_(L, R, G, beta2=1.0):
    if _use_native():
        _NATIVE.kron_factor_update_(L, R, G, float(beta2))
        return L, R
    return reference.kron_factor_update_(L, R, G, beta2=beta2)


def precond_apply_two_sided(L_root, G, R_root):
    if _use_native():
        return _NATIVE.precond_apply_two_sided(L_root, G, R_root)
    return reference.precond_apply_two_sided(L_root, G, R_root)


if os.environ.get("PYTORCH_OPT_NATIVE") == "1":
    load_native()
