# pytorch-opt

Second-order and structure-aware optimizers for PyTorch: natural gradient
descent, quasi-Fisher methods, Kronecker-factored preconditioners, and
trust-region Newton methods — all as `torch.optim.Optimizer` subclasses.

Three layers (see `docs/2026-08-25-pytorch-opt-design.md`):

- **`pytorch_opt.ops`** — shared numerics (inverse matrix roots, Newton–Schulz
  orthogonalization, Kronecker factor updates). Pure-torch reference
  implementations are the ground truth; a compiled C++/ATen extension provides
  the fast path on CPU **and** CUDA (ATen dispatch — no hand-written kernels
  needed), validated by parity tests.
- **`pytorch_opt.curvature`** — HVP / Gauss–Newton / Fisher vector products,
  Kronecker factor tracking, Steihaug–Toint CG, Lanczos spectra, damping
  policies.
- **`pytorch_opt.optim`** — one readable file per optimizer.

## Install / test

```bash
pip install -e .
python -m pytest -q            # full suite, CPU + CUDA if available
```

The package always works with no compiler present (reference backend). To
build/load the native backend: `pytorch_opt.ops.load_native()` (JIT, cached)
or set `PYTORCH_OPT_NATIVE=1`. Check with `pytorch_opt.ops.native_available()`.

## Quickstart

```python
import torch, torch.nn.functional as F
from pytorch_opt import KFAC, Muon, Shampoo, TrustNCG

# Muon — 2D+ weights get orthogonalized momentum, the rest AdamW (routing by
# ndim; put embeddings/heads in a group with use_muon=False).
opt = Muon(model.parameters(), lr=0.02)
loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()

# Shampoo — Kronecker preconditioner, stale-root schedule, optional grafting.
opt = Shampoo(model.parameters(), lr=0.03, precondition_frequency=20, graft="adagrad")

# KFAC — takes the MODEL (installs hooks), not params. Empirical Fisher by
# default; sampled true-Fisher via fisher_mode="sampled" + update_curvature().
opt = KFAC(model, lr=0.01, damping=1e-3)
out = model(x); loss = F.cross_entropy(out, y)
opt.zero_grad(); loss.backward(); opt.step()

# TrustNCG — REQUIRES a closure returning the loss WITH its graph (no
# backward() inside). For curvature="ggn", return (loss, outputs).
opt = TrustNCG(model.parameters(), delta0=1.0)
def closure():
    return F.cross_entropy(model(x), y)
opt.step(closure)
```

Every optimizer exposes `.diagnostics` (per-step dict: conditioning, damping,
trust radius/ρ, staleness, timing splits), `state_dict()` round-trip, and a
`state_layout()` tagging state replicable/shardable (distributed-readiness).
`pytorch_opt.diag.hessian_eigs(closure, params, k)` gives Lanczos top-k
Hessian eigenvalues.

## Roster

| optimizer | family | status |
|---|---|---|
| `Muon` | orthogonalized momentum | ✅ proven |
| `Shampoo` | Kronecker full-matrix preconditioner | ✅ proven |
| `KFAC` | natural gradient (quasi-Fisher) | ✅ proven |
| `TrustNCG` | trust-region Newton-CG | ✅ proven |
| `NGD` (exact) | natural gradient (dense-Fisher oracle) | ✅ proven |
| `SOAP` | Shampoo eigenbasis + Adam | ✅ proven |
| `AdaHessian` | Hutchinson diagonal Hessian | ✅ proven |
| `Sophia` | clipped diagonal (Hutchinson / GNB) | ✅ proven |
| `EKFAC` | K-FAC eigenbasis rescaling | ✅ proven |
| `HessianFree` | Martens CG-Newton | ✅ proven |
| `PSGD` (Kron) | affine gradient-whitening preconditioner | ✅ proven |

"Proven" = analytic ground-truth tests (e.g. Shampoo's first preconditioned
step equals the gradient's polar factor U Vᵀ; TrustNCG and HessianFree solve a
quadratic in one step; K-FAC's update equals the dense Kronecker solve and its
sampled Fisher recovers the Newton direction; SOAP is bitwise-Adam under
identity rotations; EKFAC reduces exactly to K-FAC under eigenvalue scales;
AdaHessian/Sophia recover exact diagonals on diagonal quadratics; PSGD
provably whitens a known gradient covariance), plus native/reference parity
and determinism, round-trip, and diagnostics contracts. Test names in
`tests/`. Full suite: 227 tests, ~19 s on CPU + RTX 5090.

## Native backend notes

Dev-box reality: RTX 5090 (sm_120) + torch cu130, system nvcc 12.0 — hand-CUDA
kernels can't be compiled by that toolchain, and they aren't needed for
correctness: the Tier-A C++/ATen extension runs on CUDA tensors through ATen's
dispatcher (parity-proven on the 5090). Tier-B fused kernels are viable via the
pip toolchain — `pip install nvidia-cuda-nvcc-cu13` ships nvcc 13.3 with
sm_120 support at `site-packages/nvidia/cu13/bin/nvcc` (point `CUDA_HOME` there
for `torch.utils.cpp_extension`); kernels themselves are future work.
