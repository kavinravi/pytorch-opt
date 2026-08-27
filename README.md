# pytorch-opt

Second-order and structure-aware optimizers for PyTorch: natural gradient
descent, quasi-Fisher methods, Kronecker-factored preconditioners, and
trust-region Newton methods — all as `torch.optim.Optimizer` subclasses.

Three layers (see [docs/design.md](docs/design.md)):

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
trust radius/ρ, staleness, timing splits), exact `state_dict()` round-trip,
and a `state_layout()` tagging state replicable/shardable
(distributed-readiness). `pytorch_opt.diag.hessian_eigs(closure, params, k)`
gives Lanczos top-k Hessian eigenvalues.

## Optimizers

| optimizer | family | proven by |
|---|---|---|
| `Muon` | orthogonalized momentum | hand-computed step; Newton–Schulz band + polar direction |
| `Shampoo` | Kronecker full-matrix preconditioner | first step ≡ polar factor `UVᵀ` of the gradient |
| `SOAP` | Adam in Shampoo's eigenbasis | exactly Adam under identity rotations |
| `KFAC` | Kronecker-factored natural gradient | ≡ dense Kronecker solve; sampled Fisher → Newton direction |
| `EKFAC` | K-FAC eigenbasis rescaling | exact reduction to K-FAC; Frobenius optimality |
| `NGD` | exact natural gradient (dense Fisher) | one-step optimum on linear-Gaussian models |
| `TrustNCG` | trust-region Newton-CG | one-step quadratic optimum; saddle escape |
| `HessianFree` | Martens CG-Newton | one-step quadratic optimum; LM damping adapts |
| `AdaHessian` | Hutchinson diagonal Hessian | exact diagonal on diagonal quadratics |
| `Sophia` | clipped diagonal (Hutchinson / GNB) | exact diagonal; hand-computed clipped step |
| `PSGD` | Kronecker gradient-whitening | provably whitens a known gradient covariance |

Every optimizer additionally passes convergence tests and the cross-optimizer
contracts (bitwise determinism, `state_dict` round-trip, diagnostics schema,
state-layout tags). Details and the full test map: [docs/verification.md](docs/verification.md).

## Native backend notes

The native tier is C++/ATen and needs only a host C++ compiler — the compiled
extension runs on CUDA tensors through ATen's dispatcher, so a system CUDA
toolkit that lags your GPU architecture (or torch's CUDA version) does not
block it. Hand-written fused CUDA kernels are future work; on systems where
the toolkit lags, `pip install nvidia-cuda-nvcc-cu13` provides a current nvcc
(point `CUDA_HOME` at `site-packages/nvidia/cu13` for
`torch.utils.cpp_extension`).
