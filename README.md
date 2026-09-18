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

## Consistent AdamW fallback for language models

Muon, Shampoo, SOAP, and KFAC accept the same explicit parameter split.
`matrix_param_groups` selects **named module weights**, and assigns every other
trainable parameter to AdamW exactly once. It excludes biases, vectors and
parameters marked `_no_weight_decay` from weight decay. Shared weights,
including tied token embeddings/output heads, must stay in the AdamW group.

```python
from pytorch_opt import KFAC, Muon, Shampoo, SOAP, matrix_param_groups

# These must be actual module paths in your model. For a bare Mamba2 mixer:
selected_modules = ["in_proj", "out_proj"]
groups = matrix_param_groups(
    model, selected_modules,
    lr=0.01, adamw_lr=3e-4, weight_decay=0.1, adamw_wd=0.1,
)
# Choose ONE optimizer, rebuilding groups for each independently initialized model:
opt = KFAC(model, params=groups, fisher_mode="sampled")
# opt = Muon(groups)
# opt = Shampoo(groups, max_preconditioner_dim=4096)
# opt = SOAP(groups, max_preconditioner_dim=4096)
# Control arm: torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
```

Each group has `use_preconditioner`, `param_names`, `lr`, and `weight_decay`.
Keep the same selected module paths across optimizer arms for a given model.
The AdamW fallback defaults to `adamw_betas=(0.9, 0.95)` and `adamw_eps=1e-8`
in all four optimizers. Explicit fallback groups use their ordinary `lr` and
`weight_decay`, so standard PyTorch learning-rate schedulers handle both groups.
Rebuild the groups for each optimizer; optimizers populate their group dictionaries.

Without explicit groups, Muon/Shampoo/SOAP still select parameters by tensor
rank, and use constructor `adamw_lr`/`adamw_wd` for the remaining parameters.
That convenience mode cannot identify embeddings or output heads. Use the
explicit split for experiments. Oversized selected Shampoo/SOAP matrices now
raise an error instead of silently changing optimizer. Choose a sufficient
`max_preconditioner_dim` and measure memory use before a full-size run.

KFAC tracks only selected modules. It raises before updating parameters if a
selected layer receives gradients without collecting curvature. For Mamba-2,
construct the mixer with `use_mem_eff_path=False` to expose the `out_proj`
module call; the fused path passes its weights directly to a kernel and bypasses
hooks. Keep the same forward implementation across arms for controlled timing
comparisons. Transformer projections must likewise execute their `nn.Linear`
modules; functional projections, including some attention implementations, need
an integration change. This package does not modify either model automatically.

For KFAC, losses must be means over batch rows, or over tokens for sequence
logits. With gradient accumulation or AMP scaling, tell empirical KFAC the
scale applied to that mean loss **before backward**:

```python
opt.set_grad_scale(1.0 / accumulation_steps)
# With GradScaler: opt.set_grad_scale(scaler.get_scale() / accumulation_steps)
(loss / accumulation_steps).backward()
# Unscale parameter gradients with scaler.unscale_(opt) before clipping/stepping.
```

This setting corrects curvature, not parameter gradients. Masked or weighted
losses need a scale consistent with their actual reduction; the tracker cannot
infer it. Linear statistics treat each token as a row. Conv2d retains the
mean-over-batch KFC convention. Factors use at least FP32 even under autocast.
For sampled Fisher, call `opt.update_curvature(logits)` before the training
backward frees the graph. Logits may be `[batch, tokens, vocabulary]`; the
auxiliary backward ignores the training loss scale and leaves `.grad` untouched.
Use non-reentrant activation checkpointing with this extra backward.

Optimizer state includes both update rules and KFAC curvature/cadence settings.
Resume requires the same parameter split. Checkpoints predating this routing
change are rejected; their SGD/diagonal fallback states cannot resume the new
algorithm exactly. The old KFAC `sgd_lr` option is replaced by the AdamW options.
Shampoo/SOAP now report `n_adamw_params` in diagnostics. Model/RNG/data-loader
state and the generator passed to sampled Fisher remain the training loop's
checkpoint responsibility. Distributed curvature synchronization is not
implemented here.

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
block it.
