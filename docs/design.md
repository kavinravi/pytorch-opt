# Design

pytorch-opt provides second-order and structure-aware optimizers as
`torch.optim.Optimizer` subclasses, built in three layers so that every
optimizer's step logic stays a single readable Python file while the expensive
shared numerics live in one place.

## Layers

### `pytorch_opt.ops` — shared numerics, backend-dispatched

| op | contract |
|---|---|
| `inverse_matrix_root(A, p, damping, method)` | `(A + damping·I)^(-1/p)` for symmetric PSD `A`, batched. `method ∈ {eigh, newton, auto}`: eigendecomposition with eigenvalue clamping, or a coupled Schur–Newton iteration with convergence/divergence guards. Computed internally in float64 (`root_dtype`), returned in the input dtype. |
| `newton_schulz_orthogonalize(G, steps, coeffs)` | Quintic Newton–Schulz approximation of the orthogonal polar factor `UVᵀ`. Fro-normalizes, transposes so rows ≤ cols, batched. Output singular values land in a band around 1 by construction of the coefficients. |
| `kron_factor_update_(L, R, G, beta2)` | In-place Kronecker factor accumulation `L ← β₂L + (1−β₂)GGᵀ`, `R ← β₂R + (1−β₂)GᵀG`; `beta2=1` is pure sum. |
| `precond_apply_two_sided(L_root, G, R_root)` | `L_root @ G @ R_root`. |

Two backends behind one API: **reference** (pure torch — the numerical ground
truth) and **native** (compiled C++/ATen, JIT-loaded via
`ops.load_native()` or `PYTORCH_OPT_NATIVE=1`). Because ATen dispatches by
tensor device, the same compiled extension serves CPU and CUDA tensors with no
hand-written kernels. `set_backend("auto"|"native"|"reference")`; a failed
native build degrades to reference with a warning, never an import error.

### `pytorch_opt.curvature` — curvature primitives

- `make_hvp` — Hessian-vector products via double backward (graph built once,
  cheap repeated products for CG).
- `make_ggn_vp` — Gauss–Newton products `JᵀH_out Jv` via the double-vjp jvp
  trick; output-Hessian applications provided for MSE and cross-entropy.
- `make_fisher_vp` — true Fisher; for MSE/CE losses the Fisher equals the GGN.
- `per_sample_grads` / `empirical_fisher_vp` — dense empirical Fisher pieces
  (torch.func vmap) for small models.
- `KronTracker` — forward/backward hooks on `nn.Linear`/`nn.Conv2d` capturing
  activation and pre-activation-gradient second moments (K-FAC factors), with
  both empirical and sampled-label (true Fisher) accumulation. Conv2d uses the
  KFC convention (patches unfolded, spatial locations folded into batch); the
  1×1-spatial case reduces exactly to the Linear case.
- Damping policies: fixed Tikhonov, K-FAC π-split factored damping,
  Levenberg–Marquardt adaptation.
- `steihaug_cg` — trust-region truncated CG with negative-curvature and
  boundary handling; `lanczos_eigs` — top-k spectra of implicit operators;
  `ParamVector` — flatten/unflatten for vector-space algorithms.

### `pytorch_opt.optim` — the optimizers

| optimizer | family |
|---|---|
| `Muon` | orthogonalized momentum (Newton–Schulz) |
| `Shampoo` | Kronecker full-matrix preconditioner |
| `SOAP` | Adam in Shampoo's eigenbasis |
| `KFAC` | Kronecker-factored natural gradient |
| `EKFAC` | K-FAC eigenbasis with rescaled diagonal |
| `NGD` | exact natural gradient (dense Fisher; small models) |
| `TrustNCG` | trust-region Newton-CG (Steihaug) |
| `HessianFree` | damped CG-Newton (Martens) with warm starts and backtracking |
| `AdaHessian` | Hutchinson diagonal Hessian, Adam-style |
| `Sophia` | clipped diagonal (Hutchinson or Gauss–Newton–Bartlett) |
| `PSGD` | Kronecker (affine) gradient-whitening preconditioner |

## API conventions

Three deliberate asymmetries, stated rather than papered over:

1. **Closure-based optimizers** (`TrustNCG`, `HessianFree`, `NGD`,
   `AdaHessian`, `Sophia`): `step(closure)` takes a closure returning the loss
   *with its autograd graph* (no `backward()` inside) — these methods need
   curvature products or multiple objective evaluations per step. GGN-based
   modes take `(loss, outputs)`.
2. **`KFAC`/`EKFAC` take the model**, not a parameter iterable — they install
   module hooks and key curvature state by module path.
3. **`Muon` routes parameters**: 2D+ weights get orthogonalized momentum
   (conv kernels flattened), everything else an internal AdamW; override per
   param group with `use_muon`.

## Cross-cutting contracts

- Deterministic under a fixed seed (stochastic estimators draw from internal
  seeded generators whose state round-trips through `state_dict`).
- Exact `state_dict()`/`load_state_dict()` round-trip for every optimizer,
  including hook-keyed factors, trust radii, and generator states.
- `.diagnostics` — per-step dict (conditioning, damping, trust radius / ρ,
  CG iterations, preconditioner staleness, curvature/step timing splits).
- `state_layout()` — every state key tagged `replicable` (must agree across
  ranks) or `shardable` (per-parameter), so a future distributed layer can
  shard state without redesign; curvature refreshes are already decoupled from
  stepping (`precondition_frequency` / `inv_every`).

## Numeric conventions

- Losses are mean-over-batch; the MSE convention is `0.5·Σ(y−t)²/B` (output
  Hessian `I/B`).
- Matrix roots/inverses are computed in float64 internally and cast back.
- torch's row-major `.flatten()` fixes the Kronecker orientation:
  `vec(iG · V · iA) = kron(iG, iA) · vec(V)` for symmetric `iA` — all
  Kronecker algebra in the codebase is written and tested against this
  identity.
