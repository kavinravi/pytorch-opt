# pytorch-opt — Design Spec

- **Date:** 2026-08-25
- **Status:** Implemented 2026-08-26 (v1 spine + full extension roster; 227 tests green). See docs/2026-08-26-build-report.md.
- **Owner:** Kavin Ravi

---

## The prompt

Build **pytorch-opt**: an extension of PyTorch that provides natural gradient descent,
quasi-Fisher methods, and the common second-order / structure-aware optimizers —
trust-region Newton-CG, SOAP, Shampoo, Muon, K-FAC, and relatives — all available as
optimizers compatible with the `torch.optim.Optimizer` interface.

It is a **true C++/CUDA extension**: the expensive shared numerics run in compiled native
code; each optimizer's step logic stays a readable Python file that can be checked
line-by-line against its paper. It is first and foremost a **personal research testbed**:
correctness, instrumentation, and ease of modification outrank raw throughput and API
ceremony. It is built single-GPU but **distributed-ready by design** — no distributed
code in v1, and no design dead-ends that would block sharding later.

## Goals

1. One `pip install -e .`-able package, `import pytorch_opt`, that **always imports and
   fully works with no compiler present** (pure-torch fallbacks), and gets faster when the
   native extension is built.
2. A v1 spine of four optimizers — one per family — proven correct against **analytic
   ground truth**, not just "loss goes down":
   - **K-FAC** (natural gradient / quasi-Fisher family)
   - **TrustNCG** (trust-region second-order family)
   - **Shampoo** (Kronecker-preconditioner family)
   - **Muon** (orthogonalized-momentum family)
3. Shared curvature/numerics layers that make the extension roster thin additions:
   **NGD (exact), SOAP, AdaHessian, Sophia, EKFAC, Hessian-free, PSGD** (priority order).
4. First-class instrumentation: per-step diagnostics for curvature conditioning, damping,
   step acceptance, preconditioner staleness, and timing splits.
5. Reproducibility: deterministic under fixed seed; full `state_dict` round-trip for every
   optimizer, including hook-based state (K-FAC factors).

## Non-goals (v1)

- No distributed execution (no collectives, no rank logic, no FSDP/DDP integration).
- No ROCm/TPU support.
- No LR schedulers (use torch's).
- No transformer-scale benchmarks; target models are MLPs, small CNNs, small transformers
  on one device.
- No attempt to beat reference implementations on wall-clock; parity of *math* is the bar.

---

## Architecture: three layers

### L0 — native numerics (`pytorch_opt/csrc/`, exposed as `pytorch_opt.ops`)

The only code that earns native compilation. Two tiers:

- **Tier A (C++/ATen, compiles with g++ alone):** ops written as C++ against the ATen API.
  Because ATen dispatches by tensor device, the *same compiled extension* runs on CPU and
  on CUDA tensors (cuBLAS/cuSOLVER underneath) — no nvcc required. This tier is mandatory.
- **Tier B (hand-written `.cu` kernels):** fused elementwise/reduction kernels where
  kernel-launch overhead or memory traffic actually dominates. Compiled only when a
  CUDA toolkit matching torch's major CUDA version is present; otherwise skipped.
  This tier is best-effort and never load-bearing.

Ops (each with a pure-torch **reference implementation that is the numerical ground
truth**; native is validated against it in parity tests):

| op | contract |
|---|---|
| `inverse_matrix_root(A, p, damping, method)` | `(A + damping·I)^(-1/p)` for symmetric PSD `A`; `method ∈ {eigh, newton, auto}`. `eigh`: eigendecomposition with eigenvalue clamping. `newton`: coupled Newton–Schur iteration (Anil et al. style) with convergence/divergence guards. Internally computes in float64 by default (`root_dtype` option), returns input dtype. Batched over leading dims. |
| `newton_schulz_orthogonalize(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315))` | Quintic Newton–Schulz iteration approximating the orthogonal polar factor `U Vᵀ` of `G`. Normalizes by Frobenius norm, transposes internally so rows ≤ cols. Singular values of output land in the known NS5 band around 1, not exactly 1. |
| `kron_factor_update_(L, R, G, beta2)` | In-place `L ← β₂L + (1−β₂)·G Gᵀ`, `R ← β₂R + (1−β₂)·Gᵀ G`; `beta2=1.0` means pure sum-accumulate (`L += G Gᵀ`). |
| `precond_apply_two_sided(L_root, G, R_root)` | `L_root @ G @ R_root`. |
| `symeig_psd(A)` | Eigendecomposition for symmetric PSD, batched; thin wrapper (native tier optional — `torch.linalg.eigh` is already cuSOLVER-backed). |
| `cg_fused_ops` | Fused axpy/dot/norm updates for CG inner loops (Tier B candidate; reference = plain torch). |

Backend selection: `pytorch_opt.ops.set_backend("auto"|"native"|"reference")`;
`auto` uses native when the extension loaded, reference otherwise. Native build is JIT via
`torch.utils.cpp_extension.load` (ninja-cached), triggered by `pytorch_opt.ops.load_native()`
or env `PYTORCH_OPT_NATIVE=1`; failure to build degrades to reference with a warning, never
an import error.

### L1 — curvature primitives (Python, `pytorch_opt/curvature/`)

- `hvp(loss_closure, params, v)` — Hessian-vector product via double backward.
- `ggn_vp(model_fn, loss_hess, params, v)` — Gauss–Newton-vector product `Jᵀ H_out J v`
  via the double-vjp jvp trick; prebuilt output-Hessian applications for `mse` and
  `cross_entropy` (softmax: `diag(p) − p pᵀ` applied without materializing).
- `fisher_vp(...)` — true Fisher via GGN identity for exponential-family losses
  (CE, MSE ⇒ Fisher ≡ GGN); `empirical=True` computes `(1/N) Σ gᵢ (gᵢ·v)` via
  per-sample grads (`torch.func.vmap(grad)`).
- `KronTracker` — forward/backward hooks on `nn.Linear` / `nn.Conv2d` capturing
  activations `a` and pre-activation gradients `g`; maintains EMA Kronecker factors
  `A = E[ã ãᵀ]` (bias-augmented) and `G = E[g̃ g̃ᵀ]` (batch-size rescaled). Conv2d via
  `unfold` patches (KFC convention, spatial locations folded into batch). Trackable
  on/off; supports a *sampled-label* backward pass (true Fisher) as well as the
  empirical (real-gradient) pass.
- Damping policies: fixed Tikhonov; K-FAC factored damping with
  `π = sqrt( (tr(A)/dim A) / (tr(G)/dim G) )`, factor dampings `π·sqrt(λ)` and `sqrt(λ)/π`;
  Levenberg–Marquardt adaptation (shared with trust-region ρ logic).
- `lanczos_eigs(mvp_fn, dim, k)` — top-k eigenvalue estimates of any implicit operator.
- `ParamVector` — flatten/unflatten utilities to run CG over a parameter list as one vector.
- `steihaug_cg(mvp, g, radius, tol, max_iter)` — Steihaug–Toint truncated CG: handles
  negative curvature (follow direction to boundary) and radius exit (boundary crossing).

### L2 — optimizers (`pytorch_opt/optim/`, one readable file each)

All subclass `torch.optim.Optimizer`, all support `state_dict()`/`load_state_dict()`
round-trip, all emit diagnostics (below). Two **sanctioned API asymmetries**, stated
rather than papered over:

1. **`TrustNCG` requires the `closure` protocol** (like `torch.optim.LBFGS`): trust-region
   methods evaluate the objective more than once per step (actual-vs-predicted reduction).
   `step()` without a closure raises with an instructive message.
2. **`KFAC` takes `model`, not `params`**: it needs module identity to install hooks.
   State is keyed by module path for portable `state_dict`s.

**Muon routes parameters rather than owning them all:** orthogonalized momentum applies to
2D+ weights (conv kernels flattened `(out, −1)`); 1D/0D params (biases, norms) fall back to
an internal AdamW. Default routing by `ndim ≥ 2`, overridable per param-group via
`use_muon`; embeddings/heads are documented as belonging in an AdamW group.

---

## v1 optimizer specifications

Conventions: params `W`, gradient `∇`, learning rate `lr`, decoupled weight decay optional
everywhere it makes sense (`weight_decay`, AdamW-style).

### Muon (`pytorch_opt/optim/muon.py`)
- State per routed param: momentum buffer `B ← μB + ∇` (nesterov option: use `∇ + μB`).
- Update: `O = newton_schulz_orthogonalize(B, steps)`;
  scale `s` by `lr_adjust ∈ {"spectral" (default): max(1, m/n)^½, "match_rms_adam": 0.2·√max(m,n), "none"}`;
  `W ← W − lr·s·O`.
- Non-routed params: internal AdamW (its own betas/eps/lr factor).
- Defaults: `lr=0.02, momentum=0.95, nesterov=True, ns_steps=5`.

### Shampoo (`pytorch_opt/optim/shampoo.py`)
- 2D params (convs reshaped `(out, −1)`): factors `L (m×m)`, `R (n×n)` updated by
  `kron_factor_update_` with `beta2` (default 0.999; `beta2=1` = classic sum).
- Preconditioned grad `P = L^(−1/4) ∇ R^(−1/4)` with roots recomputed every
  `precondition_frequency` steps (default 20), stale roots used between recomputes.
- 1D params or any dim > `max_preconditioner_dim` (default 1024): diagonal AdaGrad-style
  fallback preconditioner.
- Optional grafting `graft ∈ {"none" (default), "sgd", "adagrad"}`: direction from Shampoo,
  per-layer magnitude from the grafted method.
- Momentum on the preconditioned step optional. Defaults: `lr=0.03, eps=1e-8` (added to
  factors before rooting), `root_method="auto"`.
- Bootstrap: before the first root computation, steps use the grafted/diagonal direction.

### KFAC (`pytorch_opt/optim/kfac.py`)
- Wraps `KronTracker`. For each tracked module with weight grad `∇W (out×in)` (bias folded
  in as an extra input column): natural-gradient step
  `Δ = (G + γ_G I)^(−1) ∇W̃ (A + γ_A I)^(−1)` with π-split factored damping (above),
  then unfold bias column back out.
- `fisher_mode ∈ {"empirical" (default), "sampled"}`; sampled mode exposes
  `kfac.update_curvature(outputs, kind="categorical"|"gaussian")` which samples labels
  from the model's predictive distribution and runs the extra backward, tracking only
  that pass.
- Cadences: `stats_every` (factor EMA update, default 1), `inv_every` (inverse/root
  recompute, default 10; stale inverses between).
- Untracked params (norm layers etc.): plain SGD-with-momentum path inside the optimizer.
- Optional KL-style step clipping `max_grad_norm`. Defaults:
  `lr=0.01, damping=1e-3, ema_decay=0.95, momentum=0.9`.

### TrustNCG (`pytorch_opt/optim/trust_ncg.py`)
- Closure-based full-parameter method over a `ParamVector`.
- Subproblem `min_s g·s + ½ sᵀHs, ‖s‖ ≤ Δ` solved by `steihaug_cg`; forcing tolerance
  `η_k = min(0.5, √‖g‖)·‖g‖`.
- Curvature `curvature ∈ {"hessian" (default), "ggn"}` via L1 `hvp`/`ggn_vp`.
- Acceptance: `ρ = (f(x) − f(x+s)) / (−m(s))`; accept if `ρ > 0.15`; `Δ ← 0.25Δ` if
  `ρ < 0.25`; `Δ ← min(2Δ, Δ_max)` if `ρ > 0.75` and `‖s‖ ≈ Δ` (Nocedal–Wright Alg. 4.1).
- Defaults: `delta0=1.0, delta_max=1e3, max_cg_iter=250, damping=0.0`.

---

## Extension roster (post-v1, priority order)

Each is a thin file over L0/L1 once the spine is proven:

1. **NGD (exact)** — dense Fisher `(F + λI)^(−1) g` via per-sample grads; small nets only;
   doubles as cross-validation oracle for K-FAC/EKFAC tests.
2. **SOAP** — Shampoo factor eigenbases + Adam run in the rotated space; moments projected
   on basis refresh. Parity property: with identity rotations SOAP ≡ Adam exactly.
3. **AdaHessian** — Hutchinson diagonal Hessian (`z ⊙ Hz`, Rademacher z) with spatial
   averaging for convs; Adam-style step on it.
4. **Sophia** — EMA clipped-diagonal method; both Sophia-H (Hutchinson) and Sophia-G
   (Gauss–Newton–Bartlett sampled) estimators.
5. **EKFAC** — K-FAC eigenbasis + per-eigencomponent second-moment rescaling.
6. **Hessian-free** — Martens-style CG-Newton: LM damping, CG-iterate backtracking,
   warm-started CG; shares everything with TrustNCG.
7. **PSGD (Kron)** — Kronecker-factored whitening preconditioner (Xi-Lin Li).

## Distributed-readiness rules (design constraints, no code)

- Every optimizer state entry is declared **replicable** (must agree across ranks:
  Kronecker factors, roots, trust radius) or **shardable** (per-param, one owner:
  momenta, diagonal accumulators) via a `state_layout()` classmethod returning the tags.
- Curvature refresh is decoupled from stepping (`precondition_frequency` / `inv_every`)
  so factor inversions can later round-robin across ranks.
- No op may assume all params live on one device object; per-param device is read from
  the param.

## Instrumentation

`optimizer.diagnostics` → dict of the latest step's metrics; `optimizer.history` optional
ring buffer (`track_history=N`). Common schema plus per-optimizer keys:

- Shampoo/K-FAC: factor condition-number estimates (free at root/inverse recompute),
  preconditioner staleness (steps since refresh), effective damping.
- TrustNCG: `Δ`, `ρ`, CG iterations, termination reason (converged/boundary/neg-curv),
  accepted flag.
- Muon: update RMS, routed/non-routed param counts.
- All: wall-clock split `{curvature_ms, step_ms}`.
- `pytorch_opt.diag.hessian_eigs(closure, params, k)` — Lanczos top-k spectrum utility.

## Correctness strategy (the proof, not vibes)

Analytic ground truth, then parity, then convergence — in that order:

| claim | test |
|---|---|
| `inverse_matrix_root` correct | matches closed form on constructed SPD matrices; `newton` ≡ `eigh` to tight tol |
| NS5 orthogonalizes | singular values in band; orthogonal input ≈ fixed point; direction ≈ SVD polar factor |
| `hvp` correct | equals `Av` on quadratic `½xᵀAx`; finite differences on an MLP |
| `ggn_vp` correct | equals dense `JᵀHJ v` built column-by-column on tiny nets (MSE and CE) |
| Fisher ≡ GGN for CE/MSE | sampled-Fisher MC estimate → GGN (statistical, fixed seed) |
| Steihaug correct | full-radius quadratic solved to optimum in one outer step; boundary and neg-curvature branches exercised |
| TrustNCG | 1-step optimum on convex quadratic; Rosenbrock → (1,1); escapes `x²−y²` saddle |
| Shampoo | **single-step identity**: with sum-accumulate and eps→0, first preconditioned grad `L^(−1/4) G R^(−1/4) = U Vᵀ` (polar factor of G) — checked against SVD |
| K-FAC algebra | vec/Kron orientation identity `(G+γI)^(−1) ∇ (A+γI)^(−1) ≡ unvec(((A+γI)⊗(G+γI))^(−1) vec ∇)`; factors match hand-computed moments; B=1 empirical factors reproduce `G ⊗ A` exactly |
| K-FAC ≈ Newton | single linear layer + MSE + sampled Fisher: direction → damped Newton direction (statistical) |
| Muon | hand-computed single step; routing table correct |
| native ops | parity vs reference backend, CPU and CUDA, forced-`native` fixture |
| every optimizer | toy-MLP convergence under fixed seed and budget; deterministic trajectory twice; `state_dict` round-trip (k steps + save + m steps ≡ load + m steps) |

Test suite must stay fast (< ~3 min CPU); GPU tests parametrized by available device.

## Packaging & environment

- Flat layout: `pytorch_opt/` package at repo root (`ops.py`, `csrc/`, `curvature/`,
  `optim/`, `diag.py`), `tests/` at root, `pyproject.toml` (setuptools), editable install.
- Python ≥ 3.10 (dev env: 3.12.3), torch ≥ 2.2 (dev env: 2.13.0+cu130).
- Dev machine reality: RTX 5090 (sm_120) with torch cu130, but system nvcc is CUDA 12.0 —
  it can neither target sm_120 nor match torch's CUDA major. Hence Tier A (ATen C++,
  g++-only) is the mandatory native path and runs on the 5090 through ATen dispatch;
  Tier B `.cu` kernels compile only if a CUDA 13 toolkit is provisioned (pip
  `nvidia-cuda-nvcc-cu13` route documented as a stretch task).
- Deterministic seeding throughout; sampled-Fisher paths accept a `torch.Generator`.

## Delivery order

1. Spec + implementation plan committed.
2. Package skeleton; L0 reference ops + tests.
3. L1 curvature + tests.
4. L2 spine in TDD order Muon → Shampoo → TrustNCG → KFAC, each proven before the next.
5. Diagnostics + determinism + round-trip tests woven in per optimizer.
6. Tier A native extension + parity tests (CPU + CUDA); Tier B best-effort.
7. Extended roster in priority order, same bar per optimizer.
