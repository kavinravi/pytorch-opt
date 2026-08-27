# pytorch-opt v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline execution — subagents not authorized this session; user asleep, autonomous overnight run explicitly approved 2026-08-25). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and *prove* the pytorch-opt v1 spine — ops layer (reference + native ATen-C++), curvature layer, and four optimizers (Muon, Shampoo, TrustNCG, KFAC) — green under `pytest` with analytic ground-truth tests; then extend the roster.

**Architecture:** Three layers per the spec: `pytorch_opt.ops` (backend-dispatched numerics: pure-torch reference = ground truth; JIT-compiled ATen-C++ native = fast path, CPU+CUDA via dispatcher), `pytorch_opt.curvature` (hvp/ggn/fisher products, Kronecker tracker, Steihaug CG, Lanczos, ParamVector, damping), `pytorch_opt.optim` (one readable file per optimizer, `torch.optim.Optimizer` subclasses).

**Tech Stack:** Python 3.12, torch 2.13.0+cu130, pytest 9, ninja + g++ 13 (JIT C++ extension), RTX 5090 available. System nvcc is CUDA 12.0 → **no hand-CUDA kernels tonight** (Tier B gated off); native tier is ATen-C++ only.

**Spec:** `docs/2026-08-25-pytorch-opt-design.md` (read it first; math conventions live there).

## Global Constraints

- Package must import and pass the full suite with **no compiler and no GPU** (reference backend, CPU).
- Every native op is validated against its reference implementation in parity tests; reference is ground truth.
- Matrix roots/inverses computed internally in float64 (`root_dtype` option), returned in input dtype.
- Loss convention everywhere (tests, curvature, trackers): **mean over batch**; MSE is `0.5 * (y - t).pow(2).sum() / B`.
- vec convention: torch `.flatten()` is row-major ⇒ `vec(iG @ V @ iA) == kron(iG, iA) @ vec(V)` for symmetric iA. All Kronecker algebra written against this identity.
- Every optimizer: `state_dict()` round-trip exact; deterministic trajectory under fixed seed (CPU); `.diagnostics` dict after each step; `state_layout()` classmethod tagging state `replicable|shardable`.
- Suite runtime target < ~3 min CPU. Tiny models only (`pytorch_opt/_testing.py`).
- Commit at the end of every task (conventional commits, Co-Authored-By Claude trailer).
- TDD per task: write test → watch fail → implement → watch pass → commit.

---

### Task 0: Skeleton + editable install + smoke test

**Files:** Create `pyproject.toml`, `README.md`, `pytorch_opt/__init__.py`, `pytorch_opt/_testing.py`, `tests/conftest.py`, `tests/test_smoke.py`; empty pkg dirs `pytorch_opt/{ops,curvature,optim}/__init__.py`.

**Interfaces produced:** `pytorch_opt.__version__ = "0.1.0.dev0"`. `_testing.py`: `TinyMLP(d_in=8, h=16, d_out=1|k)`, `TinyCNN()` (Conv2d(1,4,3)→ReLU→Conv2d(4,4,3,stride=2)→flatten→Linear), `make_regression(n=64, d=8, *, seed=0, device="cpu") -> (X, y)` (y from a fixed random linear map + 0.05 noise), `make_classification(n=64, d=8, k=3, *, seed=0, device)`, `mse_half(pred, t)` (the Global-Constraints MSE), `run_steps(model, opt, X, y, loss_fn, n) -> list[float]` — handles three protocols: TrustNCG (closure returning loss w/ graph; for ggn mode `(loss, outputs)`), KFAC (`opt.update_curvature(out)` when `fisher_mode=="sampled"`), plain (backward+step).

- [ ] Write `pyproject.toml` (setuptools, name `pytorch-opt`, version 0.1.0.dev0, requires-python >=3.10, deps `["torch>=2.2"]`), minimal README, package inits.
- [ ] `pip install -e .` succeeds.
- [ ] `tests/conftest.py`: `device` fixture parametrized over `["cpu"] + (["cuda"] if torch.cuda.is_available() else [])`; autouse fixture `torch.manual_seed(0)`.
- [ ] `tests/test_smoke.py`: import package, assert version; build TinyMLP, one fwd/bwd; run pytest → green.
- [ ] Commit `feat: package skeleton, test harness`.

### Task 1: ParamVector

**Files:** Create `pytorch_opt/curvature/paramvec.py`, `tests/test_paramvec.py`.

**Produces:** `class ParamVector(params)`: `.numel`, `.gather() -> flat detached`, `.gather_grads(fill_none=0.0)`, `.add_(flat, alpha=1.0)` (in-place into params, no_grad), `.assign_(flat)`, `.unflatten(flat) -> list[Tensor]` views reshaped per-param. Mixed devices allowed: flat lives on first param's device; scatter casts back per-param.

- [ ] Tests: round-trip gather→assign_ identity; add_ then gather reflects change; unflatten shapes match; grads gather with a None grad filled with zeros; mixed-dtype params rejected with clear error.
- [ ] Implement; pass; commit `feat(curvature): ParamVector flatten/unflatten`.

### Task 2: Reference op — inverse_matrix_root

**Files:** Create `pytorch_opt/ops/reference.py`, `tests/test_ops_reference.py`.

**Produces:** `inverse_matrix_root(A, p, damping=0.0, method="auto", root_dtype=torch.float64, max_iter=100, tol=1e-10) -> Tensor` — `(A + damping I)^(-1/p)`, batched `(..., n, n)`, symmetric-PSD input, output symmetrized, input dtype restored. `method`: `eigh` (clamp evals to ≥1e-30 then pow(-1/p)), `newton` (coupled Schur–Newton: `alpha=-1/p`; `z=(1+p)/(2‖M‖_F)`; `X0=z^{1/p}·I`, `M0=z·M`; iterate `Mi=(1-alpha)I+alpha·M`, `X←X@Mi`, `M←Mi^p@M` (binary-exponentiation matrix power); stop at `max|M-I|<tol`; divergence guard: if error > 1.2·prev, return previous X), `auto`→`eigh` in reference.

- [ ] Failing tests:
```python
def test_root_matches_closed_form(device):
    # A = Q diag(l) Q^T with known l  ->  A^{-1/p} = Q diag(l^{-1/p}) Q^T
    torch.manual_seed(1); n = 6
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device=device))
    l = torch.linspace(0.5, 4.0, n, dtype=torch.float64, device=device)
    A = Q @ torch.diag(l) @ Q.T
    for p in (1, 2, 4):
        want = Q @ torch.diag(l.pow(-1.0 / p)) @ Q.T
        got = ops_ref.inverse_matrix_root(A, p)
        assert torch.allclose(got, want, atol=1e-10)

def test_newton_matches_eigh(device):       # p in (2,4), damped random PSD, atol 1e-7
def test_damping_applied(device):           # damping=d equals eigh of (A + d I)
def test_batched_and_dtype(device):         # (3,5,5) float32 in -> float32 out, matches per-slice fp64
```
- [ ] Implement; pass on cpu+cuda; commit `feat(ops): inverse_matrix_root (eigh + coupled Newton), fp64 internals`.

### Task 3: Reference ops — newton_schulz, kron factors, precond apply

**Files:** Modify `pytorch_opt/ops/reference.py`; extend `tests/test_ops_reference.py`.

**Produces:** `newton_schulz_orthogonalize(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7)` (fp32 internal, Fro-normalize, transpose so rows≤cols, quintic `X←aX+(bA+cA²)X` with `A=XXᵀ`, batched); `kron_factor_update_(L, R, G, beta2=1.0)` (in-place; `beta2==1` pure add; else EMA with `alpha=1-beta2`); `precond_apply_two_sided(L_root, G, R_root)`.

- [ ] Failing tests:
```python
def test_ns_orthogonalizes(device):
    G = torch.randn(16, 32, device=device)
    O = ops_ref.newton_schulz_orthogonalize(G)
    s = torch.linalg.svdvals(O.double())
    assert s.min() > 0.3 and s.max() < 1.6          # NS5 band (loose)
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    assert (O.double() - U @ Vh).norm() / (U @ Vh).norm() < 0.35   # polar-direction proximity

def test_ns_orthogonal_fixed_point(device):   # G orthogonal -> O ≈ G (atol 5e-2), incl. tall via transpose
def test_ns_batched_matches_loop(device)
def test_kron_factor_update_sum_and_ema(device)   # manual G@G.T bookkeeping, both beta2 modes
def test_precond_apply(device)
```
- [ ] Implement; pass; commit `feat(ops): newton_schulz, kron factor update, two-sided apply`.

### Task 4: ops dispatch layer

**Files:** Create `pytorch_opt/ops/__init__.py` (public API + backend registry), `pytorch_opt/ops/native_build.py` (loader shell; real csrc in Task 13), `tests/test_ops_dispatch.py`.

**Produces:** module-level public functions with reference signatures; `set_backend(name)` / `get_backend()` (`auto|native|reference`); `load_native(verbose=False) -> bool` (JIT `torch.utils.cpp_extension.load`; failure ⇒ warn + False, never raise on import); `native_available() -> bool`. `auto` = native if loaded else reference. Env `PYTORCH_OPT_NATIVE=1` triggers load at import, guarded.

- [ ] Tests: default backend auto→reference dispatch returns reference results; `set_backend("reference")` explicit; unknown backend raises ValueError; `native_available()` False before any load.
- [ ] Implement; pass; commit `feat(ops): backend dispatch (auto/native/reference)`.

### Task 5: Curvature products — hvp, ggn_vp, fisher_vp

**Files:** Create `pytorch_opt/curvature/products.py`, `tests/test_products.py`.

**Produces:**
- `make_hvp(loss, params) -> (hvp_fn, flat_grad)`: grads once with `create_graph=True`; `hvp_fn(vflat)->flat` via `autograd.grad(flat_g @ v, params, retain_graph=True)`.
- `make_ggn_vp(outputs, hess_mvp, params) -> ggn_fn`: w-trick jvp (`w=zeros_like(outputs, requires_grad=True)`; `g_w=grad(outputs, params, grad_outputs=w, create_graph=True)`; `Jv = grad(Σ g_w·v, w, retain_graph=True)`), then `u = hess_mvp(Jv)`, then `JᵀHJv = grad(outputs, params, grad_outputs=u.detach(), retain_graph=True)`; flat in/out.
- `mse_hess_mvp(outputs)` → `u ↦ u/B`; `ce_hess_mvp(logits)` → `u ↦ (p⊙u − p⊙Σ_c(p⊙u))/B`, `p=softmax(logits).detach()`.
- `fisher_vp = make_fisher_vp(outputs, loss_type, params)` — true-Fisher ≡ GGN for `{"mse","ce"}` (delegates).
- `per_sample_grads(model, X, y, loss_fn) -> (B, n_params)` via `torch.func.functional_call` + `vmap(grad)` (per-sample loss WITHOUT the 1/B); `empirical_fisher_vp` from it: `Fv = Gᵀ(Gv)/B`.

- [ ] Failing tests:
```python
def test_hvp_quadratic(device):        # f=0.5 x^T A x  ->  hvp(v)=Av exact (atol 1e-6)
def test_hvp_mlp_finite_diff(device)   # central FD on grad, atol 1e-4 (fp64 model)
def test_ggn_dense_mse(device):
    # TinyMLP(4,8,3) fp64; J built column-by-column with unit grad_outputs;
    # H_out = I/B; assert ggn_fn(v) == (J.T @ H @ J) @ v  (atol 1e-8)
def test_ggn_dense_ce(device)          # dense H_out blocks diag(p)-pp^T; same bar
def test_true_fisher_equals_ggn_ce(device):
    # ENUMERATED (deterministic): F = (1/B) Σ_b Σ_c p_bc g_bc g_bc^T with
    # g_bc = ∇_θ CE(logits_b, c);  assert F @ v == ggn_fn(v)  (atol 1e-6, fp64)
def test_empirical_fisher_vp_dense(device)  # matches (1/B) Σ g_i (g_i·v) built densely
```
- [ ] Implement; pass; commit `feat(curvature): hvp/ggn/fisher products with dense-verified tests`.

### Task 6: Steihaug CG + Lanczos

**Files:** Create `pytorch_opt/curvature/steihaug.py`, `pytorch_opt/curvature/lanczos.py`, `tests/test_steihaug.py`, `tests/test_lanczos.py`.

**Produces:** `steihaug_cg(mvp, g, radius, tol=None, max_iter=250) -> (s, info)`; `info = {"iters", "reason": "converged|boundary|neg_curvature|max_iter|zero_grad"}`; default `tol = min(0.5, sqrt(‖g‖))·‖g‖`; boundary crossing solves `‖s+τd‖=Δ` positive root `τ = (−s·d + sqrt((s·d)² + ‖d‖²(Δ²−‖s‖²)))/‖d‖²`. `lanczos_eigs(mvp, dim, k=6, iters=None, *, device, dtype, generator=None)` — full reorthogonalization, dense tridiag eigvalsh, returns top-k descending.

- [ ] Failing tests:
```python
def test_steihaug_solves_interior(device):  # SPD A, huge radius: s == -A^{-1} g (atol 1e-6), reason converged
def test_steihaug_boundary(device):         # radius 0.1: ||s|| == 0.1 (1e-8), reason boundary, m(s) < m(cauchy clipped)
def test_steihaug_negative_curvature(device):  # A = diag(1, -1), g along +e1: reason neg_curvature, ||s||==radius, m(s)<0
def test_lanczos_topk(device):              # dense sym 40x40: top-5 vs eigvalsh, rtol 1e-4
```
- [ ] Implement; pass; commit `feat(curvature): Steihaug-Toint CG and Lanczos spectrum`.

### Task 7: KronTracker + damping policies

**Files:** Create `pytorch_opt/curvature/kron.py`, `pytorch_opt/curvature/damping.py`, `tests/test_kron_tracker.py`.

**Produces:**
- `KronTracker(model, ema_decay=0.95, modules=(nn.Linear, nn.Conv2d))`; `.enabled` bool (default False); `.factors[name] -> {"A": Tensor, "G": Tensor}` keyed by module path from `named_modules()`; `ema_decay=None` ⇒ running mean (for tests/statistical estimates). First update assigns (no zero-bias).
  Linear: fwd pre-hook saves `a=(B,in)` detached (only when `enabled` and `module.training`); full-backward hook takes `g=grad_output[0]`, sets `ĝ=g·B`, updates `G += ĝᵀĝ/B` (EMA), `A += ãᵀã/B` with `ã=[a,1]` iff bias. Conv2d: `ã = unfold(x, kernel, dilation, padding, stride)` → `(B·L, Ck²[+1])`, `A = ãᵀã/(B·L)`; `ĝ = grad_out.permute→(B·L, O)·B`, `G = ĝᵀĝ/(B·L)`. (KFC spatial convention; calibrated by the L=1≡Linear test.)
  `.sampled_backward(outputs, kind="categorical"|"gaussian", generator=None)`: samples targets from predictive (`multinomial` on softmax / `normal(outputs, 1)`), builds mean-reduced loss (CE / mse_half), runs `autograd.grad(loss, tracked_params, retain_graph=True, allow_unused=True)` **with tracker force-enabled for just that backward**; never touches `.grad`.
- `damping.py`: `kfac_factored_damping(A, G, damping) -> (gamma_A, gamma_G)` with `pi = sqrt((tr(A)/dim_A)/(tr(G)/dim_G))` (guard: nonfinite/zero ⇒ pi=1), `gamma_A = pi·sqrt(damping)`, `gamma_G = sqrt(damping)/pi`; `lm_update(damping, rho, lo=0.25, hi=0.75, factor=0.9) -> new damping` (shrink if rho>hi, grow if rho<lo).

- [ ] Failing tests:
```python
def test_linear_factors_hand_computed(device)   # fixed batch; A,G vs manual einsum incl. bias column and B-rescale
def test_ema_vs_running_mean(device)
def test_conv_1x1_output_equals_linear(device)  # 4x4 input, 4x4 kernel, L=1: conv factors == equivalent flattened Linear factors
def test_sampled_backward_does_not_touch_grads(device)  # param.grad stays None; factors move
def test_gaussian_sampled_G_approaches_identity(device) # Linear+MSE, running mean, 600 sampled passes: ||G-I||/||I|| < 0.15
def test_pi_damping_formula(device)             # hand numbers
```
- [ ] Implement; pass; commit `feat(curvature): Kronecker factor tracker (Linear/Conv2d, empirical+sampled) and damping policies`.

### Task 8: Muon

**Files:** Create `pytorch_opt/optim/muon.py`, `pytorch_opt/optim/_common.py` (diagnostics mixin: `self._diag` dict + `.diagnostics` property + timing helper; `state_layout()` convention), `tests/test_muon.py`.

**Produces:** `Muon(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, lr_adjust="spectral", adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.0)`; per-group `use_muon: Optional[bool]` (None ⇒ route by `p.ndim >= 2`); routed: momentum buffer, nesterov mix, reshape `(shape[0], -1)`, `ops.newton_schulz_orthogonalize`, scale `spectral: max(1, m/n)**0.5 | match_rms_adam: 0.2*sqrt(max(m,n)) | none`, decoupled wd, `p -= lr·s·O`; unrouted: standard AdamW math (bias-corrected). Diagnostics: `{"update_rms", "n_muon_params", "n_adamw_params", "step_ms"}`. `state_layout()` = all shardable.

- [ ] Failing tests:
```python
def test_routing(device)              # 2D weight -> momentum_buffer; bias -> exp_avg
def test_hand_step(device):
    # one param (8,5), fixed grad; expected = p0 - lr*scale*NS(g + mu*g)  (nesterov, first step)
def test_use_muon_override(device)    # force 2D param into adamw group; state proves it
def test_conv_kernel_flattening(device)  # 4D param routed, update shape preserved
def test_converges_tiny_mlp(device)   # 150 steps regression: loss < 0.15 * initial
```
- [ ] Implement; pass; commit `feat(optim): Muon (NS5 orthogonalized momentum + AdamW routing)`.

### Task 9: Shampoo

**Files:** Create `pytorch_opt/optim/shampoo.py`, `tests/test_shampoo.py`.

**Produces:** `Shampoo(params, lr=0.03, beta2=0.999, eps=1e-8, precondition_frequency=20, start_step=0, max_preconditioner_dim=1024, graft="none"|"sgd"|"adagrad", momentum=0.0, weight_decay=0.0, root_method="auto", root_dtype=torch.float64, diag_eps=1e-10)`. ndim≥2 reshaped `(shape[0], -1)`; factors via `ops.kron_factor_update_`; roots `L^{-1/4}, R^{-1/4}` via `ops.inverse_matrix_root(·, 4, damping=eps)` recomputed when `step % precondition_frequency == 0 and step >= start_step`; stale roots reused between; before first roots ⇒ diagonal-adagrad direction. 1D or any side > max dim ⇒ diagonal path (`acc` EMA by beta2; `g/(sqrt(acc)+diag_eps)`). Grafting rescales `P` to the graft method's per-layer update norm. Momentum on preconditioned update; decoupled wd. Diagnostics: `{"stale_steps", "cond_L", "cond_R"(at refresh), "graft_ratio", "n_diag_params", "step_ms", "curvature_ms"}`. `state_layout()`: factors/roots replicable; momenta/diag shardable.

- [ ] Failing tests:
```python
def test_first_step_is_polar_factor(device):
    # THE identity test: beta2=1.0, eps=1e-12, graft none, momentum 0, freq 1:
    # update dir after 1 step == U V^T from svd(G)  (fp64, atol 1e-5)
def test_stale_roots_between_refreshes(device)   # freq=5: roots buffers identical steps 1..4, change at 5
def test_1d_and_oversize_fall_back_to_diagonal(device)  # bias + max_preconditioner_dim=4 case; matches manual adagrad
def test_grafting_sgd_norm(device)               # ||update||_layer == ||g||_layer (rtol 1e-5)
def test_converges_tiny_mlp(device)              # 200 steps: loss < 0.2 * initial
```
- [ ] Implement; pass; commit `feat(optim): Shampoo (Kronecker-preconditioned, stale-root schedule, grafting)`.

### Task 10: TrustNCG

**Files:** Create `pytorch_opt/optim/trust_ncg.py`, `tests/test_trust_ncg.py`.

**Produces:** `TrustNCG(params, delta0=1.0, delta_max=1e3, eta=0.15, shrink_threshold=0.25, expand_threshold=0.75, curvature="hessian"|"ggn", ggn_loss="mse"|"ce", damping=0.0, max_cg_iter=250, cg_tol=None)`. Closure contract (asymmetry #1, docstring + error message): closure re-evaluates and **returns loss with graph, no backward()**; for `curvature="ggn"` returns `(loss, outputs)`. Step: build `make_hvp`/`make_ggn_vp` (+`damping·v`), Steihaug solve at radius Δ, `pred = g·s + ½ s·mvp(s)`; apply `s` via ParamVector; re-eval closure under `no_grad` → ρ; reject (revert) if `ρ ≤ eta`; radius update per spec constants; Δ persisted via overridden `state_dict()/load_state_dict()` (`{"trust": {"delta": float}, "base": super()}`). `step(closure)` returns the pre-step loss. Missing closure ⇒ `TypeError("TrustNCG.step() requires a closure ...")`. Diagnostics: `{"delta", "rho", "cg_iters", "cg_reason", "accepted", "pred_reduction", "step_ms"}`. `state_layout()`: delta replicable.

- [ ] Failing tests:
```python
def test_one_step_optimum_on_quadratic(device):  # n=12 SPD, delta0=100: x1 ≈ A^{-1} b (atol 1e-5); rho≈1 accepted
def test_ggn_mode_linear_regression(device)      # GGN==H for linear+MSE: same 1-step optimum via (loss, outputs) closure
def test_rosenbrock(device)                      # from (-1.2, 1.0): <=150 steps to |x-1|<1e-4, |y-1|<1e-4
def test_escapes_saddle(device)                  # f=x^2-y^2 from (1, 1e-3): some step reason==neg_curvature; f< -0.5 eventually
def test_rejection_reverts_params(device)        # closure with a cliff: rejected step leaves params identical, delta shrunk
def test_requires_closure(device)                # TypeError with instructive message
```
- [ ] Implement; pass; commit `feat(optim): trust-region Newton-CG (Steihaug), hessian+ggn curvature`.

### Task 11: KFAC

**Files:** Create `pytorch_opt/optim/kfac.py`, `tests/test_kfac.py`.

**Produces:** `KFAC(model, lr=0.01, damping=1e-3, ema_decay=0.95, momentum=0.9, stats_every=1, inv_every=10, fisher_mode="empirical"|"sampled", weight_decay=0.0, max_grad_norm=None, sgd_lr=None)` (asymmetry #2: takes model). Tracked modules → KronTracker; empirical mode: tracker enabled with cadence gate `steps % stats_every == 0` (hooks check counter; eval-mode forwards never tracked); sampled mode: tracker only active inside `update_curvature(outputs, kind="categorical", generator=None)` (cadence-gated no-op otherwise). Per tracked module at `steps % inv_every == 0`: eigh of A and G once → damped inverses `iA=(A+γ_A I)^{-1}`, `iG=(G+γ_G I)^{-1}` with π-split; harvest `cond = (λmax+γ)/(λmin+γ)`. Update: `∇W̃ = [∇W, ∇b]` (bias column iff bias; conv grads reshaped `(out, Ck²)`), `Δ = iG @ ∇W̃ @ iA`, split back; momentum buffers on nat grads; optional global-norm clip across kfac params; decoupled wd; untracked params: SGD+momentum at `sgd_lr or lr`. Custom `state_dict()` = `{"factors": {module_path: {A, G, iA, iG}}, "steps": int, "base": super()}`; load matches by path. Diagnostics: `{"damping", "mean_cond_A", "mean_cond_G", "inv_stale_steps", "nat_grad_norm", "step_ms", "curvature_ms"}`. `state_layout()`: factors/inverses replicable; momenta shardable.

- [ ] Failing tests:
```python
def test_kron_vec_orientation():
    # torch-flatten identity: vec(iG V iA) == kron(iG, iA) @ vec(V)  (symmetric iA) — pins all conventions
def test_b1_factors_reproduce_dense_fisher(device):
    # B=1 Linear: kron(G_factor, A_factor) == vec(g a^T[aug]) outer itself (atol 1e-8, fp64)
def test_update_equals_dense_kron_solve(device):
    # one Linear, factors frozen: optimizer's Δ == unvec(kron(iG, iA) @ vec(∇W̃)) (atol 1e-8)
def test_sampled_fisher_approaches_newton(device):
    # Linear(4,3)+MSE, gaussian sampling, running-mean tracker, 800 curvature passes,
    # damping 1e-6: cosine(nat_grad, dense damped-Newton dir) > 0.99
def test_beats_sgd_on_toy_classification(device)  # 60 full-batch steps, both tuned-fixed lr: kfac loss < sgd loss and < 0.5*init
def test_conv_path_trains(device)                 # TinyCNN 40 steps: loss < 0.7*init; factor shapes correct
```
- [ ] Implement; pass; commit `feat(optim): KFAC natural gradient (factored pi-damping, empirical+sampled Fisher)`.

### Task 12: Cross-optimizer contract tests

**Files:** Create `tests/test_common.py`; modify `pytorch_opt/_testing.py` (factories).

**Produces:** `FACTORIES: dict[str, Callable[[nn.Module], Optimizer]]` for muon/shampoo/kfac/trustncg (importable by later roster tasks — extension optimizers get added here and inherit all contract tests).

- [ ] Parametrized tests (cpu-only for determinism):
```python
def test_deterministic_trajectory(name)   # same seed twice -> bitwise-equal params after 5 steps
def test_state_dict_round_trip(name)      # 3 steps, snapshot(model+opt), 3 more -> params A;
                                          # restore snapshot, 3 more -> params B; A == B exactly
def test_diagnostics_schema(name)         # required keys per optimizer present after a step
def test_state_layout_declared(name)      # every state key tagged replicable|shardable
```
- [ ] Implement (fix optimizers where round-trip fails — likely spots: TrustNCG delta, KFAC factors); commit `test: cross-optimizer determinism, round-trip, diagnostics contracts`.

### Task 13: Native Tier A extension + parity

**Files:** Create `pytorch_opt/ops/csrc/pytorch_opt_ops.cpp`, flesh out `pytorch_opt/ops/native_build.py`, `tests/test_ops_native_parity.py`.

**Produces:** C++/ATen implementations of `inverse_matrix_root` (eigh + newton, fp64 internal, same guards), `newton_schulz_orthogonalize`, `kron_factor_update_`, `precond_apply_two_sided`, pybind'd as `pytorch_opt_native`; loader compiles with `extra_cflags=["-O3"]`, no nvcc (ATen dispatch covers CUDA tensors). `load_native()` wires module into dispatch; `set_backend("native")` errors if not loaded.

- [ ] Parity tests: session-scoped fixture attempts `load_native()`; `pytest.skip` with the build error if unavailable. For each op × {cpu, cuda} × dtypes {f32 in/f64 internal}: native == reference (atol 1e-7 on fp64 paths, 1e-5 on NS fp32). Plus `test_auto_backend_prefers_native`.
- [ ] Implement; compile; pass incl. on the 5090; commit `feat(ops): ATen C++ native backend with parity proof`.
- [ ] Time-boxed (≤45 min) stretch AFTER v1 gate: pip `nvidia-cuda-nvcc-cu13` toolchain probe for Tier B fused CG kernel; abandon cleanly on any friction (document outcome in README).

### Task 14: v1 gate — full verification + docs

- [ ] `pytest -q` full suite green, CPU and CUDA paths; record runtime.
- [ ] README: install, quickstart per optimizer (incl. closure/model asymmetries), roster status table, native-build notes, test instructions.
- [ ] Update spec status line; commit `docs: README quickstart + v1 status`.
- [ ] **Gate:** only after this may roster extension begin.

### Tasks 15+: Extension roster (same bar: analytic test + convergence + contract tests + commit each)

Priority order, each its own task/commit, each added to `FACTORIES`:

1. **NGD (exact)** `optim/ngd.py` — dense `F̂ = (1/B) GᵀG` from `per_sample_grads`; solve `(F̂+λI)d = g` (cholesky). Tests: linear-Gaussian closed-form natural gradient match; doubles as oracle for KFAC comparisons.
2. **SOAP** `optim/soap.py` — Shampoo factors' eigenbases `Q_L,Q_R` (refresh every `precondition_frequency`); Adam moments kept in rotated space, re-projected on refresh (`m̃ ← Q_newᵀ Q_old m̃`). Analytic test: rotations forced to identity ⇒ **exactly Adam** (compare vs torch.optim.Adam trajectories); plus convergence.
3. **AdaHessian** `optim/adahessian.py` — Hutchinson `D = E[z ⊙ Hz]` (Rademacher, `make_hvp` on closure-supplied loss... uses backward-through-graph: requires `loss.backward(create_graph=True)` loop convention — document), spatial averaging for conv dims, Adam-style with `D²`. Analytic test: diagonal quadratic ⇒ `z⊙Hz` exact diag (deterministic since z²=1).
4. **Sophia** `optim/sophia.py` — Sophia-H (Hutchinson) + Sophia-G (sampled GNB via tracker-style extra backward); update `clip(m/max(γ·h, eps), 1)·lr`. Tests: clip saturation behavior on quadratic; convergence.
5. **EKFAC** `optim/ekfac.py` — KFAC eigenbases + per-component second moments `s`; `Δ = Q_G (∇̃_eig / (s+λ)) Q_Aᵀ`. Test: with s set from factors' eigenvalue products EKFAC == KFAC; Frobenius approximation ≤ KFAC's on toy Fisher; convergence.
6. **HessianFree** `optim/hessianfree.py` — Martens CG-Newton: GGN default, LM damping via `lm_update`, CG backtracking over recorded iterates, warm-start `0.95·s_prev`. Tests: quadratic one-step; Rosenbrock; damping adapts (recorded).
7. **PSGD-Kron** `optim/psgd.py` — Xi-Lin Li whitening variant. Test: on static Gaussian gradients preconditioner whitens (`E[(Pg)(Pg)ᵀ] → I` loosely); convergence. (Lowest priority; OK to defer with a stated reason.)

Each roster task follows the Task-8 pattern: failing tests → implement → pass → commit; `test_common.py` picks them up via `FACTORIES` automatically.

### Task N-final: Morning report

- [ ] `docs/2026-08-26-overnight-report.md`: what shipped, proof (test counts + names of analytic tests + runtime + devices), native-build outcome, roster status, deviations from plan with reasons, suggested next steps. Final commit. Leave branch `v1` unmerged; report offers merge command.

## Self-review notes (done at write time)

- Spec coverage: L0 ops ✓ (T2–4, 13), L1 ✓ (T1, 5–7), spine ✓ (T8–11), asymmetries ✓ (T10/T11 tests), Muon routing ✓, distributed tags ✓ (T12), diagnostics ✓ (per-task + T12), determinism/round-trip ✓ (T12), packaging/no-compiler import ✓ (T0/T4), native parity ✓ (T13), roster ✓ (T15+), Tier B honestly gated ✓.
- `cg_fused_ops` from spec: reference path is plain torch inside `steihaug.py` (no separate op needed until Tier B exists) — deliberate simplification, noted here.
- `symeig_psd` from spec: `torch.linalg.eigh` used directly (already cuSOLVER-backed); wrapping adds nothing — deliberate simplification.
- Type consistency: `ops.*` signatures identical across reference/native/dispatch; `FACTORIES` names match test params; ParamVector API used by TrustNCG matches T1.
