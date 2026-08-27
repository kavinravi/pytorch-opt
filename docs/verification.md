# Verification

The test suite is built on three principles, in priority order: **analytic
ground truth** (closed-form identities an implementation must reproduce),
**parity** (the compiled backend must match the pure-torch reference, which is
the ground truth by definition), and **contracts** (determinism, persistence,
diagnostics — properties every optimizer must satisfy). "Loss goes down" is
the weakest evidence used, and never the only evidence.

Run everything with:

```bash
python -m pytest -q
```

CUDA-capable machines automatically run the suite on both CPU and GPU via a
parametrized device fixture.

## Analytic identities per optimizer

| optimizer | identity proven in tests |
|---|---|
| `Muon` | single step reproduces the hand-computed nesterov + Newton–Schulz + spectral-scaling update; NS output singular values in the design band; (semi-)orthogonal input maps to an exact scalar multiple of itself |
| `Shampoo` | with sum accumulation and vanishing damping, the **first preconditioned gradient equals `UVᵀ`**, the polar factor of the gradient (`L^{-1/4} G R^{-1/4} = US^{-1/2}·S·S^{-1/2}Vᵀ`) |
| `KFAC` | update equals the dense Kronecker solve `unvec(kron(iG, iA)·vec(∇))`; at batch size 1 the factors reproduce the dense Fisher exactly (`kron(G, A) = ffᵀ`); with sampled (true-Fisher) statistics on a linear-Gaussian model the natural gradient converges to the Newton direction (cosine > 0.99) |
| `EKFAC` | with scalings frozen to the factor-eigenvalue outer product and zero damping, **EKFAC reduces exactly to K-FAC**; the optimal eigenbasis diagonal is never worse than the Kronecker product in Frobenius norm (George et al.) |
| `NGD` | on a linear-Gaussian model (Fisher ≡ Hessian) one step with `lr=1` lands on the least-squares optimum |
| `TrustNCG` | solves a convex quadratic to optimality in one outer step (ρ ≈ 1); converges on Rosenbrock; escapes the `x²−y²` saddle through the negative-curvature branch; a rejected step reverts parameters bit-exactly and shrinks the radius |
| `HessianFree` | one-step quadratic optimum; LM damping demonstrably adapts; CG backtracking selects an improving iterate |
| `AdaHessian` | on `H = diag(d)` the Hutchinson estimate is **exact for any Rademacher probe** (`z⊙Hz = d`), making the first step a hand-checkable scaled-Newton step; conv kernels get spatially averaged estimates |
| `Sophia` | exact diagonal on diagonal quadratics; hand-computed first step with elementwise clipping engaged; GNB estimates are nonnegative with correct shapes |
| `SOAP` | with diagonal Kronecker factors (eigenbases = signed permutations, which elementwise Adam commutes with) SOAP's trajectory is **exactly Adam's**, including across basis refreshes — this pins the second-moment transport `v ← (C_L²)ᵀ v (C_R²)`; with non-diagonal factors it measurably differs from Adam |
| `PSGD` | fed gradients with a known Kronecker covariance, the fitted preconditioner **whitens**: `E[(PG)(PG)ᵀ] ∝ I` to within tolerance, several times closer to identity than the raw covariance |

## Curvature-layer proofs

- `hvp` equals `Av` on quadratics and matches central finite differences on an
  MLP.
- `ggn_vp` equals the densely assembled `JᵀH_outJ` (Jacobian built row by row)
  for both MSE and CE.
- True Fisher ≡ GGN for CE verified by **enumerating** the label distribution
  (no sampling noise).
- `steihaug_cg` solves interior problems to the CG limit, lands exactly on the
  boundary when constrained, and returns descent directions under negative
  curvature (never worse than the clipped Cauchy point).
- `lanczos_eigs` at full dimension reproduces `eigvalsh`.
- Kronecker trackers match hand-computed factor moments; the Conv2d path at
  1×1 spatial output equals the equivalent Linear layer exactly (calibrating
  the KFC scaling convention).

## Backend parity

Every native (C++/ATen) op is compared against its reference implementation on
random inputs — batched and unbatched, CPU and CUDA — at tolerances set by the
float64 internal path. If the native extension cannot be built, parity tests
skip and everything else runs on the reference backend.

## Cross-optimizer contracts

Parametrized over every optimizer in the roster:

- **Determinism** — two runs from the same seed produce bitwise-identical
  parameters (stochastic estimators use internal seeded generators).
- **Round-trip** — train, snapshot (`state_dict` of model + optimizer), train
  further; restoring the snapshot and repeating produces identical parameters.
- **Diagnostics schema** — each optimizer's `.diagnostics` exposes its
  documented keys after a step.
- **State layout** — every state key an optimizer creates is tagged
  `replicable` or `shardable` by `state_layout()`.
