// pytorch-opt native ops (Tier A): C++/ATen implementations of the shared
// numerics. ATen dispatches by tensor device, so this single compiled
// extension serves CPU and CUDA tensors (cuBLAS/cuSOLVER underneath) without
// any hand-written kernels. Semantics mirror pytorch_opt/ops/reference.py,
// which is the ground truth (see parity tests).

#include <torch/extension.h>

#include <string>

namespace {

at::Tensor sym(const at::Tensor& A) { return 0.5 * (A + A.mT()); }

at::Tensor coupled_newton(const at::Tensor& M, int64_t p, int64_t max_iter, double tol) {
  const auto n = M.size(-1);
  const auto I = at::eye(n, M.options()).expand_as(M);
  const double alpha = -1.0 / static_cast<double>(p);
  auto fro = M.pow(2).sum(at::IntArrayRef{-2, -1}, /*keepdim=*/true).sqrt();
  auto z = (1.0 + static_cast<double>(p)) / (2.0 * fro.clamp_min(1e-300));
  auto X = I * z.pow(-alpha);
  auto Mk = M * z;
  at::Tensor prevX = X;
  double prev_err = -1.0;
  for (int64_t i = 0; i < max_iter; ++i) {
    auto Mi = (1.0 - alpha) * I + alpha * Mk;
    X = at::matmul(X, Mi);
    Mk = at::matmul(at::linalg_matrix_power(Mi, p), Mk);
    const double err = (Mk - I).abs().max().item<double>();
    if (prev_err >= 0.0 && err > prev_err * 1.2) {  // diverging: keep last good
      X = prevX;
      break;
    }
    prevX = X;
    prev_err = err;
    if (err < tol) break;
  }
  return X;
}

at::Tensor inverse_matrix_root(const at::Tensor& A, int64_t p, double damping,
                               const std::string& method, bool root_f64,
                               int64_t max_iter, double tol) {
  TORCH_CHECK(p >= 1, "p must be a positive integer, got ", p);
  TORCH_CHECK(method == "auto" || method == "eigh" || method == "newton",
              "method must be auto|eigh|newton, got ", method);
  const auto orig = A.scalar_type();
  auto M = sym(A.to(root_f64 ? at::kDouble : at::kFloat));
  if (damping != 0.0) {
    M = M + damping * at::eye(M.size(-1), M.options()).expand_as(M);
  }
  at::Tensor root;
  if (method == "newton") {
    root = coupled_newton(M, p, max_iter, tol);
  } else {
    auto eig = at::linalg_eigh(M, "L");
    auto evals = std::get<0>(eig).clamp_min(1e-30);
    auto evecs = std::get<1>(eig);
    root = at::matmul(at::matmul(evecs, at::diag_embed(evals.pow(-1.0 / static_cast<double>(p)))),
                      evecs.mT());
  }
  return sym(root).to(orig);
}

at::Tensor newton_schulz_orthogonalize(const at::Tensor& G, int64_t steps,
                                       double a, double b, double c, double eps) {
  TORCH_CHECK(G.dim() >= 2, "newton_schulz_orthogonalize expects ndim >= 2");
  const auto work = (G.scalar_type() == at::kDouble) ? at::kDouble : at::kFloat;
  auto X = G.to(work);
  const bool transposed = X.size(-2) > X.size(-1);
  if (transposed) X = X.mT();
  auto nrm = X.pow(2).sum(at::IntArrayRef{-2, -1}, /*keepdim=*/true).sqrt();
  X = X / (nrm + eps);
  for (int64_t i = 0; i < steps; ++i) {
    auto A_ = at::matmul(X, X.mT());
    auto B_ = b * A_ + c * at::matmul(A_, A_);
    X = a * X + at::matmul(B_, X);
  }
  if (transposed) X = X.mT();
  return X.to(G.scalar_type());
}

void kron_factor_update_(at::Tensor L, at::Tensor R, const at::Tensor& G, double beta2) {
  TORCH_CHECK(G.dim() == 2, "G must be 2D");
  auto GGt = at::matmul(G, G.mT());
  auto GtG = at::matmul(G.mT(), G);
  if (beta2 == 1.0) {
    L.add_(GGt);
    R.add_(GtG);
  } else {
    L.mul_(beta2).add_(GGt, 1.0 - beta2);
    R.mul_(beta2).add_(GtG, 1.0 - beta2);
  }
}

at::Tensor precond_apply_two_sided(const at::Tensor& L_root, const at::Tensor& G,
                                   const at::Tensor& R_root) {
  return at::matmul(at::matmul(L_root, G), R_root);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inverse_matrix_root", &inverse_matrix_root,
        "(A + damping I)^(-1/p), symmetric PSD, batched");
  m.def("newton_schulz_orthogonalize", &newton_schulz_orthogonalize,
        "quintic Newton-Schulz polar-factor approximation");
  m.def("kron_factor_update_", &kron_factor_update_,
        "in-place Kronecker factor accumulation");
  m.def("precond_apply_two_sided", &precond_apply_two_sided, "L @ G @ R");
}
