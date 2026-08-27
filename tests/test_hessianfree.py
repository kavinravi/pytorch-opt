import torch
from torch import nn

from pytorch_opt._testing import TinyMLP, make_regression, mse_half, run_steps
from pytorch_opt.optim.hessianfree import HessianFree


def test_one_step_optimum_on_quadratic(device):
    torch.manual_seed(0)
    n = 10
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device=device))
    A = Q @ torch.diag(torch.linspace(0.5, 3.0, n, dtype=torch.float64, device=device)) @ Q.T
    b = torch.randn(n, dtype=torch.float64, device=device)
    x = nn.Parameter(torch.zeros(n, dtype=torch.float64, device=device))
    opt = HessianFree([x], curvature="hessian", damping=1e-10, cg_tol=1e-13,
                      max_cg_iter=200)

    def closure():
        return 0.5 * x @ A @ x - b @ x

    opt.step(closure)
    assert torch.allclose(x.detach(), torch.linalg.solve(A, b), atol=1e-6)
    assert opt.diagnostics["accepted"]


def test_ggn_linear_regression_one_step(device):
    torch.manual_seed(1)
    X = torch.randn(20, 5, dtype=torch.float64, device=device)
    w_true = torch.randn(5, 1, dtype=torch.float64, device=device)
    y = X @ w_true
    model = nn.Linear(5, 1, bias=False).double().to(device)
    opt = HessianFree(model.parameters(), damping=1e-10, cg_tol=1e-13, max_cg_iter=200)

    def closure():
        out = model(X)
        return mse_half(out, y), out

    opt.step(closure)
    assert torch.allclose(model.weight.detach().T, w_true, atol=1e-5)


def test_damping_adapts_and_rosenbrock_descends(device):
    x = nn.Parameter(torch.tensor([-1.2, 1.0], dtype=torch.float64, device=device))
    opt = HessianFree([x], curvature="hessian", damping=1.0, max_cg_iter=50)

    def closure():
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    lams = []
    for _ in range(60):
        opt.step(closure)
        lams.append(opt.diagnostics["damping"])
    assert len(set(lams)) > 1                     # LM adaptation actually moved
    with torch.no_grad():
        assert float(closure()) < 1e-3            # descended essentially to optimum


def test_warm_start_stored(device):
    torch.manual_seed(2)
    model = TinyMLP(4, 6, 1).to(device)
    X, y = make_regression(n=32, d=4, device=device)
    opt = HessianFree(model.parameters(), max_cg_iter=20)

    def closure():
        out = model(X)
        return mse_half(out, y), out

    assert opt._prev_s is None
    opt.step(closure)
    assert opt._prev_s is not None and opt._prev_s.norm() > 0


def test_converges_tiny_mlp(device):
    torch.manual_seed(0)
    model = TinyMLP().to(device)
    X, y = make_regression(device=device)
    opt = HessianFree(model.parameters(), max_cg_iter=30)
    losses = run_steps(model, opt, X, y, mse_half, 25)
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
