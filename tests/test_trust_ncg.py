import pytest
import torch
from torch import nn

from pytorch_opt._testing import mse_half
from pytorch_opt.optim import TrustNCG


def test_requires_closure(device):
    opt = TrustNCG([nn.Parameter(torch.zeros(2, device=device))])
    with pytest.raises(TypeError, match="requires a closure"):
        opt.step()


def test_one_step_optimum_on_quadratic(device):
    torch.manual_seed(0)
    n = 12
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device=device))
    A = Q @ torch.diag(torch.linspace(0.5, 3.0, n, dtype=torch.float64, device=device)) @ Q.T
    b = torch.randn(n, dtype=torch.float64, device=device)
    x = nn.Parameter(torch.zeros(n, dtype=torch.float64, device=device))
    opt = TrustNCG([x], delta0=100.0, cg_tol=1e-12, max_cg_iter=200)

    def closure():
        return 0.5 * x @ A @ x - b @ x

    opt.step(closure)
    assert torch.allclose(x.detach(), torch.linalg.solve(A, b), atol=1e-6)
    d = opt.diagnostics
    assert d["accepted"] and abs(d["rho"] - 1.0) < 1e-6


def test_ggn_mode_linear_regression(device):
    torch.manual_seed(1)
    X = torch.randn(20, 5, dtype=torch.float64, device=device)
    w_true = torch.randn(5, 1, dtype=torch.float64, device=device)
    y = X @ w_true
    model = nn.Linear(5, 1, bias=False).double().to(device)
    opt = TrustNCG(model.parameters(), delta0=100.0, curvature="ggn",
                   ggn_loss="mse", cg_tol=1e-12, max_cg_iter=200)

    def closure():
        out = model(X)
        return mse_half(out, y), out

    opt.step(closure)
    assert torch.allclose(model.weight.detach().T, w_true, atol=1e-5)


def test_rosenbrock(device):
    x = nn.Parameter(torch.tensor([-1.2, 1.0], dtype=torch.float64, device=device))
    opt = TrustNCG([x], delta0=1.0)

    def closure():
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    for _ in range(150):
        opt.step(closure)
    assert torch.allclose(x.detach(), torch.ones(2, dtype=torch.float64, device=device), atol=1e-4)


def test_escapes_saddle(device):
    x = nn.Parameter(torch.tensor([1.0, 1e-3], dtype=torch.float64, device=device))
    opt = TrustNCG([x], delta0=1.0)

    def closure():
        return x[0] ** 2 - x[1] ** 2

    reasons = []
    for _ in range(6):
        opt.step(closure)
        reasons.append(opt.diagnostics["cg_reason"])
    assert "neg_curvature" in reasons
    with torch.no_grad():
        assert float(closure()) < -0.5


def test_rejection_reverts_params(device):
    x = nn.Parameter(torch.zeros(1, dtype=torch.float64, device=device))
    opt = TrustNCG([x], delta0=100.0, cg_tol=1e-12)

    def closure():
        return ((x - 5.0) ** 2 + 50.0 * torch.relu(x - 1.0)).sum()

    opt.step(closure)
    d = opt.diagnostics
    assert not d["accepted"]
    assert torch.equal(x.detach(), torch.zeros(1, dtype=torch.float64, device=device))
    assert d["delta"] < 100.0
    for _ in range(30):
        opt.step(closure)
    assert abs(float(x.detach()) - 1.0) < 0.05, float(x.detach())
    with torch.no_grad():
        assert abs(float(closure()) - 16.0) < 0.5
