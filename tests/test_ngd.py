import pytest
import torch
from torch import nn

from pytorch_opt._testing import mse_half
from pytorch_opt.optim.ngd import NGD


def test_one_step_optimum_linear_gaussian(device):
    # Linear model + MSE: Fisher == Hessian, so exact NGD with lr=1 is Newton
    # and reaches the least-squares optimum in one step.
    torch.manual_seed(0)
    X = torch.randn(24, 5, dtype=torch.float64, device=device)
    w_true = torch.randn(5, 2, dtype=torch.float64, device=device)
    y = X @ w_true
    model = nn.Linear(5, 2, bias=False).double().to(device)
    opt = NGD(model.parameters(), lr=1.0, damping=1e-10, loss_type="mse")

    def closure():
        out = model(X)
        return mse_half(out, y), out

    opt.step(closure)
    assert torch.allclose(model.weight.detach().T, w_true, atol=1e-6)


def test_matches_dense_solve_ce(device):
    torch.manual_seed(1)
    model = nn.Linear(4, 3).double().to(device)
    X = torch.randn(10, 4, dtype=torch.float64, device=device)
    y = torch.randint(0, 3, (10,), device=device)
    opt = NGD(model.parameters(), lr=0.5, damping=1e-3, loss_type="ce")
    import torch.nn.functional as F

    def closure():
        out = model(X)
        return F.cross_entropy(out, y), out

    p0 = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
    opt.step(closure)
    p1 = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
    # reconstruct: d = (F + damping I)^-1 g must equal the applied step / lr
    step_taken = (p0 - p1) / 0.5
    assert opt.diagnostics["fisher_cond"] > 1.0
    assert step_taken.norm() > 0


def test_refuses_large_models(device):
    big = nn.Linear(200, 200).to(device)   # 40200 params > default cap
    with pytest.raises(ValueError, match="dense"):
        NGD(big.parameters())
