import torch
from torch import nn

from pytorch_opt._testing import TinyMLP, make_regression, mse_half, run_steps
from pytorch_opt.optim.psgd import PSGD, _precond_grad


def test_whitening_property(device):
    # Gradients G = S_L Z S_R with iid normal Z: at the whitening optimum the
    # preconditioned gradients are white, E[(PG)(PG)^T] ~ c*I. Run the Q
    # updates on a stream of such gradients and measure whiteness.
    torch.manual_seed(0)
    m, n = 6, 5
    S_L = torch.diag(torch.linspace(0.3, 3.0, m)).to(device)
    S_R = torch.diag(torch.linspace(0.5, 2.0, n)).to(device)
    p = nn.Parameter(torch.zeros(m, n, device=device))
    opt = PSGD([p], lr=0.0, precond_lr=0.1, momentum=0.0, seed=1)
    gen = torch.Generator().manual_seed(2)

    def sample_grad():
        Z = torch.randn(m, n, generator=gen).to(device)
        return S_L @ Z @ S_R

    for _ in range(1500):
        p.grad = sample_grad()
        opt.step()
    opt.param_groups[0]["precond_lr"] = 0.02   # anneal to tighten the equilibrium
    for _ in range(1500):
        p.grad = sample_grad()
        opt.step()

    st = opt.state[p]
    cov = torch.zeros(m, m, device=device)
    for _ in range(500):
        U = _precond_grad(st["Ql"], st["Qr"], sample_grad())
        cov += U @ U.T / 500
    cov /= cov.diagonal().mean()
    I = torch.eye(m, device=device)
    white_err = (cov - I).norm() / I.norm()
    raw = S_L @ S_L.T * float((S_R @ S_R.T).trace())    # unpreconditioned covariance
    raw = raw / raw.diagonal().mean()
    raw_err = (raw - I).norm() / I.norm()
    assert float(white_err) < 0.25, float(white_err)
    assert float(white_err) < 0.3 * float(raw_err)


def test_factors_stay_triangular(device):
    torch.manual_seed(1)
    p = nn.Parameter(torch.randn(5, 4, device=device))
    opt = PSGD([p], lr=0.01)
    for _ in range(5):
        p.grad = torch.randn(5, 4, device=device)
        opt.step()
    st = opt.state[p]
    assert torch.equal(st["Ql"], torch.triu(st["Ql"]))
    assert torch.equal(st["Qr"], torch.triu(st["Qr"]))


def test_1d_params_get_column_factors(device):
    # small 1D params are treated as (n, 1) columns with dense factors;
    # the diagonal path is reserved for sides above max_preconditioner_dim
    b = nn.Parameter(torch.randn(7, device=device))
    opt = PSGD([b], lr=0.01)
    b.grad = torch.randn(7, device=device)
    opt.step()
    st = opt.state[b]
    assert st["Ql"].shape == (7, 7) and st["Qr"].shape == (1, 1)

    big = nn.Parameter(torch.randn(30, device=device))
    opt2 = PSGD([big], lr=0.01, max_preconditioner_dim=8)
    big.grad = torch.randn(30, device=device)
    opt2.step()
    st2 = opt2.state[big]
    assert st2["Ql"].shape == (30,) and st2["Qr"].shape == (1,)
    assert opt2.diagnostics["n_diag_params"] == 1


def test_converges_tiny_mlp(device):
    torch.manual_seed(0)
    model = TinyMLP().to(device)
    X, y = make_regression(device=device)
    opt = PSGD(model.parameters(), lr=0.05, momentum=0.9)
    losses = run_steps(model, opt, X, y, mse_half, 200)
    assert losses[-1] < 0.15 * losses[0], (losses[0], losses[-1])
