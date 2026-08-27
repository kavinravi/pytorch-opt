import torch
import torch.nn.functional as F
from torch import nn

from pytorch_opt._testing import TinyMLP, make_classification, make_regression, mse_half, run_steps
from pytorch_opt.optim.sophia import Sophia


def test_hutchinson_diag_exact_on_diagonal_quadratic(device):
    d = torch.tensor([0.5, 2.0, 3.5, 1.25], dtype=torch.float64, device=device)
    x = nn.Parameter(torch.randn(4, dtype=torch.float64, device=device))
    opt = Sophia([x], lr=0.01)

    def closure():
        return 0.5 * (d * x * x).sum()

    opt.step(closure)
    assert torch.allclose(opt.state[x]["hess_diag"], d, atol=1e-10)


def test_first_step_hand_computed_with_clipping(device):
    d = torch.tensor([1.0, 4.0], dtype=torch.float64, device=device)
    x = nn.Parameter(torch.tensor([10.0, 0.001], dtype=torch.float64, device=device))
    x0 = x.detach().clone()
    lr, b1, gamma, eps = 0.01, 0.96, 0.05, 1e-12
    opt = Sophia([x], lr=lr, betas=(b1, 0.99), gamma=gamma, eps=eps)

    def closure():
        return 0.5 * (d * x * x).sum()

    opt.step(closure)
    g = d * x0
    m = (1 - b1) * g
    want = x0 - lr * (m / (gamma * d).clamp_min(eps)).clamp(-1.0, 1.0)
    assert torch.allclose(x.detach(), want, atol=1e-10)
    assert opt.diagnostics["clip_fraction"] > 0    # the x=10 coord clips


def test_gnb_estimator_shapes_and_positivity(device):
    torch.manual_seed(0)
    model = TinyMLP(6, 8, 3).to(device)
    X, y = make_classification(n=32, d=6, k=3, device=device)
    opt = Sophia(model.parameters(), lr=5e-3, estimator="gnb", estimate_freq=1)

    def closure():
        out = model(X)
        return F.cross_entropy(out, y), out

    opt.step(closure)
    for p in model.parameters():
        h = opt.state[p]["hess_diag"]
        assert h.shape == p.shape and (h >= 0).all()


def test_converges_tiny_mlp(device):
    torch.manual_seed(0)
    model = TinyMLP().to(device)
    X, y = make_regression(device=device)
    opt = Sophia(model.parameters(), lr=2e-2, estimate_freq=5)
    losses = run_steps(model, opt, X, y, mse_half, 200)
    assert losses[-1] < 0.25 * losses[0], (losses[0], losses[-1])
