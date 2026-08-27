import torch
from torch import nn

from pytorch_opt._testing import TinyMLP, make_regression, mse_half, run_steps
from pytorch_opt.optim.adahessian import AdaHessian


def test_diag_exact_on_diagonal_quadratic(device):
    # For H = diag(d): z * (H z) = d * z^2 = d exactly, for ANY Rademacher z.
    d = torch.tensor([0.5, 2.0, 3.5, 1.25], dtype=torch.float64, device=device)
    x = nn.Parameter(torch.randn(4, dtype=torch.float64, device=device))
    opt = AdaHessian([x], lr=0.1)

    def closure():
        return 0.5 * (d * x * x).sum()

    opt.step(closure)
    assert torch.allclose(opt.state[x]["hess_diag"], d, atol=1e-10)


def test_first_step_is_scaled_newton(device):
    d = torch.tensor([0.5, 2.0, 3.5, 1.25], dtype=torch.float64, device=device)
    x = nn.Parameter(torch.randn(4, dtype=torch.float64, device=device))
    x0 = x.detach().clone()
    lr, eps = 0.1, 1e-8
    opt = AdaHessian([x], lr=lr, eps=eps)

    def closure():
        return 0.5 * (d * x * x).sum()

    opt.step(closure)
    g = d * x0
    want = x0 - lr * g / (d + eps)      # bias-corrected m=g, v=d^2
    assert torch.allclose(x.detach(), want, atol=1e-8)


def test_conv_spatial_averaging(device):
    conv = nn.Conv2d(2, 3, 3).to(device)
    X = torch.randn(4, 2, 6, 6, device=device)
    opt = AdaHessian(conv.parameters(), lr=0.01)

    def closure():
        return conv(X).pow(2).mean()

    opt.step(closure)
    hd = opt.state[conv.weight]["hess_diag"]
    assert hd.shape == (3, 2, 1, 1)     # averaged over the 3x3 kernel dims


def test_converges_tiny_mlp(device):
    torch.manual_seed(0)
    model = TinyMLP().to(device)
    X, y = make_regression(device=device)
    opt = AdaHessian(model.parameters(), lr=0.05)
    losses = run_steps(model, opt, X, y, mse_half, 150)
    assert losses[-1] < 0.15 * losses[0], (losses[0], losses[-1])
