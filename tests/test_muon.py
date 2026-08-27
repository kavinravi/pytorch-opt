import torch
from torch import nn

from pytorch_opt import ops
from pytorch_opt._testing import TinyMLP, make_regression, mse_half, run_steps
from pytorch_opt.optim import Muon


def test_routing(device):
    model = nn.Linear(4, 3).to(device)
    opt = Muon(model.parameters())
    mse_half(model(torch.randn(5, 4, device=device)), torch.randn(5, 3, device=device)).backward()
    opt.step()
    assert "momentum_buffer" in opt.state[model.weight]
    assert "exp_avg" in opt.state[model.bias]
    d = opt.diagnostics
    assert d["n_muon_params"] == 1 and d["n_adamw_params"] == 1


def test_hand_step(device):
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(8, 5, device=device))
    p0 = p.detach().clone()
    g = torch.randn(8, 5, device=device)
    p.grad = g.clone()
    lr, mu = 0.02, 0.95
    opt = Muon([p], lr=lr, momentum=mu, nesterov=True)
    opt.step()
    u = g + mu * g                      # first step: buf == g, nesterov mix
    O = ops.newton_schulz_orthogonalize(u)
    scale = max(1.0, 8 / 5) ** 0.5
    want = p0 - lr * scale * O
    assert torch.allclose(p.detach(), want, atol=1e-7)


def test_use_muon_override(device):
    w = nn.Parameter(torch.randn(6, 6, device=device))
    opt = Muon([{"params": [w], "use_muon": False}])
    w.grad = torch.randn_like(w)
    opt.step()
    assert "exp_avg" in opt.state[w] and "momentum_buffer" not in opt.state[w]


def test_conv_kernel_flattening(device):
    w = nn.Parameter(torch.randn(4, 3, 3, 3, device=device))
    w.grad = torch.randn_like(w)
    opt = Muon([w])
    opt.step()
    assert opt.state[w]["momentum_buffer"].shape == w.shape
    assert opt.diagnostics["n_muon_params"] == 1


def test_converges_tiny_mlp(device):
    torch.manual_seed(0)
    model = TinyMLP().to(device)
    X, y = make_regression(device=device)
    opt = Muon(model.parameters(), lr=0.02, adamw_lr=3e-3)
    losses = run_steps(model, opt, X, y, mse_half, 150)
    assert losses[-1] < 0.15 * losses[0], (losses[0], losses[-1])
