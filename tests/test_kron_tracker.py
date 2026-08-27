import torch
import torch.nn.functional as F
from torch import nn

from pytorch_opt._testing import mse_half
from pytorch_opt.curvature.damping import kfac_factored_damping, lm_update
from pytorch_opt.curvature.kron import KronTracker


def test_linear_factors_hand_computed(device):
    torch.manual_seed(1)
    lin = nn.Linear(3, 2).to(device)
    X = torch.randn(4, 3, device=device)
    y = torch.randn(4, 2, device=device)
    tr = KronTracker(lin)
    with tr.track():
        out = lin(X)
        loss = mse_half(out, y)
        loss.backward()
    a_aug = torch.cat([X, X.new_ones(4, 1)], dim=1)
    A_want = a_aug.T @ a_aug / 4
    ghat = out.detach() - y            # B * dL/dout for mse_half
    G_want = ghat.T @ ghat / 4
    assert torch.allclose(tr.factors[""]["A"], A_want, atol=1e-6)
    assert torch.allclose(tr.factors[""]["G"], G_want, atol=1e-6)


def _one_pass(lin, X, y, tr):
    with tr.track():
        loss = mse_half(lin(X), y)
        g = torch.autograd.grad(loss, list(lin.parameters()))
    return g


def test_ema_vs_running_mean(device):
    torch.manual_seed(2)
    lin = nn.Linear(3, 2, bias=False).to(device)
    X1, y1 = torch.randn(4, 3, device=device), torch.randn(4, 2, device=device)
    X2, y2 = torch.randn(4, 3, device=device), torch.randn(4, 2, device=device)

    def batch_A(X):
        return X.T @ X / 4

    ema = KronTracker(lin, ema_decay=0.9)
    _one_pass(lin, X1, y1, ema)
    _one_pass(lin, X2, y2, ema)
    want = 0.9 * batch_A(X1) + 0.1 * batch_A(X2)
    assert torch.allclose(ema.factors[""]["A"], want, atol=1e-6)

    rm = KronTracker(lin, ema_decay=None)
    _one_pass(lin, X1, y1, rm)
    _one_pass(lin, X2, y2, rm)
    want = 0.5 * (batch_A(X1) + batch_A(X2))
    assert torch.allclose(rm.factors[""]["A"], want, atol=1e-6)


def test_conv_1x1_output_equals_linear(device):
    torch.manual_seed(3)
    conv = nn.Conv2d(2, 5, kernel_size=4).to(device)   # 4x4 input -> 1x1 output
    lin = nn.Linear(32, 5).to(device)
    with torch.no_grad():
        lin.weight.copy_(conv.weight.reshape(5, -1))
        lin.bias.copy_(conv.bias)
    X = torch.randn(3, 2, 4, 4, device=device)
    y = torch.randn(3, 5, device=device)
    tc, tl = KronTracker(conv), KronTracker(lin)
    with tc.track():
        loss = mse_half(conv(X).reshape(3, 5), y)
        torch.autograd.grad(loss, list(conv.parameters()))
    with tl.track():
        loss = mse_half(lin(X.reshape(3, -1)), y)
        torch.autograd.grad(loss, list(lin.parameters()))
    assert torch.allclose(tc.factors[""]["A"], tl.factors[""]["A"], atol=1e-5)
    assert torch.allclose(tc.factors[""]["G"], tl.factors[""]["G"], atol=1e-5)


def test_sampled_backward_does_not_touch_grads(device):
    torch.manual_seed(4)
    lin = nn.Linear(4, 3).to(device)
    tr = KronTracker(lin)
    out = lin(torch.randn(6, 4, device=device))
    gen = torch.Generator().manual_seed(0)
    tr.sampled_backward(out, kind="categorical", generator=gen)
    assert all(p.grad is None for p in lin.parameters())
    assert "" in tr.factors and tr.factors[""]["A"].shape == (5, 5)
    assert tr.factors[""]["G"].shape == (3, 3)


def test_gaussian_sampled_G_approaches_identity(device):
    torch.manual_seed(5)
    lin = nn.Linear(4, 3, bias=False).to(device)
    X = torch.randn(8, 4, device=device)
    tr = KronTracker(lin, ema_decay=None)
    out = lin(X)
    gen = torch.Generator().manual_seed(0)
    for _ in range(600):
        tr.sampled_backward(out, kind="gaussian", generator=gen)
    G = tr.factors[""]["G"]
    I = torch.eye(3, device=device)
    assert (G - I).norm() / I.norm() < 0.15


def test_pi_damping_formula():
    A = torch.diag(torch.tensor([4.0, 4.0]))
    G = torch.diag(torch.tensor([1.0]))
    ga, gg = kfac_factored_damping(A, G, 0.01)
    assert abs(ga - 0.2) < 1e-12 and abs(gg - 0.05) < 1e-12
    # degenerate factors fall back to pi=1
    ga, gg = kfac_factored_damping(torch.zeros(2, 2), G, 0.04)
    assert abs(ga - 0.2) < 1e-12 and abs(gg - 0.2) < 1e-12


def test_lm_update():
    assert lm_update(1e-3, rho=0.9) < 1e-3
    assert lm_update(1e-3, rho=0.1) > 1e-3
    assert lm_update(1e-3, rho=0.5) == 1e-3
    assert lm_update(1e-12, rho=0.9) == 1e-8


def test_no_grad_forward_is_ignored(device):
    # a validation pass under torch.no_grad() (with the model still in train
    # mode) must neither crash nor pollute the factors
    lin = nn.Linear(3, 2).to(device)
    tr = KronTracker(lin)
    tr.enabled = True
    with torch.no_grad():
        lin(torch.randn(4, 3, device=device))
    assert "" not in tr.factors
