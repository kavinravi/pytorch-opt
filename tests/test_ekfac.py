import torch
import torch.nn.functional as F
from torch import nn

from pytorch_opt._testing import TinyMLP, make_classification, mse_half, run_steps
from pytorch_opt.curvature.products import per_sample_grads
from pytorch_opt.optim.ekfac import EKFAC
from pytorch_opt.optim.kfac import KFAC


def _twin_linears(device, seed=0):
    torch.manual_seed(seed)
    a = nn.Linear(4, 3).double().to(device)
    b = nn.Linear(4, 3).double().to(device)
    b.load_state_dict(a.state_dict())
    return a, b


def test_reduces_to_kfac_with_eigenvalue_scales(device):
    # With s frozen to outer(evG, evA) and vanishing damping, EKFAC == KFAC.
    m_k, m_e = _twin_linears(device)
    lam = 0.0   # identity is exact only at zero damping (pi-split vs flat cross-terms)
    kf = KFAC(m_k, lr=0.3, damping=lam, momentum=0.0, inv_every=1)
    ek = EKFAC(m_e, lr=0.3, damping=lam, momentum=0.0, inv_every=1)
    X = torch.randn(16, 4, dtype=torch.float64, device=device)
    y = torch.randn(16, 3, dtype=torch.float64, device=device)
    for model, opt in ((m_k, kf), (m_e, ek)):
        mse_half(model(X), y).backward()
    # refresh EKFAC bases first so we can inject the eigenvalue-product scales
    ek._refresh_inverses()
    inv = ek._inv[""]
    ek._scales[""] = torch.outer(inv["evG"], inv["evA"])
    ek._scale_decay = 1.0                      # freeze the injected scales
    kf.step()
    ek.step()
    assert torch.allclose(m_k.weight.detach(), m_e.weight.detach(), atol=1e-6)
    assert torch.allclose(m_k.bias.detach(), m_e.bias.detach(), atol=1e-6)


def test_scales_initialized_from_first_rotated_gradient(device):
    torch.manual_seed(1)
    model = nn.Linear(3, 2).double().to(device)
    opt = EKFAC(model, lr=0.1, inv_every=1, momentum=0.0)
    X = torch.randn(8, 3, dtype=torch.float64, device=device)
    y = torch.randn(8, 2, dtype=torch.float64, device=device)
    mse_half(model(X), y).backward()
    gW = model.weight.grad.clone()
    gb = model.bias.grad.clone()
    opt.step()
    inv = opt._inv[""]
    V = torch.cat([gW, gb.unsqueeze(1)], dim=1)
    Vt = inv["QG"].T @ V @ inv["QA"]
    assert torch.allclose(opt._scales[""], Vt.pow(2), atol=1e-10)


def test_optimal_diagonal_beats_kron_in_frobenius(device):
    # George et al. Thm: the eigenbasis diagonal s* = diag(Q^T F Q) gives a
    # Fisher approximation at least as good (Frobenius) as A (x) G.
    torch.manual_seed(2)
    model = nn.Linear(3, 2, bias=False).double().to(device)
    X = torch.randn(8, 3, dtype=torch.float64, device=device)
    y = torch.randn(8, 2, dtype=torch.float64, device=device)

    def psl(out, t):
        return 0.5 * (out - t).pow(2).sum()

    G = per_sample_grads(model, X, y, psl)             # (B, 6), vec_row(gW)
    Fm = G.T @ G / 8
    opt = EKFAC(model, inv_every=1)
    mse_half(model(X), y).backward()
    opt._refresh_inverses()
    inv = opt._inv[""]
    A = opt.tracker.factors[""]["A"]
    Gf = opt.tracker.factors[""]["G"]
    Q = torch.kron(inv["QG"], inv["QA"])               # vec_row convention
    s_star = torch.diag(Q.T @ Fm @ Q)
    err_ekfac = (Fm - Q @ torch.diag(s_star) @ Q.T).norm()
    err_kfac = (Fm - torch.kron(Gf, A)).norm()
    assert float(err_ekfac) <= float(err_kfac) + 1e-12


def test_converges_on_classification(device):
    X, y = make_classification(n=128, d=8, k=3, device=device)
    torch.manual_seed(0)
    model = TinyMLP(8, 16, 3).to(device)
    opt = EKFAC(model, lr=0.05, damping=1e-2, momentum=0.9)
    losses = run_steps(model, opt, X, y, F.cross_entropy, 60)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
