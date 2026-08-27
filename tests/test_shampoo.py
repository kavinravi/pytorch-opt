import torch
from torch import nn

from pytorch_opt._testing import TinyMLP, make_regression, mse_half, run_steps
from pytorch_opt.optim import Shampoo


def test_first_step_is_polar_factor(device):
    torch.manual_seed(0)
    for shape in [(6, 6), (8, 5)]:
        p = nn.Parameter(torch.randn(*shape, dtype=torch.float64, device=device))
        p0 = p.detach().clone()
        G = torch.randn(*shape, dtype=torch.float64, device=device)
        p.grad = G.clone()
        lr = 0.1
        opt = Shampoo([p], lr=lr, beta2=1.0, eps=1e-12, precondition_frequency=1)
        opt.step()
        upd = (p0 - p.detach()) / lr
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        assert torch.allclose(upd, U @ Vh, atol=1e-8), shape


def test_stale_roots_between_refreshes(device):
    torch.manual_seed(1)
    p = nn.Parameter(torch.randn(4, 3, device=device))
    opt = Shampoo([p], precondition_frequency=5)
    roots = []
    for i in range(7):
        p.grad = torch.randn(4, 3, device=device)
        opt.step()
        roots.append(opt.state[p]["L_root"].clone())
    for i in range(1, 5):       # steps 1..4 reuse the roots from step 0
        assert torch.equal(roots[i], roots[0])
    assert not torch.equal(roots[5], roots[0])   # refresh at step 5
    assert opt.diagnostics["stale_steps"] >= 0


def test_1d_and_oversize_fall_back_to_diagonal(device):
    torch.manual_seed(2)
    b = nn.Parameter(torch.randn(7, device=device))
    w = nn.Parameter(torch.randn(6, 5, device=device))
    opt = Shampoo([b, w], max_preconditioner_dim=4, lr=0.1, beta2=1.0, diag_eps=1e-10)
    gb, gw = torch.randn_like(b), torch.randn_like(w)
    b0, w0 = b.detach().clone(), w.detach().clone()
    b.grad, w.grad = gb.clone(), gw.clone()
    opt.step()
    assert opt.diagnostics["n_diag_params"] == 2
    assert "L" not in opt.state[w]
    want_b = b0 - 0.1 * gb / (gb.abs() + 1e-10)      # first-step adagrad
    assert torch.allclose(b.detach(), want_b, atol=1e-6)


def test_grafting_sgd_norm(device):
    torch.manual_seed(3)
    p = nn.Parameter(torch.randn(5, 4, device=device))
    p0 = p.detach().clone()
    g = torch.randn(5, 4, device=device)
    p.grad = g.clone()
    lr = 1.0
    opt = Shampoo([p], lr=lr, graft="sgd", beta2=1.0, precondition_frequency=1)
    opt.step()
    upd = p0 - p.detach()
    assert abs(float(upd.norm()) - float(g.norm())) / float(g.norm()) < 1e-5


def test_converges_tiny_mlp(device):
    torch.manual_seed(0)
    model = TinyMLP().to(device)
    X, y = make_regression(device=device)
    opt = Shampoo(model.parameters(), lr=0.05, beta2=0.99, precondition_frequency=5, graft="adagrad")
    losses = run_steps(model, opt, X, y, mse_half, 200)
    assert losses[-1] < 0.2 * losses[0], (losses[0], losses[-1])
