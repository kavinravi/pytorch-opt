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


def test_1d_and_explicit_matrix_fallback_match_adamw(device):
    torch.manual_seed(2)
    b = nn.Parameter(torch.randn(7, device=device))
    w = nn.Parameter(torch.randn(6, 5, device=device))
    refs = [nn.Parameter(p.detach().clone()) for p in (b, w)]
    opt = Shampoo([dict(params=[b, w], use_preconditioner=False, lr=0.1,
                        weight_decay=0.2)], max_preconditioner_dim=4)
    adamw = torch.optim.AdamW(refs, lr=0.1, betas=(0.9, 0.95), weight_decay=0.2)
    for _ in range(3):
        for p, ref in zip((b, w), refs):
            p.grad = torch.randn_like(p)
            ref.grad = p.grad.clone()
        opt.step()
        adamw.step()
    assert opt.diagnostics["n_adamw_params"] == 2
    assert "L" not in opt.state[w]
    for p, ref in zip((b, w), refs):
        torch.testing.assert_close(p, ref)


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
