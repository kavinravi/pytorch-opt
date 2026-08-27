import torch
import torch.nn.functional as F
from torch import nn

from pytorch_opt._testing import TinyCNN, TinyMLP, make_classification, mse_half, run_steps
from pytorch_opt.optim import KFAC


def test_kron_vec_orientation():
    # torch .flatten() is row-major: vec(iG V iA) == kron(iG, iA) @ vec(V)
    # for symmetric iA. This identity pins every orientation convention in KFAC.
    torch.manual_seed(0)
    iG = torch.randn(3, 3, dtype=torch.float64)
    iG = iG @ iG.T
    iA = torch.randn(4, 4, dtype=torch.float64)
    iA = iA @ iA.T
    V = torch.randn(3, 4, dtype=torch.float64)
    lhs = (iG @ V @ iA).flatten()
    rhs = torch.kron(iG, iA) @ V.flatten()
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_b1_factors_reproduce_dense_fisher(device):
    # With batch size 1 the Kronecker factorization is exact:
    # kron(G, A) == vec(ghat a_aug^T) outer itself.
    torch.manual_seed(1)
    lin = nn.Linear(3, 2).double().to(device)
    opt = KFAC(lin, ema_decay=None)
    X = torch.randn(1, 3, dtype=torch.float64, device=device)
    y = torch.randn(1, 2, dtype=torch.float64, device=device)
    out = lin(X)
    mse_half(out, y).backward()
    A = opt.tracker.factors[""]["A"]
    G = opt.tracker.factors[""]["G"]
    a_aug = torch.cat([X, X.new_ones(1, 1)], dim=1)
    ghat = out.detach() - y
    f = (ghat.T @ a_aug).flatten()   # vec of the per-sample W_aug gradient
    assert torch.allclose(torch.kron(G, A), torch.outer(f, f), atol=1e-10)


def test_update_equals_dense_kron_solve(device):
    torch.manual_seed(2)
    lin = nn.Linear(4, 3).double().to(device)
    opt = KFAC(lin, lr=0.5, momentum=0.9, inv_every=1, damping=1e-2)
    X = torch.randn(6, 4, dtype=torch.float64, device=device)
    y = torch.randn(6, 3, dtype=torch.float64, device=device)
    mse_half(lin(X), y).backward()
    w0 = lin.weight.detach().clone()
    b0 = lin.bias.detach().clone()
    gW, gb = lin.weight.grad.clone(), lin.bias.grad.clone()
    opt.step()
    inv = opt._inv[""]
    V = torch.cat([gW, gb.unsqueeze(1)], dim=1)
    want = inv["iG"] @ V @ inv["iA"]
    # cross-check against the dense Kronecker solve
    dense = (torch.kron(inv["iG"], inv["iA"]) @ V.flatten()).reshape(3, 5)
    assert torch.allclose(want, dense, atol=1e-10)
    got_W = (w0 - lin.weight.detach()) / 0.5
    got_b = (b0 - lin.bias.detach()) / 0.5
    assert torch.allclose(got_W, want[:, :-1], atol=1e-8)
    assert torch.allclose(got_b, want[:, -1], atol=1e-8)


def test_sampled_fisher_approaches_newton(device):
    torch.manual_seed(3)
    lin = nn.Linear(4, 3, bias=False).double().to(device)
    X = torch.randn(16, 4, dtype=torch.float64, device=device)
    y = torch.randn(16, 3, dtype=torch.float64, device=device)
    opt = KFAC(lin, ema_decay=None, fisher_mode="sampled", damping=1e-6,
               inv_every=1, momentum=0.0)
    out = lin(X)
    gen = torch.Generator().manual_seed(0)
    for _ in range(800):
        opt.update_curvature(out, kind="gaussian", generator=gen)
    loss = mse_half(lin(X), y)
    loss.backward()
    w0 = lin.weight.detach().clone()
    opt.step()
    kfac_dir = (w0 - lin.weight.detach()).flatten()
    grad = lin.weight.grad
    newton_dir = (grad @ torch.linalg.inv(X.T @ X / 16)).flatten()
    cos = torch.dot(kfac_dir, newton_dir) / (kfac_dir.norm() * newton_dir.norm())
    assert float(cos) > 0.99, float(cos)


def test_beats_sgd_on_toy_classification(device):
    X, y = make_classification(n=128, d=8, k=3, device=device)

    def train(opt_fn, steps=60):
        torch.manual_seed(0)
        model = TinyMLP(8, 16, 3).to(device)
        opt = opt_fn(model)
        losses = run_steps(model, opt, X, y, F.cross_entropy, steps)
        return losses

    kfac_losses = train(lambda m: KFAC(m, lr=0.05, damping=1e-2, momentum=0.9))
    sgd_losses = train(lambda m: torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9))
    assert kfac_losses[-1] < sgd_losses[-1], (kfac_losses[-1], sgd_losses[-1])
    assert kfac_losses[-1] < 0.5 * kfac_losses[0]


def test_conv_path_trains(device):
    torch.manual_seed(0)
    model = TinyCNN().to(device)
    X = torch.randn(32, 1, 8, 8, device=device)
    y = torch.randint(0, 3, (32,), device=device)
    opt = KFAC(model, lr=0.02, damping=1e-2, momentum=0.9)
    losses = run_steps(model, opt, X, y, F.cross_entropy, 40)
    assert losses[-1] < 0.7 * losses[0], (losses[0], losses[-1])
    A1 = opt.tracker.factors["conv1"]["A"]
    G1 = opt.tracker.factors["conv1"]["G"]
    assert A1.shape == (10, 10) and G1.shape == (4, 4)   # 1*3*3 + bias, out=4
