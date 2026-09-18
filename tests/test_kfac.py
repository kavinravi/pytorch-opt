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


def test_scaled_token_loss_preserves_curvature():
    import copy
    model = nn.Linear(4, 3)
    scaled = copy.deepcopy(model)
    opt = KFAC(model, inv_every=1)
    opt_scaled = KFAC(scaled, inv_every=1)
    x, targets = torch.randn(2, 5, 4), torch.randint(3, (2, 5))
    F.cross_entropy(model(x).flatten(0, 1), targets.flatten()).backward()
    # Simulate both an accumulation divisor and an AMP loss scale.
    scale = 128 / 32
    opt_scaled.set_grad_scale(scale)
    (F.cross_entropy(scaled(x).flatten(0, 1), targets.flatten()) * scale).backward()
    for key in ("A", "G"):
        torch.testing.assert_close(opt.tracker.factors[""][key], opt_scaled.tracker.factors[""][key])
    for p in scaled.parameters():
        p.grad.div_(scale)
    opt.step()
    opt_scaled.step()
    for p, q in zip(model.parameters(), scaled.parameters()):
        torch.testing.assert_close(p, q)


def test_sampled_token_logits_ignore_training_gradient_scale_and_preserve_grads():
    import copy
    model = nn.Linear(4, 3)
    other = copy.deepcopy(model)
    opt = KFAC(model, fisher_mode="sampled")
    opt2 = KFAC(other, fisher_mode="sampled")
    opt2.set_grad_scale(1 / 32)
    x = torch.randn(2, 5, 4)
    saved = {}
    for p in other.parameters():
        p.grad = torch.randn_like(p)
        saved[p] = p.grad.clone()
    opt.update_curvature(model(x), generator=torch.Generator().manual_seed(7))
    opt2.update_curvature(other(x), generator=torch.Generator().manual_seed(7))
    for key in ("A", "G"):
        torch.testing.assert_close(opt.tracker.factors[""][key], opt2.tracker.factors[""][key])
    for p, value in saved.items():
        assert torch.equal(p.grad, value)
    assert opt2.tracker.grad_scale == 1 / 32


def test_functional_projection_cannot_silently_skip_kfac():
    import pytest
    from pytorch_opt import matrix_param_groups

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(4, 8)
            self.out_proj = nn.Linear(8, 4)
            self.fused = False

        def forward(self, x):
            x = self.in_proj(x).relu()
            return F.linear(x, self.out_proj.weight, self.out_proj.bias) if self.fused else self.out_proj(x)

    model = Projection()
    opt = KFAC(model, params=matrix_param_groups(model, ["in_proj", "out_proj"], lr=0.01))
    x = torch.randn(2, 5, 4)
    model(x).square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.fused = True
    model(x).square().mean().backward()
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match="No K-FAC curvature collected for 'out_proj'"):
        opt.step()
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))


def test_frozen_bias_does_not_add_curvature_column():
    model = nn.Linear(4, 3)
    model.bias.requires_grad_(False)
    opt = KFAC(model)
    model(torch.randn(2, 4)).square().mean().backward()
    opt.step()
    assert opt.tracker.factors[""]["A"].shape == (4, 4)


def test_checkpoint_restores_curvature_configuration_and_dtype():
    import copy
    model = nn.Linear(4, 3)
    opt = KFAC(model, fisher_mode="sampled", ema_decay=0.7)
    opt.set_grad_scale(1 / 32)
    out = model(torch.randn(2, 5, 4))
    opt.update_curvature(out)
    out.square().mean().backward()
    opt.step()
    restored = nn.Linear(4, 3).double()
    resumed = KFAC(restored)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert resumed.fisher_mode == "sampled"
    assert resumed.tracker.ema_decay == 0.7
    assert resumed.tracker.grad_scale == 1 / 32
    assert all(v.dtype == torch.float64 for f in resumed.tracker.factors.values() for v in f.values())
    assert resumed._inv[""]["iA"].dtype == torch.float64
