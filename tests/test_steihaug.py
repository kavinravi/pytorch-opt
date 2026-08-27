import torch

from pytorch_opt.curvature.steihaug import steihaug_cg


def _quad(device, n=10, lo=0.5, hi=3.0, seed=2):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g).to(device=device, dtype=torch.float64))
    A = Q @ torch.diag(torch.linspace(lo, hi, n, dtype=torch.float64, device=device)) @ Q.T
    b = torch.randn(n, generator=g).to(device=device, dtype=torch.float64)
    return A, b


def _m(A, g, s):
    return float(g @ s + 0.5 * s @ A @ s)


def test_steihaug_solves_interior(device):
    A, g = _quad(device)
    s, info = steihaug_cg(lambda v: A @ v, g, radius=1e6, tol=1e-12, max_iter=100)
    assert info["reason"] == "converged"
    assert torch.allclose(s, -torch.linalg.solve(A, g), atol=1e-8)


def test_steihaug_boundary(device):
    A, g = _quad(device)
    r = 0.1
    s, info = steihaug_cg(lambda v: A @ v, g, radius=r)
    assert info["reason"] == "boundary"
    assert abs(float(s.norm()) - r) < 1e-10
    # at least as good as the clipped Cauchy point
    gAg = float(g @ A @ g)
    t_c = min(float(g @ g) / gAg, r / float(g.norm()))
    assert _m(A, g, s) <= _m(A, g, -t_c * g) + 1e-12


def test_steihaug_negative_curvature(device):
    A = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64, device=device))
    g = torch.tensor([1.0, 0.3], dtype=torch.float64, device=device)
    r = 2.0
    s, info = steihaug_cg(lambda v: A @ v, g, radius=r)
    assert info["reason"] == "neg_curvature"
    assert abs(float(s.norm()) - r) < 1e-10
    assert _m(A, g, s) < 0


def test_steihaug_zero_grad(device):
    g = torch.zeros(4, dtype=torch.float64, device=device)
    s, info = steihaug_cg(lambda v: v, g, radius=1.0)
    assert info["reason"] == "zero_grad" and s.abs().sum() == 0
