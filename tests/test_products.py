import torch
import torch.nn.functional as F
from torch import nn

from pytorch_opt.curvature import products as cp
from pytorch_opt.curvature.paramvec import ParamVector
from pytorch_opt._testing import TinyMLP


def _dense_jacobian(outputs, params):
    flat_out = outputs.reshape(-1)
    rows = []
    for i in range(flat_out.numel()):
        e = torch.zeros_like(flat_out)
        e[i] = 1.0
        g = torch.autograd.grad(outputs, params, grad_outputs=e.view_as(outputs), retain_graph=True)
        rows.append(torch.cat([t.reshape(-1) for t in g]))
    return torch.stack(rows)


def test_hvp_quadratic(device):
    n = 12
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device=device))
    A = Q @ torch.diag(torch.linspace(0.5, 3.0, n, dtype=torch.float64, device=device)) @ Q.T
    x = nn.Parameter(torch.randn(n, dtype=torch.float64, device=device))
    loss = 0.5 * x @ A @ x
    hvp_fn, g = cp.make_hvp(loss, [x])
    v = torch.randn(n, dtype=torch.float64, device=device)
    assert torch.allclose(hvp_fn(v), A @ v, atol=1e-10)
    assert torch.allclose(g, A @ x.detach(), atol=1e-10)


def test_hvp_mlp_finite_diff(device):
    model = TinyMLP(4, 6, 2).double().to(device)
    X = torch.randn(8, 4, dtype=torch.float64, device=device)
    y = torch.randn(8, 2, dtype=torch.float64, device=device)
    params = list(model.parameters())
    pv = ParamVector(params)

    def loss_at(flat):
        pv.assign_(flat)
        return 0.5 * (model(X) - y).pow(2).sum() / 8

    theta = pv.gather()
    hvp_fn, _ = cp.make_hvp(loss_at(theta), params)
    v = torch.randn(pv.numel, dtype=torch.float64, device=device)
    got = hvp_fn(v)
    eps = 1e-5

    def grad_at(flat):
        pv.assign_(flat)
        loss = 0.5 * (model(X) - y).pow(2).sum() / 8
        return torch.cat([t.reshape(-1) for t in torch.autograd.grad(loss, params)])

    fd = (grad_at(theta + eps * v) - grad_at(theta - eps * v)) / (2 * eps)
    pv.assign_(theta)
    assert torch.allclose(got, fd, atol=1e-6)


def _dense_ggn_check(device, loss_kind):
    model = TinyMLP(4, 8, 3).double().to(device)
    X = torch.randn(6, 4, dtype=torch.float64, device=device)
    params = list(model.parameters())
    out = model(X)
    J = _dense_jacobian(out, params)          # (B*C, n)
    B, C = out.shape
    if loss_kind == "mse":
        H = torch.eye(B * C, dtype=torch.float64, device=device) / B
        mvp = cp.mse_hess_mvp(out)
    else:
        p = torch.softmax(out.detach(), dim=-1)
        H = torch.zeros(B * C, B * C, dtype=torch.float64, device=device)
        for b in range(B):
            blk = (torch.diag(p[b]) - torch.outer(p[b], p[b])) / B
            H[b * C:(b + 1) * C, b * C:(b + 1) * C] = blk
        mvp = cp.ce_hess_mvp(out)
    G_dense = J.T @ H @ J
    ggn_fn = cp.make_ggn_vp(out, mvp, params)
    v = torch.randn(J.shape[1], dtype=torch.float64, device=device)
    assert torch.allclose(ggn_fn(v), G_dense @ v, atol=1e-10)
    return G_dense, params, out


def test_ggn_dense_mse(device):
    _dense_ggn_check(device, "mse")


def test_ggn_dense_ce(device):
    _dense_ggn_check(device, "ce")


def test_true_fisher_equals_ggn_ce(device):
    # Enumerated true Fisher: F = (1/B) sum_b sum_c p_bc g_bc g_bc^T,
    # g_bc = grad of CE(logits_b, c) with sum reduction. Deterministic, no sampling.
    model = TinyMLP(3, 5, 3).double().to(device)
    X = torch.randn(4, 3, dtype=torch.float64, device=device)
    params = list(model.parameters())
    out = model(X)
    p = torch.softmax(out.detach(), dim=-1)
    n = sum(pp.numel() for pp in params)
    Fm = torch.zeros(n, n, dtype=torch.float64, device=device)
    B, C = out.shape
    for b in range(B):
        for c in range(C):
            lbc = F.cross_entropy(out[b:b + 1], torch.tensor([c], device=device), reduction="sum")
            g = torch.autograd.grad(lbc, params, retain_graph=True)
            gf = torch.cat([t.reshape(-1) for t in g])
            Fm += p[b, c] * torch.outer(gf, gf)
    Fm /= B
    ggn_fn = cp.make_fisher_vp(out, "ce", params)
    v = torch.randn(n, dtype=torch.float64, device=device)
    assert torch.allclose(ggn_fn(v), Fm @ v, atol=1e-8)


def test_empirical_fisher_vp_dense(device):
    model = TinyMLP(3, 5, 2).double().to(device)
    X = torch.randn(6, 3, dtype=torch.float64, device=device)
    y = torch.randn(6, 2, dtype=torch.float64, device=device)

    def per_sample_loss(out, t):  # sum-reduced, no 1/B
        return 0.5 * (out - t).pow(2).sum()

    G = cp.per_sample_grads(model, X, y, per_sample_loss)
    # sanity: mean per-sample grad == grad of mean loss
    loss = 0.5 * (model(X) - y).pow(2).sum() / 6
    full = torch.autograd.grad(loss, list(model.parameters()))
    full_flat = torch.cat([t.reshape(-1) for t in full])
    assert torch.allclose(G.mean(0), full_flat, atol=1e-10)

    Fd = G.T @ G / 6
    mvp = cp.empirical_fisher_vp(G)
    v = torch.randn(G.shape[1], dtype=torch.float64, device=device)
    assert torch.allclose(mvp(v), Fd @ v, atol=1e-10)
