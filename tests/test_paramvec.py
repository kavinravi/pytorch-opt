import pytest
import torch
from torch import nn

from pytorch_opt.curvature.paramvec import ParamVector


def _params(device):
    lin = nn.Linear(4, 3).to(device)
    extra = nn.Parameter(torch.randn(2, 2, 2, device=device))
    return [lin.weight, lin.bias, extra]


def test_gather_assign_round_trip(device):
    params = _params(device)
    pv = ParamVector(params)
    flat = pv.gather()
    assert flat.shape == (pv.numel,)
    orig = [p.detach().clone() for p in params]
    pv.assign_(torch.zeros_like(flat))
    assert all(p.detach().abs().sum() == 0 for p in params)
    pv.assign_(flat)
    for p, o in zip(params, orig):
        assert torch.equal(p.detach(), o)


def test_add_reflects_in_gather(device):
    pv = ParamVector(_params(device))
    before = pv.gather()
    step = torch.randn_like(before)
    pv.add_(step, alpha=-0.5)
    assert torch.allclose(pv.gather(), before - 0.5 * step, atol=1e-7)


def test_unflatten_shapes(device):
    params = _params(device)
    pv = ParamVector(params)
    parts = pv.unflatten(pv.gather())
    assert [tuple(t.shape) for t in parts] == [tuple(p.shape) for p in params]
    # values line up with the params they came from
    for t, p in zip(parts, params):
        assert torch.equal(t, p.detach())


def test_gather_grads_fills_none(device):
    params = _params(device)
    params[0].grad = torch.ones_like(params[0])
    pv = ParamVector(params)
    g = pv.gather_grads()
    parts = pv.unflatten(g)
    assert torch.equal(parts[0], torch.ones_like(params[0]))
    assert parts[1].abs().sum() == 0 and parts[2].abs().sum() == 0


def test_mixed_dtype_rejected(device):
    a = nn.Parameter(torch.randn(2, device=device))
    b = nn.Parameter(torch.randn(2, device=device, dtype=torch.float64))
    with pytest.raises(TypeError, match="dtype"):
        ParamVector([a, b])
