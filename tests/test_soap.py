import torch
from torch import nn

from pytorch_opt._testing import TinyMLP, make_regression, mse_half, run_steps
from pytorch_opt.optim.soap import SOAP


def test_equals_adam_when_factors_diagonal(device):
    # Diagonal gradients keep L and R diagonal, so the eigenbases are signed
    # permutations -- and elementwise Adam commutes with those, making SOAP
    # exactly Adam. This is SOAP's analytic identity test.
    torch.manual_seed(0)
    p_soap = nn.Parameter(torch.randn(5, 5, device=device))
    p_adam = nn.Parameter(p_soap.detach().clone())
    lr, betas, eps = 1e-2, (0.9, 0.999), 1e-8
    soap = SOAP([p_soap], lr=lr, betas=betas, eps=eps, precondition_frequency=2)
    adam = torch.optim.Adam([p_adam], lr=lr, betas=betas, eps=eps)
    for t in range(7):
        g = torch.diag(torch.randn(5, device=device))
        p_soap.grad = g.clone()
        p_adam.grad = g.clone()
        soap.step()
        adam.step()
        assert torch.allclose(p_soap.detach(), p_adam.detach(), atol=1e-6), t


def test_1d_params_are_exactly_adam(device):
    torch.manual_seed(1)
    p_soap = nn.Parameter(torch.randn(9, device=device))
    p_adam = nn.Parameter(p_soap.detach().clone())
    soap = SOAP([p_soap], lr=3e-3, betas=(0.95, 0.95))
    adam = torch.optim.Adam([p_adam], lr=3e-3, betas=(0.95, 0.95), eps=1e-8)
    for _ in range(5):
        g = torch.randn(9, device=device)
        p_soap.grad = g.clone()
        p_adam.grad = g.clone()
        soap.step()
        adam.step()
    assert torch.allclose(p_soap.detach(), p_adam.detach(), atol=1e-6)


def test_rotation_actually_used(device):
    # Non-diagonal grads: SOAP must differ from Adam (rotations active).
    torch.manual_seed(2)
    p_soap = nn.Parameter(torch.randn(4, 4, device=device))
    p_adam = nn.Parameter(p_soap.detach().clone())
    soap = SOAP([p_soap], lr=1e-2, precondition_frequency=1)
    adam = torch.optim.Adam([p_adam], lr=1e-2, betas=(0.95, 0.95), eps=1e-8)
    for _ in range(6):
        g = torch.randn(4, 4, device=device)
        p_soap.grad = g.clone()
        p_adam.grad = g.clone()
        soap.step()
        adam.step()
    assert not torch.allclose(p_soap.detach(), p_adam.detach(), atol=1e-4)


def test_converges_tiny_mlp(device):
    torch.manual_seed(0)
    model = TinyMLP().to(device)
    X, y = make_regression(device=device)
    opt = SOAP(model.parameters(), lr=1e-2, precondition_frequency=5)
    losses = run_steps(model, opt, X, y, mse_half, 200)
    assert losses[-1] < 0.15 * losses[0], (losses[0], losses[-1])
