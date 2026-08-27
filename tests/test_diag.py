import torch
from torch import nn

from pytorch_opt.diag import hessian_eigs


def test_hessian_eigs_quadratic(device):
    n = 15
    g = torch.Generator().manual_seed(9)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g).to(device=device, dtype=torch.float64))
    l = torch.linspace(0.2, 7.0, n, dtype=torch.float64, device=device)
    A = Q @ torch.diag(l) @ Q.T
    x = nn.Parameter(torch.randn(n, dtype=torch.float64, device=device))
    eigs = hessian_eigs(lambda: 0.5 * x @ A @ x, [x], k=4, iters=n,
                        generator=torch.Generator().manual_seed(0))
    want = torch.sort(l, descending=True).values[:4]
    assert torch.allclose(eigs, want.cpu().to(eigs.dtype), rtol=1e-6)
