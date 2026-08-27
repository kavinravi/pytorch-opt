import torch

from pytorch_opt.curvature.lanczos import lanczos_eigs


def test_lanczos_topk(device):
    n = 40
    g = torch.Generator().manual_seed(7)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g).to(device=device, dtype=torch.float64))
    l = torch.linspace(1.0, 10.0, n, dtype=torch.float64, device=device)
    A = Q @ torch.diag(l) @ Q.T
    gen = torch.Generator().manual_seed(0)
    got = lanczos_eigs(lambda v: A @ v, n, k=5, iters=n, device=device,
                       dtype=torch.float64, generator=gen)
    want = torch.sort(l, descending=True).values[:5]
    assert torch.allclose(got, want.cpu().to(got.dtype), rtol=1e-6)
