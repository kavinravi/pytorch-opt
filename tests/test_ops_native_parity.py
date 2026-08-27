"""Native (C++/ATen) backend vs reference ground truth, CPU and CUDA."""

import pytest
import torch

from pytorch_opt import ops
from pytorch_opt.ops import reference as ops_ref


@pytest.fixture(scope="session")
def native():
    if not ops.load_native():
        pytest.skip("native extension could not be built in this environment")
    return ops


@pytest.fixture
def native_backend(native):
    ops.set_backend("native")
    yield
    ops.set_backend("auto")


def _psd(shape, device, dtype=torch.float32, seed=11):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(*shape, generator=g).to(device=device, dtype=dtype)
    return B @ B.mT + 0.1 * torch.eye(shape[-1], device=device, dtype=dtype)


def test_root_parity(native_backend, device):
    A = _psd((6, 6), device)
    Ab = _psd((3, 5, 5), device)
    for p in (1, 2, 4):
        for method in ("eigh", "newton"):
            got = ops.inverse_matrix_root(A, p, damping=1e-6, method=method)
            want = ops_ref.inverse_matrix_root(A, p, damping=1e-6, method=method)
            assert torch.allclose(got, want, atol=1e-6), (p, method)
    got = ops.inverse_matrix_root(Ab, 4, damping=1e-6)
    want = ops_ref.inverse_matrix_root(Ab, 4, damping=1e-6)
    assert torch.allclose(got, want, atol=1e-6)


def test_ns_parity(native_backend, device):
    for shape in [(8, 5), (5, 8), (4, 6, 9)]:
        G = torch.randn(*shape, device=device)
        got = ops.newton_schulz_orthogonalize(G)
        want = ops_ref.newton_schulz_orthogonalize(G)
        assert torch.allclose(got, want, atol=1e-5), shape


def test_kron_update_parity(native_backend, device):
    g = torch.Generator().manual_seed(12)
    G1, G2 = (torch.randn(4, 7, generator=g).to(device) for _ in range(2))
    for beta2 in (1.0, 0.9):
        Ln = torch.zeros(4, 4, device=device)
        Rn = torch.zeros(7, 7, device=device)
        Lr = torch.zeros(4, 4, device=device)
        Rr = torch.zeros(7, 7, device=device)
        for G in (G1, G2):
            ops.kron_factor_update_(Ln, Rn, G, beta2=beta2)
            ops_ref.kron_factor_update_(Lr, Rr, G, beta2=beta2)
        assert torch.allclose(Ln, Lr, atol=1e-6)
        assert torch.allclose(Rn, Rr, atol=1e-6)


def test_apply_parity(native_backend, device):
    Lr = torch.randn(4, 4, device=device)
    G = torch.randn(4, 7, device=device)
    Rr = torch.randn(7, 7, device=device)
    assert torch.allclose(ops.precond_apply_two_sided(Lr, G, Rr),
                          ops_ref.precond_apply_two_sided(Lr, G, Rr), atol=1e-6)


def test_auto_backend_prefers_native(native):
    assert ops.native_available()
    assert ops.get_backend() == "auto"
    A = _psd((5, 5), "cpu")
    got = ops.inverse_matrix_root(A, 4)          # auto -> native path
    want = ops_ref.inverse_matrix_root(A, 4)
    assert torch.allclose(got, want, atol=1e-6)
