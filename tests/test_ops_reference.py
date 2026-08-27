import pytest
import torch

from pytorch_opt.ops import reference as ops_ref


def _spd(n, device, dtype=torch.float64, seed=1, lo=0.5, hi=4.0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g).to(device=device, dtype=dtype))
    l = torch.linspace(lo, hi, n, dtype=dtype, device=device)
    return Q @ torch.diag(l) @ Q.T, Q, l


# ---------------- inverse_matrix_root ----------------

def test_root_matches_closed_form(device):
    A, Q, l = _spd(6, device)
    for p in (1, 2, 4):
        want = Q @ torch.diag(l.pow(-1.0 / p)) @ Q.T
        got = ops_ref.inverse_matrix_root(A, p)
        assert torch.allclose(got, want, atol=1e-10)


def test_newton_matches_eigh(device):
    A, _, _ = _spd(8, device, seed=3)
    for p in (2, 4):
        got_n = ops_ref.inverse_matrix_root(A, p, damping=1e-3, method="newton")
        got_e = ops_ref.inverse_matrix_root(A, p, damping=1e-3, method="eigh")
        assert torch.allclose(got_n, got_e, atol=1e-7)


def test_damping_applied(device):
    A, _, _ = _spd(5, device, seed=4)
    d = 0.37
    I = torch.eye(5, dtype=A.dtype, device=device)
    want = ops_ref.inverse_matrix_root(A + d * I, 2)
    got = ops_ref.inverse_matrix_root(A, 2, damping=d)
    assert torch.allclose(got, want, atol=1e-10)


def test_batched_and_dtype(device):
    g = torch.Generator().manual_seed(5)
    B = torch.randn(3, 5, 5, generator=g).to(device)
    A = (B @ B.mT + 0.1 * torch.eye(5, device=device)).to(dtype=torch.float32)
    got = ops_ref.inverse_matrix_root(A, 4)
    assert got.dtype == torch.float32 and got.shape == A.shape
    for i in range(3):
        per = ops_ref.inverse_matrix_root(A[i].double(), 4)
        assert torch.allclose(got[i].double(), per, atol=1e-5)


def test_root_input_validation(device):
    A, _, _ = _spd(4, device)
    with pytest.raises(ValueError, match="positive integer"):
        ops_ref.inverse_matrix_root(A, 0)
    with pytest.raises(ValueError, match="method"):
        ops_ref.inverse_matrix_root(A, 2, method="qr")


# ---------------- newton_schulz ----------------

def test_ns_orthogonalizes(device):
    G = torch.randn(16, 32, device=device)
    O = ops_ref.newton_schulz_orthogonalize(G)
    s = torch.linalg.svdvals(O.double())
    assert s.min() > 0.3 and s.max() < 1.6
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    polar = U @ Vh
    assert (O.double() - polar).norm() / polar.norm() < 0.35


def test_ns_orthogonal_input_gives_scalar_multiple(device):
    # For (semi-)orthogonal input all singular values are equal, so NS5 maps it
    # to an exact scalar multiple of itself (scalar inside the NS5 band).
    for shape in [(8, 8), (12, 6)]:  # square + tall (transpose path)
        Q, _ = torch.linalg.qr(torch.randn(*shape, device=device))
        O = ops_ref.newton_schulz_orthogonalize(Q, steps=10)
        s = (O * Q).sum() / Q.pow(2).sum()
        assert 0.3 < float(s) < 1.6
        assert (O - s * Q).norm() / Q.norm() < 1e-3


def test_ns_batched_matches_loop(device):
    G = torch.randn(4, 8, 12, device=device)
    O = ops_ref.newton_schulz_orthogonalize(G)
    for i in range(4):
        assert torch.allclose(O[i], ops_ref.newton_schulz_orthogonalize(G[i]), atol=1e-5)


# ---------------- kron factors + apply ----------------

def test_kron_factor_update_sum_and_ema(device):
    g = torch.Generator().manual_seed(6)
    G1, G2 = (torch.randn(4, 7, generator=g).to(device) for _ in range(2))
    L = torch.zeros(4, 4, device=device)
    R = torch.zeros(7, 7, device=device)
    ops_ref.kron_factor_update_(L, R, G1, beta2=1.0)
    ops_ref.kron_factor_update_(L, R, G2, beta2=1.0)
    assert torch.allclose(L, G1 @ G1.T + G2 @ G2.T, atol=1e-6)
    assert torch.allclose(R, G1.T @ G1 + G2.T @ G2, atol=1e-6)

    L = torch.zeros(4, 4, device=device)
    R = torch.zeros(7, 7, device=device)
    b = 0.9
    ops_ref.kron_factor_update_(L, R, G1, beta2=b)
    ops_ref.kron_factor_update_(L, R, G2, beta2=b)
    want_L = b * ((1 - b) * (G1 @ G1.T)) + (1 - b) * (G2 @ G2.T)
    assert torch.allclose(L, want_L, atol=1e-6)


def test_precond_apply(device):
    Lr = torch.randn(4, 4, device=device)
    G = torch.randn(4, 7, device=device)
    Rr = torch.randn(7, 7, device=device)
    assert torch.allclose(ops_ref.precond_apply_two_sided(Lr, G, Rr), Lr @ G @ Rr)
