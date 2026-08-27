import pytest
import torch

from pytorch_opt import ops
from pytorch_opt.ops import reference as ops_ref


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    ops.set_backend("auto")


def test_defaults():
    assert ops.get_backend() == "auto"


def test_auto_without_native_matches_reference(device):
    A = torch.randn(5, 5, device=device)
    A = A @ A.T + 0.1 * torch.eye(5, device=device)
    assert torch.allclose(ops.inverse_matrix_root(A, 4),
                          ops_ref.inverse_matrix_root(A, 4))
    G = torch.randn(4, 6, device=device)
    assert torch.allclose(ops.newton_schulz_orthogonalize(G),
                          ops_ref.newton_schulz_orthogonalize(G))


def test_reference_backend_explicit():
    ops.set_backend("reference")
    assert ops.get_backend() == "reference"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        ops.set_backend("bogus")


def test_native_requires_load():
    if ops.native_available():
        pytest.skip("native already loaded")
    with pytest.raises(RuntimeError, match="native backend not loaded"):
        ops.set_backend("native")
