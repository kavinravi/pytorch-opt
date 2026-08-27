import torch

import pytorch_opt
from pytorch_opt._testing import TinyMLP, make_regression, mse_half


def test_version():
    assert pytorch_opt.__version__ == "0.1.0.dev0"


def test_harness_forward_backward(device):
    model = TinyMLP().to(device)
    X, y = make_regression(device=device)
    loss = mse_half(model(X), y)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert loss.isfinite()
