import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
