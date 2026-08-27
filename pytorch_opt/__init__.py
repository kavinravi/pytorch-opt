"""pytorch-opt: second-order and structure-aware optimizers for PyTorch."""

from . import diag, ops
from .optim import KFAC, Muon, NGD, Shampoo, SOAP, TrustNCG

__version__ = "0.1.0.dev0"

__all__ = ["KFAC", "Muon", "NGD", "Shampoo", "SOAP", "TrustNCG", "diag", "ops", "__version__"]
