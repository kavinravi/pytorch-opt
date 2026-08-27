"""pytorch-opt: second-order and structure-aware optimizers for PyTorch."""

from . import diag, ops
from .optim import AdaHessian, EKFAC, KFAC, Muon, NGD, Shampoo, SOAP, Sophia, TrustNCG

__version__ = "0.1.0.dev0"

__all__ = ["AdaHessian", "EKFAC", "KFAC", "Muon", "NGD", "Shampoo", "SOAP", "Sophia", "TrustNCG", "diag", "ops", "__version__"]
