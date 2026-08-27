"""pytorch-opt: second-order and structure-aware optimizers for PyTorch."""

from . import diag, ops
from .optim import AdaHessian, EKFAC, HessianFree, KFAC, Muon, NGD, PSGD, Shampoo, SOAP, Sophia, TrustNCG

__version__ = "0.1.0.dev0"

__all__ = ["AdaHessian", "EKFAC", "HessianFree", "KFAC", "Muon", "NGD", "PSGD", "Shampoo", "SOAP", "Sophia", "TrustNCG", "diag", "ops", "__version__"]
