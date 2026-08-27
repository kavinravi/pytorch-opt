"""Cross-optimizer contract tests: determinism, state_dict round-trip,
diagnostics schema, distributed-readiness state tagging. CPU-only (bitwise
determinism). Extension-roster optimizers get added to FACTORIES and inherit
all of these."""

import copy

import pytest
import torch

from pytorch_opt._testing import TinyMLP, make_regression, mse_half, run_steps
from pytorch_opt.optim import AdaHessian, EKFAC, HessianFree, KFAC, Muon, NGD, Shampoo, SOAP, Sophia, TrustNCG

FACTORIES = {
    "muon": lambda m: Muon(m.parameters(), lr=0.02),
    "shampoo": lambda m: Shampoo(m.parameters(), lr=0.03, graft="adagrad",
                                 precondition_frequency=2),
    "kfac": lambda m: KFAC(m, lr=0.02, damping=1e-2, inv_every=2),
    "trustncg": lambda m: TrustNCG(m.parameters(), delta0=0.5),
    "ngd": lambda m: NGD(m.parameters(), lr=0.5, damping=1e-3, momentum=0.5),
    "soap": lambda m: SOAP(m.parameters(), lr=1e-2, precondition_frequency=2),
    "adahessian": lambda m: AdaHessian(m.parameters(), lr=0.05, update_freq=2),
    "sophia": lambda m: Sophia(m.parameters(), lr=2e-2, estimate_freq=3),
    "ekfac": lambda m: EKFAC(m, lr=0.02, damping=1e-2, inv_every=2),
    "hessianfree": lambda m: HessianFree(m.parameters(), max_cg_iter=15),
}

REQUIRED_DIAG = {
    "muon": {"update_rms", "n_muon_params", "n_adamw_params", "step_ms"},
    "shampoo": {"stale_steps", "cond_L", "cond_R", "graft_ratio",
                "n_diag_params", "step_ms", "curvature_ms"},
    "kfac": {"damping", "mean_cond_A", "mean_cond_G", "inv_stale_steps",
             "nat_grad_norm", "step_ms", "curvature_ms"},
    "trustncg": {"delta", "rho", "cg_iters", "cg_reason", "accepted",
                 "pred_reduction", "step_ms"},
    "ngd": {"fisher_cond", "nat_grad_norm", "step_ms", "curvature_ms"},
    "soap": {"stale_steps", "n_adam_params", "step_ms", "curvature_ms"},
    "adahessian": {"hutchinson_refreshed", "step_ms", "curvature_ms"},
    "sophia": {"clip_fraction", "estimate_refreshed", "step_ms", "curvature_ms"},
    "ekfac": {"damping", "mean_cond_A", "mean_cond_G", "inv_stale_steps",
              "nat_grad_norm", "step_ms", "curvature_ms"},
    "hessianfree": {"damping", "rho", "cg_iters", "backtracked_to",
                    "accepted", "step_ms"},
}


def _build(name):
    torch.manual_seed(0)
    model = TinyMLP()
    return model, FACTORIES[name](model)


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_deterministic_trajectory(name):
    finals = []
    for _ in range(2):
        model, opt = _build(name)
        X, y = make_regression()
        run_steps(model, opt, X, y, mse_half, 5)
        finals.append(_flat_params(model))
    assert torch.equal(finals[0], finals[1])


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_state_dict_round_trip(name):
    model, opt = _build(name)
    X, y = make_regression()
    run_steps(model, opt, X, y, mse_half, 3)
    msd = copy.deepcopy(model.state_dict())
    osd = copy.deepcopy(opt.state_dict())
    run_steps(model, opt, X, y, mse_half, 3)
    pA = _flat_params(model)

    model2, opt2 = _build(name)
    model2.load_state_dict(msd)
    opt2.load_state_dict(osd)
    run_steps(model2, opt2, X, y, mse_half, 3)
    pB = _flat_params(model2)
    assert torch.equal(pA, pB)


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_diagnostics_schema(name):
    model, opt = _build(name)
    X, y = make_regression()
    run_steps(model, opt, X, y, mse_half, 1)
    missing = REQUIRED_DIAG[name] - set(opt.diagnostics)
    assert not missing, missing


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_state_layout_declared(name):
    model, opt = _build(name)
    X, y = make_regression()
    run_steps(model, opt, X, y, mse_half, 2)
    layout = type(opt).state_layout()
    assert layout and all(v in ("replicable", "shardable") for v in layout.values())
    for _, st in opt.state.items():
        for key in st:
            assert key in layout, f"state key {key!r} not tagged in state_layout()"
