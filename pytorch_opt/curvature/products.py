"""Curvature-vector products: Hessian, Gauss-Newton, and Fisher.

Conventions (project-wide): losses are mean-over-batch; the MSE convention is
0.5 * sum((y - t)^2) / B, so its output Hessian is I/B. Cross-entropy is
torch.nn.functional.cross_entropy (mean reduction), output Hessian blocks
(diag(p) - p p^T)/B. For these losses the true Fisher equals the GGN.
"""

from __future__ import annotations

import torch


def _flatten(ts) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts])


def _split_like(flat: torch.Tensor, params) -> list[torch.Tensor]:
    out, i = [], 0
    for p in params:
        n = p.numel()
        out.append(flat[i:i + n].view_as(p))
        i += n
    return out


def make_hvp(loss: torch.Tensor, params):
    """Returns (hvp_fn, flat_grad). hvp_fn maps a flat vector to H @ v (flat).

    The gradient graph is built once (create_graph=True); each hvp_fn call is a
    single extra backward. `loss` must be attached to the graph.
    """
    params = list(params)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_g = _flatten(grads)

    def hvp_fn(v: torch.Tensor) -> torch.Tensor:
        dot = (flat_g * v.detach()).sum()
        Hv = torch.autograd.grad(dot, params, retain_graph=True)
        return _flatten(Hv)

    return hvp_fn, flat_g.detach()


def mse_hess_mvp(outputs: torch.Tensor):
    B = outputs.shape[0]

    def mvp(u: torch.Tensor) -> torch.Tensor:
        return u / B

    return mvp


def ce_hess_mvp(logits: torch.Tensor):
    if logits.ndim != 2:
        raise ValueError("ce_hess_mvp expects (B, C) logits")
    p = torch.softmax(logits.detach(), dim=-1)
    B = logits.shape[0]

    def mvp(u: torch.Tensor) -> torch.Tensor:
        pu = p * u
        return (pu - p * pu.sum(dim=-1, keepdim=True)) / B

    return mvp


def make_ggn_vp(outputs: torch.Tensor, hess_mvp, params):
    """Gauss-Newton-vector product J^T H_out J v via the double-vjp jvp trick."""
    params = list(params)
    w = torch.zeros_like(outputs, requires_grad=True)
    g_w = torch.autograd.grad(outputs, params, grad_outputs=w, create_graph=True)

    def ggn_fn(v: torch.Tensor) -> torch.Tensor:
        vs = _split_like(v.detach(), params)
        dot = sum((g * vi).sum() for g, vi in zip(g_w, vs))
        Jv = torch.autograd.grad(dot, w, retain_graph=True)[0]
        u = hess_mvp(Jv)
        JtHJv = torch.autograd.grad(outputs, params, grad_outputs=u.detach(), retain_graph=True)
        return _flatten(JtHJv)

    return ggn_fn


def make_fisher_vp(outputs: torch.Tensor, loss_type: str, params):
    """True Fisher-vector product. For 'mse'/'ce' the Fisher equals the GGN."""
    if loss_type == "mse":
        return make_ggn_vp(outputs, mse_hess_mvp(outputs), params)
    if loss_type == "ce":
        return make_ggn_vp(outputs, ce_hess_mvp(outputs), params)
    raise ValueError(f"loss_type must be 'mse' or 'ce', got {loss_type!r}")


def per_sample_grads(model, X: torch.Tensor, y: torch.Tensor, loss_fn) -> torch.Tensor:
    """(B, n_params) matrix of per-sample gradients.

    `loss_fn(out, t)` gets a size-1 batch and must NOT divide by batch size
    (use reduction='sum' semantics); ordering matches model.named_parameters().
    """
    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_one(prm, buf, x, t):
        out = functional_call(model, (prm, buf), (x.unsqueeze(0),))
        return loss_fn(out, t.unsqueeze(0))

    g = vmap(grad(loss_one), in_dims=(None, None, 0, 0))(params, buffers, X, y)
    B = X.shape[0]
    return torch.cat([g[k].reshape(B, -1) for k, _ in model.named_parameters()], dim=1)


def empirical_fisher_vp(per_sample_g: torch.Tensor):
    """v -> (1/B) G^T (G v) for the (B, n) per-sample gradient matrix."""
    B = per_sample_g.shape[0]

    def mvp(v: torch.Tensor) -> torch.Tensor:
        return per_sample_g.T @ (per_sample_g @ v) / B

    return mvp
