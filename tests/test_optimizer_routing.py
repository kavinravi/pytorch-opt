"""Fixed optimizer coverage, AdamW equivalence, and language-model integration."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pytorch_opt import EKFAC, KFAC, Muon, Shampoo, SOAP, matrix_param_groups

OPTIMIZERS = (Muon, Shampoo, SOAP, KFAC, EKFAC)


def build(cls, model, names, **kwargs):
    groups = matrix_param_groups(model, names, lr=0.003, adamw_lr=0.002,
                                 weight_decay=0.01, adamw_wd=0.02)
    return cls(model, params=groups, **kwargs) if issubclass(cls, KFAC) else cls(groups, **kwargs)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(17, 8)
        self.norm = nn.LayerNorm(8)
        self.qkv = nn.Linear(8, 24)
        self.out_proj = nn.Linear(8, 8)
        self.ff_in = nn.Linear(8, 16)
        self.ff_out = nn.Linear(16, 8)
        self.head = nn.Linear(8, 17, bias=False)
        self.head.weight = self.embed.weight
        self.activation_checkpointing = False

    def block(self, x):
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out_proj(y)
        return x + self.ff_out(F.gelu(self.ff_in(self.norm(x))))

    def forward(self, tokens):
        x = self.embed(tokens)
        x = checkpoint(self.block, x, use_reentrant=False) if self.activation_checkpointing else self.block(x)
        return self.head(x)


SELECTED = ("qkv", "out_proj", "ff_in", "ff_out")


@pytest.mark.parametrize("cls", OPTIMIZERS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_shared_fallback_matches_torch_adamw_and_scheduler(cls, dtype):
    model = TinyTransformer().to(dtype=dtype)
    opt = build(cls, model, SELECTED)
    fallback = [g for g in opt.param_groups if not g["use_preconditioner"]]
    refs = [[nn.Parameter(p.detach().clone()) for p in g["params"]] for g in fallback]
    adamw = torch.optim.AdamW([dict(params=ps, lr=g["lr"], weight_decay=g["weight_decay"])
                              for ps, g in zip(refs, fallback)], betas=(0.9, 0.95), eps=1e-8)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.7)
    ref_sched = torch.optim.lr_scheduler.StepLR(adamw, step_size=1, gamma=0.7)
    for step in range(4):
        for group, copies in zip(fallback, refs):
            for i, (p, ref) in enumerate(zip(group["params"], copies)):
                # None gradients must skip both decay and moment updates.
                grad = None if step == 1 and i == 0 else torch.randn_like(p)
                p.grad = grad
                ref.grad = None if grad is None else grad.clone()
        opt.step()
        adamw.step()
        sched.step()
        ref_sched.step()
        for group, copies in zip(fallback, refs):
            for p, ref in zip(group["params"], copies):
                torch.testing.assert_close(p, ref, rtol=1e-6, atol=1e-8)
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(opt.state[p][key], adamw.state[ref][key])


def test_partition_is_complete_and_identical_across_optimizers():
    model = TinyTransformer()
    model.ff_in.weight._no_weight_decay = True
    expected = {id(dict(model.named_modules())[n].weight) for n in SELECTED}
    reference = None
    for cls in OPTIMIZERS:
        opt = build(cls, model, SELECTED)
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        selected = {id(p) for g in opt.param_groups if g["use_preconditioner"] for p in g["params"]}
        assert selected == expected
        assignments = {name: (g["use_preconditioner"], g["weight_decay"])
                       for g in opt.param_groups for name in g["param_names"]}
        assert assignments["embed.weight"][0] is False
        assert assignments["norm.weight"] == (False, 0.0)
        assert assignments["ff_in.weight"] == (True, 0.0)
        assert reference is None or assignments == reference
        reference = assignments
        if isinstance(opt, KFAC):
            opt.tracker.remove()


@pytest.mark.parametrize("cls", (Shampoo, SOAP))
def test_size_limit_cannot_change_selected_parameter_assignment(cls):
    model = TinyTransformer()
    with pytest.raises(ValueError, match="exceeds max_preconditioner_dim"):
        build(cls, model, SELECTED, max_preconditioner_dim=4)


def test_rejects_unknown_duplicate_shared_and_unsupported_selections():
    model = TinyTransformer()
    for names, message in ((["missing"], "Unknown"), (["qkv", "qkv"], "duplicates"),
                           (["head"], "shared"), (["norm"], "must be Linear")):
        with pytest.raises(ValueError, match=message):
            matrix_param_groups(model, names, lr=0.01)
    groups = matrix_param_groups(model, SELECTED, lr=0.01)
    with pytest.raises(ValueError, match="cover every"):
        KFAC(model, params=[g for g in groups if g["use_preconditioner"]])


@pytest.mark.parametrize("cls", OPTIMIZERS)
@pytest.mark.parametrize("checkpointing", (False, True))
def test_transformer_bfloat16_checkpoint_resume(cls, checkpointing):
    model = TinyTransformer()
    model.activation_checkpointing = checkpointing
    opt = build(cls, model, SELECTED)
    tokens, targets = torch.randint(17, (2, 4)), torch.randint(17, (2, 4))

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())

    step(model, opt)
    msd, osd = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    step(model, opt)
    resumed = TinyTransformer()
    resumed.activation_checkpointing = checkpointing
    resumed.load_state_dict(msd)
    opt2 = build(cls, resumed, SELECTED)
    opt2.load_state_dict(osd)
    step(resumed, opt2)
    for p, q in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    if isinstance(opt, KFAC):
        assert set(opt.tracker.factors) == set(SELECTED)
        assert all(v.dtype == torch.float32 for f in opt.tracker.factors.values() for v in f.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual Mamba-2 kernels require CUDA")
@pytest.mark.parametrize("cls,fisher_mode,checkpointing", [
    (Muon, None, False), (Shampoo, None, False), (SOAP, None, False),
    (KFAC, "empirical", False), (KFAC, "sampled", False),
    (KFAC, "sampled", True),
])
def test_actual_mamba2_bfloat16(cls, fisher_mode, checkpointing):
    Mamba2 = pytest.importorskip("mamba_ssm.modules.mamba2").Mamba2
    model = Mamba2(d_model=64, d_state=16, headdim=16, chunk_size=16,
                   use_mem_eff_path=False, device="cuda")
    options = {} if fisher_mode is None else {"fisher_mode": fisher_mode}
    opt = build(cls, model, ("in_proj", "out_proj"), **options)
    # A single optimizer step is a compatibility check, not an experiment run.
    x = torch.randn(2, 32, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = checkpoint(model, x, use_reentrant=False) if checkpointing else model(x)
        targets = torch.randint(64, (2, 32), device="cuda")
        loss = F.cross_entropy(out.flatten(0, 1), targets.flatten())
    if fisher_mode == "sampled":
        opt.update_curvature(out)
        assert all(p.grad is None for p in model.parameters())
    loss.backward()
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all("exp_avg" in opt.state[p] for g in opt.param_groups
               if not g["use_preconditioner"] for p in g["params"] if p.grad is not None)
    if isinstance(opt, KFAC):
        assert set(opt.tracker.factors) == {"in_proj", "out_proj"}


@pytest.mark.parametrize("cls", OPTIMIZERS)
def test_checkpoint_rejects_changed_parameter_assignment(cls):
    model = TinyTransformer()
    opt = build(cls, model, SELECTED)
    state = copy.deepcopy(opt.state_dict())
    state["param_groups"][0]["use_preconditioner"] = not state["param_groups"][0]["use_preconditioner"]
    with pytest.raises(ValueError, match="routing differs"):
        opt.load_state_dict(state)
    state = copy.deepcopy(opt.state_dict())
    state["param_groups"][0]["param_names"].reverse()
    if len(state["param_groups"][0]["param_names"]) > 1:
        with pytest.raises(ValueError, match="names/order"):
            opt.load_state_dict(state)
