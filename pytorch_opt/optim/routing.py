"""Explicit, shared parameter groups for optimizer comparisons."""

from torch import nn


def matrix_param_groups(model, module_names, *, lr, adamw_lr=3e-4,
                        weight_decay=0.0, adamw_wd=0.0):
    """Select named Linear/ungrouped Conv2d weights; route everything else to AdamW.

    Biases, vectors and parameters marked ``_no_weight_decay`` never decay.
    Tied weights cannot be selected: their gradients mix multiple module uses.
    All names are explicit; this function does not infer which layers are hidden.
    Groups can also be passed to torch.optim.AdamW for the control arm.
    """
    names = list(module_names)
    if len(names) != len(set(names)):
        raise ValueError("module_names contains duplicates")
    modules = dict(model.named_modules())
    owners = {}
    for module in model.modules():
        for p in module.parameters(recurse=False):
            owners[id(p)] = owners.get(id(p), 0) + 1
    selected = set()
    for name in names:
        if name not in modules:
            raise ValueError(f"Unknown module {name!r}")
        module = modules[name]
        if not isinstance(module, (nn.Linear, nn.Conv2d)) or (
            isinstance(module, nn.Conv2d) and module.groups != 1
        ):
            raise ValueError(f"{name!r} must be Linear or ungrouped Conv2d")
        if not module.weight.requires_grad:
            raise ValueError(f"Selected weight {name!r} is frozen")
        if owners[id(module.weight)] != 1:
            raise ValueError(f"Selected weight {name!r} is shared; assign it to AdamW")
        selected.add(id(module.weight))

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        primary = id(p) in selected
        decay = weight_decay if primary else adamw_wd
        if p.ndim < 2 or name.endswith("bias") or getattr(p, "_no_weight_decay", False):
            decay = 0.0
        key = primary, decay
        group = groups.setdefault(key, dict(params=[], param_names=[],
                                            use_preconditioner=primary,
                                            lr=lr if primary else adamw_lr,
                                            weight_decay=decay))
        group["params"].append(p)
        group["param_names"].append(name)
    return list(groups.values())
