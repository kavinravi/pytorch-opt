"""JIT loader for the native (C++/ATen) ops backend."""

from __future__ import annotations

import os

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")


def load(verbose: bool = False):
    from torch.utils import cpp_extension

    src = os.path.join(_CSRC, "pytorch_opt_ops.cpp")
    if not os.path.exists(src):
        raise FileNotFoundError(f"native sources not present: {src}")
    return cpp_extension.load(
        name="pytorch_opt_native",
        sources=[src],
        extra_cflags=["-O3"],
        verbose=verbose,
    )
