"""Shared plumbing for pytorch-opt optimizers: diagnostics + step timing."""

from __future__ import annotations

import time


class StepTimer:
    """Context manager measuring wall-clock milliseconds."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._final = None
        return self

    def __exit__(self, *exc):
        self._final = (time.perf_counter() - self.t0) * 1e3
        return False

    @property
    def ms(self) -> float:
        if self._final is not None:
            return self._final
        return (time.perf_counter() - self.t0) * 1e3


class DiagnosticsMixin:
    """Optimizers populate self._diag each step; read via .diagnostics."""

    @property
    def diagnostics(self) -> dict:
        return dict(getattr(self, "_diag", {}))
