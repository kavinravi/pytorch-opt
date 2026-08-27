"""Shared plumbing for pytorch-opt optimizers: diagnostics + step timing."""

from __future__ import annotations

import time


class StepTimer:
    """Context manager measuring wall-clock milliseconds."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1e3
        return False


class DiagnosticsMixin:
    """Optimizers populate self._diag each step; read via .diagnostics."""

    @property
    def diagnostics(self) -> dict:
        return dict(getattr(self, "_diag", {}))
