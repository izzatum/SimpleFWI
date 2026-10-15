"""Linear-solver contracts shared by explicit and matrix-free adapters."""
from __future__ import annotations

from typing import Protocol
import torch
from ..types import SolveReport


class LinearSolver(Protocol):
    """Solve compact vectors with degrees of freedom on the final axis."""

    def solve(self, operator, m: torch.Tensor, omega: float, rhs: torch.Tensor,
              *, adjoint: bool = False) -> tuple[torch.Tensor, SolveReport]: ...


class SolverFailure(RuntimeError):
    """A failed solve; callers must not accept its gradient or optimizer step."""

    def __init__(self, message: str, report: SolveReport):
        super().__init__(message)
        self.report = report
