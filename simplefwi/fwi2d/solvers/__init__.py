"""Validated linear solver adapters."""
from .base import LinearSolver, SolverFailure
from .direct import DirectSolver

__all__ = ['LinearSolver', 'SolverFailure', 'DirectSolver']
