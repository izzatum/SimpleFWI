"""Bounded, immutable stage screening with explicit resource assumptions.

Planning never executes a solve or interpolates a model. PML calibration and
true-residual convergence remain independent gates after spatial screening.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

from .dispersion import DispersionReport, assess_dispersion, _validate_inputs
from .types import BoundaryConfig, Grid2D


class PlanningError(ValueError):
    """No requested candidate satisfies spatial screening and memory budget."""


def _count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass(frozen=True)
class StageRequest:
    """Explicit candidates for one stage, with physical node extents preserved.

    ``extra_workspace_bytes`` reserves caller-known preconditioner, recording,
    curvature or cache memory. The base estimate covers a matrix-free FGMRES
    stage; sparse direct factorization fill is deliberately not modeled.
    """
    grids: tuple[Grid2D, ...]
    velocity_bounds: tuple[float, float]
    frequencies_hz: tuple[float, ...]
    source_count: int
    receiver_count: int
    boundary: BoundaryConfig = BoundaryConfig()
    source_batch_sizes: tuple[int, ...] = (1,)
    precision: str = 'complex128'
    krylov_restart: int = 30
    max_phase_error: float = .01
    optimizer_history_size: int = 10
    extra_workspace_bytes: int = 0

    def __post_init__(self):
        grids = tuple(self.grids)
        if not grids or len(grids) > 64 or not all(isinstance(g, Grid2D) for g in grids):
            raise ValueError('grids must contain 1 to 64 Grid2D candidates')
        first = grids[0]
        extent = tuple((n - 1) * h for n, h in zip(first.shape, first.spacing_m))
        for grid in grids[1:]:
            candidate_extent = tuple((n - 1) * h for n, h in zip(grid.shape, grid.spacing_m))
            if grid.origin_m != first.origin_m or not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
                                                         for a, b in zip(extent, candidate_extent)):
                raise ValueError('candidate grids must have identical physical origin and node extent')
        bounds, frequencies, tolerance = _validate_inputs(self.velocity_bounds, self.frequencies_hz,
                                                          self.max_phase_error)
        if len(frequencies) > 4096:
            raise ValueError('at most 4096 frequencies are allowed per stage')
        if not isinstance(self.boundary, BoundaryConfig):
            raise ValueError('boundary must be BoundaryConfig')
        if self.precision != 'complex128':
            raise ValueError('only reference complex128 precision is supported for stage screening')
        for name in ('source_count', 'receiver_count', 'krylov_restart'):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        for name in ('optimizer_history_size', 'extra_workspace_bytes'):
            object.__setattr__(self, name, _count(getattr(self, name), name, 0))
        batches = tuple(_count(b, 'source_batch_sizes') for b in self.source_batch_sizes)
        if not batches or len(batches) > 64 or max(batches) > self.source_count:
            raise ValueError('source_batch_sizes must contain 1 to 64 sizes <= source_count')
        object.__setattr__(self, 'grids', grids)
        object.__setattr__(self, 'velocity_bounds', bounds)
        object.__setattr__(self, 'frequencies_hz', frequencies)
        object.__setattr__(self, 'max_phase_error', tolerance)
        object.__setattr__(self, 'source_batch_sizes', tuple(sorted(set(batches))))


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated resident bytes, including a 25% allocator/temporary reserve.

    This estimate is not an observed peak or a guarantee for arbitrary solvers.
    All complex arrays are complex128, all real model arrays are float64.
    """
    physical_cells: int
    padded_cells: int
    pml_cells: int
    model_bytes: int
    operator_bytes: int
    wavefield_bytes: int
    krylov_bytes: int
    optimizer_bytes: int
    data_bytes: int
    extra_workspace_bytes: int
    reserve_bytes: int
    total_bytes: int
    scope: str = 'matrix_free_fgmres_no_direct_factorization'


@dataclass(frozen=True)
class StagePlan:
    """Fixed numerical choices; construct a new plan only at a stage boundary."""
    grid: Grid2D
    stencil: str
    frequencies_hz: tuple[float, ...]
    source_count: int
    receiver_count: int
    source_batch_size: int
    precision: str
    boundary: BoundaryConfig
    krylov_restart: int
    velocity_bounds: tuple[float, float]
    dispersion: DispersionReport
    memory: MemoryEstimate
    memory_budget_bytes: int
    validation_status: str = 'screened_only'
    pml_validation: str = 'required'
    linear_solve_validation: str = 'required'
    history_reset_on_stage_change: bool = True


def _estimate_memory(request, grid, batch):
    nz, nx = grid.shape
    p = request.boundary.pml_cells
    physical = nz * nx
    padded = (nz + p * (1 if request.boundary.top == 'free_surface' else 2)) * (nx + 2*p)
    # Physical model, reference, gradient and work arrays.
    model = 4 * physical * 8
    # Complex coefficients, padded model and stencil/preconditioner work reserve.
    operator = 12 * padded * 16
    # RHS, solution, adjoint and incremental/work wavefield per source batch.
    wavefield = 4 * batch * padded * 16
    restart = request.krylov_restart
    # FGMRES V_(m+1), Z_m; Hessenberg and small least-squares work per RHS.
    krylov = batch * 16 * ((2*restart + 1)*padded + (restart + 1)*restart + 4*(restart + 1))
    optimizer = (2 * request.optimizer_history_size + 3) * physical * 8
    # Observed, predicted and residual, all frequencies/sources/receivers.
    data = 3 * len(request.frequencies_hz) * request.source_count * request.receiver_count * 16
    subtotal = model + operator + wavefield + krylov + optimizer + data + request.extra_workspace_bytes
    reserve = (subtotal + 3) // 4
    return MemoryEstimate(physical, padded, padded-physical, model, operator, wavefield,
                          krylov, optimizer, data, request.extra_workspace_bytes,
                          reserve, subtotal+reserve)


def plan_stage(request: StageRequest, *, memory_budget_bytes: int) -> StagePlan:
    """Choose the smallest passing candidate grid and largest affordable batch.

    Frequencies, tolerance, reference precision and PML remain exactly as
    requested. Rejection is explicit. The finite search has <=64 grids and
    <=64 batch sizes; it never creates an unrequested refinement or stencil.
    """
    if not isinstance(request, StageRequest):
        raise TypeError('request must be StageRequest')
    budget = _count(memory_budget_bytes, 'memory_budget_bytes')
    spatial_passed = False
    for grid in sorted(request.grids, key=lambda g: g.shape[0]*g.shape[1]):
        dispersion = assess_dispersion(grid, request.velocity_bounds, request.frequencies_hz,
                                       max_phase_error=request.max_phase_error)
        if not dispersion.passed:
            continue
        spatial_passed = True
        for batch in reversed(request.source_batch_sizes):
            memory = _estimate_memory(request, grid, batch)
            if memory.total_bytes <= budget:
                return StagePlan(grid, 'second_order', request.frequencies_hz, request.source_count,
                                 request.receiver_count, batch, request.precision, request.boundary,
                                 request.krylov_restart, request.velocity_bounds, dispersion, memory, budget)
    if spatial_passed:
        raise PlanningError('no dispersion-screened candidate fits the memory budget; requested accuracy is unchanged')
    raise PlanningError('no candidate meets the requested dispersion target at the minimum allowed velocity')
