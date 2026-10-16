"""Explicit coarse-to-fine grids and low-to-high frequency continuation.

Only detached physical slowness squared crosses a stage boundary. Observations
remain measurements at fixed SI coordinates; neither data, optimizer histories,
categorical masks nor curvature operators are interpolated. Each stage uses the
existing SUM objective and a fresh native optimizer, without frequency-count
normalization. Tiny workflow tests do not calibrate PML absorption or recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import weakref

import torch
from torch.nn import functional as F

from .api import AcousticFWI, _frequencies
from .dispersion import assess_dispersion
from .inversion import InversionRunner, StepResult
from .optimizers import validate_optimizer
from .parameters import VelocityParameters
from .types import (Acquisition, BoundaryConfig, Grid2D, SolverConfig,
                    _integer, _real_scalar, validate_velocity)


@dataclass(frozen=True)
class MultiscaleStage:
    """One fixed grid, explicit frequency band and optimizer step count.

    ``boundary`` is explicit because keeping its cell count while refining a
    grid shrinks the physical PML. Frequencies retain the caller's given order.
    """
    grid: Grid2D
    frequencies_hz: tuple[float, ...]
    steps: int
    boundary: BoundaryConfig
    source_batch_size: int = 1

    def __post_init__(self):
        if not isinstance(self.grid, Grid2D):
            raise TypeError('stage grid must be Grid2D')
        if not isinstance(self.boundary, BoundaryConfig):
            raise TypeError('stage boundary must be BoundaryConfig')
        frequencies = _frequencies(_real_scalar(f, 'frequencies_hz') for f in self.frequencies_hz)
        object.__setattr__(self, 'frequencies_hz', frequencies)
        object.__setattr__(self, 'steps', _integer(self.steps, 'steps'))
        object.__setattr__(self, 'source_batch_size',
                           _integer(self.source_batch_size, 'source_batch_size', 1))


def _same_coordinates(a, b):
    # Tolerance permits arithmetic roundoff in SI coordinate construction, not
    # an origin tolerance proportional to a potentially huge global offset.
    return math.isclose(a, b, rel_tol=0., abs_tol=1e-9)


def _validate_refinement(source, target):
    if not isinstance(source, Grid2D) or not isinstance(target, Grid2D):
        raise TypeError('source_grid and target_grid must be Grid2D')
    if any(not _same_coordinates(a, b) for a, b in zip(source.origin_m, target.origin_m)):
        raise ValueError('stage grids must have the same origin; shifted domains cannot be refined')
    source_extent = tuple((n-1)*h for n, h in zip(source.shape, source.spacing_m))
    target_extent = tuple((n-1)*h for n, h in zip(target.shape, target.spacing_m))
    if any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
           for a, b in zip(source_extent, target_extent)):
        raise ValueError('stage grids must have the same physical node extent')
    if any(b > a*(1+1e-12) or nt < ns for a, b, ns, nt in
           zip(source.spacing_m, target.spacing_m, source.shape, target.shape)):
        raise ValueError('coarsening is unsupported; supply the initial velocity on the first/coarsest grid')


def refine_slowness_squared(m, source_grid, target_grid):
    """Detach and bilinearly refine physical m=1/v² on common endpoint nodes.

    Coordinates are node aligned (``align_corners=True``). Same-grid transfer
    returns an exact independent clone. Device and float32/64 precision are
    preserved. This is prolongation, with no downsampling or anti-alias claim.
    """
    _validate_refinement(source_grid, target_grid)
    if not isinstance(m, torch.Tensor):
        raise TypeError('slowness squared must be a torch.Tensor')
    if tuple(m.shape) != source_grid.shape:
        raise ValueError(f'slowness squared shape must equal source grid {source_grid.shape}')
    if m.dtype not in (torch.float32, torch.float64):
        raise ValueError('slowness squared must be real float32 or float64')
    if not bool(torch.isfinite(m).all()) or not bool((m > 0).all()):
        raise ValueError('slowness squared must be finite and strictly positive')
    if source_grid.shape == target_grid.shape:
        return m.detach().clone()
    return F.interpolate(m.detach()[None, None], size=target_grid.shape,
                         mode='bilinear', align_corners=True)[0, 0]


def _correct_transfer_bound_roundoff(velocity, bounds):
    """Repair only outward reconstruction errors <= 4*eps*the affected bound.

    Reciprocal, square, bilinear interpolation and reciprocal square root can
    put a valid frozen endpoint just outside the velocity interval. This error
    budget applies only to reconstruction after refinement, never initial user
    inputs. Inward values are unchanged, and corrections land exactly on the
    endpoint, where active-cell initialization still fails. Count and maximum
    correction are returned for the stage summary; substantive errors fail.
    """
    lower, upper = bounds
    eps = torch.finfo(velocity.dtype).eps
    if (not bool(torch.isfinite(velocity).all())
            or bool((lower-velocity > 4*eps*lower).any())
            or bool((velocity-upper > 4*eps*upper).any())):
        raise ValueError('refined velocity exceeds fixed bounds beyond the 4-epsilon transfer roundoff budget')
    outside = (velocity < lower) | (velocity > upper)
    count = int(outside.sum())
    if not count:
        return velocity, 0, 0.
    corrected = velocity.clamp(lower, upper)
    return corrected, count, float((corrected-velocity).abs().max())


@dataclass(frozen=True)
class MultiscaleStageResult:
    """Scalar stage summary; history steps and counters are stage local.

    ``pml_widths_m`` contains bottom/z and side/x widths. A free surface has
    zero top width; otherwise the top equals the reported bottom width.
    Dispersion metadata only screens the local homogeneous stencil symbol.
    Transfer correction metadata records any tiny outward bound excursions
    repaired after refinement, under the documented four-epsilon error budget.
    """
    index: int
    stage: MultiscaleStage
    initial_loss: float
    history: tuple[StepResult, ...]
    max_phase_error: float
    frequency_ceiling_hz: float
    pml_widths_m: tuple[float, float]
    transfer_bound_corrections: int = 0
    max_transfer_bound_correction_m_s: float = 0.


@dataclass(frozen=True)
class MultiscaleResult:
    """Detached final physical tensors and lightweight stage summaries."""
    grid: Grid2D
    velocity: torch.Tensor
    slowness_squared: torch.Tensor
    stages: tuple[MultiscaleStageResult, ...]


def _validate_fresh_optimizer(optimizer, model):
    validate_optimizer(optimizer, model)
    parameters = {p for group in optimizer.param_groups for p in group['params']}
    if any(p not in parameters for p in optimizer.state):
        raise ValueError('optimizer state contains parameters from another model; return a fresh optimizer')
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            if not state:
                continue
            # Native Adagrad eagerly creates its zero-step accumulator. This is
            # pristine native state, unlike a momentum/history warm start. Do
            # not reset it: nonzero initial_accumulator_value is caller policy.
            initial = group.get('initial_accumulator_value')
            if (initial is not None and set(state) == {'step', 'sum'}
                    and isinstance(state['step'], torch.Tensor)
                    and state['step'].numel() == 1 and float(state['step']) == 0.
                    and isinstance(state['sum'], torch.Tensor)
                    and state['sum'].shape == p.shape
                    and bool(torch.isfinite(state['sum']).all())
                    and bool((state['sum'] == initial).all())):
                continue
            raise ValueError('optimizer has existing state/history; optimizer_factory must return a fresh optimizer')


class MultiscaleInversion:
    """Execute a validated schedule once, retaining only its current runner.

    ``optimizer_factory(model, stage)`` must return a new native optimizer.
    Every recipe supported by :func:`make_optimizer` is supported here, with
    an appropriate ``model_factory(velocity, stage)`` for sparse parameters.
    A model factory must preserve the transferred initial physical velocity,
    bounds, shape, device and precision; it supplies target-grid masks itself.

    ``regularizer_factory(stage)`` runs once for each executed stage and must
    supply its fixed physical prior on that grid. It must not redefine a prior
    from the current iterate. ``configure_runner(index, stage, runner)`` runs
    before initial loss evaluation and can attach a fresh scheduler, recorder
    or accepted callbacks. Use distinct archive paths per stage: accepted-step
    counters restart at one. The hook cannot replace the stage model, objective,
    optimizer or source batch size. Schedulers must belong to the new optimizer
    and have fresh identities and history; native initial epoch zero is allowed.
    Accepted callbacks must be callable and must not change the stage objective.

    Same-grid transitions retain the exact detached velocity, avoiding a needless
    reciprocal round trip. After refinement only, outward reconstruction errors
    up to ``4*finfo(dtype).eps*bound`` are corrected to the exact affected bound
    and reported in the stage summary. Inward rounding is retained. Initial input
    bounds are strict, and active cells at an endpoint still fail; this correction
    never moves an active endpoint into the open interval.

    All static schedule/data/geometry/dispersion checks finish before optimizer
    creation. Factory and solve failures propagate; existing runner rollback
    semantics apply. Numerical caches are cleared on success or failure. The
    final ``runner.objective`` and ``runner.model`` remain available for fresh
    stage-local Curvature/TransformedCurvature. No optimization trajectory graph
    or intermediate model snapshots are retained internally.
    """
    def __init__(self, stages, acquisition, observed, frequencies_hz,
                 initial_velocity, velocity_bounds, solver_config,
                 optimizer_factory, *, weights=1., regularizer_factory=None,
                 model_factory=None, configure_runner=None, max_phase_error=.01):
        self.stages = tuple(stages)
        if not self.stages or any(not isinstance(s, MultiscaleStage) for s in self.stages):
            raise ValueError('stages must be a nonempty sequence of MultiscaleStage')
        if not isinstance(acquisition, Acquisition) or not isinstance(solver_config, SolverConfig):
            raise TypeError('acquisition and solver_config must be Acquisition and SolverConfig')
        self.acquisition, self.solver_config = acquisition, solver_config
        self.frequencies_hz = _frequencies(_real_scalar(f, 'frequencies_hz') for f in frequencies_hz)
        self.velocity_bounds = tuple(_real_scalar(v, 'velocity_bounds') for v in velocity_bounds)
        if (len(self.velocity_bounds) != 2 or
                not all(math.isfinite(v) for v in self.velocity_bounds) or
                not 0 < self.velocity_bounds[0] < self.velocity_bounds[1]):
            raise ValueError('velocity_bounds must be finite, positive, and strictly increasing')
        for name, factory in (('optimizer_factory', optimizer_factory),
                              ('regularizer_factory', regularizer_factory),
                              ('model_factory', model_factory), ('configure_runner', configure_runner)):
            if not callable(factory) and (name == 'optimizer_factory' or factory is not None):
                raise TypeError(f'{name} must be callable')
        self.optimizer_factory, self.regularizer_factory = optimizer_factory, regularizer_factory
        self.model_factory, self.configure_runner = model_factory, configure_runner
        validate_velocity(self.stages[0].grid, initial_velocity)
        if initial_velocity.dtype not in (torch.float32, torch.float64):
            raise ValueError('initial_velocity must use float32 or float64')
        lower, upper = self.velocity_bounds
        if not bool(((initial_velocity >= lower) & (initial_velocity <= upper)).all()):
            raise ValueError('initial_velocity must lie within fixed velocity_bounds')
        self._initial_velocity = initial_velocity.detach().clone()
        self.runner = None
        self._started = False
        self._optimizers = weakref.WeakSet()
        self._models = weakref.WeakSet()
        self._schedulers = weakref.WeakSet()
        self._metadata = []
        previous = None
        for index, stage in enumerate(self.stages):
            if previous is not None:
                _validate_refinement(previous.grid, stage.grid)
                if previous.boundary.top != stage.boundary.top:
                    raise ValueError('top boundary condition must remain fixed across stages')
                if max(stage.frequencies_hz) < max(previous.frequencies_hz):
                    raise ValueError('stage frequency maxima must be nondecreasing')
            if any(f not in self.frequencies_hz for f in stage.frequencies_hz):
                raise ValueError(f'stage {index} requests a frequency missing from master frequencies_hz')
            widths = tuple(stage.boundary.pml_cells*h for h in stage.grid.spacing_m)
            if previous is not None:
                previous_widths = self._metadata[-1][2]
                if any(w < old and not math.isclose(w, old, rel_tol=1e-12, abs_tol=1e-9)
                       for w, old in zip(widths, previous_widths)):
                    raise ValueError('physical PML widths must not shrink; increase stage boundary.pml_cells')
            report = assess_dispersion(stage.grid, self.velocity_bounds, stage.frequencies_hz,
                                       max_phase_error=max_phase_error)
            if not report.passed:
                raise ValueError(f'stage {index} fails dispersion screening at minimum allowed velocity '
                                 f'{lower:g} m/s; refine the grid or explicitly revise its frequency band '
                                 f'(frequency ceiling {report.frequency_ceiling_hz:g} Hz)')
            # Construct sequentially, without retaining earlier solvers/samplers.
            # This validates free-surface geometry and the declared device policy.
            facade = AcousticFWI(stage.grid, acquisition, stage.boundary, solver_config)
            try:
                facade._validate_precision(initial_velocity)
                if index == 0:
                    master = facade.objective(observed, self.frequencies_hz, weights=weights)
                    self._observed, self._weights = master.observed, master.weights
            finally:
                facade.solver.clear_cache()
            self._metadata.append((report.max_phase_error, report.frequency_ceiling_hz, widths))
            previous = stage
        self._metadata = tuple(self._metadata)

    def _make_model(self, velocity, stage):
        expected = velocity.detach().clone()
        model = (VelocityParameters(velocity, self.velocity_bounds) if self.model_factory is None
                 else self.model_factory(velocity, stage))
        if not isinstance(model, VelocityParameters):
            raise TypeError('model_factory must return VelocityParameters')
        if model in self._models:
            raise ValueError('model_factory must return a newly constructed model for every stage')
        actual = model.velocity().detach()
        if (model.bounds != self.velocity_bounds or tuple(actual.shape) != stage.grid.shape
                or actual.dtype != expected.dtype or actual.device != expected.device):
            raise ValueError('model_factory must preserve stage shape, fixed bounds, device and precision')
        if not torch.allclose(actual, expected, rtol=4*torch.finfo(expected.dtype).eps, atol=0.):
            raise ValueError('model_factory changed the transferred initial physical velocity')
        lower, upper = self.velocity_bounds
        if not bool(((actual[model.active_mask] > lower) & (actual[model.active_mask] < upper)).all()):
            raise ValueError('bounded mapping saturated at an endpoint; active velocities must remain strictly inside bounds')
        self._models.add(model)
        return model

    def _validate_configured_runner(self, runner, model, objective, optimizer, stage):
        if (runner.model is not model or runner.objective is not objective
                or runner.optimizer is not optimizer
                or runner.source_batch_size != stage.source_batch_size):
            raise ValueError('configure_runner cannot replace stage model, objective, optimizer or source_batch_size')
        callbacks = tuple(runner.accepted_callbacks)
        if any(not callable(callback) for callback in callbacks):
            raise TypeError('accepted_callbacks must contain callables')
        runner.accepted_callbacks = callbacks
        pending = [] if runner.scheduler is None else [runner.scheduler]
        current = set()
        while pending:
            scheduler = pending.pop()
            if getattr(scheduler, 'optimizer', None) is not optimizer:
                raise ValueError('configured scheduler must belong to the stage optimizer')
            if scheduler in self._schedulers or scheduler in current:
                raise ValueError('configured scheduler was reused; every stage requires a fresh scheduler')
            if (getattr(scheduler, 'last_epoch', -1) not in (-1, 0)
                    or getattr(scheduler, '_step_count', 0) > 1
                    or getattr(scheduler, 'num_bad_epochs', 0) != 0
                    or getattr(scheduler, 'cooldown_counter', 0) != 0
                    or (hasattr(scheduler, 'mode_worse')
                        and scheduler.best != scheduler.mode_worse)):
                raise ValueError('configured scheduler has existing history; return a fresh scheduler')
            if not callable(getattr(scheduler, 'step', None)):
                raise TypeError('configured scheduler must provide step()')
            current.add(scheduler)
            # SequentialLR and ChainedScheduler hold native child schedulers;
            # a fresh wrapper must not conceal an already-stepped child.
            pending.extend(getattr(scheduler, '_schedulers', ()))
        self._schedulers.update(current)

    def _run_stage(self, index, stage, velocity, corrections=0, max_correction=0.):
        facade = AcousticFWI(stage.grid, self.acquisition, stage.boundary, self.solver_config)
        try:
            model = self._make_model(velocity, stage)
            indices = torch.tensor([self.frequencies_hz.index(f) for f in stage.frequencies_hz],
                                   device=self._observed.device)
            regularizer = None if self.regularizer_factory is None else self.regularizer_factory(stage)
            if regularizer is not None:
                if not callable(getattr(regularizer, 'value', None)):
                    raise TypeError('regularizer_factory must return None or a physical regularizer')
                if getattr(regularizer, 'grid', stage.grid) != stage.grid:
                    raise ValueError('regularizer must use the current stage grid')
            objective = facade.objective(self._observed.index_select(0, indices), stage.frequencies_hz,
                weights=self._weights.index_select(0, indices), regularizer=regularizer)
            optimizer = self.optimizer_factory(model, stage)
            _validate_fresh_optimizer(optimizer, model)
            if optimizer in self._optimizers:
                raise ValueError('optimizer_factory reused an optimizer; every stage requires a fresh optimizer')
            self._optimizers.add(optimizer)
            runner = InversionRunner(objective, model, optimizer, source_batch_size=stage.source_batch_size)
            self.runner = runner
            if self.configure_runner is not None:
                self.configure_runner(index, stage, runner)
            self._validate_configured_runner(runner, model, objective, optimizer, stage)
            initial_loss = runner._accepted_loss()
            history = tuple(runner.step() for _ in range(stage.steps))
            final_velocity = model.velocity().detach()
            lower, upper = self.velocity_bounds
            if not bool(((final_velocity[model.active_mask] > lower)
                         & (final_velocity[model.active_mask] < upper)).all()):
                raise ValueError('bounded mapping saturated at an endpoint; active velocities must remain strictly inside bounds')
            m = model.slowness_squared().detach().clone()
            maximum, ceiling, widths = self._metadata[index]
            return MultiscaleStageResult(index, stage, initial_loss, history, maximum,
                ceiling, widths, corrections, max_correction), m
        except BaseException as exc:
            if hasattr(exc, 'add_note'):
                exc.add_note(f'Multiscale stage {index}: grid={stage.grid.shape}, frequencies_hz={stage.frequencies_hz}')
            raise
        finally:
            facade.solver.clear_cache()

    def run(self):
        """Run once; a failed or successful execution cannot silently restart."""
        if self._started:
            raise RuntimeError('MultiscaleInversion.run() may be called only once; construct a new inversion')
        self._started = True
        velocity, self._initial_velocity = self._initial_velocity, None
        results = []
        for index, stage in enumerate(self.stages):
            corrections, max_correction = 0, 0.
            if index:
                if stage.grid.shape == self.stages[index-1].grid.shape:
                    velocity = self.runner.model.velocity().detach().clone()
                else:
                    velocity = refine_slowness_squared(m, self.stages[index-1].grid, stage.grid).rsqrt()
                    velocity, corrections, max_correction = _correct_transfer_bound_roundoff(
                        velocity, self.velocity_bounds)
            # Drop the previous model, optimizer, solver and regularizer before
            # allocating a new stage. Caller callbacks may retain their own data.
            self.runner = None
            summary, m = self._run_stage(index, stage, velocity, corrections, max_correction)
            results.append(summary)
        return MultiscaleResult(self.stages[-1].grid,
            self.runner.model.velocity().detach().clone(), m, tuple(results))
