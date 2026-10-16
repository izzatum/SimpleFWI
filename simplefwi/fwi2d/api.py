"""Public SI acoustic modelling façade with bounded source working storage."""
from __future__ import annotations

import math
import numbers
import torch

from .types import Acquisition, BoundaryConfig, Grid2D, SolverConfig, validate_velocity
from .acquisition import SamplingOperator
from .operator import HelmholtzOperator
from .objective import FWIObjective
from .recording import ArrayPayload
from .inversion import StepResult
from .solvers import DirectSolver


def _frequencies(values):
    values = tuple(float(f) for f in values)
    if not values or any(not math.isfinite(f) or f <= 0 for f in values):
        raise ValueError('frequencies_hz must be nonempty, finite and positive')
    if len(set(values)) != len(values):
        raise ValueError('frequencies_hz must be unique')
    return values


class AcousticFWI:
    """Acoustic frequency-domain data in [frequency, source, receiver] order.

    Velocities are explicit SI tensors [z,x]. ``simulate`` performs detached
    forward modelling; ``objective`` exposes differentiable physical slowness
    squared through FWIObjective. Neither method changes numerical precision,
    device, grid or boundary to make a requested solve succeed. FGMRES uses the
    shifted-Helmholtz multigrid preconditioner; CPU sparse direct is the reference.
    """
    def __init__(self, grid, acquisition, boundary, solver_config):
        if not isinstance(grid, Grid2D) or not isinstance(acquisition, Acquisition):
            raise TypeError('grid and acquisition must be Grid2D and Acquisition')
        if not isinstance(boundary, BoundaryConfig) or not isinstance(solver_config, SolverConfig):
            raise TypeError('boundary and solver_config must be BoundaryConfig and SolverConfig')
        self.grid, self.acquisition, self.boundary = grid, acquisition, boundary
        self.solver_config = solver_config
        self.operator = HelmholtzOperator(grid, boundary)
        self.sources = SamplingOperator(grid, acquisition.sources_m, boundary=boundary)
        self.receivers = SamplingOperator(grid, acquisition.receivers_m, boundary=boundary)
        self.preconditioner_config = None
        if solver_config.method == 'direct':
            self.solver = DirectSolver(solver_config)
        else:
            from .solvers.krylov import FGMRESSolver
            from .solvers.multigrid import ShiftedHelmholtzMultigrid, MultigridConfig
            if torch.device(solver_config.device).type == 'cuda' and not torch.cuda.is_available():
                raise ValueError('requested CUDA device is unavailable')
            self.preconditioner_config = MultigridConfig(shift=.5, max_levels=8, coarse_max_dof=64,
                memory_budget_bytes=solver_config.memory_budget_bytes)
            preconditioner = ShiftedHelmholtzMultigrid(self.operator, self.preconditioner_config)
            self.solver = FGMRESSolver(solver_config, preconditioner=preconditioner)
        self.last_solve_reports = ()

    def _validate_precision(self, value):
        device = torch.device(self.solver_config.device)
        # Unindexed cuda means the explicitly selected current CUDA device.
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        real = torch.float64 if self.solver_config.dtype == torch.complex128 else torch.float32
        if value.device != device or value.dtype not in (real, self.solver_config.dtype):
            raise ValueError('tensor device and precision must match solver_config explicitly')

    def objective(self, observed, frequencies_hz, *, weights=1.0, regularizer=None):
        """Build the explicit physical-m objective, retaining this solver policy."""
        frequencies = _frequencies(frequencies_hz)
        if isinstance(observed, torch.Tensor):
            self._validate_precision(observed)
        return FWIObjective(self.operator, self.acquisition, frequencies, observed,
                            weights, regularizer, self.solver)

    @torch.no_grad()
    def simulate(self, velocity, frequencies_hz, *, source_batch_size=1, recorder=None):
        """Return detached complex data; discard each sequential batch wavefield.

        Optional wavefields are recorded only for selected sources/frequencies.
        Each selected batch is a separate explicitly accepted simulation record
        (loss=0); use a fresh archive, separate from optimizer step archives.
        Wavefield pressure uses normalized source units, denoted ``source_unit``.
        """
        validate_velocity(self.grid, velocity)
        self._validate_precision(velocity)
        frequencies = _frequencies(frequencies_hz)
        if (isinstance(source_batch_size, bool) or not isinstance(source_batch_size, numbers.Integral)
                or source_batch_size < 1):
            raise ValueError('source_batch_size must be a positive integer')
        ns, nr = len(self.acquisition.sources_m), len(self.acquisition.receivers_m)
        data = torch.empty((len(frequencies), ns, nr), dtype=self.solver_config.dtype,
                           device=velocity.device)
        m = velocity.reciprocal().square()
        reports = []
        record_number = 0
        try:
            for fi, frequency in enumerate(frequencies):
                for start in range(0, ns, source_batch_size):
                    stop = min(start+source_batch_size, ns)
                    sampler = SamplingOperator(self.grid, self.acquisition.sources_m[start:stop],
                                               boundary=self.boundary)
                    amplitudes = torch.tensor(self.acquisition.amplitudes[start:stop],
                                              dtype=data.dtype, device=data.device)
                    rhs = sampler.source_rhs(torch.diag(amplitudes))
                    wave, report = self.solver.solve(self.operator, m, 2*math.pi*frequency, rhs)
                    reports.append(report)
                    sampled = self.receivers.sample(wave)
                    data[fi, start:stop] = sampled
                    if recorder is not None:
                        policy = recorder.policy
                        selected = [i for i in range(start, stop)
                                    if policy.sources is None or i in policy.sources]
                        if not selected or (policy.frequencies is not None and frequency not in policy.frequencies):
                            continue
                        local = torch.tensor([i-start for i in selected], device=wave.device)
                        coords = dict(frequencies=(frequency,), sources=tuple(selected),
                                      coordinates={'spacing_m': self.grid.spacing_m})
                        payload = {}
                        if 'physical_wavefield' in policy.fields:
                            payload['physical_wavefield'] = ArrayPayload(
                                self.operator.physical(wave[local])[None], ('frequency','source','z','x'),
                                'source_unit', origin=self.grid.origin_m, region='physical', **coords)
                        if 'pml_wavefield' in policy.fields:
                            p = self.boundary.pml_cells
                            origin = (self.grid.origin_m[0]-(p if self.boundary.top == 'pml' else 0)*self.grid.spacing_m[0],
                                      self.grid.origin_m[1]-p*self.grid.spacing_m[1])
                            payload['pml_wavefield'] = ArrayPayload(
                                self.operator.unpack(wave[local])[None],
                                ('frequency','source','z_padded','x_padded'), 'source_unit',
                                origin=origin, region='pml_padded', **coords)
                        if 'predicted_data' in policy.fields:
                            payload['predicted_data'] = ArrayPayload(sampled[local][None],
                                ('frequency','source','receiver'), 'source_unit', **coords)
                        if payload:
                            record_number += 1
                            recorder.record_accepted(StepResult(record_number, 0., 0,
                                {'kind': 'simulation', 'frequency_hz': frequency}), payload)
                # No frequency sweep of numeric factors survives forward modelling.
                self.solver.clear_cache()
        finally:
            self.last_solve_reports = tuple(reports)
            self.solver.clear_cache()
        return data
