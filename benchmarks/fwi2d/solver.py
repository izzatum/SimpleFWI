"""Reproducible equal-accuracy solver/implicit-gradient/HVP benchmark.

Run from the repository root: python benchmarks/fwi2d/solver.py --output /tmp/solver.json
The fixed cases increase domain size, frequency and velocity contrast; they are
algebraic performance fixtures, not claims of dispersion or PML calibration.
No cloud resources are created. --device cuda requires an existing CUDA device.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

# Support direct script invocation without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from simplefwi.fwi2d.autograd import implicit_solve
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers.base import SolverFailure
from simplefwi.fwi2d.solvers.direct import DirectSolver
from simplefwi.fwi2d.solvers.krylov import FGMRESSolver
from simplefwi.fwi2d.solvers.multigrid import MultigridConfig, ShiftedHelmholtzMultigrid
from simplefwi.fwi2d.types import BoundaryConfig, Grid2D, SolverConfig


CASES = (
    ((8, 12), 5., 1.),
    ((16, 24), 10., 2.),
    ((24, 36), 20., 2.),
    ((32, 48), 30., 3.),
)


def accuracy_criteria(dtype):
    """Declared direct-reference gates; derivative coordinates are explicit."""
    name = str(dtype)
    if name not in ('torch.complex128', 'torch.complex64'):
        raise ValueError('benchmark accuracy requires complex128 or complex64')
    reference_precision = name == 'torch.complex128'
    return {
        'dtype': name,
        'comparison': 'relative Euclidean error against CPU complex128 direct solve',
        'maximum_relative_errors': {
            'waveform_relative_error': 1e-7 if reference_precision else 1e-3,
            'gradient_relative_error': 1e-7 if reference_precision else 1e-2,
            'hvp_relative_error': 1e-5 if reference_precision else 1e-2,
        },
        'gradient_and_hvp_coordinates': 'theta=log(m/m_reference)',
        'threshold_source': {
            'waveform': 'FWI redesign specification acceptance gate 6 precision limits',
            'gradient': ('Gate 6 numeric precision limits adopted for log-slowness '
                         'benchmark gradients; physical-gradient checks are separate tests'),
            'hvp': ('Gate 5 1e-5 derivative consistency target adopted for direct-reference '
                    'complex128 HVP comparison; complex64 1e-2 is an explicit experimental '
                    'benchmark design choice, not a specification or uncertainty acceptance gate'),
        },
        'acceptance': 'all three errors must be present, finite, nonnegative and <= their limits',
    }


def classify_accuracy(measured, dtype):
    """Return a classified copy, preserving all observed metrics and failures."""
    result = dict(measured)
    criteria = accuracy_criteria(dtype)
    result['accuracy_criteria'] = criteria
    if measured.get('status') == 'failed':
        result['accuracy_evaluation'] = 'unrun_after_failed_solve'
        return result
    failures = []
    for name, limit in criteria['maximum_relative_errors'].items():
        value = measured.get(name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0 <= value <= limit):
            failures.append(name)
    result['accuracy_failures'] = failures
    result['accuracy_evaluation'] = 'failed' if failures else 'passed'
    result['status'] = 'failed_accuracy' if failures else 'passed'
    return result


class ObservedSolver:
    def __init__(self, solver):
        self.solver, self.reports = solver, []

    def solve(self, *args, **kwargs):
        solution, report = self.solver.solve(*args, **kwargs)
        self.reports.append(report.to_dict())
        return solution, report


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed(call, device):
    synchronize(device)
    start = time.perf_counter()
    result = call()
    synchronize(device)
    return result, time.perf_counter()-start


def measure(operator, m, omega, rhs, solver, device):
    observer = ObservedSolver(solver)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    if isinstance(solver, FGMRESSolver):
        _, setup = timed(lambda: solver.preconditioner.prepare(m, omega), device)
    else:
        # Sparse factorization has no separate prepare API; cold solve reports
        # its combined assembly/factorization/RHS cost below.
        setup = None
    (cold, cold_report), cold_seconds = timed(lambda: observer.solve(operator, m, omega, rhs), device)
    (_, _), warm_seconds = timed(lambda: observer.solve(operator, m, omega, rhs), device)
    # Real trainable log-slowness coordinates keep SI-scale Hessians readable.
    theta = torch.zeros_like(m, requires_grad=True)
    state = {}
    def forward():
        state['u'] = implicit_solve(m*theta.exp(), rhs, omega, operator, observer)
        # Fixed zero target, normalized by the reference primal norm. This
        # nonzero residual exercises exact second-order (not merely GN) terms.
        state['loss'] = .5*state['u'].abs().square().sum()/cold.abs().square().sum().detach()
        return state['u']
    _, forward_seconds = timed(forward, device)
    gradient, gradient_seconds = timed(lambda: torch.autograd.grad(state['loss'], theta, create_graph=True)[0], device)
    direction = torch.linspace(-1., 1., theta.numel(), dtype=m.dtype, device=m.device).reshape(m.shape)
    direction /= direction.norm()
    hvp, hvp_seconds = timed(lambda: torch.autograd.grad((gradient*direction).sum(), theta)[0], device)
    result = {
        'status': 'solved' if isinstance(solver, FGMRESSolver) else 'passed',
        'setup_seconds': setup,
        'setup_plus_all_rhs_seconds': cold_seconds+(setup or 0.),
        'warm_all_rhs_seconds': warm_seconds,
        'implicit_forward_seconds': forward_seconds,
        'implicit_gradient_increment_seconds': gradient_seconds,
        'implicit_hvp_increment_seconds': hvp_seconds,
        'maximum_relative_residual': max(r['relative_residual'] for r in observer.reports),
        'maximum_iterations_per_rhs': max(r['iterations'] for r in observer.reports),
        'estimated_tensor_bytes': max(r['memory_bytes'] for r in observer.reports),
        'cuda_peak_allocated_bytes': (torch.cuda.max_memory_allocated(device)
                                      if torch.device(device).type == 'cuda' else None),
        'solve_calls': len(observer.reports),
        'primal_report': cold_report.to_dict(),
        'solve_reports': observer.reports,
        'solver_config': solver.config.to_dict(),
    }
    if isinstance(solver, FGMRESSolver):
        result.update(multigrid_config=solver.preconditioner.config.to_dict(),
                      hierarchy_shapes=[level.padded_shape for level in solver.preconditioner.levels],
                      basis_peak_vectors=solver.basis_peak_vectors)
    return result, (cold.detach().cpu(), gradient.detach().cpu(), hvp.detach().cpu())


def run(device='cpu', *, quick=False, maxiter=120, dtype=torch.complex128):
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no device fallback is permitted')
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    # Untimed tiny solves initialize autograd and backend kernels equally.
    warm_operator = HelmholtzOperator(Grid2D((2, 3), (.4, .6)),
        BoundaryConfig(pml_cells=2, sigma_max=5.))
    warm_model = torch.full((2, 3), .6, dtype=torch.float64)
    warm_rhs = torch.ones((2, warm_operator.n_dof), dtype=torch.complex128)
    measure(warm_operator, warm_model, 2.3, warm_rhs, DirectSolver(), 'cpu')
    warm_mg = ShiftedHelmholtzMultigrid(warm_operator,
        MultigridConfig(shift=.5, max_levels=5, coarse_max_dof=4))
    warm_solver = FGMRESSolver(SolverConfig(method='fgmres', device=device, dtype=dtype,
        rtol=1e-11 if real == torch.float64 else 2e-6,
        atol=1e-13 if real == torch.float64 else 1e-8), warm_mg)
    measure(warm_operator, warm_model.to(device=device, dtype=real), 2.3,
            warm_rhs.to(device=device, dtype=dtype), warm_solver, device)
    records = []
    for shape, frequency, contrast in CASES[:1 if quick else len(CASES)]:
        operator = HelmholtzOperator(Grid2D(shape, (10., 12.), (13., -17.)),
            BoundaryConfig(pml_cells=3, sigma_max=80.))
        velocity = torch.full(shape, 1500., dtype=torch.float64)
        velocity[shape[0]//3:2*shape[0]//3, shape[1]//3:2*shape[1]//3] *= contrast
        model = velocity.reciprocal().square()
        rhs = torch.randn((2, operator.n_dof), dtype=torch.complex128,
                          generator=torch.Generator().manual_seed(20261015))
        omega = 2*math.pi*frequency
        base, reference = measure(operator, model, omega, rhs,
                                  DirectSolver(SolverConfig(rtol=1e-11, atol=1e-13)), 'cpu')
        record = dict(shape=shape, frequency_hz=frequency, velocity_contrast=contrast,
                      n_dof=operator.n_dof, rhs_count=2, direct_cpu=base)
        config = SolverConfig(method='fgmres', rtol=1e-11 if real == torch.float64 else 2e-6,
                              atol=1e-13 if real == torch.float64 else 1e-8,
                              restart=30, maxiter=maxiter, device=device, dtype=dtype,
                              memory_budget_bytes=512*1024**2)
        mg_config = MultigridConfig(shift=.5, max_levels=8, coarse_max_dof=36,
                                   memory_budget_bytes=256*1024**2)
        solver = FGMRESSolver(config, ShiftedHelmholtzMultigrid(operator, mg_config))
        attempt_start = time.perf_counter()
        try:
            measured, values = measure(operator, model.to(device=device, dtype=real), omega,
                                       rhs.to(device=device, dtype=dtype), solver, device)
            for name, actual, expected in zip(('waveform', 'gradient', 'hvp'), values, reference):
                measured[name+'_relative_error'] = float((actual-expected).norm()/expected.norm())
            record['fgmres'] = measured
        except SolverFailure as exc:
            record['fgmres'] = dict(status='failed', message=str(exc), report=exc.report.to_dict(),
                                    elapsed_until_failure_seconds=time.perf_counter()-attempt_start,
                                    derivative_validation='unrun after failed solve',
                                    solver_config=config.to_dict(), multigrid_config=mg_config.to_dict())
        record['fgmres'] = classify_accuracy(record['fgmres'], dtype)
        records.append(record)
    return dict(python=platform.python_version(), torch=torch.__version__, device=device,
                cuda_available=torch.cuda.is_available(), cuda_validation='executed' if device.startswith('cuda') else 'unrun',
                seed=20261015, dtype=str(dtype), threads=torch.get_num_threads(),
                scope='algebraic fixtures; no dispersion, PML calibration or GPU speedup claim',
                timing='untimed tiny backend/autograd warmup; warm forward, then incremental backward with create_graph, then exact HVP; setup+all RHS also measured',
                memory='tensor estimates; CUDA peak allocator bytes when available; CPU RSS is not inferred', records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--maxiter', type=int, default=120)
    parser.add_argument('--complex64', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    # One CPU thread makes the small-system baseline reproducible and prevents
    # thread-pool startup from overwhelming the actual sparse/tensor timings.
    torch.set_num_threads(1)
    result = run(args.device, quick=args.quick, maxiter=args.maxiter,
                 dtype=torch.complex64 if args.complex64 else torch.complex128)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload+'\n')
    print(payload)


if __name__ == '__main__':
    main()
