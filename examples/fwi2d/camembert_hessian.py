"""Reproduce source-streamed physical Hessian checks on 16/24-cell windows.

Same-grid synthetic observations intentionally validate algebraic derivatives;
these results are not an independent reconstruction or continuum physics test.
No explicit Jacobian or Hessian is assembled on the Camembert grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Make the documented direct script invocation work from a source checkout.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from examples.fwi2d.camembert import DEFAULT_CONFIG, make_camembert
from simplefwi.fwi2d.curvature import Curvature
from simplefwi.fwi2d.objective import FWIObjective
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers.direct import DirectSolver
from simplefwi.fwi2d.types import SolverConfig


def relative(actual, reference):
    return float((actual-reference).norm()/reference.norm().clamp_min(torch.finfo(reference.real.dtype).tiny))


def gradient(objective, m, mask):
    m = m.detach().clone().requires_grad_()
    total = torch.zeros_like(m)
    for loss in objective.iter_data_losses(m, 1):
        total += torch.autograd.grad(loss, m)[0].detach()
    return total[mask]


def run(fixture, *, window='both', frequency=5.):
    torch.set_num_threads(1)
    torch.manual_seed(217)
    started = time.perf_counter()
    config = json.loads(Path(fixture).read_text())
    grid, true, initial, acquisition, boundary = make_camembert(fixture)
    operator = HelmholtzOperator(grid, boundary)
    solver = DirectSolver(SolverConfig(rtol=1e-10, atol=1e-12, memory_budget_bytes=512*1024**2))
    shape = (1, len(acquisition.sources_m), len(acquisition.receivers_m))
    empty = torch.zeros(shape, dtype=torch.complex128)
    weights = torch.linspace(0.7, 1.3, shape[-1], dtype=torch.float64)
    objective = FWIObjective(operator, acquisition, (frequency,), empty, weights, None, solver)
    m = initial.reciprocal().square()
    with torch.no_grad():
        observed = objective.predict(true.reciprocal().square(), 1)
        predicted = objective.predict(m, 1)
    objective.observed = observed.detach()
    zero = FWIObjective(operator, acquisition, (frequency,), predicted, weights, None, solver)
    output = {'coordinates': 'physical_slowness_squared', 'model_units': 's^2/m^2',
              'data_axes': ['frequency', 'source', 'receiver'], 'model_axes': ['z', 'x'],
              'classification': 'same-grid algebraic derivative validation',
              'frequency_hz': frequency, 'physical_shape': list(grid.shape),
              'pml_cells': boundary.pml_cells, 'source_count': shape[1],
              'receiver_count': shape[2], 'source_batch_size': 1,
              'dtype': 'float64/complex128', 'solver_rtol': solver.config.rtol,
              'solver_atol': solver.config.atol,
              'relative_data_residual': relative(predicted, observed), 'windows': {}}
    windows = ('routine', 'extended') if window == 'both' else (window,)
    for name in windows:
        z0,z1,x0,x1 = config['active_windows'][name]
        mask = torch.zeros_like(m, dtype=torch.bool); mask[z0:z1,x0:x1] = True
        c = Curvature(objective, m, active_mask=mask)
        v,w = torch.randn((2,c.n_active), dtype=m.dtype)
        v /= v.norm(); w /= w.norm()
        step = float(m.mean())*1e-3
        dm = c.inject(v)
        jv = c.jvp(v)
        with torch.no_grad():
            data_fd = (objective.predict(m+step*dm,1)-objective.predict(m-step*dm,1))/(2*step)
        hv = c.exact_hvp(v)
        fd = (gradient(objective,m+step*dm,mask)-gradient(objective,m-step*dm,mask))/(2*step)
        hw = c.exact_hvp(w)
        gn = c.data_gn_hvp(v)
        exact_zero = Curvature(zero,m,active_mask=mask).exact_hvp(v)
        cotangent = torch.randn_like(observed)
        left = (jv.conj()*cotangent).real.sum()
        right = v@c.vjp(cotangent)
        report = {'active_shape': [z1-z0,x1-x0], 'n_active': c.n_active,
                  'finite_difference_step_m': step,
                  'jvp_fd_relative_error': relative(jv,data_fd),
                  'exact_hvp_fd_relative_error': relative(hv,fd),
                  'hessian_symmetry_relative_error': float((w@hv-v@hw).abs()/(w.norm()*hv.norm()+v.norm()*hw.norm())),
                  'jacobian_adjoint_relative_error': float((left-right).abs()/torch.maximum(left.abs(),right.abs())),
                  'zero_residual_exact_gn_relative_error': relative(exact_zero,gn),
                  'nonzero_residual_exact_gn_relative_difference': relative(hv,gn)}
        assert report['jvp_fd_relative_error'] <= 1e-5, report
        assert report['exact_hvp_fd_relative_error'] <= 1e-5, report
        assert report['hessian_symmetry_relative_error'] <= 1e-8, report
        assert report['jacobian_adjoint_relative_error'] <= 1e-8, report
        assert report['zero_residual_exact_gn_relative_error'] <= 1e-8, report
        assert report['nonzero_residual_exact_gn_relative_difference'] > 1e-3, report
        output['windows'][name] = report
    output['elapsed_seconds'] = time.perf_counter()-started
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--window', choices=('routine','extended','both'), default='both')
    parser.add_argument('--frequency', type=float, default=5.)
    args = parser.parse_args()
    print(json.dumps(run(args.fixture, window=args.window, frequency=args.frequency), indent=2))
