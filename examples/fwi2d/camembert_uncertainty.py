"""Tiny active-window algebraic uncertainty check on pinned Camembert geometry.

Same-grid observations validate operator algebra, not reconstruction accuracy.
The 2x2 active window permits the explicit covariance check; production APIs
never assemble a full-grid Hessian or covariance. Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from examples.fwi2d.camembert import DEFAULT_CONFIG, make_camembert, simulate_data
from simplefwi.fwi2d.curvature import Curvature, explicit_matrix
from simplefwi.fwi2d.objective import FWIObjective
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers import DirectSolver
from simplefwi.fwi2d.uncertainty import CGConfig, LocalGaussian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default=DEFAULT_CONFIG)
    parser.add_argument('--probes', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)  # Small CPU reference: avoid thread-pool overhead.
    grid, true, initial, acquisition, boundary = make_camembert(args.fixture)
    frequencies = (5.,)
    noise_std = 1e-5  # declared std of each real and imaginary residual component
    observed = simulate_data(grid, true, acquisition, boundary, frequencies)
    objective = FWIObjective(HelmholtzOperator(grid, boundary), acquisition,
                             frequencies, observed, 1/noise_std, None, DirectSolver())
    m = initial.reciprocal().square()
    mask = torch.zeros_like(m, dtype=torch.bool)
    mask[23:25, 23:25] = True
    curvature = Curvature(objective, m, active_mask=mask, source_batch_size=2)
    # Independent m-cell prior standard deviation is ten percent of background m.
    prior = (0.1*m[mask]).square().reciprocal()
    gaussian = LocalGaussian(curvature, prior, solve_config=CGConfig(rtol=1e-10))
    data = explicit_matrix(curvature.data_gn_hvp, curvature.n_active, max_bytes=4096)
    covariance = torch.linalg.inv(data+torch.diag(prior))  # tiny diagnostic only
    v = torch.arange(1., 5., dtype=m.dtype)
    inverse = gaussian.covariance_action(v)
    inverse_report = gaussian.last_solve
    relative_error = float((inverse-covariance@v).norm()/(covariance@v).norm())
    estimate = gaussian.diagonal_estimate(args.probes, seed=19)
    modes = gaussian.leading_modes(4, seed=7)
    mode_error = float((modes.covariance_action(v)-covariance@v).norm()/(covariance@v).norm())
    if relative_error > 1e-8 or mode_error > 1e-8:
        raise RuntimeError('tiny explicit covariance comparison failed')
    print(json.dumps({
        'classification': 'same-discretization algebraic local-Gaussian diagnostic',
        'coordinates': gaussian.metadata['coordinates'], 'active_shape': [2, 2],
        'frequency_hz': frequencies, 'noise_std_each_real_imag': noise_std,
        'prior_std_fraction_of_background_m': .1,
        'inverse_relative_error': relative_error,
        'inverse_true_relative_residual': inverse_report.relative_residual,
        'diagonal_exact_s4_m4': covariance.diagonal().tolist(),
        'diagonal_estimate_s4_m4': estimate.mean.tolist(),
        'diagonal_standard_error_s4_m4': estimate.standard_error.tolist(),
        'diagonal_inner_solve_error_bound': estimate.max_solve_error_bound,
        'probes': estimate.probes, 'probe_seed': estimate.seed,
        'mode_eigenvalues': modes.eigenvalues.tolist(),
        'mode_generalized_residual_norms': modes.ritz_residual_norms.tolist(),
        'full_rank_covariance_relative_error': mode_error,
    }, indent=2))


if __name__ == '__main__':
    main()
