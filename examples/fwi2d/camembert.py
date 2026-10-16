"""Pinned Camembert geometry and explicitly classified reference data.

No inversion-accuracy claim can use same-grid data. The refined reference
keeps the physical circle and acquisition fixed while enlarging the domain.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from simplefwi.fwi2d.acquisition import SamplingOperator
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers import DirectSolver
from simplefwi.fwi2d.types import Acquisition, BoundaryConfig, Grid2D, SolverConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'tests/fwi2d/fixtures/camembert.json'


def _models(grid, circle):
    z = grid.origin_m[0] + torch.arange(grid.shape[0], dtype=torch.float64)*grid.spacing_m[0]
    x = grid.origin_m[1] + torch.arange(grid.shape[1], dtype=torch.float64)*grid.spacing_m[1]
    radius2 = ((z[:, None]-circle['centre_m'][0])**2
               + (x[None, :]-circle['centre_m'][1])**2)
    initial = torch.full(grid.shape, circle['background_m_s'], dtype=torch.float64)
    true = torch.where(radius2 <= circle['radius_m']**2, circle['inclusion_m_s'], initial)
    return true, initial


def make_camembert(config_path=DEFAULT_CONFIG, *, acquisition='surrounding'):
    """Return grid, true/initial velocity, acquisition, boundary in SI units.

    ``acquisition='surface'`` selects the separately pinned surface geometry;
    this does not silently change the pressure boundary condition.
    """
    config = json.loads(Path(config_path).read_text())
    if acquisition not in ('surrounding', 'surface'):
        raise ValueError("acquisition must be 'surrounding' or 'surface'")
    grid = Grid2D(**config['grid'])
    true, initial = _models(grid, config['circle'])
    geometry = config['acquisition' if acquisition == 'surrounding' else 'surface_acquisition']
    return grid, true, initial, Acquisition(**geometry), BoundaryConfig(**config['boundary'])


def make_refined_camembert(config_path=DEFAULT_CONFIG, *, refinement=2, margin_m=100.,
                          acquisition='surrounding'):
    """Larger/finer reference geometry, with circle unchanged in metres.

    Refine the PML too, holding its physical thickness and damping fixed.
    A free surface keeps its physical elevation: add vertical margin only
    below the model, while horizontal margins remain on both sides.
    This is an independent discretization, not an exact continuum solution.
    """
    if isinstance(refinement, bool) or not isinstance(refinement, int) or refinement < 2:
        raise ValueError('refinement must be an integer >= 2')
    grid, _, _, geometry, boundary = make_camembert(config_path, acquisition=acquisition)
    if not torch.isfinite(torch.tensor(margin_m)) or margin_m <= 0:
        raise ValueError('margin_m must be finite and positive')
    spacing = tuple(h/refinement for h in grid.spacing_m)
    margins = tuple(round(margin_m/h) for h in spacing)
    if any(abs(n*h-margin_m) > 1e-9 for n,h in zip(margins,spacing)):
        raise ValueError('margin_m must align with the refined grid')
    lower_margins = (0 if boundary.top == 'free_surface' else margins[0], margins[1])
    fine = Grid2D(
        tuple((n-1)*refinement+1+low+high
              for n,low,high in zip(grid.shape,lower_margins,margins)),
        spacing, tuple(o-low*h for o,low,h in zip(grid.origin_m,lower_margins,spacing)))
    config = json.loads(Path(config_path).read_text())
    true, initial = _models(fine, config['circle'])
    return fine, true, initial, geometry, BoundaryConfig(
        top=boundary.top, pml_cells=boundary.pml_cells*refinement,
        sigma_max=boundary.sigma_max, alpha=boundary.alpha)


def simulate_data(grid, velocity, acquisition, boundary, frequencies_hz, *, source_batch=2):
    """CPU complex128 reference observations in [frequency, source, receiver].

    Only the small sampled data persist; each frequency/batch's wavefield is
    discarded. The caller must classify same-grid versus independent data.
    """
    if isinstance(source_batch, bool) or not isinstance(source_batch, int) or source_batch < 1:
        raise ValueError('source_batch must be a positive integer')
    frequencies_hz = tuple(float(f) for f in frequencies_hz)
    if not frequencies_hz or any(not 0 < f < float('inf') for f in frequencies_hz):
        raise ValueError('frequencies_hz must contain positive finite values')
    operator = HelmholtzOperator(grid, boundary)
    sources = SamplingOperator(grid, acquisition.sources_m, boundary=boundary)
    receivers = SamplingOperator(grid, acquisition.receivers_m, boundary=boundary)
    solver = DirectSolver(SolverConfig(memory_budget_bytes=512*1024**2))
    amplitudes = torch.tensor(acquisition.amplitudes, dtype=torch.complex128)
    data = []
    for frequency in frequencies_hz:
        batches = []
        for start in range(0, len(amplitudes), source_batch):
            values = torch.diag(amplitudes)[start:start+source_batch]
            wavefield, _ = solver.solve(operator, velocity.reciprocal().square(),
                                        2*torch.pi*frequency, sources.source_rhs(values))
            batches.append(receivers.sample(wavefield))
        data.append(torch.cat(batches))
        solver.clear_cache()
    return torch.stack(data)
