"""PML calibration: true plane incidence and independent two-dimensional corners.

Plane tests take a tangential Fourier section of actual assembled 2D rows.
Only x is truncated; this removes edge diffraction from the angle experiment.
The full 2D point-source test then exercises simultaneous x/z corner stretches.
"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import torch

from simplefwi.fwi2d.acquisition import SamplingOperator
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers import DirectSolver
from simplefwi.fwi2d.types import BoundaryConfig, Grid2D, SolverConfig


def plane_field(frequency, angle, h, thickness, *, extent=600.):
    k = 2*np.pi*frequency/1500
    angle = np.deg2rad(angle)
    # Numerical wavevector points at exactly the requested angle, and obeys
    # the interior symbol. Thus dispersion is not mistaken for reflection.
    kn = brentq(lambda t: 4*(np.sin(t*h*np.cos(angle)/2)**2
                            + np.sin(t*h*np.sin(angle)/2)**2)/h**2-k*k, k, 1.5*k)
    kx, kz = kn*np.cos(angle), kn*np.sin(angle)
    p = round(thickness/h)
    grid = Grid2D((3,round(extent/h)+1),(h,h))
    boundary = BoundaryConfig(pml_cells=p,sigma_max=60.)
    operator = HelmholtzOperator(grid,boundary)
    matrix = operator.assemble(torch.full(grid.shape,1500.**-2,dtype=torch.float64),2*np.pi*frequency)
    nx = operator.padded_shape[1]-2
    row = np.arange(p*nx,(p+1)*nx)  # central physical z row, sigma_z=0
    reduced = matrix[row,:][:,row] + diags(
        matrix[row,row-nx].A.ravel()*np.exp(-1j*kz*h)
        + matrix[row,row+nx].A.ravel()*np.exp(1j*kz*h))
    source = p+round(extent/(2*h))-1
    rhs = np.zeros(nx,complex)
    rhs[source] = 1/h
    field = spsolve(reduced.tocsc(),rhs)
    positions = np.arange(60.,151.,10.)
    sampled = field[source+np.rint(positions/h).astype(int)]
    # Independently derived outgoing discrete Green function for this strip.
    incident = h*np.exp(1j*kx*positions)/(2j*np.sin(kx*h))
    return sampled, incident, kx, positions


@pytest.mark.parametrize('frequency',[2.5,5.,10.,15.])
@pytest.mark.parametrize('angle',[0.,30.,60.])
def test_plane_incidence_thickness_sweep_independent_of_grid_refinement(frequency,angle):
    errors = []
    for h in (10.,5.):
        reference, incident, kx, positions = plane_field(frequency,angle,h,1600.,extent=1200.)
        reference_error = np.max(np.abs(reference-incident)/np.abs(incident))
        assert reference_error < 1e-5, reference_error
        row = []
        for thickness in (800.,1000.):
            actual, _, _, _ = plane_field(frequency,angle,h,thickness)
            # Measure boundary contamination and independently fit the incoming
            # plane-wave coefficient; do not rely on cancellation at a probe.
            error = np.max(np.abs(actual-reference)/np.abs(incident))
            basis = np.column_stack((np.exp(1j*kx*positions), np.exp(-1j*kx*positions)))
            coefficients = np.linalg.lstsq(basis,actual-reference,rcond=None)[0]
            incoming = abs(coefficients[1])/abs(incident[0])
            assert incoming <= 1e-3, (frequency,angle,h,thickness,incoming)
            assert error <= 1e-3, (frequency,angle,h,thickness,error)
            row.append(error)
        assert row[1] < row[0], row
        errors.append(row)
    print('plane reflection',frequency,angle,errors)


@pytest.mark.parametrize('frequency',[2.5,5.,10.,15.])
def test_full_2d_corners_against_expanded_domain(frequency):
    # Receivers approach each of the four corners; PML has both stretches.
    receivers = np.array([(sz*r,sx*r) for sz in (-1,1) for sx in (-1,1) for r in (120.,180.,240.)])
    def solve(extent,thickness):
        grid=Grid2D((round(extent/10)+1,)*2,(10.,10.),(-extent/2,)*2)
        boundary=BoundaryConfig(pml_cells=round(thickness/10),sigma_max=60.)
        operator=HelmholtzOperator(grid,boundary)
        rhs=SamplingOperator(grid,[(0.,0.)],boundary=boundary).source_rhs(torch.ones(1,dtype=torch.complex128))
        field, report=DirectSolver(SolverConfig(memory_budget_bytes=1)).solve(
            operator,torch.full(grid.shape,1500.**-2,dtype=torch.float64),2*np.pi*frequency,rhs)
        assert report.relative_residual <= 1e-10
        return SamplingOperator(grid,receivers,boundary=boundary).sample(field).numpy()
    reference=solve(1000.,800.)
    errors=[]
    for thickness in (400.,600.):
        actual=solve(600.,thickness)
        error=np.max(np.abs(actual-reference)/np.abs(reference))
        assert error <= 1e-3, (frequency,thickness,error)
        errors.append(error)
    assert errors[1] < errors[0]
    print('2D corner reflection',frequency,errors)
