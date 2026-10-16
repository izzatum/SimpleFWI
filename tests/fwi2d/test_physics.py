"""Continuum references independently derived from Delta + omega**2/v**2."""
import numpy as np
import pytest
from scipy.special import hankel1
import torch

from simplefwi.fwi2d.acquisition import SamplingOperator
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers import DirectSolver
from simplefwi.fwi2d.types import BoundaryConfig, Grid2D, SolverConfig


def solve_points(grid, boundary, frequency, source, receivers):
    operator = HelmholtzOperator(grid, boundary)
    model = torch.full(grid.shape, 1500.**-2, dtype=torch.float64)
    rhs = SamplingOperator(grid, np.asarray([source]), boundary=boundary).source_rhs(
        torch.ones(1, dtype=torch.complex128))
    wavefield, report = DirectSolver(SolverConfig(memory_budget_bytes=1)).solve(
        operator, model, 2*np.pi*frequency, rhs)
    assert report.relative_residual <= 1e-10
    return SamplingOperator(grid, receivers, boundary=boundary).sample(wavefield).numpy()


def green(frequency, distances):
    # exp(-i omega t), outgoing Hankel H0^(1), positive source q=+delta:
    # integral grad G.n dl = +1, hence G = -i H0^(1)(kr)/4.
    return -.25j*hankel1(0, 2*np.pi*frequency/1500*np.asarray(distances))


def test_smooth_manufactured_solution_second_order_rectangular_nonzero_origin():
    errors = []
    for h in (20., 10., 5.):
        grid = Grid2D((round(400/h)+1, round(600/(1.5*h))+1), (h, 1.5*h), (-70., 120.))
        boundary = BoundaryConfig(pml_cells=0)
        operator = HelmholtzOperator(grid, boundary)
        z = torch.arange(grid.shape[0], dtype=torch.float64)*h
        x = torch.arange(grid.shape[1], dtype=torch.float64)*(1.5*h)
        exact = torch.sin(torch.pi*z[:,None]/400)*torch.sin(2*torch.pi*x[None,:]/600)
        model = (1500.+100*torch.cos(torch.pi*z[:,None]/400)*torch.cos(torch.pi*x[None,:]/600))**-2
        omega = 2*np.pi*.6
        # Analytic Laplacian, never operator.apply(exact).
        rhs = (-(np.pi/400)**2-(2*np.pi/600)**2+omega**2*model)*exact
        computed, report = DirectSolver(SolverConfig()).solve(operator, model, omega,
                                                              operator.pack(rhs.to(torch.complex128)))
        errors.append(float(torch.linalg.vector_norm(computed-operator.pack(exact)) /
                            torch.linalg.vector_norm(operator.pack(exact))))
        assert report.relative_residual < 1e-10
    slopes = np.log2(np.asarray(errors[:-1])/errors[1:])
    assert np.all((slopes > 1.8) & (slopes < 2.2)), (errors, slopes)
    print('manufactured relative errors / slopes:', errors, slopes)


def test_homogeneous_green_absolute_source_amplitude_sign_and_refinement():
    # Fixed physical source/exclusion/receiver coordinates under refinement.
    receivers = np.array([[0.,80.],[0.,120.],[80.,60.],[-80.,60.],[120.,0.]])
    exact = green(5., np.linalg.norm(receivers, axis=1))
    errors = []
    for h in (10.,5.):
        grid = Grid2D((round(600/h)+1,)*2,(h,h),(-300.,-300.))
        computed = solve_points(grid, BoundaryConfig(pml_cells=round(600/h),sigma_max=60.),
                                5., (0.,0.), receivers)
        errors.append(np.linalg.norm(computed-exact)/np.linalg.norm(exact))
        # Absolute complex amplitudes: no fitted multiplier or phase alignment.
        assert np.linalg.norm(computed-exact) < .01*np.linalg.norm(exact)
    assert 3.5 < errors[0]/errors[1] < 4.5, errors
    print('Green relative errors:', errors)


def test_pressure_release_image_has_negative_pressure_reflection():
    h = 5.
    grid = Grid2D((121,121),(h,h),(0.,-300.))
    source = np.array([80.,0.])
    receivers = np.array([[60.,60.],[80.,100.],[120.,120.],[160.,0.]])
    image = np.array([-80.,0.])
    direct = green(5., np.linalg.norm(receivers-source,axis=1))
    reflected = green(5., np.linalg.norm(receivers-image,axis=1))
    actual = solve_points(grid,BoundaryConfig(top='free_surface',pml_cells=120,sigma_max=60.),
                          5.,source,receivers)
    error = np.linalg.norm(actual-(direct-reflected))/np.linalg.norm(direct-reflected)
    assert error < .01
    # A sign reversal to a rigid (positive-image) surface is strongly rejected.
    assert np.linalg.norm(actual-(direct+reflected)) > .5*np.linalg.norm(reflected)
    coefficient = np.vdot(reflected,actual-direct)/np.vdot(reflected,reflected)
    assert abs(coefficient+1) < .02
    print('pressure image relative error / fitted reflection:',error,coefficient)
