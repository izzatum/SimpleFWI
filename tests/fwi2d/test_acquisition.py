import math
import pytest
import torch
from simplefwi.fwi2d.types import Grid2D, BoundaryConfig
from simplefwi.fwi2d.acquisition import SamplingOperator
from simplefwi.fwi2d.operator import HelmholtzOperator


def test_sampling_adjoint_and_affine_interpolation():
    grid = Grid2D((4,5), (2.,3.), (-7.,11.))
    sample = SamplingOperator(grid, [(-6.,12.5),(-1.,23.),(-7.,11.)])
    z = torch.arange(4, dtype=torch.float64)*2-7
    x = torch.arange(5, dtype=torch.float64)*3+11
    field = (2*z[:,None]+3*x[None,:]).to(torch.complex128)
    torch.testing.assert_close(sample.sample(field), torch.tensor([25.5,67.,19.],dtype=torch.complex128))
    torch.manual_seed(2)
    u = torch.randn(2,4,5,dtype=torch.complex128)
    v = torch.randn(2,3,dtype=torch.complex128)
    lhs = torch.vdot(sample.sample(u).flatten(),v.flatten())
    rhs = torch.vdot(u.flatten(),sample.scatter(v).flatten())
    assert abs(lhs-rhs)/max(abs(lhs),abs(rhs)) < 1e-10


def test_compact_sampling_and_area_source():
    grid = Grid2D((4,5), (2.,3.), (-7.,11.))
    boundary = BoundaryConfig(pml_cells=2)
    operator = HelmholtzOperator(grid,boundary)
    sample = SamplingOperator(grid,[(-6.,12.5),(-1.,23.)],boundary=boundary)
    u = torch.randn(operator.n_dof,dtype=torch.complex128)
    v = torch.randn(2,dtype=torch.complex128)
    torch.testing.assert_close(torch.vdot(sample.sample(u),v),torch.vdot(u,sample.scatter(v)))
    torch.testing.assert_close(sample.source_rhs(v),sample.scatter(v)/6)
    torch.testing.assert_close(sample.sample(u),SamplingOperator(grid,sample.positions_m).sample(operator.physical(u)))


@pytest.mark.parametrize('point',[(-7.00000001,11.),(-1.,23.00000001),(float('nan'),11.)])
def test_reject_outside_without_clipping(point):
    with pytest.raises(ValueError):
        SamplingOperator(Grid2D((4,5),(2.,3.),(-7.,11.)),[point])


def test_free_surface_rejects_pressure_release_node():
    with pytest.raises(ValueError,match='Dirichlet|free.surface|pressure'):
        SamplingOperator(Grid2D((4,5),(2.,3.),(-7.,11.)),[(-7.,14.)],boundary=BoundaryConfig(top='free_surface',pml_cells=2))


def test_partial_free_surface_interpolation_keeps_zero_constraint():
    grid = Grid2D((4,5),(2.,3.),(-7.,11.))
    boundary = BoundaryConfig(top='free_surface',pml_cells=2)
    operator = HelmholtzOperator(grid,boundary)
    sample = SamplingOperator(grid,[(-6.,14.)],boundary=boundary)
    u = torch.ones(operator.n_dof,dtype=torch.complex128)
    torch.testing.assert_close(sample.sample(u),torch.tensor([.5],dtype=torch.complex128))
    v = torch.tensor([2+3j],dtype=torch.complex128)
    torch.testing.assert_close(torch.vdot(sample.sample(u),v),torch.vdot(u,sample.scatter(v)))


def test_sampling_rejects_integer_values():
    sample = SamplingOperator(Grid2D((3,3),(1.,1.)),[(.5,.5)])
    with pytest.raises(TypeError):
        sample.sample(torch.ones(3,3,dtype=torch.int64))
    with pytest.raises(TypeError):
        sample.scatter(torch.ones(1,dtype=torch.int64))


@pytest.mark.parametrize('point',[(.4,.2),(.2,.4),(.1,.2),(.2,.1)])
def test_decimal_grid_rejects_exact_dirichlet_endpoints(point):
    grid = Grid2D((4,4),(.1,.1),(.1,.1))
    with pytest.raises(ValueError,match='Dirichlet'):
        SamplingOperator(grid,[point],boundary=BoundaryConfig(pml_cells=0,sigma_max=0.))


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('boundary',[None,BoundaryConfig(pml_cells=0,sigma_max=0.)])
def test_decimal_grid_does_not_clip_or_reject_near_endpoint(axis,boundary):
    grid = Grid2D((4,4),(.1,.1),(.1,.1))
    outside = [.2,.2]
    outside[axis] = math.nextafter(.4,math.inf)
    with pytest.raises(ValueError,match='outside'):
        SamplingOperator(grid,[outside],boundary=boundary)
    inside = [.2,.2]
    inside[axis] = math.nextafter(.4,-math.inf)
    sampler = SamplingOperator(grid,[inside],boundary=boundary)
    assert bool(torch.all(sampler._weights >= 0))
    field = torch.ones(sampler.field_shape,dtype=torch.complex128)
    assert sampler.sample(field).real.item() > 0


@pytest.mark.parametrize('point,index',[((.4,.2),(3,1)),((.2,.4),(1,3)),((.4,.4),(3,3))])
def test_decimal_grid_diagnostic_sampling_exact_endpoint(point,index):
    grid = Grid2D((4,4),(.1,.1),(.1,.1))
    sampler = SamplingOperator(grid,[point])
    field = torch.arange(16,dtype=torch.float64).reshape(grid.shape)
    torch.testing.assert_close(sampler.sample(field),field[index].reshape(1),rtol=0,atol=0)
    scattered = sampler.scatter(torch.ones(1,dtype=torch.float64))
    expected = torch.zeros(grid.shape,dtype=torch.float64)
    expected[index] = 1
    torch.testing.assert_close(scattered,expected,rtol=0,atol=0)
