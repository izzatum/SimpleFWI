import numpy as np
import pytest
import torch

from simplefwi.fwi2d.types import Grid2D, BoundaryConfig
from simplefwi.fwi2d.operator import HelmholtzOperator


@pytest.mark.parametrize("top", ["pml", "free_surface"])
def test_sparse_tensor_and_adjoint(top):
    torch.manual_seed(3)
    grid = Grid2D((5, 7), (8., 11.), (-13., 29.))
    op = HelmholtzOperator(grid, BoundaryConfig(top=top, pml_cells=3, sigma_max=90.))
    m = (1500. + 700. * torch.rand(grid.shape, dtype=torch.float64)).pow(-2)
    u = torch.randn(2, op.n_dof, dtype=torch.complex128)
    v = torch.randn_like(u)
    a = op.assemble(m, 2*np.pi*9)
    got = op.apply(m, 2*np.pi*9, u)
    expected = torch.from_numpy((a @ u.numpy().T).T)
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected) < 1e-12
    ah = op.apply(m, 2*np.pi*9, v, adjoint=True)
    lhs, rhs = torch.vdot(v.flatten(), got.flatten()), torch.vdot(ah.flatten(), u.flatten())
    assert abs(lhs-rhs)/max(abs(lhs), abs(rhs)) < 1e-10
    np.testing.assert_allclose(ah.numpy(), (a.conj().T @ v.numpy().T).T, rtol=1e-12, atol=1e-14)
    full = op.unpack(u)
    assert full.shape == (2, *op.padded_shape)
    assert torch.equal(op.pack(full), u)
    assert torch.count_nonzero(full[..., 0, :]) == 0


def test_independent_no_pml_rectangular_stencil():
    grid = Grid2D((5, 6), (2., 3.), (31., -19.))
    op = HelmholtzOperator(grid, BoundaryConfig(pml_cells=0, sigma_max=0.))
    m = torch.full(grid.shape, 1/1500**2, dtype=torch.float64)
    omega = 2*np.pi*7
    u = torch.arange(op.n_dof, dtype=torch.float64).to(torch.complex128) * (1+2j)
    full = np.zeros(grid.shape, dtype=np.complex128)
    full[1:-1, 1:-1] = u.numpy().reshape(3, 4)
    explicit = []
    for z in range(1, 4):
        for x in range(1, 5):
            explicit.append((full[z-1,x]-2*full[z,x]+full[z+1,x])/4 +
                            (full[z,x-1]-2*full[z,x]+full[z,x+1])/9 +
                            omega**2/1500**2*full[z,x])
    np.testing.assert_allclose(op.apply(m, omega, u).numpy(), explicit, rtol=1e-13, atol=1e-14)


def test_independent_pml_corner_stencil():
    grid = Grid2D((3, 4), (2., 3.), (31., -19.))
    op = HelmholtzOperator(grid, BoundaryConfig(pml_cells=3, sigma_max=75.))
    omega = 43.
    m = torch.full(grid.shape, 1/1700**2, dtype=torch.float64)
    u = torch.arange(op.n_dof, dtype=torch.float64).to(torch.complex128)*(1+.3j)
    full = op.unpack(u).numpy()
    def stretch(t, lo, hi):
        return 1+1j*75/omega*(max(lo-t, 0, t-hi)/3)**3
    expected = np.zeros(op.padded_shape, complex)
    for z in range(1, op.padded_shape[0]-1):
        for x in range(1, op.padded_shape[1]-1):
            sx, sz = stretch(x, 3, 6), stretch(z, 3, 5)
            expected[z,x] = (sz/stretch(x+.5,3,6)*(full[z,x+1]-full[z,x]) - sz/stretch(x-.5,3,6)*(full[z,x]-full[z,x-1]))/9
            expected[z,x] += (sx/stretch(z+.5,3,5)*(full[z+1,x]-full[z,x]) - sx/stretch(z-.5,3,5)*(full[z,x]-full[z-1,x]))/4
            expected[z,x] += omega**2/1700**2*sx*sz*full[z,x]
    np.testing.assert_allclose(op.apply(m,omega,u).numpy(), expected[1:-1,1:-1].ravel(), rtol=1e-12, atol=1e-13)


def test_derivative_includes_edge_extension():
    grid = Grid2D((4, 5), (8., 9.))
    op = HelmholtzOperator(grid, BoundaryConfig(pml_cells=3))
    m = torch.full(grid.shape, 1/1800**2, dtype=torch.float64, requires_grad=True)
    dm = torch.zeros_like(m); dm[0, 0] = 1e-7
    u = torch.ones(op.n_dof, dtype=torch.complex128)
    actual = torch.autograd.functional.jvp(lambda model: op.apply(model, 32., u), m, dm)[1]
    expected = op.model_derivative(dm, 32., u)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    assert torch.count_nonzero(actual) == 9


@pytest.mark.parametrize("omega", [0., -1., float('nan')])
def test_reject_bad_frequency(omega):
    grid = Grid2D((3,3), (1.,1.))
    op = HelmholtzOperator(grid, BoundaryConfig(pml_cells=1))
    with pytest.raises(ValueError):
        op.assemble(torch.ones(grid.shape, dtype=torch.float64), omega)


def test_reject_nonphysical_model_and_precision_mismatch():
    grid = Grid2D((3,3),(1.,1.))
    op = HelmholtzOperator(grid,BoundaryConfig(pml_cells=2))
    with pytest.raises(ValueError):
        op.assemble(-torch.ones(grid.shape,dtype=torch.float64),10.)
    with pytest.raises(TypeError):
        op.apply(torch.ones(grid.shape,dtype=torch.float64),10.,torch.ones(op.n_dof,dtype=torch.complex64))


def test_complex64_is_explicit_and_preserved():
    grid = Grid2D((3,3),(1.,1.))
    op = HelmholtzOperator(grid,BoundaryConfig(pml_cells=2))
    m = torch.full(grid.shape,1/1500**2,dtype=torch.float32)
    u = torch.ones(op.n_dof,dtype=torch.complex64)
    result = op.apply(m,40.,u)
    assert result.dtype == torch.complex64
    expected = torch.from_numpy(op.assemble(m,40.) @ u.numpy())
    torch.testing.assert_close(result,expected,rtol=2e-6,atol=2e-6)


def test_apply_retains_second_derivative_graph():
    grid = Grid2D((3,3),(2.,3.))
    op = HelmholtzOperator(grid,BoundaryConfig(pml_cells=2))
    m = torch.full(grid.shape,1/1500**2,dtype=torch.float64,requires_grad=True)
    u = torch.arange(op.n_dof,dtype=torch.float64).to(torch.complex128)
    # The squared residual is nonlinear in m even though A(m) is affine.
    def objective(model):
        return op.apply(model,30.,u).abs().square().sum()
    assert torch.autograd.gradcheck(objective,(m,),eps=1e-8,atol=1e-4,rtol=1e-5)
    assert torch.autograd.gradgradcheck(objective,(m,),eps=1e-8,atol=1e-4,rtol=1e-5)
