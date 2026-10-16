import pytest
import torch

from simplefwi.fwi2d.autograd import implicit_solve
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers.direct import DirectSolver
from simplefwi.fwi2d.types import BoundaryConfig, Grid2D


def fixture(top='pml'):
    torch.manual_seed(37)
    operator = HelmholtzOperator(Grid2D((2, 3), (0.4, 0.6)),
        BoundaryConfig(top=top, pml_cells=2, sigma_max=5.))
    m = (0.6 + 0.1*torch.rand((2, 3), dtype=torch.float64)).requires_grad_()
    rhs = torch.randn((2, operator.n_dof), dtype=torch.complex128, requires_grad=True)
    return operator, DirectSolver(), m, rhs


def dense(operator, m, rhs, omega):
    eye = torch.eye(operator.n_dof, dtype=rhs.dtype)
    matrix = operator.apply(m, omega, eye).T
    return torch.linalg.solve(matrix, rhs.T).T


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
def test_gradcheck_gradgradcheck_complex_rhs_and_extended_model(top):
    operator, solver, m, rhs = fixture(top)
    solve = lambda m, q: implicit_solve(m, q, 2.3, operator, solver)
    assert torch.autograd.gradcheck(solve, (m, rhs), fast_mode=True,
                                    atol=2e-7, rtol=2e-5)
    assert torch.autograd.gradgradcheck(solve, (m, rhs), fast_mode=True,
                                        atol=3e-7, rtol=3e-5)


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
def test_dense_value_gradient_and_hessian_agreement(top):
    operator, solver, m, rhs = fixture(top)
    actual = implicit_solve(m, rhs, 2.3, operator, solver)
    expected = dense(operator, m, rhs, 2.3)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    # A nonzero residual and a complex data cotangent exercise both real parts.
    target = torch.randn_like(actual)
    losses = [0.5*(u-target).abs().square().sum() for u in (actual, expected)]
    grads = [torch.autograd.grad(l, (m, rhs), create_graph=True) for l in losses]
    for g, ref in zip(*grads):
        torch.testing.assert_close(g, ref, atol=2e-11, rtol=2e-11)
    dm, dq = torch.randn_like(m), torch.randn_like(rhs)
    hvps = [torch.autograd.grad((g[0]*dm).sum()+(g[1].conj()*dq).real.sum(),
                               (m, rhs)) for g in grads]
    for h, ref in zip(*hvps):
        torch.testing.assert_close(h, ref, atol=1e-10, rtol=1e-10)


def test_nonlinear_parameterization_taylor_and_hvp():
    operator, solver, m, rhs = fixture()
    theta = m.detach().log().requires_grad_()
    target = torch.randn_like(rhs)
    def loss(t):
        u = implicit_solve(t.exp(), rhs.detach(), 2.3, operator, solver)
        return 0.5*(u-target).abs().square().sum()
    direction = torch.randn_like(theta)
    direction /= direction.norm()
    value = loss(theta)
    grad, = torch.autograd.grad(value, theta, create_graph=True)
    deriv = (grad*direction).sum()
    errors = [(loss(theta+e*direction)-value-e*deriv).abs().item()
              for e in (0.02, 0.01, 0.005, 0.0025)]
    slopes = torch.log2(torch.tensor(errors[:-1])/torch.tensor(errors[1:]))
    assert torch.all((slopes > 1.8) & (slopes < 2.2)), slopes
    hvp, = torch.autograd.grad(deriv, theta)
    eps = 1e-5
    plus = (theta.detach()+eps*direction).requires_grad_()
    minus = (theta.detach()-eps*direction).requires_grad_()
    fd = (torch.autograd.grad(loss(plus), plus)[0]-
          torch.autograd.grad(loss(minus), minus)[0])/(2*eps)
    assert (hvp-fd).norm()/hvp.norm() < 1e-5


def test_si_scaled_model_adjoint_and_model_dependent_rhs():
    torch.manual_seed(15)
    operator = HelmholtzOperator(Grid2D((2, 2), (10., 12.)),
        BoundaryConfig(pml_cells=2, sigma_max=90.))
    solver = DirectSolver()
    theta = torch.ones((2, 2), dtype=torch.float64, requires_grad=True)
    q = torch.randn(operator.n_dof, dtype=torch.complex128)
    # Gradcheck perturbs a dimensionless variable; physical m stays ~2.5e-7.
    def solve(t):
        return implicit_solve(2.5e-7*t.exp(), q*t.square().mean(), 50.,
                               operator, solver, adjoint=True)
    assert torch.autograd.gradcheck(solve, (theta,), fast_mode=True,
                                    atol=2e-6, rtol=2e-5)
    assert torch.autograd.gradgradcheck(solve, (theta,), fast_mode=True,
                                        atol=2e-6, rtol=2e-5)
