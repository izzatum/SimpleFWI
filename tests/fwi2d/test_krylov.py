"""FGMRES accuracy, flexible right preconditioning and failure contracts."""
import importlib
import math
import pytest
import torch
from simplefwi.fwi2d import Grid2D, BoundaryConfig, SolverConfig
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers.direct import DirectSolver
from simplefwi.fwi2d.solvers.base import SolverFailure


def implementation():
    spec = importlib.util.find_spec('simplefwi.fwi2d.solvers.krylov')
    assert spec is not None, 'FGMRES implementation is required'
    return importlib.import_module(spec.name).FGMRESSolver


def fixture(top='pml', shape=(4, 5)):
    op = HelmholtzOperator(Grid2D(shape, (7., 11.), (13., -27.)),
                           BoundaryConfig(top=top, pml_cells=2, sigma_max=80.))
    m = torch.full(shape, 1/1500**2, dtype=torch.float64)
    generator = torch.Generator().manual_seed(291)
    q = torch.randn((2, op.n_dof), generator=generator, dtype=torch.complex128)
    return op, m, 2*math.pi*10, q


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
@pytest.mark.parametrize('adjoint', [False, True])
def test_restarted_tensor_solver_matches_direct(top, adjoint):
    solver_class = implementation()
    op, m, omega, q = fixture(top)
    solver = solver_class(SolverConfig(method='fgmres', restart=12, maxiter=300,
                                      rtol=1e-11, atol=1e-13))
    actual, report = solver.solve(op, m, omega, q, adjoint=adjoint)
    expected, _ = DirectSolver().solve(op, m, omega, q, adjoint=adjoint)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=1e-8)
    assert report.converged and report.relative_residual <= 1e-11
    assert report.iterations > 12
    assert actual.grad_fn is None


def test_flexible_preconditioner_changes_every_iteration():
    solver_class = implementation()
    op, m, omega, q = fixture()
    class Varying:
        calls = 0
        def apply(self, model, frequency, rhs, *, adjoint=False):
            self.calls += 1
            return rhs * (1 + .1*(self.calls % 3))
    preconditioner = Varying()
    solver = solver_class(SolverConfig(method='fgmres', restart=op.n_dof,
                                      maxiter=op.n_dof, rtol=1e-11), preconditioner)
    actual, report = solver.solve(op, m, omega, q[0])
    assert report.converged and preconditioner.calls > 3
    torch.testing.assert_close(op.apply(m, omega, actual), q[0], rtol=1e-10, atol=1e-10)


def test_nonconvergence_and_memory_budget_raise_structured_failure():
    solver_class = implementation()
    op, m, omega, q = fixture()
    for config, reason in [(SolverConfig(method='fgmres', restart=1, maxiter=1), 'maxiter'),
                           (SolverConfig(method='fgmres', memory_budget_bytes=1), 'memory_budget')]:
        with pytest.raises(SolverFailure) as failure:
            solver_class(config).solve(op, m, omega, q)
        assert not failure.value.report.converged
        assert reason in failure.value.report.convergence_reason


@pytest.mark.parametrize('scale', [0., 1e-200, 1e-15])
def test_zero_tiny_and_batched_rhs(scale):
    op, m, omega, q = fixture()
    q = q.reshape(1, 2, -1)*scale
    result, report = implementation()(SolverConfig(method='fgmres')).solve(op, m, omega, q)
    assert result.shape == q.shape and torch.isfinite(result).all()
    assert report.converged and report.absolute_residual <= 1e-12
    assert math.isfinite(report.relative_residual)


def test_precision_and_device_mismatch_rejected():
    op, m, omega, q = fixture()
    solver = implementation()(SolverConfig(method='fgmres'))
    with pytest.raises(ValueError, match='precision|dtype'):
        solver.solve(op, m, omega, q.to(torch.complex64))
    with pytest.raises(ValueError, match='device'):
        solver.solve(op, m, omega, torch.empty(q.shape, device='meta', dtype=q.dtype))


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
def test_implicit_first_and_second_derivatives_use_iterative_solves(top):
    from simplefwi.fwi2d.autograd import implicit_solve
    from simplefwi.fwi2d.solvers.multigrid import MultigridConfig, ShiftedHelmholtzMultigrid
    torch.manual_seed(37)
    op = HelmholtzOperator(Grid2D((2, 3), (.4, .6)),
                           BoundaryConfig(top=top, pml_cells=2, sigma_max=5.))
    m = (.6+.1*torch.rand((2, 3), dtype=torch.float64)).requires_grad_()
    q = torch.randn((2, op.n_dof), dtype=torch.complex128, requires_grad=True)
    mg = ShiftedHelmholtzMultigrid(op, MultigridConfig(shift=.5, max_levels=5, coarse_max_dof=4))
    solver = implementation()(SolverConfig(method='fgmres', rtol=2e-13, atol=1e-14, restart=20, maxiter=100), mg)
    fn = lambda model, rhs: implicit_solve(model, rhs, 2.3, op, solver)
    assert torch.autograd.gradcheck(fn, (m, q), fast_mode=True, atol=2e-7, rtol=2e-5)
    assert torch.autograd.gradgradcheck(fn, (m, q), fast_mode=True, atol=3e-7, rtol=3e-5)
    actual = fn(m, q)
    expected = implicit_solve(m, q, 2.3, op, DirectSolver(SolverConfig(rtol=2e-13, atol=1e-14)))
    target = torch.randn_like(q)
    gradients = [torch.autograd.grad(.5*(u-target).abs().square().sum(), m, create_graph=True)[0]
                 for u in (actual, expected)]
    torch.testing.assert_close(*gradients, rtol=1e-9, atol=1e-10)
    direction = torch.randn_like(m)
    hvps = [torch.autograd.grad((g*direction).sum(), m)[0] for g in gradients]
    torch.testing.assert_close(*hvps, rtol=1e-8, atol=1e-9)


def test_streamed_curvature_actions_through_multigrid_fgmres():
    from simplefwi.fwi2d.curvature import Curvature
    from simplefwi.fwi2d.solvers.multigrid import MultigridConfig, ShiftedHelmholtzMultigrid
    from tests.fwi2d.test_objective import fixture as objective_fixture
    objective, m = objective_fixture()
    vector = torch.linspace(-1., 1., m.numel(), dtype=m.dtype)
    vector /= vector.norm()
    reference = Curvature(objective, m)
    expected_jvp = reference.jvp(vector)
    expected_gn = reference.data_gn_hvp(vector)
    expected_hvp = reference.exact_hvp(vector)
    mg = ShiftedHelmholtzMultigrid(objective.operator,
        MultigridConfig(shift=.5, max_levels=5, coarse_max_dof=4))
    objective.solver = implementation()(SolverConfig(method='fgmres', rtol=2e-13,
        atol=1e-14, restart=20, maxiter=100), mg)
    actual = Curvature(objective, m)
    torch.testing.assert_close(actual.jvp(vector), expected_jvp, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(actual.data_gn_hvp(vector), expected_gn, rtol=1e-8, atol=1e-11)
    torch.testing.assert_close(actual.exact_hvp(vector), expected_hvp, rtol=1e-8, atol=1e-10)
    other = torch.cos(vector)
    hv = actual.exact_hvp(vector)
    torch.testing.assert_close(other@hv, vector@actual.exact_hvp(other), rtol=1e-8, atol=1e-11)
    def gradient(model):
        variable = model.detach().requires_grad_()
        return torch.autograd.grad(objective.loss(variable), variable)[0].flatten()
    epsilon = 1e-5
    dm = vector.reshape_as(m)
    finite_difference = (gradient(m+epsilon*dm)-gradient(m-epsilon*dm))/(2*epsilon)
    assert (finite_difference-hv).norm()/hv.norm() < 1e-5


def test_unrepresentable_rhs_norm_never_reports_convergence():
    op, m, omega, q = fixture()
    q = torch.full_like(q, 1e308+0j)
    with pytest.raises(SolverFailure) as failure:
        implementation()(SolverConfig(method='fgmres')).solve(op, m, omega, q)
    assert not failure.value.report.converged
    assert failure.value.report.convergence_reason == 'nonfinite_rhs_norm'
