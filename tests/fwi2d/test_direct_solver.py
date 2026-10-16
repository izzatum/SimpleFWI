"""Independent complex systems exercise the CPU adapter and its true residual."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from simplefwi.fwi2d import Grid2D, BoundaryConfig, SolverConfig
from simplefwi.fwi2d.solvers import DirectSolver, SolverFailure


class TinyOperator:
    """An intentionally nonsymmetric complex matrix with model dependence."""
    def __init__(self):
        self.grid = Grid2D((2, 2), (7., 11.), (-3., 5.))
        self.boundary = BoundaryConfig(pml_cells=1)
        self.n_dof = 4

    def dense(self, m, omega):
        a = torch.tensor([[5+1j, 1-2j, 0, .3j], [2+.5j, 6-1j, 1, 0],
                          [0, -.2j, 7+2j, 2], [1, 0, -.3, 8-1j]],
                         dtype=torch.complex128)
        return a + torch.diag(m.flatten() * omega + self.boundary.sigma_max*.001)

    def assemble(self, m, omega):
        return sp.csc_matrix(self.dense(m, omega).detach().numpy())

    def apply(self, m, omega, u, *, adjoint=False):
        a = self.dense(m, omega)
        return u @ (a.mH if adjoint else a).T


def model():
    return torch.full((2, 2), .4, dtype=torch.float64)


def rhs(batch=3):
    rng = np.random.default_rng(19)
    return torch.tensor(rng.normal(size=(batch, 4)) + 1j*rng.normal(size=(batch, 4)))


@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.parametrize('shape', [(4,), (3, 4), (2, 3, 4)])
def test_batched_forward_and_conjugate_transpose_match_dense(adjoint, shape):
    operator, m = TinyOperator(), model()
    q = rhs(int(np.prod(shape[:-1]) or 1)).reshape(shape)
    result, report = DirectSolver(SolverConfig()).solve(operator, m, 7., q, adjoint=adjoint)
    a = operator.dense(m, 7.)
    reference = torch.linalg.solve(a.mH if adjoint else a, q.reshape(-1, 4).T).T.reshape(shape)
    torch.testing.assert_close(result, reference, atol=1e-13, rtol=1e-13)
    assert report.converged and report.relative_residual <= 1e-10
    assert report.absolute_residual >= 0 and report.memory_bytes > 0


@pytest.mark.parametrize('scale', [0., 1e-200, 1e-15])
def test_zero_and_tiny_rhs_have_finite_diagnostics(scale):
    q = rhs() * scale
    result, report = DirectSolver(SolverConfig()).solve(TinyOperator(), model(), 7., q)
    assert torch.isfinite(result).all()
    assert np.isfinite(report.relative_residual)
    assert np.isfinite(report.absolute_residual)
    assert report.converged and report.absolute_residual <= 1e-12


def test_residual_failure_raises_before_returning_result():
    class InconsistentOperator(TinyOperator):
        def apply(self, m, omega, u, *, adjoint=False):
            return 2 * super().apply(m, omega, u, adjoint=adjoint)
    with pytest.raises(SolverFailure) as failure:
        DirectSolver(SolverConfig()).solve(InconsistentOperator(), model(), 7., rhs())
    assert not failure.value.report.converged
    assert failure.value.report.relative_residual > .1
    assert failure.value.report.convergence_reason == 'residual_tolerance_failed'


@pytest.mark.parametrize('omega', [0., -1., float('nan')])
def test_invalid_frequency_is_explicit(omega):
    with pytest.raises(ValueError, match='omega'):
        DirectSolver(SolverConfig()).solve(TinyOperator(), model(), omega, rhs())


def test_cpu_adapter_rejects_implicit_device_or_precision_changes():
    with pytest.raises(ValueError, match='CPU|cpu'):
        DirectSolver(SolverConfig(device='cuda'))
    with pytest.raises(ValueError, match='dtype|precision'):
        DirectSolver(SolverConfig()).solve(TinyOperator(), model(), 7., rhs().to(torch.complex64))
    with pytest.raises(ValueError, match='CPU|cpu'):
        DirectSolver(SolverConfig()).solve(TinyOperator(), model(), 7., torch.empty((2,4), device='meta', dtype=torch.complex128))


def test_singular_factorization_is_structured_failure():
    class Singular(TinyOperator):
        def assemble(self, m, omega):
            return sp.csc_matrix((4, 4), dtype=np.complex128)
    with pytest.raises(SolverFailure) as failure:
        DirectSolver(SolverConfig()).solve(Singular(), model(), 7., rhs())
    assert not failure.value.report.converged
    assert failure.value.report.convergence_reason == 'factorization_failed'


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
def test_real_helmholtz_sparse_solve_and_adjoint(top):
    from simplefwi.fwi2d.operator import HelmholtzOperator
    operator = HelmholtzOperator(Grid2D((4, 5), (7., 11.), (13., -27.)),
                                 BoundaryConfig(top=top, pml_cells=2, sigma_max=80.))
    m = torch.full(operator.grid.shape, 1/1500**2, dtype=torch.float64)
    rng = np.random.default_rng(31)
    q = torch.tensor(rng.normal(size=(3, operator.n_dof))
                     + 1j*rng.normal(size=(3, operator.n_dof)))
    solver = DirectSolver(SolverConfig())
    for adjoint in (False, True):
        u, report = solver.solve(operator, m, 2*np.pi*10, q, adjoint=adjoint)
        assert report.relative_residual <= 1e-10
        a = torch.from_numpy(operator.assemble(m, 2*np.pi*10).toarray())
        expected = torch.linalg.solve(a.mH if adjoint else a, q.T).T
        torch.testing.assert_close(u, expected, atol=1e-9, rtol=1e-11)
    assert solver.factorization_count == 1


@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.parametrize('lazy_view', ['conjugate', 'negative', 'both'])
def test_lazy_conjugate_and_negative_rhs_match_dense(adjoint, lazy_view):
    operator, m = TinyOperator(), model()
    q = rhs()
    if lazy_view in ('conjugate', 'both'):
        q = q.conj()
    if lazy_view in ('negative', 'both'):
        q = torch._neg_view(q)
    result, report = DirectSolver(SolverConfig()).solve(operator, m, 7., q, adjoint=adjoint)
    a = operator.dense(m, 7.)
    expected = torch.linalg.solve(a.mH if adjoint else a, q.T).T
    torch.testing.assert_close(result, expected, atol=1e-13, rtol=1e-13)
    assert report.converged
