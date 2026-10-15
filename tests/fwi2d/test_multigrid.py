"""Geometric transfer boundaries and exact Galerkin coarse actions."""
import importlib
import math
import pytest
import torch
from simplefwi.fwi2d import Grid2D, BoundaryConfig, SolverConfig
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers.direct import DirectSolver


def implementation():
    spec = importlib.util.find_spec('simplefwi.fwi2d.solvers.multigrid')
    assert spec is not None, 'shifted geometric multigrid is required'
    module = importlib.import_module(spec.name)
    return module.MultigridConfig, module.ShiftedHelmholtzMultigrid


def fixture(top, shape=(7, 10)):
    op = HelmholtzOperator(Grid2D(shape, (8., 13.), (9., -11.)),
                           BoundaryConfig(top=top, pml_cells=2, sigma_max=60.))
    m = torch.full(shape, 1/1500**2, dtype=torch.float64)
    m[2:5, 3:6] = 1/1900**2
    return op, m, 2*math.pi*10


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
def test_coarse_actions_are_galerkin_and_transfer_preserves_boundary(top):
    Config, MG = implementation()
    op, m, omega = fixture(top)
    mg = MG(op, Config(shift=.5, max_levels=5, coarse_max_dof=16))
    mg.prepare(m, omega)
    gen = torch.Generator().manual_seed(81)
    q = torch.randn((2, op.n_dof), generator=gen, dtype=torch.complex128)
    expected = op.apply(m, omega, q) + .5j*op.mass_diagonal(m, omega)*q
    torch.testing.assert_close(mg.levels[0].apply(q), expected, rtol=1e-13, atol=1e-13)
    assert len(mg.levels) > 1
    for index, transfer in enumerate(mg.transfers):
        coarse = torch.randn((2, mg.levels[index+1].n_dof), generator=gen, dtype=q.dtype)
        fine = torch.randn((2, mg.levels[index].n_dof), generator=gen, dtype=q.dtype)
        prolonged = transfer.prolong(coarse)
        full = transfer.prolong_padded(coarse)
        assert torch.count_nonzero(full[..., 0, :]) == 0
        assert torch.count_nonzero(full[..., -1, :]) == 0
        assert torch.count_nonzero(full[..., :, 0]) == 0
        assert torch.count_nonzero(full[..., :, -1]) == 0
        torch.testing.assert_close(torch.sum(prolonged.conj()*fine),
                                   torch.sum(coarse.conj()*transfer.restrict(fine)), rtol=1e-13, atol=1e-13)
        for adjoint in (False, True):
            expected = transfer.restrict(mg.levels[index].apply(prolonged, adjoint=adjoint))
            actual = mg.levels[index+1].apply(coarse, adjoint=adjoint)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
@pytest.mark.parametrize('adjoint', [False, True])
def test_multigrid_preconditioned_solve_matches_direct(top, adjoint):
    Config, MG = implementation()
    from simplefwi.fwi2d.solvers.krylov import FGMRESSolver
    op, m, omega = fixture(top)
    mg = MG(op, Config(shift=.5, max_levels=5, coarse_max_dof=16))
    q = torch.randn((2, op.n_dof), generator=torch.Generator().manual_seed(82), dtype=torch.complex128)
    result, report = FGMRESSolver(SolverConfig(method='fgmres', rtol=1e-11, maxiter=100, restart=15), mg).solve(op, m, omega, q, adjoint=adjoint)
    expected, _ = DirectSolver().solve(op, m, omega, q, adjoint=adjoint)
    torch.testing.assert_close(result, expected, rtol=1e-9, atol=1e-8)
    assert report.converged and report.relative_residual <= 1e-11


def test_numeric_cache_tracks_in_place_model_changes_and_releases():
    Config, MG = implementation()
    op, m, omega = fixture('pml')
    mg = MG(op, Config(shift=.5, max_levels=5, coarse_max_dof=16))
    q = torch.ones(op.n_dof, dtype=torch.complex128)
    first = mg.apply(m, omega, q)
    assert mg.cache_bytes > 0
    m.mul_(1.3)
    second = mg.apply(m, omega, q)
    assert not torch.allclose(first, second)
    fresh = MG(op, mg.config).apply(m, omega, q)
    torch.testing.assert_close(second, fresh)
    mg.clear_cache()
    assert mg.cache_bytes == 0 and not mg.levels


def test_coarse_limit_and_memory_limits_are_enforced_before_dense_allocation():
    Config, MG = implementation()
    from simplefwi.fwi2d.solvers.base import SolverFailure
    op, m, omega = fixture('pml')
    for config in [Config(shift=.5, max_levels=1, coarse_max_dof=4),
                   Config(shift=.5, max_levels=5, coarse_max_dof=16, memory_budget_bytes=1)]:
        with pytest.raises(SolverFailure):
            MG(op, config).prepare(m, omega)


@pytest.mark.parametrize('shape', [(3, 3), (3, 8), (8, 3), (8, 9), (14, 17)])
def test_nested_galerkin_action_handles_odd_even_and_thin_domains(shape):
    Config, MG = implementation()
    op = HelmholtzOperator(Grid2D(shape, (4., 7.)), BoundaryConfig(pml_cells=0))
    m = torch.full(shape, 1/1500**2, dtype=torch.float64)
    mg = MG(op, Config(shift=.5, max_levels=8, coarse_max_dof=1))
    mg.prepare(m, 50.)
    for index, transfer in enumerate(mg.transfers):
        coarse = torch.randn(mg.levels[index+1].n_dof, dtype=torch.complex128)
        torch.testing.assert_close(mg.levels[index+1].apply(coarse),
            transfer.restrict(mg.levels[index].apply(transfer.prolong(coarse))), rtol=1e-12, atol=1e-13)


def test_vcycle_adjoint_is_complex_conjugate_transpose():
    Config, MG = implementation()
    op, m, omega = fixture('free_surface')
    mg = MG(op, Config(shift=.5, max_levels=5, coarse_max_dof=16, pre_smooth=1, post_smooth=3))
    gen = torch.Generator().manual_seed(13)
    x, y = torch.randn((2, op.n_dof), generator=gen, dtype=torch.complex128)
    left = torch.vdot(mg.apply(m, omega, x), y)
    right = torch.vdot(x, mg.apply(m, omega, y, adjoint=True))
    torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
