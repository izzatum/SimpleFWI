"""Bounded Krylov storage and cache lifetime, with CPU tensor allocations."""
import pytest
import torch
from simplefwi.fwi2d import Grid2D, BoundaryConfig, SolverConfig
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers.base import SolverFailure
from simplefwi.fwi2d.solvers.krylov import FGMRESSolver
from simplefwi.fwi2d.solvers.multigrid import MultigridConfig, ShiftedHelmholtzMultigrid


def fixture():
    op = HelmholtzOperator(Grid2D((6, 9), (7., 11.)), BoundaryConfig(pml_cells=2))
    m = torch.full(op.grid.shape, 1/1500**2, dtype=torch.float64)
    q = torch.randn(op.n_dof, dtype=torch.complex128, generator=torch.Generator().manual_seed(91))
    return op, m, q


def test_restart_storage_does_not_grow_with_iteration_budget():
    op, m, q = fixture()
    reports = []
    for maxiter in (200, 400):
        solver = FGMRESSolver(SolverConfig(method='fgmres', restart=10, maxiter=maxiter))
        _, report = solver.solve(op, m, 30., q)
        assert report.iterations > 10
        assert solver.basis_peak_vectors <= 21
        reports.append(report)
    assert reports[0].memory_bytes == reports[1].memory_bytes


def test_frequency_and_source_streaming_retains_one_numeric_hierarchy():
    op, m, q = fixture()
    mg = ShiftedHelmholtzMultigrid(op, MultigridConfig(shift=.5, max_levels=6, coarse_max_dof=9))
    solver = FGMRESSolver(SolverConfig(method='fgmres', restart=12), mg)
    sizes = []
    with solver.evaluation():
        for omega in (20., 30., 40., 50.):
            for scale in (1., 2., 3.):
                _, report = solver.solve(op, m, omega, q*scale)
                sizes.append((solver.cache_bytes, report.memory_bytes))
        assert len(set(sizes)) == 1
        assert all(level.values.grad_fn is None for level in mg.levels)
    assert solver.cache_bytes == 0


def test_combined_budget_rejects_before_building_hierarchy():
    op, m, q = fixture()
    mg = ShiftedHelmholtzMultigrid(op, MultigridConfig(shift=.5, max_levels=6, coarse_max_dof=9))
    budget = mg.estimate_memory_bytes(m)
    solver = FGMRESSolver(SolverConfig(method='fgmres', memory_budget_bytes=budget), mg)
    with pytest.raises(SolverFailure, match='memory budget'):
        solver.solve(op, m, 30., q)
    assert mg.cache_bytes == 0


def test_cpu_allocation_profile_has_no_fine_dense_system():
    op, m, q = fixture()
    mg = ShiftedHelmholtzMultigrid(op, MultigridConfig(shift=.5, max_levels=6, coarse_max_dof=9))
    solver = FGMRESSolver(SolverConfig(method='fgmres', restart=10), mg)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                profile_memory=True, record_shapes=True, acc_events=True) as profile:
        _, report = solver.solve(op, m, 30., q)
    allocations = [event.cpu_memory_usage for event in profile.events() if event.cpu_memory_usage > 0]
    assert allocations and max(allocations) < op.n_dof**2*q.element_size()
    assert max(allocations) < report.memory_bytes


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware absent; allocation scaling unrun')
def test_cuda_streamed_working_memory_is_bounded_by_batch_not_source_total():
    op, m, q = fixture()
    m, q = m.cuda(), q.cuda()
    mg = ShiftedHelmholtzMultigrid(op, MultigridConfig(shift=.5, max_levels=6, coarse_max_dof=9))
    solver = FGMRESSolver(SolverConfig(method='fgmres', device='cuda', restart=12), mg)
    solver.solve(op, m, 30., q)  # initialize backend workspaces before measuring
    peaks = []
    for sources in (1, 12):
        torch.cuda.reset_peak_memory_stats()
        for source in range(sources):
            result, _ = solver.solve(op, m, 30., q*(1+source/12))
            del result
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    assert peaks[1] <= 1.2*peaks[0]+65536


def test_old_basis_tensors_are_released_before_restart_allocation(monkeypatch):
    import weakref
    op, m, q = fixture()
    references = []
    peaks = []
    allocate = torch.empty
    def observed(*args, **kwargs):
        value = allocate(*args, **kwargs)
        if value.shape in ((11, op.n_dof), (10, op.n_dof)):
            references.append(weakref.ref(value))
            peaks.append(sum(item().shape[0] for item in references if item() is not None))
        return value
    monkeypatch.setattr(torch, 'empty', observed)
    solver = FGMRESSolver(SolverConfig(method='fgmres', restart=10, maxiter=200))
    _, report = solver.solve(op, m, 30., q)
    assert report.iterations > 10
    assert max(peaks) <= 21
