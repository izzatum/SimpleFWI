"""Fixed-stage screening and memory refusal contracts."""
from dataclasses import FrozenInstanceError, replace

import pytest

from simplefwi.fwi2d.types import BoundaryConfig, Grid2D
from simplefwi.fwi2d.planning import PlanningError, StageRequest, plan_stage


def request(**kwargs):
    values = dict(grids=(Grid2D((48, 48), (10., 10.)), Grid2D((95, 95), (5., 5.))),
                  velocity_bounds=(1500., 2500.), frequencies_hz=(5., 15.),
                  source_count=8, receiver_count=32, source_batch_sizes=(1, 2, 4),
                  boundary=BoundaryConfig(pml_cells=10))
    values.update(kwargs)
    return StageRequest(**values)


def test_selects_accurate_grid_and_preserves_stage_request():
    plan = plan_stage(request(), memory_budget_bytes=10**9)
    assert plan.grid.spacing_m == (5., 5.)
    assert plan.frequencies_hz == (5., 15.)
    assert plan.source_batch_size == 4
    assert plan.precision == 'complex128'
    assert plan.dispersion.passed
    assert plan.validation_status == 'screened_only'
    assert plan.pml_validation == 'required'
    assert plan.linear_solve_validation == 'required'


def test_memory_estimate_counts_pml_krylov_and_history():
    plan = plan_stage(request(), memory_budget_bytes=10**9)
    mem = plan.memory
    assert mem.physical_cells == 95 * 95
    assert mem.padded_cells == 115 * 115
    assert mem.pml_cells == 115 * 115 - 95 * 95
    assert mem.krylov_bytes >= (2 * 30 + 1) * 4 * 115 * 115 * 16
    assert mem.optimizer_bytes >= 2 * 10 * 95 * 95 * 8
    assert mem.total_bytes >= sum((mem.model_bytes, mem.operator_bytes,
        mem.wavefield_bytes, mem.krylov_bytes, mem.optimizer_bytes,
        mem.data_bytes, mem.extra_workspace_bytes))


def test_budget_can_reduce_batch_but_not_accuracy():
    one = plan_stage(request(source_batch_sizes=(1,)), memory_budget_bytes=10**9)
    plan = plan_stage(request(), memory_budget_bytes=one.memory.total_bytes)
    assert plan.source_batch_size == 1
    assert plan.grid == one.grid
    assert plan.memory.total_bytes <= one.memory.total_bytes


def test_no_feasible_configuration_fails_explicitly():
    with pytest.raises(PlanningError, match='memory'):
        plan_stage(request(), memory_budget_bytes=100)
    with pytest.raises(PlanningError, match='dispersion'):
        plan_stage(request(grids=(Grid2D((48, 48), (10., 10.)),)), memory_budget_bytes=10**9)


def test_plan_and_inputs_are_deeply_immutable_during_parameter_updates():
    candidates = [Grid2D((95, 95), (5., 5.))]
    frequencies = [5., 15.]
    req = request(grids=candidates, frequencies_hz=frequencies)
    plan = plan_stage(req, memory_budget_bytes=10**9)
    candidates.clear()
    frequencies.append(30.)
    assert plan.frequencies_hz == (5., 15.)
    assert len(req.grids) == 1
    with pytest.raises(FrozenInstanceError):
        plan.source_batch_size = 1
    with pytest.raises(FrozenInstanceError):
        plan.grid.spacing_m = (10., 10.)


def test_free_surface_has_no_top_pml_memory():
    req = request(boundary=BoundaryConfig(top='free_surface', pml_cells=10))
    plan = plan_stage(req, memory_budget_bytes=10**9)
    assert plan.memory.padded_cells == 105 * 115


@pytest.mark.parametrize('kwargs', [dict(frequencies_hz=(0.,)), dict(source_count=0),
    dict(source_batch_sizes=(0,)), dict(krylov_restart=0), dict(precision='complex64'),
    dict(velocity_bounds=(0., 2500.)), dict(max_phase_error=0.), dict(optimizer_history_size=-1)])
def test_bad_request_is_rejected(kwargs):
    with pytest.raises(ValueError):
        plan_stage(request(**kwargs), memory_budget_bytes=10**9)


def test_different_physical_extent_is_not_silently_regridded():
    with pytest.raises(ValueError, match='extent'):
        request(grids=(Grid2D((48, 48), (10., 10.)), Grid2D((48, 48), (5., 5.))))


def test_optimizer_steps_and_repeated_closures_preserve_fixed_plan():
    import torch
    from dataclasses import asdict
    plan = plan_stage(request(), memory_budget_bytes=10**9)
    snapshot = asdict(plan)
    parameter = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameter], max_iter=3)
    closures = []

    def closure():
        optimizer.zero_grad()
        loss = (parameter - .5)**2
        loss.backward()
        closures.append(asdict(plan))
        return loss

    optimizer.step(closure)
    assert parameter.item() == pytest.approx(.5)
    assert len(closures) >= 2
    assert all(value == snapshot for value in closures)
    assert asdict(plan) == snapshot
