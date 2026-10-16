import pytest
import torch
from simplefwi.fwi2d.parameters import VelocityParameters
from simplefwi.fwi2d.inversion import InversionRunner
from tests.fwi2d.test_optimizer_inventory import Quadratic


class Capture:
    def __init__(self): self.results = []
    def record_accepted(self, result, payload): self.results.append((result, payload))


def test_events_and_detached_history_and_factories():
    model = VelocityParameters([[1600., 1700.]], (1400., 2200.), None)
    record = Capture()
    runner = InversionRunner(Quadratic(), model, lambda p: torch.optim.Adam(p, lr=0.1),
                             source_batch_size=1, recorder=record)
    scheduler = torch.optim.lr_scheduler.StepLR(runner.optimizer, step_size=1, gamma=0.5)
    runner.scheduler = scheduler
    results = runner.run(3)
    assert [s.accepted_step for s in results.history] == [1, 2, 3]
    assert len(record.results) == 3
    assert scheduler.last_epoch == 3
    assert runner.optimizer.param_groups[0]['lr'] == pytest.approx(0.0125)
    assert all(isinstance(s.loss, float) for s in results.history)
    assert results.velocity.grad_fn is None


def test_plateau_receives_post_step_loss():
    model = VelocityParameters([[1600., 1700.]], (1400., 2200.), None)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    runner = InversionRunner(Quadratic(), model, opt, scheduler=scheduler)
    result = runner.step()
    assert scheduler.best == result.loss


def test_failed_post_step_solve_does_not_emit_acceptance_and_restores_state():
    class Failure(Quadratic):
        def iter_data_losses(self, m, source_batch_size):
            if not torch.is_grad_enabled():
                raise RuntimeError('solve did not converge')
            yield from super().iter_data_losses(m, source_batch_size)
    model = VelocityParameters([[1600., 1700.]], (1400., 2200.), None)
    initial = model.velocity().detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 1)
    recorder = Capture()
    runner = InversionRunner(Failure(), model, opt, scheduler=scheduler, recorder=recorder)
    with pytest.raises(RuntimeError, match='solve did not converge'):
        runner.step()
    assert runner.accepted_steps == 0 and not recorder.results
    assert scheduler.last_epoch == 0
    torch.testing.assert_close(model.velocity(), initial)
    assert not opt.state


@pytest.mark.parametrize('name,kwargs,steps', [
    ('Adam', {'lr': 0.2}, 6),
    ('LBFGS', {'lr': 1., 'max_iter': 5, 'tolerance_grad': 1e-12,
               'tolerance_change': 1e-14}, 3),
])
def test_camembert_native_optimizer_reduces_same_grid_pde_objective(name, kwargs, steps):
    # This establishes optimizer/PDE integration, not independent-data recovery.
    from examples.fwi2d.camembert import make_camembert, simulate_data
    from simplefwi.fwi2d.objective import FWIObjective
    from simplefwi.fwi2d.operator import HelmholtzOperator
    from simplefwi.fwi2d.optimizers import make_optimizer
    from simplefwi.fwi2d.solvers import DirectSolver
    from simplefwi.fwi2d.types import SolverConfig
    grid, true, initial, acquisition, boundary = make_camembert()
    observed = simulate_data(grid, true, acquisition, boundary, (5.,), source_batch=2)
    mask = torch.zeros(grid.shape, dtype=torch.bool)
    mask[16:32, 16:32] = True
    model = VelocityParameters(initial, (1400., 2200.), mask)
    objective = FWIObjective(HelmholtzOperator(grid, boundary), acquisition,
                             (5.,), observed, 1., None,
                             DirectSolver(SolverConfig(memory_budget_bytes=512*1024**2)))
    initial_loss = objective.loss(model.slowness_squared()).item()
    optimizer = make_optimizer(name, model, **kwargs)
    result = InversionRunner(objective, model, optimizer, source_batch_size=2).run(steps)
    assert result.history[-1].loss < initial_loss
    assert torch.equal(result.velocity[~mask], initial[~mask])
    assert all(s.accepted for s in result.history)
    assert result.velocity.grad_fn is None


def test_nonfinite_native_state_rolls_back_without_recording():
    class BrokenState(torch.optim.SGD):
        def step(self, closure=None):
            result = super().step(closure)
            p = self.param_groups[0]['params'][0]
            self.state[p]['bad'] = torch.tensor(float('inf'))
            return result
    model = VelocityParameters([[1600., 1700.]], (1400., 2200.), None)
    before = model.velocity().detach().clone()
    recorder = Capture()
    optimizer = BrokenState(model.parameters(), lr=0.1)
    runner = InversionRunner(Quadratic(), model, optimizer, recorder=recorder)
    with pytest.raises(FloatingPointError, match='optimizer state'):
        runner.step()
    assert not recorder.results and runner.accepted_steps == 0
    assert not optimizer.state
    torch.testing.assert_close(model.velocity(), before)


@pytest.mark.parametrize('scheduler_kind', ['step', 'plateau'])
def test_legacy_recording_failure_preserves_accepted_lifecycle(scheduler_kind):
    model = VelocityParameters([[1600., 1700.]], (1400., 2200.))
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
                 if scheduler_kind == 'step' else
                 torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer))
    before = model.velocity().detach().clone()
    observed = []

    class FailOnce:
        def record_accepted(self, result, payload):
            observed.append((result.accepted_step, runner.accepted_steps,
                             scheduler.last_epoch))
            if len(observed) == 1:
                raise OSError('archive unavailable')

    runner = InversionRunner(Quadratic(), model, optimizer, scheduler=scheduler,
                             recorder=FailOnce())
    with pytest.raises(OSError, match='archive unavailable'):
        runner.step()
    assert runner.accepted_steps == 1
    assert scheduler.last_epoch == 1
    assert observed == [(1, 1, 1)]
    assert all(int(state['step']) == 1 for state in optimizer.state.values())
    assert not torch.equal(model.velocity(), before)
    if scheduler_kind == 'step':
        assert optimizer.param_groups[0]['lr'] == pytest.approx(.05)

    result = runner.step()
    assert result.accepted_step == 2
    assert runner.accepted_steps == scheduler.last_epoch == 2
    assert observed == [(1, 1, 1), (2, 2, 2)]
    assert all(int(state['step']) == 2 for state in optimizer.state.values())
