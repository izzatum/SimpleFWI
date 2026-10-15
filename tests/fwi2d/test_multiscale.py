"""Continuation algebra and lifecycle; tiny domains are not recovery benchmarks."""
from dataclasses import FrozenInstanceError, replace
import gc
import weakref

import pytest
import torch

import simplefwi.fwi2d as fwi
from simplefwi.fwi2d.optimizers import make_optimizer, OPTIMIZER_RECIPES


def test_public_multiscale_api_exists():
    assert hasattr(fwi, 'MultiscaleInversion'), 'public multiscale API is missing'


@pytest.fixture
def api():
    names = ('MultiscaleStage', 'MultiscaleInversion', 'MultiscaleResult',
             'MultiscaleStageResult', 'refine_slowness_squared')
    assert all(hasattr(fwi, name) for name in names), 'public multiscale API is missing'
    return fwi


def geometry(top='pml'):
    coarse = fwi.Grid2D((5, 7), (10., 10.), (20., -30.))
    fine = fwi.Grid2D((9, 13), (5., 5.), coarse.origin_m)
    acquisition = fwi.Acquisition(((35., -15.), (45., 5.)),
                                  ((35., -5.), (45., 15.)), (1., .7+.2j))
    boundaries = (fwi.BoundaryConfig(top, 2, 30.), fwi.BoundaryConfig(top, 4, 30.))
    return coarse, fine, acquisition, boundaries


def initial(grid, device='cpu', dtype=torch.float64):
    z = torch.arange(grid.shape[0], dtype=dtype, device=device) * grid.spacing_m[0]
    x = torch.arange(grid.shape[1], dtype=dtype, device=device) * grid.spacing_m[1]
    return (1/1600.**2 + 2e-10*z[:, None] + 1e-10*x[None, :]).rsqrt()


def problem(api, *, top='pml', steps=0, optimizer='Adam', configure=None,
            model_factory=None, regularizer_factory=None, device='cpu', solver_config=None):
    coarse, fine, acquisition, boundaries = geometry(top)
    stages = (api.MultiscaleStage(coarse, (2.,), steps, boundaries[0]),
              api.MultiscaleStage(fine, (2., 3.), steps, boundaries[1]))
    master = (3., 1., 2.)
    truth = initial(fine) + 35.
    observed = fwi.AcousticFWI(fine, acquisition, boundaries[1], fwi.SolverConfig()).simulate(truth, master)
    config = solver_config or fwi.SolverConfig(device=device)
    kwargs = {'lr': .025} if optimizer != 'LBFGS' else {
        'lr': .3, 'max_iter': 3, 'tolerance_grad': 1e-15,
        'tolerance_change': 1e-18, 'line_search_fn': 'strong_wolfe'}
    inversion = api.MultiscaleInversion(
        stages, acquisition, observed.to(device), master, initial(coarse, device),
        (1400., 1800.), config, lambda model, stage: make_optimizer(optimizer, model, **kwargs),
        weights=torch.tensor([2+.5j, .7-1j, 3.], dtype=torch.complex128, device=device)[:, None, None],
        configure_runner=configure, model_factory=model_factory,
        regularizer_factory=regularizer_factory)
    return inversion, observed


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_refinement_affine_si_nodes_and_identity(api, dtype):
    source = fwi.Grid2D((3, 4), (12., 8.), (17., -11.))
    target = fwi.Grid2D((5, 10), (6., 8/3), source.origin_m)
    def field(grid):
        z = grid.origin_m[0] + torch.arange(grid.shape[0], dtype=dtype)*grid.spacing_m[0]
        x = grid.origin_m[1] + torch.arange(grid.shape[1], dtype=dtype)*grid.spacing_m[1]
        return 4e-7+z[:, None]*1e-10+x[None, :]*2e-10
    m = field(source).requires_grad_()
    actual = api.refine_slowness_squared(m, source, target)
    torch.testing.assert_close(actual, field(target), rtol=2e-7 if dtype == torch.float32 else 1e-14, atol=0.)
    assert actual.dtype == m.dtype and actual.device == m.device and actual.grad_fn is None
    torch.testing.assert_close(actual[[0, -1]][:, [0, -1]], m[[0, -1]][:, [0, -1]], rtol=0., atol=0.)
    same = api.refine_slowness_squared(m, source, source)
    assert torch.equal(same, m) and same.data_ptr() != m.data_ptr() and not same.requires_grad
    constant = api.refine_slowness_squared(torch.full_like(m, 5e-7), source, target)
    torch.testing.assert_close(constant, torch.full(target.shape, 5e-7, dtype=dtype))


def test_refinement_independent_bilinear_reference_not_velocity(api):
    source = fwi.Grid2D((2, 2), (10., 20.), (15., -10.))
    target = fwi.Grid2D((3, 3), (5., 10.), source.origin_m)
    m = torch.tensor([[4., 8.], [12., 20.]], dtype=torch.float64)*1e-8
    expected = torch.tensor([[4., 6., 8.], [8., 11., 14.], [12., 16., 20.]], dtype=m.dtype)*1e-8
    result = api.refine_slowness_squared(m, source, target)
    torch.testing.assert_close(result, expected, atol=1e-23, rtol=1e-15)
    assert abs(result[1, 1].rsqrt() - m.rsqrt().mean()) > 100.


@pytest.mark.parametrize('case', ['coarsen', 'shift', 'extent', 'shape', 'nan', 'zero', 'int', 'complex'])
def test_refinement_rejects_invalid_inputs(api, case):
    coarse, fine, _, _ = geometry()
    m = initial(coarse).reciprocal().square()
    source, target = coarse, fine
    if case == 'coarsen': source, target, m = fine, coarse, initial(fine).reciprocal().square()
    if case == 'shift': target = replace(fine, origin_m=(21., -30.))
    if case == 'extent': target = replace(fine, shape=(8, 13))
    if case == 'shape': m = m[:-1]
    if case == 'nan': m[0, 0] = float('nan')
    if case == 'zero': m[0, 0] = 0.
    if case == 'int': m = m.to(torch.int64)
    if case == 'complex': m = m.to(torch.complex128)
    with pytest.raises((ValueError, TypeError)):
        api.refine_slowness_squared(m, source, target)


@pytest.mark.parametrize('change', [
    {'frequencies_hz': ()}, {'frequencies_hz': (2., 2.)}, {'frequencies_hz': (0.,)},
    {'frequencies_hz': (float('nan'),)}, {'steps': -1}, {'steps': True},
    {'steps': 1.2}, {'source_batch_size': 0}, {'source_batch_size': True},
    {'grid': None}, {'boundary': None},
])
def test_immutable_stage_rejects_invalid_configuration(api, change):
    coarse, _, _, boundaries = geometry()
    args = dict(grid=coarse, frequencies_hz=(2.,), steps=0, boundary=boundaries[0])
    args.update(change)
    with pytest.raises((ValueError, TypeError)):
        api.MultiscaleStage(**args)
    stage = api.MultiscaleStage(coarse, [2.], 0, boundaries[0])
    assert stage.frequencies_hz == (2.,)
    with pytest.raises(FrozenInstanceError): stage.steps = 2


@pytest.mark.parametrize('case', ['missing', 'duplicate_master', 'bad_master', 'decreasing',
    'coarsen', 'extent', 'origin', 'top', 'pml', 'dispersion', 'surface', 'data', 'precision', 'weights'])
def test_entire_schedule_rejected_before_optimizer_factory(api, case):
    coarse, fine, acq, boundaries = geometry()
    stages = [api.MultiscaleStage(coarse, (2.,), 1, boundaries[0]),
              api.MultiscaleStage(fine, (2., 3.), 1, boundaries[1])]
    master, weights = (2., 3.), 1.
    observed = torch.zeros((2, 2, 2), dtype=torch.complex128)
    if case == 'missing': stages[1] = replace(stages[1], frequencies_hz=(4.,))
    if case == 'duplicate_master': master = (2., 2.)
    if case == 'bad_master': master = (2., -3.)
    if case == 'decreasing': stages[1] = replace(stages[1], frequencies_hz=(1.,)); master = (2., 1.)
    if case == 'coarsen': stages[1] = replace(stages[1], grid=fwi.Grid2D((3, 4), (20., 20.), coarse.origin_m))
    if case == 'extent': stages[1] = replace(stages[1], grid=replace(fine, shape=(8, 13)))
    if case == 'origin': stages[1] = replace(stages[1], grid=replace(fine, origin_m=(0., 0.)))
    if case == 'top': stages[1] = replace(stages[1], boundary=replace(boundaries[1], top='free_surface'))
    if case == 'pml': stages[1] = replace(stages[1], boundary=replace(boundaries[1], pml_cells=2))
    if case == 'dispersion': stages[1] = replace(stages[1], frequencies_hz=(100.,)); master = (2., 100.)
    if case == 'surface':
        stages = [replace(stage, boundary=replace(stage.boundary, top='free_surface')) for stage in stages]
        acq = replace(acq, sources_m=((20., -15.), (45., 5.)))
    if case == 'data': observed[0, 0, 0] = complex('nan')
    if case == 'precision': observed = observed.to(torch.complex64)
    if case == 'weights': weights = float('inf')
    calls = []
    with pytest.raises((ValueError, TypeError)):
        api.MultiscaleInversion(stages, acq, observed, master, initial(coarse),
            (1400., 1800.), fwi.SolverConfig(), lambda *args: calls.append(args), weights=weights).run()
    assert not calls


def test_reordered_frequency_weights_and_prior_match_independent_sum(api):
    priors = []
    def regularizer(stage):
        prior = torch.full(stage.grid.shape, 1/1650.**2, dtype=torch.float64)
        result = fwi.Regularizer(stage.grid, 1e12, prior)
        priors.append(result)
        return result
    inversion, observed = problem(api, regularizer_factory=regularizer)
    original = observed.clone()
    observed.fill_(123.)  # caller mutation cannot alter the immutable master snapshot
    result = inversion.run()
    objective = inversion.runner.objective
    assert objective.frequencies_hz == (2., 3.) and len(priors) == 2
    torch.testing.assert_close(objective.observed, original[[2, 0]])
    torch.testing.assert_close(objective.weights[:, 0, 0], torch.tensor([3., 2+.5j], dtype=torch.complex128))
    m = result.slowness_squared.detach().requires_grad_()
    expected = priors[-1].value(m)
    facade = fwi.AcousticFWI(result.grid, inversion.acquisition, result.stages[-1].stage.boundary, fwi.SolverConfig())
    for frequency, index, weight in ((2., 2, 3.), (3., 0, 2+.5j)):
        single = facade.objective(original[index:index+1], (frequency,), weights=weight)
        expected = expected + single.loss(m)
    actual = objective.loss(m)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(torch.autograd.grad(actual, m)[0], torch.autograd.grad(expected, m)[0], rtol=1e-12, atol=1e-10)
    inversion.runner.closure()
    fresh = m.detach().requires_grad_()
    torch.testing.assert_close(inversion.runner.last_physical_gradient, torch.autograd.grad(objective.loss(fresh), fresh)[0], rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
@pytest.mark.parametrize('optimizer', ['Adam', 'LBFGS'])
def test_real_multistage_matches_manual_runners_and_decreases_each_stage(api, top, optimizer):
    seen, snapshots, refs = [], [], []
    def configure(index, stage, runner):
        assert not runner.optimizer.state
        refs.append(weakref.ref(runner))
        seen.append((index, stage, runner.model.slowness_squared().detach().clone()))
        def capture(current, step):
            snapshots.append((index, step.accepted_step, current.model.slowness_squared().detach().clone()))
        runner.accepted_callbacks = (capture,)
    inversion, observed = problem(api, top=top, steps=2, optimizer=optimizer, configure=configure)
    result = inversion.run()
    assert result.grid == inversion.stages[-1].grid
    assert result.velocity.grad_fn is None and result.slowness_squared.grad_fn is None
    assert [s.index for s in result.stages] == [0, 1]
    assert all(s.history[-1].loss < s.initial_loss for s in result.stages)
    assert all([r.accepted_step for r in s.history] == [1, 2] for s in result.stages)
    assert all(s.pml_widths_m == (20., 20.) and s.max_phase_error <= .01 for s in result.stages)
    transferred = api.refine_slowness_squared(snapshots[1][2], seen[0][1].grid, seen[1][1].grid)
    torch.testing.assert_close(seen[1][2], transferred, rtol=2e-15, atol=0.)
    # Independent orchestration uses PyTorch's node-coordinate transfer directly.
    velocity = initial(inversion.stages[0].grid)
    for stage, summary in zip(inversion.stages, result.stages):
        model = fwi.VelocityParameters(velocity, (1400., 1800.))
        facade = fwi.AcousticFWI(stage.grid, inversion.acquisition, stage.boundary, fwi.SolverConfig())
        indices = [(3., 1., 2.).index(f) for f in stage.frequencies_hz]
        weights = torch.tensor([2+.5j, .7-1j, 3.], dtype=torch.complex128)[indices, None, None]
        objective = facade.objective(observed[indices], stage.frequencies_hz, weights=weights)
        kwargs = {'lr': .025} if optimizer == 'Adam' else {'lr': .3, 'max_iter': 3,
            'tolerance_grad': 1e-15, 'tolerance_change': 1e-18, 'line_search_fn': 'strong_wolfe'}
        manual = fwi.InversionRunner(objective, model, make_optimizer(optimizer, model, **kwargs))
        assert summary.initial_loss == pytest.approx(objective.loss(model.slowness_squared()).item(), rel=1e-12)
        manual_result = manual.run(stage.steps)
        assert [r.loss for r in summary.history] == pytest.approx([r.loss for r in manual_result.history], rel=1e-11)
        velocity = torch.nn.functional.interpolate(manual_result.slowness_squared[None, None],
            size=inversion.stages[-1].grid.shape, mode='bilinear', align_corners=True)[0, 0].rsqrt()
        facade.solver.clear_cache()
    torch.testing.assert_close(result.velocity, manual_result.velocity, rtol=1e-12, atol=1e-9)
    gc.collect()
    assert refs[0]() is None and refs[1]() is inversion.runner
    assert not any(isinstance(value, torch.Tensor) for summary in result.stages for value in vars(summary).values())
    with pytest.raises(RuntimeError, match='once'): inversion.run()


@pytest.mark.parametrize('name', tuple(OPTIMIZER_RECIPES))
def test_all_native_optimizer_recipes_can_start_each_stage(api, name):
    def factory(velocity, stage):
        return fwi.VelocityParameters(velocity, (1400., 1800.),
            representation=OPTIMIZER_RECIPES[name].representation)
    inversion, _ = problem(api, model_factory=factory)
    inversion.optimizer_factory = lambda model, stage: make_optimizer(name, model)
    result = inversion.run()
    assert len(result.stages) == 2


def test_masked_sparse_factory_scheduler_and_final_curvature(api):
    masks, frozen, events = [], [], []
    def factory(velocity, stage):
        mask = torch.ones_like(velocity, dtype=torch.bool)
        mask[0] = False
        masks.append(mask)
        frozen.append(velocity[0].clone())
        return fwi.VelocityParameters(velocity, (1400., 1800.), mask, representation='sparse_embedding')
    def configure(index, stage, runner):
        runner.scheduler = torch.optim.lr_scheduler.StepLR(runner.optimizer, 1, gamma=.5)
        runner.accepted_callbacks = (lambda current, step: events.append((index, step.accepted_step, current.scheduler.last_epoch)),)
    inversion, _ = problem(api, optimizer='SparseAdam', steps=1, model_factory=factory, configure=configure)
    result = inversion.run()
    assert events == [(0, 1, 1), (1, 1, 1)]
    torch.testing.assert_close(result.velocity[0], frozen[-1], atol=0., rtol=0.)
    curvature = fwi.Curvature(inversion.runner.objective, result.slowness_squared, active_mask=masks[-1])
    action = curvature.gn_hvp(torch.ones(curvature.n_active, dtype=torch.float64))
    assert torch.isfinite(action).all() and action.norm() > 0


def test_saturated_final_model_fails_without_clipping(api):
    class Saturating(torch.optim.SGD):
        def step(self, closure):
            loss = closure()
            with torch.no_grad():
                for group in self.param_groups:
                    for parameter in group['params']: parameter.fill_(1000.)
            return loss
    inversion, _ = problem(api, steps=1)
    # Only the final optimizer saturates, so transition checks cannot hide it.
    inversion.optimizer_factory = lambda model, stage: (
        Saturating(model.parameters(), lr=.01) if stage is inversion.stages[-1]
        else make_optimizer('Adam', model, lr=.01))
    with pytest.raises(ValueError, match='saturat|strictly inside'):
        inversion.run()
    assert inversion.runner.objective.solver.cache_bytes == 0


@pytest.mark.parametrize('refined', [False, True])
@pytest.mark.parametrize('bound', [1000., 2000.])
def test_float32_frozen_bounds_survive_stage_transfer(api, refined, bound):
    coarse, fine, acquisition, boundaries = geometry()
    target = fine if refined else coarse
    target_boundary = boundaries[1] if refined else boundaries[0]
    stages = (api.MultiscaleStage(coarse, (2.,), 0, boundaries[0]),
              api.MultiscaleStage(target, (2.,), 0, target_boundary))
    velocity = torch.full(coarse.shape, 1500., dtype=torch.float32)
    velocity[0] = bound
    received = []
    def factory(value, stage):
        received.append(value.detach().clone())
        mask = torch.ones_like(value, dtype=torch.bool)
        mask[0] = False
        return fwi.VelocityParameters(value, (1000., 2000.), mask)
    inversion = api.MultiscaleInversion(stages, acquisition,
        torch.zeros((1, 2, 2), dtype=torch.complex64), (2.,), velocity,
        (1000., 2000.), fwi.SolverConfig(dtype=torch.complex64, rtol=1e-5, atol=1e-7),
        lambda model, stage: make_optimizer('Adam', model), model_factory=factory)
    result = inversion.run()
    assert bool(((result.velocity >= 1000.) & (result.velocity <= 2000.)).all())
    if not refined:
        assert torch.equal(received[1], received[0])
        assert torch.equal(result.velocity[0], velocity[0])
        assert result.stages[1].transfer_bound_corrections == 0
    else:
        torch.testing.assert_close(result.velocity[0], torch.full_like(result.velocity[0], bound),
                                   rtol=4*torch.finfo(torch.float32).eps, atol=0.)
        raw = api.refine_slowness_squared(velocity.reciprocal().square(), coarse, fine).rsqrt()
        correction = result.stages[1].max_transfer_bound_correction_m_s
        expected_count = int(((raw < 1000.) | (raw > 2000.)).sum())
        assert result.stages[1].transfer_bound_corrections == expected_count
        assert correction == pytest.approx(float((received[1]-raw).abs().max()))
        assert correction <= 4*torch.finfo(torch.float32).eps*bound
        if bound == 1000.:
            assert expected_count > 0 and correction > 0.
            assert bool((received[1][0] == bound).all())


def test_transfer_roundoff_does_not_relax_initial_bounds(api):
    coarse, _, acquisition, boundaries = geometry()
    velocity = torch.full(coarse.shape, 1500., dtype=torch.float32)
    velocity[0] = torch.tensor(1000., dtype=velocity.dtype).reciprocal().square().rsqrt()
    with pytest.raises(ValueError, match='initial_velocity.*bounds'):
        api.MultiscaleInversion((api.MultiscaleStage(coarse, (2.,), 0, boundaries[0]),),
            acquisition, torch.zeros((1, 2, 2), dtype=torch.complex64), (2.,), velocity,
            (1000., 2000.), fwi.SolverConfig(dtype=torch.complex64),
            lambda model, stage: make_optimizer('Adam', model))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_transfer_bound_correction_is_outward_only_and_bounded(dtype):
    from simplefwi.fwi2d.multiscale import _correct_transfer_bound_roundoff
    bounds = (1000., 2000.)
    lower, upper = torch.tensor(bounds, dtype=dtype)
    before = torch.stack((torch.nextafter(lower, lower-1), lower,
                          torch.nextafter(lower, lower+1), upper,
                          torch.nextafter(upper, upper+1)))
    actual, count, maximum = _correct_transfer_bound_roundoff(before, bounds)
    assert count == 2 and maximum > 0.
    assert actual[0] == lower and actual[-1] == upper
    assert torch.equal(actual[1:-1], before[1:-1])
    for invalid in (torch.tensor([999.], dtype=dtype), torch.tensor([2001.], dtype=dtype)):
        with pytest.raises(ValueError, match='roundoff|bounds'):
            _correct_transfer_bound_roundoff(invalid, bounds)


def test_corrected_endpoint_cannot_be_activated_on_target_grid(api):
    coarse, fine, acquisition, boundaries = geometry()
    stages = (api.MultiscaleStage(coarse, (2.,), 0, boundaries[0]),
              api.MultiscaleStage(fine, (2.,), 0, boundaries[1]))
    velocity = torch.full(coarse.shape, 1500., dtype=torch.float32)
    velocity[0] = 1000.
    def factory(value, stage):
        mask = torch.ones_like(value, dtype=torch.bool)
        if stage is stages[0]: mask[0] = False
        return fwi.VelocityParameters(value, (1000., 2000.), mask)
    inversion = api.MultiscaleInversion(stages, acquisition,
        torch.zeros((1, 2, 2), dtype=torch.complex64), (2.,), velocity,
        (1000., 2000.), fwi.SolverConfig(dtype=torch.complex64, rtol=1e-5, atol=1e-7),
        lambda model, stage: make_optimizer('Adam', model), model_factory=factory)
    with pytest.raises(ValueError, match='strictly inside'):
        inversion.run()


def test_configure_rejects_scheduler_owned_by_foreign_optimizer(api):
    foreign = torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=.8)
    scheduler = torch.optim.lr_scheduler.StepLR(foreign, 1, gamma=.5)
    inversion, _ = problem(api, steps=1,
        configure=lambda index, stage, runner: setattr(runner, 'scheduler', scheduler))
    with pytest.raises(ValueError, match='scheduler.*optimizer'):
        inversion.run()
    assert foreign.param_groups[0]['lr'] == .8 and scheduler.last_epoch == 0
    assert inversion.runner.accepted_steps == 0 and inversion.runner.objective.solver.cache_bytes == 0


@pytest.mark.parametrize('kind', ['step', 'plateau', 'chained', 'sequential'])
def test_fresh_native_schedulers_are_stage_local(api, kind):
    schedulers = []
    def configure(index, stage, runner):
        opt = runner.optimizer
        if kind == 'step': scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.5)
        elif kind == 'plateau': scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        elif kind == 'chained': scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.5),
            torch.optim.lr_scheduler.ExponentialLR(opt, .9)])
        else: scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [
            torch.optim.lr_scheduler.ConstantLR(opt, total_iters=2),
            torch.optim.lr_scheduler.ExponentialLR(opt, .9)], milestones=[2])
        runner.scheduler = scheduler
        schedulers.append(scheduler)
    inversion, _ = problem(api, steps=1, configure=configure)
    inversion.run()
    assert schedulers[0] is not schedulers[1]
    assert schedulers[1].optimizer is inversion.runner.optimizer
    assert all(getattr(s, 'last_epoch', 1) == 1 for s in schedulers)


@pytest.mark.parametrize('kind', ['step', 'plateau', 'chained'])
def test_configure_rejects_scheduler_with_existing_history(api, kind):
    def configure(index, stage, runner):
        if kind == 'step': scheduler = torch.optim.lr_scheduler.StepLR(runner.optimizer, 1, gamma=.5)
        elif kind == 'plateau': scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(runner.optimizer)
        else: scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.StepLR(runner.optimizer, 1, gamma=.5)])
        runner.optimizer.step()  # no gradients: native ordering, no physical update
        scheduler.step(1.) if kind == 'plateau' else scheduler.step()
        runner.scheduler = scheduler
    inversion, _ = problem(api, steps=1, configure=configure)
    with pytest.raises(ValueError, match='scheduler.*fresh|scheduler.*history'):
        inversion.run()
    assert inversion.runner.accepted_steps == 0


def test_configure_rejects_reused_scheduler_even_without_steps(api):
    retained = []
    def configure(index, stage, runner):
        if not retained:
            retained.append(torch.optim.lr_scheduler.StepLR(runner.optimizer, 1))
        else:
            retained[0].optimizer = runner.optimizer
        runner.scheduler = retained[0]
    inversion, _ = problem(api, configure=configure)
    with pytest.raises(ValueError, match='scheduler.*fresh|scheduler.*reused'):
        inversion.run()


@pytest.mark.parametrize('field', ['model', 'objective', 'optimizer', 'source_batch_size', 'accepted_callbacks'])
def test_configure_cannot_replace_core_stage_configuration(api, field):
    def configure(index, stage, runner):
        if field == 'model': runner.model = fwi.VelocityParameters(runner.model.velocity(), (1400., 1800.))
        elif field == 'objective': runner.objective = object()
        elif field == 'optimizer': runner.optimizer = make_optimizer('Adam', runner.model)
        elif field == 'source_batch_size': runner.source_batch_size = 2
        else: runner.accepted_callbacks = (None,)
    inversion, _ = problem(api, configure=configure)
    with pytest.raises((ValueError, TypeError), match='configure_runner|accepted_callbacks'):
        inversion.run()
    assert inversion.runner.accepted_steps == 0


@pytest.mark.parametrize('failure', ['numerical', 'accepted_callback', 'configure'])
def test_failure_preserves_runner_lifecycle_and_clears_cache(api, failure):
    events = []
    def configure(index, stage, runner):
        def accepted(current, step):
            events.append(step.accepted_step)
            if failure == 'accepted_callback': raise OSError('archive unavailable')
        runner.accepted_callbacks = (accepted,)
        if failure == 'configure':
            runner._accepted_loss()  # build actual factors before the external error
            raise OSError('configuration unavailable')
    inversion, _ = problem(api, steps=1, configure=configure)
    if failure == 'numerical':
        class Broken(torch.optim.SGD):
            def step(self, closure):
                result = closure()
                with torch.no_grad():
                    self.param_groups[0]['params'][0].fill_(float('nan'))
                return result
        inversion.optimizer_factory = lambda model, stage: Broken(model.parameters(), lr=.01)
    with pytest.raises((FloatingPointError, OSError)) as caught:
        inversion.run()
    assert any('Multiscale stage 0' in note for note in getattr(caught.value, '__notes__', ()))
    assert inversion.runner.objective.solver.cache_bytes == 0
    assert inversion.runner.accepted_steps == (1 if failure == 'accepted_callback' else 0)
    assert events == ([1] if failure == 'accepted_callback' else [])
    if failure == 'numerical':
        torch.testing.assert_close(inversion.runner.model.velocity(), initial(inversion.stages[0].grid))
        assert not inversion.runner.optimizer.state
    with pytest.raises(RuntimeError, match='once'): inversion.run()


@pytest.mark.parametrize('top', ['pml', 'free_surface'])
def test_stage_sampling_preserves_si_points_amplitudes_and_area_normalization(api, top):
    captured = []
    def configure(index, stage, runner):
        objective = runner.objective
        assert objective.sources.positions_m == objective.acquisition.sources_m
        assert objective.receivers.positions_m == objective.acquisition.receivers_m
        assert objective.acquisition.amplitudes == (1+0j, .7+.2j)
        rhs = objective.source_rhs(runner.model.slowness_squared())
        amplitudes = torch.tensor(objective.acquisition.amplitudes, dtype=rhs.dtype)
        area = stage.grid.spacing_m[0]*stage.grid.spacing_m[1]
        torch.testing.assert_close(rhs.sum(-1)*area, amplitudes, atol=1e-15, rtol=1e-15)
        assert objective.operator.layout.top_pad == (0 if top == 'free_surface' else stage.boundary.pml_cells)
        captured.append(weakref.ref(objective.sources))
    inversion, _ = problem(api, top=top, configure=configure)
    inversion.run()
    gc.collect()
    assert captured[0]() is None and captured[1]() is inversion.runner.objective.sources
    assert inversion.runner.objective.solver.cache_bytes == 0


@pytest.mark.parametrize('invalid', ['wrong_shape', 'changed_velocity', 'wrong_bounds', 'wrong_dtype', 'wrong_type', 'reused'])
def test_model_factory_must_honor_transferred_physical_state(api, invalid):
    retained = []
    def factory(velocity, stage):
        if invalid == 'reused' and retained: return retained[0]
        if invalid == 'wrong_shape': velocity = velocity[:-1]
        if invalid == 'changed_velocity': velocity = velocity + 1.
        if invalid == 'wrong_dtype': velocity = velocity.float()
        if invalid == 'wrong_type': return velocity
        bounds = (1300., 1900.) if invalid == 'wrong_bounds' else (1400., 1800.)
        model = fwi.VelocityParameters(velocity, bounds)
        retained.append(model)
        return model
    inversion, _ = problem(api, model_factory=factory)
    with pytest.raises((ValueError, TypeError), match='model_factory'):
        inversion.run()


@pytest.mark.parametrize('invalid', ['foreign_parameters', 'history', 'reused'])
def test_optimizer_factory_must_start_fresh_on_new_model(api, invalid):
    retained = []
    def factory(model, stage):
        if invalid == 'foreign_parameters':
            model = fwi.VelocityParameters(model.velocity(), model.bounds)
        if invalid == 'reused' and retained:
            # Rebinding an otherwise empty old optimizer is still reuse.
            optimizer = retained[0]
            optimizer.param_groups[0]['params'] = list(model.parameters())
            return optimizer
        optimizer = make_optimizer('Adam', model)
        retained.append(optimizer)
        if invalid == 'history':
            for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        return optimizer
    inversion, _ = problem(api)
    inversion.optimizer_factory = factory
    with pytest.raises(ValueError, match='optimizer|parameters'):
        inversion.run()


def test_frequency_only_bands_preserve_order_and_final_theta_curvature(api):
    coarse, _, acquisition, boundaries = geometry()
    master = (3., 2., 1.)
    observed = fwi.AcousticFWI(coarse, acquisition, boundaries[0], fwi.SolverConfig()).simulate(initial(coarse)+10., master)
    stages = [api.MultiscaleStage(coarse, band, 1, boundaries[0]) for band in ((1.,), (3., 2.))]
    seen = []
    inversion = api.MultiscaleInversion(stages, acquisition, observed, master, initial(coarse),
        (1400., 1800.), fwi.SolverConfig(), lambda model, stage: make_optimizer('Adam', model, lr=.01),
        configure_runner=lambda index, stage, runner: seen.append(runner.objective.frequencies_hz))
    result = inversion.run()
    assert seen == [(1.,), (3., 2.)]
    assert all(stage.history[-1].loss < stage.initial_loss for stage in result.stages)
    curvature = fwi.TransformedCurvature(inversion.runner.objective, inversion.runner.model)
    theta = inversion.runner.model.raw_parameter
    action = curvature.exact_hvp(torch.ones_like(theta).flatten())
    assert torch.isfinite(action).all() and action.norm() > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA hardware is required')
@pytest.mark.parametrize('top', ['pml', 'free_surface'])
def test_cuda_multistage_fgmres_matches_cpu_direct(api, top):
    config = fwi.SolverConfig(method='fgmres', device='cuda', dtype=torch.complex128,
        rtol=1e-12, atol=1e-15, maxiter=300, restart=50)
    cpu, _ = problem(api, top=top, steps=1)
    gpu, _ = problem(api, top=top, steps=1, device='cuda', solver_config=config)
    cpu_result, gpu_result = cpu.run(), gpu.run()
    assert gpu_result.velocity.is_cuda and gpu_result.slowness_squared.is_cuda
    assert gpu.runner.objective.observed.is_cuda and gpu.runner.objective.weights.is_cuda
    assert all(p.is_cuda for p in gpu.runner.model.parameters())
    assert gpu.runner.objective.solver.config.method == 'fgmres'
    for a, b in zip(cpu_result.stages, gpu_result.stages):
        assert b.initial_loss == pytest.approx(a.initial_loss, rel=1e-7, abs=1e-13)
        assert b.history[-1].loss == pytest.approx(a.history[-1].loss, rel=1e-7, abs=1e-13)
    torch.testing.assert_close(gpu_result.velocity.cpu(), cpu_result.velocity, rtol=1e-9, atol=1e-5)
    torch.testing.assert_close(gpu_result.slowness_squared.cpu(), cpu_result.slowness_squared, rtol=1e-8, atol=1e-16)
    # Compare resident waveforms and physical gradients at exactly the same m.
    m_cpu = cpu_result.slowness_squared.detach().requires_grad_()
    m_gpu = m_cpu.detach().cuda().requires_grad_()
    p_cpu, p_gpu = cpu.runner.objective.predict(m_cpu), gpu.runner.objective.predict(m_gpu)
    assert p_gpu.is_cuda
    assert (p_gpu.cpu()-p_cpu).norm()/p_cpu.norm() <= 1e-7
    g_cpu, = torch.autograd.grad(cpu.runner.objective.loss(m_cpu), m_cpu)
    g_gpu, = torch.autograd.grad(gpu.runner.objective.loss(m_gpu), m_gpu)
    assert g_gpu.is_cuda and (g_gpu.cpu()-g_cpu).norm()/g_cpu.norm() <= 1e-7
