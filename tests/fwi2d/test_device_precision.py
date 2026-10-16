"""Precision candidates; CUDA checks remain explicitly skipped without hardware."""
import pytest
import torch
from simplefwi.fwi2d import Grid2D, BoundaryConfig, SolverConfig
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.autograd import implicit_solve
from simplefwi.fwi2d.solvers.direct import DirectSolver
from simplefwi.fwi2d.solvers.krylov import FGMRESSolver
from simplefwi.fwi2d.solvers.multigrid import MultigridConfig, ShiftedHelmholtzMultigrid


def fixture(device, dtype):
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    op = HelmholtzOperator(Grid2D((3, 4), (.4, .6)), BoundaryConfig(pml_cells=2, sigma_max=5.))
    mg = ShiftedHelmholtzMultigrid(op, MultigridConfig(shift=.5, max_levels=5, coarse_max_dof=4))
    solver = FGMRESSolver(SolverConfig(method='fgmres', device=device, dtype=dtype,
        rtol=1e-12 if real == torch.float64 else 2e-6, atol=1e-14 if real == torch.float64 else 1e-8,
        maxiter=100, restart=20), mg)
    theta = torch.zeros(op.grid.shape, dtype=real, device=device, requires_grad=True)
    q = torch.randn((2, op.n_dof), generator=torch.Generator().manual_seed(62), dtype=torch.complex128).to(device=device, dtype=dtype)
    return op, solver, theta, q


def values_and_gradient(op, solver, theta, q):
    m = .6*theta.exp()
    u = implicit_solve(m, q, 2.3, op, solver)
    gradient, = torch.autograd.grad(.5*u.abs().square().sum(), theta)
    # theta=log(m/.6), so convert to the required physical-m gradient.
    return u.detach(), (gradient/m).detach()


@pytest.mark.parametrize('dtype', [torch.complex128, torch.complex64])
def test_cpu_precision_waveform_and_physical_gradient(dtype):
    op, solver, theta, q = fixture('cpu', dtype)
    u, g = values_and_gradient(op, solver, theta, q)
    theta64 = theta.detach().double().requires_grad_()
    reference, gradient = values_and_gradient(op, DirectSolver(), theta64, q.to(torch.complex128))
    wave_limit, grad_limit = ((1e-7, 1e-7) if dtype == torch.complex128 else (1e-3, 1e-2))
    assert (u-reference).norm()/reference.norm() <= wave_limit
    assert (g-gradient).norm()/gradient.norm() <= grad_limit


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=[pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware absent; Taylor/inversion unrun')])])
@pytest.mark.parametrize('dtype', [torch.complex128, torch.complex64])
def test_candidate_taylor_and_small_inversion(dtype, device):
    op, solver, theta, q = fixture(device, dtype)
    with torch.no_grad():
        truth = .6*(theta+.12).exp()
        target, _ = DirectSolver().solve(op, truth.cpu().double(), 2.3, q.cpu().to(torch.complex128))
        target = target.to(device=device, dtype=dtype)
    scale = target.abs().square().sum()
    def loss(t):
        u = implicit_solve(.6*t.exp(), q, 2.3, op, solver)
        return .5*(u-target).abs().square().sum()/scale
    direction = torch.ones_like(theta)/theta.numel()**.5
    value = loss(theta)
    gradient, = torch.autograd.grad(value, theta)
    derivative = (gradient*direction).sum()
    epsilons = (.08, .04, .02, .01)
    errors = torch.stack([(loss(theta+epsilon*direction)-value-epsilon*derivative).abs().detach()
                          for epsilon in epsilons])
    slopes = torch.log2(errors[:-1]/errors[1:])
    assert bool(((slopes > 1.8) & (slopes < 2.2)).all()), slopes
    optimizer = torch.optim.SGD([theta], lr=.2)
    initial = float(loss(theta).detach())
    for _ in range(5):
        optimizer.zero_grad()
        current = loss(theta)
        current.backward()
        optimizer.step()
    assert float(loss(theta).detach()) < initial


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware absent; device validation unrun')
@pytest.mark.parametrize('dtype', [torch.complex128, torch.complex64])
def test_cuda_waveform_gradient_and_resident_hierarchy(dtype):
    op, solver, theta, q = fixture('cuda', dtype)
    actual, gradient = values_and_gradient(op, solver, theta, q)
    expected, ref_gradient = values_and_gradient(op, DirectSolver(),
        theta.detach().cpu().double().requires_grad_(), q.cpu().to(torch.complex128))
    wave_limit, grad_limit = ((1e-7, 1e-7) if dtype == torch.complex128 else (1e-3, 1e-2))
    assert actual.device.type == gradient.device.type == 'cuda'
    assert float((actual.cpu()-expected).norm()/expected.norm()) <= wave_limit
    assert float((gradient.cpu()-ref_gradient).norm()/ref_gradient.norm()) <= grad_limit
    mg = solver.preconditioner
    assert all(level.values.device.type == 'cuda' for level in mg.levels)
    assert mg._lu.device.type == mg._model.device.type == 'cuda'
    assert all(t.device.type == 'cuda' for transfer in mg.transfers for axis in transfer.axes for t in axis)
