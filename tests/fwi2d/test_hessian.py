import torch
from simplefwi.fwi2d.curvature import Curvature
from tests.fwi2d.test_objective import fixture


def test_exact_hvp_finite_difference_symmetry_and_regularizer_once():
    objective, m = fixture()
    c = Curvature(objective, m)
    v, w = torch.randn(2, c.n_active, dtype=m.dtype)
    v /= v.norm(); w /= w.norm()
    def gradient(flat):
        x = flat.detach().reshape_as(m).requires_grad_()
        return torch.autograd.grad(objective.loss(x), x)[0].flatten()
    eps = 1e-5
    fd = (gradient(m.flatten()+eps*v)-gradient(m.flatten()-eps*v))/(2*eps)
    hv = c.exact_hvp(v)
    assert (fd-hv).norm()/hv.norm() < 1e-5
    torch.testing.assert_close(w@hv, v@c.exact_hvp(w), rtol=1e-8, atol=1e-11)
    regularizer = objective.regularizer
    reg = torch.autograd.functional.hvp(regularizer.value, m, v.reshape_as(m))[1].flatten()
    torch.testing.assert_close(c.gn_hvp(v)-c.data_gn_hvp(v), reg, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(c.exact_hvp(v), Curvature(objective, m, source_batch_size=3).exact_hvp(v))


def test_zero_residual_equality_and_nonzero_residual_difference():
    objective, m = fixture()
    objective.regularizer = None
    c = Curvature(objective, m)
    v = torch.randn(c.n_active, dtype=m.dtype)
    assert (c.exact_hvp(v)-c.data_gn_hvp(v)).norm()/c.data_gn_hvp(v).norm() > 1e-3
    objective.observed = objective.predict(m).detach()
    zero = Curvature(objective, m)
    torch.testing.assert_close(zero.exact_hvp(v), zero.data_gn_hvp(v), rtol=1e-8, atol=1e-11)


def test_extreme_float32_weights_with_double_wavefields():
    from simplefwi.fwi2d.objective import FWIObjective
    from simplefwi.fwi2d.curvature import TransformedCurvature
    from simplefwi.fwi2d.parameters import VelocityParameters
    original, m = fixture()
    model = VelocityParameters(m.detach().rsqrt(), (0.5, 2.))
    # Use the transformed reference identically, including last-bit rounding.
    m = model.slowness_squared().detach()
    observed = original.predict(m).detach()
    for scale in (1e-30, 1e30):
        weights = torch.tensor(scale, dtype=torch.float32)
        objective = FWIObjective(original.operator, original.acquisition,
                                 original.frequencies_hz, observed, weights, None,
                                 original.solver)
        assert objective.weights.dtype == torch.float32
        for c in (Curvature(objective, m), TransformedCurvature(objective, model)):
            vector = torch.ones(c.n_active, dtype=m.dtype)
            exact = c.exact_hvp(vector)
            gn = c.data_gn_hvp(vector)
            assert torch.isfinite(gn).all() and bool((gn != 0).any())
            # No absolute tolerance may conceal the tiny-weight contribution.
            torch.testing.assert_close(gn, exact, rtol=1e-8, atol=0.)


def test_float64_weights_with_single_wavefields():
    from simplefwi.fwi2d.objective import FWIObjective
    from simplefwi.fwi2d.solvers.direct import DirectSolver
    from simplefwi.fwi2d.types import SolverConfig
    original, m = fixture()
    m = m.detach().float()
    solver = DirectSolver(SolverConfig(dtype=torch.complex64, rtol=1e-4, atol=1e-6))
    weights = torch.linspace(0.7, 1.3, original.observed.numel(), dtype=torch.float64).reshape_as(original.observed)
    objective = FWIObjective(original.operator, original.acquisition,
                             original.frequencies_hz, original.observed.to(torch.complex64),
                             weights, None, solver)
    objective.observed = objective.predict(m).detach()
    assert objective.weights.dtype == torch.float64
    c = Curvature(objective, m)
    vector = torch.ones(c.n_active, dtype=m.dtype)
    exact, gn = c.exact_hvp(vector), c.data_gn_hvp(vector)
    assert exact.dtype == gn.dtype == torch.float32
    assert torch.isfinite(gn).all() and bool((gn != 0).any())
    torch.testing.assert_close(gn, exact, rtol=2e-5, atol=0.)
