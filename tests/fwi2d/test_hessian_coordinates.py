import copy
import math
import pytest
import torch
from simplefwi.fwi2d.curvature import Curvature, TransformedCurvature
from simplefwi.fwi2d.parameters import VelocityParameters
from tests.fwi2d.test_objective import fixture


def test_nonlinear_theta_exact_dense_reference_and_snapshot():
    objective, m = fixture()
    mask = torch.ones_like(m, dtype=torch.bool); mask[1, 1] = False
    model = VelocityParameters(m.detach().rsqrt(), (0.5, 2.), mask)
    frozen = copy.deepcopy(model)
    c = TransformedCurvature(objective, model)
    vector = torch.randn(c.n_active, dtype=m.dtype)
    def dense_loss(active):
        theta = frozen.raw_parameter.detach().masked_scatter(frozen.active_mask, active)
        physical = frozen.slowness_squared_from(theta)
        total = objective.regularizer.value(physical)
        eye = torch.eye(objective.operator.n_dof, dtype=torch.complex128)
        for fi, frequency in enumerate(objective.frequencies_hz):
            matrix = objective.operator.apply(physical, 2*math.pi*frequency, eye).T
            wave = torch.linalg.solve(matrix, objective.source_rhs(physical).T).T
            residual = objective.weights[fi]*(objective.receivers.sample(wave)-objective.observed[fi])
            total = total+0.5*residual.abs().square().sum()
        return total
    active = frozen.raw_parameter.detach()[mask]
    hessian = torch.autograd.functional.hessian(dense_loss, active)
    expected = hessian@vector
    torch.testing.assert_close(c.exact_hvp(vector), expected, rtol=1e-9, atol=1e-10)
    eps = 1e-5
    grad = lambda t: torch.autograd.functional.jacobian(dense_loss, t)
    fd = (grad(active+eps*vector)-grad(active-eps*vector))/(2*eps)
    assert (expected-fd).norm()/expected.norm() < 1e-5
    # Pure first-order chain rule omits a nonzero transformation term.
    jac = torch.autograd.functional.jacobian(lambda t: frozen.slowness_squared_from(
        frozen.raw_parameter.detach().masked_scatter(mask,t)).flatten(), active)
    physical = Curvature(objective, frozen.slowness_squared())
    incomplete = jac.T@physical.exact_hvp(jac@vector)
    assert (expected-incomplete).norm()/expected.norm() > 1e-3
    assert c.coordinates == 'theta'
    data = torch.randn_like(objective.observed)
    torch.testing.assert_close((c.jvp(vector).conj()*data).real.sum(), vector@c.vjp(data))
    with torch.no_grad():
        model.theta.add_(2.)
        model.active_mask.fill_(False)
        model.background.mul_(1.1)
    torch.testing.assert_close(c.exact_hvp(vector), expected, rtol=1e-9, atol=1e-10)


def test_sparse_parameterization_physical_curvature():
    objective, m = fixture()
    model = VelocityParameters(m.detach().rsqrt(), (0.5, 2.), representation='sparse_embedding')
    sparse = Curvature(objective, model.slowness_squared(), active_mask=model.active_mask)
    dense = Curvature(objective, m)
    vector = torch.randn(sparse.n_active, dtype=m.dtype)
    torch.testing.assert_close(sparse.exact_hvp(vector), dense.exact_hvp(vector), rtol=1e-9, atol=1e-10)
    with pytest.raises(ValueError, match='dense'):
        TransformedCurvature(objective, model)


@pytest.mark.parametrize('regularized', [False, True])
def test_theta_zero_residual_and_exact_transformed_regularization(regularized):
    objective, m = fixture()
    if not regularized:
        objective.regularizer = None
    model = VelocityParameters(m.detach().rsqrt(), (0.5, 2.))
    objective.observed = objective.predict(model.slowness_squared()).detach()
    c = TransformedCurvature(objective, model)
    vector = torch.randn(c.n_active, dtype=m.dtype)
    torch.testing.assert_close(c.exact_hvp(vector), c.gn_hvp(vector), rtol=1e-9, atol=1e-11)
    if regularized:
        def reg(active):
            return objective.regularizer.value(model.slowness_squared_from(active.reshape_as(m)))
        expected = torch.autograd.functional.hvp(reg, model.theta.detach().flatten(), vector)[1]
        torch.testing.assert_close(c.gn_hvp(vector)-c.data_gn_hvp(vector), expected,
                                   rtol=1e-9, atol=1e-11)
