import pytest
import numpy as np
import torch
from simplefwi.fwi2d.objective import FWIObjective
from simplefwi.fwi2d.regularization import Regularizer
from simplefwi.fwi2d.operator import HelmholtzOperator
from simplefwi.fwi2d.solvers.direct import DirectSolver
from simplefwi.fwi2d.types import Acquisition, BoundaryConfig, Grid2D


def fixture():
    torch.manual_seed(73)
    grid = Grid2D((3, 4), (0.5, 0.6))
    operator = HelmholtzOperator(grid, BoundaryConfig(pml_cells=2, sigma_max=4.))
    acq = Acquisition(((0.2, 0.3), (0.7, 1.0), (0.8, 1.5)),
                      ((0.4, 0.5), (0.9, 1.3)), (1., 2j, 0.3+0.5j))
    m = (0.7+0.1*torch.rand(grid.shape, dtype=torch.float64)).requires_grad_()
    observed = torch.randn((2, 3, 2), dtype=torch.complex128)
    weights = torch.randn((2, 3, 2), dtype=torch.complex128)
    regularizer = Regularizer(grid, 0.25, torch.zeros_like(m))
    objective = FWIObjective(operator, acq, (0.3, 0.5), observed, weights,
                             regularizer, DirectSolver())
    return objective, m


def test_batched_prediction_losses_gradients_and_regularization_once():
    objective, m = fixture()
    predicted = objective.predict(m)
    assert predicted.shape == (2, 3, 2)
    torch.testing.assert_close(predicted, objective.predict(m, source_batch_size=1))
    explicit_data = 0.5*(objective.weights*(predicted-objective.observed)).abs().square().sum()
    loss = objective.loss(m)
    torch.testing.assert_close(loss, explicit_data+objective.regularizer.value(m))
    full_grad, = torch.autograd.grad(loss, m)
    for batch in (1, 2, 3, 4):
        data_loss = sum(objective.iter_data_losses(m, batch))
        torch.testing.assert_close(data_loss, explicit_data)
        grad, = torch.autograd.grad(data_loss+objective.regularizer.value(m), m)
        torch.testing.assert_close(grad, full_grad, atol=1e-12, rtol=1e-12)
    data_grad, = torch.autograd.grad(objective.data_loss(m), m)
    reg_grad, = torch.autograd.grad(objective.regularizer.value(m), m)
    torch.testing.assert_close(full_grad, data_grad+reg_grad)
    # Streaming training frees each batch graph before producing the next.
    m.grad = None
    for contribution in objective.iter_data_losses(m, 1):
        contribution.backward()
    objective.regularizer.value(m).backward()
    torch.testing.assert_close(m.grad, full_grad, atol=1e-12, rtol=1e-12)


def test_combined_objective_gradgrad_and_taylor():
    objective, m = fixture()
    assert torch.autograd.gradcheck(objective.loss, (m,), fast_mode=True)
    assert torch.autograd.gradgradcheck(objective.loss, (m,), fast_mode=True)
    direction = torch.randn_like(m)
    direction /= direction.norm()
    loss = objective.loss(m)
    grad, = torch.autograd.grad(loss, m)
    deriv = (grad*direction).sum()
    errors = [(objective.loss(m+e*direction)-loss-e*deriv).abs().item()
              for e in (0.02, 0.01, 0.005, 0.0025)]
    slopes = torch.log2(torch.tensor(errors[:-1])/torch.tensor(errors[1:]))
    assert torch.all((slopes > 1.8) & (slopes < 2.2)), slopes


def test_invalid_inputs():
    objective, m = fixture()
    with pytest.raises(ValueError):
        objective.predict(m, source_batch_size=0)
    with pytest.raises(ValueError):
        FWIObjective(objective.operator, objective.acquisition, (0., 1.),
                     objective.observed, 1., None, objective.solver)


@pytest.mark.parametrize('weight', [1.00000001, 1e-50, 1.00000001+1e-8j,
    [1.00000001, 1e-50], [1.00000001+1e-8j, 1e-50j]])
def test_python_weights_preserve_double_loss_and_gradient(weight):
    objective, m = fixture()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        actual = FWIObjective(objective.operator, objective.acquisition,
            objective.frequencies_hz, objective.observed, weight, None, objective.solver)
        expected = FWIObjective(objective.operator, objective.acquisition,
            objective.frequencies_hz, objective.observed,
            torch.tensor(weight, dtype=torch.complex128), None, objective.solver)
    finally:
        torch.set_default_dtype(original_dtype)
    actual_loss, expected_loss = actual.loss(m), expected.loss(m)
    assert actual_loss > 0
    torch.testing.assert_close(actual_loss, expected_loss, atol=0., rtol=1e-13)
    actual_grad, = torch.autograd.grad(actual_loss, m)
    expected_grad, = torch.autograd.grad(expected_loss, m)
    assert torch.count_nonzero(actual_grad) > 0
    torch.testing.assert_close(actual_grad, expected_grad, atol=0., rtol=1e-13)


@pytest.mark.parametrize('weight', [torch.tensor([1.00000001, 1e-50], dtype=torch.float64),
    np.array([1.00000001+1e-8j, 1e-50j], dtype=np.complex128), np.float64(1.00000001)])
def test_explicit_typed_weights_never_downcast(weight):
    objective, _ = fixture()
    actual = FWIObjective(objective.operator, objective.acquisition,
        objective.frequencies_hz, objective.observed.to(torch.complex64),
        weight, None, objective.solver)
    expected = torch.as_tensor(weight).expand_as(actual.weights)
    assert actual.weights.dtype == expected.dtype
    torch.testing.assert_close(actual.weights, expected, atol=0., rtol=0.)


def test_exact_fit_hessian_equals_weighted_jacobian_normal():
    objective, m = fixture()
    observed = objective.predict(m).detach()
    exact_fit = FWIObjective(objective.operator, objective.acquisition,
        objective.frequencies_hz, observed, objective.weights, None, objective.solver)
    assert exact_fit.loss(m).item() == 0.
    def weighted_real_data(model):
        return torch.view_as_real(exact_fit.weights*exact_fit.predict(model)).reshape(-1)
    jacobian = torch.autograd.functional.jacobian(weighted_real_data, m).reshape(-1, m.numel())
    expected = jacobian.T@jacobian
    actual = torch.autograd.functional.hessian(exact_fit.loss, m).reshape(m.numel(), m.numel())
    assert expected.norm() > 0.
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
