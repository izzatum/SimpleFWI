import pytest
import torch
from simplefwi.fwi2d.parameters import VelocityParameters
from simplefwi.fwi2d.inversion import InversionRunner
from tests.fwi2d.test_optimizer_inventory import Quadratic


class Regularized(Quadratic):
    class Penalty:
        def value(self, m):
            return ((m*1e6-0.35)**2).sum()*0.7
    regularizer = Penalty()
    def iter_data_losses(self, m, source_batch_size):
        for chunk in m.reshape(-1).split(source_batch_size):
            yield ((chunk*1e6-0.3)**2).sum()/2
    def loss(self, m):
        return super().loss(m)+self.regularizer.value(m)


def test_streaming_closure_equivalence_determinism_and_no_accumulation():
    model = VelocityParameters(torch.full((3, 4), 1600.), (1400., 2200.), None)
    objective = Regularized()
    runner = InversionRunner(objective, model, torch.optim.Adam(model.parameters()), source_batch_size=2)
    full = objective.loss(model.slowness_squared())
    expected, = torch.autograd.grad(full, tuple(model.parameters()))
    losses, gradients = [], []
    for _ in range(2):
        losses.append(runner.closure())
        gradients.append(next(model.parameters()).grad.clone())
    torch.testing.assert_close(losses[0], full.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], expected)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    assert losses[0].grad_fn is None


def test_lbfgs_reports_accepted_state_not_original_loss():
    model = VelocityParameters(torch.full((3, 4), 1600., dtype=torch.float64), (1400., 2200.), None)
    objective = Regularized()
    opt = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=4,
                             line_search_fn='strong_wolfe', tolerance_grad=1e-12)
    runner = InversionRunner(objective, model, opt, source_batch_size=2)
    initial = objective.loss(model.slowness_squared()).item()
    result = runner.step()
    assert result.closure_evaluations > 1
    assert result.loss < initial
    assert result.loss == pytest.approx(objective.loss(model.slowness_squared()).item(), rel=1e-14)


def test_actual_fwi_streaming_gradient_representation_equivalence_and_trial_cache():
    from tests.fwi2d.test_objective import fixture
    objective, m = fixture()
    velocity = m.detach().rsqrt()
    mask = torch.ones_like(m, dtype=torch.bool)
    mask[0, 0] = False
    dense = VelocityParameters(velocity, (0.5, 2.), mask)
    sparse = VelocityParameters(velocity, (0.5, 2.), mask, representation='sparse_embedding')
    expected_loss = objective.loss(dense.slowness_squared())
    expected_gradient, = torch.autograd.grad(expected_loss, dense.raw_parameter)
    runners = [InversionRunner(objective, dense, torch.optim.Adam(dense.parameters()), source_batch_size=1),
               InversionRunner(objective, sparse, torch.optim.SparseAdam(sparse.parameters()), source_batch_size=1)]
    losses = [r.closure() for r in runners]
    for loss in losses:
        torch.testing.assert_close(loss, expected_loss.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dense.raw_parameter.grad, expected_gradient, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(sparse.raw_parameter.grad.to_dense().squeeze(-1),
                               expected_gradient[mask], rtol=1e-12, atol=1e-12)
    factors = objective.solver.factorization_count
    loss0, gradient0 = runners[0].closure(), dense.raw_parameter.grad.clone()
    assert objective.solver.factorization_count == factors
    with torch.no_grad():
        dense.raw_parameter.add_(0.1)
    loss1 = runners[0].closure()
    assert objective.solver.factorization_count > factors
    assert not torch.isclose(loss0, loss1, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(gradient0, dense.raw_parameter.grad, rtol=1e-10, atol=1e-12)
