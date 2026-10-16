import pytest
import torch
from simplefwi.fwi2d.curvature import Curvature, explicit_matrix
from tests.fwi2d.test_objective import fixture


def test_jvp_vjp_active_boundary_and_frozen_snapshot():
    objective, m = fixture()
    mask = torch.zeros_like(m, dtype=torch.bool)
    mask[0, :] = True  # includes physical-to-PML extension derivatives
    mask[1, 1] = True
    original = m.detach().clone()
    curvature = Curvature(objective, m, active_mask=mask)
    vector = torch.randn(curvature.n_active, dtype=m.dtype)
    data = torch.randn_like(objective.observed)
    dm = curvature.inject(vector)
    eps = 1e-6
    fd = (objective.predict(original+eps*dm)-objective.predict(original-eps*dm))/(2*eps)
    torch.testing.assert_close(curvature.jvp(vector), fd, rtol=1e-7, atol=1e-9)
    left = (curvature.jvp(vector).conj()*data).real.sum()
    right = (vector*curvature.vjp(data)).sum()
    torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)
    expected = curvature.jvp(vector)
    with torch.no_grad():
        m.mul_(1.1)
        mask.fill_(False)
    torch.testing.assert_close(curvature.jvp(vector), expected)
    torch.testing.assert_close(curvature.restrict(dm), vector)


def test_weighted_gn_matches_explicit_real_jacobian():
    objective, m = fixture()
    curvature = Curvature(objective, m)
    vector = torch.randn(curvature.n_active, dtype=m.dtype)
    # Tiny independent reverse-mode explicit Jacobian, never production code.
    def real_data(flat):
        return torch.view_as_real(objective.predict(flat.reshape_as(m))).flatten()
    jac = torch.autograd.functional.jacobian(real_data, m.detach().flatten())
    weights = objective.weights.abs().square().flatten().repeat_interleave(2)
    expected = jac.T@(weights*(jac@vector))
    torch.testing.assert_close(curvature.data_gn_hvp(vector), expected, rtol=1e-8, atol=1e-11)
    batched = Curvature(objective, m, source_batch_size=2)
    torch.testing.assert_close(batched.data_gn_hvp(vector), expected, rtol=1e-8, atol=1e-11)


def test_guards_and_no_dense_default(monkeypatch):
    objective, m = fixture()
    c = Curvature(objective, m)
    calls = []
    def action(v):
        calls.append(v)
        return v
    with pytest.raises(MemoryError):
        explicit_matrix(action, 1000000, max_bytes=100)
    assert not calls
    torch.testing.assert_close(explicit_matrix(action, 3, max_bytes=1024), torch.eye(3, dtype=m.dtype))
    with pytest.raises(ValueError):
        c.jvp(torch.ones_like(m))
    with pytest.raises(ValueError):
        Curvature(objective, m, source_batch_size=0)
    original = objective.solver.solve
    batches = []
    def solve(operator, m, omega, rhs, **kwargs):
        batches.append(rhs.shape[0])
        return original(operator, m, omega, rhs, **kwargs)
    monkeypatch.setattr(objective.solver, 'solve', solve)
    monkeypatch.setattr(torch.autograd.functional, 'jacobian', lambda *a, **kw: pytest.fail('dense Jacobian'))
    monkeypatch.setattr(objective, 'predict', lambda *a, **kw: pytest.fail('all-source graph'))
    v = torch.ones(c.n_active, dtype=m.dtype)
    c.jvp(v); c.vjp(torch.ones_like(objective.observed)); c.data_gn_hvp(v); c.exact_hvp(v)
    assert max(batches) == 1
