"""Tiny dense references only; production inverse actions remain matrix-free."""
from types import SimpleNamespace

import pytest
import torch

from simplefwi.fwi2d.uncertainty import (
    CGConfig, LocalGaussian, PrecisionError, PriorPrecision, SmoothingPrecision,
)

DT = torch.float64


class MatrixCurvature:
    coordinates = 'physical_slowness_squared'
    def __init__(self, h):
        self.h = h
        self.n_active = h.shape[0]
        self.reference_model = torch.ones(self.n_active, dtype=DT)
        self.noise_whitening = 'fixed W; precision J^H W^H W J'
    def data_gn_hvp(self, v):
        return self.h @ v
    def gn_hvp(self, v):
        raise AssertionError('inversion regularizer must not enter posterior precision')


def fixture():
    a = torch.tensor([[2., 1., 0.], [0., 2., 1.], [1., 0., 1.]], dtype=DT)
    h = a.T @ a
    diagonal = torch.tensor([1., 2., 3.], dtype=DT)
    return MatrixCurvature(h), diagonal, h + torch.diag(diagonal)


def test_inverse_actions_true_residual_and_tiny_rhs():
    c, d, p = fixture()
    posterior = LocalGaussian(c, d, solve_config=CGConfig(rtol=1e-12, atol=0))
    v = torch.tensor([1., -2., .5], dtype=DT)
    for scale in (1., 1e-25, 1e-250):
        expected = torch.linalg.solve(p, scale*v)
        actual = posterior.covariance_action(scale*v)
        assert ((actual-expected)/scale).norm()/(expected/scale).norm() <= 1e-8
        assert posterior.last_solve.relative_residual < 1e-11
    assert posterior.last_solve.converged
    assert posterior.metadata['coordinates'] == 'physical_slowness_squared'
    assert posterior.metadata['noise_whitening']
    assert torch.equal(posterior.covariance_action(torch.zeros_like(v)), torch.zeros_like(v))


def test_seeded_diagonal_sampling_error_and_inverse_not_reciprocal():
    c, d, p = fixture()
    posterior = LocalGaussian(c, d)
    estimate = posterior.diagonal_estimate(2048, seed=19)
    again = posterior.diagonal_estimate(2048, seed=19)
    torch.testing.assert_close(estimate.mean, again.mean, rtol=0, atol=0)
    exact = torch.linalg.inv(p).diagonal()
    # Fixed-seed check uses six estimated standard errors, not a fragile
    # arbitrary percent tolerance. This is a diagnostic, not simultaneous CI.
    assert torch.all((estimate.mean-exact).abs() <= 6*estimate.standard_error + 1e-10)
    assert torch.all(estimate.standard_error > 0)
    assert estimate.probes == 2048 and estimate.seed == 19
    assert estimate.max_solve_error_bound < 1e-8
    assert not torch.allclose(exact, p.diagonal().reciprocal(), rtol=.01, atol=0)


def test_prior_not_double_counted_and_smoothing_inverse():
    c, d, p = fixture()
    l = torch.tensor([[1., -1., 0.], [0., 1., -1.]], dtype=DT)
    prior = PriorPrecision(d, smoothing=SmoothingPrecision(lambda v: l@v, lambda w: l.T@w))
    posterior = LocalGaussian(c, prior)
    v = torch.ones(3, dtype=DT)
    torch.testing.assert_close(posterior.covariance_action(v), torch.linalg.solve(p+l.T@l, v))
    modes = posterior.leading_modes(3, seed=4)
    torch.testing.assert_close(modes.covariance_action(v), posterior.covariance_action(v), rtol=1e-9, atol=1e-10)


def test_prior_preconditioned_lanczos_rank_convergence_and_diagnostics():
    c, d, p = fixture()
    posterior = LocalGaussian(c, d)
    v = torch.tensor([1., -2., .5], dtype=DT)
    exact = torch.linalg.solve(p, v)
    low = posterior.leading_modes(1, seed=5)
    full = posterior.leading_modes(3, seed=5)
    assert (full.covariance_action(v)-exact).norm() < (low.covariance_action(v)-exact).norm()
    torch.testing.assert_close(full.covariance_action(v), exact, rtol=1e-9, atol=1e-10)
    expected = torch.linalg.eigvalsh(c.h / torch.sqrt(d[:,None]*d[None,:])).flip(0)
    torch.testing.assert_close(full.eigenvalues, expected, rtol=1e-10, atol=1e-11)
    assert full.ritz_residual_norms.max() < 1e-9
    assert full.basis_dimension == 3 and full.exhausted_space
    assert low.rank == 1 and not low.exact_covariance


def test_improper_indefinite_and_failed_solves_rejected():
    c, d, p = fixture()
    for invalid in (torch.zeros(3, dtype=DT), -d, torch.tensor([1., float('nan'), 1.], dtype=DT)):
        with pytest.raises(ValueError, match='positive'):
            LocalGaussian(c, invalid)
    # Exact Hessians (potentially indefinite) are never an accepted action API.
    with pytest.raises(TypeError, match='data_gn_hvp'):
        LocalGaussian(SimpleNamespace(exact_hvp=lambda v: -v), d)
    bad = LocalGaussian(MatrixCurvature(-10*torch.eye(3, dtype=DT)), d)
    with pytest.raises(PrecisionError, match='positive'):
        bad.covariance_action(torch.ones(3, dtype=DT))
    limited = LocalGaussian(c, d, solve_config=CGConfig(maxiter=1, rtol=1e-14, atol=0))
    with pytest.raises(PrecisionError, match='converge'):
        limited.covariance_action(torch.ones(3, dtype=DT))
    assert not limited.last_solve.converged


def test_no_dense_default_and_bounded_mode_storage(monkeypatch):
    c, d, _ = fixture()
    posterior = LocalGaussian(c, d)
    monkeypatch.setattr(torch.linalg, 'inv', lambda *a, **k: pytest.fail('dense inverse'))
    monkeypatch.setattr(torch.linalg, 'solve', lambda *a, **k: pytest.fail('dense solve'))
    posterior.covariance_action(torch.ones(3, dtype=DT))
    posterior.diagonal_estimate(3, seed=0)
    posterior.leading_modes(2, seed=0)
    with pytest.raises(MemoryError):
        posterior.leading_modes(2, seed=0, max_bytes=1)
    with pytest.raises(ValueError):
        posterior.diagonal_estimate(1, seed=0)


def test_rank_cap_reports_partial_exploration():
    c = MatrixCurvature(torch.diag(torch.arange(1., 11., dtype=DT)))
    posterior = LocalGaussian(c, torch.ones(10, dtype=DT))
    modes = posterior.leading_modes(2, seed=4, max_basis=5)
    assert modes.basis_dimension == 5
    assert not modes.exhausted_space and not modes.exact_covariance
    assert modes.rank == 2


def test_actual_fwi_curvature_explicit_reference_prior_once():
    from simplefwi.fwi2d.curvature import Curvature, explicit_matrix
    from tests.fwi2d.test_objective import fixture as fwi_fixture
    objective, m = fwi_fixture()
    mask = torch.zeros_like(m, dtype=torch.bool)
    mask[1, :3] = True
    curvature = Curvature(objective, m, active_mask=mask)
    data = explicit_matrix(curvature.data_gn_hvp, 3, max_bytes=4096)
    d = torch.tensor([1., 2., 3.], dtype=DT)
    gaussian = LocalGaussian(curvature, d)
    v = torch.tensor([.5, -1., 2.], dtype=DT)
    expected = torch.linalg.solve(data+torch.diag(d), v)
    torch.testing.assert_close(gaussian.covariance_action(v), expected, rtol=1e-8, atol=1e-11)
    # Objective has a nonzero spatial regularizer: including it would differ.
    total = explicit_matrix(curvature.gn_hvp, 3, max_bytes=4096)
    assert (gaussian.covariance_action(v)-torch.linalg.solve(total+torch.diag(d), v)).norm() > 1e-3
    assert gaussian.metadata['noise_whitening']['weights'].shape == objective.weights.shape


def test_full_modes_include_repeated_eigenvalue_eigenspace():
    h = torch.diag(torch.tensor([2., 2., 4.], dtype=DT))
    posterior = LocalGaussian(MatrixCurvature(h), torch.ones(3, dtype=DT))
    modes = posterior.leading_modes(3, seed=3)
    assert modes.rank == 3 and modes.exhausted_space
    torch.testing.assert_close(modes.eigenvalues, torch.tensor([4., 2., 2.], dtype=DT))
