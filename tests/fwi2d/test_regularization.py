import torch
import pytest
from simplefwi.fwi2d.regularization import Regularizer
from simplefwi.fwi2d.types import Grid2D


def test_half_factor_spacing_area_and_spatial_axes():
    grid = Grid2D((3, 4), (2., 3.))
    m = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    reg = Regularizer(grid, 0.7, torch.zeros_like(m))
    # Differences along x are 1; differences along z are 4, no row wrap.
    expected = 0.7/2*6*(9/9 + 8*16/4)
    assert reg.value(m).item() == pytest.approx(expected)
    assert torch.autograd.gradcheck(reg.value, (m,))
    assert torch.autograd.gradgradcheck(reg.value, (m,))


def test_constant_nullspace_and_linear_hessian():
    grid = Grid2D((3, 4), (2., 3.))
    prior = torch.randn(grid.shape, dtype=torch.float64)
    reg = Regularizer(grid, 0.4, prior)
    assert reg.value(prior+3).item() < 1e-29
    m = torch.randn_like(prior, requires_grad=True)
    d = torch.randn_like(prior)
    g, = torch.autograd.grad(reg.value(m), m, create_graph=True)
    h, = torch.autograd.grad((g*d).sum(), m)
    zero_prior = Regularizer(grid, 0.4, torch.zeros_like(prior))
    delta = d.requires_grad_()
    expected, = torch.autograd.grad(zero_prior.value(delta), delta)
    torch.testing.assert_close(h, expected)


def test_regularization_taylor_slopes():
    torch.manual_seed(12)
    grid = Grid2D((3, 4), (2., 3.))
    m = torch.randn(grid.shape, dtype=torch.float64, requires_grad=True)
    reg = Regularizer(grid, 0.3, torch.zeros_like(m))
    direction = torch.randn_like(m)
    value = reg.value(m)
    grad, = torch.autograd.grad(value, m)
    derivative = (grad*direction).sum()
    errors = [(reg.value(m+e*direction)-value-e*derivative).abs().item()
              for e in (0.02, 0.01, 0.005, 0.0025)]
    slopes = torch.log2(torch.tensor(errors[:-1])/torch.tensor(errors[1:]))
    assert torch.all((slopes > 1.8) & (slopes < 2.2)), slopes


@pytest.mark.parametrize('beta', [-1., float('nan'), float('inf')])
def test_invalid_beta(beta):
    with pytest.raises(ValueError):
        Regularizer(Grid2D((2, 2), (1., 1.)), beta, torch.ones(2, 2))
