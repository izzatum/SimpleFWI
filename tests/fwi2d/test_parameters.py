import pytest
import torch
from simplefwi.fwi2d.parameters import VelocityParameters


def test_representations_freeze_background_and_have_equal_physics():
    initial = torch.tensor([[1500., 1600., 1700.], [1800., 1900., 2000.]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    models = [VelocityParameters(initial, (1400., 2200.), mask, representation=r)
              for r in ('dense', 'sparse_embedding')]
    for model in models:
        torch.testing.assert_close(model.velocity(), initial, atol=1e-12, rtol=1e-15)
        model.slowness_squared().sum().backward()
        p, = model.parameters()
        assert p.ndim == 2
        assert p.grad.is_sparse == (model.representation == 'sparse_embedding')
        with torch.no_grad():
            p.add_(0.7)
        assert torch.equal(model.velocity()[~mask], initial[~mask])
        assert bool(((model.velocity() >= 1400) & (model.velocity() <= 2200)).all())
    torch.testing.assert_close(models[0].slowness_squared(), models[1].slowness_squared(), rtol=0, atol=0)


@pytest.mark.parametrize('value,bounds,mask', [
    ([[1500.]], (1500., 2200.), None),
    ([[1500.]], (0., 2200.), None),
    ([[float('nan')]], (1000., 2200.), None),
    ([1500., 1600.], (1000., 2200.), None),
    ([[1500.]], (1000., 2200.), [[False]]),
])
def test_invalid_parameters(value, bounds, mask):
    with pytest.raises(ValueError):
        VelocityParameters(value, bounds, mask)


def test_pure_dense_transform_preserves_full_second_derivative_and_state():
    model = VelocityParameters([[1600., 1700.], [1800., 1900.]], (1400., 2200.),
                                [[True, False], [True, True]])
    theta = model.raw_parameter.detach().clone().requires_grad_()
    before = model.raw_parameter.detach().clone()
    torch.testing.assert_close(model.slowness_squared_from(theta), model.slowness_squared())
    assert torch.autograd.gradcheck(model.slowness_squared_from, (theta,), fast_mode=True)
    assert torch.autograd.gradgradcheck(model.slowness_squared_from, (theta,), fast_mode=True)
    assert torch.equal(model.raw_parameter, before)


@pytest.mark.parametrize('representation', ['dense', 'sparse_embedding'])
def test_python_decimal_inputs_do_not_round_through_default_float32(representation):
    value = 1500.123456789
    model = VelocityParameters([[value]], (1400., 2200.), representation=representation)
    expected = torch.tensor([[value]], dtype=torch.float64)
    assert torch.equal(model.background, expected)
    torch.testing.assert_close(model.velocity(), expected, rtol=0, atol=5e-13)
    assert model.raw_parameter.dtype == torch.float64


@pytest.mark.parametrize('representation', ['dense', 'sparse_embedding'])
@pytest.mark.parametrize('value', [1400.+1e-7, 2200.-1e-7])
def test_python_values_strictly_inside_bounds_keep_float64_margin(value, representation):
    model = VelocityParameters([[value]], (1400., 2200.), representation=representation)
    assert torch.isfinite(model.raw_parameter).all()
    torch.testing.assert_close(model.velocity(), torch.tensor([[value]], dtype=torch.float64),
                               rtol=0, atol=5e-13)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_explicit_tensor_precision_is_preserved(dtype):
    initial = torch.tensor([[1500.125]], dtype=dtype)
    model = VelocityParameters(initial, (1400., 2200.))
    assert model.raw_parameter.dtype == dtype
    assert torch.equal(model.background, initial)


def test_numpy_input_is_directly_promoted_to_documented_float64_default():
    import numpy as np
    for dtype in (np.float32, np.float64):
        initial = np.array([[1500.123456789]], dtype=dtype)
        model = VelocityParameters(initial, (1400., 2200.))
        expected = torch.from_numpy(initial).to(torch.float64)
        assert model.raw_parameter.dtype == torch.float64
        assert torch.equal(model.background, expected)


@pytest.mark.parametrize('factory', [
    lambda: [[1500.+0j]],
    lambda: [[1500.+1j]],
    lambda: torch.tensor([[1500.+0j]], dtype=torch.complex128),
    lambda: __import__('numpy').array([[1500.+1j]], dtype='complex128'),
])
def test_complex_velocity_is_rejected_without_lossy_conversion(factory):
    with pytest.raises(ValueError, match='real'):
        VelocityParameters(factory(), (1400., 2200.))
