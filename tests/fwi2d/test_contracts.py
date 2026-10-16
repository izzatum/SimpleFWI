"""SI geometry, configuration identity, and explicit dataset unit contracts."""
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import scipy.io
from simplefwi.fwi2d import (
    Acquisition, BoundaryConfig, Grid2D, SolverConfig, load_marmousi,
    validate_velocity,
)
import torch

DATA = Path(__file__).resolve().parents[2] / 'data'


def test_marmousi_units_and_origin():
    grid, velocity = load_marmousi(DATA / 'marmousi.mat', velocity_unit='m/s')
    assert grid.shape == (121, 373)
    assert grid.spacing_m == (25.0, 25.0)
    assert grid.origin_m == (0.0, -200.0)
    assert velocity.dtype == torch.float64
    assert 1460 < velocity.min().item() < 1470
    assert 5920 < velocity.max().item() < 5940
    grid50, velocity50 = load_marmousi(DATA / 'marmousi_50.mat', velocity_unit='km/s')
    assert grid50.shape == (61, 220)
    assert grid50.spacing_m == (50.0, 50.0)
    assert grid50.origin_m == (0.0, 0.0)
    assert velocity50.min().item() == pytest.approx(1500)
    assert velocity50.max().item() == pytest.approx(4700)
    with pytest.raises(TypeError):
        load_marmousi(DATA / 'marmousi.mat')
    with pytest.raises(ValueError, match='velocity_unit'):
        load_marmousi(DATA / 'marmousi.mat', velocity_unit='guess')


def test_rectangular_grid_and_immutable_value_identity():
    grid = Grid2D([3, 5], [7, 11], [-20, 50])
    assert grid.shape == (3, 5)
    assert grid.spacing_m == (7.0, 11.0)
    assert grid.origin_m == (-20.0, 50.0)
    assert grid.config_hash() == Grid2D((3, 5), (7., 11.), (-20., 50.)).config_hash()
    assert grid.config_hash() != Grid2D((3, 5), (7., 11.), (-20., 51.)).config_hash()
    with pytest.raises(FrozenInstanceError):
        grid.shape = (4, 5)
    assert validate_velocity(grid, torch.full((3, 5), 1500.)).shape == grid.shape


@pytest.mark.parametrize('kwargs', [
    {'shape': (0, 3)}, {'shape': (3.2, 3)}, {'spacing_m': (0, 1)},
    {'spacing_m': (float('nan'), 1)}, {'origin_m': (0, float('inf'))},
])
def test_invalid_grid(kwargs):
    values = dict(shape=(3, 5), spacing_m=(7, 11), origin_m=(0, 0))
    values.update(kwargs)
    with pytest.raises(ValueError):
        Grid2D(**values)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0., -1.])
def test_nonfinite_or_nonpositive_velocity_rejected(value):
    velocity = torch.full((3, 5), 1500., dtype=torch.float64)
    velocity[1, 2] = value
    with pytest.raises(ValueError, match='positive|finite'):
        validate_velocity(Grid2D((3, 5), (7, 11)), velocity)


def test_velocity_shape_and_complex_rejected():
    grid = Grid2D((3, 5), (7, 11))
    with pytest.raises(ValueError, match='shape'):
        validate_velocity(grid, torch.ones((5, 3)))
    with pytest.raises(ValueError, match='real'):
        validate_velocity(grid, torch.ones((3, 5), dtype=torch.complex128))


def test_loader_validates_coordinate_spacing_and_velocity(tmp_path):
    path = tmp_path / 'bad.mat'
    scipy.io.savemat(path, {'vel': np.ones((3, 4)), 'x': [0, 10, 21, 30], 'z': [0, 5, 10]})
    with pytest.raises(ValueError, match='uniform'):
        load_marmousi(path, velocity_unit='m/s')
    scipy.io.savemat(path, {'vel': np.zeros((3, 4)), 'dx': 10})
    with pytest.raises(ValueError, match='positive'):
        load_marmousi(path, velocity_unit='m/s')


def test_acquisition_and_configuration_values():
    acquisition = Acquisition([[0, 5], [0, 10]], [[5, 5]], [1, 2j])
    assert acquisition.sources_m == ((0., 5.), (0., 10.))
    assert acquisition.amplitudes == (1 + 0j, 2j)
    assert acquisition.config_hash() == Acquisition(((0, 5), (0, 10)), ((5, 5),), (1, 2j)).config_hash()
    assert Acquisition(((0, 5),), ((5, 5),)).amplitudes == (1 + 0j,)
    with pytest.raises(ValueError, match='amplitudes'):
        Acquisition(((0, 5),), ((5, 5),), (1, 2))
    with pytest.raises(ValueError, match='finite'):
        Acquisition(((0, float('nan')),), ((5, 5),))
    assert BoundaryConfig().top == 'pml'
    assert SolverConfig().dtype == torch.complex128
    assert SolverConfig().config_hash() == SolverConfig().config_hash()
    assert SolverConfig().config_hash() != SolverConfig(rtol=1e-9).config_hash()


@pytest.mark.parametrize('factory,kwargs', [
    (BoundaryConfig, {'top': 'reflecting'}), (BoundaryConfig, {'pml_cells': -1}),
    (BoundaryConfig, {'sigma_max': -1}), (BoundaryConfig, {'alpha': -1}),
    (SolverConfig, {'method': 'cg'}), (SolverConfig, {'rtol': -1}),
    (SolverConfig, {'atol': float('nan')}), (SolverConfig, {'maxiter': 0}),
    (SolverConfig, {'restart': 0}), (SolverConfig, {'memory_budget_bytes': 0}),
    (SolverConfig, {'dtype': torch.float32}), (SolverConfig, {'device': 'auto'}),
])
def test_invalid_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize('field', ['x', 'z', 'dx'])
@pytest.mark.parametrize('invalid_kind', ['complex', 'complex_zero_imaginary', 'text'])
def test_loader_rejects_nonreal_geometry_before_conversion(tmp_path, field, invalid_kind):
    path = tmp_path / 'nonreal_geometry.mat'
    payload = {'vel': np.full((3, 4), 1500.)}
    if field == 'dx':
        value = np.array([[10.]])
    else:
        payload.update(x=np.arange(4) * 10., z=np.arange(3) * 5.)
        value = payload[field]
    if invalid_kind == 'text':
        value = np.asarray(value).astype(str)
    else:
        value = np.asarray(value, dtype=np.complex128)
        if invalid_kind == 'complex':
            value = value + 5j
    payload[field] = value
    scipy.io.savemat(path, payload)
    with pytest.raises(ValueError, match='real'):
        load_marmousi(path, velocity_unit='m/s')


@pytest.mark.parametrize('value', [np.complex128(5 + 2j), np.complex128(5 + 0j)])
@pytest.mark.parametrize('field', ['spacing_m', 'origin_m'])
def test_grid_rejects_complex_numpy_geometry(value, field):
    kwargs = dict(shape=(3, 4), spacing_m=(5, 10))
    kwargs[field] = (value, 10)
    with pytest.raises(ValueError, match='real'):
        Grid2D(**kwargs)


@pytest.mark.parametrize('field', ['sources_m', 'receivers_m'])
def test_acquisition_rejects_complex_numpy_geometry(field):
    kwargs = dict(sources_m=((0, 5),), receivers_m=((5, 5),))
    kwargs[field] = ((np.complex128(5 + 2j), 5),)
    with pytest.raises(ValueError, match='real'):
        Acquisition(**kwargs)


@pytest.mark.parametrize('factory,field', [
    (BoundaryConfig, 'sigma_max'), (BoundaryConfig, 'alpha'),
    (SolverConfig, 'rtol'), (SolverConfig, 'atol'),
])
def test_configuration_rejects_complex_numpy_scalars(factory, field):
    with pytest.raises(ValueError, match='real'):
        factory(**{field: np.complex128(1 + 2j)})
