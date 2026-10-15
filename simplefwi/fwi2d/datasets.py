"""Load the bundled MATLAB Marmousi variants with explicit velocity units."""
from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

from .types import Grid2D, validate_velocity


def _real_geometry(values, name):
    raw = np.asarray(values)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f'{name} must contain real numeric geometry')
    return raw.astype(np.float64).reshape(-1)


def _axis(values, size, name):
    axis = _real_geometry(values, name)
    if axis.size != size or not np.isfinite(axis).all():
        raise ValueError(f'{name} coordinate count must match velocity shape and be finite')
    intervals = np.diff(axis)
    if intervals.size == 0 or intervals[0] <= 0 or not np.allclose(intervals, intervals[0], rtol=1e-10, atol=1e-10):
        raise ValueError(f'{name} coordinates must have positive uniform spacing')
    return float(intervals[0]), float(axis[0])


def load_marmousi(path: str | Path, *, velocity_unit: str) -> tuple[Grid2D, torch.Tensor]:
    """Return a physical grid and float64 SI velocity with ``[z,x]`` axes.

    ``velocity_unit`` must be ``'m/s'`` or ``'km/s'``; magnitudes are never used
    to guess units. Files with x/z vectors retain their coordinate origins.
    The spacing-only ``marmousi_50.mat`` format declares a zero origin because
    that file contains no coordinate vectors.
    """
    if velocity_unit not in ('m/s', 'km/s'):
        raise ValueError("velocity_unit must be explicitly 'm/s' or 'km/s'")
    payload = loadmat(path)
    if 'vel' not in payload:
        raise ValueError('Marmousi file must contain vel')
    raw = np.asarray(payload['vel'])
    if raw.ndim != 2 or min(raw.shape) < 2 or np.iscomplexobj(raw):
        raise ValueError('vel must be a real two-dimensional grid with at least two nodes per axis')
    if 'z' in payload and 'x' in payload:
        dz, z0 = _axis(payload['z'], raw.shape[0], 'z')
        dx, x0 = _axis(payload['x'], raw.shape[1], 'x')
    elif 'z' in payload or 'x' in payload:
        raise ValueError('both x and z coordinate vectors are required')
    elif 'dx' in payload:
        spacing = _real_geometry(payload['dx'], 'dx')
        if spacing.size != 1:
            raise ValueError('dx must be a scalar spacing in metres')
        dz = dx = float(spacing[0])
        z0 = x0 = 0.0
    else:
        raise ValueError('Marmousi file requires x/z coordinate vectors or scalar dx')
    grid = Grid2D(tuple(raw.shape), (dz, dx), (z0, x0))
    velocity = torch.tensor(np.array(raw, dtype=np.float64, copy=True), dtype=torch.float64)
    if velocity_unit == 'km/s':
        velocity = velocity * 1000.0
    return grid, validate_velocity(grid, velocity)
