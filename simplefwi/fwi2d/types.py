"""Immutable SI geometry and numerical configuration contracts.

Coordinates and array dimensions are ordered ``(z, x)``. Configuration hashes
encode values, including device and precision; tensor addresses are never used.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
import numbers
from typing import Any

import torch


def _real_scalar(value, name):
    # NumPy complex scalars accept float(...) with a lossy warning. Reject
    # their representation explicitly, including a zero imaginary component.
    if ((isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real))
            or (isinstance(value, torch.Tensor) and value.is_complex())):
        raise ValueError(f'{name} must be real')
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a real scalar') from exc


def _pair(value, name, convert=float):
    try:
        if len(value) != 2:
            raise ValueError
        result = tuple(_real_scalar(v, name) if convert is float else convert(v) for v in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a real pair ordered (z, x)') from exc
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f'{name} must be finite')
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _nonnegative(value, name):
    value = _real_scalar(value, name)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _serializable(value):
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, tuple):
        return [_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    return value


class _ValueConfiguration:
    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable values suitable for saved metadata."""
        return {field.name: _serializable(getattr(self, field.name)) for field in fields(self)}

    def config_hash(self) -> str:
        """A stable SHA256 identity, independent of Python object identity."""
        payload = {'type': type(self).__name__, 'values': self.to_dict()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                        separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Grid2D(_ValueConfiguration):
    """Physical node grid, with SI spacing and explicit first-node origin."""
    shape: tuple[int, int]
    spacing_m: tuple[float, float]
    origin_m: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        shape = _pair(self.shape, 'shape', lambda n: _integer(n, 'shape', 2))
        spacing = _pair(self.spacing_m, 'spacing_m')
        if min(spacing) <= 0:
            raise ValueError('spacing_m must be positive')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'spacing_m', spacing)
        object.__setattr__(self, 'origin_m', _pair(self.origin_m, 'origin_m'))


@dataclass(frozen=True)
class Acquisition(_ValueConfiguration):
    """Source/receiver coordinates in metres and complex source amplitudes.

    A missing amplitude sequence means unit amplitude for each source. Bounds
    and free-surface exclusions depend on a grid and are checked by sampling.
    """
    sources_m: tuple[tuple[float, float], ...]
    receivers_m: tuple[tuple[float, float], ...]
    amplitudes: tuple[complex, ...] | None = None

    def __post_init__(self):
        sources = tuple(_pair(p, 'sources_m') for p in self.sources_m)
        receivers = tuple(_pair(p, 'receivers_m') for p in self.receivers_m)
        if not sources or not receivers:
            raise ValueError('sources_m and receivers_m must not be empty')
        amplitudes = ((1 + 0j,) * len(sources) if self.amplitudes is None
                      else tuple(complex(a) for a in self.amplitudes))
        if len(amplitudes) != len(sources):
            raise ValueError('amplitudes must contain one value per source')
        if not all(math.isfinite(a.real) and math.isfinite(a.imag) for a in amplitudes):
            raise ValueError('amplitudes must be finite')
        object.__setattr__(self, 'sources_m', sources)
        object.__setattr__(self, 'receivers_m', receivers)
        object.__setattr__(self, 'amplitudes', amplitudes)


@dataclass(frozen=True)
class BoundaryConfig(_ValueConfiguration):
    """Cubic PML configuration; a free surface replaces top padding."""
    top: str = 'pml'
    pml_cells: int = 20
    sigma_max: float = 150.0
    alpha: float = 0.0

    def __post_init__(self):
        if self.top not in ('pml', 'free_surface'):
            raise ValueError("top must be 'pml' or 'free_surface'")
        object.__setattr__(self, 'pml_cells', _integer(self.pml_cells, 'pml_cells'))
        for name in ('sigma_max', 'alpha'):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))


@dataclass(frozen=True)
class SolverConfig(_ValueConfiguration):
    """Declared solver, precision and memory budget; no implicit fallback."""
    method: str = 'direct'
    rtol: float = 1e-10
    atol: float = 1e-12
    maxiter: int = 200
    restart: int = 30
    device: str | torch.device = 'cpu'
    dtype: torch.dtype = torch.complex128
    memory_budget_bytes: int = 1024**3

    def __post_init__(self):
        if self.method not in ('direct', 'fgmres'):
            raise ValueError("method must be 'direct' or 'fgmres'")
        for name in ('rtol', 'atol'):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.rtol == 0 and self.atol == 0:
            raise ValueError('at least one residual tolerance must be positive')
        for name in ('maxiter', 'restart', 'memory_budget_bytes'):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError('device must explicitly select cpu or cuda') from exc
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('device must explicitly select cpu or cuda')
        if self.dtype not in (torch.complex128, torch.complex64):
            raise ValueError('dtype must be torch.complex128 or torch.complex64')
        # Canonical string retains indices, and serializes without custom hooks.
        object.__setattr__(self, 'device', str(device))


@dataclass(frozen=True)
class SolveReport(_ValueConfiguration):
    """Measured true residuals and resource observations for one solve."""
    relative_residual: float
    absolute_residual: float
    iterations: int
    convergence_reason: str
    elapsed_seconds: float
    memory_bytes: int
    converged: bool

    def __post_init__(self):
        for name in ('relative_residual', 'absolute_residual', 'elapsed_seconds'):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in ('iterations', 'memory_bytes'):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.converged, bool):
            raise ValueError('converged must be boolean')
        if not isinstance(self.convergence_reason, str) or not self.convergence_reason:
            raise ValueError('convergence_reason must be a nonempty string')


def validate_velocity(grid: Grid2D, velocity: torch.Tensor) -> torch.Tensor:
    """Validate a physical velocity in m/s without detaching its graph."""
    if not isinstance(velocity, torch.Tensor):
        raise TypeError('velocity must be a torch.Tensor')
    if tuple(velocity.shape) != grid.shape:
        raise ValueError(f'velocity shape must equal grid shape {grid.shape}')
    if velocity.is_complex() or not velocity.is_floating_point():
        raise ValueError('velocity must be a real floating tensor')
    if not bool(torch.isfinite(velocity).all()):
        raise ValueError('velocity must be finite')
    if not bool((velocity > 0).all()):
        raise ValueError('velocity must be strictly positive')
    return velocity
