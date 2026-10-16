"""Local homogeneous phase-wavenumber screening for conservative differences.

The result is a spatial screening estimate, independent of boundary reflections,
heterogeneous scattering, source discretization and linear-solve convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

from .types import Grid2D


@dataclass(frozen=True)
class DispersionReport:
    """Worst-frequency, minimum-allowed-velocity angular symbol assessment."""
    stencil: str
    frequencies_hz: tuple[float, ...]
    minimum_velocity_m_s: float
    target_phase_error: float
    angles_rad: tuple[float, ...]
    phase_errors: tuple[float, ...]
    max_phase_error: float
    frequency_ceiling_hz: float
    points_per_wavelength_required: float
    passed: bool
    scope: str = 'local_homogeneous_symbol_only'


def _validate_inputs(velocity_bounds, frequencies_hz, max_phase_error):
    try:
        bounds = tuple(float(v) for v in velocity_bounds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('velocity_bounds must be a finite positive (minimum, maximum) pair') from exc
    if (len(bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in bounds)
            or bounds[0] > bounds[1]):
        raise ValueError('velocity_bounds must be a finite positive ordered pair')
    try:
        frequencies = tuple(float(f) for f in frequencies_hz)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('frequencies_hz must contain positive finite frequencies') from exc
    if not frequencies or not all(math.isfinite(f) and f > 0 for f in frequencies):
        raise ValueError('frequencies_hz must contain positive finite frequencies')
    tolerance = float(max_phase_error)
    if not math.isfinite(tolerance) or not 0 < tolerance <= .1:
        raise ValueError('max_phase_error must be positive and <= 0.1')
    return bounds, frequencies, tolerance


def _axis_resolution(tolerance):
    # q = pi*f*h/v; k_n/k = asin(q)/q for the axial stencil symbol.
    lo, hi = 0., 1.
    for _ in range(64):
        q = (lo + hi) / 2
        if math.asin(q) / q - 1 > tolerance:
            hi = q
        else:
            lo = q
    return (lo + hi) / 2


def _angular_error(k, hz, hx, angle):
    sz, sx = math.sin(angle), math.cos(angle)
    limit = min(math.pi / (hz * abs(sz)) if abs(sz) > 1e-15 else math.inf,
                math.pi / (hx * abs(sx)) if abs(sx) > 1e-15 else math.inf)

    def symbol(kn):
        return (4 * math.sin(kn * hz * sz / 2)**2 / hz**2
                + 4 * math.sin(kn * hx * sx / 2)**2 / hx**2)

    # Only the monotone first-Brillouin-zone branch is admissible.
    if k > limit or symbol(limit) < k*k:
        return math.inf
    lo, hi = k, limit
    for _ in range(64):
        mid = (lo + hi) / 2
        if symbol(mid) < k*k:
            lo = mid
        else:
            hi = mid
    return max(0., (lo + hi) / (2 * k) - 1)


def assess_dispersion(grid: Grid2D, velocity_bounds, frequencies_hz,
                      stencil: str = 'second_order', *, max_phase_error: float = .01,
                      angle_samples: int = 721) -> DispersionReport:
    """Screen the largest frequency at the smallest *allowed* velocity.

    Angles span [0, pi/2], measured from x towards z; symmetry covers the other
    quadrants. The axial ceiling using max(dz,dx) is conservative for every
    direction on the first symbol branch, including rectangular grids. Unknown
    stencils are rejected; this API cannot register unvalidated alternatives.
    """
    if not isinstance(grid, Grid2D):
        raise TypeError('grid must be Grid2D')
    if stencil != 'second_order':
        raise ValueError('only the second_order stencil is supported')
    bounds, frequencies, tolerance = _validate_inputs(velocity_bounds, frequencies_hz, max_phase_error)
    if (isinstance(angle_samples, bool) or not isinstance(angle_samples, numbers.Integral)
            or not 2 <= angle_samples <= 10001):
        raise ValueError('angle_samples must be an integer in [2, 10001]')
    hz, hx = grid.spacing_m
    k = 2 * math.pi * max(frequencies) / bounds[0]
    angles = tuple(i * math.pi / (2 * (angle_samples - 1)) for i in range(angle_samples))
    errors = tuple(_angular_error(k, hz, hx, angle) for angle in angles)
    q = _axis_resolution(tolerance)
    ceiling = bounds[0] * q / (math.pi * max(hz, hx))
    worst = max(errors)
    passed = worst <= tolerance + 1e-12 and max(frequencies) <= ceiling * (1 + 1e-12)
    return DispersionReport(stencil, frequencies, bounds[0], tolerance, angles, errors,
                            worst, ceiling, math.pi / q, passed)
