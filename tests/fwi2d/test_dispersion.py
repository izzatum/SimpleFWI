"""Analytic homogeneous-symbol checks, independent of Helmholtz assembly."""
import math

import pytest

from simplefwi.fwi2d.types import Grid2D
from simplefwi.fwi2d.dispersion import assess_dispersion


def test_axis_matches_exact_second_order_symbol():
    grid = Grid2D((31, 41), (5., 10.))
    report = assess_dispersion(grid, (1500., 2500.), (5., 10.))
    q = math.pi * 10. * 10. / 1500.
    assert report.max_phase_error == pytest.approx(math.asin(q) / q - 1., rel=1e-10)
    assert report.passed
    assert report.frequencies_hz == (5., 10.)
    assert report.minimum_velocity_m_s == 1500.


def test_all_sampled_angles_satisfy_discrete_symbol_on_rectangular_grid():
    grid = Grid2D((31, 41), (6., 9.), (-30., 100.))
    report = assess_dispersion(grid, (1400., 3000.), (8.,), angle_samples=91)
    k = 2 * math.pi * 8. / 1400.
    for angle, error in zip(report.angles_rad, report.phase_errors):
        kn = k * (1 + error)
        symbol = (4 * math.sin(kn * math.sin(angle) * 6. / 2)**2 / 6.**2
                  + 4 * math.sin(kn * math.cos(angle) * 9. / 2)**2 / 9.**2)
        assert symbol == pytest.approx(k*k, rel=1e-11)


def test_lower_allowed_velocity_controls_pass_and_frequency_ceiling():
    grid = Grid2D((48, 48), (10., 10.))
    assert assess_dispersion(grid, (2000., 2500.), (15.,)).passed
    lower = assess_dispersion(grid, (1500., 2500.), (15.,))
    assert not lower.passed
    assert lower.frequency_ceiling_hz < 15.
    assert lower.points_per_wavelength_required > 12.
    at_limit = assess_dispersion(grid, (1500., 2500.), (lower.frequency_ceiling_hz,))
    assert at_limit.max_phase_error == pytest.approx(.01, abs=1e-12)
    assert at_limit.passed


def test_refining_has_second_order_phase_convergence():
    errors = [assess_dispersion(Grid2D((20, 20), (h, h)), (1500., 2000.), (10.,)).max_phase_error
              for h in (10., 5., 2.5)]
    assert 3.9 < errors[1] / errors[2] < 4.1
    assert errors[0] > errors[1] > errors[2]


def test_unresolved_wavenumber_is_explicit_failure():
    result = assess_dispersion(Grid2D((10, 10), (100., 100.)), (1500., 2000.), (100.,))
    assert not result.passed
    assert math.isinf(result.max_phase_error)


@pytest.mark.parametrize('frequencies', [(), (0.,), (-1.,), (float('nan'),), (float('inf'),)])
def test_invalid_frequencies(frequencies):
    with pytest.raises(ValueError, match='frequenc'):
        assess_dispersion(Grid2D((5, 5), (10., 10.)), (1500., 2500.), frequencies)


@pytest.mark.parametrize('bounds', [(0., 2000.), (2000., 1000.), (float('nan'), 2000.), (1500., float('inf'))])
def test_invalid_velocity_bounds(bounds):
    with pytest.raises(ValueError, match='velocity'):
        assess_dispersion(Grid2D((5, 5), (10., 10.)), bounds, (10.,))


def test_unvalidated_stencil_rejected():
    with pytest.raises(ValueError, match='stencil'):
        assess_dispersion(Grid2D((5, 5), (10., 10.)), (1500., 2500.), (10.,), stencil='fourth_order')


def test_screened_plane_wave_converges_to_refined_reference():
    # Homogeneous plane waves only: no PML/source/heterogeneous validation claim.
    import cmath
    x = tuple(float(i) for i in range(0, 501, 5))
    k = 2 * math.pi * 10. / 1500.
    fields = []
    for h in (10., 5., 2.5, .3125):
        report = assess_dispersion(Grid2D((10, 10), (h, h)), (1500., 2000.), (10.,))
        kn = k * (1 + report.phase_errors[0])
        fields.append(tuple(cmath.exp(1j * kn * position) for position in x))
    reference = fields[-1]
    errors = [math.sqrt(sum(abs(a-b)**2 for a, b in zip(field, reference))/len(x))
              for field in fields[:-1]]
    assert errors[0] > errors[1] > errors[2]
    assert 3.9 < errors[0] / errors[1] < 4.2
    assert 3.9 < errors[1] / errors[2] < 4.2
