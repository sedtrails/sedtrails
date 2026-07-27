"""Regression tests for MacDonald plugin diagnostics."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from sedtrails.transport_converter.plugins.physics.macdonald import PhysicsPlugin


def test_lookup_invalid_input_emits_filterable_warning_once(monkeypatch, capsys):
    """Invalid lookup values should warn once without printing to stdout."""
    monkeypatch.setattr(PhysicsPlugin, "_macdonald_warned_oob", False)

    with pytest.warns(RuntimeWarning, match="Some inputs were invalid"):
        result = PhysicsPlugin.calculate_macdonald_susp_load_height(0.0, 1.0)

    assert np.isnan(result)
    assert capsys.readouterr().out == ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PhysicsPlugin.calculate_macdonald_susp_load_height(0.0, 1.0)

    assert caught == []
def test_shared_entrainment_frequency_is_zero_below_threshold_and_positive_above():
    config = SimpleNamespace(
        particle_density=2650.0,
        water_density=1027.0,
        gravity=9.81,
        grain_diameter=0.00025,
        kinematic_viscosity=1.36e-6,
        entrainment_turbulence_gamma=0.0,
    )
    plugin = PhysicsPlugin(config, tracer_methods={})

    turbulent_shear, turbulent_shields, active_layer, frequency = (
        plugin._calculate_entrainment_frequency(
            np.array([0.0, 1.0]),
            critical_shields=0.05,
            timestep=1.0,
        )
    )

    assert turbulent_shear[0] == pytest.approx(0.0)
    assert active_layer[0] == pytest.approx(0.0)
    assert turbulent_shields[0] == pytest.approx(0.0)
    assert frequency[0] == pytest.approx(0.0)
    assert turbulent_shields[1] > 0.05
    assert frequency[1] > 0.0


def _macdonald_eq27_z_over_h(rouse_number):
    """Independent reference implementation of MacDonald et al. (2006) Eq. 27.

    z_s/h = 0.0398 * 10 ** ( -1.08 * tanh[ 1.2 * ln(w_s / (kappa*u_star)) - 0.4 ] )

    (ERDC/CHL TR-06-20, "PTM: Particle Tracking Model, Report 1", p.25, Eq. 27,
    read directly off the report text). This is written independently of both
    calculate_macdonald_susp_load_height (which interpolates a precomputed
    lookup table) and resources/create_lookuptable_zs_over_h_rouse.py (which
    generates that table), so it can catch a formula regression in either.
    """
    tanh_arg = 1.2 * np.log(rouse_number) - 0.4
    return 0.0398 * (10 ** (-1.08 * np.tanh(tanh_arg)))


@pytest.mark.parametrize(
    "rouse_number",
    [0.02, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 19.9],
)
def test_lookup_table_matches_macdonald_eq27_across_rouse_range(rouse_number):
    """The packaged lookup table must reproduce MacDonald (2006) Eq. 27.

    calculate_macdonald_susp_load_height() linearly interpolates a
    precomputed NetCDF table (macdonald_z_over_h_lookup.nc) built by
    resources/create_lookuptable_zs_over_h_rouse.py. This checks that, for a
    unit water depth, the interpolated z_s/h matches a fresh closed-form
    evaluation of the report's Eq. 27 to well within interpolation error.
    A prior version of the generator script mis-parenthesized the tanh
    argument (tanh(1.2*(ln(rouse)-0.4)) instead of tanh(1.2*ln(rouse)-0.4)),
    which produced errors up to ~19% around rouse~1-2; the tolerance below is
    far tighter than that error, so it will catch a regression to that bug.
    """
    z_lookup = PhysicsPlugin.calculate_macdonald_susp_load_height(rouse_number, 1.0)
    z_reference = _macdonald_eq27_z_over_h(rouse_number)

    assert z_lookup == pytest.approx(z_reference, rel=2e-3)


def test_lookup_table_matches_macdonald_eq27_at_rouse_one():
    """Pinned regression value at Rouse = 1 (ln(1) = 0, a clean anchor point).

    z_s/h = 0.0398 * 10 ** (-1.08 * tanh(-0.4))
          = 0.0398 * 10 ** (-1.08 * (-0.379949...))
          ~= 0.102383   (MacDonald et al. 2006, Eq. 27, p.25)
    """
    z_lookup = PhysicsPlugin.calculate_macdonald_susp_load_height(1.0, 1.0)

    assert z_lookup == pytest.approx(0.102383, abs=5e-4)


def test_lookup_table_asymptote_matches_eq27_high_rouse_limit():
    """For rouse >= R_ASYM (=20), the plugin returns a fixed asymptotic value.

    As rouse -> inf, ln(rouse) -> inf, so tanh(1.2*ln(rouse) - 0.4) -> 1 and
    Eq. 27 approaches the closed form 0.0398 * 10**(-1.08). This checks both
    that the coded ASYM constant matches that closed-form limit, and that the
    plugin actually returns it (rather than a stale/extrapolated table value)
    once rouse crosses the asymptotic threshold.
    """
    asym_reference = 0.0398 * (10 ** (-1.08))

    for rouse_number in (20.0, 100.0, 1.0e5):
        z_lookup = PhysicsPlugin.calculate_macdonald_susp_load_height(rouse_number, 1.0)
        assert z_lookup == pytest.approx(asym_reference, rel=1e-6)

    # Sanity-check that the asymptote is a reasonable approximation just below
    # the cutoff: tanh(1.2*ln(19.9) - 0.4) hasn't fully saturated to 1 yet, so
    # this is a loose bound, not an exact match.
    assert _macdonald_eq27_z_over_h(19.9) == pytest.approx(asym_reference, rel=1e-2)
