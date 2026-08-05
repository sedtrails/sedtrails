"""Regression tests for MacDonald plugin diagnostics."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from sedtrails.particle_tracer.q3d_macdonald_motion import Q3DMacdonaldMotionMixin
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter.physics_converter import PhysicsConfig, PhysicsConverter
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


def test_lookup_interpolation_does_not_use_xarray_interp(monkeypatch):
    """Repeated lookup interpolation should operate directly on cached NumPy arrays."""
    monkeypatch.setattr(PhysicsPlugin, '_macdonald_lookup_rouse', None)
    monkeypatch.setattr(PhysicsPlugin, '_macdonald_lookup_values', None)

    def fail_interp(*args, **kwargs):
        raise AssertionError('xarray interpolation should not be used')

    monkeypatch.setattr('xarray.DataArray.interp', fail_interp)

    result = PhysicsPlugin.calculate_macdonald_susp_load_height(np.array([0.5, 1.0]), 2.0)

    assert np.all(np.isfinite(result))


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


class _MacdonaldSedtrailsDataStub:
    """Minimal data object exposing fields consumed by the MacDonald plugin."""

    def __init__(self):
        """Create deterministic one-timestep, two-node transport fields."""
        self.depth_avg_flow_velocity = {
            'x': np.array([[1.0, 1.5]]),
            'y': np.array([[0.0, 0.0]]),
            'magnitude': np.array([[1.0, 1.5]]),
        }
        self.mean_bed_shear_stress = np.array([[1.0, 1.2]])
        self.max_bed_shear_stress = np.array([[2.0, 3.0]])
        self.bed_load_transport = {'magnitude': np.array([[0.2, 0.3]])}
        self.suspended_transport = {'magnitude': np.array([[0.4, 0.5]])}
        self.water_depth = np.array([[2.0, 3.0]])
        self._physics_fields = {}

    def add_physics_field(self, name, data):
        """Mirror SedtrailsData.add_physics_field: store and expose as attribute."""
        self._physics_fields[name] = data
        setattr(self, name, data)

    def has_physics_field(self, name):
        return name in self._physics_fields


def _macdonald_config(**overrides):
    """Build a MacDonald physics config with the fields add_physics() needs."""
    config = PhysicsConfig.from_dict(config={'tracer_method': 'macdonald'})
    config.use_transport_fields = 'model-native'
    config.computationType = '2D'
    config.max_suspended_velocity_factor = None
    config.export_diagnostic_fields = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _run_macdonald_add_physics(export_diagnostic_fields):
    config = _macdonald_config(export_diagnostic_fields=export_diagnostic_fields)
    sedtrails_data = _MacdonaldSedtrailsDataStub()
    plugin = PhysicsPlugin(config, tracer_methods={})
    grain_properties = {
        'critical_shields': 0.05,
        'settling_velocity': 0.02,
        'dimensionless_grain_size': 6.0,
    }
    plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='no_probability')
    return sedtrails_data


# Fields that add_physics() must always compute and store, regardless of
# export_diagnostic_fields, because something downstream of add_physics()
# (simulation_manager.py's 2D sampling, or update_q3d_particle_position's
# required_fields check) reads them back by name.
_REQUIRED_MACDONALD_FIELDS = (
    'max_shields_number',
    'mixing_layer_thickness',
    'suspended_velocity',
    'suspended_transport_centroid_elevation',
    'total_transport_centroid_elevation',
    'rouse_number',
    'skin_roughness_height',
    'profile_roughness_height',
    'max_shear_velocity',
    'selected_shear_velocity',
    'centroid_particle_velocity',
)

# Fields that exist only for offline QC/diagnostics: nothing in the transport,
# entrainment, or deposition pipeline reads them back by name.
_DIAGNOSTIC_ONLY_MACDONALD_FIELDS = (
    'bed_load_probability',
    'suspended_probability',
    'mean_particle_probability',
    'bedload_velocity',
    'suspended_transport_centroid_elevation_over_depth',
    'total_transport_centroid_elevation_over_depth',
    'particle_advection_velocity',
    'bedform_roughness_height',
    'macdonald_total_roughness_height',
    'effective_chezy_coefficient',
    'chezy_current_shear_velocity',
    'chezy_equivalent_roughness_height',
    'shear_velocity_ratio',
    'suspended_transport_ratio',
    'bed_load_transport_ratio',
    'suspended_velocity_over_da_velocity',
    '30z_s_over_k_s_total',
    'mean_shear_velocity',
    'selected_bed_shear_stress',
    'suspended_load_velocity_lnpart',
    'particle_velocity_over_da_velocity',
    'mean_particle_velocity',
)


def test_add_physics_default_omits_diagnostic_only_fields():
    """export_diagnostic_fields defaults to False: only required fields exist."""
    sedtrails_data = _run_macdonald_add_physics(export_diagnostic_fields=False)

    for field in _REQUIRED_MACDONALD_FIELDS:
        assert sedtrails_data.has_physics_field(field), f'{field} should always be present'

    for field in _DIAGNOSTIC_ONLY_MACDONALD_FIELDS:
        assert not sedtrails_data.has_physics_field(field), f'{field} should be gated off by default'


def test_add_physics_export_diagnostics_true_includes_all_fields():
    """export_diagnostic_fields=True restores every field the old code always wrote."""
    sedtrails_data = _run_macdonald_add_physics(export_diagnostic_fields=True)

    for field in _REQUIRED_MACDONALD_FIELDS + _DIAGNOSTIC_ONLY_MACDONALD_FIELDS:
        assert sedtrails_data.has_physics_field(field), f'{field} should be present when the flag is on'


def test_add_physics_required_field_values_unaffected_by_export_diagnostics_flag():
    """The gating flag must only add/remove field registrations, never change values.

    This is the safety check for the refactor: several diagnostic-only fields
    (e.g. bedload_velocity, suspended_transport_ratio) alias local variables
    that are also required intermediates for centroid_particle_velocity and
    profile_roughness_height. Skipping their *registration* must not skip or
    otherwise disturb their *computation*, so every required field's value
    must be bit-for-bit identical whether or not the flag is set.
    """
    data_off = _run_macdonald_add_physics(export_diagnostic_fields=False)
    data_on = _run_macdonald_add_physics(export_diagnostic_fields=True)

    for field in _REQUIRED_MACDONALD_FIELDS:
        value_off = getattr(data_off, field)
        value_on = getattr(data_on, field)
        if isinstance(value_off, dict):
            for key in value_off:
                np.testing.assert_array_equal(value_off[key], value_on[key], err_msg=f'{field}[{key}] differs')
        else:
            np.testing.assert_array_equal(value_off, value_on, err_msg=f'{field} differs')


def test_soulsby_vanrijn_transport_accepts_scalar_settling_velocity():
    """Broadcast the converter's scalar settling velocity over field arrays."""
    config = _macdonald_config(use_transport_fields='SoulsbyvanRijn1997')
    converter = PhysicsConverter(config.as_dict(), {'macdonald': {}})
    settling_velocity = converter.grain_properties['settling_velocity']
    sedtrails_data = _MacdonaldSedtrailsDataStub()

    assert np.asarray(settling_velocity).ndim == 0

    PhysicsPlugin(config, tracer_methods={}).add_physics(
        sedtrails_data,
        converter.grain_properties,
        transport_probability_method='no_probability',
    )

    assert sedtrails_data.centroid_particle_velocity['magnitude'].shape == sedtrails_data.water_depth.shape
    expected_mean_shear = np.sqrt(sedtrails_data.mean_bed_shear_stress / config.water_density)
    np.testing.assert_allclose(sedtrails_data.selected_shear_velocity, expected_mean_shear)


def test_compute_dh_dt_uses_eulerian_frame_spacing():
    """Differentiate water-surface elevation using the input-frame times."""
    water_depth = np.array([[1.0], [2.0], [5.0]])
    bed_level = np.array([0.0])
    times = np.array([0.0, 10.0, 40.0])

    result = PhysicsPlugin.compute_dh_dt(water_depth, bed_level, times)

    np.testing.assert_allclose(result[:, 0], np.array([0.1, 0.1, 0.1]))


def test_compute_dh_dt_rejects_nonincreasing_input_times():
    """Reject invalid Eulerian timestamps instead of dividing by zero."""
    water_depth = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match='increase monotonically'):
        PhysicsPlugin.compute_dh_dt(water_depth, np.array([0.0]), np.array([5.0, 5.0]))


def test_time_divergence_reuses_geometry_for_all_frames():
    """Apply one set of scattered-grid derivative stencils over time."""
    grid_x, grid_y = np.meshgrid(np.arange(3.0), np.arange(3.0))
    x = grid_x.ravel()
    y = grid_y.ravel()
    base_u = 2.0 * x + y
    base_v = x + 3.0 * y
    u = np.vstack((base_u, 2.0 * base_u))
    v = np.vstack((base_v, 3.0 * base_v))

    divergence, dudx, dvdy = PhysicsPlugin.divergence_scattered_knn_time(
        x,
        y,
        u,
        v,
        k=8,
    )

    np.testing.assert_allclose(dudx[0], 2.0)
    np.testing.assert_allclose(dvdy[0], 3.0)
    np.testing.assert_allclose(divergence[0], 5.0)
    np.testing.assert_allclose(divergence[1], 13.0)


def test_q3d_divergence_is_cached_for_unchanged_input_chunk(monkeypatch):
    """Avoid recomputing full-grid divergence during every particle step."""
    plugin = PhysicsPlugin(_macdonald_config(computationType='Q3D'), tracer_methods={})
    x = np.arange(4.0)
    y = np.arange(4.0)
    u = np.ones((2, 4))
    v = np.ones((2, 4))
    calls = 0
    original = PhysicsPlugin.divergence_scattered_knn_time

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        PhysicsPlugin,
        'divergence_scattered_knn_time',
        staticmethod(counted),
    )

    first = plugin._get_q3d_divergence(x, y, u, v)
    second = plugin._get_q3d_divergence(x, y, u, v)
    plugin._get_q3d_divergence(x, y, u.copy(), v)

    assert first is second
    assert calls == 2


def test_max_shear_controls_mobility_and_mean_shear_scales_bedload_speed(monkeypatch):
    """Separate peak-stress mobility from mean-current bedload speed scaling."""
    config = _macdonald_config(export_diagnostic_fields=True)
    grain_properties = {
        'critical_shields': 0.05,
        'settling_velocity': 0.02,
        'dimensionless_grain_size': 6.0,
    }
    sedtrails_data = _MacdonaldSedtrailsDataStub()
    captured_shear_velocity = []
    original = physics_lib.compute_bed_load_velocity

    def capture_bedload_velocity(shields_number, critical_shields, shear_velocity):
        captured_shear_velocity.append(np.asarray(shear_velocity).copy())
        return original(shields_number, critical_shields, shear_velocity)

    monkeypatch.setattr(
        physics_lib,
        'compute_bed_load_velocity',
        capture_bedload_velocity,
    )
    PhysicsPlugin(config, tracer_methods={}).add_physics(
        sedtrails_data,
        grain_properties,
        transport_probability_method='no_probability',
    )

    expected_mean_shear = physics_lib.compute_shear_velocity(
        sedtrails_data.mean_bed_shear_stress,
        config.water_density,
    )
    expected_max_shields = physics_lib.compute_shields(
        sedtrails_data.max_bed_shear_stress,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.grain_diameter,
    )
    np.testing.assert_allclose(
        captured_shear_velocity[0],
        expected_mean_shear,
    )
    np.testing.assert_allclose(sedtrails_data.max_shields_number, expected_max_shields)


def test_model_native_profile_uses_chezy_shear_from_depth_averaged_speed():
    """Pair model-native profile roughness with its depth-averaged current shear."""
    config = _macdonald_config(export_diagnostic_fields=True)
    sedtrails_data = _MacdonaldSedtrailsDataStub()
    sedtrails_data.effective_chezy = np.array([[50.0, 70.0]])
    grain_properties = {
        'critical_shields': 0.05,
        'settling_velocity': 0.02,
        'dimensionless_grain_size': 6.0,
    }

    PhysicsPlugin(config, tracer_methods={}).add_physics(
        sedtrails_data,
        grain_properties,
        transport_probability_method='no_probability',
    )

    expected_shear = (
        sedtrails_data.depth_avg_flow_velocity['magnitude']
        * np.sqrt(config.gravity)
        / sedtrails_data.effective_chezy
    )
    expected_stress = config.water_density * expected_shear**2
    np.testing.assert_allclose(sedtrails_data.selected_shear_velocity, expected_shear)
    np.testing.assert_allclose(sedtrails_data.chezy_current_shear_velocity, expected_shear)
    np.testing.assert_allclose(sedtrails_data.selected_bed_shear_stress, expected_stress)


def test_chezy_log_profile_recovers_depth_averaged_speed():
    """The Chezy-equivalent log profile should average to the input speed."""
    water_depth = 4.0
    depth_averaged_speed = 1.3
    chezy = 65.0
    gravity = 9.81
    shear_velocity = depth_averaged_speed * np.sqrt(gravity) / chezy
    roughness = PhysicsPlugin.calculate_chezy_equivalent_roughness_height(
        water_depth,
        chezy,
        gravity=gravity,
    )
    z = np.linspace(0.0, water_depth, 100_001)
    velocity = PhysicsPlugin.calculate_macdonald_loglaw_velocity_at_z(
        np.full_like(z, shear_velocity),
        z,
        np.full_like(z, roughness),
    )

    integrated_average = np.trapezoid(velocity, z) / water_depth

    assert integrated_average == pytest.approx(depth_averaged_speed, rel=1e-4)


def test_vertical_diffusivity_has_depth_scaling_and_consistent_implementations():
    """Keep Q3D vertical diffusivity dimensional in grid and particle paths."""
    water_depth = np.array([1.0, 2.0])
    particle_height = 0.5 * water_depth
    flow_speed = np.ones(2)
    shear_velocity = np.zeros(2)
    kwargs = {
        'K_Ev': 0.2,
        'compute_horizontal': False,
        'E_turb_vert_min': 0.0,
    }

    _, grid_vertical = PhysicsPlugin.compute_turbulent_diffusion_coefficients(
        water_depth,
        particle_height,
        flow_speed,
        shear_velocity,
        **kwargs,
    )
    _, particle_vertical = Q3DMacdonaldMotionMixin._turbulent_diffusion_coefficients(
        water_depth,
        particle_height,
        flow_speed,
        shear_velocity,
        **kwargs,
    )

    assert grid_vertical[1] == pytest.approx(2.0 * grid_vertical[0])
    np.testing.assert_allclose(particle_vertical, grid_vertical)
