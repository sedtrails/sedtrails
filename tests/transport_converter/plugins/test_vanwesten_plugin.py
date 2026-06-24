"""Tests for the van Westen physics plugin."""

import numpy as np
import pytest

from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter.physics_converter import PhysicsConfig
from sedtrails.transport_converter.plugins.physics.vanwesten import PhysicsPlugin


class _SedtrailsDataStub:
    """Minimal data object exposing fields consumed by the van Westen plugin."""

    def __init__(self):
        """Create deterministic one-timestep, two-node transport fields."""
        self.depth_avg_flow_velocity = {
            'x': np.array([[1.0, 1.5]]),
            'y': np.array([[0.0, 0.0]]),
            'magnitude': np.array([[1.0, 1.5]]),
        }
        self.mean_bed_shear_stress = np.array([[1.0, 1.2]])
        self.max_bed_shear_stress = np.array([[2.0, 3.0]])
        self.bed_load_transport = {
            'x': np.array([[0.2, 0.3]]),
            'y': np.array([[0.0, 0.0]]),
            'magnitude': np.array([[0.2, 0.3]]),
        }
        self.suspended_transport = {
            'x': np.array([[0.4, 0.5]]),
            'y': np.array([[0.0, 0.0]]),
            'magnitude': np.array([[0.4, 0.5]]),
        }
        self.water_depth = np.array([[2.0, 3.0]])

    def add_physics_field(self, name, data):
        """Store plugin outputs as attributes for assertions."""
        setattr(self, name, data)


def _vanwesten_config(**overrides):
    """Build a van Westen physics config with optional field overrides."""
    config = PhysicsConfig.from_dict(config={'tracer_method': 'vanwesten'})
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_vanwesten_plugin_uses_macdonald_suspended_velocity_method():
    """Plugin selection should pass MacDonald inputs through to physics_lib."""
    config = _vanwesten_config(suspended_velocity_method='macdonald_2006')
    sedtrails_data = _SedtrailsDataStub()
    plugin = PhysicsPlugin(config, tracer_methods={})
    grain_properties = physics_lib.compute_grain_properties(
        config.grain_diameter,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.kinematic_viscosity,
    )

    plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='no_probability')

    max_shear_velocity = physics_lib.compute_shear_velocity(
        sedtrails_data.max_bed_shear_stress,
        config.water_density,
    )
    shields_number = physics_lib.compute_shields(
        sedtrails_data.max_bed_shear_stress,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.grain_diameter,
    )
    bed_load_velocity = physics_lib.compute_bed_load_velocity(
        shields_number,
        grain_properties['critical_shields'],
        physics_lib.compute_shear_velocity(sedtrails_data.mean_bed_shear_stress, config.water_density),
    )
    expected = physics_lib.compute_suspended_velocity(
        sedtrails_data.depth_avg_flow_velocity['magnitude'],
        bed_load_velocity,
        grain_properties['settling_velocity'],
        config.von_karman_constant,
        max_shear_velocity,
        shields_number,
        grain_properties['critical_shields'],
        method=physics_lib.SuspendedVelocityMethod.MACDONALD_2006,
        water_depth=sedtrails_data.water_depth,
        grain_diameter=config.grain_diameter,
    )

    np.testing.assert_allclose(sedtrails_data.suspended_velocity['magnitude'], expected)


def test_vanwesten_plugin_rejects_unknown_suspended_velocity_method():
    """Unknown suspended velocity methods should fail before output fields are written."""
    config = _vanwesten_config(suspended_velocity_method='not_a_method')
    sedtrails_data = _SedtrailsDataStub()
    plugin = PhysicsPlugin(config, tracer_methods={})
    grain_properties = physics_lib.compute_grain_properties(
        config.grain_diameter,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.kinematic_viscosity,
    )

    with pytest.raises(ValueError, match='Unknown suspended_velocity_method'):
        plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='no_probability')


def test_vanwesten_plugin_passes_custom_bertin_coefficient(monkeypatch):
    """The plugin should pass the configured Bertin coefficient to physics_lib."""
    captured = {}

    def fake_mixing_layer(max_bed_shear_stress, critical_shear_stress, method, bertin_coefficient):
        """Capture Bertin options and return deterministic nonzero depths."""
        captured['bertin_coefficient'] = bertin_coefficient
        captured['method'] = method
        return np.full_like(max_bed_shear_stress, 1.0, dtype=float)

    monkeypatch.setattr(physics_lib, 'compute_mixing_layer_thickness', fake_mixing_layer)
    config = _vanwesten_config(bertin_coefficient=0.123)
    sedtrails_data = _SedtrailsDataStub()
    plugin = PhysicsPlugin(config, tracer_methods={})
    grain_properties = physics_lib.compute_grain_properties(
        config.grain_diameter,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.kinematic_viscosity,
    )

    plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='stochastic_transport')

    assert captured['bertin_coefficient'] == pytest.approx(0.123)
    assert captured['method'] is physics_lib.MixingLayerMethod.BERTIN_2008


def test_vanwesten_plugin_clamps_probabilities_above_one(monkeypatch):
    """Transport-layer ratios above one should be clamped for stochastic mode."""
    _patch_probability_inputs(monkeypatch, mixing_depth=0.5, layer_depth=2.0)
    config = _vanwesten_config()
    sedtrails_data = _SedtrailsDataStub()
    plugin = PhysicsPlugin(config, tracer_methods={})
    grain_properties = {
        'critical_shields': 0.1,
        'settling_velocity': 0.01,
        'critical_shear_stress': 0.1,
    }

    plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='stochastic_transport')

    np.testing.assert_array_equal(sedtrails_data.bed_load_probability, np.ones((1, 2)))
    np.testing.assert_array_equal(sedtrails_data.suspended_probability, np.ones((1, 2)))


def test_vanwesten_plugin_uses_zero_probability_for_zero_mixing_depth(monkeypatch):
    """Zero mixing depth should produce zero stochastic transport probability."""
    _patch_probability_inputs(monkeypatch, mixing_depth=0.0, layer_depth=2.0)
    config = _vanwesten_config()
    sedtrails_data = _SedtrailsDataStub()
    plugin = PhysicsPlugin(config, tracer_methods={})
    grain_properties = {
        'critical_shields': 0.1,
        'settling_velocity': 0.01,
        'critical_shear_stress': 0.1,
    }

    plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='stochastic_transport')

    np.testing.assert_array_equal(sedtrails_data.bed_load_probability, np.zeros((1, 2)))
    np.testing.assert_array_equal(sedtrails_data.suspended_probability, np.zeros((1, 2)))


def test_vanwesten_plugin_resets_no_probability_output_after_clamping(monkeypatch):
    """No-probability mode should export all-one probability fields."""
    _patch_probability_inputs(monkeypatch, mixing_depth=0.5, layer_depth=0.25)
    config = _vanwesten_config()
    sedtrails_data = _SedtrailsDataStub()
    plugin = PhysicsPlugin(config, tracer_methods={})
    grain_properties = {
        'critical_shields': 0.1,
        'settling_velocity': 0.01,
        'critical_shear_stress': 0.1,
    }

    plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='no_probability')

    np.testing.assert_array_equal(sedtrails_data.bed_load_probability, np.ones((1, 2)))
    np.testing.assert_array_equal(sedtrails_data.suspended_probability, np.ones((1, 2)))


def test_vanwesten_plugin_reduced_velocity_applies_probability_then_resets_output(monkeypatch):
    """Reduced-velocity mode should scale speeds but export all-one probabilities."""
    _patch_probability_inputs(monkeypatch, mixing_depth=2.0, layer_depth=0.5)
    config = _vanwesten_config()
    sedtrails_data = _SedtrailsDataStub()
    plugin = PhysicsPlugin(config, tracer_methods={})
    grain_properties = {
        'critical_shields': 0.1,
        'settling_velocity': 0.01,
        'critical_shear_stress': 0.1,
    }

    plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='reduced_velocity')

    np.testing.assert_array_equal(sedtrails_data.bed_load_velocity['magnitude'], np.full((1, 2), 0.25))
    np.testing.assert_array_equal(sedtrails_data.suspended_velocity['magnitude'], np.full((1, 2), 0.25))
    np.testing.assert_array_equal(sedtrails_data.bed_load_probability, np.ones((1, 2)))
    np.testing.assert_array_equal(sedtrails_data.suspended_probability, np.ones((1, 2)))


def _patch_probability_inputs(monkeypatch, mixing_depth, layer_depth):
    """Patch physics helpers so probability calculations are isolated."""
    monkeypatch.setattr(physics_lib, 'compute_shear_velocity', lambda stress, density: np.ones_like(stress))
    monkeypatch.setattr(
        physics_lib,
        'compute_shields',
        lambda stress, gravity, particle_density, water_density, grain_diameter: np.ones_like(stress),
    )
    monkeypatch.setattr(
        physics_lib,
        'compute_bed_load_velocity',
        lambda shields, critical_shields, mean_shear_velocity: np.ones_like(shields),
    )
    monkeypatch.setattr(
        physics_lib,
        'compute_suspended_velocity',
        lambda *args, **kwargs: np.ones_like(args[4]),
    )
    monkeypatch.setattr(
        physics_lib,
        'compute_transport_layer_thickness',
        lambda transport_magnitude, velocity, particle_density, porosity: np.full_like(velocity, layer_depth),
    )
    monkeypatch.setattr(
        physics_lib,
        'compute_mixing_layer_thickness',
        lambda max_bed_shear_stress, critical_shear_stress, method, bertin_coefficient: np.full_like(
            max_bed_shear_stress,
            mixing_depth,
            dtype=float,
        ),
    )
    monkeypatch.setattr(
        physics_lib,
        'compute_directions_from_magnitude',
        lambda velocity, transport_x, transport_y, transport_magnitude: (velocity.copy(), np.zeros_like(velocity)),
    )
