import warnings

import numpy as np

from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter.physics_converter import PhysicsConfig
from sedtrails.transport_converter.plugins.physics.soulsby import PhysicsPlugin, _safe_velocity_direction


class SedtrailsDataStub:
    def __init__(self):
        self.depth_avg_flow_velocity = {
            'x': np.array([[0.0, 1.0], [0.0, 0.0]]),
            'y': np.array([[0.0, 0.0], [0.0, 2.0]]),
            'magnitude': np.array([[0.0, 1.0], [0.0, 2.0]]),
        }
        self.mean_bed_shear_stress = np.ones((2, 2))
        self.max_bed_shear_stress = np.ones((2, 2))

    def add_physics_field(self, name, data):
        setattr(self, name, data)


def _soulsby_config():
    return PhysicsConfig.from_dict(
        {'tracer_method': 'soulsby'},
        {
            'soulsby': {
                'tracer_grain_size': 0.0002,
                'background_grain_size': 0.0002,
                'soulsby_b_e': 1.7e-7,
                'soulsby_theta_s': 0.1,
                'soulsby_gamma_e': 0.1,
                'soulsby_mu_d': 0.5,
            }
        },
    )


def test_safe_velocity_direction_uses_zero_direction_for_zero_speed():
    direction_x, direction_y = _safe_velocity_direction(
        np.array([[0.0, 3.0]]),
        np.array([[0.0, 4.0]]),
        np.array([[0.0, 5.0]]),
    )

    np.testing.assert_array_equal(direction_x, np.array([[0.0, 0.6]]))
    np.testing.assert_array_equal(direction_y, np.array([[0.0, 0.8]]))


def test_soulsby_zero_flow_magnitude_does_not_warn_or_emit_nan():
    config = _soulsby_config()
    plugin = PhysicsPlugin(config, tracer_config={})
    sedtrails_data = SedtrailsDataStub()
    grain_properties = physics_lib.compute_grain_properties(
        config.tracer_grain_size,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.kinematic_viscosity,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='no_probability')

    grain_velocity = sedtrails_data.grain_velocity
    assert np.isfinite(grain_velocity['magnitude']).all()
    assert np.isfinite(grain_velocity['x']).all()
    assert np.isfinite(grain_velocity['y']).all()
    assert grain_velocity['magnitude'][0, 0] == 0.0
    assert grain_velocity['x'][0, 0] == 0.0
    assert grain_velocity['y'][0, 0] == 0.0
