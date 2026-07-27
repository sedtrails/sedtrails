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