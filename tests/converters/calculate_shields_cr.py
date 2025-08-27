# placeholder file for the test_calculate_shields_cr.py module

# TODO: refactor or remove this code
import pytest
import numpy as np
from your_module import CriticalShields, DStar


def test_calculate_shields_cr():
    """Test the calculation of critical Shields number."""
    # Constants for DStar calculation
    g = 9.81  # gravity in m/s²
    rhoS = 2650  # grain density in kg/m³
    rhoW = 1027  # water density in kg/m³
    visc_kin = 1.36e-6  # kinematic viscosity in m²/s
    d = 200e-6  # grain diameter in m

    # Calculate DStar
    dstar_instance = DStar(g, rhoS, rhoW, visc_kin, d)
    computed_DStar = dstar_instance.calculate()

    # Expected Shields parameter for computed DStar
    expected_theta_cr = 0.3 / (1 + 1.2 * computed_DStar) + 0.055 * (1 - np.exp(-0.020 * computed_DStar))

    shields = CriticalShields(computed_DStar)
    result = shields.calculate_shields_cr()

    assert pytest.approx(result, rel=1e-6) == expected_theta_cr, f'Failed for computed DStar={computed_DStar}'
