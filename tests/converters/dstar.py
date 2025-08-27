# TODO: refactor or remove this code

import pytest
import numpy as np
from dstar import DStar  # Import the class from dstar.py


@pytest.mark.parametrize(
    'g, rhoS, rhoW, visc_kin, d, expected',
    [
        (9.81, 2650, 1027, 1.36e-6, 200e-6, 4.06),  # Expected value based on given data
        (9.81, 2650, 1000, 1e-6, 300e-6, ((9.81 * (2650 / 1000 - 1) / (1e-6**2)) ** (1 / 3)) * 300e-6),
        (9.81, 2500, 1025, 1.5e-6, 150e-6, ((9.81 * (2500 / 1025 - 1) / (1.5e-6**2)) ** (1 / 3)) * 150e-6),
        (9.81, 2700, 1030, 1.2e-6, 100e-6, ((9.81 * (2700 / 1030 - 1) / (1.2e-6**2)) ** (1 / 3)) * 100e-6),
    ],
)
def test_dstar(g, rhoS, rhoW, visc_kin, d, expected):
    """Test the calculation of the dimensionless grain size (DStar)."""
    dstar_value = DStar(g, rhoS, rhoW, visc_kin, d).calculate()
    assert np.isclose(dstar_value, expected, rtol=1e-3), (
        f'Failed for g={g}, rhoS={rhoS}, rhoW={rhoW}, visc_kin={visc_kin}, d={d}'
    )
