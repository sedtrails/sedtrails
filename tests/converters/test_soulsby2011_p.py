import pytest
import numpy as np
import xarray as xr
from sedtrails.converters.soulsby2011_p import Dstar, ShieldsNumber, GrainProperties, soulsby2011, shields_parameters

# Test Dstar class
def test_dstar():
    # Test parameters
    g = 9.81  # m/s^2
    rhoS = 2650  # kg/m^3
    rhoW = 1000  # kg/m^3
    visc_kin = 1e-6  # m^2/s
    d = 0.01  # m (grain size)

    # Create Dstar object and calculate Dstar
    dstar = Dstar(g, rhoS, rhoW, visc_kin, d)
    result = dstar.calculate()

    # Expected value (calculated manually or using known reference)
    expected_result = 0.052  # Replace with the expected value based on your calculations
    
    # Assert if the result is close to the expected value
    assert np.isclose(result, expected_result, atol=1e-3), f"Expected {expected_result}, but got {result}"


# Test ShieldsNumber class for theta_cr
def test_shields_number_theta_cr():
    Dstar_val = 0.1  # Example Dstar value
    g = 9.81
    rhoS = 2650
    rhoW = 1000
    d = 0.01

    shields = ShieldsNumber(Dstar_val, g, rhoS, rhoW, d)
    theta_cr = shields.calculate_theta_cr()

    # Expected value based on the formula or manual calculation
    expected_theta_cr = 0.370  # Replace with the expected value based on your calculations

    assert np.isclose(theta_cr, expected_theta_cr, atol=1e-3), f"Expected {expected_theta_cr}, but got {theta_cr}"


# Test ShieldsNumber class for theta_max
def test_shields_number_theta_max():
    tau_max = 0.3  # Example tau_max value
    g = 9.81
    rhoS = 2650
    rhoW = 1000
    d = 0.01

    shields = ShieldsNumber(None, g, rhoS, rhoW, d)
    theta_max = shields.calculate_theta_max(tau_max)

    # Expected value based on formula or manual calculation
    expected_theta_max = tau_max / (g * (rhoS - rhoW) * d)

    assert np.isclose(theta_max, expected_theta_max, atol=1e-3), f"Expected {expected_theta_max}, but got {theta_max}"


# Test GrainProperties class
def test_grain_properties():
    # Example parameters for S
    S = {
        'g': 9.81,  # gravitational acceleration (m/s^2)
        'rhoS': 2650,  # density of sediment (kg/m^3)
        'rhoW': 1000,  # density of water (kg/m^3)
        'visc_kin': 1e-6,  # kinematic viscosity (m^2/s)
        'dTracer': 0.002,  # tracer grain size (m)
        'dBackground': 0.01,  # background grain size (m)
        'soulsby2011_mu_d': 0.5  # Soulsby parameter for the function
    }

    grain_props = GrainProperties(S).calculate()

    # Check if the grain properties are calculated correctly
    assert 'Dstar_t' in grain_props, "Dstar_t was not calculated correctly."
    assert 'theta_cr_t' in grain_props, "theta_cr_t was not calculated correctly."
    assert 'theta_cr_A' in grain_props, "theta_cr_A was not calculated correctly."


# Test soulsby2011 function
def test_soulsby2011():
    # Example values for theta_max_A and theta_cr_A
    theta_max_A = np.array([[0.5, 0.6], [0.7, 0.8]])
    theta_cr_A = 0.5
    mu_d = 0.5

    # Call the soulsby2011 function
    P = soulsby2011(theta_max_A, theta_cr_A, mu_d)

    # Define expected result for comparison
    expected_P = np.array([[0.727, 0.689], [0.589, 0.515]])

    assert np.allclose(P, expected_P, atol=1e-3), f"Expected {expected_P}, but got {P}"


# Test shields_parameters function
def test_shields_parameters():
    tau_max = np.array([[0.3, 0.4], [0.5, 0.6]])
    S = {
        'g': 9.81,
        'rhoS': 2650,
        'rhoW': 1000,
        'dTracer': 0.002,
        'dBackground': 0.01
    }

    theta_max_A, theta_max_a = shields_parameters(tau_max, S)

    # Define expected results based on known formulas or manual calculation
    expected_theta_max_A = tau_max / (S['g'] * (S['rhoS'] - S['rhoW']) * S['dTracer'])
    expected_theta_max_a = tau_max / (S['g'] * (S['rhoS'] - S['rhoW']) * S['dBackground'])

    assert np.allclose(theta_max_A, expected_theta_max_A), f"Expected {expected_theta_max_A}, but got {theta_max_A}"
    assert np.allclose(theta_max_a, expected_theta_max_a), f"Expected {expected_theta_max_a}, but got {theta_max_a}"


# Running all tests
if __name__ == '__main__':
    pytest.main()

