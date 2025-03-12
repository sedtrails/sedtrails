# placeholder file for the soulsby2011_p.py module
# This file is not used in the current version of the plugin

import numpy as np
import xarray as xr

class DStar:
    def __init__(self, g, rhoS, rhoW, visc_kin, d):
        """
        Class to calculate the dimensionless grain size (DStar).
        """
        self.g = g
        self.rhoS = rhoS
        self.rhoW = rhoW
        self.visc_kin = visc_kin
        self.d = d  # grain size (either dTracer or dBackground)

    def calculate(self):
        """Calculate the dimensionless grain size."""
        return ((self.g * (self.rhoS / self.rhoW - 1) / (self.visc_kin ** 2)) ** (1/3)) * self.d


class ShieldsNumber:
    def __init__(self, g, rhoS, rhoW, d):
        """
        Class to calculate the Shields number (shields).
        """
        self.g = g
        self.rhoS = rhoS
        self.rhoW = rhoW
        self.d = d

    def calculate_shields(self, tau):
        """Calculate the Shields number based on equation 74 in Soulsby (1997)."""
        return tau / (self.g * (self.rhoS - self.rhoW) * self.d)
    
class CriticalShields:
    def __init__(self, DStar):
        """
        Class to calculate the critical Shields number (shields_cr).
        """
        self.DStar = DStar

    def calculate_shields_cr(self):
        """Calculate the critical Shields number based on equation 77 in Soulsby (1997)."""
        return 0.3 / (1 + 1.2 * self.DStar) + 0.055 * (1 - np.exp(-0.020 * self.DStar))


class GrainProperties:
    def __init__(self, S):
        """
        Class to calculate the grain properties for tracer and background sediment.
        """
        self.S = S

    def calculate(self):
        # Calculate Dstar and ShieldsNumber for tracer sediment
        Dstar_t = Dstar(self.S['g'], self.S['rhoS'], self.S['rhoW'], self.S['visc_kin'], self.S['dTracer']).calculate()
        shields_t = ShieldsNumber(Dstar_t, self.S['g'], self.S['rhoS'], self.S['rhoW'], self.S['dTracer'])
        shields_cr_t = CriticalShields(Dstar_t)

        # Calculate w_s for tracer sediment
        w_s_t = (self.S['visc_kin'] / self.S['dTracer']) * (np.sqrt(10.36 ** 2 + 1.049 * (Dstar_t ** 3)) - 10.36)
      
        # Calculate A and shields_cr_A
        A = self.S['dTracer'] / self.S['dBackground']
        shields_cr_A = shields_cr_t * np.sqrt(8 / (3 * (A ** 2) + 6 * A - 1)) * ((3.2260 * A) / (4 * A - 2 * (A + 1 - np.sqrt(A ** 2 + 2 * A - 1 / 3))))

        # Return all the properties in a dictionary
        return {
            'Dstar_t': Dstar_t,
            'shields_cr_t': shields_cr_t,
            'w_s_t': w_s_t,
            'A': A,
            'shields_cr_A': shields_cr_A,
        }


def soulsby2011(shields_max_A, shields_cr_A, mu_d):
    """
    Function to calculate the probability/proportion of time a particle is moving.
    """
    P = np.zeros_like(shields_max_A)

    for ii in range(shields_max_A.shape[0]):
        for jj in range(shields_max_A.shape[1]):
            if shields_max_A[ii, jj] > shields_cr_A:
                P[ii, jj] = (1 + ((np.pi / (6 * mu_d)) / (shields_max_A[ii, jj] - shields_cr_A)) ** 4) ** (-1/4)
            else:
                P[ii, jj] = 0

    return P


def shields_parameters(tau_max, S):
    """
    Function to calculate the Shields parameters for tracer and background sediment.
    """
    # Create ShieldsNumber objects for Shields number calculation
    shields_max_t = ShieldsNumber(None, S['g'], S['rhoS'], S['rhoW'], S['dTracer'])
  
    # Calculate shields_A
    shields_max_A = shields_max_t.calculate_shields(tau)
    
    return shields_max_A


def main(S, tau_max):
    # Calculate grain properties
    grain_props = GrainProperties(S).calculate()

    # Convert grain properties and Shields numbers to xarrays
    grain_props_xr = {key: xr.DataArray(value) for key, value in grain_props.items()}

    # Calculate Shields parameters
    shields_max_A = shields_parameters(tau_max, S)

    # Convert Shields parameters to xarrays
    shields_max_A_xr = xr.DataArray(shields_max_A)

    # Calculate the probability (P) of particle movement
    P = soulsby2011(shields_max_A, grain_props['shields_cr_A'], S['soulsby2011_mu_d'])

    # Convert probability to xarray
    P_xr = xr.DataArray(P)

    # Return the results as xarrays
    return {
        'grain_props': grain_props_xr,
        'shields_max_A': shields_max_A_xr,
        'P': P_xr
    }


# Example usage:
S = {
    'g': 9.81,  # gravitational acceleration (m/s^2)
    'rhoS': 2650,  # density of sediment (kg/m^3)
    'rhoW': 1000,  # density of water (kg/m^3)
    'visc_kin': 1e-6,  # kinematic viscosity (m^2/s)
    'dTracer': 0.002,  # tracer grain size (m)
    'dBackground': 0.002,  # background grain size (m)
    'soulsby2011_mu_d': 0.5  # dynamic friction coefficient mu_d, ranging between 0.5 and 1.0, as per Fredsoe & Deigaard (1992)
}

tau_max = np.ones((5, 5))  # Example tau_max array for illustration

# Running the main function
results = main(S, tau_max)

# Access and print results
print(results['P'])
