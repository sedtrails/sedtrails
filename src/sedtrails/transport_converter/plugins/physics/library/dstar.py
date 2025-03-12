import numpy as np

class DStar:
    def __init__(self, g, rhoS, rhoW, visc_kin, d):
        """
        Initialize the DStar class with the given parameters.

        Parameters:
        g (float): Acceleration due to gravity (m/s^2).
        rhoS (float): Density of the sediment particles (kg/m^3).
        rhoW (float): Density of the water (kg/m^3).
        visc_kin (float): Kinematic viscosity of the water (m^2/s).
        d (float): Grain size (m).
        """
        self.g = g
        self.rhoS = rhoS
        self.rhoW = rhoW
        self.visc_kin = visc_kin
        self.d = d  # grain size 

    def calculate(self):
        """
        Calculate the dimensionless grain size (DStar).

        Returns:
        D* (float): The dimensionless grain size.
        """
        return ((self.g * (self.rhoS / self.rhoW - 1) / (self.visc_kin ** 2)) ** (1/3)) * self.d
