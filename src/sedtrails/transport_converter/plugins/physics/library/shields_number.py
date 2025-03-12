class ShieldsNumber:
    def __init__(self, g, rhoS, rhoW, d):
        """
        Initialize the ShieldsNumber class with the given parameters.

        Parameters:
        g (float): Acceleration due to gravity (m/s^2).
        rhoS (float): Density of the sediment particles (kg/m^3).
        rhoW (float): Density of the water (kg/m^3).
        d (float): Grain size (m).
        """
        self.g = g
        self.rhoS = rhoS
        self.rhoW = rhoW
        self.d = d

    def calculate_shields(self, tau_b):
        """
        Calculate the Shields number based on equation 74 in Soulsby (1997).

        Parameters:
        tau_b (float): Bed shear stress (N/m^2).

        Returns:
        float: The Shields number.
        """
        return tau_b / (self.g * (self.rhoS - self.rhoW) * self.d)
