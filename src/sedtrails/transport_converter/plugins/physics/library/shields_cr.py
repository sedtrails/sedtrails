import numpy as np

class CriticalShields:
    def __init__(self, DStar):
        """
        Class to calculate the critical Shields number (shields_cr).
        """
        self.DStar = DStar

    def calculate_shields_cr(self):
        """Calculate the critical Shields number based on equation 77 in Soulsby (1997)."""
        return 0.3 / (1 + 1.2 * self.DStar) + 0.055 * (1 - np.exp(-0.020 * self.DStar))
