"""
Physics Library for Sediment Transport Calculations

This module implements physics-based calculations for sediment transport following
methods from van Westen et al. (2025) and alternative methods from literature.

References:
-----------
van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025).
Lagrangian modelling reveals sediment pathways at evolving coasts.
Scientific Reports, 15(1), 8793.
https://www.nature.com/articles/s41598-025-92910-z#Sec8

Soulsby, R. L., Mead, C. T., Wild, B. R., & Wood, M. J. (2011).
Lagrangian model for simulating the dispersal of sand-sized particles in coastal waters.
Journal of waterway, port, coastal, and ocean engineering, 137(3), 123-131.

Soulsby, R. (1997). Dynamics of marine sands: a manual for practical applications.
Thomas Telford.

Soulsby, R. L., & Whitehouse, R. (1997). Threshold of sediment motion in coastal environments.
Pacific Coasts and Ports '97: Proceedings of the 13th Australasian Coastal and Ocean
Engineering Conference and the 6th Australasian Port and Harbour Conference; Volume 1.

Fredsoe, J., & Deigaard, R. (1992). Mechanics of coastal sediment transport.
World Scientific.

Bertin, X., Bruneau, N., Breilh, J. F., Fortunato, A. B., & Karpytchev, M. (2012).
Importance of wave age and resonance in storm surges: The case Xynthia, Bay of Biscay.
Ocean Modelling, 42, 16-30.
"""

import numpy as np
from typing import Tuple
from enum import Enum


class SuspendedVelocityMethod(Enum):
    """Available methods for computing suspended sediment velocity."""

    VAN_WESTEN_2025 = 'van_westen_2025'  # Main method from van Westen et al. (2025)
    SOULSBY_2011 = 'soulsby_2011'  # Alternative from Soulsby et al. (2011) - placeholder


class MixingLayerMethod(Enum):
    """Available methods for computing mixing layer thickness."""

    BERTIN_2008 = 'bertin_2008'  # Bertin method (current implementation)
    HARRIS_WIBERG = 'harris_wiberg'  # Harris & Wiberg method - placeholder


def compute_shear_velocity(bed_shear_stress: np.ndarray, water_density: float) -> np.ndarray:
    """
    Compute shear velocity from bed shear stress.

    Parameters:
    -----------
    bed_shear_stress : np.ndarray
        τ = Bed shear stress [N/m²]
    water_density : float
        ρ_w = Water density [kg/m³]

    Returns:
    --------
    np.ndarray
        u* = Shear velocity [m/s]

    Notes:
    ------
    u* = sqrt(τ / ρ_w)

    Reference:
    Soulsby, R. (1997). Dynamics of marine sands: a manual for practical applications.
    Thomas Telford. Equation 32
    """
    return np.sqrt(np.abs(bed_shear_stress) / water_density)


def compute_shields(
    bed_shear_stress: np.ndarray, gravity: float, sediment_density: float, water_density: float, grain_diameter: float
) -> np.ndarray:
    """
    Compute Shields parameter (dimensionless bed shear stress).

    Parameters:
    -----------
    bed_shear_stress : np.ndarray
        τ = Bed shear stress [N/m²]
    gravity : float
        g = Gravitational acceleration [m/s²]
    sediment_density : float
        ρ_s = Sediment density [kg/m³]
    water_density : float
        ρ_w = Water density [kg/m³]
    grain_diameter : float
        d = Grain diameter [m]

    Returns:
    --------
    np.ndarray
        Shields parameter [-]

    Notes:
    ------
    θ = τ / (g(ρ_s - ρ_w)d)

    Reference:
    Soulsby, R. (1997). Dynamics of marine sands: a manual for practical applications.
    Thomas Telford. Equation 74
    """
    return np.abs(bed_shear_stress) / (gravity * (sediment_density - water_density) * grain_diameter)


def compute_bed_load_velocity(
    shields_number: np.ndarray, critical_shields: float, mean_shear_velocity: np.ndarray
) -> np.ndarray:
    """
    Compute bed load velocity using Soulsby et al. (2011) Equation 7.

    Parameters:
    -----------
    shields_number : np.ndarray
        θ_max = Shields parameter [-]
    critical_shields : float
        θ_cr = Critical Shields parameter [-]
    mean_shear_velocity : np.ndarray
        u*_mean = Mean shear velocity [m/s]

    Returns:
    --------
    np.ndarray
        U_bed = Bed load velocity [m/s]

    Notes:
    ------
    U_bed = 10 * u*_mean * (1 - 0.7 * sqrt(θ_cr / θ_max))

    Only computed where θ_max > θ_cr (critical conditions).

    References: 
    Fredsoe, J., & Deigaard, R. (1992). Mechanics of coastal sediment transport.
    World Scientific. Equation 7.51

    Soulsby, R. L., Mead, C. T., Wild, B. R., & Wood, M. J. (2011).
    Lagrangian model for simulating the dispersal of sand-sized particles in coastal waters.
    Journal of waterway, port, coastal, and ocean engineering, 137(3), 123-131. Equation 7

    van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025).
    Lagrangian modelling reveals sediment pathways at evolving coasts.
    Scientific Reports, 15(1), 8793. Equation 2
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            shields_number > critical_shields,
            10.0 * mean_shear_velocity * (1 - 0.7 * np.sqrt(critical_shields / shields_number)),
            0.0,
        )


def compute_transport_layer_thickness(
    transport_magnitude: np.ndarray, velocity_magnitude: np.ndarray, sediment_density: float, porosity: float
) -> np.ndarray:
    """
    Compute representative thickness of transport layer (bed load or suspended).

    Parameters:
    -----------
    transport_magnitude : np.ndarray
        Magnitude of sediment transport [kg/m/s]
    velocity_magnitude : np.ndarray
        U_layer = Velocity magnitude [m/s]
    sediment_density : float
        ρ_s = Sediment density [kg/m³]
    porosity : float
        n = Sediment porosity [-]

    Returns:
    --------
    np.ndarray
        Transport layer thickness [m]

    Notes:
    ------
    d_layer = Q_layer / U_layer
    where Q_layer = transport_magnitude / (ρ_s * (1 - n))

    This function works for both bed load and suspended transport layers.

    Reference:
    van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025).
    Lagrangian modelling reveals sediment pathways at evolving coasts.
    Scientific Reports, 15(1), 8793. Equation 7
    """
    transport_flux = transport_magnitude / (sediment_density * (1 - porosity))

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(velocity_magnitude > 0, transport_flux / velocity_magnitude, 0.0)


def compute_suspended_velocity(
    flow_velocity_magnitude: np.ndarray,
    bed_load_velocity: np.ndarray,
    settling_velocity: float,
    von_karman_constant: float,
    max_shear_velocity: np.ndarray,
    shields_number: np.ndarray,
    critical_shields: float,
    method: SuspendedVelocityMethod = SuspendedVelocityMethod.SOULSBY_2011,
) -> np.ndarray:
    """
    Compute suspended sediment velocity.

    Parameters:
    -----------
    flow_velocity_magnitude : np.ndarray
        U_c = Flow velocity magnitude [m/s]
    bed_load_velocity : np.ndarray
        U_bed = Bed load velocity [m/s]
    settling_velocity : float
        w_s = Particle settling velocity [m/s]
    von_karman_constant : float
        κ = von Kármán constant [-] (typically 0.4)
    max_shear_velocity : np.ndarray
        u*_max = Maximum shear velocity [m/s]
    shields_number : np.ndarray
        θ_max = Shields parameter [-]
    critical_shields : float
        θ_cr = Critical Shields parameter [-]
    method : SuspendedVelocityMethod, optional
        Method to use for calculation

    Returns:
    --------
    np.ndarray
        Suspended velocity [m/s]

    Notes:
    ------
    van Westen et al. (2025) method:
    U_sus = U_c * Rs
    where Rs = ((Rb*(1-B))/(8/7-B)) * (((8/7*Rb)^(8-7B) - 1) / ((8/7*Rb)^(7-7B) - 1))
    B = w_s / (κ * u*_max)  (Rouse parameter)
    Rb = U_bed / U_c  (bed load ratio)

    Reference:
    van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025).
    Lagrangian modelling reveals sediment pathways at evolving coasts.
    Scientific Reports, 15(1), 8793.
    """
    if method == SuspendedVelocityMethod.SOULSBY_2011:
        # Suppress warnings for this entire function
        with np.errstate(divide='ignore', invalid='ignore'):
            # Critical conditions where Shields number exceeds critical threshold
            critical_conditions = shields_number > critical_shields

            # Compute Rouse parameter internally
            rouse_parameter = settling_velocity / (von_karman_constant * max_shear_velocity)

            # Compute bed load ratio
            bed_load_ratio = bed_load_velocity / flow_velocity_magnitude

            # Compute suspended sediment ratio
            # Rs = ((Rb*(1-B_rouse))/(8/7-B_rouse)) * (((8/7*Rb)**(8-7*B_rouse) - 1) / ((8/7*Rb)**(7-7*B_rouse) - 1))
            suspended_ratio = ((bed_load_ratio * (1 - rouse_parameter)) / (8 / 7 - rouse_parameter)) * (
                ((8 / 7 * bed_load_ratio) ** (8 - 7 * rouse_parameter) - 1)
                / ((8 / 7 * bed_load_ratio) ** (7 - 7 * rouse_parameter) - 1)
            )

            # Replace nans with zeros
            suspended_ratio = np.nan_to_num(suspended_ratio)

        return np.where(critical_conditions, flow_velocity_magnitude * suspended_ratio, 0.0)

    else:
        raise ValueError(f'Unknown suspended velocity method: {method}')


def compute_directions_from_magnitude(
    velocity_magnitude: np.ndarray, transport_x: np.ndarray, transport_y: np.ndarray, transport_magnitude: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity direction components from transport components and velocity magnitude.

    Parameters:
    -----------
    velocity_magnitude : np.ndarray
        Velocity magnitude [m/s]
    transport_x : np.ndarray
        X-component of transport [kg/m/s]
    transport_y : np.ndarray
        Y-component of transport [kg/m/s]
    transport_magnitude : np.ndarray
        Magnitude of transport [kg/m/s]

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (velocity_x, velocity_y) components [m/s]

    Notes:
    ------
    U_x = U_magnitude * (transport_x / transport_magnitude)
    U_y = U_magnitude * (transport_y / transport_magnitude)

    Reference:
    van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025).
    Lagrangian modelling reveals sediment pathways at evolving coasts.
    Scientific Reports, 15(1), 8793.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity_x = np.where(transport_magnitude > 0, velocity_magnitude * transport_x / transport_magnitude, 0.0)

        velocity_y = np.where(transport_magnitude > 0, velocity_magnitude * transport_y / transport_magnitude, 0.0)

    return velocity_x, velocity_y


def compute_mixing_layer_thickness(
    max_bed_shear_stress: np.ndarray,
    critical_shear_stress: float,
    method: MixingLayerMethod = MixingLayerMethod.BERTIN_2008,
) -> np.ndarray:
    """
    Compute mixing layer thickness.

    Parameters:
    -----------
    max_bed_shear_stress : np.ndarray
        τ_max = Maximum bed shear stress [N/m²]
    critical_shear_stress : float
        τ_cr = Critical shear stress [N/m²]
    method : MixingLayerMethod, optional
        Method to use for calculation

    Returns:
    --------
    np.ndarray
        Mixing layer thickness [m]

    Notes:
    ------
    Bertin (2008) method:
    d_mix = 0.041 * sqrt(max(τ_max - τ_cr, 0))

    References:
    van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025).
    Lagrangian modelling reveals sediment pathways at evolving coasts.
    Scientific Reports, 15(1), 8793.

    Bertin, X., Bruneau, N., Breilh, J. F., Fortunato, A. B., & Karpytchev, M. (2012).
    Importance of wave age and resonance in storm surges: The case Xynthia, Bay of Biscay.
    Ocean Modelling, 42, 16-30.
    """
    if method == MixingLayerMethod.BERTIN_2008:
        return 0.041 * np.sqrt(np.maximum(max_bed_shear_stress - critical_shear_stress, 0.0))
    elif method == MixingLayerMethod.HARRIS_WIBERG:
        # Placeholder for Harris & Wiberg method
        raise NotImplementedError(
            'Harris & Wiberg method not yet implemented. Please implement based on specific application requirements.'
        )
    else:
        raise ValueError(f'Unknown mixing layer method: {method}')


# Convenience function for getting all grain properties at once
def compute_grain_properties(
    grain_diameter: float, gravity: float, sediment_density: float, water_density: float, kinematic_viscosity: float
) -> dict[str, float]:
    """
    Compute all grain-related properties.

    Parameters:
    -----------
    grain_diameter : float
        d50 = Grain diameter [m]
    gravity : float
        g = Gravitational acceleration [m/s²]
    sediment_density : float
        ρ_s = Sediment density [kg/m³]
    water_density : float
        ρ_w = Water density [kg/m³]
    kinematic_viscosity : float
        ν = Kinematic viscosity [m²/s]

    Returns:
    --------
    dict[str, float]

    Dictionary containing:
        - 'dimensionless_grain_size': Dimensionless grain size [m]
        - 'critical_shields': Critical Shields number [-]
        - 'settling_velocity': Settling velocity [m/s]
        - 'critical_shear_stress': Critical shear stress [N/m²]

    References:
    -----------
    Soulsby, R. (1997). Dynamics of marine sands: a manual for practical applications.
    Thomas Telford. Equations 75, 15

    Soulsby, R. L., & Whitehouse, R. (1997). Threshold of sediment motion in coastal environments.
    Pacific Coasts and Ports '97: Proceedings of the 13th Australasian Coastal and Ocean
    Engineering Conference and the 6th Australasian Port and Harbour Conference; Volume 1. Equation 14
    """
    # Dimensionless grain size, D* (Soulsby 1997, Equation 75)
    dstar = (gravity * (sediment_density / water_density - 1) / kinematic_viscosity**2) ** (1 / 3) * grain_diameter

    # Critical Shields number, θ_cr (Soulsby & Whitehouse 1997, Equation 14)
    theta_cr = 0.3 / (1 + 1.2 * dstar) + 0.055 * (1 - np.exp(-0.020 * dstar))

    # Settling velocity, w_s (Soulsby 1997, Equation 15)
    settling_velocity = (kinematic_viscosity / grain_diameter) * (np.sqrt(10.36**2 + 1.049 * dstar**3) - 10.36)

    # Critical shear stress, τ_cr
    critical_shear_stress = (sediment_density - water_density) * gravity * grain_diameter * theta_cr

    return {
        'dimensionless_grain_size': dstar,
        'critical_shields': theta_cr,
        'settling_velocity': settling_velocity,
        'critical_shear_stress': critical_shear_stress,
    }
