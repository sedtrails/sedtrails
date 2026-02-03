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


def calculate_rouse_number(
    settling_velocity,
    shear_velocity,
    von_karman_constant: float = 0.4,
):
    """
    Compute the Rouse number P.

        P = w_s / (kappa * u_*)

    where:
        w_s   = settling velocity [m/s]
        u_*   = shear velocity [m/s]
        kappa = von Kármán constant (≈ 0.4)

    Parameters
    ----------
    settling_velocity : float or array-like
        Particle settling velocity [m/s].
    shear_velocity : array-like
        Shear velocity u_* [m/s]. Must be broadcastable to settling_velocity.
    von_karman_constant : float, optional
        von Kármán constant κ [-]. Default is 0.4.

    Returns
    -------
    array-like
        Rouse number P [-], same shape as shear_velocity.
        Values are NaN where u_* ≤ 0.

    """
    shear_velocity = np.asarray(shear_velocity)

    rouse_number = np.full_like(shear_velocity, np.nan, dtype=float)
    mask = shear_velocity > 0.0

    rouse_number[mask] = settling_velocity / (
        von_karman_constant * shear_velocity[mask]
    )

    return rouse_number


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
            rouse_parameter = calculate_rouse_number(settling_velocity, max_shear_velocity, von_karman_constant)

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

    Bertin, X., Castelle, B., Anfuso, G., & Ferreira, Ó. (2008). 
    Improvement of sand activation depth prediction under conditions 
    of oblique wave breaking. Geo-Marine Letters, 28(2), 65-75.
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
    grain_diameter: float, 
    gravity: float, 
    sediment_density: float, 
    water_density: float, 
    kinematic_viscosity: float,
    settling_velocity_method: str = "soulsby1997",  # or "macdonald2006"

) -> dict[str, float]:
    """
    Compute all grain-related properties.
    Settling velocity can be computed using:
    - "soulsby1997" (default)
    - "macdonald2006" (MacDonald et al., 2006, equation 28)

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

    MacDonald, N., Davies, M., Zundel, A., Howlett, J., Demirbilek, Z.,
    Gailani, J., Lackey, T., & Smith, J. (2006). *PTM: Particle Tracking Model. 
    Report 1: Model Theory, Implementation, and Example Applications*.      '
    U.S. Army Corps of Engineers. Equation 28
    """
    # Dimensionless grain size, D* (Soulsby 1997, Equation 75, p. 104)
    dstar = (gravity * (sediment_density / water_density - 1) / kinematic_viscosity**2) ** (1 / 3) * grain_diameter

    # Critical Shields number, θ_cr (Soulsby 1997, Equation 77, p. 106)
    theta_cr = 0.3 / (1 + 1.2 * dstar) + 0.055 * (1 - np.exp(-0.020 * dstar))

    # Settling velocity, w_s 
    if settling_velocity_method.lower() in ["soulsby1997", "soulsby", "soulsby_1997"]: #(default is Soulsby 1997, Equation 102, p. 134)
        settling_velocity = (kinematic_viscosity / grain_diameter) * (
            np.sqrt(10.36**2 + 1.049 * dstar**3) - 10.36
        )
    elif settling_velocity_method.lower() in ["macdonald2006", "macdonald", "ptm"]: # (MacDonald et al., 2006, Equation 28)
        settling_velocity = compute_settling_velocity_macdonald(
            dstar=dstar,
            grain_diameter=grain_diameter,
            kinematic_viscosity=kinematic_viscosity,
        )
    else:
        raise ValueError(
            f"Unknown settling_velocity_method='{settling_velocity_method}'. "
            "Use 'soulsby1997' or 'macdonald2006'."
        )
    # Critical shear stress, τ_cr
    critical_shear_stress = (sediment_density - water_density) * gravity * grain_diameter * theta_cr

    return {
        'dimensionless_grain_size': dstar,
        'critical_shields': theta_cr,
        'settling_velocity': settling_velocity,
        'critical_shear_stress': critical_shear_stress,
    }

def compute_settling_velocity_macdonald(
    dstar,
    grain_diameter,
    kinematic_viscosity,
):
    """
    Settling velocity w_s following MacDonald et al. (2006) (PTM report).

    Piecewise formulation (as commonly reported in PTM documentation):

        (w_s * d) / nu = sqrt(107.33 + 1.049 * D_*^3) - 10.36     for D_* >= 0.672
        (w_s * d) / nu = 0.0077 * D_*^2                            for D_* < 0.672

    Parameters
    ----------
    dstar : float or array-like
        Dimensionless grain size D_* [-].
    grain_diameter : float or array-like
        Grain diameter d [m].
    kinematic_viscosity : float or array-like
        Kinematic viscosity nu [m^2/s].

    Returns
    -------
    w_s : float or array-like
        Settling velocity [m/s].

    Reference
    ---------
    MacDonald, N. et al. (2006). PTM: Particle Tracking Model. Report 1: Model Theory,
    Implementation, and Example Applications. U.S. Army Corps of Engineers. Equation 28.    
    """
    nu = kinematic_viscosity
    d = grain_diameter

    term_large = np.sqrt(107.33 + 1.049 * dstar**3) - 10.36
    ws_large = (nu / d) * term_large

    ws_small = (nu / d) * (0.0077 * dstar**2)

    return np.where(dstar >= 0.672, ws_large, ws_small)

def calculate_skin_roughness(grain_diameter):
    """
    Skin-friction roughness following MacDonald et al. (2006).

    Commonly approximated as:
        k_s = 3 * d90

    Parameters
    ----------
    grain_diameter : float
        Representative grain size [m], typically d90.

    Returns
    -------
    float
        Skin-friction roughness length k_s [m].

    Reference
    ---------
    MacDonald, N., Davies, M., Zundel, A., Howlett, J., Demirbilek, Z.,
    Gailani, J., Lackey, T., & Smith, J. (2006). *PTM: Particle Tracking Model. 
    Report 1: Model Theory, Implementation, and Example Applications*. 
    U.S. Army Corps of Engineers. Equation 13
    """
    return 3.0 * grain_diameter



def calculate_equilibrium_bedform_height(theta_max, theta_cr, grain_diameter, water_depth):
    """
    Equilibrium bedform height (η_b) following MacDonald et al. (2006), Eq. 12.

    This computes the *form-scale* roughness contribution via an equilibrium
    bedform height as a function of the Shields parameter exceedance above
    threshold. The equation is defined only for a limited mobility range:

        1 < (θ_max / θ_cr) < 24

    Outside this range, η_b is set to 0 (i.e., no bedforms / equation not applicable).

    Parameters
    ----------
    theta_max : array-like
        Maximum Shields parameter θ over the period of interest (dimensionless).
        Can be a scalar, NumPy array, or xarray.DataArray.
    theta_cr : float or array-like
        Critical Shields parameter θ_cr for initiation of motion (dimensionless).
    grain_diameter : float or array-like
        Representative grain diameter d50 [m].
    water_depth : float or array-like
        Water depth h [m].

    Returns
    -------
    array-like
        Equilibrium bedform height η_b [m], same shape as theta_max.


    Reference
    ---------
    MacDonald, N., Davies, M., Zundel, A., Howlett, J., Demirbilek, Z.,
    Gailani, J., Lackey, T., & Smith, J. (2006).
    PTM: Particle Tracking Model. Report 1: Model Theory, Implementation,
    and Example Applications. U.S. Army Corps of Engineers. Eq. 12.
    """

    # initialize bedform height array
    eta_b = np.zeros_like(theta_max)

    # ratio of Shields parameter to critical Shields
    theta_ratio = theta_max / theta_cr

    # apply the valid range for Eq. 12: 1 < theta/theta_cr < 24
    mask = (theta_ratio > 1) & (theta_ratio < 24)

    eta_b[mask] = (
        0.11 * water_depth[mask]
        * (grain_diameter / water_depth[mask]) ** 0.3
        * (1 - np.exp(-0.5 * (theta_ratio[mask] - 1)))
        * (24 - theta_ratio[mask])
    )

    # form roughness is the bedform height
    return eta_b


def calculate_relative_density_ratio(sediment_density, water_density):
    """
    Compute the relative density ratio s = ρ_s / ρ_w.

    Parameters
    ----------
    sediment_density : float or array-like
        Particle (sediment) density ρ_s [kg m⁻³].
    water_density : float or array-like
        Water density ρ_w [kg m⁻³].

    Returns
    -------
    float or array-like
        Relative density ratio s (dimensionless).
    """
    return sediment_density / water_density