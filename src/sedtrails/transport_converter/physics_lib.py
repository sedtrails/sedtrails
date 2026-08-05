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

MacDonald, N. J. (2006). PTM: Particle Tracking Model — Report 1: Model Theory, Implementation
and Example Applications. Technical Report.
"""

import numpy as np
from typing import Tuple
from enum import Enum


class SuspendedVelocityMethod(Enum):
    """Available methods for computing suspended sediment velocity.

    Attributes
    ----------
    VAN_WESTEN_2025 : str
        Main suspended velocity method from van Westen et al. (2025).
    SOULSBY_2011 : str
        Rouse-profile ratio method from Soulsby et al. (2011).
    MACDONALD_2006 : str
        Log-profile method evaluated at suspended sediment centroid height.
    """

    VAN_WESTEN_2025 = 'van_westen_2025'  # Main method from van Westen et al. (2025)
    SOULSBY_2011 = 'soulsby_2011'  # Rouse-profile ratio from Soulsby et al. (2011)
    MACDONALD_2006 = 'macdonald_2006'  # Log-profile at centroid height, MacDonald (2006)


class MixingLayerMethod(Enum):
    """Available methods for computing mixing layer thickness.

    Attributes
    ----------
    BERTIN_2008 : str
        Bertin-style mixing layer implementation.
    HARRIS_WIBERG : str
        Placeholder for a Harris and Wiberg mixing layer method.
    """

    BERTIN_2008 = 'bertin_2008'  # Bertin method (current implementation)
    HARRIS_WIBERG = 'harris_wiberg'  # Harris & Wiberg method - placeholder


def compute_shear_velocity(bed_shear_stress: np.ndarray, water_density: float) -> np.ndarray:
    """
    Compute shear velocity from bed shear stress.

    Parameters
    ----------
    bed_shear_stress : np.ndarray
        τ = Bed shear stress [N/m²]
    water_density : float
        ρ_w = Water density [kg/m³]

    Returns
    -------
    np.ndarray
        u* = Shear velocity [m/s]

    Notes
    -----
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

    Parameters
    ----------
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

    Returns
    -------
    np.ndarray
        Shields parameter [-]

    Notes
    -----
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

    Parameters
    ----------
    shields_number : np.ndarray
        θ_max = Shields parameter [-]
    critical_shields : float
        θ_cr = Critical Shields parameter [-]
    mean_shear_velocity : np.ndarray
        u*_mean = Mean shear velocity [m/s]

    Returns
    -------
    np.ndarray
        U_bed = Bed load velocity [m/s]

    Notes
    -----
    

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

    Parameters
    ----------
    transport_magnitude : np.ndarray
        Magnitude of sediment transport [kg/m/s]
    velocity_magnitude : np.ndarray
        U_layer = Velocity magnitude [m/s]
    sediment_density : float
        ρ_s = Sediment density [kg/m³]
    porosity : float
        n = Sediment porosity [-]

    Returns
    -------
    np.ndarray
        Transport layer thickness [m]

    Notes
    -----
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
    water_depth: np.ndarray = None,
    grain_diameter: float = None,
) -> np.ndarray:
    """
    Compute suspended sediment velocity.

    Parameters
    ----------
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
    water_depth : np.ndarray, optional
        h = Water depth [m]. Required for MACDONALD_2006.
    grain_diameter : float, optional
        d50 = Grain diameter [m]. Required for MACDONALD_2006 (k"_s = 2.5 · d50).

    Returns
    -------
    np.ndarray
        Suspended velocity [m/s]

    Notes
    -----
    SOULSBY_2011 (van Westen et al. 2025 implementation):
    U_sus = U_c * Rs
    where Rs = ((Rb*(1-B))/(8/7-B)) * (((8/7*Rb)^(8-7B) - 1) / ((8/7*Rb)^(7-7B) - 1))
    B = w_s / (κ * u*_max)  (Rouse parameter)
    Rb = U_bed / U_c  (bed load ratio)

    MACDONALD_2006:
    Centroid height (Eq. 27):
      z_s / h = 0.0398 × 10^(−1.08 · tanh[1.2 · ln(w_s / (κ·u*)) − 0.4])
    Advection velocity at centroid (Eq. 29):
      u_s = 2.5 · u* · ln(30 · z_s / k"_s),  where k"_s = 2.5 · d50
    Fall velocity (Eq. 28) should be supplied via settling_velocity; for very fine
    grains (D_gr < 0.672) use w_s = (ν/d) · 0.0077 · D_gr² instead of Soulsby Eq. 102.

    References:
    van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025).
    Lagrangian modelling reveals sediment pathways at evolving coasts.
    Scientific Reports, 15(1), 8793.

    MacDonald, N. J. (2006). PTM: Particle Tracking Model — Report 1: Model Theory,
    Implementation and Example Applications. Technical Report. Equations 27–29.
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

    elif method == SuspendedVelocityMethod.MACDONALD_2006:
        if water_depth is None or grain_diameter is None:
            raise ValueError("MacDonald (2006) method requires 'water_depth' and 'grain_diameter' parameters.")

        with np.errstate(divide='ignore', invalid='ignore'):
            
            # Eq. 27: normalised centroid height z_s/h
            # z_s/h = 0.0398 × 10^(−1.08 · tanh[1.2 · ln(w_s/(κ·u*)) − 0.4])
            valid = max_shear_velocity > 0
            ws_ratio = np.where(valid, settling_velocity / (von_karman_constant * max_shear_velocity), 1.0)
            z_s = water_depth * np.where(
                valid,
                0.0398 * np.power(10.0, -1.08 * np.tanh(1.2 * np.log(ws_ratio) - 0.4)),
                0.0,
            )

            # Eq. 29: log-profile advection velocity at centroid height
            # u_s = 2.5 · u* · ln(30 · z_s / k"_s),  k"_s = 2.5 · d50
            k_s = 2.5 * grain_diameter
            log_arg = np.where(z_s > 0, 30.0 * z_s / k_s, np.nan)
            u_s = 2.5 * max_shear_velocity * np.log(log_arg)
            
            # Negative values arise below the roughness sublayer (z_s < k"_s/30); clip to zero
            u_s = np.nan_to_num(np.where(u_s > 0, u_s, 0.0), nan=0.0)

        return u_s

    else:
        raise ValueError(f'Unknown suspended velocity method: {method}')


def compute_directions_from_magnitude(
    velocity_magnitude: np.ndarray, transport_x: np.ndarray, transport_y: np.ndarray, transport_magnitude: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity direction components from transport components and velocity magnitude.

    Parameters
    ----------
    velocity_magnitude : np.ndarray
        Velocity magnitude [m/s]
    transport_x : np.ndarray
        X-component of transport [kg/m/s]
    transport_y : np.ndarray
        Y-component of transport [kg/m/s]
    transport_magnitude : np.ndarray
        Magnitude of transport [kg/m/s]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (velocity_x, velocity_y) components [m/s]

    Notes
    -----
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
    bertin_coefficient: float = 0.041,
) -> np.ndarray:
    """
    Compute mixing layer thickness.

    Parameters
    ----------
    max_bed_shear_stress : np.ndarray
        τ_max = Maximum bed shear stress [N/m²]
    critical_shear_stress : float
        τ_cr = Critical shear stress [N/m²]
    method : MixingLayerMethod, optional
        Method to use for calculation
    bertin_coefficient : float, optional
        Empirical coefficient c in the Bertin (2008) formula:
        d_mix = c * sqrt(max(τ_max − τ_cr, 0)) [m / (N/m²)^0.5].
        Default 0.041 matches the original publication.

    Returns
    -------
    np.ndarray
        Mixing layer thickness [m]

    Notes
    -----
    Bertin (2008) method:
    d_mix = c * sqrt(max(τ_max - τ_cr, 0))

    References:
    van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025).
    Lagrangian modelling reveals sediment pathways at evolving coasts.
    Scientific Reports, 15(1), 8793.

    Bertin, X., Castelle, B., Anfuso, G., & Ferreira, Ó. (2008).
    Improvement of sand activation depth prediction under conditions
    of oblique wave breaking. Geo-Marine Letters, 28(2), 65-75.
    """
    if method == MixingLayerMethod.BERTIN_2008:
        return bertin_coefficient * np.sqrt(np.maximum(max_bed_shear_stress - critical_shear_stress, 0.0))
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

    Parameters
    ----------
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

    Returns
    -------
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
    Report 1: Model Theory, Implementation, and Example Applications*.
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
        term_large = np.sqrt(107.33 + 1.049 * dstar**3) - 10.36
        ws_large = (kinematic_viscosity / grain_diameter) * term_large
        ws_small = (kinematic_viscosity / grain_diameter) * (0.0077 * dstar**2)
        settling_velocity = np.where(dstar >= 0.672, ws_large, ws_small)
        
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

def calculate_skin_roughness(
    *,
    method: str = "macdonald_d90",
    d50: float | None = None,
    d65: float | None = None,
    d84: float | None = None,
    d90: float | None = None,
):
    """
    Calculate skin-friction roughness length k_s for a flat, non-rippled sand bed.

    Parameters
    ----------
    method : str
        Method used to compute k_s. Options include:

        - "macdonald_d90" : k_s = 3.0 * d90  (MacDonald et al., 2006)
        - "soulsby_d50"   : k_s = 2.5 * d50  (Soulsby, 1997, eq. 24)
        - "vanrijn_d65"   : k_s = 2.0 * d65  (Van Rijn, 1993)
        - "soulsby_d90"   : k_s = 1.1 * d90  (Soulsby & Humphery, 1990)

    d50, d65, d84, d90 : float
        Grain-size percentiles [m]. Only the percentile required by the selected
        method must be provided.

    Returns
    -------
    float
        Skin-friction roughness length k_s [m].

    Reference
    ---------
    Soulsby, R. (1997). Dynamics of marine sands: a manual for practical applications. Thomas Telford. eq 24 and below. 

    """

    method = method.lower()

    if method == "macdonald_d90":
        if d90 is None:
            raise ValueError("d90 must be provided for method 'macdonald_d90'")
        return 3.0 * d90

    elif method == "soulsby_d50":
        if d50 is None:
            raise ValueError("d50 must be provided for method 'soulsby_d50'")
        return 2.5 * d50

    elif method == "vanrijn_d65":
        if d65 is None:
            raise ValueError("d65 must be provided for method 'vanrijn_d65'")
        return 2.0 * d65

    elif method == "soulsby_d90":
        if d90 is None:
            raise ValueError("d90 must be provided for method 'soulsby_d90'")
        return 1.1 * d90

    else:
        raise ValueError(
            f"Unknown skin roughness method '{method}'. "
            "See docstring for available options."
        )



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

    theta_max, theta_cr, grain_diameter, water_depth = np.broadcast_arrays(
        np.asarray(theta_max, dtype=float),
        np.asarray(theta_cr, dtype=float),
        np.asarray(grain_diameter, dtype=float),
        np.asarray(water_depth, dtype=float),
    )
    theta_ratio = np.full(theta_max.shape, np.nan, dtype=float)
    np.divide(theta_max, theta_cr, out=theta_ratio, where=theta_cr > 0.0)

    mask = (
        (theta_ratio > 1)
        & (theta_ratio < 24)
        & (water_depth > 0)
        & np.isfinite(theta_ratio)
        & np.isfinite(water_depth)
    )

    diameter_depth_ratio = np.zeros(theta_max.shape, dtype=float)
    np.divide(grain_diameter, water_depth, out=diameter_depth_ratio, where=water_depth > 0.0)
    eta_candidate = (
        0.11
        * water_depth
        * diameter_depth_ratio**0.3
        * (1 - np.exp(-0.5 * (theta_ratio - 1)))
        * (24 - theta_ratio)
    )
    eta_b = np.where(mask, eta_candidate, 0.0)

    return float(eta_b) if eta_b.ndim == 0 else eta_b



def calculate_current_related_bed_roughness_vanrijn2007(
    flow_velocity_magnitude,
    near_bed_peak_orbital_velocity,
    grain_diameter,
    water_depth,
    relative_density,
    gravity,
):
    """
    Current-related bed roughness k_s,c following van Rijn (2007),
    including contributions from ripples, megaripples, and dunes.

    Parameters
    ----------
    flow_velocity_magnitude : array-like
        Depth-averaged current velocity [m/s].
    near_bed_peak_orbital_velocity : array-like
        Near-bed peak orbital velocity [m/s]. 
        Can be estimated from wave parameters or provided as input if available.
        near_bed_peak_orbital_velocity=pi * Hs / (Tr * sinh(2*k*water_depth)) is
        a common approximation for wave orbital velocity (see van Rijn, 2007, below equation 5d).
    grain_diameter : float or array-like
        Median grain size d50 [m].
    water_depth : float or array-like
        Water depth h [m].
    relative_density : float, optional
        Sediment relative density s = ρ_s / ρ_w (default 2.65).
    gravity : float
        Gravitational acceleration g [m/s²].   


    Returns
    -------
    ks_c : array-like
        Total current-related bed roughness [m].
    ks_r : array-like
        Ripple-related bed roughness [m].
    ks_mr : array-like
        Megaripple-related bed roughness [m].
    ks_d : array-like
        Dune-related bed roughness [m].
    """

    # --- Mobility parameter Ψ (van Rijn 2007) ---
    # Use Uwc^2 directly (avoid sqrt then square)
    Uwc2 = flow_velocity_magnitude**2 + near_bed_peak_orbital_velocity**2
    psi = Uwc2 / ((relative_density - 1.0) * gravity * grain_diameter)

    # thresholds
    d_gravel = 0.002     # [m] defined in van Rijn 2007 
    d_sand   = 0.000062  # [m] defined in van Rijn 2007 
    d_silt   = 0.000032  # [m] defined in van Rijn 2007 

    # --- Grain-size factors (depend on d50 only; broadcast happens automatically in np.where) ---
    f_cs = np.minimum(1.0, (0.25 * d_gravel / grain_diameter) ** 1.5)           # ripple coarse-sed limiter (expresses the effect of a gradually decreasing ripple roughness 
                                                                                # for very coarse sediment beds - for those sediments f_cs< 1, for finer sediments f_cs=1 )
    f_fs = np.minimum(1.0, grain_diameter / (1.5 * d_sand))                     # megaripple fine-sed limiter no megaripple roughness for fine sediments (silt, very fine sand)

    # --- Ripple roughness k_s,r (Eq. 5a–5d logic, van Rijn 2007) ---
    ks_r = np.zeros_like(psi, dtype=float)

    ks_r = np.where(psi <= 50,
                    150.0 * f_cs * grain_diameter,
                    ks_r)

    ks_r = np.where((psi > 50) & (psi <= 250),
                    (182.5 - 0.652 * psi) * f_cs * grain_diameter,
                    ks_r)

    ks_r = np.where(psi > 250,
                    20.0 * f_cs * grain_diameter,
                    ks_r)

    # fine sediment override (your choice; keeps small baseline roughness)
    ks_r = np.where(grain_diameter < d_silt,
                    20.0 * d_silt,
                    ks_r)

    # --- Megaripple roughness k_s,mr (Eq. 6a–6d, van Rijn 2007) ---
    ks_mr = np.zeros_like(psi, dtype=float)

    ks_mr = np.where(psi <= 50,
                     0.0002 * f_fs * psi * water_depth,
                     ks_mr)

    ks_mr = np.where((psi > 50) & (psi <= 550),
                     (0.011 - 0.00002 * psi) * f_fs * water_depth,
                     ks_mr)

    ks_mr = np.where((psi >= 550) & (grain_diameter >= 1.5 * d_sand),
                     0.02, # in upper regime if the sand is coarser
                     ks_mr)

    ks_mr = np.where((psi >= 550) & (grain_diameter < 1.5 * d_sand),
                     200.0 * grain_diameter, # in upper regime if the sand is very fine
                     ks_mr)

    # no megaripples for very fine sediment
    ks_mr = np.where(grain_diameter < d_silt, 0.0, ks_mr)

    # --- Dune roughness k_s,d (Eq. 7a–7d, van Rijn 2007) ---
    ks_d = np.zeros_like(psi, dtype=float)

    ks_d = np.where(psi <= 100,
                    0.0004 * f_fs * psi * water_depth,
                    ks_d)

    ks_d = np.where((psi > 100) & (psi <= 600),
                    (0.048 - 0.00008 * psi) * f_fs * water_depth,
                    ks_d)

    ks_d = np.where(psi > 600, 0.0, ks_d)

    # no dunes for very fine sediment
    ks_d = np.where(grain_diameter < d_silt, 0.0, ks_d)

    # --- Total current-related roughness (Eq. 8, van Rijn 2007) ---
    ks_c = np.sqrt(ks_r**2 + ks_mr**2 + ks_d**2)

    return ks_c, ks_r, ks_mr, ks_d


def calculate_apparent_bed_roughness_vanrijn2007(
    ks_c,
    near_bed_peak_orbital_velocity,
    flow_velocity_magnitude,
    phi_deg,
):
    """
    Apparent bed roughness k_a following van Rijn (2007),
    with angle-dependent wave–current interaction.

    Parameters
    ----------
    ks_c : array-like
        Physical current-related bed roughness k_s,c [m].
    near_bed_peak_orbital_velocity : array-like
        Near-bed peak orbital velocity U_w [m/s].
    flow_velocity_magnitude : array-like
        Depth-averaged current velocity u_c [m/s].
    phi_deg : array-like
        Angle between wave and current direction [degrees].
        Expected range: 0 <= phi_deg <= 180.

    Returns
    -------
    ka : array-like
        Apparent bed roughness k_a [m].
    ka_ksc : array-like
        Ratio k_a / k_s,c [-], capped at 10.
    gamma : array-like
        Angle-dependent interaction coefficient gamma [-].
    """

    # --------------------------------------------------------------
    # Convert angle to radians
    # --------------------------------------------------------------
    phi = np.deg2rad(np.minimum(phi_deg, 360-phi_deg))

    # --------------------------------------------------------------
    # Angle-dependent interaction coefficient gamma (van Rijn 2007)
    # --------------------------------------------------------------
    gamma = 0.8 + phi - 0.3 * phi**2

    # --------------------------------------------------------------
    # Prevent division by zero for very weak currents
    # --------------------------------------------------------------
    uc_safe = np.maximum(flow_velocity_magnitude, 1e-6)

    # --------------------------------------------------------------
    # Apparent roughness amplification factor (hard capped)
    # --------------------------------------------------------------
    ka_ksc = np.exp(gamma * near_bed_peak_orbital_velocity / uc_safe)
    ka_ksc = np.minimum(ka_ksc, 10.0)

    # --------------------------------------------------------------
    # Apparent bed roughness
    # --------------------------------------------------------------
    ka = ka_ksc * ks_c

    return ka, ka_ksc, gamma


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
