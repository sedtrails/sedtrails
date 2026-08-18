"""A plugin for Soulsby et al. (2011) sediment transport physics calculations."""

import logging

import numpy as np
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter import SedtrailsData
from sedtrails.transport_converter.plugins import BasePhysicsPlugin

logger = logging.getLogger(__name__)


def _safe_velocity_direction(
    flow_velocity_x: np.ndarray,
    flow_velocity_y: np.ndarray,
    flow_velocity_magnitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flow unit vectors, using zero direction where speed is invalid."""
    valid_speed = np.isfinite(flow_velocity_magnitude) & (flow_velocity_magnitude > 0.0)
    direction_x = np.zeros_like(flow_velocity_magnitude, dtype=float)
    direction_y = np.zeros_like(flow_velocity_magnitude, dtype=float)

    np.divide(flow_velocity_x, flow_velocity_magnitude, out=direction_x, where=valid_speed)
    np.divide(flow_velocity_y, flow_velocity_magnitude, out=direction_y, where=valid_speed)

    return direction_x, direction_y


class PhysicsPlugin(BasePhysicsPlugin):  # all classes should be called the PhysicsPlugin
    """
    Plugin for Soulsby et al. (2011) sediment transport physics calculations.
    This plugin implements the physics calculations as described in Soulsby et al. (2011).
    """

    def __init__(self, config, tracer_config):
        """Initialize the Soulsby physics plugin.

        Parameters
        ----------
        config : PhysicsConfig
            Physics configuration supplied by the converter.
        tracer_config : dict
            Method-specific tracer configuration retained for interface
            compatibility.
        """
        super().__init__()
        self.config = config

    def add_physics(self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float], transport_probability_method: str) -> None:
        """
        Add physics using Soulsby et al. (2011) approach.
                '1. Focus on individual particle tracking velocities
        '
                '2. Different approach to settling and resuspension
        '
                '3. Particle-specific rather than layer-based calculations
        '
                'See: Soulsby, R. L., et al. (2011). Lagrangian model for simulating '
                'the dispersal of sand-sized particles in coastal waters.'

        Parameters
        ----------
        sedtrails_data : SedtrailsData
            SedTRAILS data object to process.
        grain_properties : dict[str, float]
            Precomputed grain-property values.
        transport_probability_method : str
            Transport probability method name.
        """
        logger.info('Using Soulsby et al. (2011) to compute transport velocities and add to SedTRAILS data')

        # === LOAD: Extract data ===

        # Extract constants
        g = self.config.gravity
        rho_s = self.config.particle_density
        rho_w = self.config.water_density
        kinematic_viscosity = self.config.kinematic_viscosity
        von_karman = self.config.von_karman_constant

        # Extract shear stresses
        mean_bed_shear_stress = sedtrails_data.mean_bed_shear_stress
        if mean_bed_shear_stress is None:
            raise ValueError('Missing required shear stress values in SedtrailsData.')
        max_bed_shear_stress = sedtrails_data.max_bed_shear_stress
        if mean_bed_shear_stress is None or max_bed_shear_stress is None:
            raise ValueError('Missing required shear stress values in SedtrailsData.')

        # Extract particle properties
        grain_size = self.config.tracer_grain_size
        background_grain_size = self.config.background_grain_size
        dimensionless_grain_size = grain_properties.get('dimensionless_grain_size')
        critical_shields = grain_properties.get('critical_shields')
        settling_velocity = grain_properties.get('settling_velocity')

        # Validate required grain properties
        if critical_shields is None:
            raise ValueError('critical_shields is required for Soulsby physics calculations but was not found in grain_properties')
        if dimensionless_grain_size is None:
            raise ValueError('dimensionless_grain_size is required for Soulsby physics calculations but was not found in grain_properties')
        if settling_velocity is None:
            raise ValueError('settling_velocity is required for Soulsby physics calculations but was not found in grain_properties')

        # Extract flow velocities (we should be able to change these based on configuration)
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']

        # Extract Soulsby et al. (2011) empirical parameters
        soulsby_b_e = self.config.soulsby_b_e
        soulsby_theta_s = self.config.soulsby_theta_s
        soulsby_gamma_e = self.config.soulsby_gamma_e
        soulsby_mu_d = self.config.soulsby_mu_d

        # === COMPUTE ===

        # Compute shear velocities
        mean_shear_velocity = physics_lib.compute_shear_velocity(
            mean_bed_shear_stress,
            rho_w,
        )
        max_shear_velocity = physics_lib.compute_shear_velocity(
            max_bed_shear_stress,
            rho_w,
        )

        # Compute Shields number
        shields_number = physics_lib.compute_shields(
            max_bed_shear_stress,
            g,
            rho_s,
            rho_w,
            grain_size,
        )

        # Compute bed load velocity
        bed_load_velocity = physics_lib.compute_bed_load_velocity(
            shields_number,
            critical_shields,
            mean_shear_velocity,
        )

        # Compute additional particle properties
        nd_background_grain_diameter = (g * (rho_s / rho_w - 1) / (kinematic_viscosity**2)) ** (
            1 / 3
        ) * background_grain_size  # nd = non-dimensional
        grain_size_ratio = grain_size / background_grain_size
        rouse_denominator = von_karman * max_shear_velocity
        rouse_number = np.full_like(max_shear_velocity, np.inf, dtype=float)
        np.divide(
            settling_velocity,
            rouse_denominator,
            out=rouse_number,
            where=np.isfinite(rouse_denominator) & (rouse_denominator > 0.0),
        )

        # Compute Soulsby et al. (2011) physical parameters
        background_theta_max = max_bed_shear_stress / (g * (rho_s - rho_w) * background_grain_size)
        background_theta_cr = 0.3 / (1 + 1.2 * nd_background_grain_diameter) + 0.055 * (
            1 - np.exp(-0.020 * nd_background_grain_diameter)
        )
        theta_max = max_bed_shear_stress / (g * (rho_s - rho_w) * grain_size)
        theta_cr = 0.3 / (1 + 1.2 * dimensionless_grain_size) + 0.055 * (1 - np.exp(-0.020 * dimensionless_grain_size))
        theta_cr_exp = (
            theta_cr
            * np.sqrt(8 / (3 * (grain_size_ratio**2) + 6 * grain_size_ratio - 1))
            * (
                (3.2260 * grain_size_ratio)
                / (
                    4 * grain_size_ratio
                    - 2 * (grain_size_ratio + 1 - np.sqrt(grain_size_ratio**2 + 2 * grain_size_ratio - 1 / 3))
                )
            )
        )

        # VECTORIZE THESE LOOPS!
        # Compute the transition probability b [-] (Equations 3 and 4)
        soulsby_b = np.zeros(theta_max.shape)
        if transport_probability_method != 'no_probability':
            for i in range(0, theta_max.shape[0]):
                for j in range(0, theta_max.shape[1]):
                    if background_theta_max[i][j] > background_theta_cr:
                        soulsby_b[i][j] = soulsby_b_e * (
                            1 - np.exp(-(background_theta_max[i][j] - background_theta_cr) / soulsby_theta_s)
                        )
                    else:
                        soulsby_b[i][j] = 0

        # Compute the transition probability a [-] (Equation 5)
        soulsby_a = soulsby_gamma_e * soulsby_b / (1 - soulsby_gamma_e)


        # VECTORIZE THESE LOOPS!
        # Compute probability/proportion of time a particle is moving P [-] (Equation 6)
        soulsby_P = np.zeros(theta_max.shape)
        for i in range(0, theta_max.shape[0]):
            for j in range(0, theta_max.shape[1]):
                if theta_max[i][j] > theta_cr_exp:
                    soulsby_P[i][j] = (1 + ((np.pi / (6 * soulsby_mu_d)) / (theta_max[i][j] - theta_cr_exp)) ** 4) ** (
                        -1 / 4
                    )
                else:
                    soulsby_P[i][j] = 0

        # Compute a reduction factor which is applied on the flow velocities to obtain grain velocities

        Rb = np.zeros(bed_load_velocity.shape)  # Bed load velocity reduction factor
        Rs = np.zeros(bed_load_velocity.shape)  # Suspended load velocity reduction factor
        soulsby_R = np.zeros(bed_load_velocity.shape)  # Final velocity reduction factor

        # VECTORIZE THESE LOOPS!
        # Compute Rb (Equation 8)
        for i in range(0, theta_max.shape[0]):
            for j in range(0, theta_max.shape[1]):
                if theta_max[i][j] > theta_cr_exp:
                    local_flow_velocity_magnitude = flow_velocity_magnitude[i][j]
                    if not np.isfinite(local_flow_velocity_magnitude) or local_flow_velocity_magnitude <= 0.0:
                        Rb[i][j] = 0
                        continue

                    Rb[i][j] = bed_load_velocity[i][j] / local_flow_velocity_magnitude
                    if Rb[i][j] > 1:
                        Rb[i][j] = 1  # apply velocity limiter (grain velocity cannot exceed flow velocity)
                    elif not np.isfinite(Rb[i][j]):
                        Rb[i][j] = 0

        # VECTORIZE THESE LOOPS!
        # Compute Rs (Equation 11)
        for i in range(0, theta_max.shape[0]):
            for j in range(0, theta_max.shape[1]):
                if Rb[i][j] == 0:
                    Rs[i][j] = 0
                else:
                    Rs[i][j] = (
                        Rb[i][j] * (1 - rouse_number[i][j])
                        / (8 / 7 - rouse_number[i][j])
                        * ((8 / 7 * Rb[i][j]) ** (8 - 7 * rouse_number[i][j]) - 1)
                        / ((8 / 7 * Rb[i][j]) ** (7 - 7 * rouse_number[i][j]) - 1)
                    )
                    if Rs[i][j] > 1:
                        Rs[i][j] = 1  # Apply velocity limiter (grain velocity cannot exceed flow velocity)
                    elif not np.isfinite(Rs[i][j]):
                        Rs[i][j] = 0

        # VECTORIZE THESE LOOPS!
        # Compute R
        for i in range(0, theta_max.shape[0]):
            for j in range(0, theta_max.shape[1]):
                if rouse_number[i][j] < 2.5:  # if all material is in suspension use Rs (for suspended load)
                    soulsby_R[i][j] = Rs[i][j]
                else:  # otherwise use Rb (for bed load)
                    soulsby_R[i][j] = Rb[i][j]

        # Compute grain velocities
        grain_velocity_magnitude = soulsby_P * soulsby_R * flow_velocity_magnitude  # (Equation 1)
        flow_direction_x, flow_direction_y = _safe_velocity_direction(
            flow_velocity_x,
            flow_velocity_y,
            flow_velocity_magnitude,
        )
        grain_velocity_x = flow_direction_x * grain_velocity_magnitude
        grain_velocity_y = flow_direction_y * grain_velocity_magnitude

        # Replace non-finite values with zero transport.
        grain_velocity_magnitude = np.nan_to_num(grain_velocity_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        grain_velocity_x = np.nan_to_num(grain_velocity_x, nan=0.0, posinf=0.0, neginf=0.0)
        grain_velocity_y = np.nan_to_num(grain_velocity_y, nan=0.0, posinf=0.0, neginf=0.0)

        # Empty fields for Soulsby (only in van westen)
        mixing_layer_thickness = np.zeros_like(grain_velocity_magnitude)

        # === STORE ===

        logger.info('Adding physics fields to SedtrailsData')

        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'grain_velocity',
            {'x': grain_velocity_x, 'y': grain_velocity_y, 'magnitude': grain_velocity_magnitude},
        )
        # Soulsby et al. (2011) parameters (scalar fields)
        sedtrails_data.add_physics_field('soulsby_a', soulsby_a)
        sedtrails_data.add_physics_field('soulsby_b', soulsby_b)
        sedtrails_data.add_physics_field('mixing_layer_thickness', mixing_layer_thickness)
