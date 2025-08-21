"""A plugin for Soulsby et al. (2011) sediment transport physics calculations."""

import numpy as np
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter import SedtrailsData
from sedtrails.transport_converter.plugins import BasePhysicsPlugin

class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for Soulsby et al. (2011) sediment transport physics calculations.
    This plugin implements the physics calculations as described in Soulsby et al. (2011).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def add_physics(self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float]) -> None:
        """
        Add physics using Soulsby et al. (2011) approach.
        '1. Focus on individual particle tracking velocities\n'
        '2. Different approach to settling and resuspension\n'
        '3. Particle-specific rather than layer-based calculations\n'
        'See: Soulsby, R. L., et al. (2011). Lagrangian model for simulating '
        'the dispersal of sand-sized particles in coastal waters.'
        """
        print('Using Soulsby et al. (2011) to compute transport velocities and add to SedTRAILS data...')

        # === LOAD: Extract data ===

        # Extract constants
        g = self.config.g
        rho_s = self.config.row_s
        rho_w = self.config.rho_w
        kinematic_viscosity = self.config.kinematic_viscosity
        von_karman = self.config.von_karman

        # Extract shear stresses
        mean_bed_shear_stress = sedtrails_data.mean_bed_shear_stress
        if mean_bed_shear_stress is None:
            raise ValueError("Missing required shear stress values in SedtrailsData.")
        max_bed_shear_stress = sedtrails_data.max_bed_shear_stress
        if mean_bed_shear_stress is None or max_bed_shear_stress is None:
            raise ValueError("Missing required shear stress values in SedtrailsData.")

        # Extract particle properties
        grain_size = self.config.grain_size
        background_grain_size = self.config.background_grain_size
        dimensionless_grain_size = grain_properties.get('dimensionless_grain_size')
        if dimensionless_grain_size is None:
            raise ValueError("Missing required 'dimensionless_grain_size' value in grain_prperties.")
        critical_shear_stress = grain_properties.get('critical_shear_stress')
        if critical_shear_stress is None:
            raise ValueError("Missing required 'critical_shear_stress' value in grain_prperties.")
        critical_shields = grain_properties.get('critical_shields')
        if critical_shields is None:
            raise ValueError("Missing required 'critical_shields' value in grain_prperties.")
        settling_velocity = grain_properties.get('settling_velocity')
        if settling_velocity is None:
            raise ValueError("Missing required 'settling_velocity' value in grain_prperties.")
        
        # Extract flow velocities (we should be able to change these based on configuration)
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']

        # Extract Soulsby et al. (2011) empirical parameters
        soulsby_b_e = self.config.soulsby_b_e
        soulsby_theta_s = self.config.soulsby_theta_s
        soulsby_gamma_e = self.config.soulsby_gamma_e
        soulsby_mu_d    = self.config.soulsby_mu_d

        # === COMPUTE ===

        # Compute shear velocities
        mean_shear_velocity = physics_lib.compute_shear_velocity(
            mean_bed_shear_stress,
            self.config.rho_w,
        )
        max_shear_velocity = physics_lib.compute_shear_velocity(
            max_bed_shear_stress,
            self.config.rho_w,
        )

        # Compute Shields number
        shields_number = physics_lib.compute_shields(
            max_bed_shear_stress,
            self.config.g,
            self.config.rho_s,
            self.config.rho_w,
            self.config.grain_size,
        )

        # Compute bed load velocity
        bed_load_velocity = physics_lib.compute_bed_load_velocity(
            shields_number,
            critical_shields,
            mean_shear_velocity,
            )

        # Compute additional particle properties
        nd_background_grain_diameter = (g * (rho_s / rho_w - 1) / (kinematic_viscosity**2))\
            ** (1 / 3)* background_grain_size # nd = non-diamentional
        grain_size_ratio = grain_size / background_grain_size
        rouse_number = settling_velocity / (von_karman * max_shear_velocity)

        # Compute Soulsby et al. (2011) physical parameters
        background_theta_max = max_bed_shear_stress / (g * (rho_s - rho_w) * background_grain_size)
        background_theta_cr = 0.3 / (1 + 1.2 * nd_background_grain_diameter) + 0.055 *\
            (1 - np.exp(-0.020 * nd_background_grain_diameter))
        theta_max = max_bed_shear_stress / (g * (rho_s - rho_w) * grain_size)
        theta_cr = 0.3 / (1 + 1.2 * dimensionless_grain_size) + 0.055 * (1 - np.exp(-0.020 * dimensionless_grain_size))
        theta_cr_exp = (theta_cr * np.sqrt(8 / (3 * (grain_size_ratio**2) + 6 * grain_size_ratio - 1))\
                        * ((3.2260 * grain_size_ratio) / (4 * grain_size_ratio - 2 *\
                            (grain_size_ratio + 1 - np.sqrt(grain_size_ratio**2 + 2 * grain_size_ratio - 1 / 3)))))

        # Compute the transition probability b [-] (Equations 3 and 4)
        soulsby_b = np.zeros(theta_max.shape)
        for i in range(0,theta_max.shape[0]):
            for j in range(0,theta_max.shape[1]):
                if background_theta_max[i][j] > background_theta_cr:
                    soulsby_b[i][j] = soulsby_b_e * (1 - np.exp(
                        -(background_theta_max[i][j] - background_theta_cr)
                        / soulsby_theta_s))
                else:
                    soulsby_b[i][j] = 0
        
        # Compute the transition probability a [-] (Equation 5)
        soulby_a = soulsby_gamma_e * soulsby_b / (1 - soulsby_gamma_e)
    
        # Compute probability/proportion of time a particle is moving P [-] (Equation 6)
        soulsby_P = np.zeros(theta_max.shape)
        for i in range(0,theta_max.shape[0]):
            for j in range(0,theta_max.shape[1]):
                if theta_max[i][j] > theta_cr_exp:
                    soulsby_P[i][j] = (1 + ((np.pi / (6 * soulsby_mu_d))
                                     / (theta_max[i][j] - theta_cr_exp))
                                          ** 4) ** (-1/4)
                else:
                    soulsby_P[i][j] = 0

        # Compute a reduction factor which is applied on the flow velocities to obtain grain velocities

        Rb = np.zeros(bed_load_velocity.shape) # Bed load velocity reduction factor
        Rs = np.zeros(bed_load_velocity.shape) # Suspended load velocity reduction factor
        soulsby_R = np.zeros(bed_load_velocity.shape) # Final velocity reduction factor
        
        # Compute Rb (Equation 8)
        for i in range(0, theta_max.shape[0]):
            for j in range(0, theta_max.shape[1]):
                if theta_max[i][j] > theta_cr_exp:
                    print(bed_load_velocity[i][j] / flow_velocity_magnitude[i][j])
                    Rb[i][j] = bed_load_velocity[i][j] / flow_velocity_magnitude[i][j]
                    if Rb[i][j] > 1:
                        Rb[i][j] = 1  # apply velocity limiter (grain velocity cannot exceed flow velocity)
                    else:
                        Rb[i][j] = 0
                        
        # Compute Rs (Equation 11)
        for i in range(0, theta_max.shape[0]):
            for j in range(0, theta_max.shape[1]):
                if Rb[i][j] == 0:
                    Rs[i][j] = 0
                else:
                    Rs[i][j] = Rb(1 - rouse_number)/(8/7 - rouse_number) * [(8/7 * Rb)^(8-7 * rouse_number) - 1]/\
                        [(8/7 * Rb)^(7-7 * rouse_number) - 1]
                    if Rs[i][j] > 1:
                        Rs[i][j] = 1 # Apply velocity limiter (grain velocity cannot exceed flow velocity)
                    elif np.isnan(Rs[i][j]):
                        Rs[i][j] = 0

        # Compute R             
        for i in range(0, theta_max.shape[0]):
            for j in range(0, theta_max.shape[1]):
                if (rouse_number[i][j] < 2.5):  # if all material is in suspension use Rs (for suspended load)
                    soulsby_R[i][j] = Rs[i][j]
                else:  # otherwise use Rb (for bed load)
                    soulsby_R[i][j] = Rb[i][j]

        # Compute grain velocities
        grain_velocity_magnitude = np.multiply(soulsby_P,soulsby_R,flow_velocity_magnitude) # (Equation 1)
        grain_velocity_x = np.multiply((flow_velocity_x/flow_velocity_magnitude),grain_velocity_magnitude)
        grain_velocity_y = np.multiply((flow_velocity_y/flow_velocity_magnitude),grain_velocity_magnitude)

        # === STORE ===

        print('Adding physics fields to SedtrailsData...')

        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'grain_velocity', {'x': grain_velocity_x, 'y': grain_velocity_y, 'magnitude': grain_velocity_magnitude},
            'soulsby_a',
            'soulsby_b',
        )
