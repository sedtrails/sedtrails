"""A plugin for MacDonald et al. (2006) sediment transport physics calculations."""

from random import random
import numpy as np
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter.plugins import BasePhysicsPlugin
from sedtrails.transport_converter import SedtrailsData


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for MacDonald et al. (2006) sediment transport physics calculations.
    This plugin implements the physics calculations as described in MacDonald et al. (2006).
    
    MacDonald, N.J., Davies, M.H., Zundel, A.K., Howlett, J.D., Demirbilek, Z., 
    Gailani, J.Z., ... & Smith, J. (2006). PTM: particle tracking model. 
    Report 1: Model theory, implementation, and example applications (No. ERDCCHLTR0620).
    """

    def __init__(self, config, tracer_methods: None):
        super().__init__()
        self.config = config

    def add_physics(
        self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float], transport_probability_method: str
    ) -> None:
        """
        Add physics to SedtrailsData object using MacDonald et al. (2006) approach.

        This method follows the workflow from MacDonald et al. (2006):
        2D mode:
            1. Compute shear velocities and Shields number
            2. 
            3. 

        Parameters:
        -----------
        sedtrails_data : SedtrailsData
            The SedTRAILS data object containing transport data.
        grain_properties : dict[str, float]
            Dictionary containing grain properties such as 'critical_shields' and 'settling_velocity'.

        """
        # flow velocity
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']
        
        # bed shear stress
        mean_bed_shear_stress = sedtrails_data.mean_bed_shear_stress
        max_bed_shear_stress = sedtrails_data.max_bed_shear_stress

        # bed load transport
        bed_load_transport_x = sedtrails_data.bed_load_transport['x']
        bed_load_transport_y = sedtrails_data.bed_load_transport['y']
        bed_load_transport_magnitude = sedtrails_data.bed_load_transport['magnitude']

        # suspended transport
        suspended_transport_x = sedtrails_data.suspended_transport['x']
        suspended_transport_y = sedtrails_data.suspended_transport['y']
        suspended_transport_magnitude = sedtrails_data.suspended_transport['magnitude']
        
        # water depth
        water_depth = sedtrails_data.water_depth

        # Detect number of fractions from data shape (assuming shape is [time, fractions, spatial])
        detected_fractions = bed_load_transport_magnitude.shape[1] if len(bed_load_transport_magnitude.shape) > 2 else 1

        # Error if more than 1 fraction
        if detected_fractions > 1:
            raise NotImplementedError(
                f'Multiple sediment fractions ({detected_fractions}) are not yet supported. '
                f'The physics converter currently only handles single-fraction sediment transport. '
                f'Please aggregate fractions or implement multi-fraction physics calculations.'
            )

        # For physics calculations, we need to work with squeezed data (no fraction dimension)
        # but we'll add the dimension back to velocities at the end
        if len(bed_load_transport_magnitude.shape) > 2:
            bed_load_transport_x_calc = bed_load_transport_x.squeeze(axis=1)
            bed_load_transport_y_calc = bed_load_transport_y.squeeze(axis=1)
            bed_load_transport_magnitude_calc = bed_load_transport_magnitude.squeeze(axis=1)

            suspended_transport_x_calc = suspended_transport_x.squeeze(axis=1)
            suspended_transport_y_calc = suspended_transport_y.squeeze(axis=1)
            suspended_transport_magnitude_calc = suspended_transport_magnitude.squeeze(axis=1)
            has_fraction_dim = True
        else:
            # Data already doesn't have fraction dimension
            bed_load_transport_x_calc = bed_load_transport_x
            bed_load_transport_y_calc = bed_load_transport_y
            bed_load_transport_magnitude_calc = bed_load_transport_magnitude

            suspended_transport_x_calc = suspended_transport_x
            suspended_transport_y_calc = suspended_transport_y
            suspended_transport_magnitude_calc = suspended_transport_magnitude
            has_fraction_dim = False
            
        # Compute shear velocities
        mean_shear_velocity = physics_lib.compute_shear_velocity(mean_bed_shear_stress, self.config.water_density)
        max_shear_velocity = physics_lib.compute_shear_velocity(max_bed_shear_stress, self.config.water_density)

        # Compute Shields number
        mean_shields_number = physics_lib.compute_shields(
            mean_bed_shear_stress,
            self.config.gravity,
            self.config.particle_density,
            self.config.water_density,
            self.config.grain_diameter,
        )
        max_shields_number = physics_lib.compute_shields(
            max_bed_shear_stress,
            self.config.gravity,
            self.config.particle_density,
            self.config.water_density,
            self.config.grain_diameter,
        )

        # Compute transport velocities (these will have shape [time, spatial])
        critical_shields = grain_properties.get('critical_shields')
        if critical_shields is None:
            raise ValueError("Missing required 'critical_shields' value in grain_properties.")
        
        # Bed roughness (MacDonald et al., 2006, equations 12-13)
        # Questions Vassia: here we should be passing the background sediment grain size right (not sure if config.grain_diameter is correct)? That could also in the future be spatially varying
        #                   also, should we be passing the maximum shields number or the mean shields number?
        k_s_skin = self.calculate_macdonald_skin_roughness(self.config.grain_diameter)
        k_s_form = self.calculate_macdonald_form_roughness(max_shields_number, critical_shields, self.config.grain_diameter, water_depth)
        # TO DO: Note that k_s_form is the equilibrium bedform height eta_b from MacDonald et al. (2006) Eq. 12 - we should implement the rate of change also (eq. 14-15) in future

        # Potential transport rates (MacDonald et al., 2006, equations 16-20)
        s = self.config.particle_density / self.config.water_density   # relative density ratio
        dstar = grain_properties.get('dimensionless_grain_size')        

        # question: is it worth coding up the Soulsby-vanRijn transport equations that they have hear so that 
        # we can go directly from a hydrodynamic-only model, in the same way that soulsby is, 
        # or do we remain dependent on having a sediment transport model output?
        
        # soulsby-vanRijn factors (MacDonald et al., 2006, equations 16-18)
        A_s, C_d, U_cr, q_t_soulsbyVanRijn = self.calculate_soulsby_vanrijn_potential_transport(water_depth=water_depth,
                                                                                    flow_velocity_magnitude=flow_velocity_magnitude,
                                                                                    U_rms=sedtrails_data.nonlinear_wave_velocity['magnitude'],
                                                                                    dstar=dstar,
                                                                                    s=s,
                                                                                    k_s_skin=k_s_skin,
                                                                                    k_s_form=k_s_form,
                                                                                    )
        # settling velocity - very similar to physics_lib.compute_settling_velocity, differs only for dstar < 0.672 (very fine grains)
        settling_velocity = self.compute_settling_velocity_macdonald(dstar) # gives very similar results to physics_lib.compute_settling_velocity

        rouse_number = self.compute_rouse_number(
            settling_velocity=settling_velocity,
            mean_shear_velocity=mean_shear_velocity
        )

        # suspended load height (MacDonald et al., 2006, equation 27)
        z_s = self.calculate_macdonald_susp_load_height(rouse_number, water_depth)
        
        # suspended load velocity (MacDonald et al., 2006, equation 29) 
        # Note Vassia: here we sum skin and form roughness for total roughness - eq. 29 says k_s'' indicating bedform roughness
        # Question Vassia: should we be using max or mean shear velocity here?
        suspended_velocity = self.calculate_macdonald_suspended_load_velocity(mean_shear_velocity, z_s, k_s_form+k_s_skin)
        
        # bed load velocity (MacDonald et al., 2006, equation 30) - Engelund & Fredsoe (1976), same as Soulsby et al (2011)
        bed_load_velocity = physics_lib.compute_bed_load_velocity(
            max_shields_number, critical_shields, mean_shear_velocity)

        # suspended load transport fraction 
        qs_qt = suspended_transport_magnitude_calc / (suspended_transport_magnitude_calc + bed_load_transport_magnitude_calc)
        # implement van Rijn (1984b) suspended sediment as per macdonald et al. (2006) equation 31 later
        
        # sediment advection velocity (MacDonald et al., 2006, equation 32)
        u_c = qs_qt * suspended_velocity + (1 - qs_qt) * bed_load_velocity
        
        # total transport centroid elevation (MacDonald et al., 2006, equation 34)
        # Question Vassia: should we be using mean or max shear velocity here?
        z_c = k_s_form * 10 ** (0.1739 * (u_c / max_shear_velocity) - 1.47826)
        
        # horizontal mean particle advection velocity (u_a) (MacDonald et al., 2006, equation 35)
        # Question Vassia: should we be using mean or max shear velocity here?
        # Note Vassia: here we sum skin and form roughness for total roughness - eq. 35 says k_s'' indicating bedform roughness
        mean_particle_velocity= np.zeros_like(max_shear_velocity) * np.nan
        mask = (z_c / (k_s_skin + k_s_form)) > 1 
        mean_particle_velocity[mask] = max_shear_velocity[mask]  * (5.75 * np.log10(z_c[mask]  / (k_s_skin+k_s_form[mask])) + 8.5)
        
        # calculate x and y components of mean particle velocity
        mean_particle_velocity_x, mean_particle_velocity_y = physics_lib.compute_directions_from_magnitude(
            mean_particle_velocity,
            flow_velocity_x,
            flow_velocity_y,
            flow_velocity_magnitude,
        )
        
        # ----------------------------------------------------------------------------------------
        # MacDonald probabilistic bed-particle interaction model
        
        # particle entrainment rate (van Rijn (1984b) pickup function, as used in MacDonald et al., 2006, equation 58)
        q_pickup = (0.00033 * ((max_shields_number - critical_shields) / critical_shields)**1.5 * (
            ((s - 1) * self.config.gravity * self.config.grain_diameter**3) / self.config.viscosity**2) ** 0.1 
            * np.sqrt((s - 1) * self.config.gravity * self.config.grain_diameter)
        )
        
        # frequency of particle pickup (macdonald et al., 2006, equation 60)
        freq_pickup = q_pickup / self.config.grain_diameter
        
        # active layer depth (mixing depth) (macdonald et al., 2006, equation 63)
        h_active = 5 * (max_shields_number - critical_shields) * self.config.grain_diameter
        
        # mixing factor (macdonald et al., 2006, equation 64)
        if h_active > self.config.grain_diameter:
            K_mixing = self.config.grain_diameter / h_active
        else:
            K_mixing = 1.0
        
        # burial factor (macdonald et al., 2006, equation 66)
        h_burial = 0 # burial depth not implemented yet
        K_burial = 1 - h_burial/h_active
        K_burial = np.clip(K_burial, 0, 1)  # ensure between 0 and 1
        
        # frequency of entrainment (macdonald et al., 2006, equation 57)
        freq_entrainment = K_burial * K_mixing * freq_pickup
        
        # ----------------------------------------------------------------------------------------    
        # MacDonald Q3D mode
        
        # mean particle fall time (macdonald et al., 2006, equation 36)
        t_fall = z_c / settling_velocity
        
        # mean particle wait time (macdonald et al., 2006, equation 37)
        t_wait = 1 / freq_entrainment
        
        # proportion of time particle is entrained in flow (macdonald et al., 2006, equation 38)
        p_time_entrained = t_fall / t_wait
        
        # velocity deficit coefficient (macdonald et al., 2006, equation 39)
        # I DON'T THINK I'M IMPLEMENTING THE LOGIC CORRECTLY HERE- CHECK E.G. INDEXING
        velocity_deficit_coeff = np.ones_like(mean_particle_velocity)
        if p_time_entrained > 1:
            velocity_deficit_coeff = 1
        else:
            velocity_deficit_coeff = p_time_entrained
            
        # particle position quasi 3D
        z_p = z_c # FOR NOW! THIS NEEDS TO BE IN PARTICLE PROPERTIES AND UPDATED DYNAMICALLY
        
        # # adjusted mean particle velocity (macdonald et al., 2006, equation 40)
        # if and(z_p > 0, z_p <= z_c):
        #     mean_particle_velocity = velocity_deficit_coeff * 
        #     u_zp = calculate_macdonald_particle_velocity_at_zp(max_shear_velocity, z_p, k_s)
        # mean_particle_velocity
        
        # particle deposition rate (macdonald et al., 2006, equation 67)
        if z_p < k_s_skin/4:
            is_deposited = True
        else:
            is_deposited = False
        # NB. THIS SHOULD BE IN PARTICLE LOOP, NOT APPLIED TO FIELDS
        
        # q3D entrainment
        if random.uniform(0, 1) < freq_entrainment * self.config.time_step:
            is_entrained = True
            z_p = z_c # initially set particle height to centroid height
        else:
            is_entrained = False
            z_p = z_bed # keep particle at bed level
        # NB. THIS SHOULD BE IN PARTICLE LOOP, NOT APPLIED TO FIELDS

        # if bed-particle interaction disabled, deposited particles instantaneously re-entrained to z_c if mobility M>1
        # NB. THIS SHOULD BE IN PARTICLE LOOP, NOT APPLIED TO FIELDS
        
        # macdonald ----------------------------------------------------------------------------------------

        # vanwesten -----------------------------------------------------------------------------------------
        # # Expand dimensions if necessary
        # if has_fraction_dim:
        #     shields_number = shields_number[:, np.newaxis, :]
        #     bed_load_layer_thickness = bed_load_layer_thickness[:, np.newaxis, :]
        #     suspended_layer_thickness = suspended_layer_thickness[:, np.newaxis, :]
        #     mixing_layer_thickness = mixing_layer_thickness[:, np.newaxis, :]

        #     bed_load_velocity = bed_load_velocity[:, np.newaxis, :]
        #     bed_load_velocity_x = bed_load_velocity_x[:, np.newaxis, :]
        #     bed_load_velocity_y = bed_load_velocity_y[:, np.newaxis, :]

        #     suspended_velocity = suspended_velocity[:, np.newaxis, :]
        #     suspended_velocity_x = suspended_velocity_x[:, np.newaxis, :]
        #     suspended_velocity_y = suspended_velocity_y[:, np.newaxis, :]

        # Compute transport probabilities (placeholders for now)
        with np.errstate(divide='ignore', invalid='ignore'):
            bed_load_probability = np.ones_like(bed_load_velocity) 
            suspended_probability = np.ones_like(suspended_velocity) 

        # Depending on transport_probability_method; apply transport probabilities
        if transport_probability_method == 'reduced_velocity':
            # Apply reduced velocity to bed load velocity
            bed_load_velocity_x *= bed_load_probability
            bed_load_velocity_y *= bed_load_probability
            bed_load_velocity *= bed_load_probability

            # Apply reduced velocity to suspended velocity
            suspended_velocity_x *= suspended_probability
            suspended_velocity_y *= suspended_probability
            suspended_velocity *= suspended_probability

        if transport_probability_method == 'reduced_velocity' or transport_probability_method == 'no_probability':
            # Reset probabilities to one
            bed_load_probability[:] = 1.0
            suspended_probability[:] = 1.0

        # Add physics fields to SedtrailsData

        # Physics parameters (scalar fields)
        sedtrails_data.add_physics_field('max_shields_number', max_shields_number)
        sedtrails_data.add_physics_field('bed_load_layer_thickness', bed_load_layer_thickness)
        sedtrails_data.add_physics_field('suspended_layer_thickness', suspended_layer_thickness)
        sedtrails_data.add_physics_field('mixing_layer_thickness', mixing_layer_thickness)

        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'bed_load_velocity', {'x': bed_load_velocity_x, 'y': bed_load_velocity_y, 'magnitude': bed_load_velocity}
        )
        sedtrails_data.add_physics_field(
            'suspended_velocity',
            {'x': suspended_velocity_x, 'y': suspended_velocity_y, 'magnitude': suspended_velocity},
        )

        sedtrails_data.add_physics_field(
            'mean_particle_velocity',
            {'x': mean_particle_velocity_x, 'y': mean_particle_velocity_y, 'magnitude': mean_particle_velocity},
        )

        # Probability fields
        sedtrails_data.add_physics_field('bed_load_probability', bed_load_probability)
        sedtrails_data.add_physics_field('suspended_probability', suspended_probability)


    def calculate_macdonald_skin_roughness(self, grain_size):
        """
        Skin-friction roughness (MacDonald Eq. 13).
        Typically approximated as 3*d for uniform sediments.
        """
        return 3.0 * grain_size

    def calculate_macdonald_form_roughness(self, theta_max, theta_cr, grain_size, water_depth):
        """
        Form roughness (bedform height) from MacDonald et al. (2006), Eq. 12.
        Returns k_s_form with same shape as theta_max.
        """
        # initialize bedform height array
        eta_b = np.zeros_like(theta_max)

        # ratio of Shields parameter to critical Shields
        theta_ratio = theta_max / theta_cr

        # apply the valid range for Eq. 12: 1 < theta/theta_cr < 24
        mask = (theta_ratio > 1) & (theta_ratio < 24)

        eta_b[mask] = (
            0.11 * water_depth[mask]
            * (grain_size / water_depth[mask]) ** 0.3
            * (1 - np.exp(-0.5 * (theta_ratio[mask] - 1)))
            * (24 - theta_ratio[mask])
        )

        # form roughness is the bedform height
        return eta_b

    def calculate_soulsby_vanrijn_potential_transport(self, water_depth, flow_velocity_magnitude,
                                        U_rms, dstar, s, k_s_skin, k_s_form):
        """
        Compute the Soulsby–van Rijn potential transport components
        as described in MacDonald et al. (2006), eq. 16–20.

        Parameters
        ----------
        water_depth : array
            Local water depth [m].
        flow_velocity_magnitude : array
            Depth-averaged current velocity magnitude [m/s].
        U_rms : array
            RMS wave orbital velocity near bed [m/s].
        dstar : float
            Dimensionless grain parameter.
        s : float
            Relative density = rho_s / rho.
        k_s_skin : float or array
            Skin roughness height [m].
        k_s_form : float or array
            Form roughness height (bedforms) [m].

        Returns
        -------
        A_s : array
            Combined transport factor A_s.
        C_d : array
            Drag coefficient.
        U_cr : array
            Critical depth-mean velocity.
        q_t : array
            Soulsby–van Rijn potential sediment transport rate.
        """

        d = self.config.grain_diameter
        g = self.config.gravity

        # ------------------------------
        # A_sb (bedload factor)
        # ------------------------------
        A_sb = np.full_like(water_depth, np.nan)
        mask = water_depth > 0

        A_sb[mask] = (
            0.005 * water_depth[mask] *
            (d / water_depth[mask]) ** 1.2
            / (g * (s - 1) * d) ** 1.2
        )

        # ------------------------------
        # A_ss (suspended load factor)
        # ------------------------------
        A_ss = (
            0.012 * d * dstar ** -0.6
            / (g * (s - 1) * d) ** 1.2
        )

        # Combined factor
        A_s = A_sb + A_ss

        # ------------------------------
        # drag coefficient C_d
        # ------------------------------
        z_0 = (k_s_skin + k_s_form) / 30.0
        C_d = np.full_like(water_depth, np.nan)

        C_d[mask] = 0.4 / (np.log(water_depth[mask] / z_0[mask]) - 1.0) ** 2

        # ------------------------------
        # Critical velocity U_cr
        # ------------------------------
        U_cr = np.full_like(water_depth, np.nan)

        arg = 4.0 * water_depth / d
        mask_cr = arg > 1.0  # Safe domain for log10

        if d < 0.0005:
            U_cr[mask_cr] = 0.19 * d ** 0.1 * np.log10(arg[mask_cr])
        else:
            U_cr[mask_cr] = 8.5 * d ** 0.6 * np.log10(arg[mask_cr])

        # ------------------------------
        # potential Soulsby–van Rijn total transport
        # ------------------------------
        U_comb = np.sqrt(flow_velocity_magnitude ** 2 + (0.018 / C_d) * U_rms ** 2)

        # positive excess only
        excess = np.maximum(U_comb - U_cr, 0.0)

        q_t = A_s * flow_velocity_magnitude * excess ** 2.4

        return A_s, C_d, U_cr, q_t
    
    def compute_rouse_number(self, settling_velocity, mean_shear_velocity):
        """
        Compute the Rouse number:
        
            P = w_s / (kappa * u_*)
        
        where:
            w_s  = settling velocity [m/s]
            u_*  = shear velocity [m/s]
            kappa = von Kármán constant (≈0.4)

        Parameters
        ----------
        settling_velocity : float
            Particle settling velocity [m/s].
        mean_shear_velocity : array
            Shear velocity field [m/s].

        Returns
        -------
        rouse_number : array
            Rouse number (same shape as mean_shear_velocity),
            NaN where u_* <= 0.
        """

        kappa = self.config.von_karman_constant
        # initialize result
        rouse_number = np.full_like(mean_shear_velocity, np.nan)
        # mask where shear velocity is positive
        mask = mean_shear_velocity > 0
        # compute only where valid
        rouse_number[mask] = (
            settling_velocity / (kappa * mean_shear_velocity[mask])
        )
        return rouse_number

    def compute_settling_velocity_macdonald(self, dstar):
        """
        Compute settling velocity using the MacDonald et al. (2006) formulation:

            (w_s * D_*) / nu = sqrt(107.33 + 1.049 D_*^3) - 10.36     for D_* >= 0.672
            (w_s * D_*) / nu = 0.0077 D_*^2                           for D_* < 0.672

        Parameters
        ----------
        dstar : float
            Dimensionless grain size D_*.

        Returns
        -------
        w_s : float
            Settling velocity [m/s].
        """

        nu = self.config.kinematic_viscosity
        d = self.config.grain_diameter

        if dstar >= 0.672:
            # Large-grain formula
            term = np.sqrt(107.33 + 1.049 * dstar**3) - 10.36
            w_s = (nu / d) * term
        else:
            # Small-grain formula (simplified)
            w_s = 0.0077 * nu * d

        return float(w_s)


    def calculate_macdonald_susp_load_height(self, rouse_number, water_depth):
        """Calculate height of centroid of suspended load using MacDonald et al. (2006) Eq. 27"""
        log_arg = np.log(rouse_number) - 0.4
        tanh_arg = 1.2 * log_arg
        
        # Calculate MacDonald height
        z_s = water_depth * 0.0398 * (10 ** (-1.08 * np.tanh(tanh_arg)))
        return z_s

    def calculate_macdonald_suspended_load_velocity(self, max_shear_velocity, z_s, k_s):
        """Calculate suspended load velocity using MacDonald et al. (2006) Eq. 29"""
        suspended_load_velocity = np.zeros_like(z_s)*np.nan
        mask = (30 * z_s / k_s) > 1  # ensure log argument is positive
        suspended_load_velocity[mask] = 2.5 * max_shear_velocity[mask] * np.log(30 * z_s[mask] / k_s[mask])
        
        return suspended_load_velocity

    def calculate_macdonald_particle_velocity_at_zp(self, max_shear_velocity, z_p, k_s):
        """Calculate particle velocity at height z_p using MacDonald et al. (2006) Eq. 33"""
        u_a_zp = max_shear_velocity * (5.75 * np.log10(z_p / k_s) + 8.5) 
        
        return u_a_zp