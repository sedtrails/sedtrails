"""A plugin for MacDonald et al. (2006) sediment transport physics calculations."""

import os
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
# from random import random
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

    # Shared class-level cache for MacDonald lookup table
    # Loaded lazily on first use and shared across all plugin instances
    _macdonald_lookup_da = None
    _macdonald_lookup_path = None
    _macdonald_warned_oob = False

    def __init__(self, config, tracer_methods: None):
        super().__init__()
        self.config = config
        
        # This plugin relies on shared class-level MacDonald lookup cache
        _ = PhysicsPlugin._macdonald_lookup_da
        _ = PhysicsPlugin._macdonald_lookup_path
        _ = PhysicsPlugin._macdonald_warned_oob


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

        # water depth
        water_depth = sedtrails_data.water_depth

        # Compute transport velocities (these will have shape [time, spatial])
        critical_shields = grain_properties.get('critical_shields')
        if critical_shields is None:
            raise ValueError("Missing required 'critical_shields' value in grain_properties.")
        
        # Bed roughness (MacDonald et al., 2006, equations 12-13)
        k_s_skin = physics_lib.calculate_skin_roughness(method="soulsby_d50", d50=self.config.grain_diameter) #Consider here passing the background grain diameter instead of the particle grain diameter.
        k_s_form = physics_lib.calculate_equilibrium_bedform_height(mean_shields_number, critical_shields, self.config.grain_diameter, water_depth)
        k_s_total = k_s_form + k_s_skin  
        # TO DO: Note that k_s_form is the equilibrium bedform height eta_b from MacDonald et al. (2006) Eq. 12 - we should implement the rate of change also (eq. 14-15) in future
        

        s = physics_lib.calculate_relative_density_ratio(self.config.particle_density, self.config.water_density)    # relative density ratio
        dstar = grain_properties.get('dimensionless_grain_size')  
        settling_velocity = grain_properties.get('settling_velocity')  # gives very similar results to physics_lib.compute_settling_velocity
        rouse_number = physics_lib.calculate_rouse_number(settling_velocity, max_shear_velocity, self.config.von_karman_constant)


        if self.config.use_transport_fields=='model-native':
            
            # bed load transport
            # bed_load_transport_x = sedtrails_data.bed_load_transport['x']
            # bed_load_transport_y = sedtrails_data.bed_load_transport['y']
            bed_load_transport_magnitude = sedtrails_data.bed_load_transport['magnitude']

            # suspended transport
            # suspended_transport_x = sedtrails_data.suspended_transport['x']
            # suspended_transport_y = sedtrails_data.suspended_transport['y']
            suspended_transport_magnitude = sedtrails_data.suspended_transport['magnitude']
            
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
                # bed_load_transport_x_calc = bed_load_transport_x.squeeze(axis=1)
                # bed_load_transport_y_calc = bed_load_transport_y.squeeze(axis=1)
                bed_load_transport_magnitude_calc = bed_load_transport_magnitude.squeeze(axis=1)

                # suspended_transport_x_calc = suspended_transport_x.squeeze(axis=1)
                # suspended_transport_y_calc = suspended_transport_y.squeeze(axis=1)
                suspended_transport_magnitude_calc = suspended_transport_magnitude.squeeze(axis=1)
                # has_fraction_dim = True
            else:
                # Data already doesn't have fraction dimension
                # bed_load_transport_x_calc = bed_load_transport_x
                # bed_load_transport_y_calc = bed_load_transport_y
                bed_load_transport_magnitude_calc = bed_load_transport_magnitude

                # suspended_transport_x_calc = suspended_transport_x
                # suspended_transport_y_calc = suspended_transport_y
                suspended_transport_magnitude_calc = suspended_transport_magnitude
                # has_fraction_dim = False
            
            # suspended load transport fraction 
            den = suspended_transport_magnitude_calc + bed_load_transport_magnitude_calc
            qs_qt = np.full_like(suspended_transport_magnitude_calc, 0.0, dtype=float)
            mask = den > 0
            qs_qt[mask] = suspended_transport_magnitude_calc[mask] / den[mask]
            
        elif self.config.use_transport_fields=='SoulsbyvanRijn1997':

            U_rms=np.zeros_like(sedtrails_data.water_depth)# change later to sedtrails_data.u_rms_wave # root mean square wave orbital velocity near bed [m/s]

            # soulsby-vanRijn factors (MacDonald et al., 2006, equations 16-18)
            A_s, C_d, U_cr, q_t_soulsbyVanRijn = self.calculate_soulsby_vanrijn_potential_transport(water_depth=water_depth,
                                                                                        flow_velocity_magnitude=flow_velocity_magnitude,
                                                                                        U_rms=U_rms,
                                                                                        dstar=dstar,
                                                                                        s=s,
                                                                                        k_s_skin=k_s_skin,
                                                                                        k_s_form=k_s_form,
                                                                                        )


            # suspended load transport fraction, equation 31 MacDonald et al. (2006)
            qs_qt = np.full_like(max_shear_velocity, np.nan, dtype=float)
            mask = (max_shear_velocity > 0) & (settling_velocity > 0) & np.isfinite(max_shear_velocity) & np.isfinite(settling_velocity)
            qs_qt[mask] = (0.5 * np.tanh(1.3 * np.log(max_shear_velocity[mask] / settling_velocity[mask]) - 0.3) + 0.5)            

        # suspended load height (MacDonald et al., 2006, equation 27)
        z_s = PhysicsPlugin.calculate_macdonald_susp_load_height(rouse_number, water_depth)
           
        # suspended load velocity (MacDonald et al., 2006, equation 29) 
        # Note Vassia: here we sum skin and form roughness for total roughness - eq. 29 says k_s'' indicating bedform roughness
        suspended_velocity    = PhysicsPlugin.calculate_macdonald_loglaw_velocity_at_z(mean_shear_velocity, z_s,  k_s_total)
        if self.config.max_suspended_velocity_factor is not None:
            suspended_velocity = PhysicsPlugin.cap_particle_velocity(
                suspended_velocity,
                flow_velocity_magnitude,
                self.config.max_suspended_velocity_factor,
            )
        #Question Vassia: should the max_suspended_velocity_factor be used here or in the final step - on the mean_particle_velocity?
        
        # bed load velocity (MacDonald et al., 2006, equation 30) - Engelund & Fredsoe (1976), same as Soulsby et al (2011)
        bed_load_velocity = physics_lib.compute_bed_load_velocity(max_shields_number, critical_shields, mean_shear_velocity)

        # Compute transport probabilities (placeholders for now)
        with np.errstate(divide='ignore', invalid='ignore'):
            bed_load_probability = np.ones_like(bed_load_velocity) 
            suspended_probability = np.ones_like(suspended_velocity) 

        # Depending on transport_probability_method; apply transport probabilities
        if transport_probability_method == 'reduced_velocity':
            # Apply reduced velocity to bed load velocity
            bed_load_velocity *= bed_load_probability

            # Apply reduced velocity to suspended velocity
            suspended_velocity *= suspended_probability

        if transport_probability_method == 'reduced_velocity' or transport_probability_method == 'no_probability':
            # Reset probabilities to one
            bed_load_probability[:] = 1.0
            suspended_probability[:] = 1.0

        # sediment advection velocity (MacDonald et al., 2006, equation 32)
        u_zc = qs_qt * suspended_velocity + (1 - qs_qt) * bed_load_velocity
        # total transport centroid elevation (MacDonald et al., 2006, equation 34)
        z_c = np.zeros_like(u_zc)
        mask = max_shear_velocity > 0
        z_c[mask] = k_s_total[mask] * 10 ** (0.1739 * (u_zc[mask] / max_shear_velocity[mask]) - 1.47826)
        
        # horizontal mean particle advection velocity (u_a) (MacDonald et al., 2006, equation 35)
        mean_particle_velocity = u_zc #no need to calculate it again, previously PhysicsPlugin.calculate_2d_total_load_particle_advection_velocity(max_shear_velocity, z_c,  k_s_total)
        

        # # ----------------------------------------------------------------------------------------
        # # MacDonald probabilistic bed-particle interaction model
 
        #PTM turbulence parameter (macdonald et al., 2006 below equation 54)
        timestep = 60 # we need to get this dynamically from the model, this is just a placeholder value for now
        gamma=0.4 # We need to update this to 0.05 * timestep . How do i get the timer._current_timestep information in here? 

        # standard deviation of the shear stress fluctuation (macdonald et al., 2006 equation 54)
        sigma_tau=gamma * max_bed_shear_stress# mean or max bed shear stress?
        mean_tau=max_bed_shear_stress # mean or max bed shear stress?

        # sample turbulent bed shear stress based ona normal distribution with mean_tau, sigma_tau
        turbulent_shear = np.random.normal(mean_tau, sigma_tau)
        # enforce physical constraint - shear stress cannot be negative
        turbulent_shear = np.clip(turbulent_shear, a_min=0.0, a_max=None)



        # turbulent Shields number (macdonald et al., 2006 equation 59)
        turbulent_shields = turbulent_shear/ (self.config.water_density * self.config.gravity * self.config.grain_diameter * (s - 1))

        # particle entrainment rate (van Rijn (1984b) pickup function, as used in MacDonald et al., 2006, equation 58)
        q_pickup = np.zeros_like(turbulent_shields, dtype=float)
        # Valid pickup condition (mobility > 1)
        valid = (
            np.isfinite(turbulent_shields)
            & np.isfinite(critical_shields)
            & (critical_shields > 0)
            & (turbulent_shields > critical_shields)
        )
        # Compute only where physically allowed
        theta_excess = (
            (turbulent_shields[valid] - critical_shields)
            / critical_shields
        )
        q_pickup[valid] = (
            0.00033
            * theta_excess**1.5
            * (((s - 1) * self.config.gravity * self.config.grain_diameter**3)
            / self.config.kinematic_viscosity**2) ** 0.1
            * np.sqrt((s - 1) * self.config.gravity * self.config.grain_diameter)
        )
        # frequency of particle pickup (macdonald et al., 2006, equation 61)
        freq_pickup = q_pickup / self.config.grain_diameter
        
        # active layer depth (mixing depth) (macdonald et al., 2006, equation 63)
        h_active = np.zeros_like(turbulent_shields, dtype=float)
        h_active[valid] = 5 * (turbulent_shields[valid] - critical_shields) * self.config.grain_diameter
        
        # mixing factor (macdonald et al., 2006, equation 64)
        K_mixing = np.where(
            h_active > self.config.grain_diameter,
            self.config.grain_diameter / h_active,
            1.0
        )
        
        # burial factor (macdonald et al., 2006, equation 66)
        h_burial = np.zeros_like(h_active) # burial depth not implemented yet
        K_burial = np.where(
            h_active > 0,
            1.0 - (h_burial / h_active),
            0.0 # h_active is 0 when out of the valid area, where turbulent_shear<critical shear. For those areas also the freq_pickup=0
        )
        K_burial = np.clip(K_burial, 0, 1)  # ensure between 0 and 1
        
        # frequency of entrainment (macdonald et al., 2006, equation 57)
        freq_entrainment = K_burial * K_mixing * freq_pickup
        
        # # ----------------------------------------------------------------------------------------    
        # # MacDonald Q3D mode
        
        # mean particle fall time (macdonald et al., 2006, equation 36)
        t_fall = z_c / settling_velocity
        
        # mean particle wait time (macdonald et al., 2006, equation 37)
        t_wait = 1 / freq_entrainment
        
        # proportion of time particle is entrained in flow (macdonald et al., 2006, equation 38)
        p_time_entrained = np.zeros_like(t_fall)
        mask = t_wait > 0
        p_time_entrained[mask] = t_fall[mask] / t_wait[mask]
        
        # velocity deficit coefficient (macdonald et al., 2006, equation 39)
        velocity_deficit_coeff = np.ones_like(mean_particle_velocity)
        velocity_deficit_coeff = np.where(
            p_time_entrained > 1.0,
            1.0,
            p_time_entrained
        )
            
        # particle position quasi 3D
        z_p = z_c # FOR NOW! THIS NEEDS TO BE IN PARTICLE PROPERTIES AND UPDATED DYNAMICALLY
        
        # # adjusted mean particle velocity (macdonald et al., 2006, equation 40)
        u_zp = PhysicsPlugin.calculate_2d_total_load_particle_advection_velocity(max_shear_velocity, z_p,  k_s_total)
        u_1p4zc =  PhysicsPlugin.calculate_2d_total_load_particle_advection_velocity(max_shear_velocity, 1.4*z_c,  k_s_total)
        mean_particle_velocity = PhysicsPlugin.apply_q3d_velocity_deficit(u_zp, u_zc, u_1p4zc, z_p, z_c, velocity_deficit_coeff)

        # calculate x and y components of mean particle velocity
        mean_particle_velocity_x, mean_particle_velocity_y = physics_lib.compute_directions_from_magnitude(
            mean_particle_velocity,
            flow_velocity_x,
            flow_velocity_y,
            flow_velocity_magnitude,
        )
        # Note Vassia: we could consider getting particle velocity direction from bedload and suspended load instead of flow velocity
        # In that way the effect of slopes which are considered in D3D might be included better. 
        # Question Vassia - should i check here again to cap the mean particle velocity?

        E_turb_hor, E_turb_vert = PhysicsPlugin.compute_turbulent_diffusion_coefficients(
            water_depth=water_depth,
            z_p=z_p,
            flow_velocity_magnitude=flow_velocity_magnitude,
            shear_velocity=max_shear_velocity,
            K_Et=0.15,
        )

        u_Dx, u_Dy, w_D = PhysicsPlugin.compute_random_walk_diffusion_velocities(
            E_turb_hor=E_turb_hor,
            E_turb_vert=E_turb_vert,
            dt=timestep,
        )

        # calculate velocity divergence using KNN least-squares fit
        # this is just a first order approximation for the divergence, we could consider more sophisticated methods in the future
        divU, dudx, dvdy = PhysicsPlugin.divergence_scattered_knn_time(sedtrails_data.x, 
                                                                       sedtrails_data.y, 
                                                                       sedtrails_data.depth_avg_flow_velocity['x'], 
                                                                       sedtrails_data.depth_avg_flow_velocity['y'], 
                                                                       k=12, r_max=150)
        
        # """" # --- plot vectors (units are m/s; scale controls arrow length)
        # ax.quiver(
        #     x[idx], y[idx],
        #     u[it, idx], v[it, idx],
        #     color="k",
        #     alpha=0.75,
        #     width=0.0025,
        #     headwidth=3.5,
        #     headlength=4.5,
        #     headaxislength=4.0,
        #     scale=0.003,          # larger -> shorter arrows; tune this
        #     scale_units="xy",  # interpret in data units
        #     angles="xy",
        #     zorder=2
        # )

        # # optional: add a reference arrow for magnitude
        # ref_speed = np.nanpercentile(np.sqrt(u[it]**2 + v[it]**2), 80)  # m/s
        # ax.quiverkey(
        #     ax.collections[-1] if hasattr(ax, "collections") else None,  # safe-ish
        #     X=0.85, Y=1.03, U=ref_speed,
        #     label=f"{ref_speed:.2f} m/s", labelpos="E"
        # )

        # ax.set_aspect("equal")
        # ax.set_title(f"Divergence + depth-avg velocity vectors (t={it})")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("y [m]")
        # ax.grid(True, linestyle=":", alpha=0.3)

        # fig.tight_layout()
        # plt.show()
        # """"

        dh_dt = PhysicsPlugin.compute_dh_dt(
            sedtrails_data.water_depth,
            sedtrails_data.bed_level,
            timestep
        )

        # initialise output
        w_zp = np.zeros_like(sedtrails_data.water_depth)

        # explicit mask: only compute where depth > 0
        wet = sedtrails_data.water_depth != 0

        w_zp[wet] = (
            (dh_dt[wet] / sedtrails_data.water_depth[wet]) + divU[wet]
        ) * (sedtrails_data.water_depth[wet] - z_p[wet])

        # optional but recommended safeguard
        w_zp = np.nan_to_num(w_zp, nan=0.0, posinf=0.0, neginf=0.0)





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
        
        # # macdonald ----------------------------------------------------------------------------------------

        # # vanwesten -----------------------------------------------------------------------------------------
        # # # Expand dimensions if necessary
        # # if has_fraction_dim:
        # #     shields_number = shields_number[:, np.newaxis, :]
        # #     bed_load_layer_thickness = bed_load_layer_thickness[:, np.newaxis, :]
        # #     suspended_layer_thickness = suspended_layer_thickness[:, np.newaxis, :]
        # #     mixing_layer_thickness = mixing_layer_thickness[:, np.newaxis, :]

        # #     bed_load_velocity = bed_load_velocity[:, np.newaxis, :]
        # #     bed_load_velocity_x = bed_load_velocity_x[:, np.newaxis, :]
        # #     bed_load_velocity_y = bed_load_velocity_y[:, np.newaxis, :]

        # #     suspended_velocity = suspended_velocity[:, np.newaxis, :]
        # #     suspended_velocity_x = suspended_velocity_x[:, np.newaxis, :]
        # #     suspended_velocity_y = suspended_velocity_y[:, np.newaxis, :]

        # Empty fields for Soulsby (only in van westen)
        mixing_layer_thickness = np.zeros_like(mean_particle_velocity)
        # Add physics fields to SedtrailsData

        # Physics parameters (scalar fields)
        sedtrails_data.add_physics_field('max_shields_number', max_shields_number)
        sedtrails_data.add_physics_field('mixing_layer_thickness', mixing_layer_thickness)
        sedtrails_data.add_physics_field('bed_load_probability', bed_load_probability)
        sedtrails_data.add_physics_field('suspended_probability', suspended_probability)
        sedtrails_data.add_physics_field('suspended_velocity', suspended_velocity) # this is for debugging purposes
        sedtrails_data.add_physics_field('bedload_velocity', bed_load_velocity) # this is for debugging purposes
        sedtrails_data.add_physics_field('suspended_transport_centroid_elevation', z_s) # this is for debugging purposes
        sedtrails_data.add_physics_field('suspended_transport_centroid_elevation_over_depth', z_s/water_depth) # this is for debugging purposes
        sedtrails_data.add_physics_field('total_transport_centroid_elevation', z_c) # this is for debugging purposes
        sedtrails_data.add_physics_field('total_transport_centroid_elevation_over_depth', z_c/water_depth) # this is for debugging purposes
        sedtrails_data.add_physics_field('particle_advection_velocity', u_c) # this is for debugging purposes
        sedtrails_data.add_physics_field('rouse_number', rouse_number) # this is for debugging purposes
        sedtrails_data.add_physics_field('bedform_roughness_height', k_s_form) # this is for debugging purposes
        sedtrails_data.add_physics_field('total_roughness_height', k_s_total) # this is for debugging purposes
        sedtrails_data.add_physics_field('shear_velocity_ratio', max_shear_velocity/mean_shear_velocity) # this is for debugging purposes
        sedtrails_data.add_physics_field('suspended_transport_ratio', qs_qt) # this is for debugging purposes
        sedtrails_data.add_physics_field('bed_load_transport_ratio', 1-qs_qt) # this is for debugging purposes
        sedtrails_data.add_physics_field('suspended_velocity_over_da_velocity', suspended_velocity/flow_velocity_magnitude) # this is for debugging purposes
        sedtrails_data.add_physics_field('30z_s_over_k_s_total', 30 * z_s / k_s_total) # this is for debugging purposes
        sedtrails_data.add_physics_field('max_shear_velocity', max_shear_velocity) # this is for debugging purposes
        sedtrails_data.add_physics_field('mean_shear_velocity', mean_shear_velocity) # this is for debugging purposes
        sedtrails_data.add_physics_field('suspended_load_velocity_lnpart', np.log(30 * z_s / k_s_total)) # this is for debugging purposes
        sedtrails_data.add_physics_field('particle_velocity_over_da_velocity', mean_particle_velocity/flow_velocity_magnitude) # this is for debugging purposes
        sedtrails_data.add_physics_field('diffusive_velocity_x', u_Dx)# this is for debugging purposes
        sedtrails_data.add_physics_field('diffusive_velocity_y', u_Dy)# this is for debugging purposes
        sedtrails_data.add_physics_field('diffusive_velocity_z', w_D)# this is for debugging purposes
        sedtrails_data.add_physics_field('turbulent_diffusion_coefficient_horizontal', E_turb_hor)# this is for debugging purposes
        sedtrails_data.add_physics_field('turbulent_diffusion_coefficient_vertical', E_turb_vert)# this is for debugging purposes
        sedtrails_data.add_physics_field('vertical_advection_velocity', w_zp)# this is for debugging purposes


        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'mean_particle_velocity',
            {'x': mean_particle_velocity_x, 'y': mean_particle_velocity_y, 'magnitude': mean_particle_velocity},
        )

    @staticmethod
    def compute_dh_dt(water_depth, bed_level, dt):
        """
        Time derivative of water depth dh/dt using backward difference.
        """
        dh_dt = np.zeros_like(water_depth)
        wet = water_depth != 0
        
        dh_dt[1:] = ((water_depth[1:]+bed_level[1:]) - (water_depth[:-1]+bed_level[:-1])) / dt
        dh_dt[0] = dh_dt[1]  # reasonable fill for first timestep
        
        wet = water_depth != 0
        dh_dt[~wet] = np.nan  # set dh/dt to NaN where water depth is zero (dry areas)
        return dh_dt


    @staticmethod
    def divergence_scattered_knn_time(
        x, y,
        u, v,
        k=12,
        r_max=None,
    ):
        """
        Time-aware wrapper for KNN least-squares divergence.

        Parameters
        ----------
        x, y : (N,) arrays
            Coordinates [m]
        u, v : (nt, N) arrays
            Velocity components [m/s]
        k : int
            Number of nearest neighbours
        r_max : float or None
            Max neighbour radius [m]

        Returns
        -------
        divU : (nt, N) array
            Horizontal divergence ∇·U [1/s]
        dudx, dvdy : (nt, N) arrays
            Gradient components (diagnostics)
        """

        x = np.asarray(x)
        y = np.asarray(y)
        u = np.asarray(u)
        v = np.asarray(v)

        # ---- sanity checks (VERY IMPORTANT)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")

        if u.ndim != 2 or v.ndim != 2:
            raise ValueError("u and v must be 2D arrays (nt, npoints)")

        nt, npoints = u.shape
        if x.size != npoints:
            raise ValueError(
                f"Spatial mismatch: x has {x.size} points, "
                f"but u/v have {npoints}"
            )

        divU = np.full((nt, npoints), np.nan)
        dudx = np.full((nt, npoints), np.nan)
        dvdy = np.full((nt, npoints), np.nan)

        for it in range(nt):
            divU[it], dudx[it], dvdy[it] = PhysicsPlugin.divergence_scattered_knn(
                x, y,
                u[it], v[it],
                k=k,
                r_max=r_max,
            )

        return divU, dudx, dvdy

    @staticmethod
    def divergence_scattered_knn(
        x, y, u, v,
        k=12,
        r_max=None,
        eps=1e-12
    ):
        """
        Best-effort horizontal divergence from scattered (x,y,u,v)
        using local KNN least-squares fits.

        Parameters
        ----------
        x, y : (N,) arrays
            Coordinates [m]
        u, v : (N,) arrays
            Depth-averaged velocity components [m/s]
        k : int
            Number of nearest neighbours (8–20 typical)
        r_max : float or None
            Max search radius [m]; avoids using neighbours across gaps
        eps : float
            Numerical safeguard

        Returns
        -------
        divU : (N,) array
            Estimated divergence [1/s]
        dudx, dvdy : (N,) arrays
            Components of divU (for diagnostics)
        """

        x = np.asarray(x); y = np.asarray(y)
        u = np.asarray(u); v = np.asarray(v)

        N = x.size
        divU = np.full(N, np.nan)
        dudx = np.full(N, np.nan)
        dvdy = np.full(N, np.nan)

        coords = np.column_stack((x, y))
        tree = cKDTree(coords)

        # k+1 because the nearest neighbour is the point itself
        dists, idxs = tree.query(coords, k=k+1)

        for i in range(N):
            neigh = idxs[i][1:]
            r = dists[i][1:]

            if r_max is not None:
                mask = r <= r_max
                neigh = neigh[mask]
                r = r[mask]

            if neigh.size < 3:
                continue

            dx = x[neigh] - x[i]
            dy = y[neigh] - y[i]

            # Distance-weighted least squares
            w = 1.0 / (r + eps)
            W = np.sqrt(w)

            A = np.column_stack((np.ones_like(dx), dx, dy))
            Aw = A * W[:, None]

            # u-plane
            bu, *_ = np.linalg.lstsq(Aw, u[neigh] * W, rcond=None)
            # v-plane
            bv, *_ = np.linalg.lstsq(Aw, v[neigh] * W, rcond=None)

            dudx[i] = bu[1]
            dvdy[i] = bv[2]
            divU[i] = dudx[i] + dvdy[i]

        return divU, dudx, dvdy


    def apply_q3d_velocity_deficit(
        u_zp: xr.DataArray,
        u_zc: xr.DataArray,
        u_1p4zc: xr.DataArray,
        z_p: xr.DataArray,
        z_c: xr.DataArray,
        c_A: xr.DataArray,
    ) -> xr.DataArray:
        """

        Apply the Q3-D horizontal velocity deficit formulation of
        MacDonald et al. (2006), Eq. (40).

        This function reduces the horizontal particle advection velocity
        to account for intermittent particle–bed interaction in Q3-D mode.
        The reduction depends on particle elevation relative to the
        transport centroid and on the velocity deficit coefficient c_A
        (Eq. 39).


        Returns
        -------
        xarray.DataArray
            Reduced horizontal particle advection velocity. The returned velocity
            equals:
            • c_A * u(z_p) below the centroid,
            • a linear blend between c_A * u(z_c) and u(1.4 z_c) in the transition
                zone,
            • u(z_p) above 1.4 z_c.
        """

        # Linear blending term for transition region
        blending = (z_p - z_c) / (0.4 * z_c)

        # Transition-zone velocity (z_c < z_p <= 1.4 z_c)
        u_transition = (
            c_A * u_zc
            + blending * (u_1p4zc - c_A * u_zc)
        )

        # Piecewise definition of reduced velocity
        u_particle = xr.where(
            z_p <= z_c,
            c_A * u_zp,
            xr.where(
                z_p <= 1.4 * z_c,
                u_transition,
                u_zp,
            ),
        )

        return u_particle

    @staticmethod
    def compute_turbulent_diffusion_coefficients(
        water_depth,
        z_p,
        flow_velocity_magnitude,
        shear_velocity,
        *,
        K_Et=0.15,
        K_Ev=None,
        M_b=None,
        E_turb_hor_min=0.02,
        E_turb_vert_min=0.0,
    ):
        """
        Compute turbulent diffusion coefficients following MacDonald et al. (2006).

        Parameters
        ----------
        water_depth : array
            Total water depth h [m]
        z_p : array
            Particle height above bed z_p [m]
        flow_velocity_magnitude : array
            |U| depth-averaged velocity magnitude [m/s]
        shear_velocity : array
            Shear velocity u_* [m/s]
        K_Et : float
            Horizontal diffusion scaling coefficient (≈ 0.15–0.6)
        K_Ev : float or None
            Vertical diffusion scaling coefficient (if None, defaults to K_Et)
        M_b : array or None
            Wave-breaking enhancement factor (defaults to 1)
        E_turb_hor_min : float
            Minimum horizontal diffusivity [m²/s]
        E_turb_vert_min : float
            Minimum vertical diffusivity [m²/s] (default = 0 in PTM)

        Returns
        -------
        E_turb_hor, E_turb_vert : arrays
            Horizontal and vertical turbulent diffusion coefficients [m²/s]
        """

        if K_Ev is None:
            K_Ev = K_Et  # PTM uses same scaling unless specified

        if M_b is None:
            M_b = np.ones_like(flow_velocity_magnitude)

        # ---------------------
        # Horizontal diffusion
        # Equation (45) + (48)
        # ---------------------
        E_turb_hor = M_b * K_Et * water_depth * shear_velocity
        E_turb_hor = np.nan_to_num(E_turb_hor, nan=0.0)
        E_turb_hor = np.maximum(E_turb_hor, E_turb_hor_min)

        # ---------------------
        # Vertical diffusion
        # Equations (49) + (50)
        # ---------------------
        shape = np.zeros_like(water_depth)
        valid = water_depth > 0

        shape[valid] = (
            z_p[valid]
            * (water_depth[valid] - z_p[valid]) ** 2
            / water_depth[valid] ** 3
        )

        E_turb_vert = M_b * K_Ev * flow_velocity_magnitude * shape
        E_turb_vert = np.nan_to_num(E_turb_vert, nan=0.0)
        E_turb_vert = np.maximum(E_turb_vert, E_turb_vert_min)

        return E_turb_hor, E_turb_vert


    @staticmethod
    def compute_random_walk_diffusion_velocities(
        E_turb_hor,
        E_turb_vert,
        dt,
        rng=np.random,# default gives uniformly distributed random numbers in [0,1)
    ):
        """
        Compute random-walk diffusion velocities following MacDonald et al. (2006).

        Parameters
        ----------
        E_turb_hor : array
            Horizontal turbulent diffusivity [m²/s]
        E_turb_vert : array
            Vertical turbulent diffusivity [m²/s]
        dt : float
            Time step [s]
        rng : numpy random generator
            Random number generator (default: np.random)

        Returns
        -------
        u_Dx, u_Dy, w_D : arrays
            Random diffusion velocities [m/s]
        """

        # Uniform random numbers in [0,1]
        Pi_x = rng.random(E_turb_hor.shape)
        Pi_y = rng.random(E_turb_hor.shape)
        Pi_z = rng.random(E_turb_vert.shape)

        # Equation (51): horizontal (isotropic)
        scale_h = np.sqrt(6.0 * E_turb_hor / dt)
        u_Dx = 2.0 * (Pi_x - 0.5) * scale_h
        u_Dy = 2.0 * (Pi_y - 0.5) * scale_h

        # Equation (52): vertical
        scale_v = np.sqrt(6.0 * E_turb_vert / dt)
        w_D = 2.0 * (Pi_z - 0.5) * scale_v

        return u_Dx, u_Dy, w_D


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

        argument = 4.0 * water_depth / d
        mask_cr = argument > 1.0  # Safe domain for log10

        if d < 0.0005:
            U_cr[mask_cr] = 0.19 * d ** 0.1 * np.log10(argument[mask_cr])
        else:
            U_cr[mask_cr] = 8.5 * d ** 0.6 * np.log10(argument[mask_cr])

        # ------------------------------
        # potential Soulsby–van Rijn total transport
        # ------------------------------
        U_comb = np.sqrt(flow_velocity_magnitude ** 2 + (0.018 / C_d) * U_rms ** 2)

        # positive excess only
        excess = np.maximum(U_comb - U_cr, 0.0)

        q_t = A_s * flow_velocity_magnitude * excess ** 2.4

        return A_s, C_d, U_cr, q_t
    

    @staticmethod
    def calculate_macdonald_susp_load_height(rouse_number, water_depth):
        """
        Compute suspended-load centroid height z_s using a 1D lookup table of z_over_h = z_s/H vs Rouse.

        Behaviour:
        - Reads NetCDF from SAME folder as this plugin file.
        - Linear interpolation in rouse (1D).
        - For rouse >= R_ASYM: use asymptotic value z_s/H = ASYM  -> z_s = ASYM * H
        - For rouse < rmin or invalid inputs: return NaN
        - Prints a message if invalid / out-of-range inputs occur (once).
       
        The look up table is based on equation 27 of MacDonald et al. (2006) and was generated by solving 
        the equation numerically for a range of Rouse numbers, then saving the results in a NetCDF file for 
        fast loading and interpolation. This approach is much faster than solving the equation numerically 
        for each input during runtime, while still providing accurate results across the relevant range of 
        Rouse numbers (see create_lookuptable_zs_over_h_rouse.py in transport_converter/plugins/physics/)
        """
        # ----------------------------
        # Constants
        # ----------------------------
        R_ASYM = 20.0
        ASYM = 0.0398 * (10 ** (-1.08))  # ~0.003301...

        # ----------------------------
        # Load & cache lookup table (1D: z_over_h(rouse))
        # ----------------------------
        if PhysicsPlugin._macdonald_lookup_da is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))

            # file name for the NEW 1D table
            lookup_path = os.path.join(module_dir, "macdonald_z_over_h_lookup.nc")

            if not os.path.isfile(lookup_path):
                raise FileNotFoundError(
                    f"MacDonald lookup table not found: {lookup_path}\n"
                    "Expected 'macdonald_z_over_h_lookup.nc' to be in the same folder as this plugin file."
                )

            with xr.open_dataset(lookup_path) as ds:
                if "z_over_h" not in ds.data_vars:
                    raise KeyError(
                        f"'z_over_h' not found in {lookup_path}. Found variables: {list(ds.data_vars)}"
                    )
                da = ds["z_over_h"].load()  # load into memory, close file (Windows-safe)

            # Sanity checks
            if "rouse" not in da.coords:
                raise KeyError(
                    f"Lookup variable 'z_over_h' must have coord 'rouse'. Found coords: {list(da.coords)}"
                )
            if da.ndim != 1:
                raise ValueError(f"Expected 1D lookup for 'z_over_h', got ndim={da.ndim}, dims={da.dims}")

            PhysicsPlugin._macdonald_lookup_da = da
            PhysicsPlugin._macdonald_lookup_path = lookup_path

        lookup = PhysicsPlugin._macdonald_lookup_da  # DataArray over coord 'rouse'

        # ----------------------------
        # Prepare inputs (supports scalars or arrays, incl. 3D)
        # ----------------------------
        r = np.asarray(rouse_number, dtype=float)
        h = np.asarray(water_depth, dtype=float)

        # Ensure broadcastable shapes become identical (or raise)
        r, h = np.broadcast_arrays(r, h)

        scalar_out = (r.ndim == 0)

        # invalid: rouse must be > 0; depth must be >= 0; finite values only
        invalid = (~np.isfinite(r)) | (~np.isfinite(h)) | (r <= 0) | (h < 0)

        rmin = float(lookup["rouse"].min().values)
        rmax = float(lookup["rouse"].max().values)

        # ----------------------------
        # Masks for regimes
        # ----------------------------
        # High Rouse -> asymptote (this handles your huge values like 4.8e5)
        high = (~invalid) & (r >= R_ASYM)

        # Interpolation regime: within lookup domain and below R_ASYM
        interp_ok = (~invalid) & (r >= rmin) & (r <= rmax) & (r < R_ASYM)

        # Low out-of-bounds (below table): return NaN
        low_oob = (~invalid) & (r < rmin)

        # ----------------------------
        # Compute z_over_h
        # ----------------------------
        z_over_h = np.full_like(r, np.nan, dtype=float)

        # 1) asymptote for high Rouse
        z_over_h[high] = ASYM

        # 2) interpolate for normal regime
        if np.any(interp_ok):
            r_da = xr.DataArray(r[interp_ok])
            z_over_h[interp_ok] = lookup.interp(rouse=r_da, method="linear").values

        # 3) low_oob stays NaN; invalid stays NaN

        # ----------------------------
        # Warn once if needed
        # ----------------------------
        if (np.any(invalid) or np.any(low_oob)):
            if not PhysicsPlugin._macdonald_warned_oob:
                PhysicsPlugin._macdonald_warned_oob = True
                n_tot = int(r.size)
                n_invalid = int(np.count_nonzero(invalid))
                n_low = int(np.count_nonzero(low_oob))
                n_high = int(np.count_nonzero(high))
                print(
                    "[MacDonald lookup] Some inputs were invalid or out of lookup range.\n"
                    f"  invalid (NaN/Inf/rouse<=0/depth<0): {n_invalid}/{n_tot}\n"
                    f"  rouse < rmin ({rmin:g}) -> NaN: {n_low}/{n_tot}\n"
                    f"  rouse >= {R_ASYM:g} -> asymptote z_s/H={ASYM:.6g}: {n_high}/{n_tot}\n"
                    f"  lookup rouse range: [{rmin:g}, {rmax:g}] from {PhysicsPlugin._macdonald_lookup_path}"
                )

        # ----------------------------
        # Convert to z_s (meters)
        # ----------------------------
        z_s = z_over_h * h

        return float(z_s) if scalar_out else z_s
    

    @staticmethod
    def calculate_macdonald_loglaw_velocity_at_z(
        shear_velocity: np.ndarray,
        z: np.ndarray,
        roughness_height: np.ndarray,
    ) -> np.ndarray:
        """
        Compute mean horizontal fluid velocity at elevation z using the
        logarithmic velocity profile of MacDonald et al. (2006), Eq. (29).

            u(z) = 2.5 * u_* * ln(30 z / k_s)

        All inputs are NumPy arrays.

        Parameters
        ----------
        shear_velocity : ndarray
            Shear velocity u_* [m/s].
        z : ndarray
            Elevation above the bed [m].
        roughness_height : ndarray
            Bed roughness height k_s [m].

        Returns
        -------
        ndarray
            Log-law fluid velocity at elevation z [m/s].
        """

        # Initialize output
        u = np.zeros_like(z, dtype=float)

        # Log-law argument
        argument = 30.0 * z / roughness_height

        # Validity mask (physical + numerical)
        valid = (
            (argument > 1.0)
            & np.isfinite(argument)
            & np.isfinite(shear_velocity)
            & (shear_velocity > 0.0)
        )

        # Apply log-law where valid
        u[valid] = 2.5 * shear_velocity[valid] * np.log(argument[valid])

        return u


    @staticmethod
    def cap_particle_velocity(
        velocity: xr.DataArray,
        flow_velocity_magnitude: xr.DataArray,
        max_velocity_factor: float,
    ) -> xr.DataArray:
        """
        Cap particle velocity to a specified fraction of the depth-averaged
        flow velocity.

        This is a numerical or modeling safeguard and not part of the
        MacDonald et al. (2006) formulation.

        Parameters
        ----------
        velocity : xarray.DataArray
            Particle or fluid velocity to be capped [m/s].
        flow_velocity_magnitude : xarray.DataArray
            Magnitude of the ambient flow velocity [m/s].
        max_velocity_factor : float
            Maximum allowed fraction of the flow velocity (e.g. 1.0 or 0.8).

        Returns
        -------
        xarray.DataArray
            Capped velocity [m/s].
        """

        return xr.ufuncs.minimum(
            velocity,
            max_velocity_factor * flow_velocity_magnitude,
        )




    
    @staticmethod
    def calculate_2d_total_load_particle_advection_velocity(shear_velocity, reference_height, total_roughness):
        """
        Compute horizontal mean particle advection velocity u_a following
        MacDonald et al. (2006), Equation 35.

        The formulation is:

            u_a = u_*max · (5.75 · log10(z_c / k_s) + 8.5)

        where:
            u_*max = maximum shear velocity [m/s]
            z_c    = height of centroid of total transport [m]
            k_s    = bed roughness [m] (the form uses bedform roughness, here we use total roughness)

        The equation is valid only for z_c / k_s > 1.


        Notes
        -----
        - Values are only computed where z_c / k_s > 1 and k_s > 0.

        Reference
        ---------
        MacDonald, N. et al. (2006). PTM: Particle Tracking Model.
        Report 1: Model Theory, Implementation, and Example Applications.
        U.S. Army Corps of Engineers. Eq. 35.
        """

        mean_particle_velocity = np.zeros_like(
            shear_velocity, dtype=float
        )

        argument = reference_height / total_roughness
        mask = (
            np.isfinite(argument)
            & np.isfinite(shear_velocity)
            & (total_roughness > 0)
            & (argument > 1)
        )

        mean_particle_velocity[mask] = (
            shear_velocity[mask]
            * (5.75 * np.log10(argument[mask]) + 8.5)
        )

        return mean_particle_velocity
