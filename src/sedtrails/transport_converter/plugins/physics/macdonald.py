"""A plugin for MacDonald et al. (2006) sediment transport physics calculations."""

import os
import numpy as np
from sympy import arg
import xarray as xr
from random import random
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

    _macdonald_lookup_da = None
    _macdonald_lookup_path = None
    _macdonald_warned_oob = False

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
        k_s_skin = physics_lib.calculate_skin_roughness(self.config.grain_diameter) #Consider here passing the background grain diameter instead of the particle grain diameter.
        k_s_form = physics_lib.calculate_equilibrium_bedform_height(mean_shields_number, critical_shields, self.config.grain_diameter, water_depth)
        k_s_total = k_s_form + k_s_skin  
        # TO DO: Note that k_s_form is the equilibrium bedform height eta_b from MacDonald et al. (2006) Eq. 12 - we should implement the rate of change also (eq. 14-15) in future

        s = physics_lib.calculate_relative_density_ratio(self.config.particle_density, self.config.water_density)    # relative density ratio
        dstar = grain_properties.get('dimensionless_grain_size')  
        settling_velocity = grain_properties.get('settling_velocity')  # gives very similar results to physics_lib.compute_settling_velocity
        rouse_number = physics_lib.calculate_rouse_number(settling_velocity, max_shear_velocity, self.config.von_karman_constant)


        if self.config.use_transport_fields=='D3D':
            
            # bed load transport
            bed_load_transport_x = sedtrails_data.bed_load_transport['x']
            bed_load_transport_y = sedtrails_data.bed_load_transport['y']
            bed_load_transport_magnitude = sedtrails_data.bed_load_transport['magnitude']

            # suspended transport
            suspended_transport_x = sedtrails_data.suspended_transport['x']
            suspended_transport_y = sedtrails_data.suspended_transport['y']
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
            
            # suspended load transport fraction 
            den = suspended_transport_magnitude_calc + bed_load_transport_magnitude_calc
            qs_qt = np.full_like(suspended_transport_magnitude_calc, np.nan, dtype=float)
            mask = np.isfinite(suspended_transport_magnitude_calc) & np.isfinite(bed_load_transport_magnitude_calc) & (den > 0)
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
        suspended_velocity = PhysicsPlugin.calculate_macdonald_suspended_load_velocity(
            max_shear_velocity, 
            z_s, 
            k_s_total, 
            flow_velocity_magnitude, 
            max_suspended_velocity_factor=self.config.max_suspended_velocity_factor)
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
        u_c = qs_qt * suspended_velocity + (1 - qs_qt) * bed_load_velocity
        # total transport centroid elevation (MacDonald et al., 2006, equation 34)
        z_c = k_s_total * 10 ** (0.1739 * (u_c / max_shear_velocity) - 1.47826)
        
        # horizontal mean particle advection velocity (u_a) (MacDonald et al., 2006, equation 35)
        mean_particle_velocity = u_c #no need to calculate it again, previously PhysicsPlugin.calculate_mean_particle_advection_velocity(max_shear_velocity, z_c, k_s_total)
        
        # calculate x and y components of mean particle velocity
        mean_particle_velocity_x, mean_particle_velocity_y = physics_lib.compute_directions_from_magnitude(
            mean_particle_velocity,
            flow_velocity_x,
            flow_velocity_y,
            flow_velocity_magnitude,
        )
        # Note Vassia: we could consider getting particle velocity direction from bedload and suspended load velocities instead of flow velocity
        # In that way the effect of slopes which are considered in D3D might be included better. 


        # # ----------------------------------------------------------------------------------------
        # # MacDonald probabilistic bed-particle interaction model
        
        # # particle entrainment rate (van Rijn (1984b) pickup function, as used in MacDonald et al., 2006, equation 58)
        # q_pickup = (0.00033 * ((max_shields_number - critical_shields) / critical_shields)**1.5 * (
        #     ((s - 1) * self.config.gravity * self.config.grain_diameter**3) / self.config.viscosity**2) ** 0.1 
        #     * np.sqrt((s - 1) * self.config.gravity * self.config.grain_diameter)
        # )
        
        # # frequency of particle pickup (macdonald et al., 2006, equation 60)
        # freq_pickup = q_pickup / self.config.grain_diameter
        
        # # active layer depth (mixing depth) (macdonald et al., 2006, equation 63)
        # h_active = 5 * (max_shields_number - critical_shields) * self.config.grain_diameter
        
        # # mixing factor (macdonald et al., 2006, equation 64)
        # if h_active > self.config.grain_diameter:
        #     K_mixing = self.config.grain_diameter / h_active
        # else:
        #     K_mixing = 1.0
        
        # # burial factor (macdonald et al., 2006, equation 66)
        # h_burial = 0 # burial depth not implemented yet
        # K_burial = 1 - h_burial/h_active
        # K_burial = np.clip(K_burial, 0, 1)  # ensure between 0 and 1
        
        # # frequency of entrainment (macdonald et al., 2006, equation 57)
        # freq_entrainment = K_burial * K_mixing * freq_pickup
        
        # # ----------------------------------------------------------------------------------------    
        # # MacDonald Q3D mode
        
        # # mean particle fall time (macdonald et al., 2006, equation 36)
        # t_fall = z_c / settling_velocity
        
        # # mean particle wait time (macdonald et al., 2006, equation 37)
        # t_wait = 1 / freq_entrainment
        
        # # proportion of time particle is entrained in flow (macdonald et al., 2006, equation 38)
        # p_time_entrained = t_fall / t_wait
        
        # # velocity deficit coefficient (macdonald et al., 2006, equation 39)
        # # I DON'T THINK I'M IMPLEMENTING THE LOGIC CORRECTLY HERE- CHECK E.G. INDEXING
        # velocity_deficit_coeff = np.ones_like(mean_particle_velocity)
        # if p_time_entrained > 1:
        #     velocity_deficit_coeff = 1
        # else:
        #     velocity_deficit_coeff = p_time_entrained
            
        # # particle position quasi 3D
        # z_p = z_c # FOR NOW! THIS NEEDS TO BE IN PARTICLE PROPERTIES AND UPDATED DYNAMICALLY
        
        # # # adjusted mean particle velocity (macdonald et al., 2006, equation 40)
        # # if and(z_p > 0, z_p <= z_c):
        # #     mean_particle_velocity = velocity_deficit_coeff * 
        # #     u_zp = calculate_macdonald_particle_velocity_at_zp(max_shear_velocity, z_p, k_s)
        # # mean_particle_velocity
        
        # # particle deposition rate (macdonald et al., 2006, equation 67)
        # if z_p < k_s_skin/4:
        #     is_deposited = True
        # else:
        #     is_deposited = False
        # # NB. THIS SHOULD BE IN PARTICLE LOOP, NOT APPLIED TO FIELDS
        
        # # q3D entrainment
        # if random.uniform(0, 1) < freq_entrainment * self.config.time_step:
        #     is_entrained = True
        #     z_p = z_c # initially set particle height to centroid height
        # else:
        #     is_entrained = False
        #     z_p = z_bed # keep particle at bed level
        # # NB. THIS SHOULD BE IN PARTICLE LOOP, NOT APPLIED TO FIELDS

        # # if bed-particle interaction disabled, deposited particles instantaneously re-entrained to z_c if mobility M>1
        # # NB. THIS SHOULD BE IN PARTICLE LOOP, NOT APPLIED TO FIELDS
        
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
        sedtrails_data.add_physics_field('suspended_velocity_over_da_velocity', suspended_velocity/flow_velocity_magnitude) # this is for debugging purposes
        sedtrails_data.add_physics_field('30z_s_over_k_s_total', 30 * z_s / k_s_total) # this is for debugging purposes
        sedtrails_data.add_physics_field('max_shear_velocity', max_shear_velocity) # this is for debugging purposes
        sedtrails_data.add_physics_field('mean_shear_velocity', mean_shear_velocity) # this is for debugging purposes
        sedtrails_data.add_physics_field('suspended_load_velocity_lnpart', np.log(30 * z_s / k_s_total)) # this is for debugging purposes
        sedtrails_data.add_physics_field('particle_velocity_over_da_velocity', mean_particle_velocity/flow_velocity_magnitude) # this is for debugging purposes


        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'mean_particle_velocity',
            {'x': mean_particle_velocity_x, 'y': mean_particle_velocity_y, 'magnitude': mean_particle_velocity},
        )




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
    def calculate_macdonald_suspended_load_velocity(max_shear_velocity, z_s, k_s, flow_velocity_magnitude, max_suspended_velocity_factor=None):
        """Calculate suspended load velocity using MacDonald et al. (2006) Eq. 29"""
        suspended_load_velocity = np.zeros_like(z_s, dtype=float)

        # Log-law argument
        arg = 30.0 * z_s / k_s

        # Validity mask: above roughness sublayer and finite u*
        mask = (arg > 1.0) & np.isfinite(max_shear_velocity) & (max_shear_velocity > 0)

        # Apply log-law only where physically valid
        suspended_load_velocity[mask] = 2.5 * max_shear_velocity[mask] * np.log(arg[mask])
        
        # Elsewhere u_sus remains zero (no suspended-load advection)

        # Cap suspended load velocity to a fraction of flow velocity if specified
        if max_suspended_velocity_factor is not None:
            suspended_load_velocity = np.minimum(suspended_load_velocity, max_suspended_velocity_factor * flow_velocity_magnitude)

        return suspended_load_velocity

    
    @staticmethod
    def calculate_mean_particle_advection_velocity(max_shear_velocity, reference_height, total_roughness):
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

        mean_particle_velocity = np.full_like(
            max_shear_velocity, np.nan, dtype=float
        )

        arg = reference_height / total_roughness
        mask = (
            np.isfinite(arg)
            & np.isfinite(max_shear_velocity)
            & (total_roughness > 0)
            & (arg > 1)
        )

        mean_particle_velocity[mask] = (
            max_shear_velocity[mask]
            * (5.75 * np.log10(arg[mask]) + 8.5)
        )

        return mean_particle_velocity
