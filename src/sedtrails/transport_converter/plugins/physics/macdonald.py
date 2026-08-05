"""A plugin for MacDonald et al. (2006) sediment transport physics calculations."""

import os
import warnings

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
# from random import random
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter.plugins import BasePhysicsPlugin
from sedtrails.transport_converter import SedtrailsData


class PhysicsPlugin(BasePhysicsPlugin):  # all classes should be called the PhysicsPlugin
    """
    Plugin for MacDonald et al. (2006) sediment transport physics calculations.
    This plugin implements the physics calculations as described in MacDonald et al. (2006).

    MacDonald, N.J., Davies, M.H., Zundel, A.K., Howlett, J.D., Demirbilek, Z.,
    Gailani, J.Z., ... & Smith, J. (2006). PTM: particle tracking model.
    Report 1: Model theory, implementation, and example applications (No. ERDCCHLTR0620).
    """

    # Shared class-level cache for MacDonald lookup table
    # Loaded lazily on first use and shared across all plugin instances
    _macdonald_lookup_rouse = None
    _macdonald_lookup_values = None
    _macdonald_lookup_path = None
    _macdonald_warned_oob = False

    def __init__(self, config, tracer_methods):
        super().__init__()
        self.config = config
        self._q3d_divergence_cache_inputs = None
        self._q3d_divergence_cache_value = None

        # This plugin relies on shared class-level MacDonald lookup cache
        _ = PhysicsPlugin._macdonald_lookup_rouse
        _ = PhysicsPlugin._macdonald_lookup_values
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
        max_shields_number = physics_lib.compute_shields(
            max_bed_shear_stress,
            self.config.gravity,
            self.config.particle_density,
            self.config.water_density,
            self.config.grain_diameter,
        )

        # Current/mean shear drives profiles, advection, and diffusion. Maximum
        # shear is retained separately for mobilisation and resuspension.
        selected_bed_shear_stress = mean_bed_shear_stress
        selected_shear_velocity = mean_shear_velocity
        # water depth
        water_depth = sedtrails_data.water_depth

        # Compute transport velocities (these will have shape [time, spatial])
        critical_shields = grain_properties.get('critical_shields')
        s = physics_lib.calculate_relative_density_ratio(self.config.particle_density, self.config.water_density)    # relative density ratio
        dstar = grain_properties.get('dimensionless_grain_size')
        settling_velocity = grain_properties.get('settling_velocity')  # gives very similar results to physics_lib.compute_settling_velocity

        if critical_shields is None:
            raise ValueError("Missing required 'critical_shields' value in grain_properties.")

        k_s_skin = physics_lib.calculate_skin_roughness(method="soulsby_d50", d50=self.config.grain_diameter) #Consider here passing the background grain diameter instead of the particle grain diameter.
        k_s_skin_field = np.broadcast_to(np.asarray(k_s_skin, dtype=float), water_depth.shape).copy()

        if self.config.use_transport_fields == 'model-native':
            effective_chezy = self._get_effective_chezy_field(sedtrails_data, water_depth, default=65.0)
            k_s_chezy_equivalent = PhysicsPlugin.calculate_chezy_equivalent_roughness_height(
                water_depth,
                effective_chezy,
                von_karman_constant=self.config.von_karman_constant,
                gravity=self.config.gravity,
            )
            chezy_current_shear_velocity = PhysicsPlugin.safe_divide(
                flow_velocity_magnitude * np.sqrt(self.config.gravity),
                effective_chezy,
                fill=0.0,
            )
            selected_shear_velocity = chezy_current_shear_velocity
            selected_bed_shear_stress = (
                self.config.water_density * selected_shear_velocity**2
            )
            k_s_form = np.full_like(water_depth, np.nan, dtype=float)
            k_s_total = np.full_like(water_depth, np.nan, dtype=float)
            profile_roughness_height = k_s_chezy_equivalent
        else:
            # Bed roughness (MacDonald et al., 2006, equations 12-13)
            k_s_form = physics_lib.calculate_equilibrium_bedform_height(max_shields_number, critical_shields, self.config.grain_diameter, water_depth)
            # TO DO: Note that k_s_form is the equilibrium bedform height eta_b from MacDonald et al. (2006) Eq. 12 - we should implement the rate of change also (eq. 14-15) in future
            k_s_total = k_s_form + k_s_skin_field
            effective_chezy = np.full_like(water_depth, np.nan, dtype=float)
            k_s_chezy_equivalent = np.full_like(water_depth, np.nan, dtype=float)
            chezy_current_shear_velocity = np.full_like(water_depth, np.nan, dtype=float)
            profile_roughness_height = k_s_total

        rouse_number = physics_lib.calculate_rouse_number(
            settling_velocity,
            selected_shear_velocity,
            self.config.von_karman_constant,
        )

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
            qs_qt = np.full_like(selected_shear_velocity, np.nan, dtype=float)
            settling_velocity_field = np.broadcast_to(
                np.asarray(settling_velocity, dtype=float),
                selected_shear_velocity.shape,
            )
            mask = (
                (selected_shear_velocity > 0)
                & (settling_velocity_field > 0)
                & np.isfinite(selected_shear_velocity)
                & np.isfinite(settling_velocity_field)
            )
            qs_qt[mask] = 0.5 * np.tanh(
                1.3
                * np.log(
                    selected_shear_velocity[mask]
                    / settling_velocity_field[mask]
                )
                - 0.3
            ) + 0.5

        # suspended load height (MacDonald et al., 2006, equation 27)
        z_s = PhysicsPlugin.calculate_macdonald_susp_load_height(rouse_number, water_depth)

        # suspended load velocity (MacDonald et al., 2006, equation 29)
        # Note Vassia: here we sum skin and form roughness for total roughness - eq. 29 says k_s'' indicating bedform roughness
        suspended_velocity    = PhysicsPlugin.calculate_macdonald_loglaw_velocity_at_z(selected_shear_velocity, z_s,  profile_roughness_height)
        if self.config.max_suspended_velocity_factor is not None:
            suspended_velocity = PhysicsPlugin.cap_particle_velocity(
                suspended_velocity,
                flow_velocity_magnitude,
                self.config.max_suspended_velocity_factor,
            )
        #Question Vassia: should the max_suspended_velocity_factor be used here or in the final step - on the mean_particle_velocity?

        # bed load velocity (MacDonald et al., 2006, equation 30) - Engelund & Fredsoe (1976), same as Soulsby et al (2011)
        bed_load_velocity = physics_lib.compute_bed_load_velocity(
            max_shields_number,
            critical_shields,
            mean_shear_velocity,
        )

        # sediment advection velocity (MacDonald et al., 2006, equation 32)
        u_zc = qs_qt * suspended_velocity + (1 - qs_qt) * bed_load_velocity
        # total transport centroid elevation (MacDonald et al., 2006, equation 34)
        z_c = np.zeros_like(u_zc)
        mask = selected_shear_velocity > 0
        z_c[mask] = profile_roughness_height[mask] * 10 ** (0.1739 * (u_zc[mask] / selected_shear_velocity[mask]) - 1.47826)

        # Centroid particle velocity u_zc (MacDonald et al., 2006, equation 35).
        # In 2D this is the advecting particle velocity. In Q3D it is the reference
        # velocity field; the live advecting velocity is modified per particle.
        centroid_particle_velocity = u_zc

        centroid_particle_velocity_x, centroid_particle_velocity_y = physics_lib.compute_directions_from_magnitude(
            centroid_particle_velocity,
            flow_velocity_x,
            flow_velocity_y,
            flow_velocity_magnitude,
        )

        computation_type = str(getattr(self.config, 'computationType', '2D')).upper()
        if computation_type in {'2D', 'Q3D'}:
            # export_diagnostic_fields gates fields that nothing in the transport,
            # entrainment, or deposition pipeline reads back (verified by searching
            # for consumers of every field name added below). Default False keeps
            # memory use down on larger simulations. Everything registered outside
            # the "if export_diagnostics:" block below is a required input to the
            # 2D sampling logic (simulation_manager.py) or to
            # update_q3d_particle_position(), and is always computed and stored
            # regardless of this flag.
            export_diagnostics = bool(getattr(self.config, 'export_diagnostic_fields', False))

            mixing_layer_thickness = np.zeros_like(centroid_particle_velocity)

            sedtrails_data.add_physics_field('max_shields_number', max_shields_number)
            sedtrails_data.add_physics_field('mixing_layer_thickness', mixing_layer_thickness)
            sedtrails_data.add_physics_field('suspended_velocity', suspended_velocity)
            sedtrails_data.add_physics_field('suspended_transport_centroid_elevation', z_s)
            sedtrails_data.add_physics_field('total_transport_centroid_elevation', z_c)
            sedtrails_data.add_physics_field('rouse_number', rouse_number)
            sedtrails_data.add_physics_field('skin_roughness_height', k_s_skin_field)
            sedtrails_data.add_physics_field('profile_roughness_height', profile_roughness_height)
            sedtrails_data.add_physics_field('max_shear_velocity', max_shear_velocity)
            sedtrails_data.add_physics_field('selected_shear_velocity', selected_shear_velocity)
            sedtrails_data.add_physics_field(
                'centroid_particle_velocity',
                {
                    'x': centroid_particle_velocity_x,
                    'y': centroid_particle_velocity_y,
                    'magnitude': centroid_particle_velocity,
                },
            )

            if export_diagnostics:
                # MacDonald currently uses deterministic transport in the gridded
                # 2D workflow, so these probabilities are always 1.0.
                bed_load_probability = np.ones_like(bed_load_velocity, dtype=float)
                suspended_probability = np.ones_like(suspended_velocity, dtype=float)
                mean_particle_probability = np.ones_like(centroid_particle_velocity, dtype=float)
                suspended_centroid_over_depth = PhysicsPlugin.safe_divide(z_s, water_depth)
                total_centroid_over_depth = PhysicsPlugin.safe_divide(z_c, water_depth)
                shear_velocity_ratio = PhysicsPlugin.safe_divide(max_shear_velocity, mean_shear_velocity)
                suspended_velocity_over_flow = PhysicsPlugin.safe_divide(suspended_velocity, flow_velocity_magnitude)
                particle_velocity_over_flow = PhysicsPlugin.safe_divide(centroid_particle_velocity, flow_velocity_magnitude)
                log_law_argument = PhysicsPlugin.safe_divide(30 * z_s, profile_roughness_height)
                with np.errstate(divide='ignore', invalid='ignore'):
                    suspended_load_velocity_lnpart = np.where(
                        log_law_argument > 0,
                        np.log(log_law_argument),
                        np.nan,
                    )

                sedtrails_data.add_physics_field('bed_load_probability', bed_load_probability)
                sedtrails_data.add_physics_field('suspended_probability', suspended_probability)
                sedtrails_data.add_physics_field('mean_particle_probability', mean_particle_probability)
                # bed_load_velocity, k_s_form, k_s_total, effective_chezy,
                # k_s_chezy_equivalent, qs_qt, mean_shear_velocity, and
                # selected_bed_shear_stress are already computed above regardless
                # of this flag (they are required intermediates for
                # centroid_particle_velocity / profile_roughness_height /
                # selected_shear_velocity); only their extra, otherwise-unread
                # field registrations are gated here.
                sedtrails_data.add_physics_field('bedload_velocity', bed_load_velocity)
                sedtrails_data.add_physics_field('suspended_transport_centroid_elevation_over_depth', suspended_centroid_over_depth)
                sedtrails_data.add_physics_field('total_transport_centroid_elevation_over_depth', total_centroid_over_depth)
                sedtrails_data.add_physics_field('particle_advection_velocity', centroid_particle_velocity)
                sedtrails_data.add_physics_field('bedform_roughness_height', k_s_form)
                sedtrails_data.add_physics_field('macdonald_total_roughness_height', k_s_total)
                sedtrails_data.add_physics_field('effective_chezy_coefficient', effective_chezy)
                sedtrails_data.add_physics_field('chezy_current_shear_velocity', chezy_current_shear_velocity)
                sedtrails_data.add_physics_field('chezy_equivalent_roughness_height', k_s_chezy_equivalent)
                sedtrails_data.add_physics_field('shear_velocity_ratio', shear_velocity_ratio)
                sedtrails_data.add_physics_field('suspended_transport_ratio', qs_qt)
                sedtrails_data.add_physics_field('bed_load_transport_ratio', 1 - qs_qt)
                sedtrails_data.add_physics_field('suspended_velocity_over_da_velocity', suspended_velocity_over_flow)
                sedtrails_data.add_physics_field('30z_s_over_k_s_total', log_law_argument)
                sedtrails_data.add_physics_field('mean_shear_velocity', mean_shear_velocity)
                sedtrails_data.add_physics_field('selected_bed_shear_stress', selected_bed_shear_stress)
                sedtrails_data.add_physics_field('suspended_load_velocity_lnpart', suspended_load_velocity_lnpart)
                sedtrails_data.add_physics_field('particle_velocity_over_da_velocity', particle_velocity_over_flow)
                sedtrails_data.add_physics_field(
                    'mean_particle_velocity',
                    {
                        'x': centroid_particle_velocity_x,
                        'y': centroid_particle_velocity_y,
                        'magnitude': centroid_particle_velocity,
                    },
                )
            return

        raise ValueError(f"Unsupported MacDonald computationType: {getattr(self.config, 'computationType', None)!r}")

    def _calculate_entrainment_frequency(
        self,
        max_bed_shear_stress,
        critical_shields,
        timestep,
    ):
        """Calculate shared MacDonald turbulent entrainment fields.

        Returns turbulent shear stress, turbulent Shields number, active-layer
        thickness, and entrainment frequency. Both 2D and Q3D call this method
        so they use the same random shear realization and Eqs. 53-64.
        """
        max_bed_shear_stress = np.asarray(max_bed_shear_stress, dtype=float)
        s = physics_lib.calculate_relative_density_ratio(
            self.config.particle_density,
            self.config.water_density,
        )
        gamma = getattr(
            self.config,
            'entrainment_turbulence_gamma',
            getattr(self.config, 'q3d_turbulence_gamma', 0.005 * float(timestep)),
        )
        sigma_tau = gamma * max_bed_shear_stress
        turbulent_shear = np.random.normal(max_bed_shear_stress, sigma_tau)
        turbulent_shear = np.clip(turbulent_shear, a_min=0.0, a_max=None)
        turbulent_shields = PhysicsPlugin.safe_divide(
            turbulent_shear,
            self.config.water_density
            * self.config.gravity
            * self.config.grain_diameter
            * (s - 1),
        )

        q_pickup = np.zeros_like(turbulent_shields, dtype=float)
        valid_pickup = (
            np.isfinite(turbulent_shields)
            & np.isfinite(critical_shields)
            & (critical_shields > 0)
            & (turbulent_shields > critical_shields)
        )
        theta_excess = PhysicsPlugin.safe_divide(
            turbulent_shields[valid_pickup] - critical_shields,
            critical_shields,
        )
        q_pickup[valid_pickup] = (
            0.00033
            * theta_excess**1.5
            * (
                ((s - 1) * self.config.gravity * self.config.grain_diameter**3)
                / self.config.kinematic_viscosity**2
            ) ** 0.1
            * np.sqrt((s - 1) * self.config.gravity * self.config.grain_diameter)
        )
        pickup_frequency = q_pickup / self.config.grain_diameter

        active_layer_thickness = np.zeros_like(turbulent_shields, dtype=float)
        active_layer_thickness[valid_pickup] = (
            5
            * (turbulent_shields[valid_pickup] - critical_shields)
            * self.config.grain_diameter
        )
        mixing_factor = np.where(
            active_layer_thickness > self.config.grain_diameter,
            PhysicsPlugin.safe_divide(
                self.config.grain_diameter,
                active_layer_thickness,
                fill=1.0,
            ),
            1.0,
        )
        # Burial depth is handled by particle status eligibility. The grid-level
        # MacDonald burial factor remains one wherever an active layer exists.
        burial_factor = np.where(active_layer_thickness > 0.0, 1.0, 0.0)
        entrainment_frequency = burial_factor * mixing_factor * pickup_frequency
        return (
            turbulent_shear,
            turbulent_shields,
            active_layer_thickness,
            entrainment_frequency,
        )

    def _add_2d_entrainment_frequency(
        self,
        sedtrails_data: SedtrailsData,
        grain_properties: dict[str, float],
        timestep: float,
    ) -> None:
        critical_shields = grain_properties.get('critical_shields')
        if critical_shields is None:
            raise ValueError("Missing required 'critical_shields' value in grain_properties.")
        _, _, _, entrainment_frequency = self._calculate_entrainment_frequency(
            sedtrails_data.max_bed_shear_stress,
            critical_shields,
            timestep,
        )
        sedtrails_data.add_physics_field(
            'macdonald_entrainment_frequency',
            entrainment_frequency,
        )

    def add_timestep_physics(
        self,
        sedtrails_data: SedtrailsData,
        grain_properties: dict[str, float],
        current_timestep: float,
    ) -> None:
        """Add Q3D fields that depend on the final particle timestep.

        MacDonald (2006, ERDC/CHL TR-06-20) uses ``dt`` explicitly in the
        turbulent shear implementation (Eq. 54), random-walk diffusion
        velocities (Eqs. 51-52), and probabilistic re-entrainment test (Eq. 68).
        These fields are therefore added after any CFL update has selected the actual
        particle timestep.
        """

        computation_type = str(getattr(self.config, 'computationType', '2D')).upper()
        self.config.current_timestep = float(current_timestep)
        if computation_type == 'Q3D':
            self._add_q3d_timestep_physics(sedtrails_data, grain_properties)
            return

        entrainment_config = getattr(self.config, 'entrainment', {}) or {}
        entrainment_method = str(
            entrainment_config.get('method', 'shields_threshold')
        ).lower().replace('-', '_')
        if computation_type == '2D' and entrainment_method == 'entrainment_frequency':
            self._add_2d_entrainment_frequency(
                sedtrails_data,
                grain_properties,
                float(current_timestep),
            )

    def _add_q3d_timestep_physics(self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float]) -> None:
        """Add MacDonald Q3D fields that depend on the current particle timestep.

        References are to MacDonald (2006), PTM Report 1, Chapter 3:
        fall time and velocity deficit (Eqs. 36-39), Q3D vertical velocity
        (Eqs. 41-42), turbulent diffusion/random walk (Eqs. 45, 49, 51-52),
        turbulent bed shear (Eqs. 53-54), and particle-bed entrainment
        frequency (Eqs. 57-66).
        """

        timestep = float(getattr(self.config, 'current_timestep', 60.0))
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError(f'current_timestep must be positive and finite, got {timestep!r}')

        critical_shields = grain_properties.get('critical_shields')
        settling_velocity = grain_properties.get('settling_velocity')
        if critical_shields is None:
            raise ValueError("Missing required 'critical_shields' value in grain_properties.")
        if settling_velocity is None:
            raise ValueError("Missing required 'settling_velocity' value in grain_properties.")

        water_depth = sedtrails_data.water_depth
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']
        selected_shear_velocity = sedtrails_data.selected_shear_velocity
        z_c = sedtrails_data.total_transport_centroid_elevation
        centroid_particle_velocity = sedtrails_data.centroid_particle_velocity['magnitude']
        max_bed_shear_stress = sedtrails_data.max_bed_shear_stress
        # Use the same turbulent shear, Shields, active-layer, and entrainment-
        # frequency calculation as MacDonald 2D. This produces one internally
        # consistent random shear realization for all four Q3D fields.
        (
            turbulent_shear,
            turbulent_shields,
            h_active,
            freq_entrainment,
        ) = self._calculate_entrainment_frequency(
            max_bed_shear_stress,
            critical_shields,
            timestep,
        )
        # Mean fall time from centroid elevation is t_f = z_c / w_s (Eq. 36).
        t_fall = PhysicsPlugin.safe_divide(z_c, settling_velocity, fill=0.0)
        # The proportion of time that a particle would be expected to be entrained
        # is t_f * f_e (Eq. 38),
        # used as a theshold for the velocity deficit coefficient Delta_c (Eq. 39).
        p_time_entrained = np.clip(t_fall * freq_entrainment, 0.0, 1.0)
        velocity_deficit_coeff = np.ones_like(centroid_particle_velocity)
        velocity_deficit_coeff = np.where(
            p_time_entrained > 1.0,
            1.0,
            p_time_entrained,
        )

        # Grid entrainment/re-entrainment height used to lift newly entrained
        # particles. This is not the live particle z_p; update_q3d_particle_position
        # recomputes z_p from each particle's z and local bed level every substep.
        z_entrainment = np.clip(
            np.nan_to_num(z_c, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            np.maximum(water_depth, 0.0),
        )

        # Optional grid diagnostics expose intermediate Eulerian approximations.
        # The live particle-height-dependent values are recomputed later at each
        # particle's actual z_p in update_q3d_particle_position. Horizontal/vertical
        # diffusivity follow Eqs. 45 and 49; random-walk velocities depend on dt
        # in Eqs. 51-52. Shares the export_diagnostic_fields flag with add_physics()
        # (formerly a separate q3d_export_grid_diagnostics flag).
        export_q3d_grid_diagnostics = bool(getattr(self.config, 'export_diagnostic_fields', False))
        if export_q3d_grid_diagnostics:
            horizontal_diffusion_enabled = bool(
                getattr(self.config, 'q3d_horizontal_diffusion_enabled', True)
            )
            vertical_diffusion_enabled = (
                str(getattr(self.config, 'q3d_vertical_update_scheme', 'geometric'))
                .strip()
                .lower()
                .replace('-', '_')
                == 'geometric'
            )
            E_turb_hor, E_turb_vert = PhysicsPlugin.compute_turbulent_diffusion_coefficients(
                water_depth=water_depth,
                z_p=z_entrainment,
                flow_velocity_magnitude=flow_velocity_magnitude,
                shear_velocity=selected_shear_velocity,
                K_Et=getattr(self.config, 'q3d_horizontal_diffusion_factor', 0.15),
                K_Ev=getattr(self.config, 'q3d_vertical_diffusion_factor', 0.15),
                compute_horizontal=horizontal_diffusion_enabled,
                compute_vertical=vertical_diffusion_enabled,
            )
            u_Dx, u_Dy, w_D = PhysicsPlugin.compute_random_walk_diffusion_velocities(
                E_turb_hor=E_turb_hor,
                E_turb_vert=E_turb_vert,
                dt=timestep,
                compute_horizontal=horizontal_diffusion_enabled,
                compute_vertical=vertical_diffusion_enabled,
            )

        # For 2D hydrodynamic input, PTM estimates vertical flow velocity from
        # continuity (MacDonald 2006, Eq. 42), after the Q3D vertical advection
        # concept in Eq. 41. Here divU supplies the horizontal divergence term.
        divU, dudx, dvdy = self._get_q3d_divergence(
            sedtrails_data.x,
            sedtrails_data.y,
            flow_velocity_x,
            flow_velocity_y,
            k=12,
            r_max=150,
        )
        # This is the waterlevel gradient term in Eq. 42
        dh_dt = PhysicsPlugin.compute_dh_dt(
            water_depth,
            sedtrails_data.bed_level,
            sedtrails_data.times,
        )

        # local vertical flow velocity (Macdonald 2006, Eq. 42) at the grid level, not the live particle z_p.
        w_zp = np.zeros_like(water_depth, dtype=float)
        wet = water_depth > 0
        q3d_vertical_velocity_gradient = PhysicsPlugin._combine_q3d_vertical_velocity_gradient(
            dh_dt,
            water_depth,
            divU,
        )

        if export_q3d_grid_diagnostics:
            #the local vertical flow velocity at the grid level is used to compute the vertical particle velocity at the grid level
            w_zp[wet] = q3d_vertical_velocity_gradient[wet] * (water_depth[wet] - z_entrainment[wet])
            w_zp = np.nan_to_num(w_zp, nan=0.0, posinf=0.0, neginf=0.0)

            settling_velocity_field = np.full_like(w_zp, settling_velocity, dtype=float)
            #vertical particle velocity at the grid level (this is based on Eq.20
            # from https://doer.el.erdc.dren.mil/pdf/LackeyandMcDonald_2007.pdf)
            vertical_particle_velocity = w_zp - settling_velocity_field + w_D
            vertical_particle_velocity = np.nan_to_num(
                vertical_particle_velocity,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            vertical_particle_velocity[~wet] = 0.0

        # Required grid inputs for particle-resolved Q3D motion and entrainment.
        # These are interpolated to particle positions in update_q3d_particle_position.
        sedtrails_data.add_physics_field('q3d_fall_time', t_fall)
        sedtrails_data.add_physics_field('turbulent_shear_stress', turbulent_shear)
        sedtrails_data.add_physics_field('active_layer_thickness', h_active)
        sedtrails_data.add_physics_field('q3d_entrainment_height_above_bed', z_entrainment)
        sedtrails_data.add_physics_field('macdonald_entrainment_frequency', freq_entrainment)
        sedtrails_data.add_physics_field('q3d_velocity_deficit_coefficient', velocity_deficit_coeff)
        sedtrails_data.add_physics_field('turbulent_shields_number', turbulent_shields)
        sedtrails_data.add_physics_field('q3d_vertical_velocity_gradient', q3d_vertical_velocity_gradient)

        if export_q3d_grid_diagnostics:
            # These diagnostic fields are Eulerian/grid approximations evaluated at
            # z_entrainment = z_c. They are useful for checking the MacDonald formulas,
            # but they should not drive particle motion. The particle-local equivalents
            # are recomputed in update_q3d_particle_position using each particle's live z_p.
            sedtrails_data.add_physics_field('diffusive_velocity_x', u_Dx)
            sedtrails_data.add_physics_field('diffusive_velocity_y', u_Dy)
            sedtrails_data.add_physics_field('diffusive_velocity_z', w_D)
            sedtrails_data.add_physics_field('turbulent_diffusion_coefficient_horizontal', E_turb_hor)
            sedtrails_data.add_physics_field('turbulent_diffusion_coefficient_vertical', E_turb_vert)
            sedtrails_data.add_physics_field('vertical_advection_velocity', w_zp)
            sedtrails_data.add_physics_field('vertical_particle_velocity', vertical_particle_velocity)

    def _get_q3d_divergence(self, x, y, u, v, k=12, r_max=150):
        """Return cached Q3D horizontal divergence for one input-data chunk."""
        cached_inputs = self._q3d_divergence_cache_inputs
        current_inputs = (x, y, u, v, k, r_max)
        if cached_inputs is not None:
            same_arrays = all(
                cached is current
                for cached, current in zip(
                    cached_inputs[:4],
                    current_inputs[:4],
                    strict=True,
                )
            )
            if same_arrays and cached_inputs[4:] == current_inputs[4:]:
                return self._q3d_divergence_cache_value

        result = PhysicsPlugin.divergence_scattered_knn_time(
            x,
            y,
            u,
            v,
            k=k,
            r_max=r_max,
        )
        self._q3d_divergence_cache_inputs = current_inputs
        self._q3d_divergence_cache_value = result
        return result

    @staticmethod

    def compute_dh_dt(water_depth, bed_level, times):
        """
        Return the water-surface time derivative using backward differences.

        Parameters
        ----------
        water_depth : array-like
            Water depth with time on the first axis [m].
        bed_level : array-like
            Bed elevation, either static or with time on the first axis [m].
        times : array-like or float
            Eulerian input-frame times [s]. A scalar interval is accepted for
            compatibility, but Q3D runtime calculations pass the actual times.

        Returns
        -------
        numpy.ndarray
            Water-surface elevation derivative [m/s].
        """
        water_depth = np.asarray(water_depth, dtype=float)
        bed_level = np.asarray(bed_level, dtype=float)
        if bed_level.ndim == water_depth.ndim - 1:
            bed_level = np.broadcast_to(bed_level, water_depth.shape)
        else:
            bed_level = np.broadcast_to(bed_level, water_depth.shape)

        dh_dt = np.zeros_like(water_depth, dtype=float)
        if water_depth.shape[0] < 2:
            return dh_dt

        time_values = np.asarray(times, dtype=float)
        if time_values.ndim == 0:
            time_deltas = np.full(water_depth.shape[0] - 1, float(time_values))
        elif time_values.ndim == 1 and time_values.size == water_depth.shape[0]:
            time_deltas = np.diff(time_values)
        else:
            raise ValueError(
                'times must be a scalar interval or a 1D array matching the '
                'water-depth time dimension.'
            )
        if np.any(~np.isfinite(time_deltas)) or np.any(time_deltas <= 0.0):
            raise ValueError('Eulerian input-frame times must increase monotonically.')

        water_surface = water_depth + bed_level
        delta_shape = (time_deltas.size,) + (1,) * (water_depth.ndim - 1)
        dh_dt[1:] = (
            water_surface[1:] - water_surface[:-1]
        ) / time_deltas.reshape(delta_shape)
        dh_dt[0] = dh_dt[1]  # reasonable fill for first timestep

        wet = water_depth != 0
        dh_dt[~wet] = np.nan  # set dh/dt to NaN where water depth is zero (dry areas)
        return dh_dt

    @staticmethod
    def _combine_q3d_vertical_velocity_gradient(dh_dt, water_depth, divergence):
        """Combine surface change and finite horizontal divergence terms."""
        depth_change_over_depth = PhysicsPlugin.safe_divide(dh_dt, water_depth, fill=0.0)
        finite_divergence = np.where(np.isfinite(divergence), divergence, 0.0)
        return depth_change_over_depth + finite_divergence


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

        stencils = PhysicsPlugin._build_divergence_stencils(
            x,
            y,
            k=k,
            r_max=r_max,
        )
        return PhysicsPlugin._apply_divergence_stencils(u, v, stencils)

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

        x = np.asarray(x)
        y = np.asarray(y)
        u = np.asarray(u)
        v = np.asarray(v)
        if u.ndim != 1 or v.ndim != 1:
            raise ValueError('u and v must be 1D arrays')
        if u.shape != x.shape or v.shape != x.shape or y.shape != x.shape:
            raise ValueError('x, y, u, and v must have matching shapes')

        stencils = PhysicsPlugin._build_divergence_stencils(
            x,
            y,
            k=k,
            r_max=r_max,
            eps=eps,
        )
        div_u, dudx, dvdy = PhysicsPlugin._apply_divergence_stencils(
            u[np.newaxis, :],
            v[np.newaxis, :],
            stencils,
        )
        return div_u[0], dudx[0], dvdy[0]

    @staticmethod
    def _build_divergence_stencils(x, y, k=12, r_max=None, eps=1e-12):
        """Build reusable local derivative weights for a scattered grid."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError('x and y must be matching 1D arrays')

        npoints = x.size
        stencils = [None] * npoints
        if npoints < 4:
            return stencils

        coords = np.column_stack((x, y))
        tree = cKDTree(coords)
        query_k = min(k + 1, npoints)
        dists, indices = tree.query(coords, k=query_k)
        if query_k == 1:
            dists = dists[:, np.newaxis]
            indices = indices[:, np.newaxis]

        for point_index in range(npoints):
            neighbours = np.asarray(indices[point_index, 1:], dtype=int)
            distances = np.asarray(dists[point_index, 1:], dtype=float)
            if r_max is not None:
                inside_radius = distances <= r_max
                neighbours = neighbours[inside_radius]
                distances = distances[inside_radius]
            if neighbours.size < 3:
                continue

            dx = x[neighbours] - x[point_index]
            dy = y[neighbours] - y[point_index]
            root_weights = np.sqrt(1.0 / (distances + eps))
            design = np.column_stack((np.ones_like(dx), dx, dy))
            weighted_design = design * root_weights[:, np.newaxis]
            coefficient_map = np.linalg.pinv(weighted_design) * root_weights
            stencils[point_index] = (
                neighbours,
                coefficient_map[1],
                coefficient_map[2],
            )
        return stencils

    @staticmethod
    def _apply_divergence_stencils(u, v, stencils):
        """Apply scattered-grid derivative weights to all input time slices."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if u.ndim != 2 or v.ndim != 2 or u.shape != v.shape:
            raise ValueError('u and v must be matching 2D arrays')
        if u.shape[1] != len(stencils):
            raise ValueError('Velocity fields and divergence stencils do not match')

        ntimes, npoints = u.shape
        dudx = np.full((ntimes, npoints), np.nan)
        dvdy = np.full((ntimes, npoints), np.nan)
        for point_index, stencil in enumerate(stencils):
            if stencil is None:
                continue
            neighbours, dudx_weights, dvdy_weights = stencil
            dudx[:, point_index] = u[:, neighbours] @ dudx_weights
            dvdy[:, point_index] = v[:, neighbours] @ dvdy_weights
        return dudx + dvdy, dudx, dvdy


    @staticmethod

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

        z_p = np.asarray(z_p, dtype=float)
        z_c = np.asarray(z_c, dtype=float)
        z_c_safe = np.maximum(z_c, 1e-12)

        # Linear blending term for transition region
        blending = (z_p - z_c_safe) / (0.4 * z_c_safe)

        # Transition-zone velocity (z_c < z_p <= 1.4 z_c)
        u_transition = (
            c_A * u_zc
            + blending * (u_1p4zc - c_A * u_zc)
        )

        # Piecewise definition of reduced velocity
        u_particle = np.where(
            z_p <= 0.0,
            0.0,
            np.where(
                z_p <= z_c_safe,
                c_A * u_zp,
                np.where(
                    z_p <= 1.4 * z_c_safe,
                    u_transition,
                    u_zp,
                ),
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
        compute_horizontal=True,
        compute_vertical=True,
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
            Dimensionless vertical diffusion scaling coefficient. If None,
            defaults to K_Et. Water depth supplies the length scale.
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
        E_turb_hor = np.zeros_like(water_depth, dtype=float)
        if compute_horizontal:
            E_turb_hor = M_b * K_Et * water_depth * shear_velocity
            E_turb_hor = np.nan_to_num(E_turb_hor, nan=0.0)
            E_turb_hor = np.maximum(E_turb_hor, E_turb_hor_min)

        # ---------------------
        # Depth-scaled variant of equations (49) and (50). In the report K_Ev
        # carries length units; here K_Ev is dimensionless and h supplies that
        # length explicitly.
        # ---------------------
        E_turb_vert = np.zeros_like(water_depth, dtype=float)
        if compute_vertical:
            shape = np.zeros_like(water_depth)
            valid = water_depth > 0

            shape[valid] = (
                z_p[valid]
                * (water_depth[valid] - z_p[valid]) ** 2
                / water_depth[valid] ** 3
            )

            E_turb_vert = M_b * K_Ev * water_depth * flow_velocity_magnitude * shape
            E_turb_vert = np.nan_to_num(E_turb_vert, nan=0.0)
            E_turb_vert = np.maximum(E_turb_vert, E_turb_vert_min)

        return E_turb_hor, E_turb_vert


    @staticmethod

    def compute_random_walk_diffusion_velocities(
        E_turb_hor,
        E_turb_vert,
        dt,
        rng=np.random,  # default gives uniformly distributed random numbers in [0,1)
        *,
        compute_horizontal=True,
        compute_vertical=True,
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

        u_Dx = np.zeros_like(E_turb_hor, dtype=float)
        u_Dy = np.zeros_like(E_turb_hor, dtype=float)
        if compute_horizontal:
            Pi_x = rng.random(E_turb_hor.shape)
            Pi_y = rng.random(E_turb_hor.shape)
            scale_h = np.sqrt(6.0 * E_turb_hor / dt)
            u_Dx = 2.0 * (Pi_x - 0.5) * scale_h
            u_Dy = 2.0 * (Pi_y - 0.5) * scale_h

        w_D = np.zeros_like(E_turb_vert, dtype=float)
        if compute_vertical:
            Pi_z = rng.random(E_turb_vert.shape)
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
        - Emits a RuntimeWarning if invalid / out-of-range inputs occur (once).

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
        if PhysicsPlugin._macdonald_lookup_rouse is None or PhysicsPlugin._macdonald_lookup_values is None:
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

            lookup_rouse = np.asarray(da['rouse'].values, dtype=float)
            lookup_values = np.asarray(da.values, dtype=float)
            if (
                lookup_rouse.size < 2
                or not np.all(np.isfinite(lookup_rouse))
                or not np.all(np.isfinite(lookup_values))
                or not np.all(np.diff(lookup_rouse) > 0.0)
            ):
                raise ValueError('MacDonald lookup coordinates and values must be finite and strictly increasing.')

            PhysicsPlugin._macdonald_lookup_rouse = lookup_rouse
            PhysicsPlugin._macdonald_lookup_values = lookup_values
            PhysicsPlugin._macdonald_lookup_path = lookup_path

        lookup_rouse = PhysicsPlugin._macdonald_lookup_rouse
        lookup_values = PhysicsPlugin._macdonald_lookup_values

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

        rmin = float(lookup_rouse[0])
        rmax = float(lookup_rouse[-1])

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
            z_over_h[interp_ok] = np.interp(r[interp_ok], lookup_rouse, lookup_values)

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
                warnings.warn(
                    "[MacDonald lookup] Some inputs were invalid or out of lookup range.\n"
                    f"  invalid (NaN/Inf/rouse<=0/depth<0): {n_invalid}/{n_tot}\n"
                    f"  rouse < rmin ({rmin:g}) -> NaN: {n_low}/{n_tot}\n"
                    f"  rouse >= {R_ASYM:g} -> asymptote z_s/H={ASYM:.6g}: {n_high}/{n_tot}\n"
                    f"  lookup rouse range: [{rmin:g}, {rmax:g}] from {PhysicsPlugin._macdonald_lookup_path}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # ----------------------------
        # Convert to z_s (meters)
        # ----------------------------
        z_s = z_over_h * h

        return float(z_s) if scalar_out else z_s


    @staticmethod

    def _get_effective_chezy_field(sedtrails_data: SedtrailsData, water_depth: np.ndarray, default: float = 65.0) -> np.ndarray:
        """Return effective Chezy coefficient C_eff, falling back to a constant field."""
        for name in ('effective_chezy', 'effective_chezy_coefficient', 'chezy_coefficient'):
            if hasattr(sedtrails_data, name):
                chezy = getattr(sedtrails_data, name)
                break
        else:
            chezy = default

        chezy = np.asarray(chezy, dtype=float)
        if chezy.shape == ():
            return np.full_like(water_depth, float(chezy), dtype=float)
        return np.broadcast_to(chezy, water_depth.shape).copy()

    @staticmethod

    def calculate_chezy_equivalent_roughness_height(
        water_depth: np.ndarray,
        chezy_coefficient: np.ndarray,
        von_karman_constant: float = 0.4,
        gravity: float = 9.81,
    ) -> np.ndarray:
        """Convert Chezy coefficient to depth-average-consistent roughness.

        The ``30/e`` prefactor makes the depth integral of the matching log
        profile equal the supplied Chezy depth-averaged velocity.
        """
        water_depth = np.asarray(water_depth, dtype=float)
        chezy_coefficient = np.asarray(chezy_coefficient, dtype=float)
        water_depth, chezy_coefficient = np.broadcast_arrays(water_depth, chezy_coefficient)
        roughness = np.zeros_like(water_depth, dtype=float)
        valid = (
            np.isfinite(water_depth)
            & np.isfinite(chezy_coefficient)
            & (water_depth > 0.0)
            & (chezy_coefficient > 0.0)
            & (gravity > 0.0)
        )
        roughness[valid] = (30.0 / np.e) * water_depth[valid] * np.exp(
            -float(von_karman_constant) * chezy_coefficient[valid] / np.sqrt(float(gravity))
        )
        return roughness

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

        return np.minimum(velocity, max_velocity_factor * flow_velocity_magnitude)

    @staticmethod

    def safe_divide(numerator, denominator, fill=np.nan):
        """Divide arrays while returning `fill` where the denominator is zero or invalid."""

        numerator = np.asarray(numerator, dtype=float)
        denominator = np.asarray(denominator, dtype=float)
        numerator, denominator = np.broadcast_arrays(numerator, denominator)
        result = np.full_like(numerator, fill, dtype=float)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        np.divide(numerator, denominator, out=result, where=valid)
        return result

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
