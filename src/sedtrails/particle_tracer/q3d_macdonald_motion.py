"""
Q3D and MacDonald 2D particle motion/state methods.

This module holds the MacDonald-specific ParticlePopulation behaviour: the
Q3D (quasi-3D) particle-resolved vertical motion update, and the MacDonald
2D suspended/deposited state machine (entrainment and deposition). It is
mixed into ParticlePopulation (see particle_tracer/particle_seeder.py) via
Q3DMacdonaldMotionMixin so that generic particle-tracking behaviour (seeding,
status flags, 2D advection shared by all tracer methods) stays in
particle_seeder.py while this MacDonald-only logic lives separately.

Methods here are unchanged from their prior location in particle_seeder.py;
this is a file-organization split, not a behavioral change. They still
access ParticlePopulation instance state (self.particles, self._particle_simplices,
self.grid_geometry, etc.) exactly as before, via the normal Python method
resolution order once mixed in.
"""

from typing import Any, Dict

import numpy as np


def _safe_divide(numerator, denominator, fill=0.0):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), fill, dtype=float)
    return np.divide(numerator, denominator, out=out, where=denominator != 0.0)


class Q3DMacdonaldMotionMixin:
    """Q3D and MacDonald 2D motion/state methods, mixed into ParticlePopulation.

    Not usable standalone: methods here assume the host class provides
    self.particles, self._particle_simplices, self.grid_geometry, and the
    other ParticlePopulation instance state and helper methods they call.
    """

    @staticmethod

    def _loglaw_velocity_at_z(shear_velocity, z, roughness_height):
        shear_velocity = np.asarray(shear_velocity, dtype=float)
        z = np.asarray(z, dtype=float)
        roughness_height = np.asarray(roughness_height, dtype=float)
        velocity = np.zeros_like(z, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            argument = 30.0 * z / roughness_height
            valid = (
                (argument > 1.0)
                & np.isfinite(argument)
                & np.isfinite(shear_velocity)
                & (shear_velocity > 0.0)
            )
            velocity[valid] = 2.5 * shear_velocity[valid] * np.log(argument[valid])
        return velocity

    @staticmethod

    def _apply_q3d_velocity_deficit(u_zp, u_1p4zc, z_p, z_c, deficit_coefficient):
        """
        Apply the Q3-D horizontal velocity deficit formulation of
        MacDonald et al. (2006), Eq. (40), at particle positions.

        This function reduces horizontal particle advection velocity
        to account for intermittent particle-bed interaction in Q3-D mode.
        The reduction depends on particle elevation relative to the
        transport centroid and on the velocity deficit coefficient c_A
        (Eq. 39).

        Parameters
        ----------
        u_zp : array
            Log-law horizontal velocity u(z_p) at the current particle height [m/s].
        u_1p4zc : array
            Log-law horizontal velocity u(1.4 z_c) at 1.4 times the transport centroid [m/s].
        z_p : array
            Particle height above bed [m].
        z_c : array
            Total-load transport centroid height above bed [m].
        deficit_coefficient : array
            Velocity deficit coefficient c_A [-].

        Returns
        -------
        numpy.ndarray
            Reduced horizontal particle advection velocity [m/s]. The returned velocity equals:
            - 0 at or below the bed,
            - c_A * u(z_p) below the centroid,
            - a linear blend between c_A * u(z_p) and u(1.4 z_c) in the transition zone,
            - u(z_p) above 1.4 z_c.
        """

        z_p = np.asarray(z_p, dtype=float)
        z_c_safe = np.maximum(np.asarray(z_c, dtype=float), 1e-12)
        blending = (z_p - z_c_safe) / (0.4 * z_c_safe)
        u_transition = deficit_coefficient * u_zp + blending * (u_1p4zc - deficit_coefficient * u_zp)
        return np.where(
            z_p <= 0.0,
            0.0,
            np.where(
                z_p <= z_c_safe,
                deficit_coefficient * u_zp,
                np.where(z_p <= 1.4 * z_c_safe, u_transition, u_zp),
            ),
        )

    @staticmethod
    def _flow_component_field(flow_field, component):
        """Return one static or temporally bounded flow-field component."""
        if all(key in flow_field for key in ('lower', 'upper', 'weight')):
            return {
                'lower': flow_field['lower'][component],
                'upper': flow_field['upper'][component],
                'weight': flow_field['weight'],
            }
        return flow_field[component]

    @staticmethod

    def _turbulent_diffusion_coefficients(
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
        Compute particle-level turbulent diffusion coefficients following MacDonald et al. (2006) equations 45 to 50.

        Parameters
        ----------
        water_depth : array
            Total water depth h [m] interpolated at particle positions.
        z_p : array
            Particle height above bed z_p [m].
        flow_velocity_magnitude : array
            Depth-averaged flow velocity magnitude |U| [m/s] interpolated at particle positions.
        shear_velocity : array
            Shear velocity u_* [m/s] interpolated at particle positions.
        K_Et : float
            Horizontal diffusion scaling coefficient, typically about 0.15-0.6 (below eq 45 in MacDonald et al. (2006)).
        K_Ev : float or None
            Dimensionless vertical diffusion scaling coefficient. If None,
            defaults to K_Et. Water depth supplies the length scale.
        M_b : array or None
            Wave-breaking enhancement factor at particle positions. If None, defaults to 1. (eq 47 in MacDonald et al. (2006)).
        E_turb_hor_min : float
            Minimum horizontal turbulent diffusivity [m^2/s]. Default is 0.02 in PTM below eq 48 in MacDonald et al. (2006).
        E_turb_vert_min : float
            Minimum vertical turbulent diffusivity [m^2/s]. Default is 0 in PTM (below eq 49 in MacDonald et al. (2006)).
        compute_horizontal : bool
            If false, skip the horizontal coefficient calculation and return zeros.
        compute_vertical : bool
            If false, skip the vertical coefficient calculation and return zeros.

        Returns
        -------
        tuple of arrays
            Horizontal and vertical turbulent diffusion coefficients [m^2/s].
        """
        if K_Ev is None:
            K_Ev = K_Et

        water_depth = np.asarray(water_depth, dtype=float)
        z_p = np.asarray(z_p, dtype=float)
        flow_velocity_magnitude = np.asarray(flow_velocity_magnitude, dtype=float)
        shear_velocity = np.asarray(shear_velocity, dtype=float)
        if M_b is None:
            M_b = np.ones_like(flow_velocity_magnitude)
        else:
            M_b = np.asarray(M_b, dtype=float)
            # TO DO Vassia: adjust this according to eq 47 in MacDonald et al. (2006) if needed, as the current implementation assumes M_b is directly provided at particle positions.
            # Mb = 1.0 + 5 * H_s within the breaker zone and 1 outside the breaker zone, where H_s is the significant wave height. If wave information is not available, M_b can be set to 1 for all particles as a default.

        # Horizontal diffusion coefficient, equations 46 and 48 in MacDonald et al. (2006).
        horizontal = np.zeros_like(water_depth, dtype=float)
        if compute_horizontal:
            horizontal = M_b * K_Et * water_depth * shear_velocity
            horizontal = np.maximum(np.nan_to_num(horizontal, nan=0.0), E_turb_hor_min)

        # Depth-scaled variant of equations 49 and 50. In the report K_Ev
        # carries length units; here K_Ev is dimensionless and h supplies that
        # length explicitly.
        vertical = np.zeros_like(water_depth, dtype=float)
        if compute_vertical:
            shape = np.zeros_like(water_depth, dtype=float)
            valid = water_depth > 0.0
            shape[valid] = z_p[valid] * (water_depth[valid] - z_p[valid]) ** 2 / water_depth[valid] ** 3
            vertical = M_b * K_Ev * water_depth * flow_velocity_magnitude * shape
            vertical = np.maximum(np.nan_to_num(vertical, nan=0.0), E_turb_vert_min)
        return horizontal, vertical

    def initialize_macdonald_2d_release_state(self) -> None:
        """Resolve the logical initial bed/water-column state for MacDonald 2D.

        MacDonald 2D has no particle z coordinate, so `seeding.vertical_position`
        is collapsed to deposited/suspended state. Burial eligibility remains controlled
        by the transport-probability method. Only newly released
        particles inside the domain are initialized; Van Westen never calls this
        method and retains its existing burial-depth behavior.
        """
        n_particles = len(self.particles['x'])
        if n_particles == 0:
            return

        initialized = np.asarray(
            self.particles.get('vertical_position_initialized', np.zeros(n_particles)),
            dtype=bool,
        ).copy()
        initializable = (
            np.asarray(self.particles['status_released'], dtype=bool)
            & np.asarray(self.particles['status_domain'], dtype=bool)
            & np.asarray(self.particles['status_alive'], dtype=bool)
        )
        to_initialize = initializable & ~initialized
        if not np.any(to_initialize):
            return

        modes = np.asarray(
            self.particles.get('vertical_position_mode', np.full(n_particles, 'burial_depth')),
            dtype='<U32',
        )
        values = np.asarray(
            self.particles.get('vertical_position_value', np.full(n_particles, np.nan)),
            dtype=float,
        )
        burial_depth = np.maximum(
            np.nan_to_num(np.asarray(self.particles['burial_depth'], dtype=float), nan=0.0),
            0.0,
        )
        bed_level = np.asarray(self.particles.get('bed_level', np.zeros(n_particles)), dtype=float)
        suspended = np.asarray(self.particles['status_suspended'], dtype=bool).copy()
        deposited = np.asarray(self.particles['status_deposited'], dtype=bool).copy()
        buried = np.asarray(self.particles['status_buried'], dtype=bool).copy()

        for mode in np.unique(modes[to_initialize]):
            idx = to_initialize & (modes == mode)
            if mode == 'height_above_bed':
                height_above_bed = np.maximum(np.nan_to_num(values[idx], nan=0.0), 0.0)
            elif mode == 'centroid_on_release':
                height_above_bed = np.ones(np.count_nonzero(idx), dtype=float)
            elif mode == 'absolute_z':
                absolute_z = np.where(np.isfinite(values[idx]), values[idx], bed_level[idx])
                height_above_bed = absolute_z - bed_level[idx]
            elif mode == 'bed':
                height_above_bed = np.zeros(np.count_nonzero(idx), dtype=float)
            else:
                height_above_bed = -burial_depth[idx]

            suspended[idx] = height_above_bed > 0.0
            deposited[idx] = ~suspended[idx]
            initialized[idx] = True

        buried = self.update_status_buried(burial_depth=burial_depth)
        self.particles['status_suspended'] = suspended
        self.particles['status_deposited'] = deposited
        self.particles['status_buried'] = buried
        self.particles['vertical_position_initialized'] = initialized
        self.particles['status_eligible'] = (
            np.asarray(self.particles['status_domain'], dtype=bool)
            & np.asarray(self.particles['status_alive'], dtype=bool)
            & np.asarray(self.particles['status_released'], dtype=bool)
            & np.asarray(self.particles['status_transported'], dtype=bool)
            & ~buried
        )

    def _initialize_vertical_position(
        self,
        bed_level,
        water_depth,
        entrainment_height,
        z,
        burial_depth,
        initializable,
        is_suspended,
        is_deposited,
        is_buried,
    ):
        """
        Resolve configured initial vertical particle position after bed and flow
        fields are known.

        Supported seeding.vertical_position modes:
        - burial_depth: z = bed_level - burial_depth
        - bed: z = bed_level
        - height_above_bed: z = bed_level + value
        - absolute_z: z = value in the model vertical datum
        - centroid_on_release: z = bed_level + q3d_entrainment_height_above_bed
        """
        n_particles = len(self.particles['x'])
        initialized = np.asarray(
            self.particles.get('vertical_position_initialized', np.isfinite(z)),
            dtype=bool,
        ).copy()
        modes = np.asarray(
            self.particles.get('vertical_position_mode', np.full(n_particles, 'burial_depth')),
            dtype='<U32',
        )
        values = np.asarray(
            self.particles.get('vertical_position_value', np.full(n_particles, np.nan)),
            dtype=float,
        )

        to_initialize = initializable & ~initialized
        if not np.any(to_initialize):
            self.particles['vertical_position_initialized'] = initialized
            return z, burial_depth, is_suspended, is_deposited, is_buried

        for mode in np.unique(modes[to_initialize]):
            idx = to_initialize & (modes == mode)
            if not np.any(idx):
                continue

            if mode == 'bed':
                z[idx] = bed_level[idx]
            elif mode == 'height_above_bed':
                height = np.clip(np.nan_to_num(values[idx], nan=0.0), 0.0, water_depth[idx])
                z[idx] = bed_level[idx] + height
            elif mode == 'absolute_z':
                absolute_z = values[idx].copy()
                absolute_z = np.where(np.isfinite(absolute_z), absolute_z, bed_level[idx])
                z[idx] = np.minimum(absolute_z, bed_level[idx] + water_depth[idx])
            elif mode == 'centroid_on_release':
                height = np.clip(np.nan_to_num(entrainment_height[idx], nan=0.0), 0.0, water_depth[idx])
                z[idx] = bed_level[idx] + height
            else:
                z[idx] = bed_level[idx] - burial_depth[idx]

            height_above_bed = z[idx] - bed_level[idx]
            is_suspended[idx] = height_above_bed > 0.0
            is_deposited[idx] = ~is_suspended[idx]
            below_bed = height_above_bed < 0.0
            burial_depth[idx] = np.where(below_bed, -height_above_bed, 0.0)
            initialized[idx] = True

        self.particles['vertical_position_initialized'] = initialized
        return z, burial_depth, is_suspended, is_deposited, is_buried

    def _advect_particles_with_velocity(self, active, velocity_x, velocity_y, dt):
        """Move active particles with particle-level horizontal velocities.

        Returns
        -------
        left_domain_indices, beached_indices : tuple[np.ndarray, np.ndarray]
            Particle indices that crossed an open boundary or land boundary.
        """
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        old_x = self.particles['x'][active].copy()
        old_y = self.particles['y'][active].copy()
        old_simplices = self._particle_simplices[active_indices].copy()

        new_x = old_x + velocity_x[active] * dt
        new_y = old_y + velocity_y[active] * dt
        new_simplices = self.grid_geometry.locate_points(
            new_x,
            new_y,
            start_simplices=old_simplices,
        )

        self.particles['x'][active] = new_x
        self.particles['y'][active] = new_y
        self._particle_simplices[active_indices] = new_simplices

        outside_domain = new_simplices < 0
        if not np.any(outside_domain):
            self._mark_particle_simplices_current()
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        outside_particle_indices = active_indices[outside_domain]
        boundary_classes = self.grid_geometry.classify_boundary_crossings(
            old_x[outside_domain],
            old_y[outside_domain],
            new_x[outside_domain],
            new_y[outside_domain],
        )
        boundary_classes = np.asarray(boundary_classes).astype(str)

        land_boundary = boundary_classes == 'land'
        open_boundary = ~land_boundary

        self.particles['status_domain'][outside_particle_indices] = False
        self.particles['status_mobile'][outside_particle_indices] = False

        left_domain_indices = outside_particle_indices[open_boundary]
        if left_domain_indices.size:
            self.particles['status_left_domain'][left_domain_indices] = True
            self.particles['status_alive'][left_domain_indices] = False

        beached_indices = outside_particle_indices[land_boundary]
        if beached_indices.size:
            land_local_indices = np.flatnonzero(outside_domain)[land_boundary]
            self.particles['x'][beached_indices] = old_x[land_local_indices]
            self.particles['y'][beached_indices] = old_y[land_local_indices]
            self._particle_simplices[beached_indices] = old_simplices[land_local_indices]
            self.particles['status_beached'][beached_indices] = True
            self.particles['status_domain'][beached_indices] = True

        self._mark_particle_simplices_current()
        return left_domain_indices, beached_indices

    @staticmethod

    def _select_q3d_entrainment(
        available,
        turbulent_shields,
        critical_shields,
        dt,
        *,
        mode='shields_threshold',
        entrainment_frequency=None,
        probability_law='poisson',
        rng=None,
    ):
        """
        Select available bed particles that enter Q3D suspension.

        Parameters
        ----------
        available : array of bool
            Particles that are released, alive, inside the domain, exposed, and
            not already suspended.
        turbulent_shields : array
            Local turbulent Shields number [-].
        critical_shields : array
            Particle critical Shields number [-].
        dt : float
            Outer particle timestep [s].
        mode : str
            Entrainment rule:
            - shields_threshold: deterministic turbulent_shields > critical_shields.
            - non_zero_particle_velocity: entrain every available particle; the
              later particle-level velocity calculation decides whether it moves.
            - entrainment_frequency: stochastic entrainment from f_e. The
              probability law can be poisson, P = 1 - exp(-f_e dt), or
              linear, P = min(f_e dt, 1), matching the Eq. 68 small-dt form.
        entrainment_frequency : array or None
            Entrainment frequency [1/s] for entrainment_frequency mode,
            from the shared MacDonald entrainment-frequency field.
        probability_law : str
            Probability conversion used with entrainment_frequency mode:
            poisson uses P = 1 - exp(-f_e dt); linear uses P = min(f_e dt, 1).
        rng : random generator, optional
            Random source used for stochastic entrainment.

        Returns
        -------
        entrained_now, entrainment_probability : tuple of arrays
            Boolean entrainment decision and diagnostic probability [-].
        """
        available = np.asarray(available, dtype=bool)
        mode = str(mode or 'shields_threshold').strip().lower().replace('-', '_')
        probability = np.zeros_like(turbulent_shields, dtype=float)

        if mode in {'shields', 'shields_threshold', 'critical_shields'}:
            can_entrain = (
                np.isfinite(critical_shields)
                & (critical_shields > 0.0)
                & (turbulent_shields > critical_shields)
            )
            probability[can_entrain] = 1.0
            return available & can_entrain, probability

        if mode in {'non_zero_particle_velocity', 'nonzero_particle_velocity', 'all', 'all_available', 'available'}:
            probability[available] = 1.0
            return available.copy(), probability

        if mode in {'entrainment_frequency', 'frequency'}:
            if entrainment_frequency is None:
                raise ValueError('entrainment_frequency is required when entrainment.method is entrainment_frequency.')
            frequency = np.maximum(np.nan_to_num(entrainment_frequency, nan=0.0), 0.0)
            probability_law = str(probability_law or 'poisson').strip().lower().replace('-', '_')
            frequency_dt = frequency * dt
            if probability_law in {'poisson', 'exponential'}:
                probability = np.clip(-np.expm1(-frequency_dt), 0.0, 1.0)
            elif probability_law in {'linear', 'macdonald', 'macdonald_eq68', 'eq68'}:
                probability = np.clip(frequency_dt, 0.0, 1.0)
            else:
                raise ValueError(
                    "Unsupported entrainment probability law "
                    f"{probability_law!r}. Expected poisson or linear."
                )
            probability = np.broadcast_to(probability, available.shape).astype(float, copy=True)
            random_source = np.random if rng is None else rng
            return available & (random_source.random(len(available)) < probability), probability

        raise ValueError(
            "Unsupported MacDonald entrainment method "
            f"{mode!r}. Expected shields_threshold, non_zero_particle_velocity, or entrainment_frequency."
        )

    @staticmethod

    def _normalize_q3d_vertical_update_scheme(scheme):
        scheme = str(scheme or 'geometric').strip().lower().replace('-', '_')
        aliases = {
            'geometry': 'geometric',
            'geometric_method': 'geometric',
            'macdonald_geometric': 'geometric',
            'centroid': 'centroid_floor',
            'centroid_floor_method': 'centroid_floor',
            'rouse': 'rouse_profile',
            'rouse_sample': 'rouse_profile',
            'rouse_sampling': 'rouse_profile',
        }
        scheme = aliases.get(scheme, scheme)
        if scheme not in {'geometric', 'centroid_floor', 'rouse_profile'}:
            raise ValueError(
                "Unsupported q3d_vertical_update_scheme "
                f"{scheme!r}. Expected geometric, centroid_floor, or rouse_profile."
            )
        return scheme

    @staticmethod

    def _normalize_q3d_diagnostics_level(level):
        # Legacy minimal/full option kept for backward-compatible YAML files.
        level = str(level or 'minimal').strip().lower().replace('-', '_')
        aliases = {
            'basic': 'minimal',
            'standard': 'minimal',
            'reduced': 'minimal',
            'all': 'full',
            'debug': 'full',
        }
        level = aliases.get(level, level)
        if level not in {'minimal', 'full'}:
            raise ValueError(f"Unsupported q3d_diagnostics level {level!r}. Expected minimal or full.")
        return level

    @staticmethod

    def _sample_rouse_profile_height(water_depth, rouse_number, rng=None):
        """
        Draw particle heights above bed from a fast Rouse-shaped beta distribution.

        This is an intentionally compact approximation for Q3D experiments:
        R=0 gives a uniform water-column sample, and increasing R concentrates
        samples closer to the bed while still allowing occasional high
        suspension. alpha and beta are beta-distribution shape parameters, not
        MacDonald coefficients.
        """
        water_depth = np.maximum(np.nan_to_num(water_depth, nan=0.0), 0.0)
        rouse_number = np.clip(np.nan_to_num(rouse_number, nan=0.0), 0.0, 20.0)
        random_source = np.random if rng is None else rng
        alpha = 1.0 / (1.0 + rouse_number)
        beta = 1.0 + rouse_number
        return water_depth * random_source.beta(alpha, beta)

    def _q3d_height_after_vertical_update(
        self,
        scheme,
        *,
        z_p_old,
        bed_level_old,
        bed_level_new,
        water_depth_new,
        particle_w,
        settling_velocity,
        transport_centroid_elevation_new,
        rouse_number_new,
        dt,
        rng=None,
    ):
        """
        Compute active-particle height above bed after horizontal advection.

        This is the small function to edit when experimenting with Q3D vertical
        position schemes. Inputs are already sliced to active particles.
        """
        if scheme == 'geometric':
            # Convert the vertical update to height above the new bed. z_p_old is
            # relative to the old bed; particle_w * dt is the vertical displacement in
            # absolute space; bed_level_old - bed_level_new corrects for the bed elevation
            # change after horizontal advection.
            z_p_new = z_p_old + particle_w * dt + (bed_level_old - bed_level_new)
        elif scheme == 'centroid_floor':
            z_c_new = np.clip(np.nan_to_num(transport_centroid_elevation_new, nan=0.0), 0.0, water_depth_new) # relative to bed
            candidate_z = z_p_old - np.maximum(np.nan_to_num(settling_velocity, nan=0.0), 0.0) * dt
            z_p_new = np.maximum(candidate_z, z_c_new)
        elif scheme == 'rouse_profile':
            z_p_new = self._sample_rouse_profile_height(water_depth_new, rouse_number_new, rng=rng)
        else:
            raise ValueError(f'Unsupported q3d vertical update scheme {scheme!r}')

        return np.clip(np.nan_to_num(z_p_new, nan=0.0, posinf=0.0, neginf=0.0), 0.0, water_depth_new)

    def update_q3d_particle_position(
        self,
        current_timestep: float,
        centroid_flow_field: Dict,
        hydrodynamic_flow_field: Dict,
        bed_level_field: Any,
        max_shear_velocity: Any,
        selected_shear_velocity: Any,
        profile_roughness_height: Any,
        total_transport_centroid_elevation: Any,
        q3d_velocity_deficit_coefficient: Any,
        q3d_vertical_velocity_gradient: Any,
        turbulent_shields_number: Any,
        critical_shields_number: Any,
        settling_velocity: Any,
        water_depth: Any,
        skin_roughness_height: Any,
        entrainment_height_above_bed: Any,
        rouse_number: Any = None,
        K_Et: float = 0.15,
        K_Ev: float | None = None,
        M_b: Any = None,
        E_turb_hor_min: float = 0.02,
        E_turb_vert_min: float = 0.0,
        q3d_horizontal_diffusion_enabled: bool = True,
        entrainment_method: str = 'shields_threshold',
        entrainment_frequency: Any = None,
        entrainment_probability_law: str = 'poisson',
        q3d_vertical_update_scheme: str = 'geometric',
        q3d_motion_substeps: int = 1,
        q3d_save_first_substep_diagnostics: bool | None = None,
        q3d_diagnostics: str = 'minimal',
        rng: Any = None,
    ) -> None:
        """
        Update Q3D particle-resolved horizontal and vertical motion.

        This method applies the MacDonald Q3D equations at particle positions
        instead of on the Eulerian field grid. It interpolates the required
        hydraulic fields to each particle, computes the particle height above
        bed z_p from the current particle elevation z, computes particle-level
        modified centroid particle velocity for horizontal advection, turbulent
        diffusion, vertical advection, settling, and random-walk diffusion, then
        updates x, y, z and derived particle state fields.

        References below are to MacDonald et al. (2006), PTM Report 1: fall
        time/velocity deficit (Eqs. 36-40), Q3D vertical velocity (Eqs. 41-42),
        turbulent diffusion/random walk (Eqs. 45, 49, 51-52), turbulent Shields
        and pickup/entrainment frequency (Eqs. 57-61), and stochastic
        entrainment over dt (Eq. 68).

        Parameters
        ----------
        current_timestep : float
            Particle update timestep dt [s].
        centroid_flow_field : dict
            Configured MacDonald centroid particle velocity field. This is
            interpolated to particle positions for diagnostics only.
        hydrodynamic_flow_field : dict
            Depth-averaged hydrodynamic/D3D-FM flow field used for local flow
            direction and depth-averaged velocity magnitude.
        bed_level_field : array or temporal field
            Bed level field [m] used before and after horizontal advection to
            compute the geometric vertical update over the new bed position.
        max_shear_velocity : array or temporal field
            Maximum shear velocity u_* [m/s].
        profile_roughness_height : array or temporal field
            Roughness height k_s [m] used by the MacDonald log-law velocity profile.
        total_transport_centroid_elevation : array or temporal field
            Total-load transport centroid height z_c above the bed [m].
        q3d_velocity_deficit_coefficient : array or temporal field
            Velocity deficit coefficient c_A [-].
        q3d_vertical_velocity_gradient : array or temporal field
            Continuity-based vertical velocity gradient term, equal to dh/hdt + div(U) [1/s].
        turbulent_shields_number : array or temporal field
            Turbulent Shields number from the MacDonald Q3D field calculation [-].
        critical_shields_number : float or array
            Critical Shields number for this particle population [-]. This is a grain/particle
            property and is usually passed as a scalar, then broadcast to particles.
        settling_velocity : float or array
            Particle settling velocity [m/s]. This is a grain/particle property and is usually
            passed as a scalar, then broadcast to particles.
        water_depth : array or temporal field
            Water depth h [m].
        skin_roughness_height : array or temporal field
            Skin roughness height used for the near-bed deposition threshold [m].
        entrainment_height_above_bed : array or temporal field
            Height above the bed assigned to particles at the instant they entrain [m].
        rouse_number : array or temporal field, optional
            Rouse number [-] used when q3d_vertical_update_scheme is rouse_profile.
        K_Et : float
            Horizontal turbulent diffusion scaling coefficient.
        K_Ev : float or None
            Vertical turbulent diffusion scaling coefficient. If None, defaults to K_Et.
        M_b : array or temporal field, optional
            Wave-breaking enhancement factor. If None, defaults to 1 in the diffusion helper.
        E_turb_hor_min : float
            Minimum horizontal turbulent diffusivity [m^2/s].
        E_turb_vert_min : float
            Minimum vertical turbulent diffusivity [m^2/s].
        q3d_horizontal_diffusion_enabled : bool
            If false, disable horizontal turbulent diffusion while leaving
            geometric-scheme vertical diffusion unchanged.
        entrainment_method : str
            Rule used to decide which available bed particles enter suspension.
            Supported values are shields_threshold, non_zero_particle_velocity,
            and entrainment_frequency.
        entrainment_frequency : float or array, optional
            Shared MacDonald entrainment-frequency field [1/s] used when
            entrainment_method is entrainment_frequency.
        entrainment_probability_law : str
            Probability law for entrainment_frequency mode. poisson uses
            P = 1 - exp(-f_e dt); linear uses P = min(f_e dt, 1), the
            MacDonald Eq. 68 small-timestep approximation.
        q3d_vertical_update_scheme : str
            Vertical position update scheme. Supported values are:
            - geometric: current MacDonald/geometric bed-change correction.
            - centroid_floor: settling-only absolute vertical update with a
              floor at the new total-transport centroid height.
            - rouse_profile: sample new height above bed from a Rouse-shaped
              distribution at the new horizontal location.
        q3d_motion_substeps : int
            Number of smaller Q3D motion updates inside one particle timestep.
            A value of 1 gives the original single-step update.
        q3d_save_first_substep_diagnostics : bool or None
            If true, store first-substep Q3D hydraulic and diffusion diagnostic
            arrays. If None, the legacy q3d_diagnostics level is used.
        q3d_diagnostics : str
            Legacy Q3D particle diagnostics level. minimal disables the extra
            first-substep fields; full enables them.
        rng : random generator, optional
            Random source used for turbulent random-walk velocities.

        Notes
        -----
        Q3D entrainment is handled by _select_q3d_entrainment. The default
        behavior is deterministic Shields-threshold entrainment, but the same
        transport loop can also use legacy flow-speed, all-available, or
        stochastic entrainment-frequency rules. Available particles that do not
        entrain remain deposited at the bed. Bed particles that are not
        available are marked deposited/not suspended, but their burial_depth is
        not recomputed in this method.

        The calculation order is:
        1. Interpolate/read all required grid fields and particle constants.
        2. Read current particle state arrays.
        3. Resolve configured initial vertical position for newly released particles.
        4. Use the configured Q3D entrainment rule to lift newly entrained particles to
           q3d_entrainment_height_above_bed.
        5. Split dt into q3d_motion_substeps smaller updates.
        6. For each substep, recompute z_p and particle-level Q3D velocities,
           move suspended particles horizontally, then update z with the
           configured vertical update scheme.
        7. Deposit particles that return to the near-bed threshold.
        """

        # ------------------------------------------------------------------
        # 1. Validate timestep and read/interpolate all Q3D inputs.
        #    MacDonald's random-walk velocities and entrainment probability are
        #    timestep dependent (Eqs. 51-52 and 68), so the CFL-limited particle
        #    dt must be known before this method is called.
        # ------------------------------------------------------------------
        dt = float(current_timestep)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f'current_timestep must be positive and finite, got {current_timestep!r}')

        if q3d_motion_substeps is None:
            q3d_motion_substeps = 1
        try:
            substeps = int(q3d_motion_substeps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'q3d_motion_substeps must be a positive integer, got {q3d_motion_substeps!r}'
            ) from exc
        if substeps < 1:
            raise ValueError(f'q3d_motion_substeps must be >= 1, got {q3d_motion_substeps!r}')
        dt_sub = dt / substeps
        vertical_update_scheme = self._normalize_q3d_vertical_update_scheme(q3d_vertical_update_scheme)
        if q3d_save_first_substep_diagnostics is None:
            diagnostics_level = self._normalize_q3d_diagnostics_level(q3d_diagnostics)
            save_first_substep_diagnostics = diagnostics_level == 'full'
        else:
            save_first_substep_diagnostics = bool(q3d_save_first_substep_diagnostics)
        if vertical_update_scheme == 'rouse_profile' and rouse_number is None:
            raise ValueError('rouse_number is required when q3d_vertical_update_scheme is rouse_profile.')

        if hydrodynamic_flow_field is None:
            raise ValueError('hydrodynamic_flow_field is required for Q3D particle motion.')

        water_depth_field = water_depth
        skin_roughness_height_field = skin_roughness_height

        initial_particle_fields = {
            'centroid_particle_velocity_u': self._flow_component_field(
                centroid_flow_field,
                'u',
            ),
            'centroid_particle_velocity_v': self._flow_component_field(
                centroid_flow_field,
                'v',
            ),
            'depth_avg_flow_velocity_u': self._flow_component_field(
                hydrodynamic_flow_field,
                'u',
            ),
            'depth_avg_flow_velocity_v': self._flow_component_field(
                hydrodynamic_flow_field,
                'v',
            ),
            'bed_level': bed_level_field,
            'max_shear_velocity': max_shear_velocity,
            'selected_shear_velocity': selected_shear_velocity,
            'profile_roughness_height': profile_roughness_height,
            'total_transport_centroid_elevation': total_transport_centroid_elevation,
            'q3d_velocity_deficit_coefficient': q3d_velocity_deficit_coefficient,
            'q3d_vertical_velocity_gradient': q3d_vertical_velocity_gradient,
            'turbulent_shields_number': turbulent_shields_number,
            'critical_shields_number': critical_shields_number,
            'settling_velocity': settling_velocity,
            'water_depth': water_depth,
            'skin_roughness_height': skin_roughness_height,
            'q3d_entrainment_height_above_bed': entrainment_height_above_bed,
        }
        if rouse_number is not None:
            initial_particle_fields['rouse_number'] = rouse_number
        if entrainment_frequency is not None:
            initial_particle_fields['macdonald_entrainment_frequency'] = entrainment_frequency
        if M_b is not None:
            initial_particle_fields['q3d_wave_breaking_factor'] = M_b
        self._update_particle_fields(initial_particle_fields)
        self.particles['centroid_particle_velocity_magnitude'] = np.hypot(
            self.particles['centroid_particle_velocity_u'],
            self.particles['centroid_particle_velocity_v'],
        )
        self.particles['depth_avg_flow_velocity_magnitude'] = np.hypot(
            self.particles['depth_avg_flow_velocity_u'],
            self.particles['depth_avg_flow_velocity_v'],
        )

        # Required fields after interpolation/broadcasting. Hydrodynamic flow is
        # required for direction; centroid velocity is diagnostics only.
        required_fields = (
            'bed_level',
            'max_shear_velocity',
            'selected_shear_velocity',
            'profile_roughness_height',
            'total_transport_centroid_elevation',
            'q3d_velocity_deficit_coefficient',
            'q3d_vertical_velocity_gradient',
            'turbulent_shields_number',
            'critical_shields_number',
            'settling_velocity',
            'water_depth',
            'skin_roughness_height',
            'q3d_entrainment_height_above_bed',
            'depth_avg_flow_velocity_u',
            'depth_avg_flow_velocity_v',
            'depth_avg_flow_velocity_magnitude',
        )
        if vertical_update_scheme == 'rouse_profile':
            required_fields = required_fields + ('rouse_number',)
        missing = [name for name in required_fields if name not in self.particles]
        if missing:
            raise KeyError(f'Missing fields for Q3D particle position update: {missing}')

        required_status_fields = (
            'status_domain',
            'status_left_domain',
            'status_alive',
            'status_released',
            'status_suspended',
            'status_deposited',
            'status_buried',
            'status_eligible',
        )
        missing_status = [name for name in required_status_fields if name not in self.particles]
        if missing_status:
            raise KeyError(f'Missing status fields for Q3D particle position update: {missing_status}')

        # ------------------------------------------------------------------
        # 2. Read current particle state. z may still be NaN for particles
        #    that have not reached their release time yet.
        # ------------------------------------------------------------------
        n_particles = len(self.particles['x'])
        # Local bed elevation at each particle position at the start of this timestep.
        bed_level = np.asarray(self.particles['bed_level'], dtype=float)
        water_depth = np.maximum(np.nan_to_num(self.particles['water_depth'], nan=0.0), 0.0)
        skin_roughness = np.maximum(np.nan_to_num(self.particles['skin_roughness_height'], nan=0.0), 0.0)
        # z is the live absolute particle elevation. It starts as NaN because
        # the configured vertical position can only be resolved after bed_level,
        # water_depth, and possibly the Q3D centroid/entrainment height are known.
        if 'z' not in self.particles:
            self.particles['z'] = np.full(n_particles, np.nan, dtype=float)

        # Read the existing state arrays. On the first timestep these are mostly
        # defaults; on later timesteps they are the state left by the previous
        # Q3D update. The initializer below only changes particles that have not
        # been vertically initialized yet.
        z = np.asarray(self.particles['z'], dtype=float).copy()
        burial_depth = np.maximum(
            np.nan_to_num(self.particles.get('burial_depth', np.zeros(n_particles)), nan=0.0),
            0.0,
        )
        is_inside = (
            np.asarray(self.particles['status_domain'], dtype=bool)
            & ~np.asarray(self.particles['status_left_domain'], dtype=bool)
        )
        is_alive = np.asarray(self.particles['status_alive'], dtype=bool)
        is_released = np.asarray(self.particles['status_released'], dtype=bool)
        is_suspended = np.asarray(self.particles['status_suspended'], dtype=bool)
        is_deposited = np.asarray(self.particles['status_deposited'], dtype=bool)
        is_buried = np.asarray(self.particles['status_buried'], dtype=bool)
        eligible = np.asarray(self.particles['status_eligible'], dtype=bool)

        # ------------------------------------------------------------------
        # 3. Resolve initial vertical position for newly released particles.
        # ------------------------------------------------------------------
        turbulent_shields = np.nan_to_num(self.particles['turbulent_shields_number'], nan=0.0)
        critical_shields = np.nan_to_num(self.particles['critical_shields_number'], nan=np.inf)
        entrainment_height = np.clip(
            np.nan_to_num(self.particles['q3d_entrainment_height_above_bed'], nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            water_depth,
        )

        # This applies seeding.vertical_position exactly once per particle.
        # Already initialized particles keep their current z, burial_depth, and
        # suspension/deposition status and transport-controlled burial status.
        z, burial_depth, is_suspended, is_deposited, is_buried = self._initialize_vertical_position(
            bed_level,
            water_depth,
            entrainment_height,
            z,
            burial_depth,
            is_released & is_inside & is_alive,
            is_suspended,
            is_deposited,
            is_buried,
        )

        # Vertical initialization can change burial_depth (for example absolute_z),
        # so recalculate burial and eligibility immediately in the same timestep.
        is_buried = self.update_status_buried(burial_depth=burial_depth)
        eligible = (
            is_inside
            & is_alive
            & is_released
            & ~is_buried
            & np.asarray(self.particles['status_transported'], dtype=bool)
        )
        self.particles['status_eligible'] = eligible

        # ------------------------------------------------------------------
        # 4. Decide which bed particles entrain this timestep.
        #    The stochastic mode uses MacDonald's pickup/entrainment frequency
        #    f_e (Eqs. 57 and 61) as a timestep probability (Eq. 68). The
        #    threshold mode uses the turbulent Shields number from Eq. 59.
        # ------------------------------------------------------------------
        available = eligible & ~is_suspended
        entrainment_frequency = self.particles.get('macdonald_entrainment_frequency')
        entrained_now, entrainment_probability = self._select_q3d_entrainment(
            available,
            turbulent_shields,
            critical_shields,
            dt,
            mode=entrainment_method,
            entrainment_frequency=entrainment_frequency,
            probability_law=entrainment_probability_law,
            rng=rng,
        )
        not_entrained = available & ~entrained_now

        # Entrainment is sampled once for the outer particle timestep, so the
        # probability uses dt here. Q3D motion substeps below use dt_sub only for
        # advection, diffusion, and vertical position updates.

        # Entrained particles leave the bed/burial layer and enter the water column.
        z[entrained_now] = bed_level[entrained_now] + entrainment_height[entrained_now]
        burial_depth[entrained_now] = 0.0
        is_buried[entrained_now] = False
        is_deposited[entrained_now] = False
        is_suspended[entrained_now] = True

        # Available particles that are not entrained stay at/on the bed.
        is_suspended[not_entrained] = False
        is_deposited[not_entrained] = True
        is_buried[not_entrained] = burial_depth[not_entrained] > 0.0
        z[not_entrained] = np.where(
            is_buried[not_entrained],
            bed_level[not_entrained] - burial_depth[not_entrained],
            bed_level[not_entrained],
        )

        # ------------------------------------------------------------------
        # 5. Move suspended particles in q3d_motion_substeps smaller updates.
        #    Substepping does not change the outer timestep physics; it only
        #    evaluates the Q3D random walk and vertical update on smaller dt_sub
        #    intervals,
        #    Substepping uses dt_sub in the random-walk velocity formulas from
        #    MacDonald Eqs. 51-52. This keeps each diffusive displacement proportional
        #    to sqrt(E * dt_sub), so the accumulated random walk has the correct
        #    timestep scaling over the full particle timestep.
        # ------------------------------------------------------------------
        random_source = np.random if rng is None else rng
        particle_u = np.zeros(n_particles, dtype=float)
        particle_v = np.zeros(n_particles, dtype=float)
        vertical_advection = np.zeros(n_particles, dtype=float)
        particle_w = np.zeros(n_particles, dtype=float)
        deposited_now = np.zeros(n_particles, dtype=bool)
        height_above_bed = np.maximum(np.nan_to_num(z - bed_level, nan=0.0), 0.0)
        first_substep_diagnostics_saved = False
        first_substep_z_p = height_above_bed.copy()
        first_substep_modified_u = np.zeros(n_particles, dtype=float)
        first_substep_modified_v = np.zeros(n_particles, dtype=float)
        first_substep_particle_u = np.zeros(n_particles, dtype=float)
        first_substep_particle_v = np.zeros(n_particles, dtype=float)
        first_substep_particle_w = np.zeros(n_particles, dtype=float)
        if save_first_substep_diagnostics:
            first_substep_horizontal_diffusion_velocity_x = np.zeros(n_particles, dtype=float)
            first_substep_horizontal_diffusion_velocity_y = np.zeros(n_particles, dtype=float)
            first_substep_vertical_diffusion_velocity = np.zeros(n_particles, dtype=float)
            first_substep_vertical_advection = np.zeros(n_particles, dtype=float)
            first_substep_horizontal_diffusion = np.zeros(n_particles, dtype=float)
            first_substep_vertical_diffusion = np.zeros(n_particles, dtype=float)
            first_substep_bed_level = bed_level.copy()
            first_substep_water_depth = water_depth.copy()
            first_substep_skin_roughness = skin_roughness.copy()
            first_substep_shear_velocity = np.zeros(n_particles, dtype=float)
            first_substep_profile_roughness = np.zeros(n_particles, dtype=float)
            first_substep_z_c = np.zeros(n_particles, dtype=float)
            first_substep_deficit = np.zeros(n_particles, dtype=float)
            first_substep_vertical_velocity_gradient = np.zeros(n_particles, dtype=float)
            first_substep_settling_velocity = np.zeros(n_particles, dtype=float)
            first_substep_flow_magnitude = np.zeros(n_particles, dtype=float)
            first_substep_rouse_number = np.zeros(n_particles, dtype=float)
        vertical_scheme_codes = {'geometric': 0, 'centroid_floor': 1, 'rouse_profile': 2}

        for substep_index in range(substeps):
            # Only particles that are currently suspended and not deposited are active
            # for this substep. Particles that deposit in one substep stop moving in
            # the next substeps of the same outer timestep.
            active = eligible & is_suspended & ~is_deposited
            active_indices = np.flatnonzero(active)
            active_count = active_indices.size
            if active_count == 0:
                break

            bed_level_current = np.asarray(self.particles['bed_level'], dtype=float)
            water_depth_current = np.asarray(self.particles['water_depth'], dtype=float)
            skin_roughness_current = np.asarray(self.particles['skin_roughness_height'], dtype=float)

            # z_p is the key Q3D state: current particle height above the local bed.
            # This is particle-resolved z_p, not the grid diagnostic value that
            # assumes z_p = z_c. MacDonald's velocity, diffusion, and vertical
            # advection terms are evaluated at this live particle height.
            bed_active = bed_level_current[active_indices]
            waterdepth_active = np.maximum(np.nan_to_num(water_depth_current[active_indices], nan=0.0), 0.0)
            skin_active = np.maximum(np.nan_to_num(skin_roughness_current[active_indices], nan=0.0), 0.0)
            z_p_active = np.clip(np.nan_to_num(z[active_indices] - bed_active, nan=0.0), 0.0, waterdepth_active)
            max_shear_velocity_active = np.nan_to_num(
                np.asarray(self.particles['max_shear_velocity'], dtype=float)[active_indices],
                nan=0.0,
            )
            selected_shear_velocity_active = np.nan_to_num(
                np.asarray(self.particles.get('selected_shear_velocity'), dtype=float)[active_indices],
                nan=0.0,
            )
            profile_roughness_active = np.maximum(
                np.nan_to_num(
                    np.asarray(self.particles['profile_roughness_height'], dtype=float)[active_indices],
                    nan=0.0,
                ),
                1e-12,
            )
            z_c_active = np.clip(
                np.nan_to_num(
                    np.asarray(self.particles['total_transport_centroid_elevation'], dtype=float)[active_indices],
                    nan=0.0,
                ),
                1e-12,
                np.maximum(waterdepth_active, 1e-12),
            )
            deficit_active = np.clip(
                np.nan_to_num(
                    np.asarray(self.particles['q3d_velocity_deficit_coefficient'], dtype=float)[active_indices],
                    nan=1.0,
                ),
                0.0,
                1.0,
            )

            # Compute reduced horizontal speed from the log-law velocity at
            # z_p and the reference height 1.4 z_c. The velocity deficit follows
            # MacDonald Eqs. 39-40 after the fall-time logic in Eqs. 36-38.
            u_zp_active = self._loglaw_velocity_at_z(selected_shear_velocity_active, z_p_active, profile_roughness_active)
            u_1p4zc_active = self._loglaw_velocity_at_z(
                selected_shear_velocity_active,
                1.4 * z_c_active,
                profile_roughness_active,
            )
            modified_particle_velocity_active = self._apply_q3d_velocity_deficit(
                u_zp_active,
                u_1p4zc_active,
                z_p_active,
                z_c_active,
                deficit_active,
            )

            # Convert scalar particle speed to x/y components using the local
            # depth-averaged hydrodynamic flow direction.
            da_velocity_u_active = np.nan_to_num(
                np.asarray(self.particles['depth_avg_flow_velocity_u'], dtype=float)[active_indices],
                nan=0.0,
            )
            da_velocity_v_active = np.nan_to_num(
                np.asarray(self.particles['depth_avg_flow_velocity_v'], dtype=float)[active_indices],
                nan=0.0,
            )
            da_velocity_magnitude_active = np.maximum(
                np.nan_to_num(np.asarray(self.particles['depth_avg_flow_velocity_magnitude'], dtype=float)[active_indices], nan=0.0),
                1e-12,
            )
            modified_u_active = modified_particle_velocity_active * _safe_divide(da_velocity_u_active, da_velocity_magnitude_active)
            modified_v_active = modified_particle_velocity_active * _safe_divide(da_velocity_v_active, da_velocity_magnitude_active)

            # Compute turbulent diffusivities and draw random-walk velocities for
            # this substep. E_t,h and E_t,v follow MacDonald Eqs. 45 and 49;
            # the random-walk velocities follow Eqs. 51-52. Using dt_sub keeps
            # random displacement velocity * dt_sub = O(sqrt(E * dt_sub)).
            M_b_active = None
            if M_b is not None and 'q3d_wave_breaking_factor' in self.particles:
                M_b_active = np.nan_to_num(
                    np.asarray(self.particles['q3d_wave_breaking_factor'], dtype=float)[active_indices],
                    nan=1.0,
                )
            use_horizontal_diffusion = bool(q3d_horizontal_diffusion_enabled)
            use_vertical_diffusion = vertical_update_scheme == 'geometric'
            horizontal_diffusion_active, vertical_diffusion_active = self._turbulent_diffusion_coefficients(
                waterdepth_active,
                z_p_active,
                da_velocity_magnitude_active,
                selected_shear_velocity_active,
                K_Et=K_Et,  # scalar
                K_Ev=K_Ev,  # scalar
                M_b=M_b_active,
                E_turb_hor_min=E_turb_hor_min,  # scalar
                E_turb_vert_min=E_turb_vert_min,  # scalar
                compute_horizontal=use_horizontal_diffusion,
                compute_vertical=use_vertical_diffusion,
            )
            # Only draw random values for enabled components. Besides avoiding
            # unnecessary work, this keeps disabled components from advancing
            # the random-number stream.
            random_horizontal_x_active = np.zeros(active_count, dtype=float)
            random_horizontal_y_active = np.zeros(active_count, dtype=float)
            if use_horizontal_diffusion:
                random_horizontal_x_active = (
                    2.0
                    * (random_source.random(active_count) - 0.5)
                    * np.sqrt(6.0 * horizontal_diffusion_active / dt_sub)
                )
                random_horizontal_y_active = (
                    2.0
                    * (random_source.random(active_count) - 0.5)
                    * np.sqrt(6.0 * horizontal_diffusion_active / dt_sub)
                )

            random_vertical_active = np.zeros(active_count, dtype=float)
            if use_vertical_diffusion:
                random_vertical_active = (
                    2.0
                    * (random_source.random(active_count) - 0.5)
                    * np.sqrt(6.0 * vertical_diffusion_active / dt_sub)
                )

            particle_u_active = np.nan_to_num(
                modified_u_active + random_horizontal_x_active,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            particle_v_active = np.nan_to_num(
                modified_v_active + random_horizontal_y_active,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            particle_u.fill(0.0)
            particle_v.fill(0.0)
            particle_u[active_indices] = particle_u_active
            particle_v[active_indices] = particle_v_active

            # Vertical velocity combines continuity advection (MacDonald
            # Eqs. 41-42), particle settling, and the vertical random walk from
            # Eq. 52, all evaluated at the old position for this substep.
            vertical_gradient_active = np.nan_to_num(
                np.asarray(self.particles['q3d_vertical_velocity_gradient'], dtype=float)[active_indices],
                nan=0.0,
            )
            vertical_advection_active = vertical_gradient_active * (waterdepth_active - z_p_active)
            settling_active = np.nan_to_num(
                np.asarray(self.particles['settling_velocity'], dtype=float)[active_indices],
                nan=0.0,
            )
            particle_w_active = np.nan_to_num(
                vertical_advection_active - settling_active + random_vertical_active,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            particle_w.fill(0.0)
            vertical_advection.fill(0.0)
            particle_w[active_indices] = particle_w_active
            vertical_advection[active_indices] = vertical_advection_active

            # Snapshot first-substep diagnostics. Final x/y/z/status still
            # describe the end of the outer timestep.
            if not first_substep_diagnostics_saved:
                first_substep_z_p[active_indices] = z_p_active
                first_substep_modified_u[active_indices] = modified_u_active
                first_substep_modified_v[active_indices] = modified_v_active
                first_substep_particle_u[active_indices] = particle_u_active
                first_substep_particle_v[active_indices] = particle_v_active
                first_substep_particle_w[active_indices] = particle_w_active
                if save_first_substep_diagnostics:
                    first_substep_horizontal_diffusion_velocity_x[active_indices] = random_horizontal_x_active
                    first_substep_horizontal_diffusion_velocity_y[active_indices] = random_horizontal_y_active
                    first_substep_vertical_diffusion_velocity[active_indices] = random_vertical_active
                    first_substep_vertical_advection[active_indices] = vertical_advection_active
                    first_substep_horizontal_diffusion[active_indices] = horizontal_diffusion_active
                    first_substep_vertical_diffusion[active_indices] = vertical_diffusion_active
                    first_substep_bed_level[active_indices] = bed_active
                    first_substep_water_depth[active_indices] = waterdepth_active
                    first_substep_skin_roughness[active_indices] = skin_active
                    first_substep_shear_velocity[active_indices] = max_shear_velocity_active
                    first_substep_profile_roughness[active_indices] = profile_roughness_active
                    first_substep_z_c[active_indices] = z_c_active
                    first_substep_deficit[active_indices] = deficit_active
                    first_substep_vertical_velocity_gradient[active_indices] = vertical_gradient_active
                    first_substep_settling_velocity[active_indices] = settling_active
                    first_substep_flow_magnitude[active_indices] = da_velocity_magnitude_active
                    if 'rouse_number' in self.particles:
                        first_substep_rouse_number[active_indices] = np.nan_to_num(
                            np.asarray(self.particles['rouse_number'], dtype=float)[active_indices],
                            nan=0.0,
                        )
                first_substep_diagnostics_saved = True

            # Horizontal move first using the Q3D particle velocity from the
            # depth-averaged flow direction plus random walk, then re-interpolate
            # bed/depth at the new x/y location before the vertical update.
            bed_level_current_active = bed_active.copy()
            z_p_current_active = z_p_active.copy() # relative to the bed, not absolute z, we need to know this for the vertical update.
            z_current_active = z[active_indices].copy() # absolute z, we need to know this for the vertical update.

            #Advect particles horizontally using the modified particle velocities 9including random walk). check if any particles leave the domain or beach, and update their status accordingly.
            left_domain_indices, beached_indices = self._advect_particles_with_velocity(
                active,
                particle_u,
                particle_v,
                dt_sub,
            )
            stopped_indices = np.concatenate((left_domain_indices, beached_indices))
            if stopped_indices.size:
                is_alive[left_domain_indices] = False
                is_inside[left_domain_indices] = False
                eligible[left_domain_indices] = False
                is_suspended[stopped_indices] = False
                is_deposited[beached_indices] = True
                deposited_now[beached_indices] = True
                z[beached_indices] = bed_level[beached_indices]
                burial_depth[beached_indices] = 0.0

                continuing = ~np.isin(active_indices, stopped_indices)
                active_indices = active_indices[continuing]
                active_count = active_indices.size
                if active_count == 0:
                    continue
                # If some particles stopped, we need to filter the active arrays to only include the continuing particles for the vertical update.
                bed_level_current_active = bed_level_current_active[continuing]
                z_p_current_active = z_p_current_active[continuing]
                z_current_active = z_current_active[continuing]
                particle_w_active = particle_w_active[continuing]
                settling_active = settling_active[continuing]

            # After the particle has moved to a new x,y horizontally, it might now be over a newer bed level, water depth and skin roughness.
            post_advection_fields = {
                'bed_level': bed_level_field, # this is needed to convert beween absolute z and relative z_p for the vertical update.
                'water_depth': water_depth_field, # this is needed to clip the new z_p to the water depth after the vertical update.
                'skin_roughness_height': skin_roughness_height_field, # this is needed to compute the deposition threshold for the vertical update.
            }

            #If there is another substep we will need to update the following fields for the next substep,
            # so we store them in the particle fields. If this is the last substep, we don't need to store them because they won't be used again.
            needs_next_substep_fields = substep_index < substeps - 1
            if vertical_update_scheme in {'centroid_floor', 'rouse_profile'} or needs_next_substep_fields:
                post_advection_fields['total_transport_centroid_elevation'] = total_transport_centroid_elevation
            if vertical_update_scheme == 'rouse_profile':
                post_advection_fields['rouse_number'] = rouse_number
            if needs_next_substep_fields:
                post_advection_fields.update(
                    {
                        'depth_avg_flow_velocity_u': self._flow_component_field(
                            hydrodynamic_flow_field,
                            'u',
                        ),
                        'depth_avg_flow_velocity_v': self._flow_component_field(
                            hydrodynamic_flow_field,
                            'v',
                        ),
                        'max_shear_velocity': max_shear_velocity,
                        'selected_shear_velocity': selected_shear_velocity,
                        'profile_roughness_height': profile_roughness_height,
                        'q3d_velocity_deficit_coefficient': q3d_velocity_deficit_coefficient,
                        'q3d_vertical_velocity_gradient': q3d_vertical_velocity_gradient,
                    }
                )
                if M_b is not None:
                    post_advection_fields['q3d_wave_breaking_factor'] = M_b
            # We use the updaed post advection fields to update the particle fields for the next substep or for the vertical advection
            self._update_particle_fields(post_advection_fields, indices=active_indices)
            if needs_next_substep_fields:
                self.particles['depth_avg_flow_velocity_magnitude'][active_indices] = np.hypot(
                    self.particles['depth_avg_flow_velocity_u'][active_indices],
                    self.particles['depth_avg_flow_velocity_v'][active_indices],
                )

            # Next we read the new local bed level, water depth, and skin roughness at the new x,y position for the vertical update.
            bed_level_new_active = np.asarray(self.particles['bed_level'], dtype=float)[active_indices]
            water_depth_new_active = np.maximum(
                np.nan_to_num(np.asarray(self.particles['water_depth'], dtype=float)[active_indices], nan=0.0),
                0.0,
            )
            skin_roughness_new_active = np.maximum(
                np.nan_to_num(
                    np.asarray(self.particles['skin_roughness_height'], dtype=float)[active_indices],
                    nan=0.0,
                ),
                0.0,
            )
            deposition_threshold_active = 0.25 * skin_roughness_new_active

            # Prepare verical scheme inputs.
            if vertical_update_scheme in {'centroid_floor', 'rouse_profile'}:
                z_c_new_active = np.nan_to_num(
                    np.asarray(self.particles['total_transport_centroid_elevation'], dtype=float)[active_indices],
                    nan=0.0,
                )
            else: # z_c_new_acive is only useful for the sceme that uses i (centroid floor and rouse profile).
                  # For the geometric scheme, we don't need it, so we just set it to zero, so that the function call has consistent arguments
                z_c_new_active = np.zeros(active_count, dtype=float)

            if vertical_update_scheme == 'rouse_profile':
                rouse_number_new_active = np.nan_to_num(
                    np.asarray(self.particles['rouse_number'], dtype=float)[active_indices],
                    nan=0.0,
                )
            else: # rouse_number_new_active is only useful for the sceme that uses it.
                  # For the other schemes, we don't need it, so we just set it to zero, so that the function call has consistent arguments
                rouse_number_new_active = np.zeros(active_count, dtype=float)

            z_p_after_vertical_advection = self._q3d_height_after_vertical_update(
                vertical_update_scheme,
                z_p_old=z_p_current_active,
                bed_level_old=bed_level_current_active,
                bed_level_new=bed_level_new_active,
                water_depth_new=water_depth_new_active,
                particle_w=particle_w_active,
                settling_velocity=settling_active,
                transport_centroid_elevation_new=z_c_new_active,
                rouse_number_new=rouse_number_new_active,
                dt=dt_sub,
                rng=random_source,
            )
            z[active_indices] = bed_level_new_active + z_p_after_vertical_advection

            # Deposit particles that return to the near-bed roughness threshold.
            # MacDonald treats near-bed particles as available or buried within
            # the active layer; this threshold is the local numerical criterion
            # for returning a particle from suspension to the bed.
            deposited_active = np.isfinite(z_p_after_vertical_advection) & (z_p_after_vertical_advection <= deposition_threshold_active)
            deposited_indices = active_indices[deposited_active]

            if deposited_indices.size:
                z[deposited_indices] = bed_level_new_active[deposited_active]
                z_p_after_vertical_advection[deposited_active] = 0.0
                is_deposited[deposited_indices] = True
                is_suspended[deposited_indices] = False
                deposited_now[deposited_indices] = True

            z_p_after_vertical_advection = np.maximum(z_p_after_vertical_advection, 0.0)
            burial_depth[active_indices] = 0.0
            is_buried[active_indices] = False
            is_suspended[active_indices] = z_p_after_vertical_advection > deposition_threshold_active
            is_suspended[deposited_indices] = False
            is_deposited[active_indices] = ~is_suspended[active_indices]
            # TO implement - somewhere here we should also eventually update the burial depth of particles that are deposited but not buried, and set their is_buried flag to False. This will allow them to be entrained again in the future without needing a separate burial update step to reset their state.

        # Enforce the Q3D vertical-state invariant before storing statuses:
        # suspended and deposited are mutually exclusive, and buried particles
        # are deposited bed-state particles that cannot be suspended.
        is_suspended = np.asarray(is_suspended, dtype=bool)
        is_deposited = np.asarray(is_deposited, dtype=bool)
        is_buried = np.asarray(is_buried, dtype=bool)
        eligible_buried = eligible & is_buried
        eligible_suspended = eligible & is_suspended
        is_suspended[eligible_buried] = False
        is_deposited[eligible_buried] = True
        is_deposited[eligible_suspended] = False

        bed_level = np.asarray(self.particles['bed_level'], dtype=float)
        height_above_bed = np.maximum(np.nan_to_num(z - bed_level, nan=0.0), 0.0)

        # ------------------------------------------------------------------
        # 6. Store updated particle state and diagnostics.
        # ------------------------------------------------------------------
        centroid_u = np.nan_to_num(
            self.particles.get('centroid_particle_velocity_u', np.zeros(n_particles, dtype=float)),
            nan=0.0,
        )
        centroid_v = np.nan_to_num(
            self.particles.get('centroid_particle_velocity_v', np.zeros(n_particles, dtype=float)),
            nan=0.0,
        )
        centroid_magnitude = np.nan_to_num(
            self.particles.get('centroid_particle_velocity_magnitude', np.zeros(n_particles, dtype=float)),
            nan=0.0,
        )
        self.particles['centroid_particle_velocity_x'] = centroid_u
        self.particles['centroid_particle_velocity_y'] = centroid_v
        self.particles['centroid_particle_velocity'] = centroid_magnitude
        self.particles['z'] = z
        self.particles['z_p'] = height_above_bed
        self.particles['q3d_first_substep_z_p'] = first_substep_z_p
        self.particles['first_substep_modified_centroid_particle_velocity_x'] = first_substep_modified_u
        self.particles['first_substep_modified_centroid_particle_velocity_y'] = first_substep_modified_v
        self.particles['first_substep_modified_centroid_particle_velocity'] = np.hypot(
            first_substep_modified_u,
            first_substep_modified_v,
        )
        self.particles['first_substep_horizontal_particle_velocity_x'] = first_substep_particle_u
        self.particles['first_substep_horizontal_particle_velocity_y'] = first_substep_particle_v
        self.particles['first_substep_horizontal_particle_velocity'] = np.hypot(first_substep_particle_u, first_substep_particle_v)
        self.particles['first_substep_vertical_particle_velocity'] = first_substep_particle_w
        if save_first_substep_diagnostics:
            self.particles['first_substep_horizontal_diffusion_velocity_x'] = first_substep_horizontal_diffusion_velocity_x
            self.particles['first_substep_horizontal_diffusion_velocity_y'] = first_substep_horizontal_diffusion_velocity_y
            self.particles['first_substep_horizontal_diffusion_velocity'] = np.hypot(
                first_substep_horizontal_diffusion_velocity_x,
                first_substep_horizontal_diffusion_velocity_y,
            )
            self.particles['first_substep_vertical_advection_velocity'] = first_substep_vertical_advection
            self.particles['first_substep_vertical_diffusion_velocity'] = first_substep_vertical_diffusion_velocity
            self.particles['first_substep_vertical_diffusion_coefficient'] = first_substep_vertical_diffusion
            self.particles['first_substep_horizontal_diffusion_coefficient'] = first_substep_horizontal_diffusion
            self.particles['first_substep_bed_level'] = first_substep_bed_level
            self.particles['first_substep_water_depth'] = first_substep_water_depth
            self.particles['first_substep_skin_roughness_height'] = first_substep_skin_roughness
            self.particles['first_substep_max_shear_velocity'] = first_substep_shear_velocity
            self.particles['first_substep_profile_roughness_height'] = first_substep_profile_roughness
            self.particles['first_substep_total_transport_centroid_elevation'] = first_substep_z_c
            self.particles['first_substep_q3d_velocity_deficit_coefficient'] = first_substep_deficit
            self.particles['first_substep_q3d_vertical_velocity_gradient'] = first_substep_vertical_velocity_gradient
            self.particles['first_substep_settling_velocity'] = first_substep_settling_velocity
            self.particles['first_substep_depth_avg_flow_velocity_magnitude'] = first_substep_flow_magnitude
            self.particles['first_substep_rouse_number'] = first_substep_rouse_number
        else:
            for field_name in (
                'first_substep_horizontal_diffusion_velocity_x',
                'first_substep_horizontal_diffusion_velocity_y',
                'first_substep_horizontal_diffusion_velocity',
                'first_substep_vertical_advection_velocity',
                'first_substep_vertical_diffusion_velocity',
                'first_substep_vertical_diffusion_coefficient',
                'first_substep_horizontal_diffusion_coefficient',
                'first_substep_bed_level',
                'first_substep_water_depth',
                'first_substep_skin_roughness_height',
                'first_substep_max_shear_velocity',
                'first_substep_profile_roughness_height',
                'first_substep_total_transport_centroid_elevation',
                'first_substep_q3d_velocity_deficit_coefficient',
                'first_substep_q3d_vertical_velocity_gradient',
                'first_substep_settling_velocity',
                'first_substep_depth_avg_flow_velocity_magnitude',
                'first_substep_rouse_number',
            ):
                self.particles.pop(field_name, None)
        self.particles['q3d_vertical_update_scheme_code'] = np.full(
            n_particles,
            vertical_scheme_codes[vertical_update_scheme],
            dtype=int,
        )
        self.particles['q3d_motion_substeps'] = np.full(n_particles, substeps, dtype=int)
        self.particles['status_suspended'] = is_suspended
        self.particles['status_deposited'] = is_deposited
        self.particles['status_buried'] = is_buried
        self.particles['status_available_for_entrainment'] = available
        self.particles['q3d_entrainment_probability'] = entrainment_probability
        self.particles['status_entrained_now'] = entrained_now
        self.particles['status_deposited_now'] = deposited_now
        self.particles['burial_depth'] = burial_depth
        self.particles['z_burial'] = bed_level - burial_depth

        is_mobile = eligible & is_suspended & ~is_deposited
        self.particles['status_mobile'] = is_mobile

    @staticmethod

    def _markov_settling_rate(
        settling_velocity,
        settling_height,
        shear_velocity,
        minimum_settling_height,
    ):
        """Return the two-state Markov settling rate and its turbulence correction."""
        minimum_height = float(minimum_settling_height)
        if not np.isfinite(minimum_height) or minimum_height <= 0.0:
            raise ValueError('minimum_settling_height must be positive and finite.')

        settling_velocity, settling_height, shear_velocity = np.broadcast_arrays(
            np.asarray(settling_velocity, dtype=float),
            np.asarray(settling_height, dtype=float),
            np.asarray(shear_velocity, dtype=float),
        )
        ws = np.maximum(np.nan_to_num(settling_velocity, nan=0.0), 0.0)
        height = np.maximum(
            np.nan_to_num(settling_height, nan=minimum_height, posinf=minimum_height, neginf=minimum_height),
            minimum_height,
        )
        u_star = np.maximum(np.nan_to_num(shear_velocity, nan=0.0), 0.0)
        turbulence_correction = np.divide(
            ws,
            np.hypot(ws, u_star),
            out=np.zeros_like(ws),
            where=(ws > 0.0) | (u_star > 0.0),
        )
        transition_rate = (ws / height) * turbulence_correction
        return transition_rate, turbulence_correction

    def sample_macdonald_2d_transition_fields(
        self,
        *,
        particle_velocity_field=None,
        shields_number_field=None,
        settling_height_field=None,
        shear_velocity_field=None,
        entrainment_frequency_field=None,
    ) -> None:
        """Sample entrainment and deposition inputs at the old particle positions."""
        if len(self.particles['x']) == 0:
            return

        scalar_fields = {}
        if shields_number_field is not None:
            scalar_fields['macdonald_2d_shields_number'] = shields_number_field
        if settling_height_field is not None:
            scalar_fields['macdonald_2d_settling_height'] = settling_height_field
        if shear_velocity_field is not None:
            scalar_fields['macdonald_2d_shear_velocity'] = shear_velocity_field
        if entrainment_frequency_field is not None:
            scalar_fields['macdonald_2d_entrainment_frequency'] = entrainment_frequency_field
        if scalar_fields:
            self._update_particle_fields(scalar_fields)
        if particle_velocity_field is not None:
            self._update_particle_flow_field('macdonald_2d_particle_velocity', particle_velocity_field)

    def update_macdonald_2d_entrainment(
        self,
        *,
        method,
        critical_shields_number,
        current_timestep,
        probability_law='poisson',
        rng=None,
    ) -> None:
        """Update deposited MacDonald 2D particles that enter suspension."""
        n_particles = len(self.particles['x'])
        if n_particles == 0:
            return
        if not np.isfinite(current_timestep) or current_timestep <= 0.0:
            raise ValueError('current_timestep must be positive and finite.')

        method = str(method or 'shields_threshold').strip().lower().replace('-', '_')
        if method not in {
            'shields_threshold',
            'non_zero_particle_velocity',
            'entrainment_frequency',
        }:
            raise ValueError(
                f'Unsupported MacDonald 2D entrainment method {method!r}. '
                'Expected shields_threshold, non_zero_particle_velocity, '
                'or entrainment_frequency.'
            )

        eligible = np.asarray(self.particles['status_eligible'], dtype=bool)
        deposited = np.asarray(self.particles['status_deposited'], dtype=bool).copy()
        suspended = np.asarray(self.particles['status_suspended'], dtype=bool).copy()
        candidates = eligible & deposited
        probability = np.zeros(n_particles, dtype=float)

        if method == 'shields_threshold':
            if 'macdonald_2d_shields_number' not in self.particles:
                raise ValueError(
                    "MacDonald 2D entrainment field 'macdonald_2d_shields_number' is missing."
                )
            if critical_shields_number is None:
                raise ValueError('critical_shields_number is required for shields_threshold entrainment.')
            local_shields = np.asarray(self.particles['macdonald_2d_shields_number'], dtype=float)
            critical_shields = np.broadcast_to(
                np.asarray(critical_shields_number, dtype=float),
                local_shields.shape,
            )
            entrained_now = (
                candidates
                & np.isfinite(local_shields)
                & np.isfinite(critical_shields)
                & (local_shields > critical_shields)
            )
        elif method == 'non_zero_particle_velocity':
            if 'macdonald_2d_particle_velocity_magnitude' not in self.particles:
                raise ValueError(
                    "MacDonald 2D entrainment field "
                    "'macdonald_2d_particle_velocity_magnitude' is missing."
                )
            particle_velocity = np.asarray(
                self.particles['macdonald_2d_particle_velocity_magnitude'],
                dtype=float,
            )
            nonzero_velocity = np.isfinite(particle_velocity) & (particle_velocity > 0.0)
            entrained_now = candidates & nonzero_velocity
        else:
            if 'macdonald_2d_entrainment_frequency' not in self.particles:
                raise ValueError('macdonald_2d_entrainment_frequency must be sampled before entrainment.')
            frequency = np.maximum(
                np.nan_to_num(
                    np.asarray(self.particles['macdonald_2d_entrainment_frequency'], dtype=float),
                    nan=0.0,
                ),
                0.0,
            )
            law = str(probability_law or 'poisson').strip().lower().replace('-', '_')
            if law == 'poisson':
                probability = -np.expm1(-frequency * float(current_timestep))
            elif law == 'linear':
                probability = frequency * float(current_timestep)
            else:
                raise ValueError("probability_law must be 'poisson' or 'linear'.")
            probability = np.clip(probability, 0.0, 1.0)
            random_source = np.random if rng is None else rng
            entrained_now = candidates & (random_source.random(n_particles) < probability)

        deposited[entrained_now] = False
        suspended[entrained_now] = True
        self.particles['status_deposited'] = deposited
        self.particles['status_suspended'] = suspended
        self.particles['status_entrained_now'] = entrained_now
        self.particles['macdonald_2d_entrainment_probability'] = probability
        self.particles['status_mobile'] = eligible & suspended & ~deposited

    def update_macdonald_2d_deposition(
        self,
        *,
        method,
        critical_shields_number,
        current_timestep,
        settling_velocity=None,
        minimum_settling_height=0.001,
        rng=None,
    ) -> None:
        """Evaluate MacDonald 2D suspension-to-bed transitions.

        MacDonald 2D has no explicit particle vertical coordinate. Deposition is
        therefore represented as a transition between the logical particle
        states ``status_suspended`` and ``status_deposited``. This method also
        updates ``status_mobile`` so deposited particles are excluded from the
        next horizontal position update.

        Two deposition rules are supported:

        ``shields_threshold``
            Deterministic mobility rule. An eligible suspended particle remains
            suspended only when its sampled Shields number is greater than its
            critical Shields number and its sampled transport velocity is
            nonzero. Otherwise it deposits. The simulation manager calls this
            rule before horizontal movement, so a particle deposited here does
            not move during the current timestep.

        ``markov_settling``
            Stochastic, memoryless settling rule. For each eligible mobile
            particle, the transition rate is calculated from settling velocity,
            settling height, and shear velocity. The settling probability over
            ``current_timestep`` is ``1 - exp(-rate * dt)``. The simulation
            manager calls this rule after horizontal movement, so deposition
            affects subsequent timesteps and does not undo movement already
            completed in the current timestep.

        All Eulerian inputs are sampled by
        ``sample_macdonald_2d_transition_fields`` at the particle's old
        horizontal position before movement. Only generally eligible particles
        can change suspension/deposition state; ineligible particles retain
        their previous physical state and are non-mobile.

        Parameters
        ----------
        method : str
            ``shields_threshold`` or ``markov_settling``.
        critical_shields_number : float or array
            Critical Shields number. Required only for ``shields_threshold``.
        current_timestep : float
            Positive particle timestep [s].
        settling_velocity : float or array, optional
            Particle settling velocity [m/s]. Required for ``markov_settling``.
        minimum_settling_height : float
            Positive lower bound applied to the settling height [m].
        rng : random generator, optional
            Random source used by ``markov_settling``; primarily useful for
            reproducible tests.
        """
        n_particles = len(self.particles['x'])
        if n_particles == 0:
            return

        method = str(method or 'shields_threshold').strip().lower().replace('-', '_')
        if method not in {'shields_threshold', 'markov_settling'}:
            raise ValueError(
                f'Unsupported MacDonald 2D deposition method {method!r}. '
                'Expected shields_threshold or markov_settling.'
            )
        if not np.isfinite(current_timestep) or current_timestep <= 0.0:
            raise ValueError('current_timestep must be positive and finite.')
        if method == 'shields_threshold':
            if critical_shields_number is None:
                raise ValueError('critical_shields_number is required for shields_threshold deposition.')
            required_sampled_fields = {
                'macdonald_2d_shields_number',
                'macdonald_2d_particle_velocity_magnitude',
            }
        else:
            if settling_velocity is None:
                raise ValueError('markov_settling requires settling_velocity.')
            required_sampled_fields = {
                'macdonald_2d_settling_height',
                'macdonald_2d_shear_velocity',
            }
        missing_fields = sorted(required_sampled_fields.difference(self.particles))
        if missing_fields:
            raise ValueError(
                'MacDonald 2D deposition fields must be sampled before movement; '
                f'missing {missing_fields}.'
            )

        # General eligibility protects unreleased, out-of-domain, dead,
        # non-transported, and buried particles from state changes.
        eligible = np.asarray(self.particles['status_eligible'], dtype=bool)

        previous_deposited = np.asarray(self.particles['status_deposited'], dtype=bool)
        settling_probability = np.zeros(n_particles, dtype=float)
        transition_rate = np.zeros(n_particles, dtype=float)

        if method == 'shields_threshold':
            local_shields = np.asarray(self.particles['macdonald_2d_shields_number'], dtype=float)
            particle_velocity = np.asarray(
                self.particles['macdonald_2d_particle_velocity_magnitude'],
                dtype=float,
            )
            critical_shields = np.broadcast_to(
                np.asarray(critical_shields_number, dtype=float),
                local_shields.shape,
            )
            above_threshold = (
                np.isfinite(local_shields)
                & np.isfinite(critical_shields)
                & (local_shields > critical_shields)
            )
            nonzero_velocity = np.isfinite(particle_velocity) & (particle_velocity > 0.0)
            # Entrainment is handled separately before this method. This branch
            # only tests particles that are currently suspended.
            deposited = previous_deposited.copy()
            suspended_candidates = eligible & ~previous_deposited
            deposited[suspended_candidates] = ~(
                above_threshold & nonzero_velocity
            )[suspended_candidates]
        else:
            transition_rate, _ = self._markov_settling_rate(
                settling_velocity,
                self.particles['macdonald_2d_settling_height'],
                self.particles['macdonald_2d_shear_velocity'],
                minimum_settling_height,
            )
            settling_probability = np.clip(
                -np.expm1(-transition_rate * float(current_timestep)),
                0.0,
                1.0,
            )
            # Markov settling applies only to particles that actually moved (or
            # were eligible to move) during this timestep.
            moving = eligible & np.asarray(self.particles['status_mobile'], dtype=bool)
            random_source = np.random if rng is None else rng
            settles = moving & (random_source.random(n_particles) < settling_probability)
            deposited = previous_deposited.copy()
            deposited[moving] = settles[moving]

        # Synchronize the mutually exclusive logical bed/water-column states.
        suspended = np.asarray(self.particles['status_suspended'], dtype=bool).copy()
        suspended[eligible] = ~deposited[eligible]
        self.particles['status_deposited'] = deposited
        self.particles['status_deposited_now'] = eligible & deposited & ~previous_deposited
        self.particles['status_suspended'] = suspended
        self.particles['status_mobile'] = eligible & suspended
        self.particles['macdonald_2d_settling_transition_rate'] = transition_rate
        self.particles['macdonald_2d_settling_probability'] = settling_probability
