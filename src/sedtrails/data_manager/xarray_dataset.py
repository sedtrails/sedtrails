import numpy as np
import xarray as xr


def create_sedtrails_dataset(
    N_particles,
    N_populations,
    N_timesteps,
    N_flowfields,
    name_strlen=24,
    q3d_diagnostics='minimal',
):
    ds = xr.Dataset(
        {
            # Population metadata - initialize with empty/default values
            'population_name': (('n_populations', 'name_strlen'), np.empty((N_populations, name_strlen), dtype='S1')),
            'population_particle_type': ('n_populations', np.zeros(N_populations, dtype=int)),
            'population_start_idx': ('n_populations', np.zeros(N_populations, dtype=int)),
            'population_count': ('n_populations', np.zeros(N_populations, dtype=int)),
            # Trajectory metadata
            'trajectory_id': (('n_particles', 'name_strlen'), np.empty((N_particles, name_strlen), dtype='S1')),
            'population_id': ('n_particles', np.zeros(N_particles, dtype=int)),
            # Core trajectory variables - initialize with NaN to indicate unset values
            'time': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'x': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'y': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'z': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)), # absolute particle elevation in the model vertical datum, positive upwards
            'z_burial': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'z_p': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'burial_depth': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)), # depth below the bed surface (z=0) that the particle is buried (if buried) positive downwards
            'mixing_depth': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            # Q3D diagnostics. These velocity/parameter fields are stored from
            # the first Q3D substep; x/y/z/status are stored after all substeps.
            'q3d_diagnostic_z_p_first_substep': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'centroid_particle_velocity_x': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'centroid_particle_velocity_y': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'centroid_particle_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'modified_centroid_particle_velocity_x': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'modified_centroid_particle_velocity_y': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'modified_centroid_particle_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'horizontal_particle_velocity_x': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'horizontal_particle_velocity_y': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'horizontal_particle_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'horizontal_diffusion_velocity_x': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'horizontal_diffusion_velocity_y': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'horizontal_diffusion_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'vertical_advection_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'vertical_diffusion_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'vertical_particle_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'horizontal_diffusion_coefficient': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'vertical_diffusion_coefficient': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_bed_level': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_water_depth': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_skin_roughness_height': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_max_shear_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_total_roughness_height': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_total_transport_centroid_elevation': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_q3d_velocity_deficit_coefficient': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_q3d_vertical_velocity_gradient': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_settling_velocity': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_q3d_flow_magnitude': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'diagnostic_rouse_number': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'turbulent_shields_number': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'critical_shields_number': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'q3d_entrainment_probability': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'q3d_entrainment_height_above_bed': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'q3d_vertical_update_scheme_code': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'q3d_motion_substeps': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            # Status variables - initialize with zeros (False/not active)
            'is_alive': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_exposed': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_inside': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_picked_up': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_released': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_suspended': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_deposited': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_buried': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_mobile': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'is_available_for_entrainment': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'entrained_now': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'deposited_now': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            # Transport distances per flow field
            'covered_distance': (
                ('n_flowfields', 'n_particles', 'n_timesteps'),
                np.zeros((N_flowfields, N_particles, N_timesteps)),
            ),
            'flowfield_name': (('n_flowfields', 'name_strlen'), np.empty((N_flowfields, name_strlen), dtype='S1')),
            'use_transport_fields': ('name_strlen', np.empty((name_strlen), dtype='S1')),
            'max_suspended_velocity_factor': ((), np.nan),
        },
        coords={
            'n_particles': np.arange(N_particles),
            'n_populations': np.arange(N_populations),
            'n_timesteps': np.arange(N_timesteps),
            'n_flowfields': np.arange(N_flowfields),
            'name_strlen': np.arange(name_strlen),
        },
    )

    q3d_diagnostics = str(q3d_diagnostics or 'minimal').strip().lower().replace('-', '_')
    if q3d_diagnostics != 'full':
        q3d_full_only_fields = (
            'horizontal_diffusion_velocity_x',
            'horizontal_diffusion_velocity_y',
            'horizontal_diffusion_velocity',
            'vertical_advection_velocity',
            'vertical_diffusion_velocity',
            'horizontal_diffusion_coefficient',
            'vertical_diffusion_coefficient',
            'diagnostic_bed_level',
            'diagnostic_water_depth',
            'diagnostic_skin_roughness_height',
            'diagnostic_max_shear_velocity',
            'diagnostic_total_roughness_height',
            'diagnostic_total_transport_centroid_elevation',
            'diagnostic_q3d_velocity_deficit_coefficient',
            'diagnostic_q3d_vertical_velocity_gradient',
            'diagnostic_settling_velocity',
            'diagnostic_q3d_flow_magnitude',
            'diagnostic_rouse_number',
            'turbulent_shields_number',
            'critical_shields_number',
        )
        drop_names = [name for name in q3d_full_only_fields if name in ds]
        if drop_names:
            ds = ds.drop_vars(drop_names)

    return ds


def populate_population_metadata(ds, populations):
    """
    Populate population metadata in the xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset to populate
    populations : list
        List of population objects from the simulation
    """
    particle_offset = 0

    for pop_idx, population in enumerate(populations):
        # Population metadata
        pop_name = getattr(population, 'name', f'population_{pop_idx}')
        max_len = ds.sizes['name_strlen']
        truncated_name = pop_name[:max_len].ljust(max_len)
        name_array = np.array(list(truncated_name), dtype='S1')
        ds['population_name'][pop_idx, :] = name_array

        ds['population_particle_type'][pop_idx] = getattr(population, 'particle_type', 0)
        ds['population_start_idx'][pop_idx] = particle_offset
        ds['population_count'][pop_idx] = len(population.particles['x'])

        # Assign population ID to particles
        num_particles = len(population.particles['x'])
        ds['population_id'][particle_offset : particle_offset + num_particles] = pop_idx

        # Generate trajectory IDs
        for i in range(num_particles):
            traj_id = f'traj_{particle_offset + i}'
            max_len = ds.sizes['name_strlen']
            truncated_traj_id = traj_id[:max_len].ljust(max_len)
            traj_array = np.array(list(truncated_traj_id), dtype='S1')
            ds['trajectory_id'][particle_offset + i, :] = traj_array

        particle_offset += num_particles


def populate_flowfield_metadata(ds, flow_field_names):
    """
    Populate flowfield metadata in the xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset to populate
    flow_field_names : list
        List of flow field names
    """
    for ff_idx, ff_name in enumerate(flow_field_names):
        # Truncate name if it's longer than name_strlen, or pad if shorter
        max_len = ds.sizes['name_strlen']
        truncated_name = ff_name[:max_len].ljust(max_len)
        name_array = np.array(list(truncated_name), dtype='S1')
        ds['flowfield_name'][ff_idx, :] = name_array


def collect_timestep_data(ds, populations, timestep, current_time):
    """
    Collect data from all populations for a specific timestep.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset to populate
    populations : list
        List of population objects from the simulation
    timestep : int
        Current timestep index
    current_time : float
        Current simulation time
    """
    particle_offset = 0

    for population in populations:
        num_particles = len(population.particles['x'])
        particle_slice = slice(particle_offset, particle_offset + num_particles)

        # Core trajectory variables
        ds['time'][particle_slice, timestep] = current_time
        ds['x'][particle_slice, timestep] = population.particles['x']
        ds['y'][particle_slice, timestep] = population.particles['y']
        ds['z'][particle_slice, timestep] = population.particles.get('z', np.zeros(num_particles))
        ds['z_burial'][particle_slice, timestep] = population.particles.get(
            'z_burial', np.full(num_particles, np.nan)
        )
        ds['z_p'][particle_slice, timestep] = population.particles.get('z_p', np.full(num_particles, np.nan))
        ds['burial_depth'][particle_slice, timestep] = population.particles['burial_depth']
        # ds['mixing_depth'][particle_slice, timestep] = population.particles['mixing_depth'] # TODO: implement mixing depth tracking

        # Status variables (assuming these exist in population.particles)
        ds['is_alive'][particle_slice, timestep] = population.particles.get(
            'is_alive', np.ones(num_particles, dtype=int)
        )
        ds['is_exposed'][particle_slice, timestep] = population.particles.get(
            'is_exposed', np.zeros(num_particles, dtype=int)
        )
        ds['is_inside'][particle_slice, timestep] = population.particles.get(
            'is_inside', np.ones(num_particles, dtype=int)
        )
        ds['is_picked_up'][particle_slice, timestep] = population.particles.get(
            'is_picked_up', np.zeros(num_particles, dtype=int)
        )
        ds['is_released'][particle_slice, timestep] = population.particles.get(
            'is_released', np.ones(num_particles, dtype=int)
        )
        ds['is_suspended'][particle_slice, timestep] = population.particles.get(
            'is_suspended', np.ones(num_particles, dtype=int)
        )
        ds['is_deposited'][particle_slice, timestep] = population.particles.get(
            'is_deposited', np.zeros(num_particles, dtype=int)
        )
        ds['is_buried'][particle_slice, timestep] = population.particles.get(
            'is_buried', np.zeros(num_particles, dtype=int)
        )
        ds['is_mobile'][particle_slice, timestep] = population.particles.get(
            'is_mobile', np.zeros(num_particles, dtype=int)
        )

        q3d_float_fields = (
            'q3d_diagnostic_z_p_first_substep',
            'centroid_particle_velocity_x',
            'centroid_particle_velocity_y',
            'centroid_particle_velocity',
            'modified_centroid_particle_velocity_x',
            'modified_centroid_particle_velocity_y',
            'modified_centroid_particle_velocity',
            'horizontal_particle_velocity_x',
            'horizontal_particle_velocity_y',
            'horizontal_particle_velocity',
            'horizontal_diffusion_velocity_x',
            'horizontal_diffusion_velocity_y',
            'horizontal_diffusion_velocity',
            'vertical_advection_velocity',
            'vertical_diffusion_velocity',
            'vertical_particle_velocity',
            'horizontal_diffusion_coefficient',
            'vertical_diffusion_coefficient',
            'diagnostic_bed_level',
            'diagnostic_water_depth',
            'diagnostic_skin_roughness_height',
            'diagnostic_max_shear_velocity',
            'diagnostic_total_roughness_height',
            'diagnostic_total_transport_centroid_elevation',
            'diagnostic_q3d_velocity_deficit_coefficient',
            'diagnostic_q3d_vertical_velocity_gradient',
            'diagnostic_settling_velocity',
            'diagnostic_q3d_flow_magnitude',
            'diagnostic_rouse_number',
            'turbulent_shields_number',
            'critical_shields_number',
            'q3d_entrainment_probability',
            'q3d_entrainment_height_above_bed',
        )
        for field_name in q3d_float_fields:
            if field_name not in ds:
                continue
            ds[field_name][particle_slice, timestep] = population.particles.get(
                field_name,
                np.full(num_particles, np.nan),
            )

        q3d_int_fields = (
            'q3d_vertical_update_scheme_code',
            'q3d_motion_substeps',
            'is_available_for_entrainment',
            'entrained_now',
            'deposited_now',
        )
        for field_name in q3d_int_fields:
            if field_name not in ds:
                continue
            ds[field_name][particle_slice, timestep] = population.particles.get(
                field_name,
                np.zeros(num_particles, dtype=int),
            )

        particle_offset += num_particles
