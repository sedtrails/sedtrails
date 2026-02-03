import numpy as np
import xarray as xr


def create_sedtrails_dataset(N_particles, N_populations, N_timesteps, N_flowfields, name_strlen=24):
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
            'z': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'burial_depth': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            'mixing_depth': (('n_particles', 'n_timesteps'), np.full((N_particles, N_timesteps), np.nan)),
            # Status variables - initialize with zeros (False/not active)
            'status_alive': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'status_buried': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'status_domain': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'status_transported': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'status_released': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            'status_mobile': (('n_particles', 'n_timesteps'), np.zeros((N_particles, N_timesteps), dtype=int)),
            # Transport distances per flow field
            'covered_distance': (
                ('n_flowfields', 'n_particles', 'n_timesteps'),
                np.zeros((N_flowfields, N_particles, N_timesteps)),
            ),
            'flowfield_name': (('n_flowfields', 'name_strlen'), np.empty((N_flowfields, name_strlen), dtype='S1')),
            'use_transport_fields': ('name_strlen', np.empty((name_strlen), dtype='S1')),
        },
        coords={
            'n_particles': np.arange(N_particles),
            'n_populations': np.arange(N_populations),
            'n_timesteps': np.arange(N_timesteps),
            'n_flowfields': np.arange(N_flowfields),
            'name_strlen': np.arange(name_strlen),
        },
    )

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
        ds['burial_depth'][particle_slice, timestep] = population.particles['burial_depth']
        # ds['mixing_depth'][particle_slice, timestep] = population.particles['mixing_depth'] # TODO: implement mixing depth tracking

        # Status variables (assuming these exist in population.particles)
        ds['status_alive'][particle_slice, timestep] = population.particles.get(
            'status_alive', np.ones(num_particles, dtype=int)
        )
        ds['status_buried'][particle_slice, timestep] = population.particles.get(
            'status_buried', np.zeros(num_particles, dtype=int)
        )
        ds['status_domain'][particle_slice, timestep] = population.particles.get(
            'status_domain', np.ones(num_particles, dtype=int)
        )
        ds['status_transported'][particle_slice, timestep] = population.particles.get(
            'status_transported', np.zeros(num_particles, dtype=int)
        )
        ds['status_released'][particle_slice, timestep] = population.particles.get(
            'status_released', np.ones(num_particles, dtype=int)
        )
        ds['status_mobile'][particle_slice, timestep] = population.particles.get(
            'status_mobile', np.zeros(num_particles, dtype=int)
        )

        particle_offset += num_particles
