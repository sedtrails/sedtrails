"""
Module for visualizing particle trajectories from SedTRAILS NetCDF output files.
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def _decode_netcdf_name(raw_value) -> str:
    """Decode a NetCDF string value stored as bytes, fixed-width bytes, or char arrays."""
    if raw_value is None:
        return ''

    if isinstance(raw_value, str):
        return raw_value.strip()

    if isinstance(raw_value, (bytes, np.bytes_)):
        return raw_value.decode('utf-8', errors='ignore').strip('\x00').strip()

    arr = np.asarray(raw_value)

    # Char-array representation (e.g., dtype='|S1' with trailing nulls)
    if arr.ndim > 0 and arr.size > 0 and arr.dtype.kind in ('S', 'U'):
        flattened = arr.ravel().tolist()
        chars = []
        for item in flattened:
            if isinstance(item, (bytes, np.bytes_)):
                text = item.decode('utf-8', errors='ignore')
            else:
                text = str(item)
            text = text.replace('\x00', '')
            if text:
                chars.append(text)
        return ''.join(chars).strip()

    return str(raw_value).strip()


def read_netcdf(results_file_path: Path) -> xr.Dataset:
    """Read a SedTRAILS NetCDF file and return the xarray Dataset.

    Parameters
    ----------
    results_file_path : pathlib.Path
        Path to the SedTRAILS NetCDF file.

    Returns
    -------
    xr.Dataset
        The xarray Dataset containing the SedTRAILS results.

    """
    results_file_path = Path(results_file_path)
    if not results_file_path.exists():
        raise FileNotFoundError(f"NetCDF file '{results_file_path}' not found.")
    ds = xr.open_dataset(results_file_path, engine='netcdf4', decode_times=False)
    return ds


def _reference_datetime64(ds: xr.Dataset, time_var: xr.DataArray) -> np.datetime64 | None:
    for value in (
        time_var.attrs.get('reference_date'),
        ds.attrs.get('reference_date'),
    ):
        if value is not None:
            return np.datetime64(str(value), 'ns')

    for units in (
        time_var.attrs.get('units'),
        time_var.encoding.get('units'),
        ds.attrs.get('time_units'),
        ds.attrs.get('units'),
    ):
        match = re.match(r'^\s*seconds\s+since\s+(.+?)\s*$', str(units), flags=re.IGNORECASE)
        if match:
            return np.datetime64(match.group(1), 'ns')

    return None


def _time_values_as_seconds(ds: xr.Dataset, time_var: xr.DataArray) -> np.ndarray:
    values = np.asarray(time_var.values)
    if np.issubdtype(values.dtype, np.datetime64):
        values_ns = values.astype('datetime64[ns]')
        valid = ~np.isnat(values_ns)
        reference = _reference_datetime64(ds, time_var)
        if reference is None and np.any(valid):
            reference = values_ns[valid].min()
        if reference is None:
            return np.full(values_ns.shape, np.nan, dtype=float)

        seconds = np.full(values_ns.shape, np.nan, dtype=float)
        seconds[valid] = (values_ns[valid] - reference) / np.timedelta64(1, 's')
        return seconds

    return np.asarray(values, dtype=float)


def _trajectory_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y/time arrays in plotting shape: (n_particles, n_timesteps)."""
    x_var = ds['x']
    y_var = ds['y']

    if x_var.ndim == 1 and y_var.ndim == 1:
        x_data = np.asarray(x_var.values, dtype=float)[:, np.newaxis]
        y_data = np.asarray(y_var.values, dtype=float)[:, np.newaxis]
    elif (
        x_var.ndim == 2
        and y_var.ndim == 2
        and 'time' in ds
        and ds['time'].ndim == 1
        and x_var.dims[0] == ds['time'].dims[0]
        and y_var.dims[0] == ds['time'].dims[0]
        and x_var.shape == y_var.shape
    ):
        x_data = np.asarray(x_var.values, dtype=float).T
        y_data = np.asarray(y_var.values, dtype=float).T
    else:
        raise ValueError(
            "Expected SedTRAILS time-major trajectory arrays shaped as "
            "(n_timesteps, n_particles), or 1D checkpoint arrays."
        )

    n_particles, n_timesteps = x_data.shape
    if 'time' not in ds:
        time_data = np.broadcast_to(np.arange(n_timesteps, dtype=float), (n_particles, n_timesteps))
    elif ds['time'].ndim == 0:
        time_value = _time_values_as_seconds(ds, ds['time'])
        time_data = np.full((n_particles, n_timesteps), float(time_value))
    elif ds['time'].ndim == 1:
        time_values = _time_values_as_seconds(ds, ds['time'])
        if time_values.size == n_timesteps:
            time_data = np.broadcast_to(time_values, (n_particles, n_timesteps))
        else:
            raise ValueError("'time' length does not match the timestep dimension.")
    else:
        raise ValueError("Unsupported 'time' variable shape for trajectory plotting.")

    return x_data, y_data, time_data


def plot_trajectories(ds, save_plot=False, output_dir=None):
    """Plot particle trajectories from a SedTRAILS dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing particle trajectory variables, including ``x``,
        ``y``, and ``time``.
    save_plot : bool, optional
        If true, save the generated figure as ``particle_trajectories.png``.
    output_dir : str or pathlib.Path, optional
        Directory where the figure is saved. If omitted, the source dataset
        directory is used when available.

    Notes
    -----
    This function creates a four-panel matplotlib figure and displays it with
    ``plt.show()``. It does not return the figure or axes.
    """

    # Extract trajectory data
    x_data, y_data, time_data = _trajectory_arrays(ds)

    n_particles, n_timesteps = x_data.shape

    print(f'\nPlotting trajectories for {n_particles} particles over {n_timesteps} timesteps...')

    # Create the plot with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Extract population information for color coding
    population_ids = ds['population_id'].values if 'population_id' in ds else np.zeros(n_particles, dtype=int)
    n_populations = int(ds.sizes['n_populations']) if 'n_populations' in ds.sizes else 1

    # Get population names
    population_names = []
    if 'population_name' in ds:
        population_var = ds['population_name']
        n_available_names = int(population_var.sizes.get('n_populations', population_var.shape[0]))
        n_to_decode = min(n_populations, n_available_names)

        for i in range(n_to_decode):
            # Handle both 1D fixed-width strings and 2D character arrays
            if population_var.ndim == 1:
                raw_name = population_var[i].values
            else:
                raw_name = population_var[i, :].values
            decoded = _decode_netcdf_name(raw_name)
            population_names.append(decoded or f'Population {i}')

        if len(population_names) < n_populations:
            population_names.extend([f'Population {i}' for i in range(len(population_names), n_populations)])
    else:
        population_names = [f'Population {i}' for i in range(n_populations)]

    # Define colors for populations
    try:
        pop_cmap = plt.get_cmap('Set1')
        pop_colors = pop_cmap(np.linspace(0, 1, n_populations))
    except (AttributeError, ValueError):
        # Fallback to basic colors
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        pop_colors = [base_colors[i % len(base_colors)] for i in range(n_populations)]

    # Plot 1: All trajectories on spatial map (individual particle colors)
    ax1.set_title(f'(a) Particle Trajectories - Individual Colors (n={n_particles})')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')

    # Plot each particle trajectory
    try:
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, n_particles))
    except (AttributeError, ValueError):
        # Fallback to a basic color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [color_cycle[i % len(color_cycle)] for i in range(n_particles)]

    for i in range(n_particles):
        # Filter out NaN values (particles may not exist for all timesteps)
        mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
        if np.any(mask):
            x_traj = x_data[i, mask]
            y_traj = y_data[i, mask]

            # Plot trajectory line
            ax1.plot(
                x_traj,
                y_traj,
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=f'Particle {i}' if n_particles <= 10 else None,
            )

            # Mark start and end points
            # start point marker
            ax1.scatter(
                x_traj[0], y_traj[0], color=colors[i], marker='x', s=50, linewidth=1, zorder=5
            )
            # end point marker
            ax1.scatter(
                x_traj[-1], y_traj[-1], color=colors[i], marker='o', s=50, edgecolor='black', linewidth=1, zorder=5
            )

    # Add legend if few particles
    if n_particles <= 10:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Plot 2: Time series of distances from initial position
    ax2.set_title('(b) Distance from Initial Position vs Time')
    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('Distance from Initial Position [m]')

    # Find the minimum time across all particles to use as reference
    min_time = np.nanmin(time_data) if np.any(np.isfinite(time_data)) else 0.0

    for i in range(n_particles):
        # Calculate distance from initial position for each timestep
        mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
        if np.any(mask):
            x_traj = x_data[i, mask]
            y_traj = y_data[i, mask]
            time_traj = time_data[i, mask]

            # Get initial position (first valid point)
            x0 = x_traj[0]
            y0 = y_traj[0]

            # Convert time to hours relative to simulation start
            time_hours = (time_traj - min_time) / 3600.0

            # Calculate distance from initial position
            distances = np.sqrt((x_traj - x0) ** 2 + (y_traj - y0) ** 2)
            ax2.plot(time_hours, distances, color=colors[i], alpha=0.7, linewidth=1)

    ax2.grid(True, alpha=0.3)

    # Plot 3: Trajectories colored by population
    ax3.set_title('(c) Particle Trajectories - Colored by Population')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')

    # Plot trajectories grouped by population
    for pop_idx in range(n_populations):
        particles_in_pop = []
        # Find particles belonging to this population
        for i in range(n_particles):
            if population_ids[i] == pop_idx:
                particles_in_pop.append(i)

        # Plot trajectories for this population
        for i in particles_in_pop:
            mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
            if np.any(mask):
                x_traj = x_data[i, mask]
                y_traj = y_data[i, mask]

                # Plot trajectory line with population color
                ax3.plot(
                    x_traj,
                    y_traj,
                    color=pop_colors[pop_idx],
                    alpha=0.7,
                    linewidth=1,
                    label=population_names[pop_idx] if i == particles_in_pop[0] else '',
                )

                # Mark start and end points
                ax3.scatter(
                    x_traj[0],
                    y_traj[0],
                    color=pop_colors[pop_idx],
                    marker='x',
                    s=50,
                    linewidth=1,
                    zorder=5,
                )
                ax3.scatter(
                    x_traj[-1],
                    y_traj[-1],
                    color=pop_colors[pop_idx],
                    marker='o',
                    s=50,
                    edgecolor='black',
                    linewidth=1,
                    zorder=5,
                )

    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    # Plot 4: Distance from initial position by population with statistics
    ax4.set_title('(d) Distance from Initial Position by Population')
    ax4.set_xlabel('Time [hours]')
    ax4.set_ylabel('Distance from Initial Position [m]')

    # Create containers for population statistics
    population_stats = {}

    # First pass: collect all data for each population
    for pop_idx in range(n_populations):
        population_stats[pop_idx] = {
            'times': [],
            'distances': [],
            'name': population_names[pop_idx],
            'color': pop_colors[pop_idx],
        }

        # Find particles belonging to this population
        particles_in_pop = [i for i in range(n_particles) if population_ids[i] == pop_idx]

        # Plot individual particle distances for this population
        for i in particles_in_pop:
            mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
            if np.any(mask):
                x_traj = x_data[i, mask]
                y_traj = y_data[i, mask]
                time_traj = time_data[i, mask]

                # Get initial position
                x0 = x_traj[0]
                y0 = y_traj[0]

                # Convert time to hours relative to simulation start
                time_hours = (time_traj - min_time) / 3600.0

                # Calculate distance from initial position
                distances = np.sqrt((x_traj - x0) ** 2 + (y_traj - y0) ** 2)

                # Plot individual particle line (thin, transparent)
                ax4.plot(time_hours, distances, color=pop_colors[pop_idx], alpha=0.3, linewidth=0.8)

                # Store data for statistics
                population_stats[pop_idx]['times'].append(time_hours)
                population_stats[pop_idx]['distances'].append(distances)

    # Second pass: compute and plot population statistics
    for pop_idx in range(n_populations):
        if population_stats[pop_idx]['times']:
            # Create a common time grid for interpolation
            all_times = np.concatenate(population_stats[pop_idx]['times'])
            min_t, max_t = np.min(all_times), np.max(all_times)
            common_time = np.linspace(min_t, max_t, 100)

            # Interpolate all particle distances onto common time grid
            interpolated_distances = []
            for time_arr, dist_arr in zip(
                population_stats[pop_idx]['times'], population_stats[pop_idx]['distances'], strict=True
            ):
                if len(time_arr) > 1:  # Need at least 2 points for interpolation
                    interp_dist = np.interp(common_time, time_arr, dist_arr)
                    interpolated_distances.append(interp_dist)

            if interpolated_distances:
                # Convert to array for easy statistics
                distances_array = np.array(interpolated_distances)

                # Compute mean and standard deviation
                mean_distances = np.mean(distances_array, axis=0)
                std_distances = np.std(distances_array, axis=0)

                # Plot mean line (thick)
                ax4.plot(
                    common_time,
                    mean_distances,
                    color=pop_colors[pop_idx],
                    linewidth=3,
                    label=f'{population_names[pop_idx]} (mean)',
                )

                # Plot standard deviation bands
                ax4.fill_between(
                    common_time,
                    mean_distances - std_distances,
                    mean_distances + std_distances,
                    color=pop_colors[pop_idx],
                    alpha=0.2,
                    label=f'{population_names[pop_idx]} (+/-1 std)',
                )

    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        if output_dir is None or output_dir == '.':
            # Extract the source file path from the dataset encoding
            source_file = ds.encoding.get('source', '.')
            output_dir = Path(source_file).parent
        else:
            output_dir = Path(output_dir)

        output_file = output_dir / 'particle_trajectories.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {output_file}')

    plt.show()
