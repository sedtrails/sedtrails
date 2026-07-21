"""
Module for visualizing particle trajectories from SedTRAILS NetCDF output files.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection


def _decode_netcdf_name(raw_value) -> str:
    """Decode a NetCDF string value stored as bytes, fixed-width bytes, or char arrays."""
    if raw_value is None:
        return ''

    if isinstance(raw_value, str):
        return raw_value.strip()

    if isinstance(raw_value, (bytes, np.bytes_)):
        return raw_value.decode('utf-8', errors='ignore').replace('\x00', '').strip()

    arr = np.asarray(raw_value)
    if arr.ndim == 0:
        return _decode_netcdf_name(arr.item())


    # Scalar numpy bytes/str (0-dim array or item)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, (bytes, np.bytes_)):
            return item.decode('utf-8', errors='ignore').strip('\x00').strip()
        return str(item).strip()

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


def _infer_particle_dim(ds: xr.Dataset) -> str | None:
    """Infer the particle dimension used by the trajectory arrays."""
    if 'x' not in ds:
        return None

    x_var = ds['x']
    if x_var.ndim == 1:
        return x_var.dims[0]
    if x_var.ndim == 2 and 'time' in ds and ds['time'].ndim == 1 and x_var.dims[0] == ds['time'].dims[0]:
        return x_var.dims[1]
    if 'n_particles' in ds.sizes:
        return 'n_particles'
    if 'population_id' in ds and ds['population_id'].ndim == 1:
        return ds['population_id'].dims[0]
    return None


def _particle_count(ds: xr.Dataset) -> int:
    particle_dim = _infer_particle_dim(ds)
    if particle_dim is None:
        return 0
    return int(ds.sizes[particle_dim])


def _select_sample_indices(
    n_particles: int,
    population_ids: np.ndarray | None = None,
    *,
    max_particles: int | None = None,
    sample_fraction: float | None = None,
    sample_seed: int = 0,
) -> np.ndarray:
    """Return deterministic particle indices for plotting."""
    if max_particles is not None and sample_fraction is not None:
        raise ValueError("'max_particles' and 'sample_fraction' are mutually exclusive.")
    if max_particles is not None and max_particles < 1:
        raise ValueError("'max_particles' must be at least 1.")
    if sample_fraction is not None and not (0.0 < sample_fraction <= 1.0):
        raise ValueError("'sample_fraction' must be greater than 0 and less than or equal to 1.")

    if n_particles <= 0:
        return np.array([], dtype=int)
    if max_particles is None and sample_fraction is None:
        return np.arange(n_particles, dtype=int)

    if max_particles is not None:
        target = min(int(max_particles), n_particles)
    else:
        target = int(np.ceil(n_particles * float(sample_fraction)))
        target = max(1, min(target, n_particles))

    if target >= n_particles:
        return np.arange(n_particles, dtype=int)

    rng = np.random.default_rng(sample_seed)
    if population_ids is None or len(population_ids) != n_particles:
        return np.sort(rng.choice(n_particles, size=target, replace=False))

    selected: list[np.ndarray] = []
    unique_populations = np.unique(population_ids)
    population_indices = [np.flatnonzero(population_ids == pop_id) for pop_id in unique_populations]
    ideal_counts = np.array([len(indices) * target / n_particles for indices in population_indices], dtype=float)
    counts = np.floor(ideal_counts).astype(int)

    if target >= len(population_indices):
        counts = np.maximum(counts, 1)

    overflow = int(counts.sum() - target)
    if overflow > 0:
        removable_order = np.argsort(ideal_counts - counts)
        for idx in removable_order:
            if overflow == 0:
                break
            minimum = 1 if target >= len(population_indices) else 0
            if counts[idx] > minimum:
                counts[idx] -= 1
                overflow -= 1

    remainder = int(target - counts.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(ideal_counts - np.floor(ideal_counts)))
        for idx in fractional_order:
            if remainder == 0:
                break
            capacity = len(population_indices[idx]) - counts[idx]
            if capacity > 0:
                counts[idx] += 1
                remainder -= 1

    for indices, count in zip(population_indices, counts, strict=True):
        if count > 0:
            selected.append(rng.choice(indices, size=min(count, len(indices)), replace=False))

    if not selected:
        return np.array([], dtype=int)

    selected_indices = np.unique(np.concatenate(selected).astype(int))
    if selected_indices.size < target:
        remaining_pool = np.setdiff1d(np.arange(n_particles, dtype=int), selected_indices, assume_unique=True)
        if remaining_pool.size:
            extra = rng.choice(
                remaining_pool, size=min(target - selected_indices.size, remaining_pool.size), replace=False
            )
            selected_indices = np.concatenate((selected_indices, extra))
    elif selected_indices.size > target:
        selected_indices = rng.choice(selected_indices, size=target, replace=False)

    return np.sort(selected_indices)


def _sample_dataset(
    ds: xr.Dataset,
    *,
    max_particles: int | None = None,
    sample_fraction: float | None = None,
    sample_seed: int = 0,
) -> tuple[xr.Dataset, int, int]:
    """Return a dataset sampled along the particle dimension before materialization."""
    n_total = _particle_count(ds)
    population_ids = None
    if 'population_id' in ds:
        population_ids = np.asarray(ds['population_id'].values)

    indices = _select_sample_indices(
        n_total,
        population_ids,
        max_particles=max_particles,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
    )
    if len(indices) == n_total:
        return ds, n_total, n_total

    particle_dim = _infer_particle_dim(ds)
    if particle_dim is None:
        return ds, n_total, n_total
    return ds.isel({particle_dim: indices}), n_total, len(indices)


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
            'Expected SedTRAILS time-major trajectory arrays shaped as '
            '(n_timesteps, n_particles), or 1D checkpoint arrays.'
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


def _decode_population_names(ds: xr.Dataset, n_populations: int) -> list[str]:
    population_names = []
    if 'population_name' in ds:
        population_var = ds['population_name']
        n_available_names = int(population_var.sizes.get('n_populations', population_var.shape[0]))
        n_to_decode = min(n_populations, n_available_names)

        for i in range(n_to_decode):
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

    return population_names


def _population_colors(n_populations: int):
    try:
        pop_cmap = plt.get_cmap('Set1')
        return pop_cmap(np.linspace(0, 1, n_populations))
    except (AttributeError, ValueError):
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        return [base_colors[i % len(base_colors)] for i in range(n_populations)]


def _particle_colors(n_particles: int):
    try:
        cmap = plt.get_cmap('viridis')
        return cmap(np.linspace(0, 1, n_particles))
    except (AttributeError, ValueError):
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return [color_cycle[i % len(color_cycle)] for i in range(n_particles)]


def _line_segments(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segments = []
    plotted_indices = []
    endpoint_indices = []
    starts = []
    ends = []

    for i in range(x_data.shape[0]):
        mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
        if np.any(mask):
            x_traj = x_data[i, mask]
            y_traj = y_data[i, mask]
            points = np.column_stack((x_traj, y_traj))
            if len(points) >= 2:
                segments.append(points)
                plotted_indices.append(i)
            endpoint_indices.append(i)
            starts.append(points[0])
            ends.append(points[-1])

    return (
        segments,
        np.asarray(plotted_indices, dtype=int),
        np.asarray(endpoint_indices, dtype=int),
        np.asarray(starts, dtype=float) if starts else np.empty((0, 2), dtype=float),
        np.asarray(ends, dtype=float) if ends else np.empty((0, 2), dtype=float),
    )


def _distance_line_segments(
    x_data: np.ndarray,
    y_data: np.ndarray,
    time_data: np.ndarray,
    min_time: float,
) -> tuple[list[np.ndarray], np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    distance_segments = []
    distance_indices = []
    particle_distances = {}

    for i in range(x_data.shape[0]):
        mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
        if np.any(mask):
            x_traj = x_data[i, mask]
            y_traj = y_data[i, mask]
            time_traj = time_data[i, mask]
            time_hours = (time_traj - min_time) / 3600.0
            distances = np.sqrt((x_traj - x_traj[0]) ** 2 + (y_traj - y_traj[0]) ** 2)
            particle_distances[i] = (time_hours, distances)
            if len(time_hours) >= 2:
                distance_segments.append(np.column_stack((time_hours, distances)))
                distance_indices.append(i)

    return distance_segments, np.asarray(distance_indices, dtype=int), particle_distances


def _add_endpoint_markers(ax, starts, ends, *, colors, markers: str, marker_size: float):
    if markers == 'none':
        return
    if markers == 'start-end' and len(starts):
        ax.scatter(starts[:, 0], starts[:, 1], color=colors, marker='x', s=marker_size, linewidth=1, zorder=5)
    if markers in {'end', 'start-end'} and len(ends):
        ax.scatter(
            ends[:, 0],
            ends[:, 1],
            color=colors,
            marker='o',
            s=marker_size,
            edgecolor='black',
            linewidth=0.5,
            zorder=5,
        )


def _resolve_output_file(ds: xr.Dataset, output: str | Path | None) -> Path:
    """Resolve a trajectory-plot output path.

    ``None`` selects the NetCDF source directory. Any supplied path is used
    literally, so ``'.'`` selects the current working directory.
    """
    default_name = 'particle_trajectories.png'

    if output is not None:
        output_path = Path(output)
        if output_path.is_dir():
            return output_path / default_name
        return output_path

    source_file = ds.encoding.get('source', '.')
    source_dir = Path(source_file).parent
    return source_dir / default_name


def _normalize_panels(panels: str | list[str] | tuple[str, ...]) -> list[str]:
    valid_panels = ('spatial', 'distance', 'population', 'population-distance')

    if isinstance(panels, str):
        requested = [panel.strip().lower() for panel in panels.split(',') if panel.strip()]
    else:
        requested = [str(panel).strip().lower() for panel in panels if str(panel).strip()]

    if not requested:
        requested = ['spatial']
    if 'all' in requested:
        return list(valid_panels)

    invalid = sorted(set(requested) - set(valid_panels))
    if invalid:
        raise ValueError(
            f"'panels' contains invalid values: {', '.join(invalid)}. Valid values are: all, {', '.join(valid_panels)}."
        )

    ordered = [panel for panel in valid_panels if panel in requested]
    if 'spatial' not in ordered:
        ordered.insert(0, 'spatial')
    return ordered


def _create_panel_axes(panels: list[str]):
    if len(panels) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        return fig, {panels[0]: ax}

    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    axes_by_panel = dict(zip(panels, axes_flat, strict=True))
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    return fig, axes_by_panel


def plot_trajectories(
    ds,
    output=None,
    max_particles: int | None = None,
    sample_fraction: float | None = None,
    sample_seed: int = 0,
    markers: str = 'start-end',
    marker_size: float = 12.0,
    panels: str | list[str] | tuple[str, ...] = 'spatial',
    show: bool | None = None,
):
    """Plot particle trajectories from a SedTRAILS dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing particle trajectory variables, including ``x``,
        ``y``, and ``time``.
    output : str or pathlib.Path, optional
        Output target for the figure. If this is an existing directory, the
        plot is saved as ``particle_trajectories.png`` inside that directory.
        Otherwise, it is treated as an exact output filename. When omitted,
        the default output is ``particle_trajectories.png`` in the source
        dataset directory when available.
    max_particles : int, optional
        Maximum number of particles to plot. Sampling is deterministic and
        stratified by population when population IDs are available.
    sample_fraction : float, optional
        Fraction of particles to plot. Mutually exclusive with
        ``max_particles``.
    sample_seed : int, optional
        Seed used for deterministic particle sampling.
    markers : {'none', 'end', 'start-end'}, optional
        Which endpoint markers to draw.
    marker_size : float, optional
        Marker size for start/end points.
    panels : str or sequence of str, optional
        Panels to draw. ``spatial`` is the fast default. Use ``all`` for the
        previous four-panel figure, or a comma-separated subset of ``spatial``,
        ``distance``, ``population``, and ``population-distance``.
    show : bool or None, optional
        Whether to display the figure. By default, figures are not shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the selected panels.
    axes_by_panel : dict[str, matplotlib.axes.Axes]
        Axes keyed by panel name.

    Notes
    -----
    This function creates a four-panel matplotlib figure. It returns the figure
    and axes to support testing or further customization.
    """
    if markers not in {'none', 'end', 'start-end'}:
        raise ValueError("'markers' must be one of: 'none', 'end', or 'start-end'.")
    if marker_size <= 0:
        raise ValueError("'marker_size' must be greater than 0.")
    selected_panels = _normalize_panels(panels)

    output_path = _resolve_output_file(ds, output)
    if show is None:
        show = False

    sampled_ds, n_total_particles, n_sampled_particles = _sample_dataset(
        ds,
        max_particles=max_particles,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
    )

    x_data, y_data, time_data = _trajectory_arrays(sampled_ds)

    n_particles, n_timesteps = x_data.shape

    if n_sampled_particles != n_total_particles:
        print(
            f'\nPlotting trajectories for {n_sampled_particles} of {n_total_particles} '
            f'particles over {n_timesteps} timesteps...'
        )
    else:
        print(f'\nPlotting trajectories for {n_particles} particles over {n_timesteps} timesteps...')

    fig, axes_by_panel = _create_panel_axes(selected_panels)

    # Extract population information for color coding
    population_ids = (
        np.asarray(sampled_ds['population_id'].values, dtype=int)
        if 'population_id' in sampled_ds
        else np.zeros(n_particles, dtype=int)
    )
    n_populations = int(sampled_ds.sizes['n_populations']) if 'n_populations' in sampled_ds.sizes else 1

    # Plot 1: All trajectories on spatial map (individual particle colors)
    ax1 = axes_by_panel['spatial']
    ax1.set_title(f'(a) Particle Trajectories - Individual Colors (n={n_particles})')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')

    # Plot each particle trajectory
    colors = np.asarray(_particle_colors(n_particles))
    spatial_segments, spatial_indices, endpoint_indices, starts, ends = _line_segments(x_data, y_data)
    if spatial_segments:
        ax1.add_collection(LineCollection(spatial_segments, colors=colors[spatial_indices], alpha=0.7, linewidths=1))
        ax1.autoscale()
    endpoint_colors = colors[endpoint_indices] if len(starts) else []
    _add_endpoint_markers(ax1, starts, ends, colors=endpoint_colors, markers=markers, marker_size=marker_size)

    # Add legend if few particles
    if n_particles <= 10:
        for i, color in enumerate(colors):
            ax1.plot([], [], color=color, alpha=0.7, linewidth=1, label=f'Particle {i}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    if selected_panels == ['spatial']:
        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f'Plot saved to: {output_path}')

        if show:
            plt.show()
        elif output_path is not None:
            plt.close(fig)

        return fig, axes_by_panel

    # Plot 2: Time series of distances from initial position
    ax2 = axes_by_panel.get('distance')
    ax4 = axes_by_panel.get('population-distance')

    if ax2 is not None:
        ax2.set_title('(b) Distance from Initial Position vs Time')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Distance from Initial Position [m]')

    # Find the minimum time across all particles to use as reference
    min_time = np.nanmin(time_data) if np.any(np.isfinite(time_data)) else 0.0

    particle_distances = {}
    if ax2 is not None or ax4 is not None:
        distance_segments, distance_indices, particle_distances = _distance_line_segments(
            x_data, y_data, time_data, min_time
        )
        if ax2 is not None and distance_segments:
            ax2.add_collection(
                LineCollection(distance_segments, colors=colors[distance_indices], alpha=0.7, linewidths=1)
            )
            ax2.autoscale()

    if ax2 is not None:
        ax2.grid(True, alpha=0.3)

    # if 'population' in selected_panels or 'population-distance' in selected_panels:
    population_names = _decode_population_names(sampled_ds, n_populations)
    pop_colors = _population_colors(n_populations)

    # Plot 3: Trajectories colored by population
    ax3 = axes_by_panel.get('population')
    if ax3 is not None:
        ax3.set_title('(c) Particle Trajectories - Colored by Population')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')

    # Plot trajectories grouped by population
    if ax3 is not None:
        for pop_idx in range(n_populations):
            particles_in_pop = np.flatnonzero(population_ids == pop_idx)
            pop_segments = []
            pop_starts = []
            pop_ends = []
            for i in particles_in_pop:
                mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
                if np.any(mask):
                    points = np.column_stack((x_data[i, mask], y_data[i, mask]))
                    if len(points) >= 2:
                        pop_segments.append(points)
                    pop_starts.append(points[0])
                    pop_ends.append(points[-1])

            if pop_segments:
                ax3.add_collection(LineCollection(pop_segments, colors=[pop_colors[pop_idx]], alpha=0.7, linewidths=1))
                ax3.plot([], [], color=pop_colors[pop_idx], alpha=0.7, linewidth=1, label=population_names[pop_idx])
            if pop_starts:
                marker_colors = [pop_colors[pop_idx]] * len(pop_starts)
                _add_endpoint_markers(
                    ax3,
                    np.asarray(pop_starts, dtype=float),
                    np.asarray(pop_ends, dtype=float),
                    colors=marker_colors,
                    markers=markers,
                    marker_size=marker_size,
                )

        ax3.legend()
        ax3.autoscale()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')

    # Plot 4: Distance from initial position by population with statistics
    if ax4 is not None:
        ax4.set_title('(d) Distance from Initial Position by Population')
        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Distance from Initial Position [m]')

    # Create containers for population statistics
    population_stats = {}

    # First pass: collect all data for each population
    if ax4 is not None:
        for pop_idx in range(n_populations):
            population_stats[pop_idx] = {
                'times': [],
                'distances': [],
                'name': population_names[pop_idx],
                'color': pop_colors[pop_idx],
            }

            particles_in_pop = np.flatnonzero(population_ids == pop_idx)

            # Plot individual particle distances for this population
            pop_distance_segments = []
            for i in particles_in_pop:
                if i in particle_distances:
                    time_hours, distances = particle_distances[i]
                    if len(time_hours) >= 2:
                        pop_distance_segments.append(np.column_stack((time_hours, distances)))
                    population_stats[pop_idx]['times'].append(time_hours)
                    population_stats[pop_idx]['distances'].append(distances)
            if pop_distance_segments:
                ax4.add_collection(
                    LineCollection(pop_distance_segments, colors=[pop_colors[pop_idx]], alpha=0.3, linewidths=0.8)
                )

    # Second pass: compute and plot population statistics
    if ax4 is not None:
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
        ax4.autoscale()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {output_path}')

    if show:
        plt.show()
    elif output_path is not None:
        plt.close(fig)

    return fig, axes_by_panel
