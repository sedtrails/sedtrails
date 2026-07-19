"""Skipped-by-default integration test for quick particle diagnostics plots.

This is the resurrected quick-and-dirty plotter, turned into a test that can
run a SedTRAILS config end-to-end and optionally save diagnostic plots for:
- particle position maps
- burial_depth, z, and status_buried time series

Run controls:

- Enable the test with ``SEDTRAILS_RUN_PLOT_INTEGRATION=1``.
- Override the config file with ``SEDTRAILS_PLOT_CONFIG=/path/to/config.yaml``.
- The default config is [tests/integration/sedtrails-example-burial-diagnostic.yaml](c:/sedtrails/tests/integration/sedtrails-example-burial-diagnostic.yaml).

Plot controls:

- Plotting is enabled by default.
- Set ``SEDTRAILS_PLOT_PARTICLE_MAPS=0`` to disable the particle position map.
- Set ``SEDTRAILS_PLOT_TIME_SERIES=0`` to disable the burial_depth/z/status_buried time series.
- Set ``SEDTRAILS_PLOT_SHOW=1`` to optionally open the figures after the run.
- Plots are written to ``tests/integration/plots`` by default.
- Override the plot output directory with ``SEDTRAILS_PLOT_OUTPUT_DIR=/path/to/plots``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml
import xarray as xr

from sedtrails.simulation_orchestrator.simulation_manager import Simulation

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get('SEDTRAILS_RUN_PLOT_INTEGRATION'),
        reason='Skipped by default. Set SEDTRAILS_RUN_PLOT_INTEGRATION=1 to run the plot integration test.',
    ),
]

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'sedtrails-example-burial-diagnostic.yaml'
DEFAULT_PLOT_OUTPUT_DIR = Path(__file__).resolve().parent / 'plots'
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off', ''}


def _resolve_config_path() -> Path:
    config_override = os.environ.get('SEDTRAILS_PLOT_CONFIG')
    config_path = Path(config_override) if config_override else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    return config_path.resolve()


def _resolve_plot_output_dir() -> Path:
    output_override = os.environ.get('SEDTRAILS_PLOT_OUTPUT_DIR')
    if output_override:
        return Path(output_override).expanduser().resolve()
    return DEFAULT_PLOT_OUTPUT_DIR.resolve()


def _resolve_input_data_path(source_config: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path

    repo_relative_parts = list(path.parts)
    while repo_relative_parts and repo_relative_parts[0] == '..':
        repo_relative_parts.pop(0)

    candidates = [
        (source_config.parent / path).resolve(),
        (REPO_ROOT.joinpath(*repo_relative_parts)).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _prepare_config_copy(source_config: Path, output_dir: Path, tmp_path: Path) -> Path:
    """Copy a YAML config to a temp file while rewriting file paths for isolation."""
    with source_config.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    config['inputs']['data'] = str(_resolve_input_data_path(source_config, config['inputs']['data']))
    config.setdefault('outputs', {})
    config['outputs']['directory'] = str(output_dir)

    temp_config = tmp_path / source_config.name
    with temp_config.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return temp_config


def _open_results(results_file: Path) -> xr.Dataset:
    if not results_file.exists():
        raise FileNotFoundError(f'Results file not found: {results_file}')
    return xr.open_dataset(results_file, decode_times=False)


def _time_hours(time_values: np.ndarray) -> np.ndarray:
    time_values = np.asarray(time_values, dtype=float)
    if time_values.size == 0:
        return time_values
    return (time_values - float(time_values[0])) / 3600.0


def _decode_netcdf_name(raw_name) -> str:
    if isinstance(raw_name, bytes):
        return raw_name.decode('utf-8', errors='ignore').strip('\x00 ').strip()

    values = np.asarray(raw_name)
    if values.dtype.kind == 'S':
        return b''.join(np.ravel(values).tolist()).decode('utf-8', errors='ignore').strip('\x00 ').strip()
    if values.dtype.kind == 'U':
        return ''.join(np.ravel(values).tolist()).strip('\x00 ').strip()
    return str(raw_name).strip()


def _population_metadata(ds: xr.Dataset) -> tuple[np.ndarray, list[str]]:
    population_ids = np.asarray(ds['population_id'].values, dtype=int) if 'population_id' in ds else np.zeros(0, dtype=int)
    if population_ids.size == 0 and 'x' in ds and ds['x'].ndim == 2:
        population_ids = np.zeros(ds['x'].shape[1], dtype=int)

    n_populations = int(ds.sizes.get('n_populations', np.max(population_ids) + 1 if population_ids.size else 1))
    names = [f'population_{idx}' for idx in range(n_populations)]

    if 'population_name' in ds:
        population_var = ds['population_name']
        n_available = int(population_var.sizes.get('n_populations', population_var.shape[0]))
        for idx in range(min(n_populations, n_available)):
            raw_name = population_var[idx].values if population_var.ndim == 1 else population_var[idx, :].values
            decoded = _decode_netcdf_name(raw_name)
            if decoded:
                names[idx] = decoded

    return population_ids, names


def _population_colors(n_populations: int):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Set1' if n_populations <= 8 else 'tab20')
    if n_populations <= 1:
        return [cmap(0)]
    return [cmap(value) for value in np.linspace(0, 1, n_populations)]


def _plot_position_maps(ds: xr.Dataset, output_dir: Path) -> Path:
    """Plot particle position paths as x/y maps, colored by population."""
    import matplotlib

    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x = np.asarray(ds['x'].values, dtype=float)
    y = np.asarray(ds['y'].values, dtype=float)
    population_ids, population_names = _population_metadata(ds)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError('Expected time-major x/y arrays with shape (n_timesteps, n_particles).')

    n_timesteps, n_particles = x.shape
    if population_ids.shape[0] != n_particles:
        raise ValueError('population_id length does not match number of particles.')

    pop_colors = _population_colors(max(1, len(population_names)))
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    for particle_index in range(n_particles):
        pop_idx = int(population_ids[particle_index])
        color = pop_colors[pop_idx % len(pop_colors)]
        ax.plot(x[:, particle_index], y[:, particle_index], lw=1.0, alpha=0.8, color=color)
        ax.scatter(x[0, particle_index], y[0, particle_index], s=18, marker='o', color=color)
        ax.scatter(x[-1, particle_index], y[-1, particle_index], s=18, marker='x', color=color)

    legend_handles = [
        Line2D([0], [0], color=pop_colors[idx], lw=2.0, label=population_names[idx])
        for idx in range(len(population_names))
    ]

    ax.set_title(f'Particle position maps by population ({n_particles} particles, {n_timesteps} timesteps)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25)
    ax.legend(handles=legend_handles, title='Population', loc='best')

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / 'particle_position_maps.png'
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def _plot_time_series(ds: xr.Dataset, output_dir: Path) -> Path:
    """Plot burial_depth, z, and status_buried time series split by population."""
    import matplotlib

    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    time_values = _time_hours(np.asarray(ds['time'].values, dtype=float))
    burial_depth = np.asarray(ds['burial_depth'].values, dtype=float)
    z = np.asarray(ds['z'].values, dtype=float)
    status_buried = np.asarray(ds['status_buried'].values, dtype=float)
    population_ids, population_names = _population_metadata(ds)

    if burial_depth.ndim != 2 or z.ndim != 2 or status_buried.ndim != 2:
        raise ValueError('Expected time-major burial_depth, z, and status_buried arrays.')

    n_timesteps, n_particles = burial_depth.shape
    if z.shape != burial_depth.shape or status_buried.shape != burial_depth.shape:
        raise ValueError('Time-series arrays must all have the same shape.')
    if population_ids.shape[0] != n_particles:
        raise ValueError('population_id length does not match number of particles.')

    n_populations = max(1, len(population_names))
    pop_colors = _population_colors(n_populations)
    fig, axes = plt.subplots(3, n_populations, figsize=(6.2 * n_populations, 9.0), sharex=True, squeeze=False)
    series = [
        ('burial_depth', burial_depth),
        ('z', z),
        ('status_buried', status_buried),
    ]

    for pop_idx, pop_name in enumerate(population_names):
        particle_indices = np.flatnonzero(population_ids == pop_idx)
        for row_idx, (label, data) in enumerate(series):
            axis = axes[row_idx, pop_idx]
            for particle_index in particle_indices:
                color = pop_colors[pop_idx % len(pop_colors)]
                if label == 'status_buried':
                    axis.step(time_values, data[:, particle_index], where='post', lw=1.0, alpha=0.8, color=color)
                else:
                    axis.plot(time_values, data[:, particle_index], lw=1.0, alpha=0.8, color=color)
            if pop_idx == 0:
                axis.set_ylabel(label)
            axis.set_title(f'Population: {pop_name} (n={len(particle_indices)})')
            axis.grid(True, alpha=0.25)

    for axis in axes[-1, :]:
        axis.set_xlabel('time (hours since first saved slot)')
    fig.suptitle(f'Particle time series by population ({n_particles} particles, {n_timesteps} timesteps)')

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / 'particle_time_series.png'
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


@pytest.mark.integration
def test_particle_diagnostics_plotter(tmp_path):
    """Run the example config and optionally save quick diagnostic plots."""
    config_path = _resolve_config_path()
    output_dir = tmp_path / 'results'
    plot_output_dir = _resolve_plot_output_dir()
    temp_config = _prepare_config_copy(config_path, output_dir, tmp_path)

    Simulation(str(temp_config), enable_dashboard=False).run()

    results_file = output_dir / 'sedtrails_results.nc'
    with _open_results(results_file) as ds:
        for field_name in ('x', 'y', 'burial_depth', 'z', 'status_buried', 'time'):
            assert field_name in ds, f"Missing expected output field: {field_name}"

        if _env_flag('SEDTRAILS_PLOT_PARTICLE_MAPS', True):
            _plot_position_maps(ds, plot_output_dir)
        if _env_flag('SEDTRAILS_PLOT_TIME_SERIES', True):
            _plot_time_series(ds, plot_output_dir)

    if _env_flag('SEDTRAILS_PLOT_SHOW', False):
        import matplotlib.pyplot as plt

        plt.show()
