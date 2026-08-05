"""Utilities to build restart inputs from a SedTRAILS NetCDF output file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import os
import re
from typing import Any

import numpy as np
import xarray as xr
import yaml

from sedtrails.application_interfaces.validator import SedtrailsYamlLoader
from sedtrails.particle_tracer.timer import convert_duration_string_to_seconds
from sedtrails.transport_converter.format_converter import FormatConverter


@dataclass
class RestartSummary:
    """Summary of artifacts generated for a restart configuration."""

    output_config: Path
    seed_files: dict[str, Path]
    restart_time: str
    retained_particles: int


@dataclass
class RestartParticleState:
    """Particle state extracted from a SedTRAILS output file for restart seeding."""

    x: np.ndarray
    y: np.ndarray
    pop_ids: np.ndarray
    alive_mask: np.ndarray
    in_domain_mask: np.ndarray
    released_mask: np.ndarray
    restart_seconds: float | None
    particle_fields: dict[str, np.ndarray]


_RESTART_PARTICLE_FIELDS = (
    'z',
    'z_p',
    'burial_depth',
    'status_suspended',
    'status_deposited',
    'status_buried',
    'vertical_position_initialized',
)


def _optional_restart_fields(
    ds: xr.Dataset,
    n_particles: int,
    time_dim: str | None = None,
    slot_idx: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract optional Q3D state fields from one output sample."""
    fields = {}
    for name in _RESTART_PARTICLE_FIELDS:
        if name not in ds:
            continue
        variable = ds[name]
        if time_dim is not None and time_dim in variable.dims:
            variable = variable.isel({time_dim: slot_idx})
        values = np.asarray(variable.values)
        if values.ndim == 1 and values.shape[0] == n_particles:
            fields[name] = values
    if 'vertical_position_initialized' not in fields and 'z' in fields:
        fields['vertical_position_initialized'] = np.isfinite(fields['z'])
    return fields


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        raise ValueError('Missing time.start in base configuration.')
    text = str(value).strip()
    if text.endswith('Z'):
        text = text[:-1]
    return datetime.fromisoformat(text)


def _format_datetime(value: datetime) -> str:
    return value.strftime('%Y-%m-%d %H:%M:%S')


def _config_reference_date(config: dict[str, Any], default: str | None = None) -> datetime | None:
    reference_date_value = config.get('general', {}).get('input_model', {}).get('reference_date', default)
    if reference_date_value is None:
        return None
    return _to_datetime(reference_date_value)


def _parse_time_units_reference_date(units: Any) -> datetime | None:
    if units is None:
        return None

    match = re.match(r'^\s*seconds\s+since\s+(.+?)\s*$', str(units), flags=re.IGNORECASE)
    if not match:
        return None

    return _to_datetime(match.group(1))


def _dataset_reference_date(ds: xr.Dataset, config: dict[str, Any]) -> datetime | None:
    """Return the reference date used by output time values."""
    for attr_name in ('reference_date', 'time_reference_date'):
        if attr_name in ds.attrs:
            return _to_datetime(ds.attrs[attr_name])

    for attr_name in ('time_units', 'units'):
        reference_date = _parse_time_units_reference_date(ds.attrs.get(attr_name))
        if reference_date is not None:
            return reference_date

    if 'time' in ds:
        for attr_name in ('time_units', 'units'):
            reference_date = _parse_time_units_reference_date(ds['time'].attrs.get(attr_name))
            if reference_date is not None:
                return reference_date

    return _config_reference_date(config)


def _infer_restart_datetime(
    config: dict[str, Any],
    restart_seconds: float | None,
    dataset_reference_date: datetime | None = None,
) -> datetime:
    """Infer restart datetime from NetCDF time semantics and config."""
    base_start = _to_datetime(config.get('time', {}).get('start'))
    if restart_seconds is None:
        return base_start

    if dataset_reference_date is not None:
        return dataset_reference_date + timedelta(seconds=restart_seconds)

    reference_date = _config_reference_date(config)
    if reference_date is not None:
        return reference_date + timedelta(seconds=restart_seconds)

    # Heuristic fallback: very large values usually indicate absolute epoch seconds.
    if restart_seconds > 1.0e8:
        return datetime.utcfromtimestamp(restart_seconds)

    return base_start + timedelta(seconds=restart_seconds)


def _restart_format_config(config: dict[str, Any], config_path: Path) -> dict[str, Any] | None:
    input_model = config.get('general', {}).get('input_model', {})
    inputs = config.get('inputs', {})
    input_file = inputs.get('data')
    input_format = input_model.get('format')
    if not input_file or not input_format:
        return None

    input_path = Path(str(input_file))
    if not input_path.is_absolute():
        input_path = config_path.parent / input_path

    return {
        'input_file': str(input_path),
        'input_format': input_format,
        'reference_date': input_model.get('reference_date', '1970-01-01'),
        'morfac': input_model.get('morfac', 1.0),
        'sediment_fraction_index': input_model.get('sediment_fraction_index', 0),
        'sediment_fraction_name': input_model.get('sediment_fraction_name'),
    }


def _validate_restart_time_matches_input(
    config: dict[str, Any],
    config_path: Path,
    restart_dt: datetime,
) -> None:
    """Validate that the generated restart start is covered by the original forcing."""
    format_config = _restart_format_config(config, config_path)
    if format_config is None:
        return

    reference_date = _config_reference_date(config, default='1970-01-01')
    if reference_date is None:
        return

    forcing_start, forcing_end = FormatConverter(format_config).get_time_bounds()
    restart_seconds = (restart_dt - reference_date).total_seconds()
    if forcing_start <= restart_seconds <= forcing_end:
        return

    restart_time = _format_datetime(restart_dt)
    raise ValueError(
        f'Restart time {restart_time} ({restart_seconds:.3f}s since reference_date '
        f'{_format_datetime(reference_date)}) is outside the original input forcing window '
        f'[{forcing_start:.3f}, {forcing_end:.3f}]s. '
        'The generated restart would start before or after available forcing data.'
    )


def _status_values_to_mask(values: Any, default: bool = True) -> np.ndarray:
    """Convert status values to a boolean mask while treating NaN fill values as false."""
    arr = np.asarray(values)
    if arr.dtype.kind in ('f', 'c'):
        return np.isfinite(arr) & (arr != 0)
    return arr.astype(bool)


def _population_ids(ds: xr.Dataset, n_particles: int) -> np.ndarray:
    if 'population_id' not in ds:
        return np.zeros(n_particles, dtype=int)

    pop_ids = np.asarray(ds['population_id'].values).astype(int)
    if pop_ids.shape[0] != n_particles:
        raise ValueError("'population_id' length does not match number of particles.")
    return pop_ids


def _last_written_time_index(ds: xr.Dataset) -> int:
    """Return the last written v2 timestep without reading the full trajectory arrays."""
    n_timesteps = int(ds.sizes.get('n_timesteps', ds.sizes.get('time', 0)))
    if n_timesteps <= 0:
        raise ValueError('NetCDF output contains no timesteps.')

    written_slots = ds.attrs.get('written_slots')
    if written_slots is not None:
        try:
            written_slots = int(written_slots)
        except (TypeError, ValueError):
            written_slots = 0
        if written_slots > 0:
            return min(written_slots, n_timesteps) - 1

    if 'time' in ds and ds['time'].ndim == 1:
        time_values = np.asarray(ds['time'].values, dtype=float)
        finite = np.flatnonzero(np.isfinite(time_values))
        if finite.size:
            return int(finite[-1])

    return n_timesteps - 1


def _extract_checkpoint_state(ds: xr.Dataset) -> RestartParticleState:
    x_data = np.asarray(ds['x'].values, dtype=float)
    y_data = np.asarray(ds['y'].values, dtype=float)
    if x_data.ndim != 1 or y_data.ndim != 1:
        raise ValueError("Checkpoint output must contain 1D 'x' and 'y' particle arrays.")

    n_particles = x_data.shape[0]
    alive_mask = np.ones(n_particles, dtype=bool)
    if 'status_alive' in ds:
        alive_mask = _status_values_to_mask(ds['status_alive'].values)

    in_domain_mask = np.ones(n_particles, dtype=bool)
    if 'status_domain' in ds:
        in_domain_mask = _status_values_to_mask(ds['status_domain'].values)
    released_mask = np.ones(n_particles, dtype=bool)
    if 'status_released' in ds:
        released_mask = _status_values_to_mask(ds['status_released'].values)

    restart_seconds = None
    if 'time' in ds:
        time_values = np.asarray(ds['time'].values, dtype=float)
        finite_times = time_values[np.isfinite(time_values)]
        if finite_times.size:
            restart_seconds = float(np.max(finite_times))

    return RestartParticleState(
        x=x_data,
        y=y_data,
        pop_ids=_population_ids(ds, n_particles),
        alive_mask=np.asarray(alive_mask, dtype=bool),
        in_domain_mask=np.asarray(in_domain_mask, dtype=bool),
        released_mask=np.asarray(released_mask, dtype=bool),
        restart_seconds=restart_seconds,
        particle_fields=_optional_restart_fields(ds, n_particles),
    )


def _is_time_particle_layout(ds: xr.Dataset) -> bool:
    if 'time' not in ds or ds['time'].ndim != 1:
        return False
    if ds['x'].ndim != 2 or ds['y'].ndim != 2:
        return False
    return ds['x'].dims[0] == ds['time'].dims[0] and ds['y'].dims[0] == ds['time'].dims[0]


def _extract_time_particle_state(ds: xr.Dataset) -> RestartParticleState:
    slot_idx = _last_written_time_index(ds)
    time_dim = ds['time'].dims[0]

    x_data = np.asarray(ds['x'].isel({time_dim: slot_idx}).values, dtype=float)
    y_data = np.asarray(ds['y'].isel({time_dim: slot_idx}).values, dtype=float)
    if x_data.ndim != 1 or y_data.ndim != 1:
        raise ValueError("Expected time-major 'x' and 'y' arrays with particle slices.")

    n_particles = x_data.shape[0]
    alive_mask = np.ones(n_particles, dtype=bool)
    if 'status_alive' in ds:
        alive_mask = _status_values_to_mask(ds['status_alive'].isel({time_dim: slot_idx}).values)

    in_domain_mask = np.ones(n_particles, dtype=bool)
    if 'status_domain' in ds:
        in_domain_mask = _status_values_to_mask(ds['status_domain'].isel({time_dim: slot_idx}).values)
    released_mask = np.ones(n_particles, dtype=bool)
    if 'status_released' in ds:
        released_mask = _status_values_to_mask(ds['status_released'].isel({time_dim: slot_idx}).values)

    restart_seconds = None
    time_values = np.asarray(ds['time'].values, dtype=float)
    if 0 <= slot_idx < time_values.size and np.isfinite(time_values[slot_idx]):
        restart_seconds = float(time_values[slot_idx])

    return RestartParticleState(
        x=x_data,
        y=y_data,
        pop_ids=_population_ids(ds, n_particles),
        alive_mask=np.asarray(alive_mask, dtype=bool),
        in_domain_mask=np.asarray(in_domain_mask, dtype=bool),
        released_mask=np.asarray(released_mask, dtype=bool),
        restart_seconds=restart_seconds,
        particle_fields=_optional_restart_fields(ds, n_particles, time_dim, slot_idx),
    )


def _extract_restart_state(ds: xr.Dataset) -> RestartParticleState:
    """Extract restart particle state from v2 trajectory output or a checkpoint."""
    if ds.attrs.get('sedtrails_file_kind') == 'checkpoint' or (ds['x'].ndim == 1 and ds['y'].ndim == 1):
        return _extract_checkpoint_state(ds)
    if _is_time_particle_layout(ds):
        return _extract_time_particle_state(ds)

    raise ValueError(
        'Restart generation requires SedTRAILS time-major trajectory output '
        "('x'/'y' shaped as n_timesteps x n_particles with 1D 'time') "
        'or sedtrails_checkpoint.nc. Legacy particle-major trajectory files are not supported.'
    )


def _path_relative_to_cwd(path: Path) -> str:
    """Return path as POSIX text relative to current working directory when possible."""
    try:
        relative = Path(os.path.relpath(path.resolve(), Path.cwd().resolve()))
    except ValueError:
        # Different drives on Windows cannot be relativized.
        return str(path.resolve()).replace('\\', '/')

    text = str(relative).replace('\\', '/')
    if not relative.is_absolute() and not text.startswith(('.', '..')):
        text = f'./{text}'
    return text


def _seconds_to_duration_string(total_seconds: int) -> str:
    """Convert seconds to a compact SedTRAILS duration string (e.g., 1D2H30M)."""
    if total_seconds <= 0:
        return '0S'

    remaining = int(total_seconds)
    days, remaining = divmod(remaining, 86400)
    hours, remaining = divmod(remaining, 3600)
    minutes, seconds = divmod(remaining, 60)

    parts = []
    if days:
        parts.append(f'{days}D')
    if hours:
        parts.append(f'{hours}H')
    if minutes:
        parts.append(f'{minutes}M')
    if seconds:
        parts.append(f'{seconds}S')

    return ''.join(parts) if parts else '0S'


def _open_restart_dataset(netcdf_path: Path) -> xr.Dataset:
    """Open a SedTRAILS result NetCDF without xarray backend plugin discovery."""
    return xr.open_dataset(netcdf_path, engine='netcdf4', decode_times=False)


def _validate_restart_dataset_schema(ds: xr.Dataset, netcdf_path: Path) -> None:
    """Validate that the input file has the particle trajectory fields needed for restart."""
    missing = [name for name in ('x', 'y') if name not in ds]
    if not missing:
        return

    available = ', '.join(map(str, list(ds.data_vars)[:10])) or 'none'
    if len(ds.data_vars) > 10:
        available = f'{available}, ...'
    raise ValueError(
        f"'{netcdf_path}' is not a SedTRAILS trajectory output file. "
        "Restart generation expects the NetCDF file written by `sedtrails run` "
        "(usually `results/sedtrails_results.nc`) passed with `--file`. "
        f"Missing required particle variable(s): {', '.join(missing)}. "
        f'Available data variables: {available}. '
        'Eulerian forcing files belong in `inputs.data` of the base config, not in `--file`.'
    )


def create_restart_from_netcdf(
    netcdf_file: str,
    base_config_file: str,
    output_config_file: str = 'sedtrails-restart.yaml',
    seed_points_dir: str | None = None,
) -> RestartSummary:
    """
    Create a restart YAML and per-population seed files from NetCDF output.

    The generated configuration keeps existing settings from ``base_config_file`` but
    replaces each population seeding strategy with ``file_points`` using particle
    coordinates from the last valid timestep for each particle in ``netcdf_file``.

    Parameters
    ----------
    netcdf_file : str
        Path to the NetCDF results file.
    base_config_file : str
        Path to the base configuration file.
    output_config_file : str
        Path where the restart configuration file is written.
    seed_points_dir : str | None
        Directory containing restart seed-point files.

    Returns
    -------
    RestartSummary
        Computed value returned by the function.
    """

    netcdf_path = Path(netcdf_file)
    config_path = Path(base_config_file)
    output_path = Path(output_config_file)

    if not netcdf_path.is_file():
        raise FileNotFoundError(f'NetCDF file not found: {netcdf_path}')
    if not config_path.is_file():
        raise FileNotFoundError(f'Base configuration file not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.load(handle, Loader=SedtrailsYamlLoader)

    if not isinstance(config, dict):
        raise ValueError('Base configuration must parse to a dictionary.')

    populations = config.get('particles', {}).get('populations', [])
    if not populations:
        raise ValueError('Base configuration has no particles.populations entries.')

    ds = _open_restart_dataset(netcdf_path)
    try:
        _validate_restart_dataset_schema(ds, netcdf_path)

        restart_state = _extract_restart_state(ds)
        n_particles = restart_state.x.shape[0]
        keep_mask = (
            np.isfinite(restart_state.x)
            & np.isfinite(restart_state.y)
            & restart_state.alive_mask
            & restart_state.in_domain_mask
            & restart_state.released_mask
        )
        if not np.any(keep_mask):
            raise ValueError('No valid particle positions available to seed restart.')

        restart_dt = _infer_restart_datetime(config, restart_state.restart_seconds, _dataset_reference_date(ds, config))
        _validate_restart_time_matches_input(config, config_path, restart_dt)
        restart_time = _format_datetime(restart_dt)

        original_duration = config.get('time', {}).get('duration')
        if original_duration is None:
            raise ValueError('Missing time.duration in base configuration.')
        original_duration_seconds = convert_duration_string_to_seconds(str(original_duration))

        original_start = _to_datetime(config.get('time', {}).get('start'))
        elapsed_seconds = max(0, int((restart_dt - original_start).total_seconds()))
        remaining_seconds = max(0, original_duration_seconds - elapsed_seconds)
        if remaining_seconds <= 0:
            raise ValueError('No remaining duration left in original run. Restart is not needed.')
        remaining_duration = _seconds_to_duration_string(remaining_seconds)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if seed_points_dir is None:
            seeds_dir = output_path.parent / f'{output_path.stem}_seeds'
        else:
            seeds_dir = Path(seed_points_dir)
        seeds_dir.mkdir(parents=True, exist_ok=True)

        seed_files: dict[str, Path] = {}

        for pop_idx, population in enumerate(populations):
            pop_name = str(population.get('name', f'population_{pop_idx + 1}'))

            selected_indices = [
                i
                for i in range(n_particles)
                if keep_mask[i] and int(restart_state.pop_ids[i]) == pop_idx
            ]

            population.setdefault('seeding', {})
            population['seeding']['release_start'] = restart_time

            if not selected_indices:
                # Disable seeding for this population on restart (avoid reseeding new particles).
                population['seeding']['quantity'] = 0
                continue

            points_file = seeds_dir / f'{pop_name}.restart_points.csv'
            state_fields = [name for name in _RESTART_PARTICLE_FIELDS if name in restart_state.particle_fields]
            with open(points_file, 'w', encoding='utf-8') as handle:
                handle.write(','.join(('x', 'y', *state_fields)) + '\n')
                for particle_index in selected_indices:
                    values = [
                        f'{float(restart_state.x[particle_index]):.8f}',
                        f'{float(restart_state.y[particle_index]):.8f}',
                    ]
                    values.extend(str(restart_state.particle_fields[name][particle_index]) for name in state_fields)
                    handle.write(','.join(values) + '\n')

            population['seeding']['quantity'] = 1
            population['seeding']['strategy'] = {
                'file_points': {
                    'path': _path_relative_to_cwd(points_file),
                    'x_col': 'x',
                    'y_col': 'y',
                    'has_header': True,
                    'deduplicate': False,
                }
            }

            seed_files[pop_name] = points_file

        if not seed_files:
            raise ValueError('No populations had active particles to include in restart files.')

        config.setdefault('time', {})
        config['time']['start'] = restart_time
        config['time']['duration'] = remaining_duration

        with open(output_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        retained_particles = int(np.count_nonzero(keep_mask))
        return RestartSummary(
            output_config=output_path,
            seed_files=seed_files,
            restart_time=restart_time,
            retained_particles=retained_particles,
        )
    finally:
        ds.close()
