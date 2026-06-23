"""
Parallel worker utilities for multiprocessing-based HPC runs.

All functions here live at module level so they are inherited by forked
worker processes without pickling (Linux fork context copies the parent's
full address space, including this module's globals).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import netCDF4 as nc4
import numpy as np

# Set by Simulation._run_parallel before forking; workers inherit it read-only via CoW.
# Reset to None after all workers finish to release the reference.
_PRELOADED_INPUT_DATA = None


def _available_ram_mb() -> float | None:
    """Return MemAvailable from /proc/meminfo in MB, or None if unavailable."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def _set_worker_memory_limit(n_tasks: int) -> None:
    """Cap this worker's virtual address space to its fair share of available RAM.

    On Linux with default overcommit settings, malloc() never returns NULL
    regardless of memory pressure - the process stalls silently in swap until
    the OOM killer sends SIGKILL, which Python cannot catch. Setting RLIMIT_AS
    forces mmap()-based allocations (used by numpy for large arrays) to fail
    immediately with a real MemoryError instead, which _worker_fn catches and
    reports with a concrete fix suggestion.

    The limit is: current virtual size (inherited CoW DFM pages + Python
    runtime) plus this worker's equal share of the remaining available RAM.
    """
    avail = _available_ram_mb()
    if avail is None:
        return
    try:
        import resource
        virt_mb = 0.0
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmSize:'):
                    virt_mb = int(line.split()[1]) / 1024
                    break
        limit_bytes = int((virt_mb + avail / n_tasks) * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, resource.RLIM_INFINITY))
    except Exception:
        pass


def _warn_if_oom_risk(dfm_mb: float, n_workers: int, read_interval_s: float,
                      duration_s: float, logger) -> None:
    """Warn before forking if estimated peak memory likely exceeds available RAM.

    Each worker allocates physics arrays for one time chunk (~2.5x the chunk
    size). The DFM itself is shared via CoW and counted only once.
    """
    avail = _available_ram_mb()
    if avail is None:
        return
    chunk_frac = min(1.0, read_interval_s / duration_s) if duration_s > 0 else 1.0
    per_worker_mb = dfm_mb * chunk_frac * 2.5
    estimated_mb = dfm_mb + n_workers * per_worker_mb
    if estimated_mb > avail * 0.85:
        two_day_s = 2 * 86400
        two_day_frac = min(1.0, two_day_s / duration_s) if duration_s > 0 else chunk_frac
        two_day_per_worker_mb = dfm_mb * two_day_frac * 2.5
        logger.warning(
            'Estimated peak memory ~%.0f MB (%d workers x ~%.0f MB physics, DFM %.0f MB) '
            'may exceed available ~%.0f MB - workers risk being killed by the OS without a '
            'Python error. Reduce inputs.read_interval (e.g. read_interval: 2D -> '
            '~%.0f MB/worker, estimated total ~%.0f MB).',
            estimated_mb, n_workers, per_worker_mb, dfm_mb, avail,
            two_day_per_worker_mb, dfm_mb + n_workers * two_day_per_worker_mb,
        )


def _worker_fn(config_file: str, task_id: int, n_tasks: int) -> None:
    """Worker entry point called by multiprocessing.Pool (Linux fork context).

    Each worker runs a full Simulation on its own particle slice. The DFM
    dataset preloaded in the parent is injected directly to avoid reloading
    it from disk (the CoW pages are already in memory).
    """
    os.environ['SEDTRAILS_TASK_ID'] = str(task_id)
    os.environ['SEDTRAILS_N_TASKS'] = str(n_tasks)

    try:
        cache_root = Path(os.environ.get('SEDTRAILS_NUMBA_CACHE_ROOT', tempfile.gettempdir()))
        cache_dir = cache_root / f'sedtrails_numba_{os.getpid()}'
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = str(cache_dir)
        import numba
        numba.set_num_threads(1)
        # numba.config.CACHE_DIR (not NUMBA_CACHE_DIR) redirects JIT cache to a
        # per-worker path so concurrent workers do not race on shared cache files.
        numba.config.CACHE_DIR = str(cache_dir)
    except Exception:
        pass

    # Local import avoids a circular reference: simulation_manager imports this module.
    # Keep it after cache setup because importing Simulation imports numba-backed helpers.
    from sedtrails.simulation_orchestrator.simulation_manager import Simulation

    try:
        sim = Simulation(config_file, enable_dashboard=False)
        if _PRELOADED_INPUT_DATA is not None:
            sim.format_converter.format_plugin.input_data = _PRELOADED_INPUT_DATA
        _set_worker_memory_limit(n_tasks)
        sim.run()
    except MemoryError as e:
        avail = _available_ram_mb()
        avail_str = (f'{avail:.0f} MB available across {n_tasks} workers'
                     if avail else f'{n_tasks} workers')
        raise MemoryError(
            f'Worker {task_id} ran out of memory allocating physics arrays ({e}).\n'
            f'  Node: {avail_str}.\n'
            f'  Fix: reduce inputs.read_interval in your config (e.g. read_interval: 2D) '
            f'to limit how many DFM time steps each worker holds in memory at once.'
        ) from None


def _merge_outputs(base_output_dir: Path, logger) -> None:
    """Merge per-worker NetCDF files and remove task subdirectories."""
    task_files = sorted(base_output_dir.glob('task_*/sedtrails_results.nc'))
    if not task_files:
        raise FileNotFoundError(f'No task output files found in {base_output_dir}')

    result_path, n_total = _merge_netcdf_particle_shards(
        task_files,
        base_output_dir / 'sedtrails_results.nc',
    )

    checkpoint_files = sorted(base_output_dir.glob('task_*/sedtrails_checkpoint.nc'))
    if checkpoint_files:
        _merge_netcdf_particle_shards(
            checkpoint_files,
            base_output_dir / 'sedtrails_checkpoint.nc',
        )

    for task_dir in sorted({f.parent for f in task_files}):
        shutil.rmtree(task_dir)

    logger.info('Merged %d worker outputs -> %s (%s particles total)',
                len(task_files), result_path, n_total)


def _merge_netcdf_particle_shards(task_files: list[Path], out_path: Path) -> tuple[Path, int]:
    """Stream-copy worker NetCDF particle shards into one output file.

    Worker files contain complete simulation metadata but only a slice of each
    population. The merged file restores population-contiguous particle ranges
    and keeps the improved NetCDF schema, including numeric trajectory IDs.
    """
    datasets = [nc4.Dataset(str(path), 'r') for path in task_files]
    tmp_path = out_path.with_name(f'.{out_path.name}.tmp')
    try:
        if tmp_path.exists():
            tmp_path.unlink()

        first = datasets[0]
        _validate_compatible_shards(datasets)
        n_populations = len(first.dimensions['n_populations'])
        task_counts = [_population_counts(ds, n_populations) for ds in datasets]
        final_counts = np.sum(np.vstack(task_counts), axis=0).astype(np.int64)
        final_starts = _population_starts(final_counts)
        n_total = int(np.sum(final_counts))
        repr_volume = _merged_repr_volume(datasets, task_counts, final_counts)

        with nc4.Dataset(str(tmp_path), 'w', format='NETCDF4') as dst:
            _copy_global_attrs(first, dst)
            _create_merged_dimensions(first, dst, n_total)
            _create_merged_variables(first, dst)
            _copy_nonparticle_variables(first, dst)
            _write_merged_static_metadata(
                first,
                dst,
                final_counts,
                final_starts,
                repr_volume,
                n_total,
            )
            _copy_particle_variables_by_population(datasets, dst, task_counts, final_starts)
            dst.sync()

        tmp_path.replace(out_path)
    finally:
        for ds in datasets:
            ds.close()
        if tmp_path.exists():
            tmp_path.unlink()

    return out_path, n_total


def _validate_compatible_shards(datasets: list[nc4.Dataset]) -> None:
    """Raise a clear error when worker files cannot be merged safely."""
    first = datasets[0]
    expected_layout = getattr(first, 'trajectory_layout', None)
    expected_schema = getattr(first, 'sedtrails_output_schema', None)

    for ds in datasets[1:]:
        if getattr(ds, 'trajectory_layout', None) != expected_layout:
            raise ValueError('Cannot merge worker outputs with different trajectory_layout values.')
        if getattr(ds, 'sedtrails_output_schema', None) != expected_schema:
            raise ValueError('Cannot merge worker outputs with different sedtrails_output_schema values.')

    for dim_name in ('n_timesteps', 'n_populations', 'n_flowfields', 'name_strlen'):
        if dim_name not in first.dimensions:
            continue
        expected = len(first.dimensions[dim_name])
        for ds in datasets[1:]:
            if dim_name not in ds.dimensions or len(ds.dimensions[dim_name]) != expected:
                raise ValueError(f'Cannot merge worker outputs with different {dim_name} dimensions.')

    if 'time' in first.variables:
        expected_time = np.asarray(first['time'][...])
        for ds in datasets[1:]:
            if 'time' not in ds.variables:
                raise ValueError("Cannot merge worker outputs when a shard is missing 'time'.")
            if not np.allclose(np.asarray(ds['time'][...]), expected_time, equal_nan=True):
                raise ValueError('Cannot merge worker outputs with different time values.')

    for var_name in ('population_name', 'population_particle_type', 'flowfield_name'):
        if var_name not in first.variables:
            continue
        expected = np.asarray(first[var_name][...])
        for ds in datasets[1:]:
            if var_name not in ds.variables:
                raise ValueError(f"Cannot merge worker outputs when a shard is missing '{var_name}'.")
            if not np.array_equal(np.asarray(ds[var_name][...]), expected):
                raise ValueError(f'Cannot merge worker outputs with different {var_name} values.')


def _population_counts(ds: nc4.Dataset, n_populations: int) -> np.ndarray:
    """Return per-population particle counts for one worker file."""
    if 'population_count' in ds.variables:
        return np.asarray(ds['population_count'][:], dtype=np.int64)
    if 'population_id' in ds.variables:
        pop_ids = np.asarray(ds['population_id'][:], dtype=np.int64)
        return np.bincount(pop_ids, minlength=n_populations).astype(np.int64)
    raise KeyError("Worker output is missing 'population_count' and 'population_id'.")


def _population_starts(counts: np.ndarray) -> np.ndarray:
    """Return contiguous population start indices from counts."""
    starts = np.zeros_like(counts, dtype=np.int64)
    if counts.size > 1:
        starts[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return starts


def _merged_repr_volume(
    datasets: list[nc4.Dataset],
    task_counts: list[np.ndarray],
    final_counts: np.ndarray,
) -> np.ndarray:
    """Estimate final representative volumes from worker-local metadata."""
    if 'population_repr_volume' not in datasets[0].variables:
        return np.full(final_counts.shape, np.nan, dtype=float)

    repr_volume = np.full(final_counts.shape, np.nan, dtype=float)
    for pop_idx, total_count in enumerate(final_counts):
        if total_count <= 0:
            continue
        for ds, counts in zip(datasets, task_counts, strict=True):
            local_count = int(counts[pop_idx])
            if local_count <= 0:
                continue
            local_repr = float(ds['population_repr_volume'][pop_idx])
            if np.isfinite(local_repr):
                repr_volume[pop_idx] = local_repr * local_count / int(total_count)
            break
    return repr_volume


def _copy_global_attrs(src: nc4.Dataset, dst: nc4.Dataset) -> None:
    """Copy global NetCDF attributes."""
    for name in src.ncattrs():
        dst.setncattr(name, src.getncattr(name))


def _create_merged_dimensions(src: nc4.Dataset, dst: nc4.Dataset, n_particles: int) -> None:
    """Create dimensions, replacing only the particle dimension size."""
    for name, dim in src.dimensions.items():
        size = n_particles if name == 'n_particles' else (None if dim.isunlimited() else len(dim))
        dst.createDimension(name, size)


def _create_merged_variables(src: nc4.Dataset, dst: nc4.Dataset) -> None:
    """Create destination variables matching the source schema."""
    for name, var in src.variables.items():
        fill_value = getattr(var, '_FillValue', None)
        kwargs = _variable_filter_kwargs(var)
        chunksizes = _variable_chunksizes(var, dst)
        if chunksizes is not None:
            kwargs['chunksizes'] = chunksizes
        if fill_value is not None:
            kwargs['fill_value'] = fill_value

        out_var = dst.createVariable(name, var.datatype, var.dimensions, **kwargs)
        for attr_name in var.ncattrs():
            if attr_name != '_FillValue':
                out_var.setncattr(attr_name, var.getncattr(attr_name))


def _variable_filter_kwargs(var) -> dict:
    """Return compression/filter kwargs compatible with createVariable."""
    filters = var.filters()
    kwargs = {}
    if filters.get('zlib'):
        kwargs['zlib'] = True
        kwargs['complevel'] = filters.get('complevel', 4)
        kwargs['shuffle'] = filters.get('shuffle', True)
    if filters.get('fletcher32'):
        kwargs['fletcher32'] = True
    return kwargs


def _variable_chunksizes(var, dst: nc4.Dataset) -> tuple[int, ...] | None:
    """Return chunk sizes clamped to destination dimensions."""
    if not var.dimensions:
        return None
    chunking = var.chunking()
    if chunking in (None, 'contiguous'):
        return None

    chunks = []
    for dim_name, chunk_size in zip(var.dimensions, chunking, strict=True):
        dim = dst.dimensions[dim_name]
        dim_size = len(dim)
        chunks.append(max(1, min(int(chunk_size), dim_size)))
    return tuple(chunks)


def _copy_nonparticle_variables(src: nc4.Dataset, dst: nc4.Dataset) -> None:
    """Copy variables that do not vary along particles."""
    for name, var in src.variables.items():
        if 'n_particles' in var.dimensions or name in _STATIC_MERGE_VARIABLES:
            continue
        dst[name][...] = var[...]


_STATIC_MERGE_VARIABLES = {
    'population_start_idx',
    'population_count',
    'population_repr_volume',
    'population_id',
    'trajectory_id',
}


def _write_merged_static_metadata(
    src: nc4.Dataset,
    dst: nc4.Dataset,
    final_counts: np.ndarray,
    final_starts: np.ndarray,
    repr_volume: np.ndarray,
    n_particles: int,
) -> None:
    """Write static metadata whose values change after shard concatenation."""
    for name in ('population_name', 'population_particle_type', 'flowfield_name'):
        if name in src.variables and name in dst.variables:
            dst[name][...] = src[name][...]

    if 'population_start_idx' in dst.variables:
        dst['population_start_idx'][:] = final_starts
    if 'population_count' in dst.variables:
        dst['population_count'][:] = final_counts
    if 'population_repr_volume' in dst.variables:
        dst['population_repr_volume'][:] = repr_volume
    if 'trajectory_id' in dst.variables:
        if dst['trajectory_id'].dimensions == ('n_particles',):
            dst['trajectory_id'][:] = np.arange(n_particles, dtype=np.int64)
        else:
            _write_legacy_trajectory_ids(dst['trajectory_id'], n_particles)
    if 'population_id' in dst.variables:
        for pop_idx, (start, count) in enumerate(zip(final_starts, final_counts, strict=True)):
            if count > 0:
                dst['population_id'][start:start + count] = pop_idx


def _write_legacy_trajectory_ids(var, n_particles: int) -> None:
    """Write legacy fixed-width character trajectory IDs when needed."""
    name_strlen = var.shape[1]
    ids = np.array(
        [list(f'traj_{i}'.ljust(name_strlen)[:name_strlen]) for i in range(n_particles)],
        dtype='S1',
    )
    var[:, :] = ids


def _copy_particle_variables_by_population(
    datasets: list[nc4.Dataset],
    dst: nc4.Dataset,
    task_counts: list[np.ndarray],
    final_starts: np.ndarray,
) -> None:
    """Copy particle-varying variables into population-contiguous output ranges."""
    next_offsets = final_starts.astype(np.int64).copy()
    for ds, counts in zip(datasets, task_counts, strict=True):
        local_starts = _local_population_starts(ds, counts)
        for pop_idx, count in enumerate(counts):
            count = int(count)
            if count <= 0:
                continue
            src_start = int(local_starts[pop_idx])
            dst_start = int(next_offsets[pop_idx])
            _copy_particle_range(ds, dst, src_start, dst_start, count)
            next_offsets[pop_idx] += count


def _local_population_starts(ds: nc4.Dataset, counts: np.ndarray) -> np.ndarray:
    """Return source population starts for one worker output."""
    if 'population_start_idx' in ds.variables:
        return np.asarray(ds['population_start_idx'][:], dtype=np.int64)
    return _population_starts(counts)


def _copy_particle_range(
    src: nc4.Dataset,
    dst: nc4.Dataset,
    src_start: int,
    dst_start: int,
    count: int,
    *,
    chunk_size: int = 65_536,
) -> None:
    """Copy one contiguous source particle range into the destination file."""
    for name, src_var in src.variables.items():
        if 'n_particles' not in src_var.dimensions or name in _STATIC_MERGE_VARIABLES:
            continue
        dst_var = dst[name]
        particle_axis = src_var.dimensions.index('n_particles')
        copied = 0
        while copied < count:
            n = min(chunk_size, count - copied)
            src_slice = _particle_slice(src_var.ndim, particle_axis, src_start + copied, n)
            dst_slice = _particle_slice(dst_var.ndim, particle_axis, dst_start + copied, n)
            dst_var[dst_slice] = src_var[src_slice]
            copied += n


def _particle_slice(ndim: int, particle_axis: int, start: int, count: int) -> tuple:
    """Build a slicing tuple for one particle slab."""
    selection = [slice(None)] * ndim
    selection[particle_axis] = slice(start, start + count)
    return tuple(selection)
