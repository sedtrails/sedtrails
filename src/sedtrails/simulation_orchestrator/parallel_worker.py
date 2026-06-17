"""
Parallel worker utilities for multiprocessing-based HPC runs.

All functions here live at module level so they are inherited by forked
worker processes without pickling (Linux fork context copies the parent's
full address space, including this module's globals).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

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
    regardless of memory pressure — the process stalls silently in swap until
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
            'Estimated peak memory ~%.0f MB (%d workers × ~%.0f MB physics, DFM %.0f MB) '
            'may exceed available ~%.0f MB — workers risk being killed by the OS without a '
            'Python error. Reduce inputs.read_interval (e.g. read_interval: 2D → '
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
    # Local import avoids a circular reference: simulation_manager imports this module.
    from sedtrails.simulation_orchestrator.simulation_manager import Simulation

    os.environ['SEDTRAILS_TASK_ID'] = str(task_id)
    os.environ['SEDTRAILS_N_TASKS'] = str(n_tasks)

    try:
        import numba
        numba.set_num_threads(1)
        # numba.config.CACHE_DIR (not NUMBA_CACHE_DIR) — redirects JIT cache to a
        # per-worker /tmp path so concurrent workers don't race on shared NFS .nbi/.nbc files.
        numba.config.CACHE_DIR = f'/tmp/sedtrails_numba_{os.getpid()}'
    except Exception:
        pass

    try:
        sim = Simulation(config_file)
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
    """Concatenate per-worker NetCDF files along n_particles and remove task subdirectories."""
    import shutil

    task_files = sorted(base_output_dir.glob('task_*/sedtrails_results.nc'))
    if not task_files:
        raise FileNotFoundError(f'No task output files found in {base_output_dir}')

    datasets = [xr.open_dataset(f) for f in task_files]

    # Assign non-overlapping n_particles coordinates before concatenation
    offset = 0
    reindexed = []
    for ds in datasets:
        n = ds.sizes['n_particles']
        reindexed.append(ds.assign_coords(n_particles=np.arange(offset, offset + n)))
        offset += n

    merged = xr.concat(reindexed, dim='n_particles', data_vars='minimal', coords='minimal')
    merged = merged.load()

    # Per-worker outputs each contain only that worker's particle slice, so
    # population_count and population_start_idx are wrong after concat.
    # Recalculate from the merged population_id array.
    if 'population_id' in merged and 'population_count' in merged:
        pop_ids = merged['population_id'].values
        for pop_idx in range(merged.sizes['n_populations']):
            mask = pop_ids == pop_idx
            count = int(np.sum(mask))
            start = int(np.argmax(mask)) if count > 0 else 0
            merged['population_count'][pop_idx] = count
            merged['population_start_idx'][pop_idx] = start

    out_path = base_output_dir / 'sedtrails_results.nc'
    merged.to_netcdf(out_path)

    for ds in datasets:
        ds.close()
    for f in task_files:
        shutil.rmtree(f.parent)

    n_total = merged.sizes.get('n_particles', '?')
    logger.info('Merged %d worker outputs -> %s (%s particles total)',
                len(task_files), out_path, n_total)
