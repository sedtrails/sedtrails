import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from sedtrails.application_interfaces.configuration_controller import ConfigurationController
from sedtrails.data_manager import DataManager
from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.particle_tracer import ParticleSeeder
from sedtrails.particle_tracer.data_retriever import FieldDataRetriever  # Updated import
from sedtrails.particle_tracer.particle import Particle
from sedtrails.particle_tracer.timer import Duration, Time, Timer
from sedtrails.pathway_visualizer import SimulationDashboard
from sedtrails.simulation_orchestrator.global_logger import log_simulation_state, setup_logging
from sedtrails.simulation_orchestrator.runtime_plan import (
    build_plan_sedtrails_data,
    build_population_runtime_plans,
    unique_flow_field_names,
    validate_population_runtime_configurations,
)
from sedtrails.transport_converter.format_converter import FormatConverter, SedtrailsData
from sedtrails.transport_converter.physics_converter import PhysicsConverter
from sedtrails.simulation_orchestrator import parallel_worker as _pw


class _BedLevelOverride:
    """[BETA] Loads a separate bed-level file and provides temporally interpolated
    bed levels using real simulation time, bypassing the Eulerian repeat wrapping
    applied to the main hydrodynamic input.

    Expected file structure - must be structurally identical to the main DFM
    input file, containing at minimum:
      dimension  nNodes               - spatial grid nodes (same grid as main file)
      variable   time     (nTime,)    - seconds since <reference_date>
      variable   bedlevel (nTime, nNodes) - bed level [m NAP]

    Additional variables in the file (e.g. bedlevel_change_rate) are ignored.
    Time intervals can differ freely from the main file; linear interpolation
    is applied between monthly (or any) steps.
    """

    def __init__(self, nc_path: str, logger=None):
        import netCDF4 as _nc4

        if logger:
            logger.warning(
                '[BETA] inputs.bed_level_data is set: %s  '
                'Bed levels will be read from this file at real simulation time '
                '(no Eulerian repeat wrapping). This is a beta feature.',
                nc_path,
            )

        with _nc4.Dataset(nc_path) as ds:
            t_var = ds.variables['time']
            self._times    = np.asarray(t_var[:], dtype=float)
            self._bedlevel = np.asarray(ds.variables['bedlevel'][:], dtype=float)

        self._t_min = self._times[0]
        self._t_max = self._times[-1]

        bed_start        = self._bedlevel[0, :]
        bed_min          = np.nanmin(self._bedlevel, axis=0)
        self.max_erosion = np.maximum(bed_start - bed_min, 0.0)  # (nNodes,)

        if logger:
            logger.info(
                '[BETA] Bed-level override: %d nodes, %d time steps, '
                't=[%.0f, %.0f] s, max_erosion p95=%.3f m',
                self._bedlevel.shape[1], len(self._times),
                self._t_min, self._t_max,
                float(np.nanpercentile(self.max_erosion, 95)),
            )

    def _bounds_indices(self, time_seconds: float) -> tuple[int, int, float]:
        """Return bracketing time indices and interpolation weight (time clamped to file range)."""
        t  = float(np.clip(time_seconds, self._t_min, self._t_max))
        i1 = int(np.searchsorted(self._times, t, side='right'))
        i1 = min(max(i1, 1), len(self._times) - 1)
        i0 = i1 - 1
        t0, t1 = self._times[i0], self._times[i1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return i0, i1, w

    def get_bed_level_bounds(self, time_seconds: float) -> dict:
        """Return lower/upper bed-level slices with interpolation weight.

        Mirrors ``FieldDataRetriever.get_scalar_field_bounds`` so the result can
        be passed anywhere a temporal bed-level field is accepted. Returning the
        temporal form is required: the static-array path in
        ``update_bed_level_change_after_movement`` only re-samples particles that
        moved, which would miss bed change under immobile (buried) particles.
        """
        i0, i1, w = self._bounds_indices(time_seconds)
        return {
            'lower': self._bedlevel[i0],
            'upper': self._bedlevel[i1],
            'weight': w,
            'lower_index': i0,
            'upper_index': i1,
        }

    def get_bed_level(self, time_seconds: float) -> np.ndarray:
        """Linearly interpolate bed level to *time_seconds* (clamped to file range)."""
        i0, i1, w = self._bounds_indices(time_seconds)
        if w <= 0.0:
            return self._bedlevel[i0].copy()
        return (1.0 - w) * self._bedlevel[i0] + w * self._bedlevel[i1]


class Simulation:
    """Encapsulate the particle simulation process.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS YAML configuration file.
    enable_dashboard : bool, optional
        Override for the dashboard setting from the configuration. If omitted,
        the configuration value is used.
    """

    _DASHBOARD_FULL_GRID_CELL_LIMIT = 100_000
    _DASHBOARD_LARGE_GRID_UPDATE_STRIDE = 10
    _OUTPUT_COORDINATE_FIELD_COUNT = 5
    _OUTPUT_STATUS_FIELD_COUNT = 8
    _DEFAULT_COMPRESSION_AUTO_THRESHOLD_MB = 1024
    _OUTPUT_DTYPE_BYTES = {
        'float32': 4,
        'f4': 4,
        'float64': 8,
        'f8': 8,
        'uint8': 1,
        'u1': 1,
        'int32': 4,
        'i4': 4,
    }

    def __init__(
        self,
        config_file: str,
        enable_dashboard: Optional[bool] = None,
        report_domain_exits: bool = True,
    ):
        """
        Initialize the simulation with the given configuration.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.
        enable_dashboard : bool, optional
            Override the dashboard setting from configuration. If None, uses config value.
        report_domain_exits : bool, default True
            If true, write final CLI/log summaries for particles that leave
            the domain or beach on land. Per-timestep update messages are
            controlled by ``general.report_domain_exit_updates``.
        """
        self._config_file = config_file
        self._enable_dashboard_override = enable_dashboard
        self._report_domain_exits = report_domain_exits
        self._report_domain_exit_updates = False

        self._start_time = None
        self._config_is_read = False
        self._populations_config = None
        self._profile_enabled = self._is_profile_enabled()
        self._profile_timings = {}
        self._profile_summary_logged = False
        self._active_progress_bar = None
        self._dashboard_throttle_logged = False

        # Validate config file exists early
        if not os.path.exists(config_file):
            raise ConfigurationError(f'Configuration file not found: {config_file}')

        # Try to read config and update logger directory
        try:
            # self._controller = ConfigurationController(self._config_file)
            self._controller = ConfigurationController(self._config_file)
            self._controller.load_config(self._config_file)
            self._report_domain_exit_updates = bool(
                self._controller.get('general.report_domain_exit_updates', False)
            )

            # TODO: logger has a circular dependency with controller. The logger needs refactoring.
            self.logger = logging.getLogger(__name__)

            self._config_is_read = True

            # self.logger_manager.setup_logger()
            # self._controller.log_after_load_config()

        except Exception:
            # Global exception handler will catch and log this
            raise
        # Initialize other components
        self.format_converter = FormatConverter(self._get_format_config())
        self.physics_converter = PhysicsConverter(self._get_physics_config())
        self.data_manager = DataManager(self._get_output_dir())
        self.particles: list[Particle] = []  # List to hold particles
        self.dashboard = self._create_dashboard()  #
        self.writer = self.data_manager.writer

        setup_logging(output_dir=str(self.writer.output_dir))  # Initialize logging in the results directory
        self.logger = logging.getLogger(__name__)
        self.logger.info('Configuration loaded')
        if self._profile_enabled:
            self.logger.info('Profiling enabled via SEDTRAILS_PROFILE')

    @staticmethod
    def _is_profile_enabled() -> bool:
        """Return whether lightweight simulation profiling is enabled."""
        return os.environ.get('SEDTRAILS_PROFILE', '').strip().lower() in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _parse_positive_int(value: Any, name: str) -> int:
        """Return a positive integer from a config or environment value."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ConfigurationError(f'{name} must be an integer, got {value!r}.') from None
        if parsed < 1:
            raise ConfigurationError(f'{name} must be greater than or equal to 1.')
        return parsed

    @staticmethod
    def _parallel_env_int(name: str, default: int) -> int:
        """Return an integer parallel-runtime environment variable."""
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            raise ConfigurationError(f'{name} must be an integer, got {raw_value!r}.') from None

    @classmethod
    def _parallel_task_context_from_env(cls) -> tuple[int, int]:
        """Return validated parallel task identity from the worker environment."""
        task_id = cls._parallel_env_int('SEDTRAILS_TASK_ID', 0)
        n_tasks = cls._parallel_env_int('SEDTRAILS_N_TASKS', 1)
        if n_tasks < 1:
            raise ConfigurationError('SEDTRAILS_N_TASKS must be greater than or equal to 1.')
        if task_id < 0 or task_id >= n_tasks:
            raise ConfigurationError(
                'SEDTRAILS_TASK_ID must satisfy 0 <= SEDTRAILS_TASK_ID < SEDTRAILS_N_TASKS.'
            )
        return task_id, n_tasks

    def _parallel_worker_count(self) -> int:
        """Return the configured local parallel worker count."""
        configured = self._controller.get('compute.n_workers', None)
        if configured is not None:
            return self._parse_positive_int(configured, 'compute.n_workers')

        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_cpus is not None:
            return self._parse_positive_int(slurm_cpus, 'SLURM_CPUS_PER_TASK')

        return 1

    def _clear_parallel_preloaded_input(self, preload_input: bool) -> None:
        """Release parent references to input data shared with worker processes."""
        _pw._PRELOADED_INPUT_DATA = None
        if preload_input:
            try:
                self.format_converter.format_plugin.input_data = None
            except Exception:
                pass

    @contextmanager
    def _profile_section(self, name: str):
        """Measure a section when profiling is enabled."""
        if not self._profile_enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stats = self._profile_timings.setdefault(name, {'count': 0, 'total': 0.0, 'max': 0.0})
            stats['count'] += 1
            stats['total'] += elapsed
            stats['max'] = max(stats['max'], elapsed)

    def _log_profile_summary(self, status: str = 'completed') -> None:
        """Write the aggregated profiling timings to the simulation logger."""
        if not self._profile_enabled:
            return

        if self._profile_summary_logged:
            return

        self._profile_summary_logged = True

        if not self._profile_timings:
            self.logger.info('Profiling enabled, but no sections were recorded (status=%s)', status)
            return

        self.logger.info('=== SEDTRAILS PROFILE SUMMARY (%s) ===', status)
        total_wall = sum(s['total'] for s in self._profile_timings.values())
        for name, stats in sorted(self._profile_timings.items(), key=lambda item: item[1]['total'], reverse=True):
            count = stats['count']
            total = stats['total']
            average = total / count if count else 0.0
            pct = total / total_wall * 100 if total_wall > 0 else 0.0
            self.logger.info(
                'profile %-42s count=%6d total=%10.6fs avg=%10.6fs max=%10.6fs %5.1f%%',
                name,
                count,
                total,
                average,
                stats['max'],
                pct,
            )

    @staticmethod
    def _particle_status_mask(population, status_name: str) -> np.ndarray:
        particles = getattr(population, 'particles', {})
        status = particles.get(status_name)
        if status is None:
            size = len(particles.get('x', []))
            return np.zeros(size, dtype=bool)
        return np.asarray(status, dtype=bool)

    @classmethod
    def _left_domain_mask(cls, population) -> np.ndarray:
        return cls._particle_status_mask(population, 'status_left_domain')

    @classmethod
    def _beached_mask(cls, population) -> np.ndarray:
        return cls._particle_status_mask(population, 'status_beached')

    @staticmethod
    def _population_name(population, fallback_index: int) -> str:
        config = getattr(population, 'population_config', {}) or {}
        if isinstance(config, dict):
            name = config.get('name')
        else:
            nested_config = getattr(config, 'population_config', None)
            if isinstance(nested_config, dict):
                name = nested_config.get('name')
            else:
                name = getattr(config, 'name', None)
        return str(name or f'population_{fallback_index + 1}')

    def _report_new_domain_exits(
        self,
        population,
        population_index: int,
        flow_field_name: str,
        reported_left_domain: np.ndarray,
        current_time: float,
        current_timestep: float,
    ) -> int:
        """Report newly left-domain particles for one population update."""

        if not self._report_domain_exits:
            return 0

        current_left_domain = self._left_domain_mask(population)
        if current_left_domain.shape != reported_left_domain.shape:
            return 0

        previous_total = int(np.count_nonzero(reported_left_domain))
        np.logical_or(reported_left_domain, current_left_domain, out=reported_left_domain)
        total_left = int(np.count_nonzero(reported_left_domain))
        newly_left_count = total_left - previous_total
        if newly_left_count == 0:
            return 0

        if getattr(self, '_report_domain_exit_updates', False):
            population_name = self._population_name(population, population_index)
            population_size = int(current_left_domain.size)
            self.logger.info(
                'Particles left domain: +%d in %s via %s at t=%.3fs (dt=%.3fs; population total=%d/%d)',
                newly_left_count,
                population_name,
                flow_field_name,
                current_time,
                current_timestep,
                total_left,
                population_size,
            )
        return newly_left_count

    def _report_new_beached_particles(
        self,
        population,
        population_index: int,
        flow_field_name: str,
        reported_beached: np.ndarray,
        current_time: float,
        current_timestep: float,
    ) -> int:
        """Report newly beached particles for one population update."""

        if not self._report_domain_exits:
            return 0

        current_beached = self._beached_mask(population)
        if current_beached.shape != reported_beached.shape:
            return 0

        previous_total = int(np.count_nonzero(reported_beached))
        np.logical_or(reported_beached, current_beached, out=reported_beached)
        total_beached = int(np.count_nonzero(reported_beached))
        newly_beached_count = total_beached - previous_total
        if newly_beached_count == 0:
            return 0

        if getattr(self, '_report_domain_exit_updates', False):
            population_name = self._population_name(population, population_index)
            population_size = int(current_beached.size)
            self.logger.info(
                'Particles beached on land: +%d in %s via %s at t=%.3fs (dt=%.3fs; population total=%d/%d)',
                newly_beached_count,
                population_name,
                flow_field_name,
                current_time,
                current_timestep,
                total_beached,
                population_size,
            )
        return newly_beached_count

    def _report_domain_exit_summary(self, populations) -> None:
        """Report final left-domain particle counts."""

        if not self._report_domain_exits:
            return

        total_left = 0
        total_particles = 0
        per_population = []
        for population_index, population in enumerate(populations):
            left_domain = self._left_domain_mask(population)
            population_left = int(np.count_nonzero(left_domain))
            population_size = int(left_domain.size)
            total_left += population_left
            total_particles += population_size
            if population_left:
                per_population.append(
                    f'{self._population_name(population, population_index)}={population_left}/{population_size}'
                )

        if total_left:
            details = f" ({', '.join(per_population)})" if per_population else ''
            self.logger.info('Particles left domain during run: %d/%d%s', total_left, total_particles, details)
        else:
            self.logger.info('Particles left domain during run: 0/%d', total_particles)

    def _report_beached_summary(self, populations, beached_masks: list[np.ndarray] | None = None) -> None:
        """Report final beached-particle counts."""

        if not self._report_domain_exits:
            return

        total_beached = 0
        total_particles = 0
        per_population = []
        for population_index, population in enumerate(populations):
            beached = self._beached_mask(population)
            if beached_masks is not None and population_index < len(beached_masks):
                beached_history = np.asarray(beached_masks[population_index], dtype=bool)
                if beached_history.shape == beached.shape:
                    beached = beached_history
            population_beached = int(np.count_nonzero(beached))
            population_size = int(beached.size)
            total_beached += population_beached
            total_particles += population_size
            if population_beached:
                per_population.append(
                    f'{self._population_name(population, population_index)}={population_beached}/{population_size}'
                )

        if total_beached:
            details = f" ({', '.join(per_population)})" if per_population else ''
            self.logger.info('Particles beached on land during run: %d/%d%s', total_beached, total_particles, details)
        else:
            self.logger.info('Particles beached on land during run: 0/%d', total_particles)

    def _create_dashboard(self):
        """Create and return a dashboard instance."""
        # Use override if provided, otherwise fall back to config setting
        dashboard_enabled = (
            self._enable_dashboard_override
            if self._enable_dashboard_override is not None
            else self._controller.get('visualization.dashboard.enable', False)
        )

        if dashboard_enabled and self._parallel_worker_count() > 1:
            self.logger.warning(
                'Dashboard disabled because compute.n_workers > 1 uses parallel worker processes.'
            )
            return None

        if dashboard_enabled:
            reference_date = self._controller.get('general.input_model.reference_date', '1970-01-01')
            figsize = (12, 8)
            dashboard = SimulationDashboard(reference_date=reference_date)
            dashboard.initialize_dashboard(figsize)
            # Force initial display and bring window to front
            dashboard.fig.show()
            dashboard.fig.canvas.draw()
            dashboard.fig.canvas.flush_events()

            # Try to bring window to front (cross-platform)
            try:
                dashboard.fig.canvas.manager.window.raise_()
                dashboard.fig.canvas.manager.window.activateWindow()
            except AttributeError:
                pass  # Some backends don't support this

            return dashboard
        else:
            return None

    def _should_update_dashboard(self, sedtrails_data, timer) -> bool:
        """Throttle full-grid dashboard redraws for large grids."""
        if self.dashboard is None:
            return False

        grid_size = np.size(sedtrails_data.x)
        if grid_size <= self._DASHBOARD_FULL_GRID_CELL_LIMIT:
            return True

        stride = self._controller.get(
            'visualization.dashboard.large_grid_update_stride',
            self._DASHBOARD_LARGE_GRID_UPDATE_STRIDE,
        )
        try:
            stride = max(1, int(stride))
        except (TypeError, ValueError):
            stride = self._DASHBOARD_LARGE_GRID_UPDATE_STRIDE

        if not self._dashboard_throttle_logged:
            self.logger.info(
                'Dashboard full-grid updates throttled to every %d steps for %d grid cells',
                stride,
                grid_size,
            )
            self._dashboard_throttle_logged = True

        return timer.step_count % stride == 0

    def _dashboard_update_interval_seconds(self) -> int:
        """Return the configured dashboard redraw interval in seconds."""
        update_interval = self._controller.get('visualization.dashboard.update_interval', '1H')
        return Duration(update_interval).seconds

    @staticmethod
    def _missing_particle_field_like(particle_x: np.ndarray) -> np.ndarray:
        """Return a same-shaped NaN particle field for unavailable dashboard data."""
        return np.full(np.asarray(particle_x).shape, np.nan, dtype=float)

    @classmethod
    def _dashboard_particle_data(cls, population) -> dict[str, np.ndarray]:
        """Build dashboard particle arrays from a population."""
        particle_x = population.particles['x']
        particle_data = {
            'x': particle_x,
            'y': population.particles['y'],
        }

        burial_depth = population.particles.get('burial_depth')
        if burial_depth is None:
            burial_depth = cls._missing_particle_field_like(particle_x)
        else:
            burial_depth = np.asarray(burial_depth)
            if np.all(np.isnan(burial_depth)):
                burial_depth = cls._missing_particle_field_like(particle_x)
        particle_data['burial_depth'] = burial_depth

        mixing_depth = population.particles.get('mixing_depth')
        particle_data['mixing_depth'] = (
            cls._missing_particle_field_like(particle_x) if mixing_depth is None else np.asarray(mixing_depth)
        )

        for status_name in ('status_left_domain', 'status_beached'):
            status = population.particles.get(status_name)
            particle_data[status_name] = (
                np.zeros(np.asarray(particle_x).shape, dtype=bool) if status is None else np.asarray(status, dtype=bool)
            )

        return particle_data

    def _create_simulation_time(self) -> Time:
        """Build simulation time using the same reference date as the input data."""
        return Time(
            _start=self._controller.get('time.start'),
            duration=Duration(self._controller.get('time.duration')),
            time_step=Duration(self._controller.get('time.timestep')),
            read_input_interval=Duration(self._controller.get('inputs.read_interval')),
            reference_date=self._controller.get('general.input_model.reference_date', '1970-01-01 00:00:00'),
        )

    @staticmethod
    def _needs_sedtrails_reload(sedtrails_data, current_time_seconds: float) -> bool:
        """Return whether the current time is outside the loaded SedTRAILS data chunk."""
        if sedtrails_data is None:
            return True

        times = np.asarray(sedtrails_data.times)
        if times.size == 0:
            return True

        return current_time_seconds < times[0] or current_time_seconds > times[-1]

    @staticmethod
    def _is_after_loaded_sedtrails_data(sedtrails_data, current_time_seconds: float) -> bool:
        """Return whether current time is after the last timestamp in the loaded data."""
        if sedtrails_data is None:
            return False

        times = np.asarray(sedtrails_data.times)
        if times.size == 0:
            return False

        return current_time_seconds > times[-1]

    @classmethod
    def _should_attempt_sedtrails_reload(
        cls, sedtrails_data, current_time_seconds: float, input_data_exhausted: bool
    ) -> bool:
        """Return whether the loop should try to load another SedTRAILS data chunk."""
        return not input_data_exhausted and cls._needs_sedtrails_reload(sedtrails_data, current_time_seconds)

    @staticmethod
    def _map_eulerian_field_time(
        current_time_seconds: float,
        repeat_eulerian_fields: bool,
        input_time_bounds: tuple[float, float] | None,
    ) -> float:
        """
        Map simulation time to Eulerian forcing time.

        When repeat_eulerian_fields is enabled, requests after the final input
        timestamp wrap to the first input timestamp. The returned value is used
        only for field loading and interpolation; output still records the
        simulation clock.
        """
        if not repeat_eulerian_fields or input_time_bounds is None:
            return current_time_seconds

        start, end = input_time_bounds
        cycle_duration = end - start
        if cycle_duration <= 0 or current_time_seconds <= end:
            return current_time_seconds

        return start + ((current_time_seconds - start) % cycle_duration)

    @staticmethod
    def _validate_simulation_start_matches_input(
        simulation_time: Time,
        input_time_bounds: tuple[float, float] | None,
    ) -> None:
        """Validate that the configured simulation start is covered by forcing data."""
        if input_time_bounds is None:
            return

        forcing_start, forcing_end = input_time_bounds
        simulation_start = float(simulation_time.start)
        if forcing_start <= simulation_start <= forcing_end:
            return

        configured_start = getattr(simulation_time, '_start', str(simulation_start))
        raise ConfigurationError(
            'time.start must be within the input forcing window. '
            f'time.start={configured_start!r} is {simulation_start:.3f}s since reference_date '
            f'{simulation_time.reference_date!r}, but forcing covers '
            f'[{forcing_start:.3f}, {forcing_end:.3f}]s. '
            'repeat_eulerian_fields only repeats forcing after a valid simulation start.'
        )

    @classmethod
    def _validate_simulation_time_matches_input(cls, simulation_time: Time, sedtrails_data) -> None:
        """Backward-compatible validation helper for loaded SedTRAILS data."""
        times = np.asarray(sedtrails_data.times, dtype=float)
        if times.size == 0:
            raise ConfigurationError('Input forcing contains no timestamps.')

        first_input_time = float(times[0])
        if simulation_time.end < first_input_time:
            raise ConfigurationError(
                'Simulation time window ends before the first available input field timestamp. '
                f'Simulation start/end are {simulation_time.start:.3f}s/{simulation_time.end:.3f}s, '
                f'but input fields start at {first_input_time:.3f}s since '
                f'{simulation_time.reference_date}. Adjust `time.start`, `time.duration`, or '
                '`general.input_model.reference_date` so the simulation reaches the forcing data.'
            )

        if simulation_time.start < first_input_time:
            raise ConfigurationError(
                'Simulation starts before the first available input field timestamp. '
                f'Simulation start/end are {simulation_time.start:.3f}s/{simulation_time.end:.3f}s, '
                f'but input fields start at {first_input_time:.3f}s since '
                f'{simulation_time.reference_date}. Adjust `time.start` or '
                '`general.input_model.reference_date` so the simulation starts inside the forcing data.'
            )

        cls._validate_simulation_start_matches_input(simulation_time, (first_input_time, float(times[-1])))

    def _output_save_interval_seconds(self) -> int:
        """Return the configured trajectory output cadence in seconds."""
        save_interval_seconds = Duration(self._controller.get('outputs.save_interval', '1H')).seconds
        if save_interval_seconds <= 0:
            raise ConfigurationError('outputs.save_interval must be a positive duration')
        return save_interval_seconds

    def _output_sync_interval_seconds(self, save_interval_seconds: int | float | None = None) -> int:
        """Return the configured NetCDF sync cadence in seconds.

        If unset, syncing defaults to the trajectory save interval.
        """
        if save_interval_seconds is None:
            save_interval_seconds = self._output_save_interval_seconds()

        sync_interval = self._controller.get('outputs.sync_interval', None)
        if sync_interval in (None, ''):
            return int(save_interval_seconds)

        sync_interval_seconds = Duration(sync_interval).seconds
        if sync_interval_seconds <= 0:
            raise ConfigurationError('outputs.sync_interval must be a positive duration')
        return sync_interval_seconds

    @staticmethod
    def _sync_every_n_writes(save_interval_seconds: int | float, sync_interval_seconds: int | float) -> int:
        """Convert save/sync durations to a streaming writer cadence."""
        if save_interval_seconds <= 0:
            raise ConfigurationError('outputs.save_interval must be a positive duration')
        if sync_interval_seconds <= 0:
            raise ConfigurationError('outputs.sync_interval must be a positive duration')
        return max(1, int(np.ceil(float(sync_interval_seconds) / float(save_interval_seconds))))

    def _output_netcdf_sync_interval_is_configured(self, sync_interval: int | None) -> bool:
        """Return whether ``outputs.netcdf.sync_interval`` was set explicitly."""
        has_configured_value = getattr(self._controller, 'has_configured_value', None)
        if callable(has_configured_value):
            return bool(has_configured_value('outputs.netcdf.sync_interval'))
        if self._controller.get('outputs.netcdf.sync_interval', None) is not None:
            return True
        return sync_interval is not None

    def _output_netcdf_options(self) -> dict[str, Any]:
        """Return NetCDF writer options, with defaults tuned for large particle tracks."""
        netcdf_config = self._controller.get('outputs.netcdf', {}) or {}
        sync_interval = netcdf_config.get('sync_interval')
        legacy_sync_interval = self._controller.get('outputs.sync_interval', None)
        if legacy_sync_interval not in (None, '') and not self._output_netcdf_sync_interval_is_configured(
            sync_interval
        ):
            save_interval_seconds = self._output_save_interval_seconds()
            sync_interval_seconds = self._output_sync_interval_seconds(save_interval_seconds)
            sync_interval = self._sync_every_n_writes(save_interval_seconds, sync_interval_seconds)
        elif sync_interval is None:
            legacy_sync_interval = self._controller.get('outputs.sync_interval', None)
            if legacy_sync_interval not in (None, ''):
                save_interval_seconds = self._output_save_interval_seconds()
                sync_interval_seconds = self._output_sync_interval_seconds(save_interval_seconds)
                sync_interval = self._sync_every_n_writes(save_interval_seconds, sync_interval_seconds)
            else:
                sync_interval = 10

        return {
            'coordinate_dtype': netcdf_config.get('coordinate_dtype', 'float32'),
            'status_dtype': netcdf_config.get('status_dtype', 'uint8'),
            'compression': netcdf_config.get('compression', 'auto'),
            'compression_auto_threshold_mb': int(
                netcdf_config.get(
                    'compression_auto_threshold_mb',
                    self._DEFAULT_COMPRESSION_AUTO_THRESHOLD_MB,
                )
            ),
            'compression_level': int(netcdf_config.get('compression_level', 1)),
            'shuffle': bool(netcdf_config.get('shuffle', True)),
            'time_chunk': int(netcdf_config.get('time_chunk', 1)),
            'particle_chunk': int(netcdf_config.get('particle_chunk', 65_536)),
            'sync_interval': sync_interval,
            'reopen_interval': netcdf_config.get('reopen_interval', None),
        }

    @classmethod
    def _estimate_netcdf_payload_bytes(
        cls,
        n_particles: int,
        n_output_slots: int,
        coordinate_dtype: str,
        status_dtype: str,
    ) -> int:
        """Estimate uncompressed particle payload bytes for NetCDF output."""
        coordinate_bytes = cls._OUTPUT_DTYPE_BYTES[str(coordinate_dtype)]
        status_bytes = cls._OUTPUT_DTYPE_BYTES[str(status_dtype)]
        bytes_per_particle_slot = (
            cls._OUTPUT_COORDINATE_FIELD_COUNT * coordinate_bytes
            + cls._OUTPUT_STATUS_FIELD_COUNT * status_bytes
        )
        return max(0, int(n_particles)) * max(1, int(n_output_slots)) * bytes_per_particle_slot

    @classmethod
    def _resolve_output_netcdf_options(
        cls,
        netcdf_options: dict[str, Any],
        n_particles: int,
        n_output_slots: int,
    ) -> dict[str, Any]:
        """Resolve auto compression into concrete NetCDF writer options."""
        resolved_options = dict(netcdf_options)
        compression = resolved_options.get('compression', 'auto')
        threshold_mb = int(
            resolved_options.pop(
                'compression_auto_threshold_mb',
                cls._DEFAULT_COMPRESSION_AUTO_THRESHOLD_MB,
            )
        )

        if compression == 'auto':
            estimated_bytes = cls._estimate_netcdf_payload_bytes(
                n_particles,
                n_output_slots,
                resolved_options['coordinate_dtype'],
                resolved_options['status_dtype'],
            )
            resolved_options['compression'] = estimated_bytes >= threshold_mb * 1024 * 1024
        elif isinstance(compression, bool):
            resolved_options['compression'] = compression
        else:
            raise ConfigurationError(
                "outputs.netcdf.compression must be true, false, or 'auto'."
            )

        return resolved_options

    def _output_checkpoint_options(self, writer_options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return checkpoint policy and shared NetCDF encoding options."""
        netcdf_config = self._controller.get('outputs.netcdf', {}) or {}
        if writer_options is None:
            writer_options = self._resolve_output_netcdf_options(
                self._output_netcdf_options(),
                n_particles=0,
                n_output_slots=1,
            )
        return {
            'enabled': bool(netcdf_config.get('checkpoint', True)),
            'interval': int(netcdf_config.get('checkpoint_interval', 0)),
            'writer_kwargs': {
                key: writer_options[key]
                for key in (
                    'coordinate_dtype',
                    'status_dtype',
                    'compression',
                    'compression_level',
                    'shuffle',
                    'particle_chunk',
                )
            },
        }

    def _resolved_output_writer_options(
        self,
        total_particles: int,
        n_output_slots: int,
        store_tracks: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Return concrete trajectory, snapshot, and checkpoint writer options."""
        raw_netcdf_options = self._output_netcdf_options()
        netcdf_options = self._resolve_output_netcdf_options(
            raw_netcdf_options,
            total_particles,
            n_output_slots if store_tracks else 1,
        )
        snapshot_netcdf_options = self._resolve_output_netcdf_options(
            raw_netcdf_options,
            total_particles,
            1,
        )
        checkpoint_options = self._output_checkpoint_options(snapshot_netcdf_options)
        return netcdf_options, snapshot_netcdf_options, checkpoint_options

    def _output_store_tracks(self) -> bool:
        """Return whether output should store the full trajectory cube."""
        if self._controller.get('outputs.store_end_positions', False):
            return False
        return bool(self._controller.get('outputs.store_tracks', True))

    def _maybe_write_checkpoint(
        self,
        populations,
        current_time: int | float,
        simulation_time: Time,
        saved_slots: int,
        checkpoint_options: dict[str, Any],
        *,
        final: bool = False,
    ) -> None:
        """Write the compact checkpoint when the configured checkpoint policy says so."""
        if not checkpoint_options.get('enabled', True):
            return

        interval = int(checkpoint_options.get('interval', 0))
        if not final and (interval <= 0 or saved_slots % interval != 0):
            return

        self.data_manager.writer.write_checkpoint(
            'sedtrails_checkpoint.nc',
            populations,
            float(current_time),
            reference_date=str(simulation_time.reference_date),
            time_units=f'seconds since {simulation_time.reference_date}',
            **checkpoint_options.get('writer_kwargs', {}),
        )

    @staticmethod
    def _estimate_output_timesteps(simulation_time: Time, save_interval_seconds: int | float) -> int:
        """Count output slots for initial, scheduled, and final trajectory samples."""
        if save_interval_seconds <= 0:
            raise ConfigurationError('outputs.save_interval must be a positive duration')

        duration_seconds = float(simulation_time.duration.seconds)
        if duration_seconds <= 0:
            return 1

        interval_count = int(np.floor(duration_seconds / float(save_interval_seconds)))
        count = 1 + interval_count
        if interval_count * float(save_interval_seconds) < duration_seconds - 1.0e-9:
            count += 1
        return max(1, count)

    @staticmethod
    def _next_scheduled_output_time(
        simulation_time: Time,
        save_interval_seconds: int | float,
        output_index: int,
    ) -> float:
        """Return the next scheduled output time, capped at the simulation end."""
        return float(min(simulation_time.start + output_index * float(save_interval_seconds), simulation_time.end))

    @staticmethod
    def _limit_timestep_to_output_schedule(
        current_time: int | float,
        current_timestep: int | float,
        next_output_time: int | float,
    ) -> float:
        """Shorten a CFL step only when it would cross the next output boundary."""
        remaining_to_output = float(next_output_time) - float(current_time)
        if 0.0 < remaining_to_output < float(current_timestep):
            return remaining_to_output
        return float(current_timestep)

    @staticmethod
    def _is_output_sample_due(
        sample_time: int | float,
        next_output_time: int | float,
        end_time: int | float,
    ) -> bool:
        """Return whether a trajectory sample should be stored at this time."""
        tolerance = 1.0e-9
        return sample_time + tolerance >= next_output_time or sample_time + tolerance >= end_time

    @staticmethod
    def _should_update_bed_level_after_movement(tracer_plan) -> bool:
        """Return whether post-move bed-level resampling should run.

        Rules for current 2D workflows:
        - Always run for ``vanwesten`` to preserve burial-depth bookkeeping.
        - Also run for any tracer configured with ``no_probability`` so
          particle ``z`` follows bed level after movement (passive/soulsby included).

        Future quasi-3D tracers can introduce exceptions here when vertical
        position is solved independently from bed level.
        """
        return (
            tracer_plan.method_name == 'vanwesten'
            or tracer_plan.transport_probability_method == 'no_probability'
        )

    @staticmethod
    def _initialize_population_output_status(populations, current_time: int | float) -> None:
        """Populate required status arrays before the initial trajectory sample is written."""
        for population in populations:
            particles = population.particles
            n_particles = len(particles['x'])
            particles.setdefault('status_alive', np.ones(n_particles, dtype=bool))
            particles.setdefault('status_buried', np.zeros(n_particles, dtype=bool))
            particles.setdefault('status_transported', np.zeros(n_particles, dtype=bool))

            if 'status_domain' not in particles:
                simplices = getattr(population, '_particle_simplices', None)
                if simplices is None:
                    particles['status_domain'] = np.ones(n_particles, dtype=bool)
                else:
                    particles['status_domain'] = np.asarray(simplices) >= 0

            if 'status_released' not in particles:
                release_time = particles.get('release_time')
                if release_time is None:
                    particles['status_released'] = np.ones(n_particles, dtype=bool)
                else:
                    particles['status_released'] = float(current_time) >= np.asarray(release_time, dtype=float)

            particles['status_mobile'] = (
                np.asarray(particles['status_domain'], dtype=bool)
                & np.asarray(particles['status_alive'], dtype=bool)
                & ~np.asarray(particles['status_buried'], dtype=bool)
                & np.asarray(particles['status_released'], dtype=bool)
                & np.asarray(particles['status_transported'], dtype=bool)
            )

    def _get_format_config(self):
        """
        Returns configuration parameters required for the format converter.
        """

        format_config = {
            'input_file': self._controller.get('inputs.data'),
            'input_format': self._controller.get('general.input_model.format'),  # Specify the input format
            'reference_date': self._controller.get('general.input_model.reference_date'),
            'morfac': self._controller.get('general.input_model.morfac', 1.0),
            'domain_config': self._get_domain_config(),
        }

        return format_config

    def _get_domain_config(self):
        """Return domain config with relative Tekal polygon files resolved."""

        domain_config = dict(self._controller.get('domain', {}) or {})
        config_dir = Path(self._config_file).parent

        inner_files = domain_config.get('inner_boundary_pol_files')
        if inner_files:
            domain_config['inner_boundary_pol_files'] = self._resolve_polygon_files(inner_files, config_dir)

        boundary_class_files = domain_config.get('boundary_class_pol_files')
        if boundary_class_files:
            domain_config['boundary_class_pol_files'] = {
                boundary_class: self._resolve_polygon_files(pol_files, config_dir)
                for boundary_class, pol_files in boundary_class_files.items()
            }
        return domain_config

    @staticmethod
    def _resolve_polygon_files(pol_files, config_dir: Path) -> list[str]:
        """Resolve one or more polygon file paths relative to the config file."""

        if isinstance(pol_files, (str, Path)):
            pol_files = [pol_files]

        resolved_files = []
        for pol_file in pol_files:
            path = Path(pol_file)
            if not path.is_absolute():
                path = config_dir / path
            resolved_files.append(str(path))
        return resolved_files

    def _get_output_dir(self):
        """
        Returns the output directory for the simulation.

        If the output directory is not explicitly specified or is a default value,
        uses the directory containing the config file as the base directory.

        Returns
        -------
        Path
            Path object representing the output directory
        """
        output_dir = self._controller.get('outputs.directory')

        # Check if output_dir is a default or relative path that should be relative to config file
        if output_dir is None or output_dir in ['.', './output', 'output', './results', 'results']:
            # Use the config file's directory as the base
            config_file_dir = Path(self._config_file).parent
            if output_dir in ['./output', 'output']:
                # Preserve the 'output' subdirectory but make it relative to config file
                output_dir = config_file_dir / 'output'
            elif output_dir in ['./results', 'results']:
                # Preserve the 'results' subdirectory but make it relative to config file
                output_dir = config_file_dir / 'results'
            else:
                # Use config file directory directly
                output_dir = config_file_dir
        else:
            # Convert to Path object for consistency
            output_dir = Path(output_dir)

        # In parallel runs each worker writes to its own task_XXXX/ subdirectory,
        # which the parent merges into a single output file after all workers finish.
        task_id, n_tasks = self._parallel_task_context_from_env()
        if n_tasks > 1:
            output_dir = output_dir / f'task_{task_id:04d}'

        return output_dir

    def _get_physics_config(self):
        """
        Returns configuration parameters required for the physics converter.
        """
        from sedtrails.transport_converter.physics_converter import PhysicsConfig

        config = PhysicsConfig(
            gravity=self._controller.get('physics.constants.g', 9.81),
            von_karman_constant=self._controller.get('physics.constants.von_karman', 0.40),
            kinematic_viscosity=self._controller.get('physics.constants.kinematic_viscosity', 1.36e-6),
            water_density=self._controller.get('physics.constants.rho_w', 1027.0),
            particle_density=self._controller.get('physics.constants.rho_s', 2650.0),
            porosity=self._controller.get('physics.constants.porosity', 0.4),
            grain_diameter=self._controller.get('physics.constants.grain_diameter', 2.5e-4),
            morfac=self._controller.get('physics.constants.morphology_factor', 1.0),
            bertin_coefficient=self._controller.get('physics.constants.bertin_coefficient', 0.041),
            # trapped_exposed_method=self._controller.get('physics.trapped_exposed_method', 'reduced_velocity'), # other option; 'probabilistic_exposure'
        )

        return config

    def _remove_permanently_buried_populations(self, populations, runtime_plans, bed_level_override=None) -> None:
        """Remove particles whose burial depth can never be exposed.

        Parameters
        ----------
        populations : sequence
            Seeded particle populations.
        runtime_plans : sequence
            Runtime plans paired with the seeded populations.
        bed_level_override : _BedLevelOverride, optional
            [BETA] When set, its max erosion (from the separate bed-level file)
            replaces the main input's max erosion; max BSS still comes from the
            main hydrodynamic input.

        Raises
        ------
        NotImplementedError
            If removal is enabled but the format plugin cannot provide maximum
            exposure fields.
        ValueError
            If a physics converter does not expose ``critical_shear_stress``.
        """
        # Only scan the full dataset when at least one population requests the
        # optimization.
        if not any(pop.population_config.remove_permanently_buried for pop in populations):
            return

        plugin = self.format_converter.format_plugin
        if not hasattr(plugin, 'get_max_exposure_depth_fields'):
            raise NotImplementedError(
                f"Format plugin '{type(plugin).__name__}' does not implement "
                "'get_max_exposure_depth_fields'. Cannot compute max exposure depth for permanent burial removal."
            )

        with self._profile_section('get_max_exposure_depth'):
            max_erosion, max_bss = plugin.get_max_exposure_depth_fields()
        if bed_level_override is not None:
            max_erosion = bed_level_override.max_erosion

        from sedtrails.transport_converter import physics_lib

        for plan in runtime_plans:
            pop = plan.population
            if not pop.population_config.remove_permanently_buried:
                continue

            critical_shear_stress = plan.tracer.converter.grain_properties.get('critical_shear_stress')
            if critical_shear_stress is None:
                raise ValueError("Physics converter does not provide 'critical_shear_stress' in grain_properties.")

            max_mixing = physics_lib.compute_mixing_layer_thickness(
                max_bss,
                critical_shear_stress,
                bertin_coefficient=plan.tracer.converter.config.bertin_coefficient,
            )
            pop.remove_permanently_buried_particles(max_erosion + max_mixing)

    @property
    def config(self):
        """
        Returns the full configuration settings for the simulation.

        Returns
        -------
        dict
            Full validated simulation configuration.
        """
        if not self._config_is_read:
            self._controller.load_config(self._config_file)
        return self._controller.get_config()  # delagates to the controller

    @property
    def populations_config(self):
        """
        Returns the configuration paramters for 'populations'.

        Returns
        -------
        dict
            Population configuration mapping.
        """
        if self._populations_config is None:
            self._populations_config = self.config.get('particles', {}).get('populations', {})
        return self._populations_config

    @property
    def start_time(self):
        """
        Get the start time parameter for the simulation.

        Returns
        -------
        object
            Configured simulation start time value.
        """
        if not self._start_time:
            self._start_time = self._controller.get('time.start_time')  # defaults to Unix epoch
        return self._start_time

    @property
    def flow_field(self) -> SedtrailsData:
        """
        Returns input flow field data in SedtrailsData format.

        Returns
        -------
        SedtrailsData
            The flow field value.
        """

        return self.format_converter.convert_to_sedtrails()

    def validate_config(self) -> bool:
        """
        Validates the configuration file.

        Returns
        -------
        bool
            Boolean result of the check.
        """
        if not self._config_is_read:  # assure config is read only once
            try:
                self._controller.load_config(self._config_file)
                self._config_is_read = True
                return True
            except Exception as e:
                raise ConfigurationError(f'Error validating configuration file: {e}')  # noqa: B904
                return False  # validation fails
            else:
                return True  # validation succeeds
        else:
            # if config is already read, the file is already validated
            return True

    def get_parameter(self, key: str) -> Any:  # TODO: this is not used anywhere. Is it needed?
        """
        Returns the value of a specific parameter in the configuration file.

        Parameters
        ----------
        key : str
            The dot-separated key to retrieve.

        Returns
        -------
        Any
            The value associated with the key in the configuration file.

        Raises
        ------
        Warning
            If the key is not found in the configuration file.
        """

        import warnings

        if not self._config_is_read:  # assure config is read only once
            self._controller.load_config(self._config_file)

        value = self._controller.get(key, None)
        if value is None:
            warnings.warn(f'Key "{key}" not found in configuration file', UserWarning, stacklevel=2)
        return value

    def run(self):
        """
        Execute the simulation, dispatching to parallel workers when requested.

        Returns
        -------
        object
            Value returned by the simulation implementation.
        """
        try:
            # Workers set SEDTRAILS_N_TASKS before calling run(); skip parallel dispatch
            # to prevent infinite recursion (workers always go straight to _run_impl).
            if self._parallel_task_context_from_env()[1] > 1:
                return self._run_impl()

            n_workers = self._parallel_worker_count()
            if n_workers > 1:
                return self._run_parallel(n_workers)
            return self._run_impl()
        except Exception:
            if self._active_progress_bar is not None:
                self._active_progress_bar.close()
                self._active_progress_bar = None
            self._log_profile_summary(status='interrupted')
            raise

    def _run_parallel(self, n_workers: int) -> None:
        """Fork n_workers subprocesses each handling a particle slice, then auto-merge.

        When ``compute.preload_input`` is true, the DFM dataset is loaded once
        in the parent and shared with workers via Linux copy-on-write. When it
        is false, each worker reads its configured input chunks independently.
        """
        import gc
        import multiprocessing

        base_output_dir = self._get_output_dir()
        self.logger.info('Starting parallel run with %d workers -> %s', n_workers, base_output_dir)

        try:
            fork_context = multiprocessing.get_context('fork')
        except ValueError as exc:
            raise ConfigurationError(
                'compute.n_workers > 1 requires a Python multiprocessing fork context. '
                'Run this backend on Linux/WSL, or set compute.n_workers: 1.'
            ) from exc

        preload_input = bool(self._controller.get('compute.preload_input', False))
        worker_timeout = self._parse_positive_int(
            os.environ.get('SEDTRAILS_WORKER_TIMEOUT', 82800),
            'SEDTRAILS_WORKER_TIMEOUT',
        )
        try:
            read_interval_s = float(Duration(self._controller.get('inputs.read_interval')).seconds)
            duration_s = float(Duration(self._controller.get('time.duration')).seconds)
        except Exception:
            read_interval_s = duration_s = float('inf')

        dfm_mb = 0.0
        if preload_input:
            try:
                # Preload DFM into numpy arrays in the parent so all workers share them via CoW.
                plugin = self.format_converter.format_plugin
                plugin.load()
                try:
                    dfm_mb = plugin.input_data.nbytes / 1e6
                    _pw._warn_if_oom_risk(dfm_mb, n_workers, read_interval_s, duration_s, self.logger)
                except Exception:
                    dfm_mb = 0.0
                plugin.input_data.load()
                _pw._PRELOADED_INPUT_DATA = plugin.input_data
                if dfm_mb > 0.0:
                    self.logger.info('DFM preloaded (%.0f MB), forking %d workers', dfm_mb, n_workers)
                else:
                    self.logger.info('DFM preloaded, forking %d workers', n_workers)
            except Exception:
                self._clear_parallel_preloaded_input(preload_input)
                gc.collect()
                raise
        else:
            _pw._PRELOADED_INPUT_DATA = None
            self.logger.info(
                'Parallel input preload disabled; workers will read input chunks independently'
            )
        wall_t0 = time.perf_counter()

        args = [(self._config_file, i, n_workers) for i in range(n_workers)]
        worker_error = None
        # Explicit fork context: CoW memory sharing only works with fork, not spawn.
        try:
            pool_context = fork_context.Pool(processes=n_workers)
        except Exception:
            self._clear_parallel_preloaded_input(preload_input)
            gc.collect()
            raise

        with pool_context as pool:
            result = pool.starmap_async(_pw._worker_fn, args)
            try:
                result.get(timeout=worker_timeout)
            except multiprocessing.TimeoutError as exc:
                worker_error = exc
                self.logger.warning(
                    'Worker timeout after %.0fs - terminating stuck workers', worker_timeout)
                pool.terminate()
            except Exception as exc:
                worker_error = exc
                self.logger.warning(
                    'Worker exception: %s; terminating pool without merging partial outputs', exc)
                pool.terminate()

        # Drop references before the merge so the parent can reclaim memory.
        self._clear_parallel_preloaded_input(preload_input)
        gc.collect()

        if worker_error is not None:
            raise RuntimeError(
                'Parallel worker pool failed; partial worker outputs were not merged.'
            ) from worker_error

        _pw._merge_outputs(base_output_dir, self.logger)
        self.logger.info('Total wall time: %.1fs (%d workers)',
                         time.perf_counter() - wall_t0, n_workers)

    def _build_plan_retrievers(self, sedtrails_data, runtime_plans):
        """Build population-scoped field retrievers for one loaded data chunk.

        Parameters
        ----------
        sedtrails_data : SedtrailsData
            Converted Eulerian fields for the current input chunk.
        runtime_plans : tuple
            Population-specific tracer runtime plans.

        Returns
        -------
        dict[int, FieldDataRetriever]
            Field retrievers indexed by population index.
        """
        input_model_config = self._controller.get('general.input_model', {})
        default_fraction_index = input_model_config.get('sediment_fraction_index', 0)
        default_fraction_name = input_model_config.get('sediment_fraction_name')

        return {
            runtime_plan.population_index: FieldDataRetriever(
                build_plan_sedtrails_data(
                    sedtrails_data,
                    runtime_plan.tracer,
                    population_config=runtime_plan.population_config,
                    default_fraction_index=default_fraction_index,
                    default_fraction_name=default_fraction_name,
                )
            )
            for runtime_plan in runtime_plans
        }

    def _run_impl(self):
        """
        Executes the particle simulation workflow.
        """

        # Loading configuration
        if not self._config_is_read:  # assure config is read only once
            self._controller.load_config(self._config_file)
            self._config_is_read = True

        wall_t0 = time.perf_counter()

        populations_config = self._controller.get('particles.populations', [])
        validate_population_runtime_configurations(populations_config)

        # Time configuration
        simulation_time = self._create_simulation_time()

        timer = Timer(simulation_time=simulation_time, cfl_condition=self._controller.get('time.cfl_condition'))
        repeat_eulerian_fields = self._controller.get('inputs.repeat_eulerian_fields', False)
        input_time_bounds = self.format_converter.get_time_bounds()
        self._validate_simulation_start_matches_input(simulation_time, input_time_bounds)
        if repeat_eulerian_fields and input_time_bounds is not None:
            self.logger.info(
                'Repeating Eulerian flow fields from %.3fs after forcing end %.3fs',
                input_time_bounds[0],
                input_time_bounds[1],
            )

        # [BETA] Separate bed-level file - read at real simulation time, no repeat wrapping.
        bed_level_override = None
        _bed_level_data_path = self._controller.get('inputs.bed_level_data', None)
        if _bed_level_data_path:
            bed_level_override = _BedLevelOverride(_bed_level_data_path, self.logger)

        # Load only x/y field coordinates needed for the population seeder.
        with self._profile_section('get_seeding_field_data'):
            seeding_field_data = self.format_converter.get_seeding_field_data()

        seeder = ParticleSeeder(populations_config)  # intialize seeder with population config
        populations = seeder.seed(seeding_field_data)  # seed particles for all populations
        runtime_plans = build_population_runtime_plans(populations_config, populations, self._get_physics_config())
        flow_field_names = unique_flow_field_names(runtime_plans)

        self._remove_permanently_buried_populations(populations, runtime_plans, bed_level_override)

        # Set initial values
        sedtrails_data = None

        # Initialize progress bar
        pbar = tqdm(
            total=100,
            desc='Computing positions',
            unit='%',
            bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}, {postfix}]',
            smoothing=0,
            disable=not sys.stderr.isatty(),
        )
        self._active_progress_bar = pbar

        log_simulation_state(
            self.logger,
            {
                'status': 'simulation_started',
                'command': ' '.join(sys.argv),
                'python_version': sys.version.split()[0],
                'config_file': self._config_file,
                'working_directory': os.getcwd(),
            },
        )
        # Create SedTrails dataset using DataManager's writer (composition)
        total_particles = sum([len(pop.particles['x']) for pop in populations])
        store_tracks = self._output_store_tracks()
        save_interval_seconds = self._output_save_interval_seconds() if store_tracks else None
        n_output_slots = (
            self._estimate_output_timesteps(simulation_time, save_interval_seconds)
            if store_tracks
            else 0
        )
        netcdf_options, _, checkpoint_options = self._resolved_output_writer_options(
            total_particles,
            n_output_slots,
            store_tracks,
        )

        self._initialize_population_output_status(populations, timer.current)
        nc_handle = None
        slot_idx = 0
        last_saved_time = None
        next_output_time = simulation_time.end

        if store_tracks:
            self.logger.info(
                'Streaming output: %d slots at %gs interval -> %s',
                n_output_slots,
                save_interval_seconds,
                self.data_manager.writer.output_dir / 'sedtrails_results.nc',
            )

            nc_handle = self.data_manager.writer.open_output(
                'sedtrails_results.nc',
                n_output_slots,
                total_particles,
                len(populations),
                len(flow_field_names) if flow_field_names else 1,
                populations,
                flow_field_names,
                **netcdf_options,
            )
            nc_handle.reference_date = str(simulation_time.reference_date)
            nc_handle.time_units = f'seconds since {simulation_time.reference_date}'
            nc_handle.time_start = self._controller.get('time.start')
            nc_handle.time_end_seconds_since_reference_date = float(simulation_time.end)
            nc_handle.outputs_save_interval_seconds = float(save_interval_seconds)
            nc_handle['time'].units = nc_handle.time_units
            nc_handle['time'].reference_date = nc_handle.reference_date

            # Store the seeded initial state before the first physics update.
            with self._profile_section('record_output'):
                nc_handle = self.data_manager.writer.record_output(nc_handle, populations, slot_idx, timer.current)
            last_saved_time = timer.current
            slot_idx += 1
            self._maybe_write_checkpoint(
                populations,
                timer.current,
                simulation_time,
                slot_idx,
                checkpoint_options,
            )
            next_output_time = self._next_scheduled_output_time(
                simulation_time,
                save_interval_seconds,
                slot_idx,
            )
        else:
            self.logger.info(
                'End-position output enabled: final state will be written to %s',
                self.data_manager.writer.output_dir / 'sedtrails_results.nc',
            )
        left_domain_reported_masks = [self._left_domain_mask(population).copy() for population in populations]
        beached_ever_masks = [self._beached_mask(population).copy() for population in populations]
        # Main simulation loop with variable timestep
        input_data_exhausted = False
        input_exhaustion_warning_logged = False
        plan_retrievers = {}
        dashboard_flow_field = None
        try:
            while not timer.stop and timer.current < simulation_time.end:
                # Check if current time is within loaded SedTRAILS data
                current_time_seconds = timer.current
                field_time_seconds = self._map_eulerian_field_time(
                    current_time_seconds,
                    repeat_eulerian_fields,
                    input_time_bounds,
                )
                if self._should_attempt_sedtrails_reload(sedtrails_data, field_time_seconds, input_data_exhausted):
                    # Avoid recreating SedTRAILS data if current time is before the first time step
                    if (
                        sedtrails_data is not None
                        and field_time_seconds < sedtrails_data.times[0]
                        and not repeat_eulerian_fields
                    ):
                        timer.advance()
                        continue
                    # Convert to SedTRAILS format
                    with self._profile_section('convert_to_sedtrails'):
                        sedtrails_data = self.format_converter.convert_to_sedtrails(
                            current_time=field_time_seconds, reading_interval=simulation_time.read_input_interval.seconds
                        )
                    plan_retrievers = self._build_plan_retrievers(sedtrails_data, runtime_plans)

                    if self._is_after_loaded_sedtrails_data(sedtrails_data, field_time_seconds):
                        input_data_exhausted = True
                        if not input_exhaustion_warning_logged:
                            self.logger.warning(
                                'Simulation time %.3fs is beyond the final input field timestamp %.3fs; '
                                'reusing the last available fields for remaining timesteps.',
                                current_time_seconds,
                                float(np.asarray(sedtrails_data.times)[-1]),
                            )
                            input_exhaustion_warning_logged = True

                # TODO: integrate loop over flow fields into CFL Condition
                # Collect flow fields for CFL computation
                max_velocity = 0.0
                for runtime_plan in runtime_plans:
                    retriever = plan_retrievers[runtime_plan.population_index]
                    for flow_field_name in runtime_plan.tracer.flow_field_names:
                        with self._profile_section('get_flow_max_velocity.cfl'):
                            max_velocity = max(
                                max_velocity,
                                retriever.get_flow_max_velocity_bound(field_time_seconds, flow_field_name),
                            )

                # Compute CFL-based timestep across all flow fields
                with self._profile_section('compute_cfl_timestep'):
                    timer.compute_cfl_timestep_from_max_velocity(
                        max_velocity,
                        sedtrails_data.metadata.min_resolution,
                        sedtrails_data.metadata.timestep,
                    )
                    timer.current_timestep = min(timer.current_timestep, simulation_time.end - timer.current)
                    if store_tracks and slot_idx < n_output_slots:
                        timer.current_timestep = self._limit_timestep_to_output_schedule(
                            timer.current,
                            timer.current_timestep,
                            next_output_time,
                        )
                    if timer.current_timestep <= 0:
                        raise ConfigurationError(
                            f'Computed non-positive timestep {timer.current_timestep} '
                            f'at simulation time {timer.current}.'
                        )

                # Main loop
                dashboard_flow_field = None
                plot_interval_seconds = None
                dashboard_update_due = False
                if self._should_update_dashboard(sedtrails_data, timer):
                    plot_interval_seconds = self._dashboard_update_interval_seconds()
                    dashboard_update_due = self.dashboard.should_update(timer.current, plot_interval_seconds)

                for runtime_plan in runtime_plans:
                    population = runtime_plan.population
                    tracer_plan = runtime_plan.tracer
                    retriever = plan_retrievers[runtime_plan.population_index]

                    if tracer_plan.transport_probability_method == 'no_probability':
                        mixing_depth = None
                    else:
                        with self._profile_section('get_scalar_field_bounds.mixing_layer_thickness'):
                            mixing_depth = retriever.get_scalar_field_bounds(field_time_seconds, 'mixing_layer_thickness')
                    with self._profile_section('get_scalar_field_bounds.bed_level'):
                        if bed_level_override is not None:
                            bed_level = bed_level_override.get_bed_level_bounds(current_time_seconds)
                        else:
                            bed_level = retriever.get_scalar_field_bounds(field_time_seconds, 'bed_level')

                    for flow_field_name in tracer_plan.flow_field_names:
                        if tracer_plan.method_name == 'vanwesten':
                            with self._profile_section('get_scalar_field_bounds.transport_probability'):
                                transport_prob = retriever.get_scalar_field_bounds(
                                    field_time_seconds, flow_field_name.replace('velocity', 'probability')
                                )
                        else:
                            transport_prob = 1.0

                        with self._profile_section('update_information'):
                            population.update_information(
                                current_time=timer.current,
                                mixing_depth=mixing_depth,
                                bed_level=bed_level,
                                transport_probability=transport_prob,
                            )

                        if tracer_plan.method_name == 'vanwesten':
                            transport_probability_method = tracer_plan.transport_probability_method
                            if transport_probability_method != 'no_probability':
                                with self._profile_section('update_burial_depth'):
                                    population.update_burial_depth()


                        with self._profile_section('update_status'):
                            population.update_status()

                        with self._profile_section('get_flow_field_bounds.update_position'):
                            flow_field = retriever.get_flow_field_bounds(field_time_seconds, flow_field_name)
                        if (
                            dashboard_update_due
                            and runtime_plan.population_index == 0
                            and flow_field_name == tracer_plan.flow_field_names[0]
                        ):
                            with self._profile_section('get_flow_field.dashboard'):
                                dashboard_flow_field = retriever.get_flow_field(field_time_seconds, flow_field_name)

                        with self._profile_section('update_position'):
                            population.update_position(flow_field=flow_field, current_timestep=timer.current_timestep)
                        self._report_new_domain_exits(
                            population,
                            runtime_plan.population_index,
                            flow_field_name,
                            left_domain_reported_masks[runtime_plan.population_index],
                            timer.current,
                            timer.current_timestep,
                        )
                        self._report_new_beached_particles(
                            population,
                            runtime_plan.population_index,
                            flow_field_name,
                            beached_ever_masks[runtime_plan.population_index],
                            timer.current,
                            timer.current_timestep,
                        )

                        # Re-sample bed level at the new positions after movement.
                        # Keep this INSIDE the flow-field loop so vanwesten
                        # burial bookkeeping compares temporal bed change only,
                        # and so no_probability tracers keep z aligned to bed
                        # level after advection.
                        if self._should_update_bed_level_after_movement(tracer_plan):
                            with self._profile_section('update_bed_level_after_movement'):
                                population.update_bed_level_change_after_movement(bed_level)

                # Update dashboard if enabled
                if dashboard_update_due and dashboard_flow_field is not None:
                    # For dashboard, use first population data
                    first_population = populations[0]
                    particle_data = self._dashboard_particle_data(first_population)
                    dashboard_retriever = plan_retrievers[runtime_plans[0].population_index]
                    with self._profile_section('get_scalar_field.dashboard_bed_level'):
                        if bed_level_override is not None:
                            bathymetry = bed_level_override.get_bed_level(current_time_seconds)
                        else:
                            bathymetry = dashboard_retriever.get_scalar_field(field_time_seconds, 'bed_level')['magnitude']
                    mesh_geometry = sedtrails_data.mesh_geometry() if hasattr(sedtrails_data, 'mesh_geometry') else None

                    self.dashboard.update(
                        dashboard_flow_field,
                        bathymetry,
                        particle_data,
                        timer.current,
                        timer.current_timestep,
                        plot_interval_seconds,
                        simulation_start_time=simulation_time.start,
                        simulation_end_time=simulation_time.end,
                        mesh_geometry=mesh_geometry,
                    )

                timer.advance()

                if (
                    store_tracks
                    and nc_handle is not None
                    and slot_idx < n_output_slots
                    and self._is_output_sample_due(timer.current, next_output_time, simulation_time.end)
                ):
                    with self._profile_section('record_output'):
                        nc_handle = self.data_manager.writer.record_output(
                            nc_handle, populations, slot_idx, timer.current
                        )
                    last_saved_time = timer.current
                    slot_idx += 1
                    self._maybe_write_checkpoint(
                        populations,
                        timer.current,
                        simulation_time,
                        slot_idx,
                        checkpoint_options,
                    )
                    if slot_idx < n_output_slots:
                        next_output_time = self._next_scheduled_output_time(
                            simulation_time,
                            save_interval_seconds,
                            slot_idx,
                        )

                # Update progress bar
                if simulation_time.duration.seconds > 0:  # Avoid undefined progress when duration is zero
                    elapsed_time = timer.current - simulation_time.start
                    progress_percent = (elapsed_time / simulation_time.duration.seconds) * 100
                    pbar.update(progress_percent - pbar.n)  # increment by delta
                else:
                    pbar.update(0)  # avoid ZeroDivisionError if duration is 0
                pbar.set_postfix(
                    {
                        'Step': timer.step_count,
                        'Time': f'{timer.current:.0f}s',
                        'dt': f'{timer.current_timestep:.2f}s',
                    }
                )
            # End of Simulation
            pbar.close()
            self._active_progress_bar = None
            print('\nSimulation completed successfully!')
            self._report_domain_exit_summary(populations)
            self._report_beached_summary(populations, beached_ever_masks)

            if store_tracks:
                # Save final particle state if simulation ended between two save boundaries
                if last_saved_time is None or timer.current > last_saved_time:
                    if slot_idx < n_output_slots:
                        nc_handle = self.data_manager.writer.record_output(
                            nc_handle, populations, slot_idx, timer.current)
                        slot_idx += 1
                output_file = self.data_manager.writer.close_output(nc_handle)
                nc_handle = None  # prevent double-close in finally
            else:
                output_file = self.data_manager.writer.write_end_positions(
                    'sedtrails_results.nc',
                    populations,
                    float(timer.current),
                    reference_date=str(simulation_time.reference_date),
                    time_units=f'seconds since {simulation_time.reference_date}',
                    **checkpoint_options.get('writer_kwargs', {}),
                )
            self._maybe_write_checkpoint(
                populations,
                timer.current,
                simulation_time,
                slot_idx,
                checkpoint_options,
                final=True,
            )

            print(f'Simulation results saved to: {output_file}')
            self.logger.info('Total wall time: %.1fs', time.perf_counter() - wall_t0)
            self._log_profile_summary(status='completed')

            # Keep dashboard open after simulation ends
            if self.dashboard is not None:
                self.dashboard.keep_window_open()
        finally:
            if nc_handle is not None:
                try:
                    self.data_manager.writer.close_output(nc_handle)
                except Exception:
                    pass

# if __name__ == '__main__':
#     sim = Simulation(config_file='examples/config.example_natascia.yaml')
#     sim.run()

#     # NOTE: This will failed on the output saving. But that's success
