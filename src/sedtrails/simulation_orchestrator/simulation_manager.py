import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xarray as xr
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
from sedtrails.transport_converter.format_converter import FormatConverter, SedtrailsData
from sedtrails.transport_converter.physics_converter import PhysicsConverter


class Simulation:
    """Class to encapsulate the particle simulation process."""

    _DASHBOARD_FULL_GRID_CELL_LIMIT = 100_000
    _DASHBOARD_LARGE_GRID_UPDATE_STRIDE = 10

    def __init__(self, config_file: str, enable_dashboard: Optional[bool] = None):
        """
        Initialize the simulation with the given configuration.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.
        enable_dashboard : bool, optional
            Override the dashboard setting from configuration. If None, uses config value.
        """
        self._config_file = config_file
        self._enable_dashboard_override = enable_dashboard

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

            # TODO: logger has a circular dependency with controller. The logger needs refactoring.
            self.logger = logging.getLogger(__name__)

            self._config_is_read = True

            # self.logger_manager.setup_logger()
            # self._controller.log_after_load_config()

        except Exception:
            # Global exception handler will catch and log this
            raise

        for population in self.populations_config:  # FIXME: this just keeps the last one
            tracer_config = population['tracer_methods']
        # Initialize other components
        self.format_converter = FormatConverter(self._get_format_config())
        # Added population_config for the soulsby method
        self.physics_converter = PhysicsConverter(self._get_physics_config(), tracer_config)
        self.data_manager = DataManager(self._get_output_dir())
        self.data_manager.set_mesh()  # TODO: was this ever answered? is it needed?
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
        for name, stats in sorted(self._profile_timings.items(), key=lambda item: item[1]['total'], reverse=True):
            count = stats['count']
            total = stats['total']
            average = total / count if count else 0.0
            self.logger.info(
                'profile %-42s count=%6d total=%10.6fs avg=%10.6fs max=%10.6fs',
                name,
                count,
                total,
                average,
                stats['max'],
            )

    def _create_dashboard(self):
        """Create and return a dashboard instance."""
        # Use override if provided, otherwise fall back to config setting
        dashboard_enabled = (
            self._enable_dashboard_override
            if self._enable_dashboard_override is not None
            else self._controller.get('visualization.dashboard.enable', False)
        )

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

    def _get_format_config(self):
        """
        Returns configuration parameters required for the format converter.
        """

        format_config = {
            'input_file': self._controller.get('inputs.data'),
            'input_format': self._controller.get('general.input_model.format'),  # Specify the input format
            'reference_date': self._controller.get('general.input_model.reference_date'),
            'morfac': self._controller.get('general.input_model.morfac', 1.0),
        }

        return format_config

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

        return output_dir

    def _get_physics_config(self):
        """
        Returns configuration parameters required for the physics converter.
        """
        from sedtrails.transport_converter.physics_converter import PhysicsConfig

        # TODO implement configuration controller for multiple populations in a robust way
        # TODO make sure the tracer_methods in config file match exactly with the plugin file name
        tracer_method = self._controller.get('particles.populations')[0].get('tracer_methods', 'vanwesten')
        if isinstance(tracer_method, dict):
            tracer_method = list(tracer_method.keys())[0]
        config = PhysicsConfig(
            tracer_method=tracer_method,
            gravity=self._controller.get('physics.constants.g', 9.81),
            von_karman_constant=self._controller.get('physics.constants.von_karman', 0.40),
            kinematic_viscosity=self._controller.get('physics.constants.kinematic_viscosity', 1.36e-6),
            water_density=self._controller.get('physics.constants.rho_w', 1027.0),
            particle_density=self._controller.get('physics.constants.rho_s', 2650.0),
            porosity=self._controller.get('physics.constants.porosity', 0.4),
            grain_diameter=self._controller.get('physics.constants.grain_diameter', 2.5e-4),
            morfac=self._controller.get('physics.constants.morphology_factor', 1.0),
            # trapped_exposed_method=self._controller.get('physics.trapped_exposed_method', 'reduced_velocity'), # other option; 'probabilistic_exposure'
        )

        return config

    @property
    def config(self):
        """
        Returns the full configuration settings for the simulation.
        """
        if not self._config_is_read:
            self._controller.load_config(self._config_file)
        return self._controller.get_config()  # delagates to the controller

    @property
    def populations_config(self):
        """
        Returns the configuration paramters for 'populations'.
        """
        if self._populations_config is None:
            self._populations_config = self.config.get('particles', {}).get('populations', {})
        return self._populations_config

    @property
    def start_time(self):
        """
        Get the start time parameter for the simulation.
        """
        if not self._start_time:
            self._start_time = self._controller.get('time.start_time')  # defaults to Unix epoch
        return self._start_time

    @property
    def flow_field(self) -> SedtrailsData:
        """
        Returns input flow field data in SedtrailsData format.
        """

        return self.format_converter.convert_to_sedtrails()

    def validate_config(self) -> bool:
        """
        Validates the configuration file.

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
        """Execute the simulation and flush profile timings on failure."""
        try:
            return self._run_impl()
        except Exception:
            if self._active_progress_bar is not None:
                self._active_progress_bar.close()
                self._active_progress_bar = None
            self._log_profile_summary(status='interrupted')
            raise

    def _run_impl(self):
        """
        Executes the particle simulation workflow.
        """

        # Loading configuration
        if not self._config_is_read:  # assure config is read only once
            self._controller.load_config(self._config_file)
            self._config_is_read = True

        # Time configuration
        simulation_time = self._create_simulation_time()

        timer = Timer(simulation_time=simulation_time, cfl_condition=self._controller.get('time.cfl_condition'))

        # Load only x/y field coordinates needed for the population seeder.
        with self._profile_section('get_seeding_field_data'):
            seeding_field_data = self.format_converter.get_seeding_field_data()

        populations_config = self._controller.get('particles.populations', [])
        seeder = ParticleSeeder(populations_config)  # intialize seeder with population config
        populations = seeder.seed(seeding_field_data)  # seed particles for all populations

        # Set initial values
        sedtrails_data = None

        # Initialize progress bar
        pbar = tqdm(
            total=100,
            desc='Computing positions',
            unit='%',
            bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}, {postfix}]',
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

        # Determine flow field names from configuration
        # tracer_methods is obligatory, so all these checks are not needed.
        flow_field_names = []
        for population in populations_config:
            if 'tracer_methods' in population:
                if 'vanwesten' in population['tracer_methods']:
                    flow_field_names = population['tracer_methods']['vanwesten']['flow_field_name']
                    break
                elif 'soulsby' in population['tracer_methods']:
                    flow_field_names = population['tracer_methods']['soulsby']['flow_field_name']
                    break
                elif 'macdonald' in population['tracer_methods']:
                    # MacDonald method uses both bed load and suspended velocities
                    flow_field_names = population['tracer_methods']['macdonald']['flow_field_name']
                    break
                elif 'passive_tracer' in population['tracer_methods']:
                    flow_field_names = population['tracer_methods']['passive_tracer']['flow_field_name']
                break  # Use the first population's flow fields for now

        # Create SedTrails dataset using DataManager's writer (composition)
        total_particles = sum([len(pop.particles['x']) for pop in populations])
        estimated_timesteps = (simulation_time.duration.seconds // simulation_time.time_step.seconds) + 1
        max_timesteps = estimated_timesteps * 2  # Initial buffer

        xr_data = self.data_manager.writer.create_dataset(
            N_particles=total_particles,
            N_populations=len(populations),
            N_timesteps=max_timesteps,
            N_flowfields=len(flow_field_names) if flow_field_names else 1,
        )

        # Initialize metadata using DataManager's writer (composition)
        self.data_manager.writer.add_metadata(xr_data, populations, flow_field_names)

        # Main simulation loop with variable timestep
        input_data_exhausted = False
        input_exhaustion_warning_logged = False
        while not timer.stop:
            # Check if current time is within loaded SedTRAILS data
            current_time_seconds = timer.current
            if self._should_attempt_sedtrails_reload(sedtrails_data, current_time_seconds, input_data_exhausted):
                # Avoid recreating SedTRAILS data if current time is before the first time step
                if sedtrails_data is not None and current_time_seconds < sedtrails_data.times[0]:
                    timer.advance()
                    continue
                # Convert to SedTRAILS format
                with self._profile_section('convert_to_sedtrails'):
                    sedtrails_data = self.format_converter.convert_to_sedtrails(
                        current_time=current_time_seconds, reading_interval=simulation_time.read_input_interval.seconds
                    )

                # Convert physics fields with transport probability configuration
                for pop in self.populations_config:
                    with self._profile_section('convert_physics'):
                        self.physics_converter.convert_physics(
                            sedtrails_data=sedtrails_data,
                            transport_probability_method=pop.get('transport_probability'),
                        )

                # Create new FieldDataRetriever with updated data
                retriever = FieldDataRetriever(sedtrails_data)  # TODO: should the retriever only be created once?

                if self._is_after_loaded_sedtrails_data(sedtrails_data, current_time_seconds):
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
            tracer_methods = {}
            # TODO: this loops over populations_config, but only the last one is used. This must be fixed
            # to handle multiple populations with different tracer methods and flow fields
            for ip, population in enumerate(populations_config):
                tracer_cfg = population.get('tracer_methods', {})
                if isinstance(tracer_cfg, dict) and tracer_cfg:
                    tracer_methods[ip] = next(iter(tracer_cfg.keys()))
                elif isinstance(tracer_cfg, str):
                    tracer_methods[ip] = tracer_cfg
                else:
                    tracer_methods[ip] = 'passive_tracer'

            field_cache = {}
            cache_time = timer.current

            def get_flow_field_cached(
                flow_field_name: str,
                profile_name: str,
                _cache=field_cache,
                _retriever=retriever,
                _time=cache_time,
            ):
                cache_key = ('flow', _time, flow_field_name)
                if cache_key not in _cache:
                    with self._profile_section(profile_name):
                        _cache[cache_key] = _retriever.get_flow_field(_time, flow_field_name)
                return _cache[cache_key]

            def get_flow_field_bounds_cached(
                flow_field_name: str,
                profile_name: str,
                _cache=field_cache,
                _retriever=retriever,
                _time=cache_time,
            ):
                cache_key = ('flow_bounds', _time, flow_field_name)
                if cache_key not in _cache:
                    with self._profile_section(profile_name):
                        _cache[cache_key] = _retriever.get_flow_field_bounds(_time, flow_field_name)
                return _cache[cache_key]

            def get_flow_max_velocity_cached(
                flow_field_name: str,
                profile_name: str,
                _cache=field_cache,
                _retriever=retriever,
                _time=cache_time,
            ):
                cache_key = ('flow_max_velocity', _time, flow_field_name)
                if cache_key not in _cache:
                    with self._profile_section(profile_name):
                        _cache[cache_key] = _retriever.get_flow_max_velocity_bound(_time, flow_field_name)
                return _cache[cache_key]

            def get_scalar_field_cached(
                scalar_field_name: str,
                profile_name: str,
                _cache=field_cache,
                _retriever=retriever,
                _time=cache_time,
            ):
                cache_key = ('scalar', _time, scalar_field_name)
                if cache_key not in _cache:
                    with self._profile_section(profile_name):
                        _cache[cache_key] = _retriever.get_scalar_field(_time, scalar_field_name)['magnitude']
                return _cache[cache_key]

            def get_scalar_field_bounds_cached(
                scalar_field_name: str,
                profile_name: str,
                _cache=field_cache,
                _retriever=retriever,
                _time=cache_time,
            ):
                cache_key = ('scalar_bounds', _time, scalar_field_name)
                if cache_key not in _cache:
                    with self._profile_section(profile_name):
                        _cache[cache_key] = _retriever.get_scalar_field_bounds(_time, scalar_field_name)
                return _cache[cache_key]

            max_velocity = 0.0
            for flow_field_name in flow_field_names:
                max_velocity = max(
                    max_velocity,
                    get_flow_max_velocity_cached(flow_field_name, 'get_flow_max_velocity.cfl'),
                )

            # Compute CFL-based timestep across all flow fields
            with self._profile_section('compute_cfl_timestep'):
                timer.compute_cfl_timestep_from_max_velocity(
                    max_velocity,
                    sedtrails_data.metadata.min_resolution,
                    sedtrails_data.metadata.timestep,
                )

            # Main loop
            for ip, population in enumerate(populations):
                if ip not in tracer_methods:
                    raise ConfigurationError(
                        f'Tracer method not configured for population {ip}. '
                        f'Check that all populations have "tracer_methods" defined in configuration.'
                    )
                tracer_method = tracer_methods[ip]
                for flow_field_name in flow_field_names:
                    # Obtain scalar field information
                    # TODO: Consider moving van westen specific fields to the plugin itself
                    bed_level = get_scalar_field_bounds_cached('bed_level', 'get_scalar_field.bed_level')
                    mixing_depth = None
                    if tracer_method == 'vanwesten':
                        transport_prob = get_scalar_field_bounds_cached(
                            flow_field_name.replace('velocity', 'probability'),
                            'get_scalar_field.transport_probability',
                        )
                        mixing_depth = get_scalar_field_bounds_cached(
                            'mixing_layer_thickness',
                            'get_scalar_field.mixing_layer_thickness',
                        )
                    elif tracer_method == 'soulsby':
                        transport_prob = 1.0
                        mixing_depth = get_scalar_field_bounds_cached(
                            'mixing_layer_thickness',
                            'get_scalar_field.mixing_layer_thickness',
                        )
                    elif tracer_method == 'macdonald':
                        transport_prob = get_scalar_field_bounds_cached(
                            'mean_particle_probability',
                            'get_scalar_field.transport_probability',
                        )
                        mixing_depth = float('inf')
                    else:
                        transport_prob = 1.0

                    # Update information at particle positions
                    with self._profile_section('update_information'):
                        population.update_information(
                            current_time=timer.current,
                            mixing_depth=mixing_depth,
                            bed_level=bed_level,
                            transport_probability=transport_prob,
                        )

                    # Update particle burial depth
                    if tracer_method == 'vanwesten':
                        population.update_burial_depth()

                    # Determining status
                    population.update_status()

                    is_macdonald_q3d = (
                        tracer_method == 'macdonald'
                        and str(getattr(self.physics_converter.config, 'computationType', '2D')).upper() == 'Q3D'
                    )

                    # Get flow field information
                    flow_field = get_flow_field_bounds_cached(flow_field_name, 'get_flow_field.update_position')

                    if is_macdonald_q3d:
                        hydrodynamic_flow_field = get_flow_field_bounds_cached(
                            'depth_avg_flow_velocity',
                            'get_flow_field.depth_avg_flow_velocity',
                        )
                        q3d_entrainment_mode = getattr(
                            self.physics_converter.config, 'q3d_entrainment_mode', 'shields_threshold'
                        )
                        q3d_entrainment_frequency = None
                        if str(q3d_entrainment_mode).lower().replace('-', '_') in {
                            'entrainment_frequency',
                            'frequency',
                        }:
                            q3d_entrainment_frequency = get_scalar_field_bounds_cached(
                                'q3d_entrainment_frequency',
                                'get_scalar_field.q3d_entrainment_frequency',
                            )
                        with self._profile_section('update_q3d_particle_motion'):
                            population.update_q3d_particle_motion(
                                current_timestep=timer.current_timestep,
                                centroid_flow_field=flow_field,
                                hydrodynamic_flow_field=hydrodynamic_flow_field,
                                bed_level_field=bed_level,
                                max_shear_velocity=get_scalar_field_bounds_cached(
                                    'max_shear_velocity',
                                    'get_scalar_field.max_shear_velocity',
                                ),
                                total_roughness_height=get_scalar_field_bounds_cached(
                                    'total_roughness_height',
                                    'get_scalar_field.total_roughness_height',
                                ),
                                total_transport_centroid_elevation=get_scalar_field_bounds_cached(
                                    'total_transport_centroid_elevation',
                                    'get_scalar_field.total_transport_centroid_elevation',
                                ),
                                q3d_velocity_deficit_coefficient=get_scalar_field_bounds_cached(
                                    'q3d_velocity_deficit_coefficient',
                                    'get_scalar_field.q3d_velocity_deficit_coefficient',
                                ),
                                q3d_vertical_velocity_gradient=get_scalar_field_bounds_cached(
                                    'q3d_vertical_velocity_gradient',
                                    'get_scalar_field.q3d_vertical_velocity_gradient',
                                ),
                                turbulent_shields_number=get_scalar_field_bounds_cached(
                                    'turbulent_shields_number',
                                    'get_scalar_field.turbulent_shields_number',
                                ),
                                critical_shields_number=self.physics_converter.grain_properties.get(
                                    'critical_shields'
                                ),
                                settling_velocity=self.physics_converter.grain_properties.get('settling_velocity'),
                                water_depth=get_scalar_field_bounds_cached(
                                    'water_depth',
                                    'get_scalar_field.water_depth',
                                ),
                                skin_roughness_height=get_scalar_field_bounds_cached(
                                    'skin_roughness_height',
                                    'get_scalar_field.skin_roughness_height',
                                ),
                                entrainment_height_above_bed=get_scalar_field_bounds_cached(
                                    'q3d_entrainment_height_above_bed',
                                    'get_scalar_field.q3d_entrainment_height_above_bed',
                                ),
                                rouse_number=get_scalar_field_bounds_cached(
                                    'rouse_number',
                                    'get_scalar_field.rouse_number',
                                ),
                                K_Et=getattr(
                                    self.physics_converter.config, 'q3d_horizontal_diffusion_factor', 0.15
                                ),
                                q3d_entrainment_mode=q3d_entrainment_mode,
                                q3d_entrainment_frequency=q3d_entrainment_frequency,
                                q3d_vertical_update_scheme=getattr(
                                    self.physics_converter.config, 'q3d_vertical_update_scheme', 'geometric'
                                ),
                                q3d_motion_substeps=getattr(
                                    self.physics_converter.config, 'q3d_motion_substeps', 1
                                ),
                            )
                    else:
                        # Update particle position
                        with self._profile_section('update_position'):
                            population.update_position(flow_field=flow_field, current_timestep=timer.current_timestep)

            # Collect data from all populations for this timestep using DataManager
            self.data_manager.collect_timestep_data(xr_data, populations, timer.step_count, timer.current)

            # Check if we need to expand the time dimension
            if timer.step_count >= max_timesteps - 10:  # 10-step safety margin
                old_max = max_timesteps
                max_timesteps = int(max_timesteps * 2.0)  # Expand by 100%
                self.logger.info(f'Expanding time dimension from {old_max} to {max_timesteps}')
                xr_data = self._expand_time_dimension(xr_data, max_timesteps)

            # Update dashboard if enabled
            if self._should_update_dashboard(sedtrails_data, timer):
                plot_interval_seconds = self._dashboard_update_interval_seconds()
                if self.dashboard.should_update(timer.current, plot_interval_seconds):
                    # For dashboard, use first population data
                    first_population = populations[0]
                    particle_data = self._dashboard_particle_data(first_population)
                    bathymetry = get_scalar_field_cached('bed_level', 'get_scalar_field.dashboard_bed_level')
                    dashboard_flow_field = get_flow_field_cached(flow_field_names[0], 'get_flow_field.dashboard')
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

            # Saving and plotting
            # TODO: enable saving and plotting again: addapt writer with structure issue 297
            # TODO: remove default insertion on configuration retrieval
            # interval_output = self._controller.get('output.interval_output', '1H')
            # interval_plot = self._controller.get('output.interval_plot', '1D')

            # # Data manager
            # if timer.step_count == 1 or (timer.current - simulation_time.start) // interval_output > (
            #     (timer.current - simulation_time.start - timer.current_timestep) // interval_output
            # ):
            #     # self.data_manager.add_data()

            # # Plotting
            # TODO: enable plotting from saved data file and from memory.
            # if timer.step_count == 1 or (timer.current - simulation_time.start) // interval_plot > (
            #     (timer.current - simulation_time.start - timer.current_timestep) // interval_plot
            # ):
            #     plot_particle_trajectory(
            #         flow_data=flow_field,
            #         trajectory_x=self.data_manager.get_trajectory_x(),
            #         trajectory_y=self.data_manager.get_trajectory_y(),
            #         title=f'Particle Trajectory - {simulation_time.duration.seconds} seconds, {timer.step_count} steps',
            #         save_path=f'{self.data_manager.output_dir}/trajectory_plot_{timer.step_count:05d}.png',
            #     )

            # Calculate progress percentage based on simulation time
            # elapsed_time = timer.current - simulation_time.start
            # progress_percent = (elapsed_time / simulation_time.duration.seconds) * 100

            # # Update progress bar
            # pbar.n = progress_percent
            # pbar.set_postfix(
            #     {
            #         'Step': timer.step_count,
            #         'Time': f'{timer.current:.0f}s',
            #         'dt': f'{timer.current_timestep:.2f}s',
            #     }
            # )
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

        # Write final results to NetCDF using DataManager's writer (composition)
        actual_timesteps = timer.step_count + 1
        output_file = self.data_manager.writer.write(
            xr_data, filename='sedtrails_results.nc', trim_to_actual_timesteps=True, actual_timesteps=actual_timesteps
        )
        print(f'Simulation results saved to: {output_file}')
        self._log_profile_summary(status='completed')

        # Keep dashboard open after simulation ends
        if self.dashboard is not None:
            self.dashboard.keep_window_open()

        # Finalize results
        # self.data_manager.dump()  # Write remaining data to disk. # TODO: not working

    def _expand_time_dimension(
        self,
        xr_data: xr.Dataset,
        new_max_timesteps: int,
    ) -> xr.Dataset:
        time_dim = 'n_timesteps' if 'n_timesteps' in xr_data.sizes else 'time'

        current_size = xr_data.sizes[time_dim]

        if new_max_timesteps <= current_size:
            raise ValueError(f'new_max_timesteps={new_max_timesteps} must be larger than current size={current_size}')

        # Replace/normalize the timestep coordinate to guarantee uniqueness
        xr_data = xr_data.assign_coords({time_dim: np.arange(current_size)})

        new_coord = np.arange(new_max_timesteps)

        expanded_vars = {}

        for var_name, var in xr_data.data_vars.items():
            if time_dim not in var.dims:
                expanded_vars[var_name] = var
                continue

            time_axis = var.dims.index(time_dim)

            pad_shape = list(var.shape)
            pad_shape[time_axis] = new_max_timesteps - current_size

            if np.issubdtype(var.dtype, np.floating) or np.issubdtype(var.dtype, np.complexfloating):
                pad_data = np.full(
                    pad_shape,
                    np.nan,
                    dtype=var.dtype,
                )
            else:
                pad_data = np.zeros(
                    pad_shape,
                    dtype=var.dtype,
                )

            pad_coords = {}
            for dim in var.dims:
                if dim == time_dim:
                    pad_coords[dim] = np.arange(current_size, new_max_timesteps)
                elif dim in var.coords:
                    pad_coords[dim] = var.coords[dim]

            pad_array = xr.DataArray(
                pad_data,
                dims=var.dims,
                coords=pad_coords,
                attrs=var.attrs.copy(),
            )

            expanded_vars[var_name] = xr.concat(
                [var, pad_array],
                dim=time_dim,
            )

        coords = {}
        for coord_name, coord in xr_data.coords.items():
            if coord_name == time_dim:
                coords[coord_name] = new_coord
            elif time_dim not in coord.dims:
                coords[coord_name] = coord

        expanded_dataset = xr.Dataset(
            expanded_vars,
            coords=coords,
            attrs=xr_data.attrs.copy(),
        )

        return expanded_dataset


# if __name__ == '__main__':
#     sim = Simulation(config_file='examples/config.example_natascia.yaml')
#     sim.run()

#     # NOTE: This will failed on the output saving. But that's success
