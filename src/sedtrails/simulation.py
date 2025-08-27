import os
import sys
from tqdm import tqdm

from sedtrails.particle_tracer.particle_seeder import PopulationConfig
from sedtrails.particle_tracer.population import ParticlePopulation
from sedtrails.transport_converter.format_converter import FormatConverter, SedtrailsData
from sedtrails.transport_converter.physics_converter import PhysicsConverter
from sedtrails.particle_tracer.data_retriever import FieldDataRetriever  # Updated import
from sedtrails.particle_tracer.particle import Particle
from sedtrails.configuration_interface.configuration_controller import ConfigurationController
from sedtrails.data_manager import DataManager
from sedtrails.particle_tracer.timer import Time, Duration, Timer
from sedtrails.logger.logger import LoggerManager
from sedtrails.exceptions.exceptions import ConfigurationError
from typing import Any


def setup_global_exception_logging(logger_manager):
    """Setup global exception logging for unhandled exceptions."""
    original_excepthook = sys.excepthook

    def exception_handler(exc_type, exc_value, exc_traceback):
        # Don't log KeyboardInterrupt (Ctrl+C)
        if issubclass(exc_type, KeyboardInterrupt):
            original_excepthook(exc_type, exc_value, exc_traceback)
            return

        # Log all other exceptions
        logger_manager.log_exception(exc_value, 'Global Exception Handler')
        logger_manager.log_simulation_state(
            {'status': 'simulation_failed', 'error_type': exc_type.__name__, 'error_message': str(exc_value)}
        )

        # Call original handler
        original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = exception_handler


class Simulation:
    """Class to encapsulate the particle simulation process."""

    def __init__(self, config_file: str):
        """
        Initialize the simulation with the given configuration.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.
        """
        self._config_file = config_file

        self._start_time = None
        self._config_is_read = False
        self._population_config = None

        # Validate config file exists early
        if not os.path.exists(config_file):
            raise ConfigurationError(f'Configuration file not found: {config_file}')

        # Try to read config and update logger directory
        try:
            self._controller = ConfigurationController(self._config_file)
            self._controller.load_config(self._config_file)
            output_dir = self._controller.get('folder_settings.output_dir', 'results')

            self.logger_manager = LoggerManager(output_dir)
            self._controller = ConfigurationController(self._config_file, self.logger_manager)
            self.logger_manager.setup_logger()
            self._controller.log_after_load_config()

            self._config_is_read = True

        except Exception:
            # Global exception handler will catch and log this
            raise

        # Initialize other components
        self.format_converter = FormatConverter(self._get_format_config())
        self.physics_converter = PhysicsConverter(self._get_physics_config())
        self.data_manager = DataManager(self._get_output_dir())
        self.data_manager.set_mesh()
        self.particles: list[Particle] = []  # List to hold particles

        # Setup global exception handling
        setup_global_exception_logging(self.logger_manager)

    def _get_format_config(self):
        """
        Returns configuration parameters required for the format converter.
        """

        format_config = {
            'input_file': self._controller.get('folder_settings.input_data'),
            'input_format': self._controller.get('general.input_model.format'),  # Specify the input format
            'reference_date': self._controller.get('general.input_model.reference_date'),
            'morfac': self._controller.get('general.input_model.morfac', 1.0),
        }

        return format_config

    def _get_output_dir(self):
        """
        Returns the output directory for the simulation.
        """
        return self._controller.get('folder_settings.output_dir')

    def _get_physics_config(self):
        """
        Returns configuration parameters required for the physics converter.
        """
        from sedtrails.transport_converter.physics_converter import PhysicsConfig

        config = PhysicsConfig(
            tracer_method=self._controller.get('physics.tracer_method', 'van_westen'),
            gravity=self._controller.get('physics.constants.g', 9.81),
            von_karman_constant=self._controller.get('physics.constants.von_karman', 0.40),
            kinematic_viscosity=self._controller.get('physics.constants.kinematic_viscosity', 1.36e-6),
            water_density=self._controller.get('physics.constants.rho_w', 1027.0),
            particle_density=self._controller.get('physics.constants.rho_s', 2650.0),
            porosity=self._controller.get('physics.porosity', 0.4),
            grain_diameter=self._controller.get('physics.grain_diameter', 2.5e-4),
            morfac=self._controller.get('physics.morfac', 1.0),
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
    def population_config(self):
        """
        Returns the particle population configuration.
        """
        if self._population_config is None:
            self._population_config = self.config.get('particles', {}).get('population', {})
        return self._population_config

    @property
    def start_time(self):
        """
        Get the start time for the simulation.
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

    def get_parameter(self, key: str) -> Any:
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

        if not self._config_is_read:
            self._controller.load_config(self._config_file)

        value = self._controller.get(key, None)
        if value is None:
            warnings.warn(f'Key "{key}" not found in configuration file', UserWarning, stacklevel=2)
        return value

    def run(self):
        """
        Execute the complete particle simulation workflow.
        """

        # Loading configuration
        if not self._config_is_read:  # assure config is read only once
            self._controller.load_config(self._config_file)
            self._config_is_read = True

        # Time configuration
        simulation_time = Time(
            _start=self._controller.get('time.start'),
            duration=Duration(self._controller.get('time.duration')),
            time_step=Duration(self._controller.get('time.timestep')),
            read_input_timestep=Duration(self._controller.get('time.read_input_timestep')),
        )

        timer = Timer(simulation_time=simulation_time, cfl_condition=self._controller.get('time.cfl_condition'))

        # Load SedTRAILS data for 'x' and 'y' (needed for population seerder)
        sedtrails_data = self.format_converter.convert_to_sedtrails(
            current_time=simulation_time._start, reading_interval=1
        )

        populations = []  # TODO: integrate into ParticlePopulation class
        populations_config = self._controller.get('particles.populations')
        for population_config in populations_config:
            p_config = PopulationConfig(population_config)

            populations.append(
                ParticlePopulation(
                    field_x=sedtrails_data.x,
                    field_y=sedtrails_data.y,
                    population_config=p_config,
                )
            )

        # Set initial values
        sedtrails_data = None

        # Initialize progress bar
        pbar = tqdm(
            total=100,
            desc='Computing positions',
            unit='%',
            bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}, {postfix}]',
        )

        # Main simulation loop with variable timestep
        while not timer.stop:
            # Check if current time is within loaded SedTRAILS data
            current_time_seconds = timer.current
            if (
                sedtrails_data is None
                or current_time_seconds < sedtrails_data.times[0]
                or current_time_seconds > sedtrails_data.times[-2]
            ):
                # Convert to SedTRAILS format
                sedtrails_data = self.format_converter.convert_to_sedtrails(
                    current_time=current_time_seconds, reading_interval=simulation_time.read_input_timestep.seconds
                )

                # Convert physics fields with transport probability configuration
                self.physics_converter.convert_physics(
                    sedtrails_data=sedtrails_data,
                    transport_probability_method=self.population_config.get('transport_probability'),
                )

                # Create new FieldDataRetriever with updated data
                retriever = FieldDataRetriever(sedtrails_data)

            # TODO: integrate loop over flow fields into CFL Condition
            # Collect flow fields for CFL computation
            for population_config in populations_config:
                tracer_methods = population_config['tracer_methods']
                flow_field_names = population_config['tracer_methods']['vanwesten']['flow_field_name']

            flow_data_list = []
            for flow_field_name in flow_field_names:
                flow_data_list.append(retriever.get_flow_field(timer.current, flow_field_name))

            # Compute CFL-based timestep across all flow fields
            timer.compute_cfl_timestep(flow_data_list, sedtrails_data)

            # Main loop
            for population in populations:
                for method in tracer_methods:
                    for flow_field_name in flow_field_names:
                        # Obtain scalar field information
                        mixing_depth = retriever.get_scalar_field(timer.current, 'mixing_layer_thickness')['magnitude']
                        bed_level = retriever.get_scalar_field(timer.current, 'bed_level')['magnitude']
                        transport_prob = retriever.get_scalar_field(
                            timer.current, flow_field_name.replace('velocity', 'probability')
                        )['magnitude']

                        # Update information at particle positions
                        population.update_information(
                            current_time=timer.current,
                            mixing_depth=mixing_depth,
                            bed_level=bed_level,
                            transport_probability=transport_prob,
                        )

                        # Update particle burial depth
                        population.update_burial_depth()

                        # Determining status
                        population.update_status()

                        # Get flow field information
                        flow_field = retriever.get_flow_field(timer.current, flow_field_name)

                        # Update particle position
                        population.update_position(flow_field=flow_field, current_timestep=timer.current_timestep)

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
            elapsed_time = timer.current - simulation_time.start
            progress_percent = (elapsed_time / simulation_time.duration.seconds) * 100

            # Update progress bar
            pbar.n = progress_percent
            pbar.set_postfix(
                {
                    'Step': timer.step_count,
                    'Time': f'{timer.current:.0f}s',
                    'dt': f'{timer.current_timestep:.2f}s',
                }
            )
            pbar.refresh()

        # End of Simulation
        pbar.close()

        # Finalize results
        self.data_manager.dump()  # Write remaining data to disk


if __name__ == '__main__':
    sim = Simulation(config_file='examples/config.example.yaml')
    sim.run()

    # NOTE: This will failed on the output saving. But that's success
