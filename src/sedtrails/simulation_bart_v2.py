import time
import os
import sys
import numpy as np
from tqdm import tqdm

from sedtrails.particle_tracer.particle_seeder import PopulationConfig
from sedtrails.particle_tracer.population import ParticlePopulation
from sedtrails.transport_converter.format_converter import FormatConverter, SedtrailsData
from sedtrails.transport_converter.physics_converter import PhysicsConverter
from sedtrails.particle_tracer.data_retriever import FieldDataRetriever  # Updated import
from sedtrails.particle_tracer.particle import Particle
from sedtrails.particle_tracer.particle import Sand
from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator
from sedtrails.configuration_interface.configuration_controller import ConfigurationController
from sedtrails.data_manager import DataManager
from sedtrails.particle_tracer.timer import Time, Duration, Timer
from sedtrails.logger.logger import LoggerManager
from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.pathway_visualizer.visualization_utils import plot_particle_trajectory
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
            self._config_is_read = True
            output_dir = self._controller.get('folder_settings.output_dir', 'results')

        except Exception:
            # Global exception handler will catch and log this
            raise

        # Initialize other components
        self.format_converter = FormatConverter(self._get_format_config())
        self.physics_converter = PhysicsConverter(self._get_physics_config())
        self.data_manager = DataManager(self._get_output_dir())
        self.data_manager.set_mesh()
        self.particles: list[Particle] = []  # List to hold particles

        self.logger_manager = LoggerManager(output_dir)
        self.logger_manager.setup_logger()

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
            # TODO: These parameters are missing from json schema.
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

        # ------------- STEP 0: Loading configuration -----------------

        if not self._config_is_read:  # assure config is read only once
            self._controller.load_config(self._config_file)
            self._config_is_read = True

            self.logger_manager.log_simulation_state(
                {'state': 'config_loading', 'config_file_path': self._config_file, 'timestamp': time.time()}
            )

        # Log the command that started the simulation
        self.logger_manager.log_simulation_state(
            {
                'status': 'simulation_started',
                'command': ' '.join(sys.argv),
                'config_file': self._config_file,
                'working_directory': os.getcwd(),
                'python_version': sys.version.split()[0],
            }
        )

        # Extract population configuration once
        pop_config = self.population_config
        tracer_methods = pop_config.get('tracer_methods', {})
        transport_prob_config = pop_config.get('transport_probability', {})
        # seeding_strategy = pop_config.get('seeding', {}).get('strategy', {})

        # ------------- STEP 1: Time configuration ----------------------

        start_time = self._controller.get('time.start', None)
        if start_time is None:
            start_time = '1970-01-01 00:00:00'  # TODO: This should be the start of flow field time series

        simulation_time = Time(
            start_time,
            duration=Duration(self._controller.get('time.duration')),
            time_step=Duration(self._controller.get('time.timestep')),
            read_input_timestep=Duration(self._controller.get('time.read_input_timestep')),
        )

        read_input_seconds = simulation_time.read_input_timestep.seconds
        timer = Timer(simulation_time=simulation_time)
        
        # Get CFL condition from config
        cfl_condition = self._controller.get('time.cfl_condition', 0.7)

        # ------------- STEP 2: Seeding -------------------

        # FIXME: Link to Yaml configuration
        # FIXME: Yaml configuration does not support multiple populations (?)

        # Example usage of the ParticleFactory  to create population of particles
        config_random = PopulationConfig(
            {
                'population': {
                    'particle_type': 'sand',
                    'seeding': {
                        'strategy': {'random': {'bbox': '70950,450200,71050,450300', 'nlocations': 10, 'seed': 42}},
                        'quantity': 1,
                        'release_start': 1,
                    },
                }
            }
        )

        # Load SedTRAILS data for 'x' and 'y'
        sedtrails_data = self.format_converter.convert_to_sedtrails(
            current_time=start_time, reading_interval=1)

        populations = [ParticlePopulation(
            field_x=sedtrails_data.x,
            field_y=sedtrails_data.y,
            population_config=config_random,
        )]

        trajectory_x = []
        trajectory_y = []

        # ------------- STEP 3: Main simulation loop ---------------------

        # Calculate total simulation duration for progress bar
        step_count = 0
        sedtrails_data = None
        total_duration = simulation_time.end - simulation_time.start
        pbar = tqdm(total=100, desc='Computing positions', unit='%', bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}, {postfix}]')
        
        # Main simulation loop with variable timestep
        while not timer.stop:
            step_count += 1
            
            # Check if current time is within loaded SedTRAILS data
            current_time_seconds = timer.current
            if (sedtrails_data is None or 
                current_time_seconds < sedtrails_data.times[0] or 
                current_time_seconds > sedtrails_data.times[-2]):

                # Convert to SedTRAILS format
                sedtrails_data = self.format_converter.convert_to_sedtrails(
                    current_time=current_time_seconds, reading_interval=read_input_seconds)

                # Convert physics fields with transport probability configuration
                self.physics_converter.convert_physics(sedtrails_data, transport_prob_config)

                # Create new FieldDataRetriever with updated data
                retriever = FieldDataRetriever(sedtrails_data)


            # Collect flow fields for CFL computation
            flow_data_list = []
            for method in tracer_methods:
                for flow_field_name in tracer_methods[method]['flow_field_name']:
                    flow_data_list.append(retriever.get_flow_field(timer.current, flow_field_name))

            # Compute CFL-based timestep across all flow fields
            if cfl_condition > 0:
                timer.compute_cfl_timestep(flow_data_list, sedtrails_data, cfl_condition)

            # ============== MAIN LOOP ===============
            for population in populations:
                for method in tracer_methods:
                    for flow_field_name in tracer_methods[method]['flow_field_name']:

                        # Obtain scalar field information
                        mixing_depth = retriever.get_scalar_field(timer.current, 'mixing_layer_thickness')['magnitude']
                        bed_level = retriever.get_scalar_field(timer.current, 'bed_level')['magnitude']
                        transport_prob = retriever.get_scalar_field(timer.current, flow_field_name.replace('velocity', 'probability'))['magnitude']

                        # Update information at particle positions
                        population.update_information(
                            current_time=timer.current,
                            mixing_depth=mixing_depth,
                            bed_level=bed_level,
                            transport_probability=transport_prob
                        )
            
                        # Update particle burial depth
                        population.update_burial_depth()
            
                        # Determining status
                        population.update_status()

                        # Get flow field information
                        flow_field = retriever.get_flow_field(timer.current, flow_field_name)

                        # Update particle position
                        population.update_position(flow_field=flow_field, 
                                                   current_timestep=timer.current_timestep)

            timer.advance()

            # TODO: Save to trajectory and Netcdf here...
            # data_manager.add_positions(population)

            # Store trajectory
            trajectory_x.append(population.particles['x'].copy())
            trajectory_y.append(population.particles['y'].copy())
            
            # TEMP: Visualize particle positions (only at user interval)
            plot_interval = 864000  # seconds; adjust as needed
            if step_count == 1 or (timer.current - simulation_time.start) // plot_interval > ((timer.current - simulation_time.start - timer.current_timestep) // plot_interval):
                plot_particle_trajectory(
                    flow_data=flow_field,
                    trajectory_x=trajectory_x,
                    trajectory_y=trajectory_y,
                    title=f'Particle Trajectory - {simulation_time.duration.seconds} seconds, {step_count} steps',
                    save_path=f"{self.data_manager.output_dir}/trajectory_plot_{step_count:05d}.png",
                )

            # Calculate progress percentage based on simulation time
            elapsed_time = timer.current - simulation_time.start
            progress_percent = (elapsed_time / total_duration) * 100
            
            # Update progress bar
            pbar.n = progress_percent
            pbar.set_postfix({
                'Step': step_count,
                'Time': f'{timer.current:.0f}s',
                'dt': f'{timer.current_timestep:.2f}s',
                # 'Pos': f'({self.particles[0].x:.0f}, {self.particles[0].y:.0f})'
            })
            pbar.refresh()

        # =====================
        # End of Simulation

        pbar.close()

        self.logger_manager.log_simulation_state(
            {
                'status': 'simulation_complete',
                'total_steps': step_count,
                'total_time_sec': round(time.time(), 2),
                'final_position': f'({self.particles[0].x:.2f}, {self.particles[0].y:.2f})',
                'cfl_condition': cfl_condition,
            }
        )

        # Finalize results
        self.data_manager.dump()  # Write remaining data to disk

        self.logger_manager.log_simulation_state(
            {
                'status': 'visualization_complete',
                'output_plot_path': self.data_manager.output_dir + '/trajectory_plot.png',
            }
        )


if __name__ == '__main__':
    sim = Simulation(config_file='examples/config.example_bart.yaml')
    sim.run()