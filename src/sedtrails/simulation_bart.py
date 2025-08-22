import time
import os
import sys
import numpy as np

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

        # ------------- STEP 0: Loading configuration -----------------------------------
        # -------------------------------------------------------------------------------
        from tqdm import tqdm

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
        seeding_strategy = pop_config.get('seeding', {}).get('strategy', {})

        # ------------- STEP 1: Seeding -------------------------------------------------
        # -------------------------------------------------------------------------------

        # Particle seeding parameters
        # TODO: this should be handle by the seeding tool.
        if 'point' in seeding_strategy:
            for point in seeding_strategy['point']['locations']:
                _point = point.split(',')

                sand = Sand()
                sand.x = float(_point[0])
                sand.y = float(_point[1])
                # sand.release_time = simulation_time.start.seconds
                sand.name = 'Sand Particle'  # Set release time to start of simulation
                self.particles.append(sand)

                self.logger_manager.log_simulation_state(
                    {
                        'state': 'particles_initialized',
                        'num_particles': 1,
                        'particle_positions': {},
                        'seeding_strategy': seeding_strategy,
                    }
                )

        # Store trajectory
        trajectory_numba_x = [self.particles[0].x]  # TODO: must handle multiple particles
        trajectory_numba_y = [self.particles[0].y]

        # ------------- STEP 2: Time configuration --------------------------------------
        # -------------------------------------------------------------------------------

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

        # ------------- STEP 3: Main simulation loop ------------------------------------
        # -------------------------------------------------------------------------------

        # Get initial flow data and set up numba calculator
        sedtrails_data = self.format_converter.convert_to_sedtrails(
            current_time=timer.current,
            reading_interval=read_input_seconds
        )

        # Convert physics fields with transport probability configuration
        self.physics_converter.convert_physics(sedtrails_data, transport_prob_config)
        
        # Create FieldDataRetriever
        retriever = FieldDataRetriever(sedtrails_data)

        self.logger_manager.log_simulation_state(
            {
                'state': 'initial_data_loaded',
                'num_timesteps': len(sedtrails_data.times),
                'flow_field_name': 'suspended_velocity',
                'min_resolution': sedtrails_data.min_resolution,
                'cfl_condition': cfl_condition,
            }
        )

        # Initialize numba calculator
        flow_data = retriever.get_flow_field(simulation_time.start, 'suspended_velocity')

        self.logger_manager.log_simulation_state(
            {
                'state': 'numba_compilation_started',
                'grid_size_x': len(flow_data['x']),
                'grid_size_y': len(flow_data['y']),
            }
        )
        compile_start = time.time()
        numba_calc = create_numba_particle_calculator(grid_x=flow_data['x'], grid_y=flow_data['y'])
        compile_time = time.time() - compile_start
        self.logger_manager.log_simulation_state({'status': 'compilation_complete', 'time_sec': round(compile_time, 2)})

        # Warm up with one calculation to trigger JIT compilation
        self.logger_manager.log_simulation_state({'status': 'warming_up_jit'})
        warmup_start = time.time()
        _ = numba_calc['update_particles'](
            np.array([p.x for p in self.particles]),
            np.array([p.y for p in self.particles]),
            flow_data['u'],
            flow_data['v'],
            1, # simulation_time.time_step.seconds, # TEMP: 1 second timestep
        )
        warmup_time = time.time() - warmup_start
        self.logger_manager.log_simulation_state({'status': 'warmup_complete', 'time_sec': round(warmup_time, 2)})

        covered_dist_dict = {}
        step_count = 0
        
        # Calculate total simulation duration for progress bar
        total_duration = simulation_time.end - simulation_time.start
        
        print('Simulation loop is starting...')
        # Initialize time-based progress bar
        pbar = tqdm(total=100, desc='Computing positions', unit='%', bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}, {postfix}]')
        
        # Main simulation loop with variable timestep
        while not timer.stop:
            step_count += 1
            
            # Check if current time is within loaded SedTRAILS data (temporary memory fix)
            current_time_seconds = timer.current
            if current_time_seconds < sedtrails_data.times[0] or current_time_seconds > sedtrails_data.times[-1]:
                # Need to reload data

                # Convert to SedTRAILS format
                sedtrails_data = self.format_converter.convert_to_sedtrails(
                    current_time=current_time_seconds,
                    reading_interval=read_input_seconds
                )

                # Convert physics fields with transport probability configuration
                self.physics_converter.convert_physics(sedtrails_data, transport_prob_config)

                # Create new FieldDataRetriever with updated data
                retriever = FieldDataRetriever(sedtrails_data)

                # Pass field name to method
                plot_particle_trajectory(
                    flow_data=retriever.get_flow_field(timer.current, 'suspended_velocity'),
                    trajectory_x=trajectory_numba_x,
                    trajectory_y=trajectory_numba_y,
                    title=f'Particle Trajectory - {simulation_time.duration.seconds} seconds, {step_count} steps',
                    save_path=self.data_manager.output_dir + f'/trajectory_plot_{timer.current}.png'
                )

            # Collect flow fields for CFL computation
            flow_data_list = []
            for method in tracer_methods:
                for flow_field_name in tracer_methods[method]['flow_field_name']:
                    # Updated: Pass field name to method
                    flow_data = retriever.get_flow_field(timer.current, flow_field_name)
                    flow_data_list.append(flow_data)

            # Compute CFL-based timestep across all flow fields
            if cfl_condition > 0:
                timer.compute_cfl_timestep(flow_data_list, sedtrails_data, cfl_condition)


            # Get the scalar field data
            mixing_data = retriever.get_scalar_field(timer.current, 'mixing_layer_thickness')['magnitude']

            # Next loop over multiple methods and flow_fields as provided in config
            for method in tracer_methods:
                for flow_field_name in tracer_methods[method]['flow_field_name']:

                    # CHECK PARTICLE STATES HERE! Below example of interpolation
                    particle_x = np.array([p.x for p in self.particles])
                    particle_y = np.array([p.y for p in self.particles])
                    interpolated_values = numba_calc['interpolate_field'](mixing_data, particle_x, particle_y)
                    
                    # Updated: Pass field name to method instead of setting attribute
                    flow_data = retriever.get_flow_field(timer.current, flow_field_name)

                    # Update particle position using Numba calculator
                    new_x, new_y = numba_calc['update_particles'](
                        np.array([self.particles[0].x]),
                        np.array([self.particles[0].y]),
                        flow_data['u'],
                        flow_data['v'],
                        timer.current_timestep,
                    )

                    # Store per flow_field_name the covered distance
                    if flow_field_name not in covered_dist_dict:
                        covered_dist_dict[flow_field_name] = 0.0
                    covered_dist_dict[flow_field_name] += np.sqrt((new_x[0] - self.particles[0].x) ** 2 + (new_y[0] - self.particles[0].y) ** 2)

                    # Update particle with new position
                    self.particles[0].x = new_x[0]
                    self.particles[0].y = new_y[0]

            # Store trajectory
            trajectory_numba_x.append(self.particles[0].x)
            trajectory_numba_y.append(self.particles[0].y)

            # # Save data
            # THE DATA MANAGER MAKES THE SIMULATION SLOWER OVER TIME, THEREFORE TURNED OFF FOR NOW.
            # Other data that should be stored; 
            # - covered distance by different transport methods and/or flow fields
            # - CFL per timestep
            # - number of active particles
            # - vertical position
            # - burial depth
            # - other states?
            # self.data_manager.add_data(
            #     particle_id=self.particles[0].id,
            #     time=timer.current,
            #     x=self.particles[0].x,
            #     y=self.particles[0].y,
            # )

            # Advance timer
            timer.advance()
            
            # Calculate progress percentage based on simulation time
            elapsed_time = timer.current - simulation_time.start
            progress_percent = (elapsed_time / total_duration) * 100
            
            # Update progress bar
            pbar.n = progress_percent
            pbar.set_postfix({
                'Step': step_count,
                'Time': f'{timer.current:.0f}s',
                'dt': f'{timer.current_timestep:.2f}s',
                'Pos': f'({self.particles[0].x:.0f}, {self.particles[0].y:.0f})'
            })
            pbar.refresh()

        pbar.close()
        
        simulation_end_time = time.time()
        total_time = simulation_end_time - compile_start  # Total time including compilation

        self.logger_manager.log_simulation_state(
            {
                'status': 'simulation_complete',
                'total_steps': step_count,
                'total_time_sec': round(total_time, 2),
                'final_position': f'({self.particles[0].x:.2f}, {self.particles[0].y:.2f})',
                'cfl_condition': cfl_condition,
            }
        )

        # Finalize results
        self.data_manager.dump()  # Write remaining data to disk

        # Convert trajectory to numpy arrays
        trajectory_numba_x = np.array(trajectory_numba_x)
        trajectory_numba_y = np.array(trajectory_numba_y)

        self.logger_manager.log_simulation_state(
            {'status': 'creating_visualization', 'trajectory_points': len(trajectory_numba_x)}
        )

        # Plot flow field with particle trajectory using the function
        # Updated: Pass field name to method
        final_flow = retriever.get_flow_field(timer.current, 'suspended_velocity')

        plot_particle_trajectory(
            flow_data=final_flow,
            trajectory_x=trajectory_numba_x,
            trajectory_y=trajectory_numba_y,
            title=f'Particle Trajectory - {simulation_time.duration.seconds} seconds, {step_count} steps',
            save_path=self.data_manager.output_dir + '/trajectory_plot.png',
        )

        self.logger_manager.log_simulation_state(
            {
                'status': 'visualization_complete',
                'output_plot_path': self.data_manager.output_dir + '/trajectory_plot.png',
            }
        )


if __name__ == '__main__':
    sim = Simulation(config_file='examples/config.example_bart.yaml')
    sim.run()