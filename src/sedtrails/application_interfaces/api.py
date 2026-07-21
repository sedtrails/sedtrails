"""
Application Interface (API)
===========================

This module provides the main Application Interface (Python API) for SedTRAILS.
It exposes functions and classes to configure and run SedTRAILS simulations
programmatically, and provides access to visualization and data analysis
functionalities.

This is the primary interface for Python users who want to use SedTRAILS
programmatically rather than through the CLI.
"""

import logging
from typing import Any, Dict, Optional

from sedtrails.application_interfaces.restart import RestartSummary, create_restart_from_netcdf

from .nc_inspector import NetCDFInspector

# ============================================================================
# Simulation Functions
# ============================================================================


def run_simulation(
    config_file: str,
    verbose: bool = False,
    enable_dashboard: Optional[bool] = None,
    report_domain_exits: bool = True,
) -> str:
    """
    Run a SedTRAILS simulation from a configuration file.

    This is the main high-level function for running simulations. It handles
    configuration loading, validation, execution, and output generation.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.
    verbose : bool, optional
        Enable verbose logging output. Default is False.
    enable_dashboard : bool, optional
        Override the dashboard setting from configuration. If None, uses config value.
    report_domain_exits : bool, optional
        Report particles that leave the model domain or beach on land during and after the run.

    Returns
    -------
    str
        Path to the output results directory.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    Examples
    --------
    >>> import sedtrails
    >>> output = sedtrails.run_simulation('config.yml')
    >>> print(f"Results saved to: {output}")

    >>> # Run with dashboard enabled regardless of config
    >>> sedtrails.run_simulation('config.yml', enable_dashboard=True)
    """
    from sedtrails.exceptions.exceptions import ConfigurationError
    from sedtrails.simulation_orchestrator.simulation_manager import Simulation

    # Set up logging if verbose
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Create simulation instance with dashboard override
    try:
        simulation = Simulation(
            config_file,
            enable_dashboard=enable_dashboard,
            report_domain_exits=report_domain_exits,
        )
    except Exception as e:
        raise ConfigurationError(f'Failed to initialize simulation: {e}') from e

    # Run the simulation
    simulation.run()

    output_dir = str(simulation.data_manager.output_dir)
    return output_dir


def create_simulation_runner(config_file: str) -> 'SimulationRunner':
    """
    Create a SimulationRunner instance for advanced control over simulation execution.

    This function provides access to a more detailed interface for users who need
    fine-grained control over the simulation process, such as step-by-step execution,
    inspection of intermediate states, or custom callbacks.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.

    Returns
    -------
    SimulationRunner
        A SimulationRunner instance configured with the given configuration file.

    Examples
    --------
    >>> import sedtrails
    >>> runner = sedtrails.create_simulation_runner('config.yml')
    >>> runner.validate()
    >>> runner.run(enable_dashboard=True)  # Override config dashboard setting
    >>> runner.save_results('output.nc')
    """
    return SimulationRunner(config_file)


# ============================================================================
# Configuration Functions
# ============================================================================


def load_configuration(config_file: str) -> Dict[str, Any]:
    """
    Load and validate a SedTRAILS configuration file.

    This function reads a YAML configuration file, validates it against the
    SedTRAILS schema, and returns the configuration as a dictionary.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.

    Returns
    -------
    dict
        The validated configuration dictionary.

    Raises
    ------
    ConfigurationError
        If the configuration file is invalid or cannot be loaded.
    FileNotFoundError
        If the configuration file does not exist.

    Examples
    --------
    >>> import sedtrails
    >>> config = sedtrails.load_configuration('config.yml')
    >>> print(config['particles']['populations'])
    """
    from sedtrails.application_interfaces.configuration_controller import ConfigurationController
    from sedtrails.exceptions.exceptions import ConfigurationError

    try:
        controller = ConfigurationController(config_file)
        controller.load_config(config_file)
        return controller.get_config()
    except Exception as e:
        raise ConfigurationError(f'Failed to load configuration: {e}') from e


def validate_configuration(config_file: str) -> bool:
    """
    Validate a SedTRAILS configuration file without returning the configuration data.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.

    Returns
    -------
    bool
        True if the configuration is valid, False otherwise.

    Examples
    --------
    >>> import sedtrails
    >>> if sedtrails.validate_configuration('config.yml'):
    ...     print("Configuration is valid!")
    ... else:
    ...     print("Configuration has errors")
    """
    try:
        load_configuration(config_file)
        return True
    except Exception:
        return False


def create_config_template(output_file: str = './sedtrails-template.yml') -> None:
    """
    Create a SedTRAILS configuration template file.

    This function generates a YAML configuration file with default values and
    documentation for all available configuration options.

    Parameters
    ----------
    output_file : str, optional
        Path where the template file will be created.
        Default is './sedtrails-template.yml'.

    Examples
    --------
    >>> import sedtrails
    >>> sedtrails.create_config_template('my_template.yml')
    """
    from sedtrails.application_interfaces.validator import YAMLConfigValidator

    validator = YAMLConfigValidator()
    validator.create_config_template(output_file)


def create_restart_config(
    results_file: str,
    base_config_file: str,
    output_config_file: str = 'sedtrails-restart.yaml',
    seed_points_dir: str | None = None,
) -> RestartSummary:
    """
    Create a restart-ready configuration from SedTRAILS NetCDF output.

    This function writes a new YAML configuration file and per-population seed
    point files using the last valid particle positions in an existing results
    file. The generated configuration can be used to continue a completed or
    interrupted simulation from the retained particle locations.

    Parameters
    ----------
    results_file : str
        Path to the SedTRAILS NetCDF results file to restart from.
    base_config_file : str
        Path to the original SedTRAILS configuration YAML file.
    output_config_file : str, optional
        Path where the restart configuration file will be written.
        Default is 'sedtrails-restart.yaml'.
    seed_points_dir : str or None, optional
        Directory where restart seed point CSV files will be written. If None,
        a directory is created next to ``output_config_file``.

    Returns
    -------
    RestartSummary
        Summary of the generated restart configuration, seed files, restart
        time, and number of retained particles.

    Raises
    ------
    FileNotFoundError
        If ``results_file`` or ``base_config_file`` does not exist.
    ValueError
        If the input files do not contain the data required to build restart
        files, or no valid particles remain for restart seeding.

    Examples
    --------
    >>> import sedtrails
    >>> summary = sedtrails.create_restart_config(
    ...     'results.nc',
    ...     'config.yml',
    ...     output_config_file='restart.yml',
    ... )
    >>> print(summary.output_config)
    """

    return create_restart_from_netcdf(
        netcdf_file=results_file,
        base_config_file=base_config_file,
        output_config_file=output_config_file,
        seed_points_dir=seed_points_dir,
    )


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_trajectories(
    results_file: str,
    output: str | None = None,
    max_particles: int | None = None,
    sample_fraction: float | None = None,
    sample_seed: int = 0,
    markers: str = 'start-end',
    marker_size: float = 12.0,
    panels: str = 'spatial',
    show: bool | None = None,
) -> None:
    """
    Plot particle trajectories from a SedTRAILS NetCDF results file.

    Parameters
    ----------
    results_file : str
        Path to the SedTRAILS NetCDF results file.
    output : str, optional
        Output target. If ``output`` is an existing directory, the plot is
        written as ``particle_trajectories.png`` inside it. Otherwise,
        ``output`` is treated as a filename. If omitted, the default output
        is ``particle_trajectories.png`` in the NetCDF file directory.
    max_particles : int, optional
        Maximum number of particles to plot. Mutually exclusive with
        ``sample_fraction``.
    sample_fraction : float, optional
        Fraction of particles to plot. Mutually exclusive with
        ``max_particles``.
    sample_seed : int, optional
        Seed used for deterministic sampling.
    markers : str, optional
        Endpoint marker mode: ``none``, ``end``, or ``start-end``.
    marker_size : float, optional
        Marker size for start/end points. Default is 12.
    panels : str, optional
        Panels to draw. Default is ``spatial`` for the fast single-panel plot.
        Use ``all`` for the previous four-panel figure, or a comma-separated
        subset of ``spatial``, ``distance``, ``population``, and
        ``population-distance``.
    show : bool, optional
        Whether to display the figure. Defaults to display-only when no output
        path is requested.

    Examples
    --------
    >>> import sedtrails
    >>> sedtrails.plot_trajectories('results.nc')

    >>> # Save in an existing folder using the default file name
    >>> sedtrails.plot_trajectories('results.nc', output='plots')

    >>> # Save to an explicit file
    >>> sedtrails.plot_trajectories('results.nc', output='plots/custom_name.png')
    """
    from sedtrails.pathway_visualizer import plot_trajectories as _plot
    from sedtrails.pathway_visualizer import read_netcdf

    ds = read_netcdf(results_file)
    try:
        _plot(
            ds,
            output=output,
            max_particles=max_particles,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
            markers=markers,
            marker_size=marker_size,
            panels=panels,
            show=show,
        )
    finally:
        ds.close()


def inspect_netcdf(results_file: str) -> 'NetCDFInspector':
    """
    Create an inspector for examining SedTRAILS NetCDF results metadata.

    Parameters
    ----------
    results_file : str
        Path to the SedTRAILS NetCDF results file.

    Returns
    -------
    NetCDFInspector
        An inspector instance that can be used to examine the file metadata.

    Examples
    --------
    >>> import sedtrails
    >>> inspector = sedtrails.inspect_netcdf('results.nc')
    >>> inspector.print_metadata()
    >>> inspector.inspect_populations()
    """
    from sedtrails.application_interfaces.nc_inspector import NetCDFInspector

    return NetCDFInspector(results_file)


# ============================================================================
# Analysis Functions (Placeholder for future implementation)
# ============================================================================


def analyze_simulation(
    input_file: str,
    output_file: str = 'analysis.nc',
) -> str:
    """
    Perform statistical analysis on simulation results.

    Note: This function is not yet fully implemented.

    Parameters
    ----------
    input_file : str
        Path to the SedTRAILS NetCDF results file.
    output_file : str, optional
        Path to save the analysis results. Default is 'analysis.nc'.

    Returns
    -------
    str
        Path to the analysis output file.

    Raises
    ------
    NotImplementedError
        This functionality is not yet implemented.
    """
    raise NotImplementedError(
        'Statistical analysis functionality is not yet fully implemented. This feature is planned for a future release.'
    )


def analyze_network(
    input_file: str,
    output_file: str = 'network_analysis.nc',
) -> str:
    """
    Perform network connectivity analysis on simulation results.

    Note: This function is not yet fully implemented.

    Parameters
    ----------
    input_file : str
        Path to the SedTRAILS NetCDF results file.
    output_file : str, optional
        Path to save the network analysis results. Default is 'network_analysis.nc'.

    Returns
    -------
    str
        Path to the network analysis output file.

    Raises
    ------
    NotImplementedError
        This functionality is not yet implemented.
    """
    raise NotImplementedError(
        'Network analysis functionality is not yet fully implemented. This feature is planned for a future release.'
    )


# ============================================================================
# Advanced API Classes
# ============================================================================


class SimulationRunner:
    """
    Advanced interface for controlling SedTRAILS simulations programmatically.

    This class provides fine-grained control over the simulation process,
    allowing users to validate, configure, and execute simulations step-by-step.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.

    Attributes
    ----------
    config_file : str
        Path to the configuration file.
    simulation : Simulation or None
        The underlying Simulation object (created after validate() is called).

    Examples
    --------
    >>> runner = SimulationRunner('config.yml')
    >>> runner.validate()
    >>> runner.run(enable_dashboard=True)
    >>> runner.save_results('output.nc')
    """

    def __init__(self, config_file: str):
        """
        Initialize the SimulationRunner.

        Parameters
        ----------
        config_file : str
            Path to the SedTRAILS configuration YAML file.
        """
        self.config_file = config_file
        self.simulation = None
        self._config = None

    def validate(self) -> bool:
        """
        Validate the configuration file.

        Returns
        -------
        bool
            True if validation succeeds.

        Raises
        ------
        ConfigurationError
            If validation fails.
        """
        from sedtrails.simulation_orchestrator.simulation_manager import Simulation

        try:
            # Create simulation without dashboard for validation
            self.simulation = Simulation(self.config_file, enable_dashboard=False)
            self.simulation.validate_config()
            return True
        except Exception as e:
            from sedtrails.exceptions.exceptions import ConfigurationError

            raise ConfigurationError(f'Validation failed: {e}') from e

    def run(self, enable_dashboard: Optional[bool] = None, verbose: bool = False) -> None:
        """
        Run the simulation.

        Parameters
        ----------
        enable_dashboard : bool, optional
            Override the dashboard setting from the configuration.
            If None, uses the configuration value.
        verbose : bool, optional
            Enable verbose logging output. Default is False.

        Raises
        ------
        RuntimeError
            If validate() has not been called first.
        """
        if self.simulation is None:
            raise RuntimeError('Must call validate() before run()')

        if verbose:
            logging.basicConfig(level=logging.INFO)

        # Create new simulation instance with dashboard override
        from sedtrails.simulation_orchestrator.simulation_manager import Simulation

        self.simulation = Simulation(self.config_file, enable_dashboard=enable_dashboard)
        self.simulation.run()

    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration.

        Returns
        -------
        dict
            The configuration dictionary.

        Raises
        ------
        RuntimeError
            If validate() has not been called first.
        """
        if self.simulation is None:
            raise RuntimeError('Must call validate() before get_config()')

        return self.simulation.config

    def save_results(self, output_file: str) -> str:
        """
        Save simulation results to a file.

        Note: This method is a placeholder. The current Simulation class
        handles output automatically during run().

        Parameters
        ----------
        output_file : str
            Path to save the results.

        Returns
        -------
        str
            Path to the saved file.
        """
        # TODO: This will need to be implemented when Simulation class
        # supports separate save functionality
        logging.info(f'Results will be saved to {output_file}')
        logging.warning('save_results() is not fully implemented yet')
        return output_file


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Simulation
    'run_simulation',
    'create_simulation_runner',
    'SimulationRunner',
    # Configuration
    'load_configuration',
    'validate_configuration',
    'create_config_template',
    'create_restart_config',
    # Visualization
    'plot_trajectories',
    'inspect_netcdf',
    # Analysis (not yet implemented)
    'analyze_simulation',
    'analyze_network',
]
