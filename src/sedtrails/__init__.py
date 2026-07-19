"""
SedTRAILS: Sediment Transport and Lagrangian Simulator
======================================================

SedTRAILS is a particle tracking system for coastal environments that simulates
sediment transport using Lagrangian methods.

Main API Functions
------------------
Simulation:
    - run_simulation: Run a simulation from a configuration file
    - create_simulation_runner: Create an advanced SimulationRunner instance

Configuration:
    - load_configuration: Load and validate a configuration file
    - validate_configuration: Validate a configuration file
    - create_config_template: Create a configuration template

Visualization:
    - plot_trajectories: Plot particle trajectories from results
    - inspect_netcdf: Inspect NetCDF file metadata

Examples
--------
>>> import sedtrails
>>> sedtrails.run_simulation('config.yml', output='results.nc')
>>> sedtrails.plot_trajectories('results.nc')
"""

from .__version__ import __version__

# Import Application Interface (API) functions and classes
from .application_interfaces.api import (
    # Simulation
    run_simulation,
    create_simulation_runner,
    SimulationRunner,
    # Configuration
    load_configuration,
    validate_configuration,
    create_config_template,
    create_restart_config,
    # Visualization
    plot_trajectories,
    inspect_netcdf,
    # Analysis (not yet fully implemented)
    analyze_simulation,
    analyze_network,
)

# # Backward compatibility - direct access to NetCDFInspector
# from .application_interfaces.nc_inspector import NetCDFInspector

__all__ = [
    # Version
    '__version__',
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
    # Analysis
    'analyze_simulation',
    'analyze_network',
    # Backward compatibility
    # 'NetCDFInspector',
]
