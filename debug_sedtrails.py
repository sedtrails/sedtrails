#!/usr/bin/env python3
"""
Debug script for SedTRAILS simulation

This script allows you to run a SedTRAILS simulation in debug mode within VS Code.
You can set breakpoints, step through the code, and inspect variables to troubleshoot issues.

Usage:
    1. Set breakpoints in this file or in the SedTRAILS source code
    2. Run this script in VS Code debugger (F5 or Run -> Start Debugging)
    3. The debugger will stop at breakpoints allowing you to inspect the simulation

This script replicates the behavior of:
    sedtrails run -c U:\\MangroveConnectivity\\pyST\\02_soulsby-debugging\\001_defaults\\config.example_soulsby.yaml
"""

import logging
import sys
from pathlib import Path

# Add the source directory to the path so we can import sedtrails
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    """Main debug function that runs the SedTRAILS simulation."""

    # Configuration
    # config_file = r"C:\surfdrive\650_SedTRAILS\pyTesting\02_soulsby-debugging\005_finer\config.example_soulsby.yaml"
    # config_file = r"C:\surfdrive\650_SedTRAILS\pyTesting\02_soulsby-debugging\006_coarser\config.example_soulsby.yaml"
    # config_file = r"C:\surfdrive\650_SedTRAILS\pyTesting\02_soulsby-debugging\007_finer_hidExp_off\config.example_soulsby.yaml"
    # config_file = r"C:\surfdrive\650_SedTRAILS\pyTesting\02_soulsby-debugging\008_coarser_hidExp_off\config.example_soulsby.yaml"
    # config_file = r"C:\surf\650_SedTRAILS\pyTesting\02_soulsby-debugging\004_grain-size-test\config.example_soulsby.yaml"
    #  config_file = r"C:\surf\650_SedTRAILS\pyTesting\02_soulsby-debugging\009_missingGrainSizeField\config.example_soulsby.yaml"
    #  config_file = r'.\examples\config.example_sfincs.yaml'
    # config_file = r'.\examples\config.example_soulsby.yaml'
    config_file = r'.\examples\sedtrails-example-burial-pr401.yaml'
    
    # Set up logging for verbose output (equivalent to CLI verbose mode)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print('Debug: Starting SedTRAILS simulation...')
    print(f'Debug: Configuration file: {config_file}')
    print(f'Debug: Working directory: {Path.cwd()}')

    # Check if config file exists
    config_path = Path(config_file)
    if not config_path.exists():
        print(f'ERROR: Configuration file not found: {config_file}')
        return 1

    try:
        # Import SedTRAILS (this is where initial import errors might occur)
        print('Debug: Importing SedTRAILS...')
        import sedtrails

        print(f'Debug: SedTRAILS version: {sedtrails.__version__}')

        # This is equivalent to what the CLI does:
        # sedtrails.application_interfaces.api.run_simulation(config_file=config_file, verbose=True)
        print('Debug: Starting simulation...')

        # You can set a breakpoint here to step into the run_simulation function
        output_dir = sedtrails.run_simulation(config_file=config_file, verbose=True)

        print('Debug: Simulation completed successfully!')
        print(f'Debug: Results saved to: {output_dir}')
        return 0

    except ImportError as e:
        print(f'ERROR: Failed to import SedTRAILS: {e}')
        print('This might be due to missing dependencies or environment issues.')
        return 1

    except Exception as e:
        print(f'ERROR: Simulation failed: {e}')
        print(f'ERROR Type: {type(e).__name__}')

        # Print the full stack trace for debugging
        import traceback

        print('\nFull traceback:')
        traceback.print_exc()
        return 1


def debug_config_loading(config_file):
    """Debug function to test just the configuration loading part."""

    try:
        print('Debug: Testing configuration loading...')
        import sedtrails

        # Test configuration loading
        config = sedtrails.load_configuration(config_file)
        print('Debug: Configuration loaded successfully!')
        print(f'Debug: Config keys: {list(config.keys())}')

        # Test configuration validation
        is_valid = sedtrails.validate_configuration(config_file)
        print(f'Debug: Configuration valid: {is_valid}')

        return config

    except Exception as e:
        print(f'ERROR: Configuration loading failed: {e}')
        import traceback

        traceback.print_exc()
        return None


def debug_simulation_creation(config_file):
    """Debug function to test just the simulation object creation."""

    try:
        print('Debug: Testing simulation creation...')

        # Import the Simulation class directly
        from sedtrails.simulation_orchestrator.simulation_manager import Simulation

        # Create simulation instance (this is where many issues occur)
        print('Debug: Creating Simulation instance...')
        simulation = Simulation(config_file, enable_dashboard=None)
        print('Debug: Simulation instance created successfully!')

        # You can inspect the simulation object here
        print(f'Debug: Simulation config loaded: {simulation.config is not None}')
        print(f'Debug: Data manager created: {simulation.data_manager is not None}')

        return simulation

    except Exception as e:
        print(f'ERROR: Simulation creation failed: {e}')
        import traceback

        traceback.print_exc()
        return None


if __name__ == '__main__':
    """
    Main entry point for debugging.
    
    You can uncomment different debug functions to test specific parts:
    """

    # Uncomment one of these debug functions to test specific parts:

    # 1. Full simulation (equivalent to CLI command)
    exit_code = main()

    # 2. Test just configuration loading (uncomment to test config only)
    # config = debug_config_loading()
    # exit_code = 0 if config is not None else 1

    # 3. Test just simulation creation (uncomment to test simulation creation only)
    # simulation = debug_simulation_creation()
    # exit_code = 0 if simulation is not None else 1

    # Exit with the appropriate code
    sys.exit(exit_code)
