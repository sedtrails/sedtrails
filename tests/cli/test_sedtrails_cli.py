"""
Unit tests for the SedTRAILS CLI commands using Typer's CliRunner.
"""

import pytest
import yaml
import os
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import Mock, patch
from sedtrails.configuration_interface.cli import app


class TestSedtrailsCLI:
    """
    Test suite for the Sedtrails CLI commands.
    """

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
            'folder_settings': {'input_data': 'input_data.nc', 'output_dir': 'output'},
            'time': {'timestep': '30S', 'duration': '1D'},
            'particles': {'count': 100, 'release_locations': [[0.0, 0.0]]},
        }

    @pytest.fixture
    def mock_simulation(self):
        """Mock Simulation class."""
        with patch('sedtrails.configuration_interface.cli.main.Simulation') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance

    def test_run_simulation_default_success(self, runner, sample_config_data, mock_simulation):
        """Test successful run with default config and output file names."""
        with runner.isolated_filesystem():
            # Create default config file
            with open('sedtrails.yml', 'w') as f:
                yaml.dump(sample_config_data, f)

            # Create necessary directories
            os.makedirs('output', exist_ok=True)
            Path('input_data.nc').touch()

            result = runner.invoke(app, ['run'])

            assert result.exit_code == 0
            assert "Validating configuration from 'sedtrails.yml'" in result.stdout
            assert 'Configuration validated successfully.' in result.stdout
            assert 'Running simulation...' in result.stdout
            assert "Simulation complete. Output saved to 'sedtrails.nc'." in result.stdout

            # Verify Simulation was called correctly
            mock_simulation.validate_config.assert_called_once()
            mock_simulation.run.assert_called_once()

    def test_run_simulation_custom_config(self, runner, sample_config_data, mock_simulation):
        """Test run with custom config file."""
        with runner.isolated_filesystem():
            custom_config = 'my_config.yml'

            # Create custom config file
            with open(custom_config, 'w') as f:
                yaml.dump(sample_config_data, f)

            # Create necessary files/directories
            os.makedirs('output', exist_ok=True)
            Path('input_data.nc').touch()

            result = runner.invoke(app, ['run', '--config', custom_config])

            assert result.exit_code == 0
            assert f"Validating configuration from '{custom_config}'" in result.stdout
            assert 'Configuration validated successfully.' in result.stdout
            assert 'Running simulation...' in result.stdout
            assert "Simulation complete. Output saved to 'sedtrails.nc'." in result.stdout

    def test_run_simulation_custom_output(self, runner, sample_config_data, mock_simulation):
        """Test run with custom output file."""
        with runner.isolated_filesystem():
            custom_output = 'my_results.nc'

            # Create config file
            with open('sedtrails.yml', 'w') as f:
                yaml.dump(sample_config_data, f)

            # Create necessary files/directories
            os.makedirs('output', exist_ok=True)
            Path('input_data.nc').touch()

            result = runner.invoke(app, ['run', '--output', custom_output])

            assert result.exit_code == 0
            assert "Validating configuration from 'sedtrails.yml'" in result.stdout
            assert f"Simulation complete. Output saved to '{custom_output}'." in result.stdout

    def test_run_simulation_custom_config_and_output(self, runner, sample_config_data, mock_simulation):
        """Test run with both custom config and output files."""
        with runner.isolated_filesystem():
            custom_config = 'my_config.yml'
            custom_output = 'my_results.nc'

            # Create custom config file
            with open(custom_config, 'w') as f:
                yaml.dump(sample_config_data, f)

            # Create necessary files/directories
            os.makedirs('output', exist_ok=True)
            Path('input_data.nc').touch()

            result = runner.invoke(app, ['run', '--config', custom_config, '--output', custom_output])

            assert result.exit_code == 0
            assert f"Validating configuration from '{custom_config}'" in result.stdout
            assert f"Simulation complete. Output saved to '{custom_output}'." in result.stdout

    def test_run_simulation_short_options(self, runner, sample_config_data, mock_simulation):
        """Test run with short option flags."""
        with runner.isolated_filesystem():
            custom_config = 'config.yml'
            custom_output = 'output.nc'

            # Create config file
            with open(custom_config, 'w') as f:
                yaml.dump(sample_config_data, f)

            # Create necessary files/directories
            os.makedirs('output', exist_ok=True)
            Path('input_data.nc').touch()

            result = runner.invoke(app, ['run', '-c', custom_config, '-o', custom_output])

            assert result.exit_code == 0
            assert f"Validating configuration from '{custom_config}'" in result.stdout
            assert f"Simulation complete. Output saved to '{custom_output}'." in result.stdout

    def test_run_simulation_validation_error(self, runner, sample_config_data, mock_simulation):
        """Test run when validation fails."""
        with runner.isolated_filesystem():
            # Create config file
            with open('sedtrails.yml', 'w') as f:
                yaml.dump(sample_config_data, f)

            # Mock validation to raise an exception
            mock_simulation.validate_config.side_effect = Exception('Invalid configuration')

            result = runner.invoke(app, ['run'])

            assert result.exit_code == 1
            assert 'Error validating configuration: Invalid configuration' in result.stdout
            mock_simulation.validate_config.assert_called_once()
            mock_simulation.run.assert_not_called()

    def test_run_simulation_runtime_error(self, runner, sample_config_data, mock_simulation):
        """Test run when simulation fails during execution."""
        with runner.isolated_filesystem():
            # Create config file and necessary files
            with open('sedtrails.yml', 'w') as f:
                yaml.dump(sample_config_data, f)

            os.makedirs('output', exist_ok=True)
            Path('input_data.nc').touch()

            # Mock simulation to raise an exception during run
            mock_simulation.run.side_effect = Exception('Simulation failed')

            result = runner.invoke(app, ['run'])

            assert result.exit_code == 1
            assert 'Configuration validated successfully.' in result.stdout
            assert 'Error running simulation: Simulation failed' in result.stdout
            mock_simulation.validate_config.assert_called_once()
            mock_simulation.run.assert_called_once()

    @patch('sedtrails.configuration_interface.cli.main.Simulation')
    def test_run_simulation_constructor_called_correctly(self, mock_simulation_class, runner, sample_config_data):
        """Test that Simulation constructor is called with correct config file."""
        with runner.isolated_filesystem():
            custom_config = 'test_config.yml'

            # Create config file
            with open(custom_config, 'w') as f:
                yaml.dump(sample_config_data, f)

            # Create necessary files/directories
            os.makedirs('output', exist_ok=True)
            Path('input_data.nc').touch()

            runner.invoke(app, ['run', '--config', custom_config])

            # Verify Simulation was instantiated with correct config file
            mock_simulation_class.assert_called_once_with(custom_config)
