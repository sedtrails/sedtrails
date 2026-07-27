"""
Unit tests for the SedTRAILS CLI commands.
"""

import pytest
import yaml
from click.testing import CliRunner
from typer.main import get_command
from types import SimpleNamespace
from unittest.mock import patch
from sedtrails.application_interfaces.cli import app


class TestSedtrailsCLI:
    """
    Test suite for the Sedtrails CLI commands.
    """

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def cli_command(self):
        """Create a Click command from the Typer app."""
        return get_command(app)

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
            'folder_settings': {'input_data': 'input_data.nc', 'output_dir': 'output'},
            'time': {'timestep': '30S', 'duration': '1D'},
            'particles': {'count': 100, 'release_locations': [[0.0, 0.0]]},
        }

    def test_root_help(self, runner, cli_command):
        """Test the root help command."""
        result = runner.invoke(cli_command, ['--help'])

        assert result.exit_code == 0
        assert 'Sedtrails: Configure, run, and analyze sediment particle tracking.' in result.stdout
        assert 'run' in result.stdout

    def test_subcommand_help(self, runner, cli_command):
        """Test subcommand help rendering."""
        result = runner.invoke(cli_command, ['run', '--help'])
        run_command = cli_command.get_command(None, 'run')
        config_option = next(param for param in run_command.params if param.name == 'config_file')

        assert result.exit_code == 0
        assert 'Run a simulation based on a configuration file.' in result.stdout
        assert '--config' in config_option.opts
        assert '-c' in config_option.opts

    @pytest.fixture
    def mock_run_simulation(self):
        """Mock run_simulation API function."""
        with patch('sedtrails.application_interfaces.api.run_simulation') as mock:
            mock.return_value = 'output'
            yield mock

    def test_run_simulation_default_success(self, runner, cli_command, sample_config_data, mock_run_simulation, tmp_path, monkeypatch):
        """Test successful run with default config file name."""
        monkeypatch.chdir(tmp_path)
        with open('sedtrails.yml', 'w') as f:
            yaml.dump(sample_config_data, f)

        result = runner.invoke(cli_command, ['run'])

        assert result.exit_code == 0
        assert "Starting simulation from 'sedtrails.yml'..." in result.stdout
        assert "Simulation complete. Results saved to 'output'." in result.stdout

        # Verify run_simulation was called correctly
        mock_run_simulation.assert_called_once_with(
            config_file='sedtrails.yml',
            verbose=True,
            report_domain_exits=True,
        )

    def test_run_simulation_custom_config(self, runner, cli_command, sample_config_data, mock_run_simulation, tmp_path, monkeypatch):
        """Test run with custom config file."""
        monkeypatch.chdir(tmp_path)
        custom_config = 'my_config.yml'

        with open(custom_config, 'w') as f:
            yaml.dump(sample_config_data, f)

        result = runner.invoke(cli_command, ['run', '--config', custom_config])

        assert result.exit_code == 0
        assert f"Starting simulation from '{custom_config}'..." in result.stdout
        assert "Simulation complete. Results saved to 'output'." in result.stdout

        # Verify run_simulation was called with custom config
        mock_run_simulation.assert_called_once_with(
            config_file=custom_config,
            verbose=True,
            report_domain_exits=True,
        )

    def test_run_simulation_short_option(self, runner, cli_command, sample_config_data, mock_run_simulation, tmp_path, monkeypatch):
        """Test run with short option flag."""
        monkeypatch.chdir(tmp_path)
        custom_config = 'config.yml'

        with open(custom_config, 'w') as f:
            yaml.dump(sample_config_data, f)

        result = runner.invoke(cli_command, ['run', '-c', custom_config])

        assert result.exit_code == 0
        assert f"Starting simulation from '{custom_config}'..." in result.stdout
        assert "Simulation complete. Results saved to 'output'." in result.stdout

        # Verify run_simulation was called with short option
        mock_run_simulation.assert_called_once_with(
            config_file=custom_config,
            verbose=True,
            report_domain_exits=True,
        )

    def test_run_simulation_can_disable_domain_exit_reporting(
        self, runner, cli_command, sample_config_data, mock_run_simulation, tmp_path, monkeypatch
    ):
        """Test run with domain exit reporting disabled."""
        monkeypatch.chdir(tmp_path)
        with open('sedtrails.yml', 'w') as f:
            yaml.dump(sample_config_data, f)

        result = runner.invoke(cli_command, ['run', '--no-report-domain-exits'])

        assert result.exit_code == 0
        mock_run_simulation.assert_called_once_with(
            config_file='sedtrails.yml',
            verbose=True,
            report_domain_exits=False,
        )

    def test_viz_trajectories_defaults_to_output_path_and_new_plot_settings(
        self, runner, cli_command, tmp_path, monkeypatch
    ):
        """Test default trajectories visualization settings."""
        monkeypatch.chdir(tmp_path)

        with patch('sedtrails.application_interfaces.api.plot_trajectories') as mock_plot_trajectories:
            result = runner.invoke(cli_command, ['viz', 'trajectories'])

        assert result.exit_code == 0
        mock_plot_trajectories.assert_called_once_with(
            'sedtrails_results.nc',
            output=None,
            max_particles=10000,
            sample_fraction=None,
            sample_seed=0,
            markers='end',
            marker_size=3.0,
            panels='spatial',
            show=None,
        )

    def test_run_simulation_error(self, runner, cli_command, sample_config_data, mock_run_simulation, tmp_path, monkeypatch):
        """Test run when simulation fails."""
        monkeypatch.chdir(tmp_path)
        with open('sedtrails.yml', 'w') as f:
            yaml.dump(sample_config_data, f)

        # Mock run_simulation to raise an exception
        mock_run_simulation.side_effect = Exception('Simulation failed')

        result = runner.invoke(cli_command, ['run'])

        assert result.exit_code == 1
        assert 'Error running simulation: Simulation failed' in result.stdout
        mock_run_simulation.assert_called_once()

    def test_config_restart_command_success(self, runner, cli_command):
        """Test successful generation of restart config via CLI."""
        with patch('sedtrails.application_interfaces.api.create_restart_config') as mock_create_restart:
            mock_create_restart.return_value = SimpleNamespace(
                output_config='restart.yaml',
                restart_time='2020-01-01 00:02:00',
                retained_particles=10,
                seed_files={'population_1': 'seeds/population_1.restart_points.csv'},
            )

            result = runner.invoke(
                cli_command,
                [
                    'config',
                    'restart',
                    '--file',
                    'results.nc',
                    '--config',
                    'base.yaml',
                    '--output',
                    'restart.yaml',
                ],
            )

            assert result.exit_code == 0
            assert "Restart config written to 'restart.yaml'" in result.stdout
            assert "Restart time set to '2020-01-01 00:02:00'" in result.stdout
            assert 'Retained particles: 10' in result.stdout
            assert 'population_1' in result.stdout

            mock_create_restart.assert_called_once_with(
                results_file='results.nc',
                base_config_file='base.yaml',
                output_config_file='restart.yaml',
                seed_points_dir=None,
            )
