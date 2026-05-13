"""
Format Converter: converts input data formats into SedtrailsData format.

This module reads various input data formats (e.g., NetCDF files from different
hydrodynamic models) and converts them into the SedtrailsData structure for
use in the SedTRAILS particle tracking system.
"""

from dataclasses import dataclass
from types import ModuleType
from typing import Dict, Optional, Union

import numpy as np

from sedtrails.transport_converter.sedtrails_data import SedtrailsData


@dataclass
class SeederFieldData:
    """
    Minimal field data needed by ParticleSeeder: only spatial coordinates.
    """

    x: np.ndarray
    y: np.ndarray


class FormatConverter:
    """
    A class to convert various input data formats to the SedtrailsData format.

    This class provides methods to read data from different file formats and
    convert them to the SedtrailsData format for use in the SedTRAILS particle
    tracking system.
    """

    def __init__(self, config: Dict):
        """
        Initialize the FormatConverter.

        Parameters:
        -----------
        config : dict
            Configuration dictionary containing settings for the converter.
            Must include 'input_file', 'input_format', optionally 'reference_date' (default
            "1970-01-01" (Unix epoch)) and 'morfac' (default 1.0)
        """
        self.config = config
        self._reference_date: Union[str, None] = None
        self.input_data = None
        self._format_plugin = None
        self._input_format: Union[str, None] = None
        self._input_file: Union[str, None] = None
        self._morfac: Union[float, None] = None

    def __post_init__(self):
        """
        Config validation and initialization.
        """

        if not isinstance(self.config, dict):
            raise TypeError(f'Config must be a dictionary, got {type(self.config)}')

    @property
    def input_file(self):
        """Get the input file path."""
        if self._input_file is None:
            self._input_file = self.config.get('input_file')
            if not self._input_file:
                raise ValueError('Input file path must be provided in the configuration')
        return self._input_file

    @property
    def input_format(self) -> str | None:
        """Get the format to convert to."""
        if self._input_format is None:
            self._input_format = self.config.get('input_format')
            if not self._input_format:
                raise ValueError('Input format must be specified in the configuration')
        return self._input_format

    @property
    def reference_date(self) -> np.datetime64:
        """Get the reference date as a numpy datetime64 object."""

        if self._reference_date is None:
            self._reference_date = self.config.get('reference_date', '1970-01-01')
        return np.datetime64(self._reference_date)  # Default to Unix epoch

    @property
    def morfac(self) -> float:
        """Get the morphological acceleration factor."""
        if self._morfac is None:
            self._morfac = self.config.get('morfac', 1.0)
        return self._morfac

    @property
    def format_plugin(self):
        """
        Get the format plugin instance based on the specified format.
        """

        import importlib  # lazy import for performance

        if self._format_plugin is None:
            # Dynamically import the format plugin based on the input type
            plugin_module_name = f'sedtrails.transport_converter.plugins.format.{self.input_format}'
            plugin_module: Optional[ModuleType] = None
            try:
                plugin_module = importlib.import_module(plugin_module_name)
                # Initialize the format plugin with the input file and morfac
                self._format_plugin = plugin_module.FormatPlugin(self.input_file, morfac=self.morfac)
            except ImportError as e:
                raise ImportError(
                    f'Failed to import format plugin module: {plugin_module_name} '
                    f'Ensure the module exists and is correctly named.'
                ) from e

        return self._format_plugin

    def convert_to_sedtrails(self, current_time=None, reading_interval=None) -> SedtrailsData:
        """
        Converts dataset to SedtrailsData format.

        Parameters:
        -----------
        current_time : float, optional
            Current simulation time in seconds
        reading_interval : float, optional
            Reading interval in seconds

        Returns:
        --------
        SedtrailsData:
            Data in SedtrailsData format
        """

        if self._format_plugin is None:
            plugin = self.format_plugin
        else:
            plugin = self._format_plugin

        sedtrails_data = plugin.convert(current_time, reading_interval, self.reference_date)

        return sedtrails_data

    def get_seeding_field_data(self) -> SeederFieldData:
        """
        Read only the data required by ParticleSeeder.seed: x and y coordinates.

        Returns
        -------
        SeederFieldData
            Minimal coordinate container for seeding workflows.
        """
        plugin = self.format_plugin

        if hasattr(plugin, 'get_seeding_coordinates'):
            x, y = plugin.get_seeding_coordinates()
        else:
            # Backward-compatible fallback for plugins that only expose full conversion.
            sedtrails_data = plugin.convert(None, None, self.reference_date)
            x, y = sedtrails_data.x, sedtrails_data.y

        return SeederFieldData(x=np.asarray(x), y=np.asarray(y))


if __name__ == '__main__':
    print('Please see the examples directory for usage examples.')

    conf = {
        'input_file': 'sedtrails/sample-data/inlet_sedtrails.nc',
        'input_format': 'fm_netcdf',
        'reference_date': '1970-01-01',
        'morfac': 1.0,
    }

    converter = FormatConverter(conf)
    converter.convert_to_sedtrails()
