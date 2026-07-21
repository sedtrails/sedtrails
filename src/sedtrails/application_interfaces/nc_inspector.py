#!/usr/bin/env python3
"""
Functions to inspect metadata of SedTrails NetCDF results.
"""

from pathlib import Path
import numpy as np
import xarray as xr


class NetCDFInspector:
    """Inspect and print metadata from a SedTRAILS NetCDF file.

    Parameters
    ----------
    nc_file : str or pathlib.Path
        Path to the NetCDF file to inspect.

    Attributes
    ----------
    nc_file : pathlib.Path
        Path to the NetCDF file with SedTRAILS results.
    data : xarray.Dataset or None
        Dataset loaded from `nc_file` when the file can be opened.
    """

    @staticmethod
    def _decode_text_value(value) -> str:
        """Decode NetCDF text values from bytes, strings, or char arrays."""
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace').replace('\x00', '').strip()
        if isinstance(value, str):
            return value.replace('\x00', '').strip()
        if isinstance(value, np.ndarray):
            flat = value.ravel()
            chars: list[str] = []
            for item in flat:
                if isinstance(item, bytes):
                    decoded = item.decode('utf-8', errors='replace')
                    if decoded != '\x00':
                        chars.append(decoded)
                else:
                    text = str(item)
                    if text != '\x00':
                        chars.append(text)
            return ''.join(chars).replace('\x00', '').strip()
        return str(value)

    def __init__(self, nc_file: str) -> None:
        """Initialize the Inspector with the path to a NetCDF file.

        Parameters
        ----------
        nc_file : str or pathlib.Path
            Path to the NetCDF file with SedTRAILS results.

        Attributes
        ----------
        nc_file : pathlib.Path
            Path to the NetCDF file with SedTRAILS results.
        data : xarray.Dataset or None
            The xarray Dataset loaded from the NetCDF file.

        Raises
        ------
        FileExistsError
            If `nc_file` does not exist.
        RuntimeError
            If the NetCDF file cannot be opened.
        """
        self.nc_file = Path(nc_file)
        self.data = None

        if not self.nc_file.exists():
            print(f'Current working directory: {Path.cwd()}')
            raise FileExistsError(f"Error: File '{str(self.nc_file)}' not found!")
        else:
            print(f'Inspecting NetCDF file: {str(self.nc_file)}')
            try:
                self.data = xr.open_dataset(self.nc_file)
            except Exception as e:
                raise RuntimeError(f'Error reading NetCDF file: {e}') from e

    def print_metadata(self) -> None:
        """
        Print comprehensive metadata about the NetCDF dataset.

        Returns
        -------
        None
            This method returns None after printing dataset metadata.
        """

        if self.data is None:
            raise RuntimeError('No data loaded. Unable to print metadata.')

        print('=' * 84)
        print('SEDTRAILS NETCDF FILE METADATA')
        print('=' * 84)
        print(f'FILE: {self.nc_file}')
        print(f'SIZE: {self.data.nbytes / 1024 / 1024:.2f} MB')

        # File attributes
        print('\nGLOBAL ATTRIBUTES:')
        print('-' * 60)
        for attr_name, attr_value in self.data.attrs.items():
            print(f'  {attr_name}: {attr_value}')

        # Dimensions
        print('\nDIMENSIONS:')
        print('-' * 60)
        for dim_name, dim_size in self.data.sizes.items():
            print(f'  {dim_name}: {dim_size}')

        # Coordinates
        print('\nCOORDINATES:')
        print('-' * 60)
        for coord_name, coord in self.data.coords.items():
            print(f'  {coord_name}: {coord.dims} {coord.dtype} {coord.shape}')

        # Data variables
        print('\nDATA VARIABLES:')
        print('-' * 60)
        for var_name, var in self.data.data_vars.items():
            print(f'  {var_name}: {var.dims} {var.dtype} {var.shape}')
            if var.attrs:
                for attr_name, attr_value in var.attrs.items():
                    print(f'    {attr_name}: {attr_value}')

        print('')
        print('=' * 84)
        return None

    def inspect_populations(self) -> None:
        """Extract and display population information.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset containing sedTRAILS results.

        Returns
        -------
        None
            This method returns None after printing population metadata.
        """

        if self.data is None:
            raise RuntimeError('No data loaded. Unable to inspect populations.')

        print('=' * 84)
        print('PARTICLE POPULATIONS:')
        print('-' * 60)

        if 'population_name' in self.data.variables:
            n_populations = self.data.sizes['n_populations']

            for i in range(n_populations):
                population_name_var = self.data['population_name']
                if population_name_var.ndim == 2:
                    name = self._decode_text_value(population_name_var[i, :].values)
                else:
                    name = self._decode_text_value(population_name_var[i].values)

                start_idx = self.data['population_start_idx'][i].values
                count = self.data['population_count'][i].values
                particle_type_var = self.data['population_particle_type']
                if particle_type_var.ndim == 2:
                    particle_type = self._decode_text_value(particle_type_var[i, :].values)
                else:
                    particle_type = self._decode_text_value(particle_type_var[i].values)

                print(f'  Population {i + 1}: {name}')
                print(f'    Particle type: {particle_type}')
                print(f'    Start index: {start_idx}')
                print(f'    Particle count: {count} \n')
        print('=' * 84)

        return None
