#!/usr/bin/env python3
"""
Functions to inspect metadata of SedTrails NetCDF results.
"""

from pathlib import Path
import xarray as xr


class NetCDFInspector:
    """A class to inspect and print metadata from SedTRAILS NetCDF files."""

    def __init__(self, nc_file: str) -> None:
        """Initialize the Inspector with the path to a NetCDF file.

        Attributes
        ----------
        nc_file : Path
            Path to the NetCDF file with sedTRAILS results.
        data : xr.Dataset or None
            The xarray Dataset loaded from the NetCDF file.
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
                # Decode population name (stored as bytes)
                name_bytes = self.data['population_name'][i, :].values
                name = ''.join([char.decode('utf-8') for char in name_bytes if char != b'\x00'])

                start_idx = self.data['population_start_idx'][i].values
                count = self.data['population_count'][i].values
                particle_type = self.data['population_particle_type'][i].values

                print(f'  Population {i + 1}: {name}')
                print(f'    Particle type: {particle_type}')
                print(f'    Start index: {start_idx}')
                print(f'    Particle count: {count} \n')
        print('=' * 84)

        return None
