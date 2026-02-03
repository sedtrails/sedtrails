"""
NetCDF Writer
=============

Writes NetCDF files produced by the SedTrails Particle Tracer System.

"""

import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from .xarray_dataset import create_sedtrails_dataset, populate_population_metadata, populate_flowfield_metadata


class NetCDFWriter:
    """
    A class for writing NetCDF files for the SedTrails Particle Tracer System using xarray.

    This class provides methods to:
    - Create xarray datasets with SedTrails structure
    - Add metadata to datasets (populations, flow fields, simulation info)
    - Write xarray datasets to NetCDF files with optional timestep trimming

    Examples
    --------

    # Method 1: Step-by-step approach
    output_dir = "some/path"
    writer = NetCDFWriter(output_dir)

    # Create the xarray dataset
    dataset = writer.create_dataset(
    N_particles=total_particles,
    N_populations=n_populations,
    N_timesteps=n_timesteps,
    N_flowfields=n_flowfields
    )

    # Populate metadata to the dataset
    writer.add_metadata(dataset, populations, flow_field_names)

    # Write the dataset to a netcdf file
    writer.write(dataset, filename)

    # Optional: Write with timestep trimming
    writer.write(dataset, filename, trim_to_actual_timesteps=True, actual_timesteps=100)

    # Method 2: All-in-one approach
    writer.create_and_write_simulation_results(
    populations, flow_field_names, n_timesteps, filename
    )

    Attributes
    ----------
    output_dir : pathlib.Path
        The directory where output files (NetCDF, images, etc.) are stored.

    """

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        # If the output directory already exists, we add a timestamp to avoid overwriting
        # if output_dir.exists():
        #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #     output_dir = output_dir.parent / f'{output_dir.name}_{timestamp}'
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_filename(self, filename):
        """
        Validates the file name to ensure it is a NetCDF file.
        """
        if not filename.endswith('.nc'):
            raise ValueError('Output file must have a .nc extension.')

    def write(self, xr_dataset, filename, trim_to_actual_timesteps=False, actual_timesteps=None):
        """
        Write an xr.Dataset to a NetCDF file in the output directory.

        Parameters
        ----------
        xr_dataset : xr.Dataset
            The xarray dataset to write.
        filename : str
            The name of the NetCDF file to write (should end with .nc).
        trim_to_actual_timesteps : bool, optional
            Whether to trim the dataset to actual timesteps used (default: False)
        actual_timesteps : int, optional
            Number of actual timesteps to keep (if None, tries to determine automatically)

        Returns
        -------
        pathlib.Path
            Path to the written file

        """
        self._validate_filename(filename)
        output_path = self.output_dir / filename
        if not isinstance(xr_dataset, xr.Dataset):
            raise TypeError('Input must be an xr.Dataset.')

        output_dataset = xr_dataset

        # Handle trimming if requested
        if trim_to_actual_timesteps:
            if actual_timesteps is None:
                # Try to determine actual timesteps by finding the last non-NaN time value
                if 'time' in xr_dataset and 'n_timesteps' in xr_dataset.dims:
                    time_data = xr_dataset['time'].values
                    if not np.all(np.isnan(time_data)):
                        # Find the last timestep with any non-NaN values
                        non_nan_mask = ~np.isnan(time_data)
                        if np.any(non_nan_mask):
                            last_timestep = np.max(np.where(non_nan_mask)[1]) + 1
                            actual_timesteps = last_timestep

            if actual_timesteps is not None and 'n_timesteps' in xr_dataset.dims:
                output_dataset = xr_dataset.isel(n_timesteps=slice(0, actual_timesteps))

        # Add standard metadata if not present
        if 'title' not in output_dataset.attrs:
            output_dataset.attrs['title'] = 'SedTrails Particle Simulation Results'
        if 'institution' not in output_dataset.attrs:
            output_dataset.attrs['institution'] = 'SedTrails Particle Tracer System'
        if 'created_on' not in output_dataset.attrs:
            output_dataset.attrs['created_on'] = datetime.now().isoformat()

        output_dataset.to_netcdf(output_path)
        return output_path

    def create_dataset(self, N_particles, N_populations, N_timesteps, N_flowfields, name_strlen=24):
        """
        Create an xarray dataset with the SedTrails structure.

        Parameters
        ----------
        N_particles : int
            Number of particles
        N_populations : int
            Number of populations
        N_timesteps : int
            Number of timesteps
        N_flowfields : int
            Number of flow fields
        name_strlen : int, optional
            Maximum length for string variables (default: 24)

        Returns
        -------
        xr.Dataset
            The created xarray dataset
        """
        return create_sedtrails_dataset(
            N_particles=N_particles,
            N_populations=N_populations,
            N_timesteps=N_timesteps,
            N_flowfields=N_flowfields,
            name_strlen=name_strlen,
        )

    def add_metadata(self, dataset, populations, flow_field_names, use_transport_fields='D3D', simulation_metadata=None):
        """
        Add metadata to the xarray dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to add metadata to
        populations : list
            List of population objects from the simulation
        flow_field_names : list
            List of flow field names used in the simulation
        simulation_metadata : dict, optional
            Additional simulation metadata to include as global attributes

        Returns
        -------
        xr.Dataset
            The dataset with metadata added (modifies in place and returns)
        """
        # Populate population and flow field metadata
        populate_population_metadata(dataset, populations)
        if flow_field_names:
            populate_flowfield_metadata(dataset, flow_field_names)
        
        if use_transport_fields:
            padded = use_transport_fields.ljust(dataset.sizes["name_strlen"])[:dataset.sizes["name_strlen"]]
            dataset["use_transport_fields"].values[:] = np.frombuffer(padded.encode("ascii"), dtype="S1")

        # Add simulation metadata as global attributes if provided
        if simulation_metadata:
            dataset.attrs.update(simulation_metadata)

        # Add standard metadata
        dataset.attrs['title'] = 'SedTRAILS Particle Simulation Results'
        dataset.attrs['institution'] = 'SedTRAILS Particle Tracer System'
        dataset.attrs['created_on'] = datetime.now().isoformat()

        return dataset

    def create_and_write_simulation_results(
        self,
        populations,
        flow_field_names,
        N_timesteps,
        filename='simulation_results.nc',
        simulation_metadata=None,
        name_strlen=24,
    ):
        """
        Convenience method that creates dataset, adds metadata, and writes to file in one call.

        Parameters
        ----------
        populations : list
            List of population objects from the simulation
        flow_field_names : list
            List of flow field names used in the simulation
        N_timesteps : int
            Number of timesteps in the simulation
        filename : str, optional
            The name of the NetCDF file to write (default: 'simulation_results.nc')
        simulation_metadata : dict, optional
            Additional simulation metadata to include as global attributes
        name_strlen : int, optional
            Maximum length for string variables (default: 24)

        Returns
        -------
        pathlib.Path
            Path to the written file

        """
        # Calculate dimensions
        total_particles = sum([len(pop.particles['x']) for pop in populations])
        n_populations = len(populations)
        n_flowfields = len(flow_field_names) if flow_field_names else 1

        # Create dataset
        dataset = self.create_dataset(
            N_particles=total_particles,
            N_populations=n_populations,
            N_timesteps=N_timesteps,
            N_flowfields=n_flowfields,
            name_strlen=name_strlen,
        )

        # Add metadata
        self.add_metadata(dataset, populations, flow_field_names, simulation_metadata)

        # Write to file
        return self.write(dataset, filename)


if __name__ == '__main__':
    pass
