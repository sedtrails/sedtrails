import os
import xugrid as xu
import numpy as np
from pathlib import Path


class SimulationDataBuffer:
    """
    Temporarily stores chunks of simulation data in memory, while they wait to be written to a simulation file.

    Attributes
    ----------
    buffer : dict
        Dictionary holding simulation data arrays (particle ID, positions, and times at the moment).
    """

    def __init__(self):
        """
        Initialize an empty simulation data buffer.
        """
        self.buffer = {
            'particle_id': [],
            'time': [],
            'x': [],
            'y': [],
            # Add other fields as needed (e.g., velocity, status, etc.)
        }

    def add(self, particle_id, time, x, y):
        """
        Add a new data point to the buffer.

        Parameters
        ----------
        particle_id : int
            Unique identifier for the particle.
        time : float or int
            Simulation time or time step.
        x : float
            X-coordinate of the particle.
        y : float
            Y-coordinate of the particle.
        """
        self.buffer['particle_id'].append(particle_id)
        self.buffer['time'].append(time)
        self.buffer['x'].append(x)
        self.buffer['y'].append(y)

    def clear(self):
        """
        Clear the buffer.
        """
        for key in self.buffer:
            self.buffer[key] = []

    def get_data(self):
        """
        Return the buffer as a dictionary of numpy arrays.

        Returns
        -------
        dict
            Dictionary with keys as field names and values as numpy arrays.
        """
        return {k: np.array(v) for k, v in self.buffer.items()}

    def to_ugrid_dataset(self, node_x, node_y, face_node_connectivity, fill_value=-1):
        """
        Convert the buffer contents to a xu.UgridDataset for writing.

        Parameters
        ----------
        node_x : np.ndarray
            Mesh node x-coordinates.
        node_y : np.ndarray
            Mesh node y-coordinates.
        face_node_connectivity : np.ndarray
            Mesh face connectivity.
        fill_value : int, optional
            Fill value for unused nodes (default: -1).

        Returns
        -------
        xu.UgridDataset
            UGRID-compliant dataset containing the buffered simulation data.
        """
        mesh = xu.Ugrid2d(
            node_x=node_x, node_y=node_y, face_node_connectivity=face_node_connectivity, fill_value=fill_value
        )
        ds = mesh.to_dataset()
        data = self.get_data()
        # Use 'time' as the coordinate and dimension
        ds = ds.assign_coords(time=('time', data['time']))
        ds['particle_id'] = ('time', data['particle_id'])
        ds['x'] = ('time', data['x'])
        ds['y'] = ('time', data['y'])
        return xu.UgridDataset(ds)

    def write_to_disk(self, node_x, node_y, face_node_connectivity, fill_value, writer, filename):
        """
        Write the current buffer to disk as a UGRID NetCDF file and clear the buffer.

        Parameters
        ----------
        node_x : np.ndarray
            Mesh node x-coordinates.
        node_y : np.ndarray
            Mesh node y-coordinates.
        face_node_connectivity : np.ndarray
            Mesh face connectivity.
        fill_value : int
            Fill value for unused nodes.
        writer : NetCDFWriter
            Writer instance to handle NetCDF output.
        filename : str
            Name of the output NetCDF file.
        """
        ugrid_ds = self.to_ugrid_dataset(node_x, node_y, face_node_connectivity, fill_value)
        writer.write(ugrid_ds, filename)
        self.clear()

    @staticmethod
    def merge_output_files(output_dir, merged_filename='merged_output.nc'):
        """
        Merge all .sim_buffer_*.nc files in the output directory into a single NetCDF file.

        Merging is based on the assumption that all files have the same structure and can be
        combined by coordinates, e.g., time, observation index, etc. This way, the merged
        file will contain the entire simulation duration and all particles.
        Each file chunk should have non-overlapping coordinate values.

        Parameters
        ----------
        output_dir : Path or str
            Directory containing the intermediate NetCDF files.
        merged_filename : str
            Name of the merged output file.
        """
        output_dir = Path(output_dir)
        print('Output dir:', output_dir)
        files = sorted(
            output_dir / f for f in os.listdir(output_dir) if f.startswith('.sim_buffer_') and f.endswith('.nc')
        )
        if not files:
            raise FileNotFoundError('No .sim_buffer_*.nc files found to merge.')
        ds = xu.open_mfdataset(files, combine='by_coords')
        ds.ugrid.to_netcdf(output_dir / merged_filename)
