from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
import xugrid as xu

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData, SedtrailsMetadata


class FormatPlugin(BaseFormatPlugin):
    """
    Plugin for converting SFINCS NetCDF to SedTRAILS format.
    """

    def __init__(self, input_file: str, morfac: float = 1.0):
        """
        Initialize the plugin with the input file.

        Parameters:
        -----------
        input_file : str
            Path to the SFINCS NetCDF file.
        """
        super().__init__()
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')
        self.input_data = None  # holds Dataset after reading
        self._input_variables: List[str] = []

    @property
    def variables(self) -> List[str]:
        """
        Get the variables in the input dataset.

        Returns:
        --------
        List
            List of variable names in the input dataset.
        """

        if self.input_data is None:  # Ensure input_data is loaded
            self.load()

        if not self._input_variables:  # Ensures variables are only loaded once
            try:
                ds = self.input_data
                if isinstance(self.input_data, xu.UgridDataset):
                    ds = (
                        getattr(self.input_data, 'dataset', None)
                        or getattr(self.input_data, '_dataset', None)
                        or getattr(self.input_data, 'ds', None)
                    )
                    if ds is None:
                        raise ValueError('Underlying xarray Dataset not found on UgridDataset')
                self._input_variables = list(ds.data_vars)
            except AttributeError as e:
                raise ValueError('Input data could not retrieve variables') from e
            else:
                print(f'Variables in {self.input_file}:')

        return self._input_variables

    def convert(
        self,
        current_time=None,
        reading_interval=None,
        reference_date: Optional[np.datetime64] = None,
    ) -> SedtrailsData:
        """
        SedtrailsData from SFINCS Netcdf.

        Parameters:
        -----------
        current_time : float, optional
            Current simulation time in seconds
        reading_interval : float, optional
            Reading interval in seconds
        reference_date : np.datetime64, optional
            Reference date for converting time values

        Returns:
        --------
        SedtrailsData
            The converted SedtrailsData object.
        """

        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        # Read the NetCDF file
        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)

        # Determine if we need to slice based on current_time and reading_interval
        time_start_idx, time_end_idx = self._calculate_time_slice(current_time, reading_interval, time_info)

        # Apply time slicing if needed
        if time_start_idx is not None or time_end_idx is not None:
            time_slice = slice(time_start_idx, time_end_idx)
            time_info = self._slice_time_info(time_info, time_slice)

        # Map the variables to SedtrailsData structure
        mapped_data = self._map_sfincs_variables(time_info, time_start_idx, time_end_idx)
        seconds_since_ref = time_info['seconds_since_reference']
        self.reference_date = time_info['reference_date']

        # Calculate magnitudes for vector quantities
        # Flow velocity magnitude
        depth_avg_velocity_magnitude = np.sqrt(
            mapped_data['flow_velocity_x'] ** 2 + mapped_data['flow_velocity_y'] ** 2
        )

        # Create dictionaries for vector quantities
        depth_avg_flow_velocity = {
            'x': mapped_data['flow_velocity_x'],
            'y': mapped_data['flow_velocity_y'],
            'magnitude': depth_avg_velocity_magnitude,
        }

        # Create SedtrailsMetadata object
        metadata = SedtrailsMetadata(
            flowfield_domain={
                'x_min': np.min(mapped_data['x']),
                'x_max': np.max(mapped_data['x']),
                'y_min': np.min(mapped_data['y']),
                'y_max': np.max(mapped_data['y']),
            }
        )

        # Create SedtrailsData object
        sedtrails_data = SedtrailsData(
            times=seconds_since_ref,
            reference_date=self.reference_date,
            x=mapped_data['x'],
            y=mapped_data['y'],
            bed_level=mapped_data['bed_level'],
            depth_avg_flow_velocity=depth_avg_flow_velocity,
            water_depth=mapped_data['water_depth'],
            metadata=metadata,
            fractions=0,  # Default to 0 fractions
            # this has no meaning for SFINCS runs
            bed_load_transport=None,
            suspended_transport=None,
            mean_bed_shear_stress=None,
            max_bed_shear_stress=None,
            sediment_concentration=None,
            nonlinear_wave_velocity=None,
            node_x=mapped_data.get('node_x'),
            node_y=mapped_data.get('node_y'),
            face_node_connectivity=mapped_data.get('face_node_connectivity'),
            face_node_fill_value=-1,
        )

        return sedtrails_data

    def get_seeding_coordinates(self):
        """
        Return only the spatial coordinates required for particle seeding.
        """
        self.load()

        # Fast path: xugrid already provides face centroids.
        if isinstance(self.input_data, xu.UgridDataset):
            grid = self.input_data.grid
            if not isinstance(grid, xu.Ugrid2d):
                raise TypeError(f'Expected Ugrid2d, got {type(grid).__name__}')
            return grid.face_x, grid.face_y

        node_x_var, node_y_var, face_nodes_var, start_index, fill_value = self._get_face_node_mesh_variables()

        return compute_face_centroids(
            node_x_var,
            node_y_var,
            face_nodes_var,
            start_index=start_index,
            fill_value=fill_value,
        )

    def load(self) -> Any:
        """
        Reads and loads a SFINCS NetCDF file using xugrid.
        """

        if self.input_data is None:
            try:
                # First try using xugrid's open_dataset which handles UGRID conventions
                self.input_data = xu.load_dataset(
                    self.input_file,
                    decode_times=True,
                    decode_timedelta=True,
                )
            except Exception as e:
                print(f'Could not open file with xugrid: {e} \n Trying with Xarray...')
                # Fallback to regular xarray
                try:
                    try:
                        self.input_data = xr.open_dataset(
                            self.input_file, decode_times=True, use_cftime=True, decode_timedelta=True
                        )
                    except TypeError:
                        # Older xarray versions may not support use_cftime
                        self.input_data = xr.open_dataset(self.input_file, decode_times=True, decode_timedelta=True)
                except Exception as e:
                    # If time decoding fails, retry without decoding
                    msg = str(e)
                    if 'decode time' in msg or 'decode_times' in msg or 'cftime' in msg:
                        try:
                            self.input_data = xr.open_dataset(
                                self.input_file, decode_times=False, decode_timedelta=False
                            )
                        except Exception as e:
                            raise IOError(f'Failed to open NetCDF file: {e}') from e
                        else:
                            print(f'Successfully loaded (Xarray, decode_times=False): {self.input_file}')
                    else:
                        raise IOError(f'Failed to open NetCDF file: {e}') from e
                else:
                    print(f'Successfully loaded (Xarray): {self.input_file}')
            else:
                print(f'Successfully loaded (Xugrid): {self.input_file}')

    def _get_variable(self, name: str):
        """
        Return an xarray DataArray for variable `name` for either xr.Dataset or xu.UgridDataset.
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        ds = self.input_data
        #   if isinstance(self.input_data, xu.UgridDataset):
        #       ds = (
        #           getattr(self.input_data, 'dataset', None)
        #           or getattr(self.input_data, '_dataset', None)
        #           or getattr(self.input_data, 'ds', None)
        #       )
        #       if ds is None:
        #           raise KeyError(f"Underlying xarray Dataset not found on UgridDataset for variable '{name}'")

        if name in ds:
            return ds[name]
        raise KeyError(f"Variable '{name}' not found in dataset")

    def _get_time_info(self, input_data: Union[xu.UgridDataset, xr.Dataset], reference_date: np.datetime64) -> Dict:
        """
        Get and transforms time information of a dataset.

        Parameters:
        -----------
        input_data : xu.UgridDataset
            The input dataset containing time information.

        reference_data : np.datetime64
            The reference date to calculate time in seconds.

        Returns:
        --------
        Dict
            Dictionary containing time values, start time, end time,
            and time in seconds since reference date
        """

        # check reference_date is a numpy datetime64
        if not isinstance(reference_date, np.datetime64):
            raise TypeError('reference_date must be a numpy datetime64 object')

        if input_data is None:
            raise ValueError('Dataset not loaded. Call read_netcdf_dfm() first.')

        # Get the time variable
        time_var = input_data['time']
        time_values = time_var.values
        time_start = time_values[0]
        time_end = time_values[-1]

        # Get original time units and calendar from the attributes
        orig_units = getattr(time_var, 'units', None)
        orig_calendar = getattr(time_var, 'calendar', 'standard')

        # Normalize time values if they are not already datetime-like
        if not np.issubdtype(np.asarray(time_values).dtype, np.datetime64):
            decoded = None
            if orig_units is not None:
                try:
                    decoded = xr.coding.times.decode_cf_datetime(time_values, orig_units, orig_calendar)
                except Exception:
                    decoded = None
            if decoded is not None:
                try:
                    decoded = xr.coding.times.cftime_to_nptime(decoded)
                except Exception:
                    pass
                time_values = decoded
            else:
                # Fallback: treat numeric values as seconds since reference_date
                time_values = reference_date + np.asarray(time_values, dtype=float) * np.timedelta64(1, 's')

        # Convert time values to seconds since reference_date
        try:
            seconds_since_ref = np.array([float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values])
        except Exception:
            try:
                time_values = xr.coding.times.cftime_to_nptime(time_values)
                seconds_since_ref = np.array(
                    [float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values]
                )
            except Exception as inner_e:
                raise TypeError('Time values could not be converted to seconds since reference_date') from inner_e

        return {
            'time_values': time_values,
            'time_start': time_start,
            'time_end': time_end,
            'original_units': orig_units,
            'original_calendar': orig_calendar,
            'seconds_since_reference': seconds_since_ref,
            'reference_date': reference_date,
            'num_times': len(time_values),
        }

    def _slice_time_info(self, time_info: Dict, time_slice: slice) -> Dict:
        """Slice time info to specified range."""
        sliced_info = time_info.copy()
        sliced_info['time_values'] = time_info['time_values'][time_slice]
        sliced_info['seconds_since_reference'] = time_info['seconds_since_reference'][time_slice]
        sliced_info['num_times'] = len(sliced_info['time_values'])
        if len(sliced_info['time_values']) > 0:
            sliced_info['time_start'] = sliced_info['time_values'][0]
            sliced_info['time_end'] = sliced_info['time_values'][-1]
        return sliced_info

    def _map_sfincs_variables(
        self, time_info, time_start_idx: Optional[int] = None, time_end_idx: Optional[int] = None
    ) -> Dict:
        """
        Map SFINCS variables to SedtrailsData structure.

        Parameters:
        -----------
        time_info : Dict
            Time information
        time_start_idx : int, optional
            Start time index for slicing
        time_end_idx : int, optional
            End time index for slicing

        Returns:
        --------
        Dict
            Dictionary with mapped variables
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call read_data() first.')

        # Get time information
        num_times = time_info['num_times']
        time_slice = (
            slice(time_start_idx, time_end_idx)
            if time_start_idx is not None or time_end_idx is not None
            else slice(None)
        )

        # Derive face coordinates and keep generic UGRID geometry for visualization.
        node_x_var, node_y_var, face_nodes_var, start_index, fill_value = self._get_face_node_mesh_variables()

        face_x, face_y = compute_face_centroids(
            node_x_var,
            node_y_var,
            face_nodes_var,
            start_index=start_index,
            fill_value=fill_value,
        )

        # Variable mapping for SFINCS files
        variable_map = {
            'bed_level': 'zb',  # Bed level
            'water_depth': 'h',  # Water depth
            'flow_velocity_x': 'u',  # X-component of flow velocity
            'flow_velocity_y': 'v',  # Y-component of flow velocity
        }

        # Extract data from dataset
        data = {}

        # First, get spatial coordinates (typically not time-dependent)
        data['x'] = face_x
        data['y'] = face_y
        data['node_x'] = np.asarray(node_x_var)
        data['node_y'] = np.asarray(node_y_var)
        data['face_node_connectivity'] = normalize_face_node_connectivity(
            face_nodes_var,
            start_index=start_index,
            fill_value=fill_value,
        )

        # Determine the spatial grid dimensions
        grid_shape = data['x'].shape

        # Extract time-dependent variables
        time_dependent_vars = [
            'bed_level',
            'water_depth',
            'flow_velocity_x',
            'flow_velocity_y',
        ]

        for key in time_dependent_vars:
            var_name = variable_map.get(key, key)
            try:
                var = self._get_variable(var_name)
            except KeyError:
                # Default to zeros if not found
                data[key] = np.zeros((num_times, *grid_shape))
                print(f"Warning: Variable '{var_name}' not found, using zeros")
                continue

            # Check if variable has time dimension
            if 'time' in var.dims:
                # Check if variable has layer dimension
                if 'layer' in var.dims:
                    # For variables with time and layer, select layer 0 and apply time slice
                    data[key] = var.isel(layer=0, time=time_slice).values
                else:
                    # For variables with time but no layer, apply time slice
                    data[key] = var.isel(time=time_slice).values
            else:
                # For variables without time dimension, broadcast to all time steps
                data[key] = np.broadcast_to(var.values, (num_times, *var.shape))

        return data

    def _get_face_node_mesh_variables(self):
        """Return UGRID node coordinates and face-node connectivity with indexing metadata."""
        node_x_var = self._get_variable('mesh2d_node_x')
        node_y_var = self._get_variable('mesh2d_node_y')

        try:
            face_nodes_var = self._get_variable('mesh2d_face_nodes')
        except KeyError as err:
            if not isinstance(self.input_data, xu.UgridDataset):
                raise
            grid = self.input_data.grid
            if not isinstance(grid, xu.Ugrid2d):
                raise TypeError(f'Expected Ugrid2d, got {type(grid).__name__}') from err
            face_nodes_var = grid.face_node_connectivity
            return node_x_var, node_y_var, face_nodes_var, 0, -1

        start_index = face_nodes_var.attrs.get('start_index', 0)
        fill_value = face_nodes_var.encoding.get('_FillValue', face_nodes_var.attrs.get('_FillValue', -1))
        return node_x_var, node_y_var, face_nodes_var, start_index, fill_value

    def _calculate_time_slice(self, current_time, reading_interval, time_info):
        """Calculate time slice indices based on current time and reading interval."""

        # If no chunking parameters provided, load entire file
        if current_time is None or reading_interval is None:
            return None, None

        # If reading_interval is 0 or very large, load entire file
        if reading_interval <= 0 or reading_interval >= time_info['seconds_since_reference'][-1]:
            return None, None

        times_array = time_info['seconds_since_reference']

        # Find current time index
        current_idx = np.searchsorted(times_array, current_time)

        # Calculate chunk size based on reading interval and NetCDF timestep
        netcdf_timestep = times_array[1] - times_array[0] if len(times_array) > 1 else 1.0
        chunk_steps = max(10, int(reading_interval / netcdf_timestep))

        # Calculate start and end indices with some buffer
        start_idx = max(0, current_idx - chunk_steps // 4)
        end_idx = min(len(times_array), current_idx + chunk_steps)

        return start_idx, end_idx


def normalize_face_node_connectivity(mesh2d_face_nodes, start_index=1, fill_value=-999):
    """Return zero-based face-node connectivity with invalid entries set to -1."""
    faces = np.asarray(mesh2d_face_nodes)
    normalized = faces.astype(np.int64) - int(start_index)
    invalid = normalized < 0
    if fill_value is not None:
        invalid |= faces == fill_value
    normalized[invalid] = -1
    return normalized


def compute_face_centroids(mesh2d_node_x, mesh2d_node_y, mesh2d_face_nodes, start_index=1, fill_value=-999):
    """
    Compute face centroids from UGRID mesh node coordinates and face->node connectivity.

    Parameters
    ----------
    mesh2d_node_x : array-like, shape (nnode,)
    mesh2d_node_y : array-like, shape (nnode,)
    mesh2d_face_nodes : array-like, shape (nface, max_nodes_per_face)
        Node indices per face; may be padded with fill_value.
    start_index : int
        0 or 1 depending on file convention.
    fill_value : int
        Padding value in mesh2d_face_nodes.

    Returns
    -------
    face_x : np.ndarray, shape (nface,)
    face_y : np.ndarray, shape (nface,)
    """
    node_x = np.asarray(mesh2d_node_x)
    node_y = np.asarray(mesh2d_node_y)
    faces = normalize_face_node_connectivity(mesh2d_face_nodes, start_index=start_index, fill_value=fill_value)

    face_x = np.empty(faces.shape[0], dtype=np.float64)
    face_y = np.empty(faces.shape[0], dtype=np.float64)

    for i, face in enumerate(faces):
        # Drop padded indices and out-of-range
        valid = face[face >= 0]
        if valid.size == 0:
            face_x[i] = np.nan
            face_y[i] = np.nan
            continue
        face_x[i] = node_x[valid].mean()
        face_y[i] = node_y[valid].mean()

    return face_x, face_y
