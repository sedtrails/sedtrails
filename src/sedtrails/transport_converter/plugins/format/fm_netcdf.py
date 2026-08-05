"""A plugin for converting Delft3D Flexible Mesh NetCDF to SedTRAILS format."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
import xugrid as xu

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.domain_mask import (
    ConnectivityMaskResult,
    classify_boundary_edges_from_config,
    delaunay_connectivity,
    filter_connectivity_by_inner_polygons,
    inner_boundary_files_from_config,
    load_inner_boundary_polygons,
    triangulate_face_connectivity,
)
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata
from sedtrails.transport_converter.time_utils import decompress_time_info


class FormatPlugin(BaseFormatPlugin):
    """Convert Delft3D Flexible Mesh NetCDF data to SedTRAILS format.

    Parameters
    ----------
    input_file : str
        Path to the Delft3D Flexible Mesh NetCDF file.
    morfac : float, optional
        Morphological acceleration factor used to decompress model time.
    """

    def __init__(self, input_file: str, morfac: float = 1.0):
        """
        Initialize the plugin with the input file.

        Parameters:
        -----------
        input_file : str
            Path to the Delft3D Flexible Mesh NetCDF file.
        morfac : float, optional
            Morphological acceleration factor for time decompression (default: 1.0)
        """
        super().__init__()
        self.input_file = Path(input_file)
        self.morfac = morfac
        self.input_data = None  # holds Dataset after reading
        self._input_variables: List[str] = []
        self.domain_config: Dict[str, Any] = {}
        self._inner_boundary_polygons: list[np.ndarray] | None = None
        self._inner_boundary_polygons_signature: str | None = None
        self._last_inner_boundary_mask: ConnectivityMaskResult | None = None
        self._active_triangular_connectivity_cache: dict[str, Any] | None = None
        self._boundary_edge_classification_cache: dict[str, Any] | None = None

    def __post_init__(self):
        # Check if the input file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')

    @property
    def variables(self) -> List[str]:
        """
        Get the variables in the input dataset.

        Returns
        -------
        List
            List of variable names in the input dataset.
        """

        if self.input_data is None:  # Ensure input_data is loaded
            self.load()

        if not self._input_variables:  # Ensures variables are only loaded once
            try:
                self._input_variables = list(self.input_data.data_vars)
            except AttributeError as e:
                raise ValueError('Input data could not retrieve variables') from e
            else:
                print(f'Variables in {self.input_file}:')

        return self._input_variables

    def _decompress_time(self, time_info: Dict) -> Dict:
        """
        Apply morfac decompression to time values.

        Parameters:
        -----------
        time_info : Dict
            Original time information

        Returns:
        --------
        Dict
            Time information with decompressed time values
        """
        return decompress_time_info(time_info, self.morfac)

    def convert(
        self, current_time=None, reading_interval=None, reference_date: Optional[np.datetime64] = None
    ) -> SedtrailsData:
        """
        Delft3D from Flexible Mesh NetCDF.

        Parameters
        ----------
        current_time : float, optional
            Current simulation time in seconds
        reading_interval : float, optional
            Reading interval in seconds

        Returns
        -------
        SedtrailsData
            The converted SedtrailsData object.
        """

        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        # Read the NetCDF file
        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)

        # Apply morfac decompression to time before time slicing
        time_info = self._decompress_time(time_info)

        # Determine if we need to slice based on current_time and reading_interval
        time_start_idx, time_end_idx = self._calculate_time_slice(current_time, reading_interval, time_info)

        # Apply time slicing if needed
        if time_start_idx is not None or time_end_idx is not None:
            time_slice = slice(time_start_idx, time_end_idx)
            time_info = self._slice_time_info(time_info, time_slice)

        # Map the variables to SedtrailsData structure
        mapped_data = self._map_dfm_variables(time_info, time_start_idx, time_end_idx)
        seconds_since_ref = time_info['seconds_since_reference']
        self.reference_date = time_info['reference_date']

        # TODO: DFM slicing can introduce an extra leading singleton dimension; remove only that axis.
        for key, value in mapped_data.items():
            if isinstance(value, np.ndarray) and value.ndim > 2 and value.shape[0] == 1:
                mapped_data[key] = np.squeeze(value, axis=0)

        # Calculate magnitudes for vector quantities
        # Flow velocity magnitude
        depth_avg_velocity_magnitude = np.sqrt(
            mapped_data['flow_velocity_x'] ** 2 + mapped_data['flow_velocity_y'] ** 2
        )

        # Bed load magnitude
        bed_load_magnitude = np.sqrt(
            mapped_data['bed_load_transport_x'] ** 2 + mapped_data['bed_load_transport_y'] ** 2
        )

        # Suspended sediment magnitude
        suspended_transport_magnitude = np.sqrt(
            mapped_data['suspended_transport_x'] ** 2 + mapped_data['suspended_transport_y'] ** 2
        )

        # Create dictionaries for vector quantities
        depth_avg_flow_velocity = {
            'x': mapped_data['flow_velocity_x'],
            'y': mapped_data['flow_velocity_y'],
            'magnitude': depth_avg_velocity_magnitude,
        }

        bed_load_transport = {
            'x': mapped_data['bed_load_transport_x'],
            'y': mapped_data['bed_load_transport_y'],
            'magnitude': bed_load_magnitude,
        }

        suspended_transport = {
            'x': mapped_data['suspended_transport_x'],
            'y': mapped_data['suspended_transport_y'],
            'magnitude': suspended_transport_magnitude,
        }

        # Create nonlinear wave velocity dictionary with zeros
        # Using the same shape as other vector quantities
        nonlinear_wave_velocity = {
            'x': np.zeros_like(mapped_data['flow_velocity_x']),
            'y': np.zeros_like(mapped_data['flow_velocity_y']),
            'magnitude': np.zeros_like(depth_avg_velocity_magnitude),
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
            fractions=1,  # Default to 1 fraction
            bed_load_transport=bed_load_transport,
            suspended_transport=suspended_transport,
            water_depth=mapped_data['water_depth'],
            mean_bed_shear_stress=mapped_data['mean_bed_shear_stress'],
            max_bed_shear_stress=mapped_data['max_bed_shear_stress'],
            sediment_concentration=mapped_data['sediment_concentration'],
            nonlinear_wave_velocity=nonlinear_wave_velocity,
            node_x=mapped_data['x'],
            node_y=mapped_data['y'],
            face_node_connectivity=mapped_data.get('face_node_connectivity'),
            face_node_fill_value=-1,
            metadata=metadata,
        )

        self._add_inner_boundary_metadata(sedtrails_data.metadata)
        self._add_boundary_edge_metadata(
            sedtrails_data.metadata,
            sedtrails_data.x,
            sedtrails_data.y,
            sedtrails_data.face_node_connectivity,
        )

        return sedtrails_data

    def get_seeding_field_data(self):
        """Return active geometry required for particle seeding.

        Returns
        -------
        types.SimpleNamespace
            Object with ``x`` and ``y`` node coordinates, active triangular
            ``face_node_connectivity``, optional
            ``boundary_edge_classification`` metadata, and
            ``face_node_fill_value``.

        Raises
        ------
        KeyError
            If the required ``net_xcc`` or ``net_ycc`` variables are missing.
        FileNotFoundError
            If configured inner-boundary or boundary-class polygon files do
            not exist.
        ValueError
            If configured Tekal polygon blocks are malformed.
        """
        self.load()
        if 'net_xcc' not in self.input_data or 'net_ycc' not in self.input_data:
            raise KeyError("Required variables 'net_xcc' and/or 'net_ycc' not found in dataset")

        x = self.input_data['net_xcc'].values
        y = self.input_data['net_ycc'].values
        connectivity = self._active_triangular_connectivity(x, y)
        boundary_edge_classification = self._boundary_edge_classification(x, y, connectivity)
        return SimpleNamespace(
            x=x,
            y=y,
            face_node_connectivity=connectivity,
            boundary_edge_classification=boundary_edge_classification,
            face_node_fill_value=-1,
        )

    def get_seeding_coordinates(self):
        """
        Return only the spatial coordinates required for particle seeding.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X and Y coordinates used for particle seeding.
        """
        self.load()

        if 'net_xcc' not in self.input_data or 'net_ycc' not in self.input_data:
            raise KeyError("Required variables 'net_xcc' and/or 'net_ycc' not found in dataset")

        return self.input_data['net_xcc'].values, self.input_data['net_ycc'].values

    def get_max_exposure_depth_fields(self):
        """
        Compute per-node maximum erosion depth and maximum bed shear stress over the
        full dataset (all time steps), without loading every time step into memory.

        Returns
        -------
        max_erosion : np.ndarray, shape (n_nodes,)
            Maximum erosion depth per node [m]: max(bed_level_t0 - bed_level_t) over all t,
            clipped to >= 0.  Zero for nodes with a static bed level.
        max_bss : np.ndarray, shape (n_nodes,)
            Maximum bed shear stress per node over all time steps [N/m²].
        """
        self.load()

        bed_var = 'bedlevel'
        bss_var = 'max_bss_magnitude'

        if bed_var not in self.input_data:
            raise KeyError(f"Required variable '{bed_var}' not found in dataset")
        if bss_var not in self.input_data:
            raise KeyError(f"Required variable '{bss_var}' not found in dataset")

        bed = self.input_data[bed_var]
        bss = self.input_data[bss_var]

        if 'time' in bed.dims:
            bed_initial = bed.isel(time=0).values.astype(float)
            bed_min = bed.min(dim='time').values.astype(float)
            max_erosion = np.maximum(bed_initial - bed_min, 0.0)
        else:
            max_erosion = np.zeros(np.asarray(bed.values).shape, dtype=float)

        if 'time' in bss.dims:
            max_bss = bss.max(dim='time').values.astype(float)
        else:
            max_bss = np.asarray(bss.values, dtype=float)

        return max_erosion, max_bss

    def get_time_bounds(self, reference_date: Optional[np.datetime64] = None) -> tuple[float, float]:
        """
        Return input time bounds in seconds since the configured reference date.

        Parameters
        ----------
        reference_date : np.datetime64, optional
            Reference date used to convert the input time coordinate to seconds.
            If omitted, the Unix epoch is used.

        Returns
        -------
        tuple of float
            First and last input timestamps, in seconds since `reference_date`.

        Raises
        ------
        ValueError
            If the input data contains no time values.
        """
        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)
        time_info = self._decompress_time(time_info)
        times = np.asarray(time_info['seconds_since_reference'], dtype=float)
        if times.size == 0:
            raise ValueError('Input data contains no time values')
        return float(times[0]), float(times[-1])

    def _calculate_time_slice(self, current_time, reading_interval, time_info):
        """Calculate time slice indices based on current time and reading interval."""

        # If no chunking parameters provided, load entire file
        if current_time is None or reading_interval is None:
            return None, None

        times_array = np.asarray(time_info['seconds_since_reference'], dtype=float)
        if times_array.size == 0:
            return None, None

        forcing_span = times_array[-1] - times_array[0]

        # If reading_interval is 0 or spans the forcing window, load entire file.
        if reading_interval <= 0 or forcing_span <= 0 or reading_interval >= forcing_span:
            return None, None

        # Find current time index
        current_idx = np.searchsorted(times_array, current_time)

        # Calculate chunk size based on reading interval and NetCDF timestep
        netcdf_timestep = times_array[1] - times_array[0] if len(times_array) > 1 else 1.0
        chunk_steps = max(10, int(reading_interval / netcdf_timestep))

        # Calculate start and end indices with some buffer
        start_idx = max(0, current_idx - chunk_steps // 4)
        end_idx = min(len(times_array), current_idx + chunk_steps)

        return start_idx, end_idx

    def load(self) -> Any:
        """
        Reads and loads a Delft3D Flexible Mesh NetCDF file using xugrid.

        Returns
        -------
        Any
            Requested value.
        """

        if self.input_data is None:
            try:
                # First try using xugrid's open_dataset which handles UGRID conventions
                self.input_data = xu.open_dataset(self.input_file, decode_timedelta=True)
            except Exception as e:
                print(f'Could not open file with xugrid: {e} \n Trying with Xarray...')
                # Fallback to regular xarray
                try:
                    self.input_data = xr.open_dataset(self.input_file, decode_timedelta=True)
                except Exception as e:
                    raise IOError(f'Failed to open NetCDF file: {e}') from e

                else:
                    print(f'Successfully loaded (Xarray): {self.input_file}')
            else:
                print('Successfully loaded (Xugrid)', self.input_file)

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

        # Convert time values to seconds since reference_date
        seconds_since_ref = np.array([float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values])

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

    def _map_dfm_variables(
        self, time_info, time_start_idx: Optional[int] = None, time_end_idx: Optional[int] = None
    ) -> Dict:
        """
        Map Delft3D Flexible Mesh variables to SedtrailsData structure.

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

        # Variable mapping for DFM files
        variable_map = {
            'x': 'net_xcc',  # X-coordinates
            'y': 'net_ycc',  # Y-coordinates
            'bed_level': 'bedlevel',  # Bed level
            'water_depth': 'waterdepth',  # Water depth
            'flow_velocity_x': 'sea_water_x_velocity',  # X-component of flow velocity
            'flow_velocity_y': 'sea_water_y_velocity',  # Y-component of flow velocity
            'mean_bed_shear_stress': 'mean_bss_magnitude',  # Mean bed shear stress
            'max_bed_shear_stress': 'max_bss_magnitude',  # Max bed shear stress
            'bed_load_transport_x': 'bedload_x_comp',  # X-component of bed load sediment transport
            'bed_load_transport_y': 'bedload_y_comp',  # Y-component of bed load sediment transport
            'suspended_transport_x': 'susload_x_comp',  # X-component of suspended sediment transport
            'suspended_transport_y': 'susload_y_comp',  # Y-component of suspended sediment transport
            'sediment_concentration': 'suspended_sed_conc',  # Suspended sediment concentration
        }

        # Extract data from dataset
        data = {}

        # First, get spatial coordinates (typically not time-dependent)
        for key in ['x', 'y']:
            var_name = variable_map[key]
            if var_name in self.input_data:
                data[key] = self.input_data[var_name].values
            else:
                raise KeyError(f"Required variable '{var_name}' not found in dataset")

        data['face_node_connectivity'] = self._active_triangular_connectivity(data['x'], data['y'])

        # Determine the spatial grid dimensions
        grid_shape = data['x'].shape

        # Extract time-dependent variables
        time_dependent_vars = [
            'bed_level',
            'water_depth',
            'mean_bed_shear_stress',
            'max_bed_shear_stress',
            'sediment_concentration',
            'flow_velocity_x',
            'flow_velocity_y',
            'bed_load_transport_x',
            'bed_load_transport_y',
            'suspended_transport_x',
            'suspended_transport_y',
        ]

        for key in time_dependent_vars:
            var_name = variable_map[key]
            if var_name in self.input_data:
                var = self.input_data[var_name]

                # Check if variable has time dimension
                if 'time' in var.dims:
                    # Check if variable has layer dimension
                    if 'layer' in var.dims:
                        if key in {'flow_velocity_x', 'flow_velocity_y'} and var.sizes['layer'] > 1:
                            raise ValueError(
                                f"Variable '{var_name}' contains {var.sizes['layer']} vertical layers, but "
                                'SedTRAILS requires depth-averaged flow velocity. Provide a depth-averaged '
                                'velocity variable or preprocess the layers with thickness weighting.'
                            )
                        # For variables with time and layer, select layer 0 and apply time slice
                        data[key] = var.isel(layer=0, time=time_slice).values
                    else:
                        # For variables with time but no layer, apply time slice
                        data[key] = var.isel(time=time_slice).values
                else:
                    # For variables without time dimension, broadcast to all time steps
                    data[key] = np.broadcast_to(var.values, (num_times, *var.shape))
            else:
                # Default to zeros if not found
                data[key] = np.zeros((num_times, *grid_shape))
                print(f"Warning: Variable '{var_name}' not found, using zeros")

        return data

    def _active_triangular_connectivity(self, node_x: np.ndarray, node_y: np.ndarray) -> np.ndarray:
        cache = self._active_triangular_connectivity_cache
        if self._geometry_cache_matches(cache, node_x, node_y):
            self._last_inner_boundary_mask = cache['mask_result']
            return cache['triangles']

        source_connectivity = self._source_face_node_connectivity(node_count=np.asarray(node_x).size)
        if source_connectivity is None:
            candidate_connectivity = delaunay_connectivity(node_x, node_y)
        else:
            candidate_connectivity = source_connectivity

        mask_result = filter_connectivity_by_inner_polygons(
            node_x,
            node_y,
            candidate_connectivity,
            self._get_inner_boundary_polygons(),
        )
        self._last_inner_boundary_mask = mask_result
        triangles = triangulate_face_connectivity(mask_result.connectivity)
        self._active_triangular_connectivity_cache = {
            'node_x': np.asarray(node_x),
            'node_y': np.asarray(node_y),
            'domain_signature': self._domain_config_signature(),
            'mask_result': mask_result,
            'triangles': triangles,
        }
        return triangles

    def _get_inner_boundary_polygons(self) -> list[np.ndarray]:
        domain_signature = self._domain_config_signature()
        if self._inner_boundary_polygons is None or self._inner_boundary_polygons_signature != domain_signature:
            self._inner_boundary_polygons = load_inner_boundary_polygons(getattr(self, 'domain_config', {}))
            self._inner_boundary_polygons_signature = domain_signature
        return self._inner_boundary_polygons

    def _add_inner_boundary_metadata(self, metadata: SedtrailsMetadata) -> None:
        inner_files = inner_boundary_files_from_config(getattr(self, 'domain_config', {}))
        if not inner_files and not self._get_inner_boundary_polygons():
            return

        metadata.add('inner_boundary_pol_files', inner_files)
        metadata.add('inner_boundary_polygon_count', len(self._get_inner_boundary_polygons()))
        if self._last_inner_boundary_mask is not None:
            metadata.add('inner_boundary_masked_face_count', self._last_inner_boundary_mask.removed_count)
            metadata.add('inner_boundary_active_face_count', int(self._last_inner_boundary_mask.connectivity.shape[0]))

    def _add_boundary_edge_metadata(
        self,
        metadata: SedtrailsMetadata,
        node_x: np.ndarray,
        node_y: np.ndarray,
        connectivity: np.ndarray | None,
    ) -> None:
        if connectivity is None:
            return

        classification_metadata = self._boundary_edge_classification(node_x, node_y, connectivity)
        if classification_metadata is not None:
            metadata.add('boundary_edge_classification', classification_metadata)

    def _boundary_edge_classification(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        connectivity: np.ndarray | None,
    ) -> dict | None:
        if connectivity is None:
            return None

        cache = self._boundary_edge_classification_cache
        if self._geometry_cache_matches(cache, node_x, node_y, connectivity):
            return cache['metadata']

        classification = classify_boundary_edges_from_config(
            node_x,
            node_y,
            connectivity,
            getattr(self, 'domain_config', {}),
        )
        classification_metadata = None if classification is None else classification.to_metadata()
        self._boundary_edge_classification_cache = {
            'node_x': np.asarray(node_x),
            'node_y': np.asarray(node_y),
            'connectivity': np.asarray(connectivity),
            'domain_signature': self._domain_config_signature(),
            'metadata': classification_metadata,
        }
        return classification_metadata

    def _domain_config_signature(self) -> str:
        return repr(getattr(self, 'domain_config', {}) or {})

    def _geometry_cache_matches(
        self,
        cache: dict[str, Any] | None,
        node_x: np.ndarray,
        node_y: np.ndarray,
        connectivity: np.ndarray | None = None,
    ) -> bool:
        if cache is None or cache.get('domain_signature') != self._domain_config_signature():
            return False
        if not self._arrays_equal(cache.get('node_x'), node_x) or not self._arrays_equal(cache.get('node_y'), node_y):
            return False
        if connectivity is None:
            return True
        return self._arrays_equal(cache.get('connectivity'), connectivity)

    @staticmethod
    def _arrays_equal(left: np.ndarray | None, right: np.ndarray) -> bool:
        if left is None:
            return False
        right_array = np.asarray(right)
        return left.shape == right_array.shape and np.array_equal(left, right_array)

    def _source_face_node_connectivity(self, node_count: int) -> np.ndarray | None:
        for variable_name in _FACE_NODE_CONNECTIVITY_CANDIDATES:
            if variable_name in self.input_data:
                connectivity = self._normalize_face_node_connectivity(
                    self.input_data[variable_name],
                    node_count=node_count,
                    variable_name=variable_name,
                )
                if self._connectivity_compatible_with_points(connectivity, node_count):
                    return connectivity
        return None

    @staticmethod
    def _connectivity_compatible_with_points(connectivity: np.ndarray, node_count: int) -> bool:
        if connectivity.size == 0:
            return False

        valid = connectivity >= 0
        if not np.any(valid):
            return False
        if int(np.max(connectivity[valid])) >= node_count:
            return False
        return bool(np.all(np.count_nonzero(valid, axis=1) >= 3))

    @staticmethod
    def _normalize_face_node_connectivity(face_nodes_var, node_count: int, variable_name: str | None = None) -> np.ndarray:
        faces = np.asarray(face_nodes_var.values if hasattr(face_nodes_var, 'values') else face_nodes_var)
        if faces.ndim != 2:
            raise ValueError(f'Face-node connectivity must be 2-D, got shape {faces.shape}')

        if faces.shape[0] <= 8 and faces.shape[1] > 8:
            faces = faces.T

        fill_value = _variable_fill_value(face_nodes_var)
        valid_raw = faces.astype(np.float64)
        valid_mask = np.isfinite(valid_raw)
        if fill_value is not None:
            valid_mask &= valid_raw != float(fill_value)

        start_index = _variable_start_index(face_nodes_var)
        if start_index is None:
            if variable_name in {'NetElemNode', 'net_elem_node', 'net_element_node'}:
                start_index = 1
            else:
                valid_values = valid_raw[valid_mask]
                if valid_values.size and np.nanmin(valid_values) >= 1 and np.nanmax(valid_values) <= node_count:
                    start_index = 1
                else:
                    start_index = 0

        normalized = faces.astype(np.int64) - int(start_index)
        invalid = normalized < 0
        if fill_value is not None:
            invalid |= faces == fill_value
        normalized[invalid] = -1
        return normalized


_FACE_NODE_CONNECTIVITY_CANDIDATES = (
    'mesh2d_face_nodes',
    'Mesh2_face_nodes',
    'NetElemNode',
    'net_elem_node',
    'net_element_node',
)


def _variable_start_index(variable) -> int | None:
    attrs = getattr(variable, 'attrs', {}) or {}
    if 'start_index' in attrs:
        return int(attrs['start_index'])
    return None


def _variable_fill_value(variable):
    attrs = getattr(variable, 'attrs', {}) or {}
    encoding = getattr(variable, 'encoding', {}) or {}
    return encoding.get('_FillValue', attrs.get('_FillValue'))


if __name__ == '__main__':
    # Example usage
    input_file = '/sedtrails/sample-data/inlet_sedtrails.nc'
    plugin = FormatPlugin(input_file, morfac=3.0)

    plugin.load()

    print(plugin.variables)  # List available variables

    data = plugin.convert()  # Convert to SedtrailsData format
    print(data)
