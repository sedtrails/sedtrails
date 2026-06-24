"""A plugin for converting XBeach NetCDF mean output to SedTRAILS format."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay

from sedtrails.transport_converter.domain_mask import classify_boundary_edges_from_config
from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata
from sedtrails.transport_converter.time_utils import decompress_time_info


def delaunay_connectivity(node_x: np.ndarray, node_y: np.ndarray) -> np.ndarray:
    """Build triangular connectivity from flattened node coordinates."""
    x = np.asarray(node_x, dtype=float).ravel()
    y = np.asarray(node_y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f'node_x and node_y must have the same shape, got {x.shape} and {y.shape}')
    if x.size < 3:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(Delaunay(np.column_stack((x, y))).simplices, dtype=np.int64)


class FormatPlugin(BaseFormatPlugin):
    """
    Plugin for converting XBeach NetCDF mean variables to SedTRAILS format.

    XBeach writes instantaneous variables on ``globaltime`` and averaged
    variables on ``meantime``. This plugin deliberately maps only ``*_mean``
    variables and uses ``meantime`` as the SedTRAILS time coordinate.

    Parameters
    ----------
    input_file : str
        Path to the XBeach NetCDF file.
    morfac : float, default 1.0
        Morphological acceleration factor for time decompression.
    """

    TIME_DIM = 'meantime'
    FRACTION_DIM = 'sediment_classes'

    def __init__(self, input_file: str, morfac: float = 1.0):
        """
        Initialize an XBeach format plugin.

        Parameters
        ----------
        input_file : str
            Path to the XBeach NetCDF file.
        morfac : float, default 1.0
            Morphological acceleration factor for time decompression. The
            default is 1.0, which leaves input times unchanged.

        Raises
        ------
        FileNotFoundError
            If ``input_file`` does not exist.
        """
        super().__init__()
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')
        self.morfac = morfac
        self.input_data: xr.Dataset | None = None
        self._input_variables: List[str] = []
        self.domain_config: Dict[str, Any] = {}
        self._particle_connectivity_cache: dict[str, Any] | None = None
        self._boundary_edge_classification_cache: dict[str, Any] | None = None

    @property
    def variables(self) -> List[str]:
        """
        Get the variables in the input dataset.

        Returns
        -------
        list[str]
            List of variable names in the input dataset.

        Raises
        ------
        OSError
            If the input NetCDF file cannot be opened.
        ValueError
            If the loaded dataset does not expose data variables.
        """
        if self.input_data is None:
            self.load()

        if not self._input_variables:
            try:
                self._input_variables = list(self.input_data.data_vars)
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
        Convert XBeach NetCDF mean output to SedtrailsData.

        Parameters
        ----------
        current_time : float, optional
            Current simulation time in seconds.
        reading_interval : float, optional
            Reading interval in seconds.
        reference_date : np.datetime64, optional
            Reference date for converting time values.

        Returns
        -------
        SedtrailsData
            Converted SedTRAILS data. Spatial ``ny,nx`` fields are flattened
            to a single spatial axis. XBeach sediment transport components are
            summed over source ``sediment_classes`` before being stored.
            XBeach cutout cells whose ``globalx`` or ``globaly`` coordinate is
            not finite are omitted from the flattened spatial axis.

        Raises
        ------
        KeyError
            If a required XBeach coordinate, time coordinate, or ``*_mean``
            variable is missing.
        OSError
            If the input NetCDF file cannot be opened.
        TypeError
            If ``reference_date`` is not a ``numpy.datetime64``.
        ValueError
            If required variables have incompatible dimensions.
        """
        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)
        time_info = self._decompress_time(time_info)

        time_start_idx, time_end_idx = self._calculate_time_slice(current_time, reading_interval, time_info)
        if time_start_idx is not None or time_end_idx is not None:
            time_slice = slice(time_start_idx, time_end_idx)
            time_info = self._slice_time_info(time_info, time_slice)

        mapped_data = self._map_xbeach_variables(time_info, time_start_idx, time_end_idx)
        seconds_since_ref = time_info['seconds_since_reference']
        self.reference_date = time_info['reference_date']

        depth_avg_velocity_magnitude = np.sqrt(
            mapped_data['flow_velocity_x'] ** 2 + mapped_data['flow_velocity_y'] ** 2
        )
        bed_load_magnitude = np.sqrt(
            mapped_data['bed_load_transport_x'] ** 2 + mapped_data['bed_load_transport_y'] ** 2
        )
        suspended_transport_magnitude = np.sqrt(
            mapped_data['suspended_transport_x'] ** 2 + mapped_data['suspended_transport_y'] ** 2
        )

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
        nonlinear_wave_velocity = {
            'x': mapped_data['nonlinear_wave_velocity_x'],
            'y': mapped_data['nonlinear_wave_velocity_y'],
            'magnitude': np.sqrt(
                mapped_data['nonlinear_wave_velocity_x'] ** 2 + mapped_data['nonlinear_wave_velocity_y'] ** 2
            ),
        }

        metadata = SedtrailsMetadata(
            flowfield_domain={
                'x_min': np.nanmin(mapped_data['x']),
                'x_max': np.nanmax(mapped_data['x']),
                'y_min': np.nanmin(mapped_data['y']),
                'y_max': np.nanmax(mapped_data['y']),
            }
        )
        metadata.update(
            {
                'model': 'xbeach',
                'time_coordinate': self.TIME_DIM,
                'uses_mean_variables': True,
                'source_sediment_classes': mapped_data['source_sediment_classes'],
                'sediment_transport_fraction_handling': 'sum_over_source_sediment_classes',
                'max_bed_shear_stress_source': 'taubx_mean/tauby_mean magnitude',
            }
        )
        particle_connectivity = self._particle_face_connectivity(mapped_data['x'], mapped_data['y'])
        boundary_edge_classification = self._boundary_edge_classification(
            mapped_data['x'],
            mapped_data['y'],
            particle_connectivity,
        )
        if boundary_edge_classification is not None:
            metadata.add('boundary_edge_classification', boundary_edge_classification)

        return SedtrailsData(
            times=seconds_since_ref,
            reference_date=self.reference_date,
            x=mapped_data['x'],
            y=mapped_data['y'],
            bed_level=mapped_data['bed_level'],
            depth_avg_flow_velocity=depth_avg_flow_velocity,
            fractions=1,
            bed_load_transport=bed_load_transport,
            suspended_transport=suspended_transport,
            water_depth=mapped_data['water_depth'],
            mean_bed_shear_stress=mapped_data['mean_bed_shear_stress'],
            # XBeach sample output has taub*_mean but no *_mean maximum. Use the
            # mean magnitude so downstream physics can run without reading *_max.
            max_bed_shear_stress=mapped_data['mean_bed_shear_stress'],
            sediment_concentration=mapped_data['sediment_concentration'],
            nonlinear_wave_velocity=nonlinear_wave_velocity,
            node_x=mapped_data['x'],
            node_y=mapped_data['y'],
            face_node_connectivity=particle_connectivity,
            particle_face_connectivity=particle_connectivity,
            face_node_fill_value=-1,
            metadata=metadata,
        )

    def get_seeding_field_data(self):
        """
        Return XBeach particle-grid geometry required for seeding.

        Returns
        -------
        types.SimpleNamespace
            Object with flattened active XBeach cell-center coordinates,
            structured triangular connectivity for particle tracking, and
            ``face_node_fill_value``. XBeach cutout cells with non-finite
            coordinates are omitted.
        """
        self.load()
        x, y = self._get_grid_coordinates()
        particle_connectivity = self._particle_face_connectivity(x, y)
        boundary_edge_classification = self._boundary_edge_classification(x, y, particle_connectivity)
        return SimpleNamespace(
            x=x,
            y=y,
            face_node_connectivity=particle_connectivity,
            particle_face_connectivity=particle_connectivity,
            boundary_edge_classification=boundary_edge_classification,
            face_node_fill_value=-1,
        )

    def get_seeding_coordinates(self):
        """
        Return the spatial coordinates required for particle seeding.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Flattened active X and Y cell-center coordinates, each with shape
            ``(n_points,)``. XBeach cutout cells with non-finite coordinates
            are omitted.

        Raises
        ------
        KeyError
            If ``globalx`` or ``globaly`` is missing from the input dataset.
        OSError
            If the input NetCDF file cannot be opened.
        ValueError
            If ``globalx`` and ``globaly`` have incompatible shapes.
        """
        self.load()
        return self._get_grid_coordinates()

    def get_time_bounds(self, reference_date: Optional[np.datetime64] = None) -> tuple[float, float]:
        """
        Return input mean-time bounds in seconds since the reference date.

        Parameters
        ----------
        reference_date : np.datetime64, optional
            Reference date used to convert the input time coordinate to seconds.
            If omitted, the Unix epoch is used.

        Returns
        -------
        tuple[float, float]
            First and last input timestamps in seconds since `reference_date`.

        Raises
        ------
        KeyError
            If the XBeach ``meantime`` coordinate is missing.
        OSError
            If the input NetCDF file cannot be opened.
        TypeError
            If ``reference_date`` is not a ``numpy.datetime64``.
        ValueError
            If the input data contains no mean-time values.
        """
        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)
        time_info = self._decompress_time(time_info)
        times = np.asarray(time_info['seconds_since_reference'], dtype=float)
        if times.size == 0:
            raise ValueError('Input data contains no mean time values')
        return float(times[0]), float(times[-1])

    def load(self) -> Any:
        """
        Read and cache the XBeach NetCDF dataset.

        Returns
        -------
        xarray.Dataset
            The opened XBeach dataset. Repeated calls return the cached
            dataset.

        Raises
        ------
        OSError
            If the NetCDF file cannot be opened by xarray.
        """
        if self.input_data is None:
            try:
                self.input_data = xr.open_dataset(self.input_file, decode_times=False, decode_timedelta=False)
            except Exception as e:
                raise IOError(f'Failed to open XBeach NetCDF file: {e}') from e
            else:
                print(f'Successfully loaded (Xarray): {self.input_file}')
        return self.input_data

    def _decompress_time(self, time_info: Dict) -> Dict:
        """Apply morfac decompression to time values."""
        return decompress_time_info(time_info, self.morfac)

    def _get_time_info(self, input_data: xr.Dataset, reference_date: np.datetime64) -> Dict:
        """Get XBeach mean-time information in seconds since `reference_date`."""
        if not isinstance(reference_date, np.datetime64):
            raise TypeError('reference_date must be a numpy datetime64 object')
        if input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')
        if self.TIME_DIM not in input_data:
            raise KeyError(
                f"Required mean time coordinate '{self.TIME_DIM}' not found. "
                'XBeach conversion requires *_mean output on the meantime axis.'
            )

        time_var = input_data[self.TIME_DIM]
        raw_time_values = np.asarray(time_var.values)
        orig_units = time_var.attrs.get('units')
        orig_calendar = time_var.attrs.get('calendar', 'standard')
        valid_time_indices = np.arange(raw_time_values.size, dtype=np.int64)

        if np.issubdtype(raw_time_values.dtype, np.datetime64):
            valid_time_mask = ~np.isnat(raw_time_values)
            raw_time_values = raw_time_values[valid_time_mask]
            valid_time_indices = valid_time_indices[valid_time_mask]
            time_values = raw_time_values
            seconds_since_ref = np.array(
                [float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values],
                dtype=float,
            )
        else:
            fill_values = self._numeric_time_fill_values(time_var)
            valid_time_mask = self._valid_numeric_time_mask(raw_time_values, orig_units, fill_values)
            raw_time_values = raw_time_values[valid_time_mask]
            valid_time_indices = valid_time_indices[valid_time_mask]
            if orig_units and 'since' in orig_units:
                decoded = xr.coding.times.decode_cf_datetime(raw_time_values, orig_units, orig_calendar)
                try:
                    decoded = xr.coding.times.cftime_to_nptime(decoded)
                except Exception:
                    pass
                time_values = np.asarray(decoded)
                seconds_since_ref = np.array(
                    [float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values],
                    dtype=float,
                )
            else:
                seconds_since_ref = np.asarray(raw_time_values, dtype=float)
                time_values = np.array(
                    [
                        reference_date + np.timedelta64(int(round(seconds * 1_000_000)), 'us')
                        for seconds in seconds_since_ref
                    ]
                )

        if len(time_values) == 0:
            raise ValueError('Input data contains no valid mean time values')

        return {
            'time_values': time_values,
            'time_start': time_values[0],
            'time_end': time_values[-1],
            'original_units': orig_units,
            'original_calendar': orig_calendar,
            'seconds_since_reference': seconds_since_ref,
            'reference_date': reference_date,
            'num_times': len(time_values),
            'valid_time_indices': valid_time_indices,
        }

    @staticmethod
    def _numeric_time_fill_values(time_var: xr.DataArray) -> np.ndarray:
        """Return explicit numeric fill values declared on a time variable."""
        fill_values = []
        for metadata in (time_var.attrs, time_var.encoding):
            for key in ('_FillValue', 'missing_value'):
                value = metadata.get(key)
                if value is None:
                    continue
                try:
                    fill_values.extend(np.asarray(value, dtype=float).ravel())
                except (TypeError, ValueError):
                    continue
        return np.asarray(fill_values, dtype=float)

    @staticmethod
    def _valid_numeric_time_mask(
        raw_time_values: np.ndarray,
        units: str | None,
        fill_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return numeric time values that can be represented as timedeltas."""
        values = np.asarray(raw_time_values, dtype=float)
        seconds_scale = FormatPlugin._time_unit_seconds_scale(units)
        max_time_value = np.iinfo(np.int64).max / (1_000_000.0 * seconds_scale)
        valid = np.isfinite(values) & (np.abs(values) <= max_time_value)
        if fill_values is not None:
            for fill_value in np.asarray(fill_values, dtype=float).ravel():
                if np.isfinite(fill_value):
                    valid &= values != fill_value
        return valid

    @staticmethod
    def _time_unit_seconds_scale(units: str | None) -> float:
        """Return the seconds represented by one numeric time unit."""
        if not units:
            return 1.0
        unit = units.split('since', 1)[0].strip().lower()
        if not unit:
            return 1.0
        unit = unit.split()[0]
        return {
            's': 1.0,
            'sec': 1.0,
            'secs': 1.0,
            'second': 1.0,
            'seconds': 1.0,
            'm': 60.0,
            'min': 60.0,
            'mins': 60.0,
            'minute': 60.0,
            'minutes': 60.0,
            'h': 3600.0,
            'hr': 3600.0,
            'hrs': 3600.0,
            'hour': 3600.0,
            'hours': 3600.0,
            'd': 86400.0,
            'day': 86400.0,
            'days': 86400.0,
        }.get(unit, 1.0)

    def _slice_time_info(self, time_info: Dict, time_slice: slice) -> Dict:
        """Slice time info to specified range."""
        sliced_info = time_info.copy()
        sliced_info['time_values'] = time_info['time_values'][time_slice]
        sliced_info['seconds_since_reference'] = time_info['seconds_since_reference'][time_slice]
        if 'valid_time_indices' in time_info:
            sliced_info['valid_time_indices'] = time_info['valid_time_indices'][time_slice]
        sliced_info['num_times'] = len(sliced_info['time_values'])
        if len(sliced_info['time_values']) > 0:
            sliced_info['time_start'] = sliced_info['time_values'][0]
            sliced_info['time_end'] = sliced_info['time_values'][-1]
        return sliced_info

    def _map_xbeach_variables(
        self, time_info, time_start_idx: Optional[int] = None, time_end_idx: Optional[int] = None
    ) -> Dict:
        """
        Map XBeach ``*_mean`` variables to SedtrailsData fields.

        Spatial ``ny,nx`` dimensions are flattened to a single SedTRAILS
        spatial axis. XBeach sediment transport variables are summed over the
        source ``sediment_classes`` dimension before entering SedTRAILS.
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        num_times = time_info['num_times']
        time_selector = self._time_selector_from_indices(time_info.get('valid_time_indices'))
        if time_selector is None:
            time_selector = (
                slice(time_start_idx, time_end_idx)
                if time_start_idx is not None or time_end_idx is not None
                else slice(None)
            )

        x, y = self._get_grid_coordinates()
        grid_size = x.size

        bed_load_transport_x, bed_load_classes_x = self._mean_transport_component(
            'Subg_mean', time_selector, num_times, grid_size
        )
        bed_load_transport_y, bed_load_classes_y = self._mean_transport_component(
            'Svbg_mean', time_selector, num_times, grid_size
        )
        suspended_transport_x, suspended_classes_x = self._mean_transport_component(
            'Susg_mean', time_selector, num_times, grid_size
        )
        suspended_transport_y, suspended_classes_y = self._mean_transport_component(
            'Svsg_mean', time_selector, num_times, grid_size
        )
        source_sediment_classes = max(
            bed_load_classes_x,
            bed_load_classes_y,
            suspended_classes_x,
            suspended_classes_y,
        )

        data = {
            'x': x,
            'y': y,
            'bed_level': self._mean_scalar('zb_mean', time_selector, num_times, grid_size),
            'water_depth': self._mean_scalar('hh_mean', time_selector, num_times, grid_size),
            'flow_velocity_x': self._mean_scalar('ue_mean', time_selector, num_times, grid_size),
            'flow_velocity_y': self._mean_scalar('ve_mean', time_selector, num_times, grid_size),
            'sediment_concentration': self._mean_scalar('cctot_mean', time_selector, num_times, grid_size),
            'bed_load_transport_x': bed_load_transport_x,
            'bed_load_transport_y': bed_load_transport_y,
            'suspended_transport_x': suspended_transport_x,
            'suspended_transport_y': suspended_transport_y,
            'source_sediment_classes': source_sediment_classes,
        }

        taubx = self._mean_scalar('taubx_mean', time_selector, num_times, grid_size)
        tauby = self._mean_scalar('tauby_mean', time_selector, num_times, grid_size)
        data['mean_bed_shear_stress'] = np.sqrt(taubx**2 + tauby**2)

        if 'ua_mean' in self.input_data and 'thetamean_mean' in self.input_data:
            theta = self._mean_scalar('thetamean_mean', time_selector, num_times, grid_size)
            ua = self._mean_scalar('ua_mean', time_selector, num_times, grid_size)
            data['nonlinear_wave_velocity_x'] = ua * np.cos(np.deg2rad(theta))
            data['nonlinear_wave_velocity_y'] = ua * np.sin(np.deg2rad(theta))
        else:
            data['nonlinear_wave_velocity_x'] = np.zeros((num_times, grid_size), dtype=float)
            data['nonlinear_wave_velocity_y'] = np.zeros((num_times, grid_size), dtype=float)
            print("Warning: Variable 'ua_mean' not found, using zeros for nonlinear wave velocity")

        return data

    @staticmethod
    def _time_selector_from_indices(time_indices: np.ndarray | None) -> Any:
        """Return a slice for contiguous raw time indices, otherwise indices."""
        if time_indices is None:
            return None
        indices = np.asarray(time_indices, dtype=np.int64)
        if indices.size == 0:
            return indices
        if np.array_equal(indices, np.arange(indices[0], indices[-1] + 1, dtype=np.int64)):
            return slice(int(indices[0]), int(indices[-1]) + 1)
        return indices

    def _get_grid_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened active XBeach cell-center coordinates."""
        x_grid, y_grid = self._grid_coordinate_arrays()
        x = x_grid.ravel()
        y = y_grid.ravel()
        active_mask = self._active_mask_from_coordinates(x, y)
        return x[active_mask], y[active_mask]

    def _grid_coordinate_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return XBeach coordinate arrays with matching shapes."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')
        if 'globalx' not in self.input_data or 'globaly' not in self.input_data:
            raise KeyError("Required XBeach coordinates 'globalx' and/or 'globaly' not found")

        x_grid = np.asarray(self.input_data['globalx'].values, dtype=float)
        y_grid = np.asarray(self.input_data['globaly'].values, dtype=float)
        if x_grid.shape != y_grid.shape:
            raise ValueError(f'globalx and globaly must have the same shape, got {x_grid.shape} and {y_grid.shape}')
        return x_grid, y_grid

    def _grid_active_mask(self) -> np.ndarray | None:
        """Return the flattened finite-coordinate mask, or ``None`` when unavailable."""
        if self.input_data is None or 'globalx' not in self.input_data or 'globaly' not in self.input_data:
            return None
        x_grid, y_grid = self._grid_coordinate_arrays()
        return self._active_mask_from_coordinates(x_grid.ravel(), y_grid.ravel())

    @staticmethod
    def _active_mask_from_coordinates(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return points with finite XBeach coordinates."""
        x_array = np.asarray(x, dtype=float).ravel()
        y_array = np.asarray(y, dtype=float).ravel()
        if x_array.shape != y_array.shape:
            raise ValueError(f'globalx and globaly must have the same flattened shape, got {x_array.shape} and {y_array.shape}')
        return np.isfinite(x_array) & np.isfinite(y_array)

    def _particle_face_connectivity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return particle-tracking triangles for the active XBeach grid."""
        cache = self._particle_connectivity_cache
        if self._geometry_cache_matches(cache, x, y):
            return cache['connectivity']

        connectivity = self._structured_grid_connectivity()
        if connectivity is None:
            connectivity = delaunay_connectivity(x, y)
        self._particle_connectivity_cache = {
            'x': np.asarray(x),
            'y': np.asarray(y),
            'connectivity': connectivity,
        }
        return connectivity

    def _structured_grid_connectivity(self) -> np.ndarray | None:
        """Build active structured XBeach triangles, excluding cutout cells."""
        if self.input_data is None or 'globalx' not in self.input_data or 'globaly' not in self.input_data:
            return None

        x_grid, y_grid = self._grid_coordinate_arrays()
        x_shape = x_grid.shape
        y_shape = y_grid.shape
        if x_shape != y_shape or len(x_shape) != 2:
            return None

        ny, nx = x_shape
        if ny < 2 or nx < 2:
            return None

        active_mask = self._active_mask_from_coordinates(x_grid.ravel(), y_grid.ravel())
        index_map = np.full(active_mask.size, -1, dtype=np.int64)
        index_map[active_mask] = np.arange(np.count_nonzero(active_mask), dtype=np.int64)

        mapped = index_map.reshape(ny, nx)
        lower_left = mapped[:-1, :-1].ravel()
        lower_right = mapped[:-1, 1:].ravel()
        upper_left = mapped[1:, :-1].ravel()
        upper_right = mapped[1:, 1:].ravel()

        n_cells = lower_left.size
        candidate_triangles = np.empty((2 * n_cells, 3), dtype=np.int64)
        candidate_triangles[0::2, 0] = lower_left
        candidate_triangles[0::2, 1] = lower_right
        candidate_triangles[0::2, 2] = upper_right
        candidate_triangles[1::2, 0] = lower_left
        candidate_triangles[1::2, 1] = upper_right
        candidate_triangles[1::2, 2] = upper_left

        active_triangles = np.all(candidate_triangles >= 0, axis=1)
        if not np.any(active_triangles):
            return np.empty((0, 3), dtype=np.int64)
        return candidate_triangles[active_triangles]

    def _geometry_cache_matches(self, cache: dict[str, Any] | None, x: np.ndarray, y: np.ndarray) -> bool:
        """Return whether cached particle connectivity matches the active grid."""
        if cache is None:
            return False
        return self._arrays_equal(cache.get('x'), x) and self._arrays_equal(cache.get('y'), y)

    def _boundary_edge_classification(
        self,
        x: np.ndarray,
        y: np.ndarray,
        connectivity: np.ndarray | None,
    ) -> dict | None:
        """Return optional boundary-edge class metadata for active triangles."""
        if connectivity is None:
            return None

        cache = self._boundary_edge_classification_cache
        if (
            cache is not None
            and cache.get('domain_signature') == self._domain_config_signature()
            and self._arrays_equal(cache.get('x'), x)
            and self._arrays_equal(cache.get('y'), y)
            and self._arrays_equal(cache.get('connectivity'), connectivity)
        ):
            return cache['metadata']

        classification = classify_boundary_edges_from_config(
            x,
            y,
            connectivity,
            getattr(self, 'domain_config', {}),
        )
        metadata = None if classification is None else classification.to_metadata()
        self._boundary_edge_classification_cache = {
            'x': np.asarray(x),
            'y': np.asarray(y),
            'connectivity': np.asarray(connectivity),
            'domain_signature': self._domain_config_signature(),
            'metadata': metadata,
        }
        return metadata

    def _domain_config_signature(self) -> str:
        """Return a cache signature for configured domain polygons."""
        return repr(getattr(self, 'domain_config', {}) or {})

    @staticmethod
    def _arrays_equal(left: np.ndarray | None, right: np.ndarray) -> bool:
        """Return whether two arrays have equal shape and values."""
        if left is None:
            return False
        right_array = np.asarray(right)
        return left.shape == right_array.shape and np.array_equal(left, right_array)

    def _mean_scalar(self, var_name: str, time_selector: Any, num_times: int, grid_size: int) -> np.ndarray:
        """Read a scalar ``*_mean`` variable as ``(time, spatial)``."""
        var = self._require_mean_variable(var_name)
        if self.FRACTION_DIM in var.dims:
            raise ValueError(f"Expected scalar mean variable '{var_name}', found fraction dimension {var.dims}")
        return self._reshape_mean_spatial(self._select_time(var, time_selector), num_times, grid_size)

    def _mean_transport_component(
        self, var_name: str, time_selector: Any, num_times: int, grid_size: int
    ) -> tuple[np.ndarray, int]:
        """Read a transport ``*_mean`` variable as total ``(time, spatial)`` data."""
        var = self._require_mean_variable(var_name)
        var = self._select_time(var, time_selector)
        source_sediment_classes = int(var.sizes.get(self.FRACTION_DIM, 1))
        if self.FRACTION_DIM in var.dims:
            var = var.sum(dim=self.FRACTION_DIM)
        return self._reshape_mean_spatial(var, num_times, grid_size), source_sediment_classes

    def _reshape_mean_spatial(self, var: xr.DataArray, num_times: int, grid_size: int) -> np.ndarray:
        """Return a mean variable with flattened spatial dimensions."""
        if self.TIME_DIM in var.dims:
            spatial_dims = [dim for dim in var.dims if dim != self.TIME_DIM]
            var = var.transpose(self.TIME_DIM, *spatial_dims)
            values = np.asarray(var.values, dtype=float)
            return self._reshape_timed_spatial_values(values, num_times, grid_size, var.name or '<unnamed>')
        values = self._reshape_static_spatial_values(np.asarray(var.values, dtype=float), grid_size, var.name or '<unnamed>')
        return np.broadcast_to(values, (num_times, grid_size))

    def _reshape_timed_spatial_values(
        self,
        values: np.ndarray,
        num_times: int,
        grid_size: int,
        var_name: str,
    ) -> np.ndarray:
        """Flatten timed XBeach spatial values and apply the active grid mask."""
        if values.shape[0] != num_times:
            raise ValueError(f"Variable '{var_name}' has {values.shape[0]} time values, expected {num_times}")
        flat_values = values.reshape(num_times, -1)
        return self._apply_active_grid_mask(flat_values, grid_size, var_name)

    def _reshape_static_spatial_values(self, values: np.ndarray, grid_size: int, var_name: str) -> np.ndarray:
        """Flatten static XBeach spatial values and apply the active grid mask."""
        flat_values = np.asarray(values, dtype=float).reshape(1, -1)
        return self._apply_active_grid_mask(flat_values, grid_size, var_name)[0]

    def _apply_active_grid_mask(self, flat_values: np.ndarray, grid_size: int, var_name: str) -> np.ndarray:
        """Filter flattened spatial values to the finite-coordinate XBeach grid."""
        values = np.asarray(flat_values, dtype=float)
        active_mask = self._grid_active_mask()
        if active_mask is not None:
            if values.shape[1] == active_mask.size:
                values = values[:, active_mask]
            elif values.shape[1] != grid_size:
                raise ValueError(
                    f"Variable '{var_name}' has {values.shape[1]} spatial values, "
                    f"but XBeach coordinates expose {active_mask.size} raw points and {grid_size} active points"
                )

        if values.shape[1] != grid_size:
            raise ValueError(f"Variable '{var_name}' has {values.shape[1]} active spatial values, expected {grid_size}")
        return values

    def _require_mean_variable(self, var_name: str) -> xr.DataArray:
        """Return a required XBeach ``*_mean`` variable or raise a clear error."""
        if not var_name.endswith('_mean'):
            raise ValueError(f"XBeach plugin only reads *_mean variables, got '{var_name}'")
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')
        if var_name not in self.input_data:
            raise KeyError(f"Required XBeach mean variable '{var_name}' not found in dataset")
        return self.input_data[var_name]

    def _select_time(self, var: xr.DataArray, time_selector: Any) -> xr.DataArray:
        """Apply the XBeach mean-time slice when present."""
        if self.TIME_DIM in var.dims:
            return var.isel({self.TIME_DIM: time_selector})
        return var

    def _calculate_time_slice(self, current_time, reading_interval, time_info):
        """Calculate time slice indices based on current time and reading interval."""
        if current_time is None or reading_interval is None:
            return None, None
        if reading_interval <= 0 or reading_interval >= time_info['seconds_since_reference'][-1]:
            return None, None

        times_array = time_info['seconds_since_reference']
        current_idx = np.searchsorted(times_array, current_time)
        netcdf_timestep = times_array[1] - times_array[0] if len(times_array) > 1 else 1.0
        chunk_steps = max(10, int(reading_interval / netcdf_timestep))
        start_idx = max(0, current_idx - chunk_steps // 4)
        end_idx = min(len(times_array), current_idx + chunk_steps)
        return start_idx, end_idx
