"""A plugin for converting Delft3D4 NetCDF format to SedTRAILS format.

By default, the map output from Delft3D-4 is written to binary `trim-*.dat` files.
At present it is not possible to directly read these files in with Python, so it is
better to instead write the map output as `*.nc` files. To enable `*.nc` output in
Delft3D-4, add the following lines to the `*.mdf` file:
```
FlNcdf= #map#  # (or #maphis# if you also want history output)
ncFormat=4
```
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata
from sedtrails.transport_converter.time_utils import decompress_time_info


class FormatPlugin(BaseFormatPlugin):
    """Plugin for converting Delft3D4 NetCDF format to SedTRAILS format."""

    def __init__(self, input_file: str, morfac: float = 1.0):
        """
        Initialize the plugin with the input file.

        Parameters
        ----------
        input_file : str
            Path to the Delft3D4 NetCDF file.
        morfac : float, optional
            Morphological acceleration factor for time decompression (default: 1.0)
        """
        super().__init__()
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')
        self.morfac = morfac
        self.sediment_fraction_index: Optional[int] = None
        self.sediment_fraction_name: Optional[str] = None
        self.sediment_fraction_labels: list[str] | None = None
        self.input_data: Optional[xr.Dataset] = None
        self._input_variables: List[str] = []

    @property
    def variables(self) -> List[str]:
        """
        Get the variables in the input dataset.

        Returns
        -------
        List
            List of variable names in the input dataset.
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
        self, current_time=None, reading_interval=None, reference_date: Optional[np.datetime64] = None
    ) -> SedtrailsData:
        """
        Delft3D4 NetCDF to SedtrailsData.

        Parameters
        ----------
        current_time : float, optional
            Current simulation time in seconds
        reading_interval : float, optional
            Reading interval in seconds
        reference_date : np.datetime64, optional
            Reference date for converting time values

        Returns
        -------
        SedtrailsData
            The converted SedtrailsData object.
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

        mapped_data = self._map_delft3d4_variables(time_info, time_start_idx, time_end_idx)
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

        mean_bed_shear_stress = np.sqrt(mapped_data['bed_shear_stress_x'] ** 2 + mapped_data['bed_shear_stress_y'] ** 2)

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
            'x': np.zeros_like(mapped_data['flow_velocity_x']),
            'y': np.zeros_like(mapped_data['flow_velocity_y']),
            'magnitude': np.zeros_like(depth_avg_velocity_magnitude),
        }

        metadata = SedtrailsMetadata(
            flowfield_domain={
                'x_min': np.min(mapped_data['x']),
                'x_max': np.max(mapped_data['x']),
                'y_min': np.min(mapped_data['y']),
                'y_max': np.max(mapped_data['y']),
            }
        )

        fractions = 1
        for candidate_name in ('bed_load_transport_x', 'suspended_transport_x', 'sediment_concentration'):
            candidate_values = mapped_data.get(candidate_name)
            if isinstance(candidate_values, np.ndarray) and candidate_values.ndim >= 3:
                fractions = int(candidate_values.shape[1])
                break

        if mapped_data.get('sediment_fraction_labels'):
            metadata.add('sediment_fraction_labels', mapped_data['sediment_fraction_labels'])

        return SedtrailsData(
            times=seconds_since_ref,
            reference_date=self.reference_date,
            x=mapped_data['x'],
            y=mapped_data['y'],
            bed_level=mapped_data['bed_level'],
            depth_avg_flow_velocity=depth_avg_flow_velocity,
            fractions=fractions,
            bed_load_transport=bed_load_transport,
            suspended_transport=suspended_transport,
            water_depth=mapped_data['water_depth'],
            mean_bed_shear_stress=mean_bed_shear_stress,
            max_bed_shear_stress=mapped_data['max_bed_shear_stress'],
            sediment_concentration=mapped_data['sediment_concentration'],
            nonlinear_wave_velocity=nonlinear_wave_velocity,
            metadata=metadata,
        )

    def get_seeding_coordinates(self):
        """Return active Delft3D4 face centers for particle seeding.

        Returns
        -------
        tuple of numpy.ndarray
            One-dimensional x and y coordinates of active finite map faces.

        Raises
        ------
        KeyError
            If the input does not provide XZ/YZ or XCOR/YCOR coordinates.
        """
        self.load()

        if 'XZ' in self.input_data and 'YZ' in self.input_data:
            x_vals = self.input_data['XZ'].values
            y_vals = self.input_data['YZ'].values
            x_values, y_values = self._flatten_xy(x_vals, y_vals)
            valid = self._valid_face_mask(x_vals, y_vals).reshape(-1)
            return x_values[valid], y_values[valid]
        if 'XCOR' in self.input_data and 'YCOR' in self.input_data:
            x_vals = self.input_data['XCOR'].values
            y_vals = self.input_data['YCOR'].values
            x_values, y_values = self._flatten_xy(x_vals, y_vals)
            valid = self._valid_face_mask(x_vals, y_vals).reshape(-1)
            return x_values[valid], y_values[valid]

        raise KeyError("Required variables 'XZ'/'YZ' or 'XCOR'/'YCOR' not found in dataset")

    def load(self) -> Any:
        """Load the Delft3D4 NetCDF input dataset with xarray.

        Returns
        -------
        None
            The loaded dataset is retained in ``input_data``.

        Raises
        ------
        OSError
            If xarray cannot open the configured input file.
        """

        if self.input_data is None:
            try:
                time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                self.input_data = xr.open_dataset(
                    self.input_file,
                    decode_times=time_coder,
                    decode_timedelta=True,
                )
            except TypeError:
                self.input_data = xr.open_dataset(
                    self.input_file,
                    decode_times=True,
                    decode_timedelta=True,
                )
            except Exception as e:
                raise IOError(f'Failed to open NetCDF file: {e}') from e
            else:
                print(f'Successfully loaded (Xarray): {self.input_file}')

    def _get_variable(self, name: str) -> xr.DataArray:
        """Return an xarray DataArray for variable `name`."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        if name in self.input_data:
            return self.input_data[name]
        raise KeyError(f"Variable '{name}' not found in dataset")

    @staticmethod
    def _decode_fraction_labels(raw_values: Any) -> list[str] | None:
        """Decode fixed-width character arrays into stripped fraction labels."""
        values = np.asarray(raw_values)
        if values.size == 0:
            return None

        if values.ndim == 1:
            labels = []
            for value in values.tolist():
                if isinstance(value, (bytes, bytearray, np.bytes_)):
                    label = value.decode('utf-8', errors='ignore')
                else:
                    label = str(value)
                label = label.strip().replace('\x00', '')
                if label:
                    labels.append(label)
            return labels or None

        labels = []
        for row in values:
            chars = []
            for value in np.asarray(row).ravel().tolist():
                if isinstance(value, (bytes, bytearray, np.bytes_)):
                    chars.append(value.decode('utf-8', errors='ignore'))
                else:
                    chars.append(str(value))
            label = ''.join(chars).replace('\x00', '').strip()
            if label:
                labels.append(label)
        return labels or None

    def _decompress_time(self, time_info: Dict) -> Dict:
        """Apply morfac decompression to time values."""
        if self.morfac == 1.0:
            return time_info
        return decompress_time_info(time_info, self.morfac)

    def _get_time_info(self, input_data: xr.Dataset, reference_date: np.datetime64) -> Dict:
        """
        Get and transform time information of a dataset.
        """

        if not isinstance(reference_date, np.datetime64):
            raise TypeError('reference_date must be a numpy datetime64 object')

        if input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        time_var = input_data['time']
        time_values = time_var.values
        time_start = time_values[0]
        time_end = time_values[-1]

        orig_units = time_var.attrs.get('units') or time_var.encoding.get('units')
        orig_calendar = time_var.attrs.get('calendar') or time_var.encoding.get('calendar', 'standard')

        if not np.issubdtype(np.asarray(time_values).dtype, np.datetime64):
            decoded = None
            if orig_units is not None:
                try:
                    decoded = xr.coding.times.decode_cf_datetime(time_values, orig_units, orig_calendar)
                except Exception:
                    decoded = None
            if decoded is None:
                try:
                    decoded = xr.coding.times.cftime_to_nptime(time_values)
                except Exception:
                    decoded = None
            if decoded is not None:
                time_values = decoded
            else:
                time_values = reference_date + np.asarray(time_values, dtype=float) * np.timedelta64(1, 's')

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

    def _resolve_fraction_index(self, var: xr.DataArray, dim: str) -> int:
        """Resolve the configured sediment fraction index for a given fraction dimension."""
        size = int(var.sizes.get(dim, 0))
        if size <= 0:
            raise ValueError(f"Invalid fraction dimension '{dim}' with size {size}")

        if self.sediment_fraction_name:
            candidate_labels = self.sediment_fraction_labels
            if dim in var.coords:
                candidate_labels = var.coords[dim].values
            elif candidate_labels is None and self.input_data is not None and dim in self.input_data.coords:
                candidate_labels = self.input_data[dim].values

            if candidate_labels is not None:
                if isinstance(candidate_labels, list):
                    available_labels = [str(label).strip() for label in candidate_labels]
                else:
                    available_labels = self._decode_fraction_labels(candidate_labels) or []
                normalized_labels = [label.lower() for label in available_labels]
                requested_name = str(self.sediment_fraction_name).strip().lower()
                if requested_name in normalized_labels:
                    return int(normalized_labels.index(requested_name))
                raise ValueError(
                    f"Configured sediment_fraction_name '{self.sediment_fraction_name}' was not found "
                    f"in dimension '{dim}'. Available values: {available_labels}"
                )

        if self.sediment_fraction_index is None:
            return 0

        try:
            index = int(self.sediment_fraction_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'Configured sediment_fraction_index must be an integer, got {self.sediment_fraction_index!r}'
            ) from exc

        if index < 0 or index >= size:
            raise ValueError(
                f"Configured sediment_fraction_index={index} is out of bounds for dimension '{dim}' with size {size}"
            )
        return index

    def _select_first_dims(self, var: xr.DataArray, *, select_fraction_dims: bool = True) -> xr.DataArray:
        """Select extra dimensions, optionally preserving LSED-like fraction dimensions."""
        selection = {}
        fraction_dims = {'LSED', 'LSEDTOT', 'LSTSCI'}
        for dim in [
            'KMAXOUT_RESTR',
            'KMAXOUT',
            'LSED',
            'LSEDTOT',
            'LSTSCI',
            'SIG_LYR',
            'SIG_INTF',
            'nlyr',
            'nlyrp1',
            'layer',
        ]:
            if dim in var.dims:
                if dim in fraction_dims and select_fraction_dims:
                    selection[dim] = self._resolve_fraction_index(var, dim)
                elif dim not in fraction_dims:
                    selection[dim] = 0
        if selection:
            var = var.isel(**selection)
        return var

    def _ensure_grid_shape(self, values: np.ndarray, grid_shape: tuple) -> np.ndarray:
        """Trim arrays to match the grid shape when possible."""
        if values.shape[-2:] == grid_shape:
            return values
        if len(values.shape) >= 2:
            return values[..., : grid_shape[0], : grid_shape[1]]
        return values

    def _flatten_xy(self, x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flatten 2D coordinate grids to 1D vectors."""
        if x_vals.ndim == 2 and y_vals.ndim == 2:
            return x_vals.reshape(-1), y_vals.reshape(-1)
        return x_vals, y_vals

    def _flatten_spatial(self, values: np.ndarray, grid_shape: tuple) -> np.ndarray:
        """Flatten structured spatial grids to 1D spatial vectors."""
        if values.ndim >= 2 and values.shape[-2:] == grid_shape:
            new_shape = values.shape[:-2] + (grid_shape[0] * grid_shape[1],)
            return values.reshape(new_shape)
        return values

    def _valid_face_mask(self, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        """Return a face mask that excludes inactive or non-finite map coordinates."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if valid.ndim != 2:
            return valid

        if 'KCS' in self.input_data:
            cell_status = self._select_first_dims(self.input_data['KCS'])
            cell_status = self._ensure_grid_shape(
                np.asarray(cell_status.values),
                x_values.shape,
            )
            valid &= cell_status > 0

        if not np.any(valid):
            raise ValueError('Input data contains no active finite cell centers')

        return valid

    def get_time_bounds(self, reference_date: Optional[np.datetime64] = None) -> tuple[float, float]:
        """
        Return input time bounds in seconds since the configured reference date.

        This avoids running a full ``convert`` call at startup when only the
        forcing time window is needed.

        Parameters
        ----------
        reference_date : np.datetime64, optional
            Reference date used to convert the input time coordinate to seconds.
            If omitted, the Unix epoch is used.

        Returns
        -------
        tuple of float
            First and last input timestamps, in seconds since ``reference_date``.

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

    def _read_values(
        self,
        variable_name: str,
        *,
        time_slice: slice,
        num_times: int,
        select_fraction_dims: bool = True,
    ) -> np.ndarray:
        """Read one Delft3D variable over the requested time window."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        variable = self._select_first_dims(
            self.input_data[variable_name],
            select_fraction_dims=select_fraction_dims,
        )
        if 'time' in variable.dims:
            return np.asarray(variable.isel(time=time_slice).values)
        return np.broadcast_to(variable.values, (num_times, *variable.shape))

    @staticmethod
    def _broadcast_mask(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Broadcast a time-varying U/V wet mask over optional fraction axes."""
        if mask.shape[-2:] != values.shape[-2:]:
            mask = mask[..., : values.shape[-2], : values.shape[-1]]

        missing_leading_dims = values.ndim - mask.ndim
        if missing_leading_dims < 0:
            raise ValueError('Delft3D wet-mask dimensions are incompatible with vector data')
        if missing_leading_dims:
            mask = mask.reshape(mask.shape[:-2] + (1,) * missing_leading_dims + mask.shape[-2:])
        return np.broadcast_to(mask, values.shape)

    @staticmethod
    def _center_staggered_component(
        values: np.ndarray,
        wet_mask: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """Wet-weight a staggered U or V component onto face centers.

        Delft3D stores the U component on xi-oriented faces and the V component
        on eta-oriented faces. A face center uses the local component and the
        immediately preceding component along the staggered axis. The first row
        or column is replicated, matching Delft3D's max(index - 1, 1)
        boundary treatment.
        """
        previous_values = np.roll(values, 1, axis=axis)
        previous_mask = np.roll(wet_mask, 1, axis=axis)
        first_index = [slice(None)] * values.ndim
        first_index[axis] = 0
        first_index = tuple(first_index)
        previous_values[first_index] = values[first_index]
        previous_mask[first_index] = wet_mask[first_index]

        current_contribution = np.where(wet_mask != 0, values, 0.0) * wet_mask
        previous_contribution = np.where(previous_mask != 0, previous_values, 0.0) * previous_mask
        denominator = np.maximum(1.0, wet_mask + previous_mask)
        return (current_contribution + previous_contribution) / denominator

    def _local_grid_angle(self, grid_shape: tuple[int, int]) -> np.ndarray:
        """Return the face-centered ALFAS angle in radians, or zero for rectilinear input."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        if 'ALFAS' not in self.input_data:
            print("Warning: Variable 'ALFAS' not found, assuming zero grid rotation")
            return np.zeros(grid_shape, dtype=float)

        angle = self._select_first_dims(self.input_data['ALFAS'])
        if 'time' in angle.dims:
            angle = angle.isel(time=0)
        angle_values = self._ensure_grid_shape(np.asarray(angle.values), grid_shape)
        return np.deg2rad(angle_values)

    def _map_vector_pair(
        self,
        *,
        u_variable: str,
        v_variable: str,
        grid_shape: tuple[int, int],
        time_slice: slice,
        num_times: int,
        select_fraction_dims: bool,
        u_wet_mask: np.ndarray | None = None,
        v_wet_mask: np.ndarray | None = None,
        cos_angle: np.ndarray | None = None,
        sin_angle: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center and rotate one Delft3D U/V vector pair into global x/y components."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        u_values = (
            self._read_values(
                u_variable,
                time_slice=time_slice,
                num_times=num_times,
                select_fraction_dims=select_fraction_dims,
            )
            if u_variable in self.input_data
            else None
        )
        v_values = (
            self._read_values(
                v_variable,
                time_slice=time_slice,
                num_times=num_times,
                select_fraction_dims=select_fraction_dims,
            )
            if v_variable in self.input_data
            else None
        )

        if u_values is None and v_values is None:
            print(f"Warning: Variables '{u_variable}' and '{v_variable}' not found, using zeros")
            zeros = np.zeros((num_times, *grid_shape))
            return zeros, zeros

        if u_values is None or v_values is None:
            missing_name = u_variable if u_values is None else v_variable
            raise KeyError(f"Vector pair '{u_variable}'/'{v_variable}' is incomplete: '{missing_name}' is missing")

        if u_values is not None:
            if u_wet_mask is not None:
                u_mask = u_wet_mask
            elif 'KFU' in self.input_data:
                u_mask = self._read_values('KFU', time_slice=time_slice, num_times=num_times)
            elif 'KCU' in self.input_data:
                u_mask = self._read_values('KCU', time_slice=time_slice, num_times=num_times)
            else:
                print("Warning: Variables 'KFU' and 'KCU' not found, assuming all U faces are wet")
                u_mask = np.ones(u_values.shape[:1] + u_values.shape[-2:])
            u_centered = self._center_staggered_component(
                u_values,
                self._broadcast_mask(u_mask, u_values),
                axis=-2,
            )
            u_centered = self._ensure_grid_shape(u_centered, grid_shape)
        else:
            u_centered = None

        if v_values is not None:
            if v_wet_mask is not None:
                v_mask = v_wet_mask
            elif 'KFV' in self.input_data:
                v_mask = self._read_values('KFV', time_slice=time_slice, num_times=num_times)
            elif 'KCV' in self.input_data:
                v_mask = self._read_values('KCV', time_slice=time_slice, num_times=num_times)
            else:
                print("Warning: Variables 'KFV' and 'KCV' not found, assuming all V faces are wet")
                v_mask = np.ones(v_values.shape[:1] + v_values.shape[-2:])
            v_centered = self._center_staggered_component(
                v_values,
                self._broadcast_mask(v_mask, v_values),
                axis=-1,
            )
            v_centered = self._ensure_grid_shape(v_centered, grid_shape)
        else:
            v_centered = None

        if u_centered is None:
            u_centered = np.zeros_like(v_centered)
        if v_centered is None:
            v_centered = np.zeros_like(u_centered)

        if cos_angle is None or sin_angle is None:
            angle = self._local_grid_angle(grid_shape)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

        broadcast_shape = (1,) * (u_centered.ndim - 2) + grid_shape
        cos_angle = cos_angle.reshape(broadcast_shape)
        sin_angle = sin_angle.reshape(broadcast_shape)
        return (
            u_centered * cos_angle - v_centered * sin_angle,
            u_centered * sin_angle + v_centered * cos_angle,
        )

    def _map_delft3d4_variables(
        self, time_info: Dict, time_start_idx: Optional[int] = None, time_end_idx: Optional[int] = None
    ) -> Dict:
        """Map Delft3D4 variables to SedtrailsData structure."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        num_times = time_info['num_times']
        time_slice = (
            slice(time_start_idx, time_end_idx)
            if time_start_idx is not None or time_end_idx is not None
            else slice(None)
        )

        data: Dict[str, np.ndarray] = {}
        sediment_keys = {'sediment_concentration'}
        fraction_labels = None

        if 'NAMCON' in self.input_data:
            fraction_labels = self._decode_fraction_labels(self.input_data['NAMCON'].values)
            self.sediment_fraction_labels = fraction_labels

        if 'XZ' in self.input_data and 'YZ' in self.input_data:
            data['x'] = self.input_data['XZ'].values
            data['y'] = self.input_data['YZ'].values
            grid_coordinate_dims = self.input_data['XZ'].dims
        elif 'XCOR' in self.input_data and 'YCOR' in self.input_data:
            data['x'] = self.input_data['XCOR'].values
            data['y'] = self.input_data['YCOR'].values
            grid_coordinate_dims = self.input_data['XCOR'].dims
        else:
            raise KeyError("Required variables 'XZ'/'YZ' or 'XCOR'/'YCOR' not found in dataset")

        grid_shape = data['x'].shape
        valid_face_mask = self._valid_face_mask(data['x'], data['y'])

        if 'DPS0' in self.input_data:
            bottom_depth_var = self.input_data['DPS0']
        elif 'DP0' in self.input_data:
            bottom_depth_var = self.input_data['DP0']
            bottom_depth_dims = tuple(bottom_depth_var.dims[-2:])
            bottom_depth_location = str(bottom_depth_var.attrs.get('location', '')).lower()
            is_node_centered = bottom_depth_location == 'node' or tuple(dim.upper() for dim in bottom_depth_dims) == (
                'MC',
                'NC',
            )
            has_non_face_location = bottom_depth_location not in {'', 'face'}
            if is_node_centered or has_non_face_location or bottom_depth_dims != tuple(grid_coordinate_dims):
                raise ValueError(
                    'DP0 must be face-located and align with selected map coordinates. Use face-centered DPS0.'
                )
        else:
            bottom_depth_var = None

        if bottom_depth_var is not None:
            bed_level_vals = self._select_first_dims(bottom_depth_var).values
            data['bed_level'] = -self._ensure_grid_shape(bed_level_vals, grid_shape)
        else:
            data['bed_level'] = np.zeros(grid_shape)
            print("Warning: Variables 'DPS0' and 'DP0' not found, using zeros for bed level")

        variable_map = {
            'water_depth': 'DPS',
            'max_bed_shear_stress': 'TAUMAX',
            'sediment_concentration': 'R1',
        }

        for key, var_name in variable_map.items():
            if var_name in self.input_data:
                select_fraction_dims = key not in sediment_keys
                var = self._select_first_dims(self.input_data[var_name], select_fraction_dims=select_fraction_dims)

                if key in sediment_keys and fraction_labels is None:
                    for fraction_dim in ('LSED', 'LSEDTOT', 'LSTSCI'):
                        if fraction_dim in self.input_data[var_name].dims:
                            if fraction_dim in self.input_data[var_name].coords:
                                fraction_labels = [
                                    str(label).strip()
                                    for label in np.asarray(
                                        self.input_data[var_name].coords[fraction_dim].values
                                    ).tolist()
                                ]
                            break

                if 'time' in var.dims:
                    values = var.isel(time=time_slice).values
                else:
                    values = np.broadcast_to(var.values, (num_times, *var.shape))

                data[key] = self._ensure_grid_shape(values, grid_shape)
            else:
                data[key] = np.zeros((num_times, *grid_shape))
                print(f"Warning: Variable '{var_name}' not found, using zeros")

        vector_pairs = {
            'flow_velocity': ('U1', 'V1', False),
            'bed_shear_stress': ('TAUKSI', 'TAUETA', False),
            'bed_load_transport': ('SBUU', 'SBVV', True),
            'suspended_transport': ('SSUU', 'SSVV', True),
        }
        has_u_components = any(pair[0] in self.input_data for pair in vector_pairs.values())
        has_v_components = any(pair[1] in self.input_data for pair in vector_pairs.values())
        if has_u_components or has_v_components:
            u_mask_variable = next(
                (candidate for candidate in ('KFU', 'KCU') if candidate in self.input_data),
                None,
            )
            v_mask_variable = next(
                (candidate for candidate in ('KFV', 'KCV') if candidate in self.input_data),
                None,
            )
            if has_u_components and u_mask_variable is None:
                print("Warning: Variables 'KFU' and 'KCU' not found, assuming all U faces are wet")
                u_wet_mask = np.ones((num_times, *grid_shape))
            elif u_mask_variable is not None:
                u_wet_mask = self._read_values(
                    u_mask_variable,
                    time_slice=time_slice,
                    num_times=num_times,
                )
            else:
                u_wet_mask = None
            if has_v_components and v_mask_variable is None:
                print("Warning: Variables 'KFV' and 'KCV' not found, assuming all V faces are wet")
                v_wet_mask = np.ones((num_times, *grid_shape))
            elif v_mask_variable is not None:
                v_wet_mask = self._read_values(
                    v_mask_variable,
                    time_slice=time_slice,
                    num_times=num_times,
                )
            else:
                v_wet_mask = None
            local_grid_angle = self._local_grid_angle(grid_shape)
            cos_angle = np.cos(local_grid_angle)
            sin_angle = np.sin(local_grid_angle)
        else:
            u_wet_mask = None
            v_wet_mask = None
            cos_angle = None
            sin_angle = None

        for key, (u_variable, v_variable, preserve_fraction_dims) in vector_pairs.items():
            data[f'{key}_x'], data[f'{key}_y'] = self._map_vector_pair(
                u_variable=u_variable,
                v_variable=v_variable,
                grid_shape=grid_shape,
                time_slice=time_slice,
                num_times=num_times,
                select_fraction_dims=not preserve_fraction_dims,
                u_wet_mask=u_wet_mask,
                v_wet_mask=v_wet_mask,
                cos_angle=cos_angle,
                sin_angle=sin_angle,
            )

            if preserve_fraction_dims and fraction_labels is None:
                for variable_name in (u_variable, v_variable):
                    if variable_name not in self.input_data:
                        continue
                    variable = self.input_data[variable_name]
                    for fraction_dim in ('LSED', 'LSEDTOT', 'LSTSCI'):
                        if fraction_dim in variable.dims and fraction_dim in variable.coords:
                            fraction_labels = [
                                str(label).strip()
                                for label in np.asarray(variable.coords[fraction_dim].values).tolist()
                            ]
                            break
                    if fraction_labels is not None:
                        break

        if 'water_depth' in data and np.all(data['water_depth'] == 0) and 'S1' in self.input_data:
            s1 = self._select_first_dims(self.input_data['S1'])
            if 'time' in s1.dims:
                s1_vals = s1.isel(time=time_slice).values
            else:
                s1_vals = np.broadcast_to(s1.values, (num_times, *s1.shape))
            data['water_depth'] = self._ensure_grid_shape(s1_vals - data['bed_level'], grid_shape)

        spatial_mask = valid_face_mask.reshape(-1)
        data['x'], data['y'] = self._flatten_xy(data['x'], data['y'])
        data['x'] = data['x'][spatial_mask]
        data['y'] = data['y'][spatial_mask]
        for key, values in list(data.items()):
            if key in {'x', 'y'}:
                continue
            flattened_values = self._flatten_spatial(values, grid_shape)
            if isinstance(flattened_values, np.ndarray) and flattened_values.shape[-1] == spatial_mask.size:
                data[key] = flattened_values[..., spatial_mask]
            else:
                data[key] = flattened_values

        if fraction_labels is not None:
            data['sediment_fraction_labels'] = fraction_labels

        return data
