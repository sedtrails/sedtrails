"""
Field Data Retriever

Retrieves field data from the transport converter or the simulation caching and state tracker.
Performs temporal interpolation to provide accurate field data at any time point for particle tracing.
"""

from typing import Dict, Tuple

import numpy as np

from sedtrails.transport_converter.format_converter import SedtrailsData


class FieldDataRetriever:
    """
    Retrieves and interpolates field data from SedtrailsData.

    This class provides methods to extract both vector fields (flow fields) and scalar fields
    at specific times, performing temporal interpolation as needed.
    """

    # Constants for interpolation weights
    MIN_WEIGHT = 0.0
    MAX_WEIGHT = 1.0

    def __init__(self, sedtrails_data: SedtrailsData, fraction_index: int = 0):
        """
        Initialize the FieldDataRetriever.

        Parameters:
        -----------
        sedtrails_data : SedtrailsData
            The SedtrailsData object containing the fields
        fraction_index : int, optional
            Which sediment fraction to use for multi-fraction data (default: 0)
        """
        self.sedtrails_data = sedtrails_data
        self.fraction_index = fraction_index
        self._flow_max_cache = {}

    def get_interpolation_indices(self, target_time: float) -> Tuple[int, int, float]:
        """
        Get indices for interpolation between two time steps.

        Parameters
        ----------
        target_time : float
            Target time in seconds since reference_date

        Returns
        -------
        Tuple[int, int, float]
            (lower_index, upper_index, weight) where weight is the interpolation factor [0-1]
        """
        times = self.sedtrails_data.times

        # Handle edge cases
        if target_time <= times[0]:
            return 0, 0, self.MIN_WEIGHT

        if target_time >= times[-1]:
            last_index = len(times) - 1
            return last_index, last_index, self.MAX_WEIGHT

        # Find the index of the last time that is less than or equal to the target time
        lower_index = np.searchsorted(times, target_time, side='right') - 1
        upper_index = lower_index + 1

        # Calculate the interpolation weight
        time_range = times[upper_index] - times[lower_index]

        # Avoid division by zero
        if time_range == 0:
            weight = self.MIN_WEIGHT
        else:
            weight = (target_time - times[lower_index]) / time_range

        return lower_index, upper_index, weight

    def _interpolate_linearly(self, lower_value: np.ndarray, upper_value: np.ndarray, weight: float) -> np.ndarray:
        """
        Perform linear interpolation between two values.

        Parameters:
        -----------
        lower_value : np.ndarray
            Value at the lower time index
        upper_value : np.ndarray
            Value at the upper time index
        weight : float
            Interpolation weight (between MIN_WEIGHT and MAX_WEIGHT)

        Returns:
        --------
        np.ndarray
            Interpolated value
        """
        # Linear interpolation: result = (1-w)*lower + w*upper
        return (1 - weight) * lower_value + weight * upper_value

    def _extract_fraction(self, field_data):
        """
        Helper function to handle fraction dimension in field data.

        Parameters:
        -----------
        field_data : dict or np.ndarray
            Field data that may contain fraction dimensions

        Returns:
        --------
        dict or np.ndarray
            Field data with fraction dimension handled appropriately
        """
        if isinstance(field_data, dict):
            # Vector field case (e.g., velocity with x, y, magnitude components)
            x_data = field_data['x']
            if len(x_data.shape) == 1:
                # Shape: (N) - no fraction dimension
                return field_data
            elif x_data.shape[0] == 1:
                # Shape: (1, N) - single fraction, squeeze out the fraction dimension
                return {'x': field_data['x'][0], 'y': field_data['y'][0], 'magnitude': field_data['magnitude'][0]}
            else:
                # Shape: (>1, N) - multiple fractions, select the specified fraction
                return {
                    'x': field_data['x'][self.fraction_index],
                    'y': field_data['y'][self.fraction_index],
                    'magnitude': field_data['magnitude'][self.fraction_index],
                }
        else:
            # Scalar field case
            if len(field_data.shape) == 1:
                # Shape: (N) - no fraction dimension
                return field_data
            elif field_data.shape[0] == 1:
                # Shape: (1, N) - single fraction, squeeze out the fraction dimension
                return field_data[0]
            else:
                # Shape: (>1, N) - multiple fractions, select the specified fraction
                return field_data[self.fraction_index]

    def get_flow_field(self, time: float, flow_field_name: str) -> Dict[str, np.ndarray]:
        """
        Get the flow field and coordinates at the specified time.

        This method performs temporal interpolation between the two closest time steps
        in the SedtrailsData object to obtain the flow field at the requested time.

        Parameters
        ----------
        time : float
            Time in seconds since the reference date of the SedtrailsData object
        flow_field_name : str
            Name of the flow field to retrieve (e.g., 'depth_avg_flow_velocity',
            'bed_load_velocity', 'suspended_velocity')

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing coordinates and flow components:
            - 'x': X-coordinates of the grid cells
            - 'y': Y-coordinates of the grid cells
            - 'u': X-component of the flow velocity
            - 'v': Y-component of the flow velocity
            - 'magnitude': Magnitude of the flow velocity
        """
        # Get indices for interpolation
        lower_index, upper_index, weight = self.get_interpolation_indices(time)

        # If time is exactly at a time step or outside the range, no interpolation needed
        if lower_index == upper_index:
            time_slice = self.sedtrails_data[lower_index]

            if flow_field_name not in time_slice:
                raise KeyError(
                    f"Flow field '{flow_field_name}' not found in SedtrailsData. "
                    f'Available fields: {list(time_slice.keys())}'
                )

            flow_field = time_slice[flow_field_name]
            flow_field = self._extract_fraction(flow_field)

            return {
                'x': self.sedtrails_data.x,
                'y': self.sedtrails_data.y,
                'u': flow_field['x'],
                'v': flow_field['y'],
                'magnitude': flow_field['magnitude'],
            }
        else:
            # Otherwise, perform linear interpolation between the two time steps
            lower_slice = self.sedtrails_data[lower_index]
            upper_slice = self.sedtrails_data[upper_index]

            if flow_field_name not in lower_slice or flow_field_name not in upper_slice:
                raise KeyError(
                    f"Flow field '{flow_field_name}' not found in SedtrailsData. "
                    f'Available fields: {list(lower_slice.keys())}'
                )

            lower_flow = lower_slice[flow_field_name]
            upper_flow = upper_slice[flow_field_name]
            lower_flow = self._extract_fraction(lower_flow)
            upper_flow = self._extract_fraction(upper_flow)

            # Use the interpolation function for velocity components
            flow_x = self._interpolate_linearly(lower_flow['x'], upper_flow['x'], weight)
            flow_y = self._interpolate_linearly(lower_flow['y'], upper_flow['y'], weight)
            flow_mag = self._interpolate_linearly(lower_flow['magnitude'], upper_flow['magnitude'], weight)

            return {
                'x': self.sedtrails_data.x,
                'y': self.sedtrails_data.y,
                'u': flow_x,
                'v': flow_y,
                'magnitude': flow_mag,
            }

    def get_flow_field_bounds(self, time: float, flow_field_name: str) -> Dict:
        """
        Return lower/upper time slices for a flow field without full-grid interpolation.

        Particle interpolation is linear in the nodal values, so callers with a
        small number of query points can interpolate each time slice at those
        points and blend the results by `weight`.

        Parameters
        ----------
        time : float
            Simulation time in seconds.
        flow_field_name : str
            Name of the vector flow field to retrieve.

        Returns
        -------
        Dict
            Dictionary containing the requested values.
        """
        lower_index, upper_index, weight = self.get_interpolation_indices(time)
        lower_slice = self.sedtrails_data[lower_index]

        if flow_field_name not in lower_slice:
            raise KeyError(
                f"Flow field '{flow_field_name}' not found in SedtrailsData. "
                f'Available fields: {list(lower_slice.keys())}'
            )

        lower_flow = self._extract_fraction(lower_slice[flow_field_name])
        if lower_index == upper_index:
            upper_flow = lower_flow
        else:
            upper_slice = self.sedtrails_data[upper_index]
            if flow_field_name not in upper_slice:
                raise KeyError(
                    f"Flow field '{flow_field_name}' not found in SedtrailsData. "
                    f'Available fields: {list(upper_slice.keys())}'
                )
            upper_flow = self._extract_fraction(upper_slice[flow_field_name])

        return {
            'x': self.sedtrails_data.x,
            'y': self.sedtrails_data.y,
            'lower': {
                'u': lower_flow['x'],
                'v': lower_flow['y'],
                'magnitude': lower_flow['magnitude'],
            },
            'upper': {
                'u': upper_flow['x'],
                'v': upper_flow['y'],
                'magnitude': upper_flow['magnitude'],
            },
            'weight': weight,
            'lower_index': lower_index,
            'upper_index': upper_index,
        }

    def get_flow_max_velocity_bound(self, time: float, flow_field_name: str) -> float:
        """
        Return a conservative max velocity bound for CFL without interpolating a full grid.

        The bound is `(1-w) * max(lower) + w * max(upper)`, which is greater
        than or equal to the exact max of the linearly interpolated magnitude
        field for `0 <= w <= 1`.

        Parameters
        ----------
        time : float
            Simulation time in seconds.
        flow_field_name : str
            Name of the vector flow field to retrieve.

        Returns
        -------
        float
            Floating-point result of the calculation.
        """
        lower_index, upper_index, weight = self.get_interpolation_indices(time)
        lower_max = self._get_flow_time_slice_max(flow_field_name, lower_index)
        if lower_index == upper_index:
            return lower_max

        upper_max = self._get_flow_time_slice_max(flow_field_name, upper_index)
        return (1.0 - weight) * lower_max + weight * upper_max

    def _get_flow_time_slice_max(self, flow_field_name: str, time_index: int) -> float:
        cache_key = (flow_field_name, time_index, self.fraction_index)
        if cache_key in self._flow_max_cache:
            return self._flow_max_cache[cache_key]

        time_slice = self.sedtrails_data[time_index]
        if flow_field_name not in time_slice:
            raise KeyError(
                f"Flow field '{flow_field_name}' not found in SedtrailsData. "
                f'Available fields: {list(time_slice.keys())}'
            )

        flow_field = self._extract_fraction(time_slice[flow_field_name])
        magnitude = flow_field['magnitude']
        max_velocity = 0.0 if np.isnan(magnitude).all() else float(np.nanmax(magnitude))
        self._flow_max_cache[cache_key] = max_velocity
        return max_velocity

    def get_scalar_field(self, time: float, scalar_field_name: str) -> Dict[str, np.ndarray]:
        """
        Get a scalar field and coordinates at the specified time.

        This method performs temporal interpolation between the two closest time steps
        in the SedtrailsData object to obtain the scalar field at the requested time.

        Parameters
        ----------
        time : float
            Time in seconds since the reference date of the SedtrailsData object
        scalar_field_name : str
            Name of the scalar field to retrieve (e.g., 'water_depth', 'shields_number',
            'bed_load_layer_thickness', 'suspended_layer_thickness', 'mixing_layer_thickness',
            'mean_bed_shear_stress', 'max_bed_shear_stress', 'sediment_concentration')

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing coordinates and scalar field:
            - 'x': X-coordinates of the grid cells
            - 'y': Y-coordinates of the grid cells
            - 'magnitude': Scalar field values
        """
        # Get indices for interpolation
        lower_index, upper_index, weight = self.get_interpolation_indices(time)

        # If time is exactly at a time step or outside the range, no interpolation needed
        if lower_index == upper_index:
            time_slice = self.sedtrails_data[lower_index]

            if scalar_field_name not in time_slice:
                raise KeyError(
                    f"Scalar field '{scalar_field_name}' not found in SedtrailsData. "
                    f'Available fields: {list(time_slice.keys())}'
                )

            scalar_field = time_slice[scalar_field_name]
            scalar_field = self._extract_fraction(scalar_field)

            return {'x': self.sedtrails_data.x, 'y': self.sedtrails_data.y, 'magnitude': scalar_field}
        else:
            # Otherwise, perform linear interpolation between the two time steps
            lower_slice = self.sedtrails_data[lower_index]
            upper_slice = self.sedtrails_data[upper_index]

            if scalar_field_name not in lower_slice or scalar_field_name not in upper_slice:
                raise KeyError(
                    f"Scalar field '{scalar_field_name}' not found in SedtrailsData. "
                    f'Available fields: {list(lower_slice.keys())}'
                )

            lower_scalar = lower_slice[scalar_field_name]
            upper_scalar = upper_slice[scalar_field_name]
            lower_scalar = self._extract_fraction(lower_scalar)
            upper_scalar = self._extract_fraction(upper_scalar)

            # Use the interpolation function for scalar field
            scalar_interpolated = self._interpolate_linearly(lower_scalar, upper_scalar, weight)

            return {'x': self.sedtrails_data.x, 'y': self.sedtrails_data.y, 'magnitude': scalar_interpolated}

    def get_scalar_field_bounds(self, time: float, scalar_field_name: str) -> Dict:
        """
        Return lower/upper time slices for a scalar field without full-grid interpolation.

        Parameters
        ----------
        time : float
            Simulation time in seconds.
        scalar_field_name : str
            Name of the scalar field to retrieve.

        Returns
        -------
        Dict
            Dictionary containing the requested values.
        """
        lower_index, upper_index, weight = self.get_interpolation_indices(time)
        lower_slice = self.sedtrails_data[lower_index]

        if scalar_field_name not in lower_slice:
            raise KeyError(
                f"Scalar field '{scalar_field_name}' not found in SedtrailsData. "
                f'Available fields: {list(lower_slice.keys())}'
            )

        lower_scalar = self._extract_fraction(lower_slice[scalar_field_name])
        if lower_index == upper_index:
            upper_scalar = lower_scalar
        else:
            upper_slice = self.sedtrails_data[upper_index]
            if scalar_field_name not in upper_slice:
                raise KeyError(
                    f"Scalar field '{scalar_field_name}' not found in SedtrailsData. "
                    f'Available fields: {list(upper_slice.keys())}'
                )
            upper_scalar = self._extract_fraction(upper_slice[scalar_field_name])

        return {
            'x': self.sedtrails_data.x,
            'y': self.sedtrails_data.y,
            'lower': lower_scalar,
            'upper': upper_scalar,
            'weight': weight,
            'lower_index': lower_index,
            'upper_index': upper_index,
        }


# Note: The example code has been moved to examples/data_retriever_example.py
if __name__ == '__main__':
    print('Please see the examples directory for usage examples.')
