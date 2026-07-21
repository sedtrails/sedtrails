import warnings
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


@dataclass
class SedtrailsData:
    """
    A data class for internally structuring SedTrails data.

    This class holds data for multiple time steps with time as the first dimension
    for time-dependent variables.

    Attributes:
    -----------

    times: np.ndarray
        Array of time values in seconds since reference_date
    reference_date: np.datetime64
        Reference date for the time values
    x: np.ndarray
        X-coordinates of the grid cells
    y: np.ndarray
        Y-coordinates of the grid cells
    bed_level: np.ndarray
        Bed level in meters (typically time-independent)
    depth_avg_flow_velocity: Dict[str, np.ndarray]
        Depth-averaged flow velocity components in m/s
        (keys: 'x', 'y', 'magnitude', each with time as first dimension)
    fractions: int
        Number of sediment fractions
    bed_load_transport: Dict[str, np.ndarray]
        Bed load sediment transport in kg/m/s
        (keys: 'x', 'y', 'magnitude', each with time as first dimension)
        Shape: (time, fractions, spatial) or (time, spatial)
    suspended_transport: Dict[str, np.ndarray]
        Suspended sediment transport in kg/m/s
        (keys: 'x', 'y', 'magnitude', each with time as first dimension)
        Shape: (time, fractions, spatial) or (time, spatial)
    water_depth: np.ndarray
        Water depth in meters (with time as first dimension)
    mean_bed_shear_stress: np.ndarray
        Mean bed shear stress in pascal (with time as first dimension)
    max_bed_shear_stress: np.ndarray
        Max bed shear stress in pascal (with time as first dimension)
    sediment_concentration: np.ndarray
        Suspended sediment concentration in kg/m^3 (with time as first dimension)
        Shape: (time, fractions, spatial) or (time, spatial)
    nonlinear_wave_velocity: Dict[str, np.ndarray]
        Nonlinear wave velocity in m/s
        (keys: 'x', 'y', 'magnitude', each with time as first dimension)
    metadata: SedtrailsMetadata
        Additional metadata for this dataset

    # === PHYSICS FIELDS (added by PhysicsConverter) ===
    # Note: All physics fields match the structure of transport data
    # Shape: (time, fractions, spatial) if fractions exist, else (time, spatial)
    shields_number: np.ndarray = None
        Shields parameter (dimensionless bed shear stress) [-]
    bed_load_layer_thickness: np.ndarray = None
        Representative thickness of bed load transport layer [m]
    suspended_layer_thickness: np.ndarray = None
        Representative thickness of suspended transport layer [m]
    mixing_layer_thickness: np.ndarray = None
        Mixing layer thickness [m]
    bed_load_velocity: Dict[str, np.ndarray] = None
        Bed load velocity components in m/s
        (keys: 'x', 'y', 'magnitude', each matching transport data structure)
    suspended_velocity: Dict[str, np.ndarray] = None
        Suspended sediment velocity components in m/s
        (keys: 'x', 'y', 'magnitude', each matching transport data structure)
    """

    times: np.ndarray
    reference_date: np.datetime64
    x: np.ndarray
    y: np.ndarray
    bed_level: np.ndarray
    depth_avg_flow_velocity: Dict[str, np.ndarray]
    fractions: int
    bed_load_transport: Dict[str, np.ndarray]
    suspended_transport: Dict[str, np.ndarray]
    water_depth: np.ndarray
    mean_bed_shear_stress: np.ndarray
    max_bed_shear_stress: np.ndarray
    sediment_concentration: np.ndarray
    nonlinear_wave_velocity: Dict[str, np.ndarray]
    metadata: SedtrailsMetadata
    node_x: np.ndarray | None = None
    node_y: np.ndarray | None = None
    face_node_connectivity: np.ndarray | None = None
    particle_face_connectivity: np.ndarray | None = None
    face_node_fill_value: int = -1

    def __post_init__(self):
        """Initialize container for dynamic physics fields and validate metadata."""

        self._normalize_optional_mesh_geometry()
        # Validate metadata field
        self._validate_metadata()
        # TODO: do we also need to check that min max values are sensible? i.e. min <= max
        self._calculate_timestep()
        self._compute_grid_metadata()
        self._physics_fields: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {}

    def _normalize_optional_mesh_geometry(self) -> None:
        """Normalize optional UGRID-style mesh geometry for dashboard consumers."""
        if self.node_x is not None:
            self.node_x = np.asarray(self.node_x)
        if self.node_y is not None:
            self.node_y = np.asarray(self.node_y)
        if self.face_node_connectivity is not None:
            self.face_node_connectivity = np.asarray(self.face_node_connectivity, dtype=np.int64)
        if self.particle_face_connectivity is not None:
            self.particle_face_connectivity = np.asarray(self.particle_face_connectivity, dtype=np.int64)

    def mesh_geometry(self) -> Dict[str, Any] | None:
        """
        Return optional face-node mesh geometry for visualization code.

        Returns
        -------
        Dict[str, Any] | None
            Dictionary containing the requested values.
        """
        if self.node_x is None or self.node_y is None or self.face_node_connectivity is None:
            return None

        return {
            'x': self.x,
            'y': self.y,
            'node_x': self.node_x,
            'node_y': self.node_y,
            'face_node_connectivity': self.face_node_connectivity,
            'face_node_fill_value': self.face_node_fill_value,
        }

    def _calculate_timestep(self):
        """Calculate median timestep and add to metadata."""

        if len(self.times) < 2:
            # Cannot calculate timestep with fewer than 2 time points
            timestep = None
        else:
            # Calculate median timestep, this helps ignore the weird startup timesteps
            all_timesteps = np.diff(self.times)
            timestep = float(np.median(np.diff(self.times)))

            # Optional: Add validation
            if timestep <= 0:
                warnings.warn(f'Calculated timestep is non-positive: {timestep}', stacklevel=1)

            # Check if we have timesteps deviating from the median
            tolerance = 1e-6
            deviations = np.abs(all_timesteps - timestep)
            deviating_indices = np.where(deviations > tolerance)[0]

            if len(deviating_indices) > 0:
                warnings.warn(
                    f'Found {len(deviating_indices)} timesteps deviating from median ({timestep:.6f}s)', stacklevel=2
                )

        self.metadata.add('timestep', timestep)

    def _compute_grid_metadata(self):
        """
        Compute grid metadata and add to metadata: minimum resolution and outer envelope.

        Computes:
        - min_resolution: minimum distance between any two grid points
        - outer_envelope: convex hull vertices of the grid points
        """

        # Stack coordinates for distance calculations
        coords = np.column_stack((self.x.flatten(), self.y.flatten()))

        # Remove invalid points (e.g., NaNs from mesh construction)
        valid_mask = np.isfinite(coords).all(axis=1)
        coords = coords[valid_mask]

        if coords.shape[0] < 2:
            self.metadata.add('min_resolution', None)
            self.metadata.add('outer_envelope', [])
            return

        # Drop duplicate points: repeated coordinates are not additional spatial resolution.
        unique_coords = np.unique(coords, axis=0)
        if unique_coords.shape[0] < 2:
            self.metadata.add('min_resolution', None)
            self.metadata.add('outer_envelope', [])
            return

        # Compute minimum resolution using nearest-neighbor search on unique points.
        tree = cKDTree(unique_coords)
        distances, _ = tree.query(unique_coords, k=2)
        nearest_distances = distances[:, 1]
        positive_distances = nearest_distances[nearest_distances > 0.0]

        if positive_distances.size == 0:
            min_resolution = None
        else:
            min_resolution = float(np.min(positive_distances))

        min_x = float(np.min(unique_coords[:, 0]))
        max_x = float(np.max(unique_coords[:, 0]))
        min_y = float(np.min(unique_coords[:, 1]))
        max_y = float(np.max(unique_coords[:, 1]))

        # Compute outer envelope using convex hull; degenerate grids have no 2D
        # hull, so use the bounding box directly instead of warning on expected input.
        if unique_coords.shape[0] < 3 or np.linalg.matrix_rank(unique_coords - unique_coords.mean(axis=0)) < 2:
            outer_envelope = [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]
        else:
            try:
                hull = ConvexHull(unique_coords)
                outer_envelope = unique_coords[hull.vertices].tolist()
            except Exception as e:
                warnings.warn(f'Convex hull failed ({e}); using bounding box instead.', stacklevel=1)
                outer_envelope = [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]

        # Add to metadata
        self.metadata.add('min_resolution', min_resolution)
        self.metadata.add('outer_envelope', outer_envelope)

    def _validate_metadata(self):
        """Validate that metadata field exists and is the correct type."""

        # Only check that metadata is the right type
        if not isinstance(self.metadata, SedtrailsMetadata):
            raise TypeError(f'metadata must be an instance of SedtrailsMetadata, got {type(self.metadata).__name__}')

    def add_physics_field(self, name: str, data):
        """
        Add a physics field to the data structure.

        Parameters
        ----------

        name : str
            Name of the physics field
        data : np.ndarray or dict
            Physics data (scalar array or dict with 'x', 'y', 'magnitude' for vectors)
        """

        self._physics_fields[name] = data
        setattr(self, name, data)

    def has_physics_field(self, name: str) -> bool:
        """
        Check if a specific physics field exists.

        Parameters
        ----------
        name : str
            Name of the requested object.

        Returns
        -------
        bool
            Boolean result of the check.
        """

        return name in self._physics_fields

    def get_physics_fields(self) -> list:
        """
        Get list of available physics field names.

        Returns
        -------
        list
            Computed value returned by the function.
        """

        return list(self._physics_fields.keys())

    def has_physics_data(self) -> bool:
        """
        Check if any physics fields have been added.

        Returns
        -------
        bool
            Boolean result of the check.
        """

        return len(self._physics_fields) > 0

    # ------------------------------------------------------------------
    # Time slicing
    # ------------------------------------------------------------------
    def __getitem__(self, time_index: int) -> Dict:
        """
        Get data for a specific time index.

        Parameters
        ----------

        time_index : int
            Time index to extract

        Returns
        -------

        Dict
            Dictionary containing all data for the specified time index
        """

        if time_index < 0 or time_index >= len(self.times):
            raise IndexError(f'Time index {time_index} out of bounds (0-{len(self.times) - 1})')

        # Core fields (only include optional fields when present)
        data = {
            'time': self.times[time_index],
            'reference_date': self.reference_date,
            'x': self.x,
            'y': self.y,
            'bed_level': self._get_time_slice_or_static(self.bed_level, time_index),
            'fractions': self.fractions,
        }

        if self.water_depth is not None:
            data['water_depth'] = self.water_depth[time_index]

        if self.mean_bed_shear_stress is not None:
            data['mean_bed_shear_stress'] = self.mean_bed_shear_stress[time_index]

        if self.max_bed_shear_stress is not None:
            data['max_bed_shear_stress'] = self.max_bed_shear_stress[time_index]

        if self.sediment_concentration is not None:
            data['sediment_concentration'] = self.sediment_concentration[time_index]

        if self.depth_avg_flow_velocity is not None:
            data['depth_avg_flow_velocity'] = {
                'x': self.depth_avg_flow_velocity['x'][time_index],
                'y': self.depth_avg_flow_velocity['y'][time_index],
                'magnitude': self.depth_avg_flow_velocity['magnitude'][time_index],
            }

        if self.bed_load_transport is not None:
            data['bed_load_transport'] = {
                'x': self.bed_load_transport['x'][time_index],
                'y': self.bed_load_transport['y'][time_index],
                'magnitude': self.bed_load_transport['magnitude'][time_index],
            }

        if self.suspended_transport is not None:
            data['suspended_transport'] = {
                'x': self.suspended_transport['x'][time_index],
                'y': self.suspended_transport['y'][time_index],
                'magnitude': self.suspended_transport['magnitude'][time_index],
            }

        if self.nonlinear_wave_velocity is not None:
            data['nonlinear_wave_velocity'] = {
                'x': self.nonlinear_wave_velocity['x'][time_index],
                'y': self.nonlinear_wave_velocity['y'][time_index],
                'magnitude': self.nonlinear_wave_velocity['magnitude'][time_index],
            }

        # Dynamic physics fields
        for name, value in self._physics_fields.items():
            if isinstance(value, dict):  # vector field
                data[name] = {
                    'x': value['x'][time_index],
                    'y': value['y'][time_index],
                    'magnitude': value['magnitude'][time_index],
                }
            else:  # scalar field
                data[name] = value[time_index]

        return data

    def _get_time_slice_or_static(self, value: np.ndarray, time_index: int) -> np.ndarray:
        """Return a time slice when the first axis matches times, otherwise static data."""
        if value is None:
            return value
        array = np.asarray(value)
        if array.ndim > 1 and array.shape[0] == len(self.times):
            return array[time_index]
        return value
