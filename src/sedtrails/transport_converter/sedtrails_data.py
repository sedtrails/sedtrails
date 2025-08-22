import numpy as np
from typing import Dict
from dataclasses import dataclass
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

    def __post_init__(self):
        """Initialize container for dynamic physics fields and validate metadata."""
        # Validate metadata field
        self._validate_metadata()
        # TODO: do we also need to check that min max values are sensible? i.e. min <= max

        self._physics_fields: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {}

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
        """Check if a specific physics field exists."""
        return name in self._physics_fields

    def get_physics_fields(self) -> list:
        """Get list of available physics field names."""
        return list(self._physics_fields.keys())

    def has_physics_data(self) -> bool:
        """Check if any physics fields have been added."""
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

        # Core fields
        data = {
            'time': self.times[time_index],
            'reference_date': self.reference_date,
            'x': self.x,
            'y': self.y,
            'bed_level': self.bed_level,  # typically time-independent
            'fractions': self.fractions,
            'min_resolution': self.min_resolution,
            'outer_envelope': self.outer_envelope,
            'water_depth': self.water_depth[time_index],
            'mean_bed_shear_stress': self.mean_bed_shear_stress[time_index],
            'max_bed_shear_stress': self.max_bed_shear_stress[time_index],
            'sediment_concentration': self.sediment_concentration[time_index],
            'depth_avg_flow_velocity': {
                'x': self.depth_avg_flow_velocity['x'][time_index],
                'y': self.depth_avg_flow_velocity['y'][time_index],
                'magnitude': self.depth_avg_flow_velocity['magnitude'][time_index],
            },
            'bed_load_transport': {
                'x': self.bed_load_transport['x'][time_index],
                'y': self.bed_load_transport['y'][time_index],
                'magnitude': self.bed_load_transport['magnitude'][time_index],
            },
            'suspended_transport': {
                'x': self.suspended_transport['x'][time_index],
                'y': self.suspended_transport['y'][time_index],
                'magnitude': self.suspended_transport['magnitude'][time_index],
            },
            'nonlinear_wave_velocity': {
                'x': self.nonlinear_wave_velocity['x'][time_index],
                'y': self.nonlinear_wave_velocity['y'][time_index],
                'magnitude': self.nonlinear_wave_velocity['magnitude'][time_index],
            },
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
