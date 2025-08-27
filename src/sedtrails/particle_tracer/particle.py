"""
Classes for representing internal data structures of the particle tracer.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Optional, Tuple, Dict, ClassVar


@dataclass
class Particle(ABC):
    """
    Base class for particles during a simulation time step.

    Attributes
    ----------

    id : int
        The unique identifier of the particle.
    x : float
        The x-coordinate of the particle in meters.
    y : float
        The y-coordinate of the particle in meters.

    release_time : int
        The time step at which the particle is released. A non-negative integer.
    is_mobile : bool
        Whether the particle can move in the current simulation step or not. Default is True.
    name : str
        A name for the particle. Optional.

    trace : dict
        A dictionary to store the trace of the particle, which can include various attributes
        such as position, velocity, and other relevant data during the simulation.
    """

    _id_counter: ClassVar[int] = 0  # Class variable to keep track IDs

    id: int = field(init=False)  # Unique identifier for the particle
    _x: float = field(init=False)  # initial position
    _y: float = field(init=False)  # initial position
    _release_time: str = field(init=False)  # release time of the particle
    _burial_depth: float = field(init=False)  # release time of the particle
    _is_mobile: bool = field(default=True)  # whether the particle is mobile or not
    name: Optional[str] = field(default='')  # name of the particle
    trace: Dict = field(default_factory=dict)  # trace of the particle

    def __post_init__(self):
        Particle._id_counter += 1
        self.id = Particle._id_counter  # Assign a unique ID to the particle

        self._position_id = 1  # Id for indexing positions in the trace
        if not isinstance(self.name, str):
            raise TypeError(f"Expected 'name' to be a string, got {type(self.name).__name__}")

    def add_position(self, position: Tuple) -> None:
        """
        Add a position to the particle's trace.

        Parameters
        ----------
        position : Tuple[float, float]
            A tuple containing the x and y coordinates of the particle's position.
        """
        if not isinstance(position, tuple) or len(position) != 2:
            raise TypeError("Expected 'position' to be a tuple of two floats (x, y)")

        self.trace[self._position_id] = position
        self._position_id += 1

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: int | float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected 'x' to be an integer or float, got {type(value).__name__}")
        self._x = value

    @property
    def burial_depth(self) -> float:
        return self._burial_depth

    @burial_depth.setter
    def burial_depth(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected 'burial_depth' to be an integer or float, got {type(value).__name__}")
        if value < 0:
            raise ValueError('Burial depth must be non-negative')
        self._burial_depth = value

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, value: int | float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected 'y' to be an integer or float, got {type(value).__name__}")
        self._y = value

    @property
    def release_time(self) -> str:
        return self._release_time

    @release_time.setter
    def release_time(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Expected 'release_time' to be a string, got {type(value).__name__}")
        self._release_time = value

    @release_time.getter
    def release_time(self) -> str:
        return self._release_time

    @property
    def is_mobile(self) -> bool:
        return self._is_mobile

    @is_mobile.setter
    def is_mobile(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Expected 'is_mobile' to be a boolean, got {type(value).__name__}")
        self._is_mobile = value

    @is_mobile.getter
    def is_mobile(self) -> bool:
        return self._is_mobile

    @abstractmethod
    def particle_velocity(self) -> float:
        """
        A method to compute the particle's velocity.
        This method should be implemented by each particle type.

        Returns
        -------
        float
            The velocity of the particle in meters per second.
        """
        pass


@dataclass
class PhysicalProperties:
    """
    Base class for particle physical properties.

    Attributes
    ----------
    density : float
        Material density in kg/m³
    diameter : float
        Particle diameter in meters
    """

    density: float
    diameter: float

    def __post_init__(self):
        """Validate physical property values."""
        if not isinstance(self.density, (int, float)) or self.density <= 0:
            raise ValueError(f'Density must be positive, got {self.density}')
        if not isinstance(self.diameter, (int, float)) or self.diameter <= 0:
            raise ValueError(f'Diameter must be positive, got {self.diameter}')


@dataclass
class Sand(Particle):
    """
    Class representing sand particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the sand particles.
    """

    physical_properties: PhysicalProperties = field(
        default_factory=lambda: PhysicalProperties(
            density=2650.0,  # kg/m³, typical sand density
            diameter=2e-4,  # m, typical sand diameter (0.2 mm)
        )
    )

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.physical_properties, PhysicalProperties):
            raise TypeError(f'Expected PhysicalProperties, got {type(self.physical_properties).__name__}')

    def particle_velocity(self) -> float:
        """
        A method to compute the velocity of a sand particle.

        Returns
        -------
        float
            The velocity of the particle in meters per second.
        """
        pass  # TODO: implement the velocity calculation for sand particles


@dataclass
class Mud(Particle):
    """
    Class representing mud particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the mud particles.
    """

    physical_properties: PhysicalProperties = field(
        default_factory=lambda: PhysicalProperties(
            density=2650.0,  # kg/m³, typical mud density
            diameter=2e-6,  # m, typical mud diameter (2 microns)
        )
    )

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.physical_properties, PhysicalProperties):
            raise TypeError(f'Expected PhysicalProperties, got {type(self.physical_properties).__name__}')

    def particle_velocity(self) -> float:
        """
        A method to compute the velocity of a mud particle.

        Returns
        -------
        float
            The velocity of the particle in meters per second.
        """
        pass  # TODO: implement the velocity calculation for mud particles


@dataclass
class Passive(Particle):
    """
    Class representing passive particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the passive particles.
    """

    physical_properties: PhysicalProperties = field(
        default_factory=lambda: PhysicalProperties(
            density=1000.0,  # kg/m³, water density
            diameter=1e-6,  # m, typical tracer size (1 micron)
        )
    )

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.physical_properties, PhysicalProperties):
            raise TypeError(f'Expected PhysicalProperties, got {type(self.physical_properties).__name__}')

    def particle_velocity(self) -> float:
        """
        A method to compute the velocity of a passive particle.

        Returns
        -------
        float
            The velocity of the particle in meters per second.
        """
        pass  # TODO: implement the velocity calculation for passive particles


@dataclass
class InterpolatedValue:
    """
    Class for storing interpolated values of a Particle.

    Attributes
    ----------
    bed_level : float
        The bed level of the particle in meters.
    bed_load_sediment : ndarray
        The bed load sediment transport of the particle in kg/m/s.
    suspended_sediment : float
        The suspended sediment transport of the particle in kg/m/s.
    sediment_concentration : float
        The suspended sediment concentration of the particle in kg/m^3.
    depth : float
        The water depth of the particle in meters. (fluid)
    mean_bed_shear_stress : float
        The mean bed shear stress of the particle in Pa. (fluid)
    max_bed_shear_stress : float
        The maximum bed shear stress of the particle in Pa. (fluid)
    wave_velocity : ndarray
        The non-linear wave velocity of the particle in m/s. (fluid)
    depth_avg_flow_velocity : float
        The flow velocity of the particle averaged over depth in m/s. (fluid)
    """

    bed_level: float
    bed_load_sediment: ndarray
    suspended_sediment: float
    sediment_concentration: float
    water_depth: float
    mean_bed_shear_stress: float
    max_bed_shear_stress: float
    wave_velocity: ndarray
    depth_avg_flow_velocity: float
