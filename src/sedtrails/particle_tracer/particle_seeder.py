"""
Particle Seeding Tool
=====================
Manage the creation of particles, their positions (x,y) and distribution.
Manage the creation of particles, their positions (x,y) and distribution.
using various release strategies.
Seeding strategies for positions include:
- Point: Release particles at a specific locations (x,y).
- Regular Grid: Release particles in a regular grid pattern based
    on distances between particles in x and y directions, and the
    simulation. A mask can be applied to restrict the area of seeding.
- Transect: release particle along line segments  defined by two points(x1,y1) and (x2,y2).
- Random: Release particles at random locations (x,y) within an area
    constrained by a bounding box (xmin, xmax, ymin, ymax).
- Point: Release particles at a specific locations (x,y).
- Regular Grid: Release particles in a regular grid pattern based
    on distances between particles in x and y directions, and the
    simulation. A mask can be applied to restrict the area of seeding.
- Transect: release particle along line segments  defined by two points(x1,y1) and (x2,y2).
- Random: Release particles at random locations (x,y) within an area
    constrained by a bounding box (xmin, xmax, ymin, ymax).
"""

import random
from abc import ABC, abstractmethod
from sedtrails.particle_tracer.particle import Particle
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.exceptions import MissingConfigurationParameter
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from sedtrails.configuration_interface.find import find_value
from numpy import ndarray
from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator
import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull


@dataclass
class PopulationConfig:
    """
    A class to represent the seeding parameters of a population of particle.
    A population is a group of particles that share the same type and seeding strategy.

    Attributes
    ----------
    population_config : Dict
        The configuration dictionary containing the seeding paraameters for a population.
    particle_type : str
        The type of particles to be seeded (e.g., 'sand', 'mud', 'passive').
    release_start : str
        The time at which the particles for a given population are released.
    quantity : int
        The number of particles to release per release location.
    strategy_settings : Dict
        The settings for the seeding strategy, extracted from the configuration.
        These are any key-value pairs defined under the specific strategy in the configuration.
    A class to represent the seeding parameters of a population of particle.
    A population is a group of particles that share the same type and seeding strategy.

    """

    population_config: Dict  # configuration for a single population
    strategy: str = field(init=False)
    particle_type: str = field(init=False)
    release_start: str = field(init=False)  # particle for a given population are released at this time
    quantity: int = field(init=False)  # number of particles to release per release location
    burial_depth: float = field(init=False, default=0.0)  # burial depth of the particles
    strategy_settings: Dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        _strategy = find_value(self.population_config, 'seeding.strategy', {}).keys()
        if not _strategy:
            raise MissingConfigurationParameter('"strategy" is not defined as seeding parameter.')
        self.strategy = next(iter(_strategy))
        self.strategy_settings = find_value(self.population_config, f'seeding.strategy.{self.strategy}', {})
        if not self.strategy_settings:
            raise MissingConfigurationParameter(f'"{self.strategy}" settings are not defined in the configuration.')
        _quantity = find_value(self.population_config, 'seeding.quantity', {})
        if not _quantity:
            raise MissingConfigurationParameter('"quantity" is not defined as seeding parameter.')
        self.quantity = _quantity
        _release_start = find_value(self.population_config, 'seeding.release_start', {})
        if not _release_start:
            raise MissingConfigurationParameter('"release_start" is not defined in the population configuration.')
        self.release_start = _release_start
        self.particle_type = find_value(self.population_config, 'particle_type', '')
        if not self.particle_type:
            raise MissingConfigurationParameter('"particle_type" is not defined in the population configuration.')
        _burial_depth = find_value(self.population_config, 'seeding.burial_depth', {})
        if not _burial_depth:
            raise MissingConfigurationParameter('"burial_depth" is not defined in the population configuration.')
        self.burial_depth = _burial_depth.get('constant', 0.0)  # TODO: support other types of burial depth


class SeedingStrategy(ABC):
    """
    Abstract base class for seeding strategies.
    """

    @abstractmethod
    def seed(self, config: PopulationConfig) -> List[Tuple[int, float, float]]:
        """
        Asociates quantity of particles to a seeding locations for a given strategy.


        Parameters
        ----------
        config : PopulationConfig
            Configuration object containing the seeding parameters.

        Returns
        -------
        list[Tuple[int, float, float]]
            A list of tuples where each tuple contains:
            - int: The quantity of particles to  be releases at a location.
            - float: The x-coordinate of the release location.
            - float: The y-coordinate of the release location.
        """
        pass


class PointStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles at specific locations (x,y).
    """

    def seed(self, config: PopulationConfig) -> list[Tuple[int, float, float]]:
        locations = getattr(config, 'strategy_settings', {}).get('locations', [])
        if not locations:
            raise MissingConfigurationParameter('"locations" must be provided for PointStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for PointStrategy.')
        quantity = int(config.quantity)
        seed_locations = []
        for loc_str in locations:
            try:
                x_str, y_str = loc_str.split(',')
                x = float(x_str.strip())
                y = float(y_str.strip())
                seed_locations.append((quantity, x, y))
            except Exception as e:
                raise ValueError(f"Invalid location string '{loc_str}': {e}") from e
        return seed_locations


class RandomStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles at at random locations (x,y) within an area constraint by a
    bounding box 'xmin,ymin xmax,ymax'.
    """

    def seed(self, config: PopulationConfig) -> list[Tuple[int, float, float]]:
        # expects strategy_settings to contain 'bbox' and 'seed'
        bbox = getattr(config, 'strategy_settings', {}).get('bbox', None)
        if not bbox:
            raise MissingConfigurationParameter('"bbox" must be provided for RandomStrategy.')

        seed = getattr(config, 'strategy_settings', {}).get('seed', None)
        if not seed:
            raise MissingConfigurationParameter('"seed" must be provided for RandomStrategy.')
        random.seed(seed)

        nlocations = getattr(config, 'strategy_settings', {}).get('nlocations', None)
        if not nlocations:
            raise MissingConfigurationParameter('"nlocations" must be provided for RandomStrategy.')

        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for RandomStrategy.')
        quantity = int(config.quantity)
        seed_locations = []

        _bbox = bbox.replace(',', ' ').split()  # separates values with whitespaces. Order is xmin, ymin, xmax, ymax
        for _ in range(nlocations):
            x = random.uniform(float(_bbox[0]), float(_bbox[2]))
            y = random.uniform(float(_bbox[1]), float(_bbox[3]))
            seed_locations.append((quantity, x, y))
        return seed_locations


class GridStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles that follows regular grid pattern.
    The grid is defined by the distance between particles (dx, dy) and the simulation domain size.
    If dx and dy have the same value, a square grid is created.
    The origin of the grid is at the bottom left corner of the bounding box
    """

    def seed(self, config: PopulationConfig) -> list[Tuple[int, float, float]]:
        bbox = config.strategy_settings.get('bbox')
        if not bbox:
            raise MissingConfigurationParameter('"bbox" must be provided for GridStrategy.')

        separation = config.strategy_settings.get('separation')
        if not separation or 'dx' not in separation or 'dy' not in separation:
            raise MissingConfigurationParameter('"separation" with "dx" and "dy" must be provided for GridStrategy.')

        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for GridStrategy.')

        quantity = int(config.quantity)
        dx = separation['dx']
        dy = separation['dy']

        # Handle both dict and string bbox formats
        if isinstance(bbox, str):
            _bbox = bbox.replace(',', ' ').split()
            if len(_bbox) != 4:
                raise ValueError(f"Invalid bbox format. Expected 'xmin,ymin xmax,ymax', got: {bbox}")
            xmin, ymin, xmax, ymax = map(float, _bbox)
        else:
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

        seed_locations = []

        # Generate grid points
        x = xmin
        while x <= xmax:
            y = ymin
            while y <= ymax:
                seed_locations.append((quantity, x, y))
                y += dy
            x += dx

        return seed_locations


class TransectStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles along straight line segments.
    A line segment is defined by two points (x1, y1) and (x2, y2).
    Particles along each segment are equally spaced, and the distance between particles is defined by
    the number of release locations per segment (k).
    """

    def seed(self, config: PopulationConfig) -> list[Tuple[int, float, float]]:
        # expect to return a dictionary with keys 'segments', 'k'
        segments = getattr(config, 'strategy_settings', {}).get('segments', None)
        if not segments:
            raise MissingConfigurationParameter('"segments" must be provided for TransectStrategy.')
        k = getattr(config, 'strategy_settings', {}).get('k', None)
        if not k:
            raise MissingConfigurationParameter('"k" must be provided for TransectStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for TransectStrategy.')
        quantity = int(config.quantity)

        seed_locations = []
        # Process each segment
        for segment_str in segments:
            try:
                # Parse segment string like '1000,2000 3000,4000'
                points = segment_str.strip().split()
                if len(points) != 2:
                    raise ValueError(f'Segment must contain exactly 2 points, got {len(points)}')

                # Parse first point (x1, y1)
                x1_str, y1_str = points[0].split(',')
                x1, y1 = float(x1_str.strip()), float(y1_str.strip())

                # Parse second point (x2, y2)
                x2_str, y2_str = points[1].split(',')
                x2, y2 = float(x2_str.strip()), float(y2_str.strip())

                # Generate k equally spaced points along the segment
                for i in range(k):
                    frac = i / (k - 1) if k > 1 else 0
                    x = x1 + frac * (x2 - x1)
                    y = y1 + frac * (y2 - y1)
                    seed_locations.append((quantity, x, y))

            except Exception as e:
                raise ValueError(f"Invalid segment string '{segment_str}': {e}") from e

        return seed_locations


class ParticleFactory:
    @staticmethod
    def create_particles(config: PopulationConfig) -> list[Particle]:
        """
        Create a list of particles of the specified type using a seeding strategy.

        Parameters
        ----------
        config : PopulationConfig
            Configuration object containing the seeding parameters.

        Returns
        -------
        list[Particle]
            List of created particles with positions and release times set.
        """
        from sedtrails.particle_tracer.particle import Sand, Mud, Passive

        PARTICLE_MAP = {'sand': Sand, 'mud': Mud, 'passive': Passive}
        STRATEGY_MAP = {
            'point': PointStrategy(),
            'random': RandomStrategy(),
            'grid': GridStrategy(),
            'transect': TransectStrategy(),
        }

        particle_type = getattr(config, 'particle_type', '')
        if particle_type.lower() not in PARTICLE_MAP:
            raise ValueError(f'Unknown particle type: {particle_type}')
        ParticleClass = PARTICLE_MAP[particle_type.lower()]

        strategy_name = getattr(config, 'strategy', '')
        if strategy_name.lower() not in STRATEGY_MAP:
            raise ValueError(f'Unknown seeding strategy: {strategy_name}')
        StrategyClass = STRATEGY_MAP[strategy_name.lower()]
        from sedtrails.particle_tracer.particle import Sand, Mud, Passive

        PARTICLE_MAP = {'sand': Sand, 'mud': Mud, 'passive': Passive}
        STRATEGY_MAP = {
            'point': PointStrategy(),
            'random': RandomStrategy(),
            'grid': GridStrategy(),
            'transect': TransectStrategy(),
        }

        # computes seeding positions using the strategy in config
        burial_depth = getattr(config, 'burial_depth', None)
        positions = StrategyClass.seed(config)
        particles = []
        for qty, x, y in positions:
            for _ in range(qty):
                p = ParticleClass()
                p.x = x
                p.y = y
                p.burial_depth = burial_depth
                p.release_time = getattr(config, 'release_start', None)

                particles.append(p)

        return particles


@dataclass
class ParticlePopulation:
    """
    Class handle operations for population of particles.
    A population is a group of particles that share the same type and seeding strategy.

    Attributes
    ----------
    field_x : ndarray
        The x-coordinates of the flow field data where particles are seeded.
    field_y : ndarray
        The y-coordinates of the flow field data where particles are seeded.
    population_config : PopulationConfig
        Configuration for the particle population, including seeding strategy and parameters.
    particles : Dict
        A dictionary containing particle attributes such as positions and status.
    _field_interpolator : Dict
        A dictionary containing Numba functions for interpolating field data.
    _position_calculator : Dict
        A dictionary containing Numba functions for calculating particle positions.
    _current_time : ndarray
        The current time in the simulation, used for updating particle positions.
    _field_mixing_depth : ndarray
        The mixing depth of the flow field, used to determine particle behavior.
    _field_transport_probability : ndarray
        The probability of particle transport in the flow field, used to determine if particles are picked up.
    """

    field_x: ndarray
    field_y: ndarray
    population_config: PopulationConfig
    particles: Dict = field(init=False, default_factory=dict)  # a dictionary with arrays
    _field_interpolator: Any = field(init=False)  # holds a Numba function
    _position_calculator: Any = field(init=False)  # holds a Numba function
    _current_time: ndarray = field(init=False)
    _field_mixing_depth: ndarray = field(init=False)  # TODO: we're not using this field yet
    _field_transport_probability: ndarray = field(init=False)  # TODO: we're not using this field yet

    def __post_init__(self):
        # Create a Numba calculator for particle operations
        numba_functions = create_numba_particle_calculator(grid_x=self.field_x, grid_y=self.field_y)

        self._field_interpolator = numba_functions['interpolate_field']
        self._position_calculator = numba_functions['update_particles']

        # generate particles based on the configuration
        _particles = ParticleFactory.create_particles(self.population_config)
        self.particles = {
            'x': np.array([p.x for p in _particles]),
            'y': np.array([p.y for p in _particles]),
            'release_time': np.array([p.release_time for p in _particles]),
            'burial_depth': np.array([p.burial_depth for p in _particles]),
        }

        # store the outer envelope of the domain
        coords = np.column_stack((self.field_x, self.field_y))
        hull = ConvexHull(coords)
        self._outer_envelope = Path(coords[hull.vertices])

    def update_information(
        self, current_time: ndarray, mixing_depth: ndarray, transport_probability: ndarray, bed_level: ndarray
    ) -> None:
        """
        Updates field data information for particles in the population.
        Parameters
        ----------
        current_time : ndarray
            The current time in the simulation.
        mixing_depth : ndarray
            The mixing depth of the flow field.
        transport_probability : ndarray
            The probability of particle transport in the flow field.
        bed_level : ndarray
            The bed level of the flow field.
        """

        self._current_time = current_time

        if not np.isnan(mixing_depth).all():
            self.particles['mixing_depth'] = self._field_interpolator(
                mixing_depth, self.particles['x'], self.particles['y']
            )

        if not np.isnan(transport_probability).all():
            """values between 0 and 1"""
            self.particles['transport_probability'] = self._field_interpolator(
                transport_probability, self.particles['x'], self.particles['y']
            )

        if not np.isnan(bed_level).all():
            self.particles['bed_level'] = self._field_interpolator(bed_level, self.particles['x'], self.particles['y'])

    def update_burial_depth(self) -> None:
        """Updates the burial depth of particles in the population.
        This method is a placeholder and should be implemented with the actual logic for updating burial depth.
        Currently, it does not perform any operations.
        """

        # Initialize vertical position ('z') based on bed level and burial depth
        if 'z' not in self.particles:
            self.particles['z'] = self.particles['bed_level'] - self.particles['burial_depth']

        # Make sure particles can never be higher than the bed level
        i_above_bed = self.particles['z'] > self.particles['bed_level']
        self.particles['z'][i_above_bed] = self.particles['bed_level'][i_above_bed]

        # Update burial depth (is always a positive value)
        self.particles['burial_depth'] = self.particles['bed_level'] - self.particles['z']

    def update_status(self) -> None:
        """
        updates status of particles in the population.
        """
        n_particles = len(self.particles['x'])

        # Compute whether particles are picked up (or trapped) based on transport probability
        # Note: If "reduced_velocity" is chosen, "transport_probability" always equals one.
        self.particles['is_picked_up'] = np.random.rand(n_particles) < self.particles['transport_probability']

        # Compute whether particles are inside (or outside) the domain envelope
        self.particles['is_inside'] = self._outer_envelope.contains_points(
            np.column_stack((self.particles['x'], self.particles['y']))
        )

        # Compute whether particles are exposed (or buried)
        self.particles['is_exposed'] = self.particles['burial_depth'] < self.particles['mixing_depth']

        # Compute whether particles are released (or retained)
        # FIXME: Temporary implementation
        self.particles['release_time'] = np.zeros_like(self.particles['x'])
        self.particles['is_released'] = self._current_time >= self.particles['release_time']

        # Compute whether particles are alive (or dead) (still TODO)
        self.particles['is_alive'] = np.ones(n_particles, dtype=bool)

        # Compute whether particles are mobile (or static) - combination of all status flags
        self.particles['is_mobile'] = (
            self.particles['is_inside']
            & self.particles['is_alive']
            & self.particles['is_exposed']
            & self.particles['is_released']
            & self.particles['is_picked_up']
        )

    def update_position(self, flow_field: Dict, current_timestep: float) -> None:
        """
        Update the position of particles in the population based on the flow field.

        Parameters
        ----------
        flow_field : Dict
            A dictionary containing the flow field information.
        current_timestep : float
            The current time step in the simulation in seconds.
        """

        ix = self.particles['is_mobile']  # Get indices of mobile particles

        n_particles = len(self.particles['x'])
        dx = np.zeros(n_particles)
        dy = np.zeros(n_particles)

        new_x, new_y = self._position_calculator(
            self.particles['x'][ix],
            self.particles['y'][ix],
            flow_field['u'],
            flow_field['v'],
            current_timestep,
        )

        dx[ix] = new_x - self.particles['x'][ix]
        dy[ix] = new_y - self.particles['y'][ix]

        self.particles['x'][ix] = new_x
        self.particles['y'][ix] = new_y


class ParticleSeeder:
    """
    High-level interface for particle seeding operations.

    This class provides a clean, modular interface for creating particles
    from configuration dictionaries.
    """

    def __init__(self, population_configs: List[Dict[str, Any]] | Dict[str, Any]):
        """
        Attributes
        ----------
        population_configs : List[Dict[str, Any]] | Dict[str, Any]
            A dictionary containing configuration for a single population,
            or a
            List of dictionaries, each containing configuration for one population.

        """

        self.population_configs = population_configs

    def seed(self, sedtrails_data: SedtrailsData) -> List[ParticlePopulation]:
        """
        Create particles from a list of population configuration dictionaries.

        Parameters
        ----------
         sedtrails_data : SedtrailsData
            The SedtrailsData object containing the field data (x, y coordinates).

        Returns
        -------
        List[ParticlePopulation]
            A list of ParticlePopulation objects, each containing the particles
            created for a specific population configuration.
        """

        if isinstance(self.population_configs, dict):
            # If a single dictionary is provided, convert it to a list for uniform processing
            self.population_configs = [self.population_configs]

        if not self.population_configs:
            raise ValueError('No population configurations provided for seeding.')

        populations = []
        for pop_config in self.population_configs:
            config = PopulationConfig(population_config=pop_config)
            pop = ParticlePopulation(field_x=sedtrails_data.x, field_y=sedtrails_data.y, population_config=config)
            populations.append(pop)
        return populations


# if __name__ == '__main__':
#     data = SedtrailsData()

#     config_random = {
#         'population': {
#             'particle_type': 'sand',
#             'seeding': {
#                 'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'nlocations': 2, 'seed': 42}},
#                 'quantity': 500,
#                 'release_start': '2025-06-18 13:00:00',
#                 'burial_depth': {
#                     'constant': 1.0,
#                 },
#             },
#         }
#     }

#     seeder = ParticleSeeder()
#     particles = seeder.seed(config_random)
#     print(f'Created {len(particles)} particles using random strategy.')
#     print(particles[:5])  # Print first 5 particles for inspection
