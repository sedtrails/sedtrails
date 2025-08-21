"""
Particle Seeding Tool
=====================
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
"""

from abc import ABC, abstractmethod
from sedtrails.particle_tracer.particle import Particle
from sedtrails.exceptions import MissingConfigurationParameter
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass, field
from sedtrails.configuration_interface.find import find_value


@dataclass
class SeedingConfig:
    """
    A class to represent the seeding parameters of a population of particle.
    A population is a group of particles that share the same type and seeding strategy.

    Attributes
    ----------
    population_config : Dict
        The configuration dictionary containing the seeding parameters.
    particle_type : str
        The type of particles to be seeded (e.g., 'sand', 'mud', 'passive').
    release_start : str
        The time at which the particles for a given population are released.
    quantity : int
        The number of particles to release per release location.
    strategy_settings : Dict
        The settings for the seeding strategy, extracted from the configuration.
        These are any key-value pairs defined under the specific strategy in the configuration.
    """

    population_config: Dict
    strategy: str = field(init=False)
    particle_type: str = field(init=False)
    release_start: str = field(init=False)  # particle for a given population are released at this time
    quantity: int = field(init=False)  # number of particles to release per release location
    strategy_settings: Dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        _strategy = find_value(self.population_config, 'population.seeding.strategy', {}).keys()
        if not _strategy:
            raise MissingConfigurationParameter('"strategy" is not defined as seeding parameter.')
        self.strategy = next(iter(_strategy))
        self.strategy_settings = find_value(self.population_config, f'population.seeding.strategy.{self.strategy}', {})
        if not self.strategy_settings:
            raise MissingConfigurationParameter(f'"{self.strategy}" settings are not defined in the configuration.')
        _quantity = find_value(self.population_config, 'population.seeding.quantity', {})
        if not _quantity:
            raise MissingConfigurationParameter('"quantity" is not defined as seeding parameter.')
        self.quantity = _quantity
        _release_start = find_value(self.population_config, 'population.seeding.release_start', {})
        if not _release_start:
            raise MissingConfigurationParameter('"release_start" is not defined in the population configuration.')
        self.release_start = _release_start
        self.particle_type = find_value(self.population_config, 'population.particle_type', '')
        if not self.particle_type:
            raise MissingConfigurationParameter('"particle_type" is not defined in the population configuration.')


class SeedingStrategy(ABC):
    """
    Abstract base class for seeding strategies.
    """

    @abstractmethod
    def seed(self, config: SeedingConfig) -> List[Tuple[int, float, float]]:
        """
        Asociates quantity of particles to a seeding locations for a given strategy.

        Parameters
        ----------
        config : SeedingConfig
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

    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
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

    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
        # expects strategy_settings to contain 'bbox' and 'seed'
        bbox = getattr(config, 'strategy_settings', {}).get('bbox', None)
        if not bbox:
            raise MissingConfigurationParameter('"bbox" must be provided for RandomStrategy.')

        seed = getattr(config, 'strategy_settings', {}).get('seed', None)
        if not seed:
            raise MissingConfigurationParameter('"seed" must be provided for RandomStrategy.')
        random.seed(seed)

        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for RandomStrategy.')
        quantity = int(config.quantity)
        seed_locations = []

        _bbox = bbox.replace(',', ' ').split()  # separates values with whitespaces. Order is xmin, ymin, xmax, ymax
        for _ in range(quantity):
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

    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
        bbox = getattr(config, 'strategy_settings', {}).get('bbox', None)
        if not bbox:
            raise RuntimeError('Bounding box must be provided for GridStrategy.')

        grid = find_value(config.population_config, 'population.seeding.strategy.grid', {})

        if not grid or not all(k in grid.get('separation', {}) for k in ['dx', 'dy']):
            raise MissingConfigurationParameter('"grid" must be provided for GridStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for GridStrategy.')
        quantity = int(config.quantity)
        seed_locations = []
        dx = find_value(grid, 'separation.dx')
        dy = find_value(grid, 'separation.dy')
        xvals = []
        yvals = []
        x = bbox['xmin']
        while x <= bbox['xmax']:
            if x > bbox['xmax']:
                break
            xvals.append(x)
            x += dx
        if xvals and xvals[-1] > bbox['xmax']:
            xvals.pop()
        y = bbox['ymin']
        while y <= bbox['ymax']:
            if y > bbox['ymax']:
                break
            yvals.append(y)
            y += dy
        if yvals and yvals[-1] > bbox['ymax']:
            yvals.pop()
        for x in xvals:
            for y in yvals:
                seed_locations.append((quantity, x, y))
        return seed_locations


class TransectStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles along straight line segments.
    A line segment is defined by two points (x1, y1) and (x2, y2).
    Particles along each segment are equally spaced, and the distance between particles is defined by
    the number of release locations per segment (k).
    """

    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
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
    def create_particles(config: SeedingConfig) -> list[Particle]:
        """
        Create a list of particles of the specified type using a seeding strategy.

        Parameters
        ----------
        config : SeedingConfig
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

        # computes seeding positions using the strategy in config
        positions = StrategyClass.seed(config)
        particles = []
        for qty, x, y in positions:
            for _ in range(qty):
                p = ParticleClass()
                p.x = x
                p.y = y

                p.release_time = getattr(config, 'release_start', None)

                particles.append(p)

        return particles


# TODO:  save the final positions of particles to a vector based files (e.g., shapefile?, GeoJSON?)

if __name__ == '__main__':
    # Example usage
    config_point = SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                    'quantity': 1,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )

    config_transect = SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {
                        'transect': {
                            'segments': ['0,0 2,0'],
                            'k': 3,
                        }
                    },
                    'quantity': 5,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )

    config_random = SeedingConfig(
        {
            'population': {
                'particle_type': 'sand',
                'seeding': {
                    'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'seed': 42}},
                    'quantity': 5,
                    'release_start': '2025-06-18 13:00:00',
                },
            }
        }
    )

    particles = ParticleFactory.create_particles(config_random)
    print('Created particles:', particles)  # Should print the created particles with their positions and release times
