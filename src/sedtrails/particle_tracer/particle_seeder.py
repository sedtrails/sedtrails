"""
Particle Seeding Tool
=====================

Manage the creation of particles, their positions (x,y) and distribution.
using various release strategies.
Seeding strategies for positions include:
Point: Release particles at a specific locations (x,y).
Regular Grid: Release particles in a regular grid pattern based
    on distances between particles in x and y directions, and the
    simulation. A mask can be applied to restrict the area of seeding.
Transect: release particle along line segments  defined by two points(x1,y1) and (x2,y2).
Random: Release particles at random locations (x,y) within an area
    constrained by a bounding box (xmin, xmax, ymin, ymax).
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple, Union

import numpy as np
from matplotlib.path import Path
from numpy import ndarray

from sedtrails.application_interfaces.find import find_value
from sedtrails.exceptions import MissingConfigurationParameter
from sedtrails.particle_tracer.particle import Particle
from sedtrails.particle_tracer.position_calculator_numba import create_grid_geometry


class HasFieldCoordinates(Protocol):
    x: ndarray
    y: ndarray


def _is_temporal_field(field_value: Any) -> bool:
    return isinstance(field_value, dict) and {'lower', 'upper', 'weight'}.issubset(field_value)


def _is_temporal_flow_field(flow_field: Dict) -> bool:
    return _is_temporal_field(flow_field) and isinstance(flow_field.get('lower'), dict)


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
    s    strategy_settings : Dict
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


class FilePointsStrategy(SeedingStrategy):
    """
    Seeding strategy to read (x, y) locations from a file.

    Expected settings under `seeding.strategy.file_points`:

    - path: str                 # required. Path to the file with coordinates
    - x_col: str|int = 0        # optional. Column name or 0-based index for x
    - y_col: str|int = 1        # optional. Column name or 0-based index for y
    - has_header: bool = True   # optional. If False, treat as no header
    - deduplicate: bool = True  # optional. Drop duplicate rows
    - dropna: bool = True       # optional. Drop rows with NaNs in x/y
    - bbox: str|dict = None     # optional. Restrict to bbox: "xmin,ymin xmax,ymax" or dict
    - stride: int = 1           # optional. Keep every `stride`-th point (≥1)
    """

    def seed(self, config: PopulationConfig) -> list[Tuple[int, float, float]]:
        import os

        import pandas as pd

        settings = getattr(config, 'strategy_settings', {})
        path = settings.get('path', None)
        if not path:
            raise MissingConfigurationParameter('"path" must be provided for FilePointsStrategy.')

        # Windows path safety
        path = os.path.expanduser(str(path))

        x_col = settings.get('x_col', 0)
        y_col = settings.get('y_col', 1)
        has_header = bool(settings.get('has_header', True))
        deduplicate = bool(settings.get('deduplicate', True))
        dropna = bool(settings.get('dropna', True))
        stride = int(settings.get('stride', 1))
        if stride < 1:
            raise ValueError('"stride" must be >= 1.')

        # Parse optional bbox
        bbox = settings.get('bbox', None)
        xmin = ymin = xmax = ymax = None
        if bbox:
            if isinstance(bbox, str):
                parts = bbox.replace(',', ' ').split()
                if len(parts) != 4:
                    raise ValueError(f'Invalid bbox string: {bbox}')
                xmin, ymin, xmax, ymax = map(float, parts)
            elif isinstance(bbox, dict):
                xmin = float(bbox['xmin'])
                ymin = float(bbox['ymin'])
                xmax = float(bbox['xmax'])
                ymax = float(bbox['ymax'])
            else:
                raise ValueError('bbox must be str "xmin,ymin xmax,ymax" or dict with xmin/ymin/xmax/ymax')

        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be provided for FilePointsStrategy.')
        quantity = int(config.quantity)

        if not os.path.isfile(path):
            raise FileNotFoundError(f'Could not find coordinates file: {path}')

        # --- Read file, auto-delimiter handling via pandas (engine="python" allows sep=None sniffing)
        try:
            df = pd.read_csv(path, sep=None, engine='python', header=0 if has_header else None)
        except Exception:
            # Fallback: whitespace-delimited
            df = pd.read_csv(path, delim_whitespace=True, header=0 if has_header else None)

        # Resolve columns by name or index
        def _resolve_col(col, df):
            if isinstance(col, int):
                # Convert positional index to actual column name
                return df.columns[col]
            return col  # assume str

        x_name = _resolve_col(x_col, df)
        y_name = _resolve_col(y_col, df)

        if x_name not in df.columns or y_name not in df.columns:
            raise ValueError(f'Columns not found. Available: {list(df.columns)}; requested x={x_name}, y={y_name}')

        df = df[[x_name, y_name]].copy()
        df.columns = ['x', 'y']

        if dropna:
            df = df.dropna(subset=['x', 'y'])
        if deduplicate:
            df = df.drop_duplicates(subset=['x', 'y'])

        # Optional bbox mask
        if xmin is not None:
            df = df[(df['x'] >= xmin) & (df['x'] <= xmax) & (df['y'] >= ymin) & (df['y'] <= ymax)]

        # Optional stride
        if stride > 1 and not df.empty:
            df = df.iloc[::stride, :]

        if df.empty:
            raise ValueError('No valid (x, y) points found after filtering.')

        # Build seed locations
        seed_locations = [
            (quantity, float(x), float(y)) for x, y in zip(df['x'].to_numpy(), df['y'].to_numpy(), strict=True)
        ]
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
        from sedtrails.particle_tracer.particle import Mud, Passive, Sand

        PARTICLE_MAP = {'sand': Sand, 'mud': Mud, 'passive': Passive}
        STRATEGY_MAP = {
            'point': PointStrategy(),
            'random': RandomStrategy(),
            'grid': GridStrategy(),
            'transect': TransectStrategy(),
            'file_points': FilePointsStrategy(),
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
    _field_interpolator : Any
        Bound method for interpolating one nodal field to particle positions.
    _field_interpolator_multi : Any
        Bound method for interpolating multiple nodal fields with one point-location pass.
    _position_calculator_with_simplex : Any
        Bound method for advancing particles while reusing cached simplex ids.
    _position_calculator_temporal_with_simplex : Any
        Bound method for temporal particle updates while reusing cached simplex ids.
    _particle_simplices : ndarray
        Cached containing-triangle ids for each particle, used to avoid global point location on every update.
    _current_time : ndarray
        The current time in the simulation, used for updating particle positions.
    _field_mixing_depth : ndarray
        The mixing depth of the flow field, reserved for later particle-behavior logic.
    _field_transport_probability : ndarray
        The probability of particle transport in the flow field, reserved for later pickup logic.
    """

    field_x: ndarray
    field_y: ndarray
    population_config: PopulationConfig
    grid_geometry: Any = None
    particles: Dict = field(init=False, default_factory=dict)  # a dictionary with arrays
    _field_interpolator: Any = field(init=False)
    _field_interpolator_multi: Any = field(init=False)
    _position_calculator_with_simplex: Any = field(init=False)
    _position_calculator_temporal_with_simplex: Any = field(init=False)
    _particle_simplices: ndarray = field(init=False)
    _current_time: float = field(init=False)
    _field_mixing_depth: ndarray = field(init=False)  # TODO: reserved for later particle-behavior logic
    _field_transport_probability: ndarray = field(init=False)  # TODO: reserved for later pickup logic

    def __post_init__(self):
        if self.grid_geometry is None:
            self.grid_geometry = create_grid_geometry(self.field_x, self.field_y)

        # Reuse methods bound to the shared grid geometry.
        self._field_interpolator = self.grid_geometry.interpolate_field
        self._field_interpolator_multi = self.grid_geometry.interpolate_fields
        self._position_calculator_with_simplex = self.grid_geometry.update_particles_with_simplex
        self._position_calculator_temporal_with_simplex = self.grid_geometry.update_particles_temporal_with_simplex

        # generate particles based on the configuration
        _particles = ParticleFactory.create_particles(self.population_config)
        self.particles = {
            'x': np.array([p.x for p in _particles]),
            'y': np.array([p.y for p in _particles]),
            'release_time': np.array([p.release_time for p in _particles]),
            'burial_depth': np.array([p.burial_depth for p in _particles]),
        }
        self._particle_simplices = self.grid_geometry.locate_points(self.particles['x'], self.particles['y'])

        # Store the outer envelope of the domain using shared grid geometry.
        self._outer_envelope = Path(self.grid_geometry.outer_envelope)

    def update_information(
        self, current_time: Union[int, float], mixing_depth: Any, transport_probability: Any, bed_level: Any
    ) -> None:
        """
        Updates field data information for particles in the population.

        Parameters
        ----------
        current_time : float, int
            The current time in the simulation.
        mixing_depth : ndarray
            The mixing depth of the flow field.
        transport_probability : ndarray
            The probability of particle transport in the flow field.
        bed_level : ndarray
            The bed level of the flow field.
        """

        self._current_time = current_time

        self._update_particle_field('mixing_depth', mixing_depth)
        self._update_particle_field('transport_probability', transport_probability)
        self._update_particle_field('bed_level', bed_level)

    def _update_particle_field(self, name: str, field_value) -> None:
        if field_value is None:
            return

        if _is_temporal_field(field_value):
            lower_values = np.asarray(field_value['lower'])
            upper_values = np.asarray(field_value['upper'])
            if lower_values.size == 0:
                return

            weight = field_value['weight']
            if weight <= 0.0 or lower_values is upper_values:
                lower_particle_values = self._field_interpolator(lower_values, self.particles['x'], self.particles['y'])
                if np.isnan(lower_particle_values).all():
                    return
                self.particles[name] = lower_particle_values
                return

            lower_particle_values, upper_particle_values = self._field_interpolator_multi(
                (lower_values, upper_values),
                self.particles['x'],
                self.particles['y'],
            )
            if np.isnan(lower_particle_values).all() and np.isnan(upper_particle_values).all():
                return
            self.particles[name] = lower_particle_values + weight * (upper_particle_values - lower_particle_values)
            return

        if np.isscalar(field_value):
            self.particles[name] = np.full(len(self.particles['x']), field_value, dtype=float)
            return

        field_array = np.asarray(field_value)
        if field_array.size == 0:
            return

        particle_values = self._field_interpolator(field_array, self.particles['x'], self.particles['y'])
        if np.isnan(particle_values).all():
            return

        self.particles[name] = particle_values

    def update_burial_depth(self) -> None:
        """Updates the burial depth of particles in the population.
        This method is a placeholder and should be implemented with the actual logic for updating burial depth.
        Currently, it does not perform any operations.
        """

        # Initialize vertical position ('z') based on bed level and burial depth
        self.particles['z'] = (
            self.particles['bed_level'] - self.particles['burial_depth']
        )  # TODO: add to top attributes. This must go to netcdf for every timestep.

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

        # New conditional logic based on transport_probability_method
        if self.population_config.population_config['transport_probability'] == 'no_probability':
            # For no_probability method, all particles are considered exposed (always mobile)
            self.particles['is_exposed'] = np.ones(n_particles, dtype=bool)
        else:
            # For stochastic_transport and reduced_velocity methods, use burial_depth vs mixing_depth

            # if van westen method:
            # a particle is considered exposed if it is buried at a shallower depth compared to the mixing depth
            self.particles['is_exposed'] = self.particles['burial_depth'] < self.particles['mixing_depth']

            # if soulsby method:
            # self.particles['is_exposed'] = (this is where we implement Soulsby's F based on a and b)

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
        particle_indices = np.flatnonzero(ix)
        if particle_indices.size == 0:
            return

        if _is_temporal_flow_field(flow_field):
            new_x, new_y, new_simplices = self._position_calculator_temporal_with_simplex(
                self.particles['x'][ix],
                self.particles['y'][ix],
                flow_field['lower']['u'],
                flow_field['lower']['v'],
                flow_field['upper']['u'],
                flow_field['upper']['v'],
                flow_field['weight'],
                current_timestep,
                simplex_ids=self._particle_simplices[particle_indices],
            )
        else:
            new_x, new_y, new_simplices = self._position_calculator_with_simplex(
                self.particles['x'][ix],
                self.particles['y'][ix],
                flow_field['u'],
                flow_field['v'],
                current_timestep,
                simplex_ids=self._particle_simplices[particle_indices],
            )

        # TODO: implement Bart's solution for gross/net values here. Add

        self.particles['x'][ix] = new_x
        self.particles['y'][ix] = new_y
        self._particle_simplices[particle_indices] = new_simplices


class ParticleSeeder:
    """
    High-level interface for particle seeding operations.

    This class provides a clean, modular interface for creating particles
    from configuration dictionaries.

    Attributes
    ----------
    population_configs : List[Dict[str, Any]] | Dict[str, Any]
        A dictionary containing configuration for a single population,
        or a
        List of dictionaries, each containing configuration for one population.

    """

    def __init__(self, population_configs: List[Dict[str, Any]] | Dict[str, Any]):
        self.population_configs = population_configs

    def seed(self, sedtrails_data: HasFieldCoordinates) -> List[ParticlePopulation]:
        """
        Create particles from a list of population configuration dictionaries.

        Parameters
        ----------
         sedtrails_data : HasFieldCoordinates
            Any object exposing `x` and `y` field coordinate arrays.

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
        grid_geometry = create_grid_geometry(sedtrails_data.x, sedtrails_data.y)
        for pop_config in self.population_configs:
            config = PopulationConfig(population_config=pop_config)
            pop = ParticlePopulation(
                field_x=sedtrails_data.x,
                field_y=sedtrails_data.y,
                population_config=config,
                grid_geometry=grid_geometry,
            )
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
