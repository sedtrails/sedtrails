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

import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple, Union

import numpy as np
from matplotlib.path import Path
from numpy import ndarray

from sedtrails.application_interfaces.find import find_value
from sedtrails.exceptions import MissingConfigurationParameter
from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.particle_tracer.diffusion_library import BrownianDiffusionStrategy, DiffusionCalculator
from sedtrails.particle_tracer.particle import Particle
from sedtrails.particle_tracer.position_calculator_numba import (
    BOUNDARY_CLASS_LAND,
    BOUNDARY_CLASS_OPEN,
    create_grid_geometry,
)
from sedtrails.particle_tracer.timer import convert_datetime_string_to_datetime64, convert_reference_date_to_datetime64

logger = logging.getLogger(__name__)


def _read_polygon_file(path: str) -> np.ndarray:
    """Read polygon vertices from a file, auto-detecting the format.

    Supported formats
    -----------------
    - Delft3D/TELEMAC ``.pol``: ``<name>\\n<nrows> <ncols>\\n<x> <y>\\n...``
    - CSV with a header row (first token non-numeric)
    - Plain two-column text (space- or comma-separated, no header)
    """
    path = os.path.expanduser(str(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Polygon file not found: {path}')

    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f'Polygon file is empty: {path}')

    vertices: list[tuple[float, float]] = []

    # --- Delft3D .pol format ---
    if os.path.splitext(path)[1].lower() == '.pol':
        i = 0
        while i < len(lines):
            i += 1  # skip name line
            if i >= len(lines):
                break
            try:
                parts = lines[i].split()
                nrows = int(parts[0])
                i += 1
            except (ValueError, IndexError):
                continue
            for j in range(nrows):
                if i + j < len(lines):
                    coords = lines[i + j].replace(',', ' ').split()
                    if len(coords) >= 2:
                        vertices.append((float(coords[0]), float(coords[1])))
            i += nrows
        if vertices:
            return np.array(vertices)

    # --- Generic text / CSV ---
    # Detect header: first line is a header if its first token is not a float.
    def _is_numeric(token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False

    first_tokens = lines[0].replace(',', ' ').split()
    start = 1 if (first_tokens and not _is_numeric(first_tokens[0])) else 0

    for line in lines[start:]:
        parts = line.replace(',', ' ').split()
        if len(parts) >= 2:
            try:
                vertices.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    if not vertices:
        raise ValueError(f'Could not parse any polygon vertices from: {path}')
    return np.array(vertices)


def _parse_polygon(poly_spec) -> np.ndarray:
    """Return an (N, 2) array of polygon vertices.

    Parameters
    ----------
    poly_spec : str or list[str]
        Either a file path (string) or a list of ``'x,y'`` coordinate strings.
    """
    if isinstance(poly_spec, str):
        return _read_polygon_file(poly_spec)
    if isinstance(poly_spec, list):
        vertices = []
        for item in poly_spec:
            parts = str(item).replace(',', ' ').split()
            if len(parts) < 2:
                raise ValueError(f"Invalid polygon coordinate '{item}'. Expected 'x,y' or 'x y'.")
            vertices.append((float(parts[0]), float(parts[1])))
        if len(vertices) < 3:
            raise ValueError('A polygon requires at least 3 vertices.')
        return np.array(vertices)
    raise ValueError('poly must be a file path string or a list of "x,y" coordinate strings.')


def _sample_burial_depth(burial_depth_config, rng: random.Random | None = None) -> float:
    """Resolve a single burial-depth value from the population config entry.

    Parameters
    ----------
    burial_depth_config : dict or float
        Either ``{'constant': value}`` for a fixed depth, or
        ``{'random': max_value}`` to draw uniformly from ``[0, max_value]``.
        A bare float is passed through unchanged (used when the config is
        already a resolved number, e.g. from legacy test fixtures).
    rng : random.Random, optional
        A local ``random.Random`` instance to use for stochastic sampling.
        When *None* the module-level ``random`` generator is used as a
        fallback (legacy behaviour).
    """
    if isinstance(burial_depth_config, dict):
        if 'constant' in burial_depth_config:
            return float(burial_depth_config['constant'])
        if 'random' in burial_depth_config:
            _rng = rng if rng is not None else random
            return _rng.uniform(0.0, float(burial_depth_config['random']))
        raise ValueError(
            'Unsupported burial_depth configuration. '
            'Use {constant: value} or {random: max_value}.'
        )
    return float(burial_depth_config)


def _compute_seeding_area(strategy_name: str, strategy_settings: dict) -> float | None:
    """Return the 2-D seeding area in m² for area-based strategies, or None.

    Only ``random`` and ``grid`` strategies define a spatial area (via ``bbox``
    or ``poly``).  For all other strategies (point, transect, file_points) the
    concept of a seeding area is not applicable and ``None`` is returned.

    Parameters
    ----------
    strategy_name : str
        Name of the active seeding strategy.
    strategy_settings : dict
        Raw settings dict for that strategy (i.e. ``config.strategy_settings``).

    Returns
    -------
    float or None
        Area in m², or None when not computable.
    """
    if strategy_name not in ('random', 'grid'):
        return None

    poly = strategy_settings.get('poly')
    bbox = strategy_settings.get('bbox')

    if poly is not None:
        vertices = _parse_polygon(poly)
        n = len(vertices)
        area = 0.5 * abs(
            sum(
                vertices[i][0] * vertices[(i + 1) % n][1]
                - vertices[(i + 1) % n][0] * vertices[i][1]
                for i in range(n)
            )
        )
        return area

    if bbox is not None:
        if isinstance(bbox, str):
            parts = bbox.replace(',', ' ').split()
            xmin, ymin, xmax, ymax = map(float, parts)
        else:
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        return (xmax - xmin) * (ymax - ymin)

    return None


def _compute_repr_volume(config, n_particles: int) -> float | None:
    """Return representative volume [m³/particle], or None if not applicable.

    Only defined when burial depth is a random-uniform distribution and the
    seeding strategy has a computable 2-D footprint area.
    """
    burial_depth = getattr(config, 'burial_depth', None)
    if not isinstance(burial_depth, dict) or 'random' not in burial_depth:
        return None
    max_depth = float(burial_depth['random'])
    strategy_name = getattr(config, 'strategy', '')
    strategy_settings = getattr(config, 'strategy_settings', {})
    area = _compute_seeding_area(strategy_name, strategy_settings)
    if area is None or n_particles == 0:
        return None
    return area * max_depth / n_particles


def _log_seeding_box_volume(config, positions: list) -> None:
    """Log the seeding box volume and representative particle volume.

    The "seeding box" is defined only when both of the following conditions hold:

    1. The strategy is area-based (``random`` or ``grid``) so a 2-D footprint
       can be computed from the ``bbox`` or ``poly`` setting.
    2. The ``burial_depth`` is configured as ``{random: max_depth}``, so
       particles are scattered uniformly in depth from 0 to *max_depth* and
       the depth extent is well-defined.

    When both conditions are met the function logs (at INFO level):

    * seeding footprint area [m²]
    * depth range [m]
    * total box volume [m³]
    * total number of particles
    * representative volume per particle [m³]

    Parameters
    ----------
    config : PopulationConfig
        Population configuration.
    positions : list of (int, float, float)
        Seed locations returned by the active strategy, each entry being
        ``(quantity, x, y)``.
    """
    burial_depth = getattr(config, 'burial_depth', None)
    if not isinstance(burial_depth, dict) or 'random' not in burial_depth:
        return

    max_depth = float(burial_depth['random'])
    strategy_name = getattr(config, 'strategy', '')
    strategy_settings = getattr(config, 'strategy_settings', {})

    area = _compute_seeding_area(strategy_name, strategy_settings)
    if area is None:
        return

    n_particles = sum(qty for qty, *_ in positions)
    if n_particles == 0:
        return

    volume = area * max_depth
    repr_volume = volume / n_particles

    pop_name = config.population_config.get('name', strategy_name)
    logger.info(
        "Seeding box for population '%s': "
        "area=%.4g m², depth=[0, %.4g] m, volume=%.4g m³, "
        "n_particles=%d, representative volume=%.4g m³/particle",
        pop_name, area, max_depth, volume, n_particles, repr_volume,
    )


class HasFieldCoordinates(Protocol):
    """Protocol for seeding input that exposes field coordinates.

    Attributes
    ----------
    x : ndarray
        Field x-coordinate array.
    y : ndarray
        Field y-coordinate array.
    """

    x: ndarray
    y: ndarray


DEFAULT_REFERENCE_DATE = '1970-01-01 00:00:00'
DEFAULT_RELEASE_START = '__SIMULATION_START__'
MISSING = object()


def _release_time_to_seconds(release_time: str | int | float, reference_date: str | np.datetime64) -> float:
    """Convert a release time to seconds since the model reference date."""

    if release_time == DEFAULT_RELEASE_START:
        release_seconds = 0.0
    elif isinstance(release_time, (int, float)):
        release_seconds = float(release_time)
    else:
        release_time_str = str(release_time).strip()
        try:
            release_seconds = float(release_time_str)
        except ValueError:
            release_datetime = convert_datetime_string_to_datetime64(release_time_str)
            if isinstance(reference_date, np.datetime64):
                reference_datetime = reference_date.astype('datetime64[s]')
            else:
                reference_datetime = convert_reference_date_to_datetime64(str(reference_date))
            release_seconds = float((release_datetime - reference_datetime).astype('timedelta64[s]').astype(int))

    if release_seconds < 0:
        warnings.warn(
            'Computed release time is negative. Particles may be released from simulation start; '
            'check seeding.release_start and general.input_model.reference_date.',
            UserWarning,
            stacklevel=2,
        )
    return release_seconds


def _is_temporal_field(field_value: Any) -> bool:
    return isinstance(field_value, dict) and {'lower', 'upper', 'weight'}.issubset(field_value)


def _is_temporal_flow_field(flow_field: Dict) -> bool:
    return _is_temporal_field(flow_field) and isinstance(flow_field.get('lower'), dict)


def _iter_expanded_particles(positions):
    """Yield all expanded particles as ``(global_index, x, y)`` tuples."""
    global_index = 0
    for qty, x_coord, y_coord in positions:
        for _ in range(int(qty)):
            yield global_index, x_coord, y_coord
            global_index += 1


def _parallel_env_int(name: str, default: int) -> int:
    """Return an integer parallel-runtime environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        raise ConfigurationError(f'{name} must be an integer, got {raw_value!r}.') from None


def _parallel_task_context_from_env() -> tuple[int, int]:
    """Return validated parallel task identity from the worker environment."""
    task_id = _parallel_env_int('SEDTRAILS_TASK_ID', 0)
    n_tasks = _parallel_env_int('SEDTRAILS_N_TASKS', 1)
    if n_tasks < 1:
        raise ConfigurationError('SEDTRAILS_N_TASKS must be greater than or equal to 1.')
    if task_id < 0 or task_id >= n_tasks:
        raise ConfigurationError(
            'SEDTRAILS_TASK_ID must satisfy 0 <= SEDTRAILS_TASK_ID < SEDTRAILS_N_TASKS.'
        )
    return task_id, n_tasks


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
        release_start : str | int | float
            The time at which the particles for a given population are released.
            If omitted, particles are released from simulation start.
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
    release_start: str | int | float = field(init=False, default=DEFAULT_RELEASE_START)
    quantity: int = field(init=False)  # number of particles to release per release location
    burial_depth: float | dict[str, float] = field(init=False, default=0.0)  # burial depth configuration for the particles
    strategy_settings: Dict = field(init=False, default_factory=dict)
    remove_permanently_buried: bool = field(init=False, default=False)
    diffusion_method: str = field(init=False, default='brownian')
    diffusion_coefficient: float = field(init=False, default=0.0)
    diffusion_seed: int | None = field(init=False, default=None)

    def __post_init__(self):
        _strategy = find_value(self.population_config, 'seeding.strategy', {}).keys()
        if not _strategy:
            raise MissingConfigurationParameter('"strategy" is not defined as seeding parameter.')
        self.strategy = next(iter(_strategy))
        self.strategy_settings = find_value(self.population_config, f'seeding.strategy.{self.strategy}', {})
        if not self.strategy_settings:
            raise MissingConfigurationParameter(f'"{self.strategy}" settings are not defined in the configuration.')
        _quantity = find_value(self.population_config, 'seeding.quantity', MISSING)
        if _quantity is MISSING:
            raise MissingConfigurationParameter('"quantity" is not defined as seeding parameter.')
        self.quantity = _quantity
        _release_start = find_value(self.population_config, 'seeding.release_start', None)
        self.release_start = DEFAULT_RELEASE_START if _release_start is None else _release_start
        self.particle_type = find_value(self.population_config, 'particle_type', '')
        if not self.particle_type:
            raise MissingConfigurationParameter('"particle_type" is not defined in the population configuration.')
        _burial_depth = find_value(self.population_config, 'seeding.burial_depth', None)
        if _burial_depth is None:
            self.burial_depth = {'constant': 0.0}
        else:
            self.burial_depth = _burial_depth
        self.remove_permanently_buried = bool(
            find_value(self.population_config, 'seeding.remove_permanently_buried', False)
        )
        diffusion_config = find_value(self.population_config, 'diffusion', {})
        if not isinstance(diffusion_config, dict):
            raise ValueError('"diffusion" must be a mapping.')
        self.diffusion_method = diffusion_config.get('method', 'brownian')
        if self.diffusion_method not in {'none', 'brownian'}:
            raise ValueError('"diffusion.method" must be "none" or "brownian".')
        diffusion_coefficient = diffusion_config.get(
            'coefficient', find_value(self.population_config, 'characteristics.diffusion_coefficient', 0.0)
        )
        if (
            isinstance(diffusion_coefficient, (bool, np.bool_))
            or not isinstance(diffusion_coefficient, (int, float, np.number))
            or not np.isfinite(diffusion_coefficient)
            or diffusion_coefficient < 0.0
        ):
            raise ValueError(
                '"diffusion.coefficient" (or legacy "characteristics.diffusion_coefficient") must be a finite non-negative number.'
            )
        self.diffusion_coefficient = float(diffusion_coefficient)
        diffusion_seed = diffusion_config.get('seed')
        if diffusion_seed is not None and (
            isinstance(diffusion_seed, (bool, np.bool_)) or not isinstance(diffusion_seed, (int, np.integer))
        ):
            raise ValueError('"diffusion.seed" must be an integer.')
        self.diffusion_seed = None if diffusion_seed is None else int(diffusion_seed)


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
        """
        Return seed.

        Parameters
        ----------
        config : PopulationConfig
            Configuration mapping used by the operation.

        Returns
        -------
        list[Tuple[int, float, float]]
            Integer result of the calculation.
        """
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
    """Release particles at random locations within an area.

    Notes
    -----
    The area can be defined as:
    - ``bbox``: axis-aligned bounding box string ``'xmin,ymin xmax,ymax'``
    - ``poly``: an arbitrary polygon given as a list of ``'x,y'`` strings or a
      path to a polygon file (``.pol``, CSV, or plain text).  When both are
      given, ``poly`` takes precedence.

    Rejection sampling is used for ``poly``; up to
    ``max(nlocations * 1000, 10_000)`` candidate points are drawn from the
    polygon's bounding box and tested for containment.
    """

    def seed(self, config: PopulationConfig) -> list[Tuple[int, float, float]]:
        """Generate random seed locations for one population.

        Parameters
        ----------
        config : PopulationConfig
            Population configuration containing ``quantity`` and random
            strategy settings.

        Returns
        -------
        list of tuple of int and float
            Seed locations as ``(quantity, x, y)`` tuples.

        Raises
        ------
        MissingConfigurationParameter
            If the area definition, particle quantity, or number of locations
            is missing.
        ValueError
            If ``nlocations`` is invalid or too few points can be sampled
            inside a polygon.
        """
        settings = getattr(config, 'strategy_settings', {})
        bbox = settings.get('bbox', None)
        poly = settings.get('poly', None)

        if poly is None and not bbox:
            raise MissingConfigurationParameter('"bbox" or "poly" must be provided for RandomStrategy.')

        seed_val = settings.get('seed', 42)
        random.seed(seed_val)

        nlocations = settings.get('nlocations', None)
        if nlocations is None:
            raise MissingConfigurationParameter('"nlocations" must be provided for RandomStrategy.')
        nlocations = int(nlocations)
        if nlocations <= 0:
            raise ValueError('"nlocations" must be a positive integer for RandomStrategy.')

        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for RandomStrategy.')
        quantity = int(config.quantity)

        if poly is not None:
            vertices = _parse_polygon(poly)
            xmin, ymin = vertices.min(axis=0)
            xmax, ymax = vertices.max(axis=0)
            poly_path = Path(vertices)

            seed_locations: list[Tuple[int, float, float]] = []
            max_attempts = max(nlocations * 1000, 10_000)
            attempts = 0
            while len(seed_locations) < nlocations and attempts < max_attempts:
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
                if poly_path.contains_point((x, y), radius=1e-9):
                    seed_locations.append((quantity, x, y))
                attempts += 1

            if len(seed_locations) < nlocations:
                raise ValueError(
                    f'Could only generate {len(seed_locations)} of {nlocations} points inside the polygon '
                    f'after {max_attempts} attempts. The polygon may be very narrow relative to its bounding box.'
                )
        else:
            _bbox = bbox.replace(',', ' ').split()
            seed_locations = []
            for _ in range(nlocations):
                x = random.uniform(float(_bbox[0]), float(_bbox[2]))
                y = random.uniform(float(_bbox[1]), float(_bbox[3]))
                seed_locations.append((quantity, x, y))

        return seed_locations


class GridStrategy(SeedingStrategy):
    """Release particles on a regular grid.

    Notes
    -----
    The grid is defined by the distance between particles (``dx``, ``dy``). The
    seeding area can be defined as:

    - ``bbox``: axis-aligned bounding box — dict with ``xmin/ymin/xmax/ymax``
      keys, or a string ``'xmin,ymin xmax,ymax'``.
    - ``poly``: an arbitrary polygon given as a list of ``'x,y'`` strings or a
      path to a polygon file (``.pol``, CSV, or plain text).  When both are
      given, ``poly`` takes precedence.

    Grid points are generated over the area's bounding box and then filtered
    to those that fall inside the polygon.
    """

    def seed(self, config: PopulationConfig) -> list[Tuple[int, float, float]]:
        """Generate regular-grid seed locations for one population.

        Parameters
        ----------
        config : PopulationConfig
            Population configuration containing ``quantity`` and grid strategy
            settings.

        Returns
        -------
        list of tuple of int and float
            Seed locations as ``(quantity, x, y)`` tuples.

        Raises
        ------
        MissingConfigurationParameter
            If the area definition, separation settings, or particle quantity
            is missing.
        """
        settings = config.strategy_settings
        bbox = settings.get('bbox')
        poly = settings.get('poly', None)

        if poly is None and not bbox:
            raise MissingConfigurationParameter('"bbox" or "poly" must be provided for GridStrategy.')

        separation = settings.get('separation')
        if not separation or 'dx' not in separation or 'dy' not in separation:
            raise MissingConfigurationParameter('"separation" with "dx" and "dy" must be provided for GridStrategy.')

        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for GridStrategy.')

        quantity = int(config.quantity)
        dx = separation['dx']
        dy = separation['dy']

        if poly is not None:
            vertices = _parse_polygon(poly)
            xmin, ymin = vertices.min(axis=0)
            xmax, ymax = vertices.max(axis=0)
            poly_path = Path(vertices)
        else:
            poly_path = None
            if isinstance(bbox, str):
                _bbox = bbox.replace(',', ' ').split()
                if len(_bbox) != 4:
                    raise ValueError(f"Invalid bbox format. Expected 'xmin,ymin xmax,ymax', got: {bbox}")
                xmin, ymin, xmax, ymax = map(float, _bbox)
            else:
                xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

        seed_locations = []
        x = xmin
        while x <= xmax:
            y = ymin
            while y <= ymax:
                if poly_path is None or poly_path.contains_point((x, y), radius=1e-9):
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
        """
        Return seed.

        Parameters
        ----------
        config : PopulationConfig
            Configuration mapping used by the operation.

        Returns
        -------
        list[Tuple[int, float, float]]
            Integer result of the calculation.
        """
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
        """
        Return seed.

        Parameters
        ----------
        config : PopulationConfig
            Configuration mapping used by the operation.

        Returns
        -------
        list[Tuple[int, float, float]]
            Integer result of the calculation.
        """
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
    """Create particle instances from population configuration.

    Notes
    -----
    The factory dispatches to the configured seeding strategy and then creates
    one particle object for each requested release location and quantity.
    """

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

        if int(config.quantity) <= 0:
            return []

        # computes seeding positions using the strategy in config
        burial_depth = getattr(config, 'burial_depth', None)
        positions = StrategyClass.seed(config)
        _log_seeding_box_volume(config, positions)

        # In parallel runs each worker keeps every n_tasks-th expanded particle
        # starting at task_id, giving balanced non-overlapping deterministic subsets.
        task_id, n_tasks = _parallel_task_context_from_env()
        # Build a dedicated local RNG for burial-depth sampling, isolated from
        # other RNG usage. Seeded from the strategy seed when available (e.g.
        # RandomStrategy) so the simulation stays reproducible. For strategies
        # without an explicit seed (point/grid/transect) strategy_seed is None
        # and random.Random(None) seeds from system entropy - burial depths are
        # then non-reproducible across runs for those strategies.
        # TODO: add a dedicated burial_depth.seed config key for full reproducibility.
        strategy_seed = getattr(config, 'strategy_settings', {}).get('seed', None)
        burial_rng = random.Random(strategy_seed)

        particles = []
        for global_index, x, y in _iter_expanded_particles(positions):
            burial_depth_value = _sample_burial_depth(burial_depth, rng=burial_rng)
            if n_tasks > 1 and global_index % n_tasks != task_id:
                continue

            p = ParticleClass()
            p.x = x
            p.y = y
            p.release_time = getattr(config, 'release_start', None)
            p.burial_depth = burial_depth_value

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
    _field_interpolator_multi_with_simplex : Any
        Bound method for interpolating multiple nodal fields while reusing cached simplex ids.
    _position_calculator_with_simplex : Any
        Bound method for advancing particles while reusing cached simplex ids.
    _position_calculator_temporal_with_simplex : Any
        Bound method for temporal particle updates while reusing cached simplex ids.
    _position_calculator_with_boundary_class : Any
        Bound method for advancing particles and returning crossed boundary class codes.
    _position_calculator_temporal_with_boundary_class : Any
        Bound method for temporal updates and crossed boundary class codes.
    _particle_simplices : ndarray
        Cached containing-triangle ids for each particle, used to avoid global point location on every update.
    _particle_simplices_stale : bool
        Whether particle positions may have changed since ``_particle_simplices`` was refreshed.
    _last_moved_particle_indices : ndarray or None
        Particle indices updated by the latest position update.
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
    reference_date: str | np.datetime64 = DEFAULT_REFERENCE_DATE
    particles: Dict = field(init=False, default_factory=dict)  # a dictionary with arrays
    repr_volume: float = field(init=False, default=np.nan)  # representative volume [m³/particle]
    _field_interpolator: Any = field(init=False)
    _field_interpolator_multi: Any = field(init=False)
    _field_interpolator_multi_with_simplex: Any = field(init=False)
    _position_calculator_with_simplex: Any = field(init=False)
    _position_calculator_temporal_with_simplex: Any = field(init=False)
    _position_calculator_with_boundary_class: Any = field(init=False)
    _position_calculator_temporal_with_boundary_class: Any = field(init=False)
    _diffusion_calculator: DiffusionCalculator | None = field(init=False, default=None)
    _particle_simplices: ndarray = field(init=False)
    _particle_simplices_stale: bool = field(init=False, default=True)
    _last_moved_particle_indices: ndarray | None = field(init=False, default=None)
    _current_time: float = field(init=False)
    _field_mixing_depth: ndarray = field(init=False)  # TODO: reserved for later particle-behavior logic
    _field_transport_probability: ndarray = field(init=False)  # TODO: reserved for later pickup logic

    def __post_init__(self):
        if self.grid_geometry is None:
            self.grid_geometry = create_grid_geometry(self.field_x, self.field_y)

        # Reuse methods bound to the shared grid geometry.
        self._field_interpolator = self.grid_geometry.interpolate_field
        self._field_interpolator_multi = self.grid_geometry.interpolate_fields
        self._field_interpolator_multi_with_simplex = self.grid_geometry.interpolate_fields_with_simplex
        self._position_calculator_with_simplex = self.grid_geometry.update_particles_with_simplex
        self._position_calculator_temporal_with_simplex = self.grid_geometry.update_particles_temporal_with_simplex
        self._position_calculator_with_boundary_class = self.grid_geometry.update_particles_with_boundary_class
        self._position_calculator_temporal_with_boundary_class = (
            self.grid_geometry.update_particles_temporal_with_boundary_class
        )
        if self.population_config.diffusion_method == 'brownian' and self.population_config.diffusion_coefficient > 0.0:
            rng = None
            if self.population_config.diffusion_seed is not None:
                rng = np.random.default_rng(self.population_config.diffusion_seed)
            self._diffusion_calculator = DiffusionCalculator(BrownianDiffusionStrategy(), rng=rng)

        # generate particles based on the configuration
        _particles = ParticleFactory.create_particles(self.population_config)
        self.particles = {
            'x': np.array([p.x for p in _particles]),
            'y': np.array([p.y for p in _particles]),
            'release_time': np.array(
                [_release_time_to_seconds(p.release_time, self.reference_date) for p in _particles],
                dtype=float,
            ),
            'burial_depth': np.array([p.burial_depth for p in _particles]),
            'status_left_domain': np.zeros(len(_particles), dtype=bool),
            'status_beached': np.zeros(len(_particles), dtype=bool),
        }
        self._particle_simplices = self.grid_geometry.locate_points(self.particles['x'], self.particles['y'])
        self._mark_particle_simplices_current()
        self._last_moved_particle_indices = None
        self._validate_seed_locations_inside_domain()

        rv = _compute_repr_volume(self.population_config, len(self.particles['x']))
        self.repr_volume = rv if rv is not None else np.nan

        # Store the outer envelope of the domain using shared grid geometry.
        self._outer_envelope = Path(self.grid_geometry.outer_envelope)

    def remove_permanently_buried_particles(self, max_exposure_depth: ndarray) -> int:
        """Remove particles that can never be exposed given the maximum possible exposure.

        A particle at burial depth *d* can only be mobilised if the bed erodes
        and/or the mixing layer deepens enough to reach it.  If

            d > max_erosion(x, y) + max_mixing_depth(x, y)

        for the particle's location, it will stay buried for the entire
        simulation and can be dropped from the particle arrays to save memory
        and computation.

        This method is a no-op when
        ``population_config.remove_permanently_buried`` is ``False`` (the
        default).  Call it once after creating the population and before
        starting the time loop, passing pre-computed nodal arrays.

        Parameters
        ----------
        max_exposure_depth : ndarray
            Per-node field equal to ``max_erosion + max_mixing_depth`` over the
            full simulation period.  Particles whose burial depth exceeds the
            interpolated value at their location are permanently removed.
            Nodes/particles with ``NaN`` values are kept (conservative).

        Returns
        -------
        int
            Number of particles removed (0 when the flag is off or no particle
            qualifies).
        """
        if not self.population_config.remove_permanently_buried:
            return 0

        n_total = len(self.particles['x'])
        if n_total == 0:
            return 0

        pop_name = self.population_config.population_config.get('name', 'unknown')

        # Interpolate max-exposure field to each particle's current position.
        max_exposure = self._field_interpolator(
            max_exposure_depth, self.particles['x'], self.particles['y']
        )
        # NaN means outside the grid — keep those particles (conservative).
        max_exposure = np.where(np.isnan(max_exposure), np.inf, max_exposure)

        keep = self.particles['burial_depth'] <= max_exposure
        n_removed = int(np.sum(~keep))

        if n_removed == 0:
            logger.debug(
                "Population '%s': no permanently buried particles found (all %d particles are potentially mobile).",
                pop_name, n_total,
            )
            return 0

        # Remove particles from every attribute array and the simplex cache.
        for key in list(self.particles.keys()):
            self.particles[key] = self.particles[key][keep]
        self._particle_simplices = self._particle_simplices[keep]
        self._mark_particle_simplices_current()
        self._last_moved_particle_indices = None

        pct = 100.0 * n_removed / n_total
        if n_removed == n_total:
            logger.warning(
                "Population '%s': ALL %d particles removed as permanently buried. "
                "Check burial_depth configuration and max_exposure_depth field.",
                pop_name, n_total,
            )
        else:
            logger.info(
                "Population '%s': removed %d of %d permanently buried particles (%.1f%% of total); "
                "%d particles remain.",
                pop_name, n_removed, n_total, pct, n_total - n_removed,
            )
        return n_removed

    def _validate_seed_locations_inside_domain(self) -> None:
        """Raise a clear configuration error when no seeded particles are inside the field grid."""
        if self._particle_simplices.size == 0:
            return

        inside_count = int(np.count_nonzero(self._particle_simplices >= 0))
        if inside_count > 0:
            return

        x_values = np.asarray(self.particles['x'], dtype=float)
        y_values = np.asarray(self.particles['y'], dtype=float)
        raise ConfigurationError(
            'All seeded particles are outside the input field domain. '
            f'Particle x/y ranges are '
            f'{np.nanmin(x_values):.3f}..{np.nanmax(x_values):.3f} / '
            f'{np.nanmin(y_values):.3f}..{np.nanmax(y_values):.3f}; '
            f'field x/y ranges are '
            f'{np.nanmin(self.field_x):.3f}..{np.nanmax(self.field_x):.3f} / '
            f'{np.nanmin(self.field_y):.3f}..{np.nanmax(self.field_y):.3f}. '
            'Use seed coordinates in the same coordinate system as the input model grid.'
        )

    def _mark_particle_simplices_current(self) -> None:
        """Mark cached simplex ids as representing current particle positions."""
        self._particle_simplices_stale = False

    def _invalidate_particle_simplices(self) -> None:
        """Mark cached simplex ids stale after external particle-coordinate edits."""
        self._particle_simplices_stale = True

    def _particle_simplices_match_positions(self) -> bool:
        """Return whether cached simplex ids represent the current particle positions."""
        n_particles = len(self.particles['x'])
        if self._particle_simplices.shape[0] != n_particles:
            return False
        return not self._particle_simplices_stale

    def _refresh_particle_simplices(self) -> None:
        """Refresh cached simplex ids from the current particle coordinates."""
        simplex_ids = self._particle_simplices
        if simplex_ids.shape[0] != len(self.particles['x']):
            simplex_ids = None
        self._particle_simplices = self.grid_geometry.locate_points(
            self.particles['x'],
            self.particles['y'],
            simplex_ids,
        )
        self._mark_particle_simplices_current()

    def update_information(
        self, current_time: Union[int, float], mixing_depth: Any, transport_probability: Any, bed_level: Any
    ) -> None:
        """
        Update scalar field data for particles in the population.

        Parameters
        ----------
        current_time : float, int
            The current time in the simulation.
        mixing_depth : ndarray or mapping
            The mixing depth of the flow field.
        transport_probability : ndarray, mapping, or scalar
            The probability of particle transport in the flow field.
        bed_level : ndarray or mapping
            The bed level of the flow field.

        Notes
        -----
        Static array fields are batched into one interpolation pass so their
        cached simplex lookup is shared. Temporal fields are sampled through
        their lower/upper slices because each field carries its own interpolation
        weight. ``bed_level_previous`` is copied before the update so burial
        depth can later distinguish temporal bed-level change from movement.
        """

        self._current_time = current_time

        if 'bed_level' in self.particles:
            self.particles['bed_level_previous'] = self.particles['bed_level'].copy()

        batched_fields = []
        batched_names = []
        for name, field_value in (
            ('mixing_depth', mixing_depth),
            ('transport_probability', transport_probability),
            ('bed_level', bed_level),
        ):
            if self._can_batch_particle_field(field_value):
                batched_names.append(name)
                batched_fields.append(np.asarray(field_value))
            else:
                self._update_particle_field(name, field_value)

        if batched_fields:
            particle_values = self._interpolate_particle_fields(
                tuple(batched_fields),
            )
            for name, values in zip(batched_names, particle_values, strict=True):
                if self._values_are_all_nan(values):
                    continue
                self.particles[name] = values

        if 'bed_level_previous' not in self.particles and 'bed_level' in self.particles:
            self.particles['bed_level_previous'] = self.particles['bed_level'].copy()

    @staticmethod
    def _can_batch_particle_field(field_value) -> bool:
        """Return whether a field can join one multi-field interpolation pass."""
        if field_value is None or np.isscalar(field_value) or _is_temporal_field(field_value):
            return False

        field_array = np.asarray(field_value)
        return field_array.size > 0

    def _interpolate_particle_fields(self, fields):
        """Interpolate fields at particle positions and refresh cached simplex ids."""
        simplex_ids = self._particle_simplices
        if self._particle_simplices_stale or simplex_ids.shape[0] != len(self.particles['x']):
            simplex_ids = None
        particle_values, simplices = self._field_interpolator_multi_with_simplex(
            tuple(fields),
            self.particles['x'],
            self.particles['y'],
            simplex_ids=simplex_ids,
        )
        if simplices.shape[0] == len(self.particles['x']):
            self._particle_simplices = simplices
            self._mark_particle_simplices_current()
        return particle_values

    def _interpolate_particle_fields_at_indices(self, fields, particle_indices):
        """Interpolate fields for selected particles and refresh their simplex ids."""
        particle_indices = np.asarray(particle_indices, dtype=np.int64)
        if particle_indices.size == 0:
            return tuple(np.empty(0, dtype=float) for _ in fields)

        n_particles = len(self.particles['x'])
        if particle_indices.size == n_particles and np.array_equal(
            particle_indices,
            np.arange(n_particles, dtype=np.int64),
        ):
            return self._interpolate_particle_fields(fields)

        simplex_ids = None
        if self._particle_simplices.shape[0] == n_particles:
            simplex_ids = self._particle_simplices[particle_indices]

        particle_values, simplices = self._field_interpolator_multi_with_simplex(
            tuple(fields),
            self.particles['x'][particle_indices],
            self.particles['y'][particle_indices],
            simplex_ids=simplex_ids,
        )
        if simplices.shape[0] == particle_indices.shape[0] and self._particle_simplices.shape[0] == n_particles:
            self._particle_simplices[particle_indices] = simplices
        return particle_values

    @staticmethod
    def _values_are_all_nan(values) -> bool:
        """Return whether all values are NaN, avoiding a full scan when possible."""
        values = np.asarray(values)
        if values.size == 0:
            return True
        first = values.ravel()[0]
        return bool(np.isnan(first) and np.isnan(values).all())

    def _status_array(self, name: str, n_particles: int) -> np.ndarray:
        """Return a reusable boolean status array with the requested length."""
        values = self.particles.get(name)
        if not isinstance(values, np.ndarray) or values.dtype != bool or values.shape != (n_particles,):
            values = np.empty(n_particles, dtype=bool)
            self.particles[name] = values
        return values

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
                lower_particle_values = self._interpolate_particle_fields((lower_values,))[0]
                if self._values_are_all_nan(lower_particle_values):
                    return
                self.particles[name] = lower_particle_values
                return

            lower_particle_values, upper_particle_values = self._interpolate_particle_fields(
                (lower_values, upper_values),
            )
            if self._values_are_all_nan(lower_particle_values) and self._values_are_all_nan(upper_particle_values):
                return
            self.particles[name] = lower_particle_values + weight * (upper_particle_values - lower_particle_values)
            return

        if np.isscalar(field_value):
            self.particles[name] = np.full(len(self.particles['x']), field_value, dtype=float)
            return

        field_array = np.asarray(field_value)
        if field_array.size == 0:
            return

        particle_values = self._interpolate_particle_fields((field_array,))[0]
        if self._values_are_all_nan(particle_values):
            return

        self.particles[name] = particle_values

    def update_burial_depth(self) -> None:
        """Update the burial depth of particles in the population.

        Notes
        -----
        Invariant: ``particles['bed_level_previous']`` always holds the bed level
        at the particle's *current* position at the *previous* timestep, because
        ``update_bed_level_change_after_movement`` re-samples bed level at the new
        position after every move.  The difference below is therefore a pure
        temporal change (zero for a static bed; equal to local morphodynamic
        accretion/erosion for a dynamic bed).  No spatial correction is needed.
        """
        if len(self.particles['x']) == 0:
            return

        bed_level_change = self.particles['bed_level'] - self.particles['bed_level_previous']
        self.particles['burial_depth'] += bed_level_change
        self.particles['burial_depth'] = np.maximum(self.particles['burial_depth'], 0.0)
        self.particles['z'] = self.particles['bed_level'] - self.particles['burial_depth']

    def update_bed_level_change_after_movement(self, bed_level) -> None:
        """Re-sample bed level at the new particle positions after movement.

        Parameters
        ----------
        bed_level : array_like, mapping, or scalar
            Bed-level field used to update particle bed elevation and elevation
            ``z`` after movement.

        Notes
        -----
        Temporal bed-level fields are re-sampled for all particles, including
        particles that did not move, because the bed can change between field
        slices. Static array fields use ``_last_moved_particle_indices`` to
        update only the particles whose coordinates changed.

        This preserves the invariant required by ``update_burial_depth``: after
        this call ``particles['bed_level']`` holds BL at the *new* position at the
        *current* timestep, so it becomes the correct ``bed_level_previous``
        reference in the next iteration.  ``z`` is also updated here so that the
        value written to output reflects the post-move position.
        """
        if len(self.particles['x']) == 0:
            return

        n_particles = len(self.particles['x'])
        if _is_temporal_field(bed_level):
            self._update_particle_field('bed_level', bed_level)
            self.particles['z'] = self.particles['bed_level'] - self.particles['burial_depth']
            self._last_moved_particle_indices = np.empty(0, dtype=np.int64)
            return

        moved_indices = self._last_moved_particle_indices
        if (
            moved_indices is None
            or moved_indices.shape[0] == n_particles
            or not self._can_batch_particle_field(bed_level)
        ):
            # Scalars and non-batchable fields may carry new values even when no
            # particle moved (e.g. a time-varying scalar bed level), so always
            # apply the full update for them.
            self._update_particle_field('bed_level', bed_level)
            self.particles['z'] = self.particles['bed_level'] - self.particles['burial_depth']
            self._last_moved_particle_indices = np.empty(0, dtype=np.int64)
            return

        if moved_indices.size == 0:
            return

        moved_bed_level = self._interpolate_particle_fields_at_indices(
            (np.asarray(bed_level),),
            moved_indices,
        )[0]

        self.particles['bed_level'][moved_indices] = moved_bed_level
        self.particles['z'][moved_indices] = (
            self.particles['bed_level'][moved_indices] - self.particles['burial_depth'][moved_indices]
        )
        self._last_moved_particle_indices = np.empty(0, dtype=np.int64)


    def update_status(self) -> None:
        """Update boolean status arrays for particles in the population.

        Notes
        -----
        The status arrays are reused in place when their shape and dtype match
        the current population. The domain status uses cached simplex ids and
        refreshes them only when the cached coordinates no longer match the
        current particle positions.
        """
        n_particles = len(self.particles['x'])
        left_domain = self.particles.get('status_left_domain')
        if left_domain is None or left_domain.shape != (n_particles,):
            left_domain = np.zeros(n_particles, dtype=bool)
        else:
            left_domain = np.asarray(left_domain, dtype=bool)
        self.particles['status_left_domain'] = left_domain
        self.particles['status_beached'] = np.zeros(n_particles, dtype=bool)

        if n_particles == 0:
            for status_name in (
                'status_alive',
                'status_buried',
                'status_domain',
                'status_released',
                'status_transported',
                'status_mobile',
            ):
                self.particles[status_name] = np.zeros(0, dtype=bool)
            return

        if not self._particle_simplices_match_positions():
            self._refresh_particle_simplices()
        self._particle_simplices[left_domain] = -1
        self._mark_particle_simplices_current()

        status_transported = self._status_array('status_transported', n_particles)
        status_domain = self._status_array('status_domain', n_particles)
        status_buried = self._status_array('status_buried', n_particles)
        status_released = self._status_array('status_released', n_particles)
        status_alive = self._status_array('status_alive', n_particles)
        status_mobile = self._status_array('status_mobile', n_particles)

        # Compute whether particles are transported (or trapped) based on transport probability
        # Note: If "reduced_velocity" is chosen, "transport_probability" always equals one.
        transport_probability_method = self.population_config.population_config.get(
            'transport_probability', 'no_probability'
        )
        if transport_probability_method == 'no_probability':
            status_transported.fill(True)
        else:
            np.less(np.random.rand(n_particles), self.particles['transport_probability'], out=status_transported)

        np.greater_equal(self._particle_simplices, 0, out=status_domain)
        np.logical_and(status_domain, ~left_domain, out=status_domain)

        # New conditional logic based on transport_probability_method
        if transport_probability_method == 'no_probability':
            # For no_probability method, all particles are considered exposed (not buried)
            status_buried.fill(False)
        else:
            # For stochastic_transport and reduced_velocity methods, use burial_depth vs mixing_depth

            # if van westen method:
            # a particle is considered buried if it is deeper than or equal to the mixing depth
            np.greater_equal(self.particles['burial_depth'], self.particles['mixing_depth'], out=status_buried)

            # if soulsby method:
            # self.particles['status_buried'] = (this is where we implement Soulsby's F based on a and b)

        # Compute whether particles are released (or retained)
        np.less_equal(self.particles['release_time'], self._current_time, out=status_released)

        # Compute whether particles are alive (or dead) (still TODO)
        np.logical_not(left_domain, out=status_alive)

        # Compute whether particles are mobile (or static) - combination of all status flags
        np.logical_not(status_buried, out=status_mobile)
        np.logical_and(status_mobile, status_domain, out=status_mobile)
        np.logical_and(status_mobile, status_alive, out=status_mobile)
        np.logical_and(status_mobile, status_released, out=status_mobile)
        np.logical_and(status_mobile, status_transported, out=status_mobile)

    def update_position(self, flow_field: Dict, current_timestep: float) -> None:
        """
        Update the position of particles in the population based on the flow field.

        Parameters
        ----------
        flow_field : Dict
            A dictionary containing the flow field information.
        current_timestep : float
            The current time step in the simulation in seconds.

        Notes
        -----
        Only mobile particles are integrated. Their refreshed simplex ids and
        particle indices are retained so static post-move bed-level updates can
        avoid re-sampling particles that did not move.
        """

        if len(self.particles['x']) == 0:
            self._last_moved_particle_indices = np.empty(0, dtype=np.int64)
            return

        ix = np.asarray(self.particles['status_mobile'], dtype=bool).copy()  # Freeze current mobile-particle mask.
        particle_indices = np.flatnonzero(ix)
        if particle_indices.size == 0:
            self._last_moved_particle_indices = np.empty(0, dtype=np.int64)
            return
        old_x = self.particles['x'][ix].copy()
        old_y = self.particles['y'][ix].copy()
        old_simplices = self._particle_simplices[particle_indices].copy()

        if _is_temporal_flow_field(flow_field):
            new_x, new_y, new_simplices, boundary_class_codes = self._position_calculator_temporal_with_boundary_class(
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
            new_x, new_y, new_simplices, boundary_class_codes = self._position_calculator_with_boundary_class(
                self.particles['x'][ix],
                self.particles['y'][ix],
                flow_field['u'],
                flow_field['v'],
                current_timestep,
                simplex_ids=self._particle_simplices[particle_indices],
            )

        # TODO: implement Bart's solution for gross/net values here. Add
        outside_domain = new_simplices < 0
        if np.any(outside_domain):
            outside_boundary_class_codes = np.asarray(boundary_class_codes[outside_domain], dtype=np.int8)
            outside_particle_indices = particle_indices[outside_domain]
            self.particles['status_domain'][outside_particle_indices] = False
            self.particles['status_mobile'][outside_particle_indices] = False

            open_boundary = outside_boundary_class_codes == BOUNDARY_CLASS_OPEN
            if np.any(open_boundary):
                open_indices = outside_particle_indices[open_boundary]
                self.particles['status_left_domain'][open_indices] = True
                self.particles['status_alive'][open_indices] = False

            land_boundary = outside_boundary_class_codes == BOUNDARY_CLASS_LAND
            if np.any(land_boundary):
                land_local_indices = np.flatnonzero(outside_domain)[land_boundary]
                land_particle_indices = outside_particle_indices[land_boundary]
                new_x[land_local_indices] = old_x[land_local_indices]
                new_y[land_local_indices] = old_y[land_local_indices]
                new_simplices[land_local_indices] = old_simplices[land_local_indices]
                self.particles['status_beached'][land_particle_indices] = True
                self.particles['status_domain'][land_particle_indices] = True

        if self._diffusion_calculator is not None:
            diffusable = ~outside_domain
            if np.any(diffusable):
                local_indices = np.flatnonzero(diffusable)
                start_x = new_x[local_indices]
                start_y = new_y[local_indices]
                start_simplices = new_simplices[local_indices]
                diffused_x, diffused_y = self._diffusion_calculator.calc_diffusion(
                    start_x, start_y, np.zeros_like(start_x), np.zeros_like(start_y),
                    self.population_config.diffusion_coefficient, current_timestep,
                )
                diffused_simplices = self.grid_geometry.locate_points(
                    diffused_x, diffused_y, start_simplices
                )
                diffused_outside = diffused_simplices < 0
                if np.any(diffused_outside):
                    crossed_classes = self.grid_geometry.classify_boundary_crossings(
                        start_x[diffused_outside], start_y[diffused_outside],
                        diffused_x[diffused_outside], diffused_y[diffused_outside],
                    )
                    outside_particles = particle_indices[local_indices[diffused_outside]]
                    self.particles['status_domain'][outside_particles] = False
                    self.particles['status_mobile'][outside_particles] = False
                    open_boundary = crossed_classes == 'open'
                    if np.any(open_boundary):
                        open_indices = outside_particles[open_boundary]
                        self.particles['status_left_domain'][open_indices] = True
                        self.particles['status_alive'][open_indices] = False
                    land_boundary = crossed_classes == 'land'
                    if np.any(land_boundary):
                        land_local = np.flatnonzero(diffused_outside)[land_boundary]
                        land_particles = outside_particles[land_boundary]
                        diffused_x[land_local] = start_x[land_local]
                        diffused_y[land_local] = start_y[land_local]
                        diffused_simplices[land_local] = start_simplices[land_local]
                        self.particles['status_beached'][land_particles] = True
                        self.particles['status_domain'][land_particles] = True
                        self.particles['status_mobile'][land_particles] = False
                new_x[local_indices] = diffused_x
                new_y[local_indices] = diffused_y
                new_simplices[local_indices] = diffused_simplices

        self.particles['x'][ix] = new_x
        self.particles['y'][ix] = new_y
        self._particle_simplices[particle_indices] = new_simplices
        self._mark_particle_simplices_current()
        self._last_moved_particle_indices = particle_indices


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
        grid_geometry = create_grid_geometry(
            sedtrails_data.x,
            sedtrails_data.y,
            triangles=_geometry_triangles_from_field_data(sedtrails_data),
            boundary_edge_classification=getattr(sedtrails_data, 'boundary_edge_classification', None),
        )
        for pop_config in self.population_configs:
            config = PopulationConfig(population_config=pop_config)
            pop = ParticlePopulation(
                field_x=sedtrails_data.x,
                field_y=sedtrails_data.y,
                population_config=config,
                grid_geometry=grid_geometry,
                reference_date=getattr(sedtrails_data, 'reference_date', DEFAULT_REFERENCE_DATE),
            )
            populations.append(pop)
        return populations


def _geometry_triangles_from_field_data(sedtrails_data: HasFieldCoordinates) -> np.ndarray | None:
    """Return triangle connectivity compatible with the provided x/y coordinates."""

    connectivity = getattr(sedtrails_data, 'particle_face_connectivity', None)
    if connectivity is None:
        connectivity = getattr(sedtrails_data, 'face_node_connectivity', None)
    if connectivity is None:
        return None

    triangles = np.asarray(connectivity, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        return None
    if triangles.shape[0] == 0:
        return triangles

    n_points = np.asarray(sedtrails_data.x).size
    valid = triangles >= 0
    if np.any(valid) and int(np.max(triangles[valid])) >= n_points:
        return None
    if np.any(np.sum(valid, axis=1) != 3):
        return None
    return triangles


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
