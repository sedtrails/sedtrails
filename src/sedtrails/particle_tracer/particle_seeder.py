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


_VERTICAL_POSITION_BED_STATE_ZERO_BURIAL_MODES = {'bed', 'height_above_bed', 'centroid_on_release'}


def _burial_depth_for_vertical_position_mode(sampled_burial_depth: float, vertical_position_mode: str) -> float:
    """Return the burial depth used by 2D/status logic for a vertical seeding mode.

    Q3D keeps ``vertical_position_mode`` and ``vertical_position_value`` so it can
    resolve full z/z_p after bed level and water depth are known. Methods that do
    not track z still need a bed-state burial depth, so above-bed/on-bed modes
    map to zero burial depth.
    """
    mode = str(vertical_position_mode or 'burial_depth').lower().replace('-', '_')
    if mode in _VERTICAL_POSITION_BED_STATE_ZERO_BURIAL_MODES:
        return 0.0
    return float(sampled_burial_depth)


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


def _safe_divide(numerator, denominator, fill=0.0):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), fill, dtype=float)
    return np.divide(numerator, denominator, out=out, where=denominator != 0.0)


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
    burial_depth: float | dict[str, float] = field(init=False, default=0.0)
    vertical_position_mode: str = field(init=False, default='burial_depth')
    vertical_position_value: Any = field(init=False, default=None)
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

        vertical_position = find_value(self.population_config, 'seeding.vertical_position', None)
        if vertical_position is None:
            return
        if not isinstance(vertical_position, dict):
            raise TypeError('"seeding.vertical_position" must be a dictionary with a mode and optional value.')

        mode = str(vertical_position.get('mode', 'burial_depth')).lower()
        mode_aliases = {
            'buried': 'burial_depth',
            'in_bed': 'burial_depth',
            'on_bed': 'bed',
            'height': 'height_above_bed',
            'z_p': 'height_above_bed',
            'z': 'absolute_z',
            'centroid': 'centroid_on_release',
            'transport_centroid': 'centroid_on_release',
        }
        mode = mode_aliases.get(mode, mode)
        allowed_modes = {'burial_depth', 'bed', 'height_above_bed', 'absolute_z', 'centroid_on_release'}
        if mode not in allowed_modes:
            raise ValueError(
                '"seeding.vertical_position.mode" must be one of '
                f'{sorted(allowed_modes)}, got {mode!r}.'
            )
        if mode in {'height_above_bed', 'absolute_z'} and 'value' not in vertical_position:
            raise MissingConfigurationParameter(
                f'"seeding.vertical_position.value" is required for mode {mode!r}.'
            )

        self.vertical_position_mode = mode
        self.vertical_position_value = vertical_position.get('value')


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
        vertical_position_mode = getattr(config, 'vertical_position_mode', 'burial_depth')
        vertical_position_value = getattr(config, 'vertical_position_value', None)
        positions = StrategyClass.seed(config)
        _log_seeding_box_volume(config, positions)

        # Build a dedicated local RNG for burial-depth sampling, isolated from
        # other RNG usage. Seeded from the strategy seed when available (e.g.
        # RandomStrategy) so the simulation stays reproducible. For strategies
        # without an explicit seed (point/grid/transect) strategy_seed is None
        # and random.Random(None) seeds from system entropy — burial depths are
        # then non-reproducible across runs for those strategies.
        # TODO: add a dedicated burial_depth.seed config key for full reproducibility.
        strategy_seed = getattr(config, 'strategy_settings', {}).get('seed', None)
        burial_rng = random.Random(strategy_seed)

        particles = []
        for qty, x, y in positions:
            for _ in range(qty):
                p = ParticleClass()
                p.x = x
                p.y = y
                p.release_time = getattr(config, 'release_start', None)
                sampled_burial_depth = _sample_burial_depth(burial_depth, rng=burial_rng)
                p.burial_depth = _burial_depth_for_vertical_position_mode(
                    sampled_burial_depth,
                    vertical_position_mode,
                )
                p.vertical_position_mode = vertical_position_mode
                p.vertical_position_value = vertical_position_value

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
        vertical_position_values = [
            np.nan if getattr(p, 'vertical_position_value', None) is None else p.vertical_position_value
            for p in _particles
        ]
        self.particles = {
            'x': np.array([p.x for p in _particles]),
            'y': np.array([p.y for p in _particles]),
            'release_time': np.array(
                [_release_time_to_seconds(p.release_time, self.reference_date) for p in _particles],
                dtype=float,
            ),
            'burial_depth': np.array([p.burial_depth for p in _particles]),
            'vertical_position_mode': np.array(
                [getattr(p, 'vertical_position_mode', 'burial_depth') for p in _particles],
                dtype='<U32',
            ),
            'vertical_position_value': np.array(vertical_position_values, dtype=float),
            'vertical_position_initialized': np.zeros(len(_particles), dtype=bool),
            'status_suspended': np.zeros(len(_particles), dtype=bool),
            'status_deposited': np.ones(len(_particles), dtype=bool),
            'status_buried': np.array([p.burial_depth > 0.0 for p in _particles], dtype=bool),
            'status_left_domain': np.zeros(len(_particles), dtype=bool),
            'status_beached': np.zeros(len(_particles), dtype=bool),
        }
        self._particle_simplices = self.grid_geometry.locate_points(self.particles['x'], self.particles['y'])
        self._mark_particle_simplices_current()
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
                if np.isnan(values).all():
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

    def _update_particle_field(self, name: str, field_value) -> None:
        self._update_particle_fields({name: field_value})

    def _particle_interpolation_target(self, indices=None):
        if indices is None:
            return None, self.particles['x'], self.particles['y'], len(self.particles['x'])

        indices = np.asarray(indices)
        if indices.dtype == bool:
            indices = np.flatnonzero(indices)
        else:
            indices = indices.astype(np.int64, copy=False).ravel()

        return indices, self.particles['x'][indices], self.particles['y'][indices], indices.size

    def _assign_particle_field_values(self, name: str, values, indices=None) -> None:
        values = np.asarray(values, dtype=float)
        if indices is None:
            self.particles[name] = values
            return

        n_particles = len(self.particles['x'])
        existing = self.particles.get(name)
        if existing is None or np.asarray(existing).shape != (n_particles,):
            target = np.full(n_particles, np.nan, dtype=float)
        else:
            target = np.asarray(existing, dtype=float)

        target[indices] = values
        self.particles[name] = target

    def _update_particle_fields(self, field_values: Dict[str, Any], indices=None) -> None:
        indices, x_points, y_points, target_size = self._particle_interpolation_target(indices)
        if target_size == 0:
            return

        arrays_to_interpolate = []
        interpolation_jobs = []

        for name, field_value in field_values.items():
            if field_value is None:
                continue

            if _is_temporal_field(field_value):
                lower_values = np.asarray(field_value['lower'])
                upper_values = np.asarray(field_value['upper'])
                if lower_values.size == 0:
                    continue

                weight = float(field_value['weight'])
                lower_is_scalar = lower_values.size == 1
                upper_is_scalar = upper_values.size == 1

                if weight <= 0.0 or lower_values is upper_values or upper_values.size == 0:
                    if lower_is_scalar:
                        self._assign_particle_field_values(
                            name,
                            np.full(target_size, float(lower_values.ravel()[0]), dtype=float),
                            indices=indices,
                        )
                    else:
                        interpolation_jobs.append(('single', name))
                        arrays_to_interpolate.append(lower_values)
                    continue

                if lower_is_scalar and upper_is_scalar:
                    lower_value = float(lower_values.ravel()[0])
                    upper_value = float(upper_values.ravel()[0])
                    self._assign_particle_field_values(
                        name,
                        np.full(target_size, lower_value + weight * (upper_value - lower_value), dtype=float),
                        indices=indices,
                    )
                    continue

                if lower_is_scalar:
                    interpolation_jobs.append(('lower_scalar_temporal', name, float(lower_values.ravel()[0]), weight))
                    arrays_to_interpolate.append(upper_values)
                    continue

                if upper_is_scalar:
                    interpolation_jobs.append(('upper_scalar_temporal', name, float(upper_values.ravel()[0]), weight))
                    arrays_to_interpolate.append(lower_values)
                    continue

                interpolation_jobs.append(('temporal', name, weight))
                arrays_to_interpolate.extend((lower_values, upper_values))
                continue

            if np.isscalar(field_value):
                self._assign_particle_field_values(
                    name,
                    np.full(target_size, field_value, dtype=float),
                    indices=indices,
                )
                continue

            field_array = np.asarray(field_value)
            if field_array.size == 0:
                continue
            if field_array.size == 1:
                self._assign_particle_field_values(
                    name,
                    np.full(target_size, float(field_array.ravel()[0]), dtype=float),
                    indices=indices,
                )
                continue

            interpolation_jobs.append(('single', name))
            arrays_to_interpolate.append(field_array)

        if not arrays_to_interpolate:
            return

        if indices is None:
            interpolated_values = self._interpolate_particle_fields(tuple(arrays_to_interpolate))
        else:
            interpolated_values = self._field_interpolator_multi(tuple(arrays_to_interpolate), x_points, y_points)
        value_index = 0
        for job in interpolation_jobs:
            kind = job[0]
            name = job[1]
            if kind == 'single':
                particle_values = interpolated_values[value_index]
                value_index += 1
            elif kind == 'temporal':
                weight = job[2]
                lower_particle_values = interpolated_values[value_index]
                upper_particle_values = interpolated_values[value_index + 1]
                value_index += 2
                if np.isnan(lower_particle_values).all() and np.isnan(upper_particle_values).all():
                    continue
                particle_values = lower_particle_values + weight * (upper_particle_values - lower_particle_values)
            elif kind == 'lower_scalar_temporal':
                lower_value = job[2]
                weight = job[3]
                upper_particle_values = interpolated_values[value_index]
                value_index += 1
                particle_values = lower_value + weight * (upper_particle_values - lower_value)
            else:
                upper_value = job[2]
                weight = job[3]
                lower_particle_values = interpolated_values[value_index]
                value_index += 1
                particle_values = lower_particle_values + weight * (upper_value - lower_particle_values)

            if np.isnan(particle_values).all():
                continue
            self._assign_particle_field_values(name, particle_values, indices=indices)

    def _update_particle_flow_field(self, prefix: str, flow_field: Dict, indices=None) -> None:
        # This method updates the particle flow field attributes (u, v, magnitude) at the particle positions, handling both temporal and non-temporal flow fields.
        # This is because the particle level Macdonald needs flow direction and flow magnitude to compute particle motion.
        # For temporal flow fields, it performs interpolation between the lower and upper time steps based on the provided weight.
        # The resulting flow field attributes are stored in the particles dictionary with keys prefixed by the given prefix, e.g. depth_avg_flow_velocity_u/v/magnitude.

        indices, x_points, y_points, target_size = self._particle_interpolation_target(indices)
        if target_size == 0:
            return

        if _is_temporal_flow_field(flow_field):
            weight = flow_field['weight']
            if weight <= 0.0 or flow_field['lower'] is flow_field['upper']:
                lower_u, lower_v, lower_magnitude = self._field_interpolator_multi(
                    (
                        np.asarray(flow_field['lower']['u']),
                        np.asarray(flow_field['lower']['v']),
                        np.asarray(flow_field['lower']['magnitude']),
                    ),
                    x_points,
                    y_points,
                )
                self._assign_particle_field_values(f'{prefix}_u', lower_u, indices=indices)
                self._assign_particle_field_values(f'{prefix}_v', lower_v, indices=indices)
                self._assign_particle_field_values(f'{prefix}_magnitude', lower_magnitude, indices=indices)
                return

            lower_u, lower_v, lower_magnitude, upper_u, upper_v, upper_magnitude = self._field_interpolator_multi(
                (
                    np.asarray(flow_field['lower']['u']),
                    np.asarray(flow_field['lower']['v']),
                    np.asarray(flow_field['lower']['magnitude']),
                    np.asarray(flow_field['upper']['u']),
                    np.asarray(flow_field['upper']['v']),
                    np.asarray(flow_field['upper']['magnitude']),
                ),
                x_points,
                y_points,
            )
            self._assign_particle_field_values(f'{prefix}_u', lower_u + weight * (upper_u - lower_u), indices=indices)
            self._assign_particle_field_values(f'{prefix}_v', lower_v + weight * (upper_v - lower_v), indices=indices)
            self._assign_particle_field_values(
                f'{prefix}_magnitude',
                lower_magnitude + weight * (upper_magnitude - lower_magnitude),
                indices=indices,
            )
            return

        u, v, magnitude = self._field_interpolator_multi(
            (
                np.asarray(flow_field['u']),
                np.asarray(flow_field['v']),
                np.asarray(flow_field['magnitude']),
            ),
            x_points,
            y_points,
        )
        self._assign_particle_field_values(f'{prefix}_u', u, indices=indices)
        self._assign_particle_field_values(f'{prefix}_v', v, indices=indices)
        self._assign_particle_field_values(f'{prefix}_magnitude', magnitude, indices=indices)

    @staticmethod
    def _loglaw_velocity_at_z(shear_velocity, z, roughness_height):
        shear_velocity = np.asarray(shear_velocity, dtype=float)
        z = np.asarray(z, dtype=float)
        roughness_height = np.asarray(roughness_height, dtype=float)
        velocity = np.zeros_like(z, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            argument = 30.0 * z / roughness_height
            valid = (
                (argument > 1.0)
                & np.isfinite(argument)
                & np.isfinite(shear_velocity)
                & (shear_velocity > 0.0)
            )
            velocity[valid] = 2.5 * shear_velocity[valid] * np.log(argument[valid])
        return velocity

    @staticmethod
    def _apply_q3d_velocity_deficit(u_zp, u_1p4zc, z_p, z_c, deficit_coefficient):
        """
        Apply the Q3-D horizontal velocity deficit formulation of
        MacDonald et al. (2006), Eq. (40), at particle positions.

        This function reduces horizontal particle advection velocity
        to account for intermittent particle-bed interaction in Q3-D mode.
        The reduction depends on particle elevation relative to the
        transport centroid and on the velocity deficit coefficient c_A
        (Eq. 39).

        Parameters
        ----------
        u_zp : array
            Log-law horizontal velocity u(z_p) at the current particle height [m/s].
        u_1p4zc : array
            Log-law horizontal velocity u(1.4 z_c) at 1.4 times the transport centroid [m/s].
        z_p : array
            Particle height above bed [m].
        z_c : array
            Total-load transport centroid height above bed [m].
        deficit_coefficient : array
            Velocity deficit coefficient c_A [-].

        Returns
        -------
        numpy.ndarray
            Reduced horizontal particle advection velocity [m/s]. The returned velocity equals:
            - 0 at or below the bed,
            - c_A * u(z_p) below the centroid,
            - a linear blend between c_A * u(z_p) and u(1.4 z_c) in the transition zone,
            - u(z_p) above 1.4 z_c.
        """

        z_p = np.asarray(z_p, dtype=float)
        z_c_safe = np.maximum(np.asarray(z_c, dtype=float), 1e-12)
        blending = (z_p - z_c_safe) / (0.4 * z_c_safe)
        u_transition = deficit_coefficient * u_zp + blending * (u_1p4zc - deficit_coefficient * u_zp)
        return np.where(
            z_p <= 0.0,
            0.0,
            np.where(
                z_p <= z_c_safe,
                deficit_coefficient * u_zp,
                np.where(z_p <= 1.4 * z_c_safe, u_transition, u_zp),
            ),
        )

    @staticmethod
    def _turbulent_diffusion_coefficients(
        water_depth,
        z_p,
        flow_velocity_magnitude,
        shear_velocity,
        *,
        K_Et=0.15,
        K_Ev=None,
        M_b=None,
        E_turb_hor_min=0.02,
        E_turb_vert_min=0.0,
        compute_horizontal=True,
        compute_vertical=True,
    ):
        """
        Compute particle-level turbulent diffusion coefficients following MacDonald et al. (2006) equations 45 to 50.

        Parameters
        ----------
        water_depth : array
            Total water depth h [m] interpolated at particle positions.
        z_p : array
            Particle height above bed z_p [m].
        flow_velocity_magnitude : array
            Depth-averaged flow velocity magnitude |U| [m/s] interpolated at particle positions.
        shear_velocity : array
            Shear velocity u_* [m/s] interpolated at particle positions.
        K_Et : float
            Horizontal diffusion scaling coefficient, typically about 0.15-0.6 (below eq 45 in MacDonald et al. (2006)).
        K_Ev : float or None
            Vertical diffusion scaling coefficient. If None, defaults to K_Et.
        M_b : array or None
            Wave-breaking enhancement factor at particle positions. If None, defaults to 1. (eq 47 in MacDonald et al. (2006)).
        E_turb_hor_min : float
            Minimum horizontal turbulent diffusivity [m^2/s]. Default is 0.02 in PTM below eq 48 in MacDonald et al. (2006).
        E_turb_vert_min : float
            Minimum vertical turbulent diffusivity [m^2/s]. Default is 0 in PTM (below eq 49 in MacDonald et al. (2006)).
        compute_horizontal : bool
            If false, skip the horizontal coefficient calculation and return zeros.
        compute_vertical : bool
            If false, skip the vertical coefficient calculation and return zeros.

        Returns
        -------
        tuple of arrays
            Horizontal and vertical turbulent diffusion coefficients [m^2/s].
        """
        if K_Ev is None:
            K_Ev = K_Et

        water_depth = np.asarray(water_depth, dtype=float)
        z_p = np.asarray(z_p, dtype=float)
        flow_velocity_magnitude = np.asarray(flow_velocity_magnitude, dtype=float)
        shear_velocity = np.asarray(shear_velocity, dtype=float)
        if M_b is None:
            M_b = np.ones_like(flow_velocity_magnitude)
        else:
            M_b = np.asarray(M_b, dtype=float)
            # TO DO Vassia: adjust this according to eq 47 in MacDonald et al. (2006) if needed, as the current implementation assumes M_b is directly provided at particle positions.
            # Mb = 1.0 + 5 * H_s within the breaker zone and 1 outside the breaker zone, where H_s is the significant wave height. If wave information is not available, M_b can be set to 1 for all particles as a default.

        # Horizontal diffusion coefficient, equations 46 and 48 in MacDonald et al. (2006).
        horizontal = np.zeros_like(water_depth, dtype=float)
        if compute_horizontal:
            horizontal = M_b * K_Et * water_depth * shear_velocity
            horizontal = np.maximum(np.nan_to_num(horizontal, nan=0.0), E_turb_hor_min)

        # Vertical diffusion coefficient, equations 49 and 50 in MacDonald et al. (2006).
        vertical = np.zeros_like(water_depth, dtype=float)
        if compute_vertical:
            shape = np.zeros_like(water_depth, dtype=float)
            valid = water_depth > 0.0
            shape[valid] = z_p[valid] * (water_depth[valid] - z_p[valid]) ** 2 / water_depth[valid] ** 3
            vertical = M_b * K_Ev * flow_velocity_magnitude * shape
            vertical = np.maximum(np.nan_to_num(vertical, nan=0.0), E_turb_vert_min)
        return horizontal, vertical

    def _initialize_vertical_position(
        self,
        bed_level,
        water_depth,
        entrainment_height,
        z,
        burial_depth,
        is_released,
        is_suspended,
        is_deposited,
        is_buried,
    ):
        """
        Resolve configured initial vertical particle position after bed and flow
        fields are known.

        Supported seeding.vertical_position modes:
        - burial_depth: z = bed_level - burial_depth
        - bed: z = bed_level
        - height_above_bed: z = bed_level + value
        - absolute_z: z = value in the model vertical datum
        - centroid_on_release: z = bed_level + q3d_entrainment_height_above_bed
        """
        n_particles = len(self.particles['x'])
        initialized = np.asarray(
            self.particles.get('vertical_position_initialized', np.isfinite(z)),
            dtype=bool,
        ).copy()
        modes = np.asarray(
            self.particles.get('vertical_position_mode', np.full(n_particles, 'burial_depth')),
            dtype='<U32',
        )
        values = np.asarray(
            self.particles.get('vertical_position_value', np.full(n_particles, np.nan)),
            dtype=float,
        )

        to_initialize = is_released & ~initialized
        if not np.any(to_initialize):
            self.particles['vertical_position_initialized'] = initialized
            return z, burial_depth, is_suspended, is_deposited, is_buried

        for mode in np.unique(modes[to_initialize]):
            idx = to_initialize & (modes == mode)
            if not np.any(idx):
                continue

            if mode == 'bed':
                z[idx] = bed_level[idx]
            elif mode == 'height_above_bed':
                height = np.clip(np.nan_to_num(values[idx], nan=0.0), 0.0, water_depth[idx])
                z[idx] = bed_level[idx] + height
            elif mode == 'absolute_z':
                absolute_z = values[idx].copy()
                absolute_z = np.where(np.isfinite(absolute_z), absolute_z, bed_level[idx])
                z[idx] = np.minimum(absolute_z, bed_level[idx] + water_depth[idx])
            elif mode == 'centroid_on_release':
                height = np.clip(np.nan_to_num(entrainment_height[idx], nan=0.0), 0.0, water_depth[idx])
                z[idx] = bed_level[idx] + height
            else:
                z[idx] = bed_level[idx] - burial_depth[idx]

            height_above_bed = z[idx] - bed_level[idx]
            is_suspended[idx] = height_above_bed > 0.0
            is_buried[idx] = height_above_bed < 0.0
            # Deposited is the bed-state complement of suspended. Buried
            # particles are a bed-state subset, so is_buried implies deposited.
            is_deposited[idx] = ~is_suspended[idx]
            burial_depth[idx] = np.where(is_buried[idx], -height_above_bed, 0.0)
            initialized[idx] = True

        self.particles['vertical_position_initialized'] = initialized
        return z, burial_depth, is_suspended, is_deposited, is_buried

    def _advect_particles_with_velocity(self, active, velocity_x, velocity_y, dt):
        """Move active particles with particle-level horizontal velocities.

        Returns
        -------
        left_domain_indices, beached_indices : tuple[np.ndarray, np.ndarray]
            Particle indices that crossed an open boundary or land boundary.
        """
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        old_x = self.particles['x'][active].copy()
        old_y = self.particles['y'][active].copy()
        old_simplices = self._particle_simplices[active_indices].copy()

        new_x = old_x + velocity_x[active] * dt
        new_y = old_y + velocity_y[active] * dt
        new_simplices = self.grid_geometry.locate_points(
            new_x,
            new_y,
            start_simplices=old_simplices,
        )

        self.particles['x'][active] = new_x
        self.particles['y'][active] = new_y
        self._particle_simplices[active_indices] = new_simplices

        outside_domain = new_simplices < 0
        if not np.any(outside_domain):
            self._mark_particle_simplices_current()
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        outside_particle_indices = active_indices[outside_domain]
        boundary_classes = self.grid_geometry.classify_boundary_crossings(
            old_x[outside_domain],
            old_y[outside_domain],
            new_x[outside_domain],
            new_y[outside_domain],
        )
        boundary_classes = np.asarray(boundary_classes).astype(str)

        land_boundary = boundary_classes == 'land'
        open_boundary = ~land_boundary

        self.particles['status_domain'][outside_particle_indices] = False
        self.particles['status_mobile'][outside_particle_indices] = False

        left_domain_indices = outside_particle_indices[open_boundary]
        if left_domain_indices.size:
            self.particles['status_left_domain'][left_domain_indices] = True
            self.particles['status_alive'][left_domain_indices] = False

        beached_indices = outside_particle_indices[land_boundary]
        if beached_indices.size:
            land_local_indices = np.flatnonzero(outside_domain)[land_boundary]
            self.particles['x'][beached_indices] = old_x[land_local_indices]
            self.particles['y'][beached_indices] = old_y[land_local_indices]
            self._particle_simplices[beached_indices] = old_simplices[land_local_indices]
            self.particles['status_beached'][beached_indices] = True
            self.particles['status_domain'][beached_indices] = True

        self._mark_particle_simplices_current()
        return left_domain_indices, beached_indices

    @staticmethod
    def _select_q3d_entrainment(
        available,
        turbulent_shields,
        critical_shields,
        dt,
        *,
        mode='shields_threshold',
        entrainment_frequency=None,
        probability_law='poisson',
        rng=None,
    ):
        """
        Select available bed particles that enter Q3D suspension.

        Parameters
        ----------
        available : array of bool
            Particles that are released, alive, inside the domain, exposed, and
            not already suspended.
        turbulent_shields : array
            Local turbulent Shields number [-].
        critical_shields : array
            Particle critical Shields number [-].
        dt : float
            Outer particle timestep [s].
        mode : str
            Entrainment rule:
            - shields_threshold: deterministic turbulent_shields > critical_shields.
            - non_zero_particle_velocity: entrain every available particle; the
              later particle-level velocity calculation decides whether it moves.
            - entrainment_frequency: stochastic entrainment from f_e. The
              probability law can be poisson, P = 1 - exp(-f_e dt), or
              linear, P = min(f_e dt, 1), matching the Eq. 68 small-dt form.
        entrainment_frequency : array or None
            Entrainment frequency [1/s] for entrainment_frequency mode,
            usually from the MacDonald q3d_entrainment_frequency field.
        probability_law : str
            Probability conversion used with entrainment_frequency mode:
            poisson uses P = 1 - exp(-f_e dt); linear uses P = min(f_e dt, 1).
        rng : random generator, optional
            Random source used for stochastic entrainment.

        Returns
        -------
        entrained_now, entrainment_probability : tuple of arrays
            Boolean entrainment decision and diagnostic probability [-].
        """
        available = np.asarray(available, dtype=bool)
        mode = str(mode or 'shields_threshold').strip().lower().replace('-', '_')
        probability = np.zeros_like(turbulent_shields, dtype=float)

        if mode in {'shields', 'shields_threshold', 'critical_shields'}:
            can_entrain = (
                np.isfinite(critical_shields)
                & (critical_shields > 0.0)
                & (turbulent_shields > critical_shields)
            )
            probability[can_entrain] = 1.0
            return available & can_entrain, probability

        if mode in {'non_zero_particle_velocity', 'nonzero_particle_velocity', 'all', 'all_available', 'available'}:
            probability[available] = 1.0
            return available.copy(), probability

        if mode in {'entrainment_frequency', 'frequency'}:
            if entrainment_frequency is None:
                raise ValueError('q3d_entrainment_frequency is required when q3d_entrainment_mode is frequency.')
            frequency = np.maximum(np.nan_to_num(entrainment_frequency, nan=0.0), 0.0)
            probability_law = str(probability_law or 'poisson').strip().lower().replace('-', '_')
            frequency_dt = frequency * dt
            if probability_law in {'poisson', 'exponential'}:
                probability = np.clip(-np.expm1(-frequency_dt), 0.0, 1.0)
            elif probability_law in {'linear', 'macdonald', 'macdonald_eq68', 'eq68'}:
                probability = np.clip(frequency_dt, 0.0, 1.0)
            else:
                raise ValueError(
                    "Unsupported q3d_entrainment_probability_law "
                    f"{probability_law!r}. Expected poisson or linear."
                )
            probability = np.broadcast_to(probability, available.shape).astype(float, copy=True)
            random_source = np.random if rng is None else rng
            return available & (random_source.random(len(available)) < probability), probability

        raise ValueError(
            "Unsupported q3d_entrainment_mode "
            f"{mode!r}. Expected shields_threshold, non_zero_particle_velocity, or entrainment_frequency."
        )

    @staticmethod
    def _normalize_q3d_vertical_update_scheme(scheme):
        scheme = str(scheme or 'geometric').strip().lower().replace('-', '_')
        aliases = {
            'geometry': 'geometric',
            'geometric_method': 'geometric',
            'macdonald_geometric': 'geometric',
            'centroid': 'centroid_floor',
            'centroid_floor_method': 'centroid_floor',
            'rouse': 'rouse_profile',
            'rouse_sample': 'rouse_profile',
            'rouse_sampling': 'rouse_profile',
        }
        scheme = aliases.get(scheme, scheme)
        if scheme not in {'geometric', 'centroid_floor', 'rouse_profile'}:
            raise ValueError(
                "Unsupported q3d_vertical_update_scheme "
                f"{scheme!r}. Expected geometric, centroid_floor, or rouse_profile."
            )
        return scheme

    @staticmethod
    def _normalize_q3d_diagnostics_level(level):
        # Legacy minimal/full option kept for backward-compatible YAML files.
        level = str(level or 'minimal').strip().lower().replace('-', '_')
        aliases = {
            'basic': 'minimal',
            'standard': 'minimal',
            'reduced': 'minimal',
            'all': 'full',
            'debug': 'full',
        }
        level = aliases.get(level, level)
        if level not in {'minimal', 'full'}:
            raise ValueError(f"Unsupported q3d_diagnostics level {level!r}. Expected minimal or full.")
        return level

    @staticmethod
    def _sample_rouse_profile_height(water_depth, rouse_number, rng=None):
        """
        Draw particle heights above bed from a fast Rouse-shaped beta distribution.

        This is an intentionally compact approximation for Q3D experiments:
        R=0 gives a uniform water-column sample, and increasing R concentrates
        samples closer to the bed while still allowing occasional high
        suspension. alpha and beta are beta-distribution shape parameters, not
        MacDonald coefficients.
        """
        water_depth = np.maximum(np.nan_to_num(water_depth, nan=0.0), 0.0)
        rouse_number = np.clip(np.nan_to_num(rouse_number, nan=0.0), 0.0, 20.0)
        random_source = np.random if rng is None else rng
        alpha = 1.0 / (1.0 + rouse_number)
        beta = 1.0 + rouse_number
        return water_depth * random_source.beta(alpha, beta)

    def _q3d_height_after_vertical_update(
        self,
        scheme,
        *,
        z_p_old,
        bed_level_old,
        bed_level_new,
        water_depth_new,
        particle_w,
        settling_velocity,
        transport_centroid_elevation_new,
        rouse_number_new,
        dt,
        rng=None,
    ):
        """
        Compute active-particle height above bed after horizontal advection.

        This is the small function to edit when experimenting with Q3D vertical
        position schemes. Inputs are already sliced to active particles.
        """
        if scheme == 'geometric':
            # Convert the vertical update to height above the new bed. z_p_old is
            # relative to the old bed; particle_w * dt is the vertical displacement in
            # absolute space; bed_level_old - bed_level_new corrects for the bed elevation
            # change after horizontal advection.
            z_p_new = z_p_old + particle_w * dt + (bed_level_old - bed_level_new)
        elif scheme == 'centroid_floor':
            z_c_new = np.clip(np.nan_to_num(transport_centroid_elevation_new, nan=0.0), 0.0, water_depth_new) # relative to bed 
            candidate_z = z_p_old - np.maximum(np.nan_to_num(settling_velocity, nan=0.0), 0.0) * dt
            z_p_new = np.maximum(candidate_z, z_c_new)
        elif scheme == 'rouse_profile':
            z_p_new = self._sample_rouse_profile_height(water_depth_new, rouse_number_new, rng=rng)
        else:
            raise ValueError(f'Unsupported q3d vertical update scheme {scheme!r}')

        return np.clip(np.nan_to_num(z_p_new, nan=0.0, posinf=0.0, neginf=0.0), 0.0, water_depth_new)

    def update_q3d_particle_position(
        self,
        current_timestep: float,
        centroid_flow_field: Dict,
        hydrodynamic_flow_field: Dict,
        bed_level_field: Any,
        max_shear_velocity: Any,
        selected_shear_velocity: Any,
        profile_roughness_height: Any,
        total_transport_centroid_elevation: Any,
        q3d_velocity_deficit_coefficient: Any,
        q3d_vertical_velocity_gradient: Any,
        turbulent_shields_number: Any,
        critical_shields_number: Any,
        settling_velocity: Any,
        water_depth: Any,
        skin_roughness_height: Any,
        entrainment_height_above_bed: Any,
        rouse_number: Any = None,
        K_Et: float = 0.15,
        K_Ev: float | None = None,
        M_b: Any = None,
        E_turb_hor_min: float = 0.02,
        E_turb_vert_min: float = 0.0,
        q3d_horizontal_diffusion_enabled: bool = True,
        q3d_entrainment_mode: str = 'shields_threshold',
        q3d_entrainment_frequency: Any = None,
        q3d_entrainment_probability_law: str = 'poisson',
        q3d_vertical_update_scheme: str = 'geometric',
        q3d_motion_substeps: int = 1,
        q3d_save_first_substep_diagnostics: bool | None = None,
        q3d_diagnostics: str = 'minimal',
        rng: Any = None,
    ) -> None:
        """
        Update Q3D particle-resolved horizontal and vertical motion.

        This method applies the MacDonald Q3D equations at particle positions
        instead of on the Eulerian field grid. It interpolates the required
        hydraulic fields to each particle, computes the particle height above
        bed z_p from the current particle elevation z, computes particle-level
        modified centroid particle velocity for horizontal advection, turbulent
        diffusion, vertical advection, settling, and random-walk diffusion, then
        updates x, y, z and derived particle state fields.

        References below are to MacDonald et al. (2006), PTM Report 1: fall
        time/velocity deficit (Eqs. 36-40), Q3D vertical velocity (Eqs. 41-42),
        turbulent diffusion/random walk (Eqs. 45, 49, 51-52), turbulent Shields
        and pickup/entrainment frequency (Eqs. 57-61), and stochastic
        entrainment over dt (Eq. 68).

        Parameters
        ----------
        current_timestep : float
            Particle update timestep dt [s].
        centroid_flow_field : dict
            Configured MacDonald centroid particle velocity field. This is
            interpolated to particle positions for diagnostics only.
        hydrodynamic_flow_field : dict
            Depth-averaged hydrodynamic/D3D-FM flow field used for local flow
            direction and depth-averaged velocity magnitude.
        bed_level_field : array or temporal field
            Bed level field [m] used before and after horizontal advection to
            compute the geometric vertical update over the new bed position.
        max_shear_velocity : array or temporal field
            Maximum shear velocity u_* [m/s].
        profile_roughness_height : array or temporal field
            Roughness height k_s [m] used by the MacDonald log-law velocity profile.
        total_transport_centroid_elevation : array or temporal field
            Total-load transport centroid height z_c above the bed [m].
        q3d_velocity_deficit_coefficient : array or temporal field
            Velocity deficit coefficient c_A [-].
        q3d_vertical_velocity_gradient : array or temporal field
            Continuity-based vertical velocity gradient term, equal to dh/hdt + div(U) [1/s].
        turbulent_shields_number : array or temporal field
            Turbulent Shields number from the MacDonald Q3D field calculation [-].
        critical_shields_number : float or array
            Critical Shields number for this particle population [-]. This is a grain/particle
            property and is usually passed as a scalar, then broadcast to particles.
        settling_velocity : float or array
            Particle settling velocity [m/s]. This is a grain/particle property and is usually
            passed as a scalar, then broadcast to particles.
        water_depth : array or temporal field
            Water depth h [m].
        skin_roughness_height : array or temporal field
            Skin roughness height used for the near-bed deposition threshold [m].
        entrainment_height_above_bed : array or temporal field
            Height above the bed assigned to particles at the instant they entrain [m].
        rouse_number : array or temporal field, optional
            Rouse number [-] used when q3d_vertical_update_scheme is rouse_profile.
        K_Et : float
            Horizontal turbulent diffusion scaling coefficient.
        K_Ev : float or None
            Vertical turbulent diffusion scaling coefficient. If None, defaults to K_Et.
        M_b : array or temporal field, optional
            Wave-breaking enhancement factor. If None, defaults to 1 in the diffusion helper.
        E_turb_hor_min : float
            Minimum horizontal turbulent diffusivity [m^2/s].
        E_turb_vert_min : float
            Minimum vertical turbulent diffusivity [m^2/s].
        q3d_horizontal_diffusion_enabled : bool
            If false, disable horizontal turbulent diffusion while leaving
            geometric-scheme vertical diffusion unchanged.
        q3d_entrainment_mode : str
            Rule used to decide which available bed particles enter suspension.
            Supported values are shields_threshold, non_zero_particle_velocity,
            and entrainment_frequency.
        q3d_entrainment_frequency : float or array, optional
            MacDonald q3d_entrainment_frequency field [1/s] used when
            q3d_entrainment_mode is entrainment_frequency.
        q3d_entrainment_probability_law : str
            Probability law for entrainment_frequency mode. poisson uses
            P = 1 - exp(-f_e dt); linear uses P = min(f_e dt, 1), the
            MacDonald Eq. 68 small-timestep approximation.
        q3d_vertical_update_scheme : str
            Vertical position update scheme. Supported values are:
            - geometric: current MacDonald/geometric bed-change correction.
            - centroid_floor: settling-only absolute vertical update with a
              floor at the new total-transport centroid height.
            - rouse_profile: sample new height above bed from a Rouse-shaped
              distribution at the new horizontal location.
        q3d_motion_substeps : int
            Number of smaller Q3D motion updates inside one particle timestep.
            A value of 1 gives the original single-step update.
        q3d_save_first_substep_diagnostics : bool or None
            If true, store first-substep Q3D hydraulic and diffusion diagnostic
            arrays. If None, the legacy q3d_diagnostics level is used.
        q3d_diagnostics : str
            Legacy Q3D particle diagnostics level. minimal disables the extra
            first-substep fields; full enables them.
        rng : random generator, optional
            Random source used for turbulent random-walk velocities.

        Notes
        -----
        Q3D entrainment is handled by _select_q3d_entrainment. The default
        behavior is deterministic Shields-threshold entrainment, but the same
        transport loop can also use legacy flow-speed, all-available, or
        stochastic entrainment-frequency rules. Available particles that do not
        entrain remain deposited at the bed. Bed particles that are not
        available are marked deposited/not suspended, but their burial_depth is
        not recomputed in this method.

        The calculation order is:
        1. Interpolate/read all required grid fields and particle constants.
        2. Read current particle state arrays.
        3. Resolve configured initial vertical position for newly released particles.
        4. Use the configured Q3D entrainment rule to lift newly entrained particles to
           q3d_entrainment_height_above_bed.
        5. Split dt into q3d_motion_substeps smaller updates.
        6. For each substep, recompute z_p and particle-level Q3D velocities,
           move suspended particles horizontally, then update z with the
           configured vertical update scheme.
        7. Deposit particles that return to the near-bed threshold.
        """

        # ------------------------------------------------------------------
        # 1. Validate timestep and read/interpolate all Q3D inputs.
        #    MacDonald's random-walk velocities and entrainment probability are
        #    timestep dependent (Eqs. 51-52 and 68), so the CFL-limited particle
        #    dt must be known before this method is called.
        # ------------------------------------------------------------------
        dt = float(current_timestep)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f'current_timestep must be positive and finite, got {current_timestep!r}')

        if q3d_motion_substeps is None:
            q3d_motion_substeps = 1
        try:
            substeps = int(q3d_motion_substeps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'q3d_motion_substeps must be a positive integer, got {q3d_motion_substeps!r}'
            ) from exc
        if substeps < 1:
            raise ValueError(f'q3d_motion_substeps must be >= 1, got {q3d_motion_substeps!r}')
        dt_sub = dt / substeps
        vertical_update_scheme = self._normalize_q3d_vertical_update_scheme(q3d_vertical_update_scheme)
        if q3d_save_first_substep_diagnostics is None:
            diagnostics_level = self._normalize_q3d_diagnostics_level(q3d_diagnostics)
            save_first_substep_diagnostics = diagnostics_level == 'full'
        else:
            save_first_substep_diagnostics = bool(q3d_save_first_substep_diagnostics)
        if vertical_update_scheme == 'rouse_profile' and rouse_number is None:
            raise ValueError('rouse_number is required when q3d_vertical_update_scheme is rouse_profile.')

        if hydrodynamic_flow_field is None:
            raise ValueError('hydrodynamic_flow_field is required for Q3D particle motion.')

        water_depth_field = water_depth
        skin_roughness_height_field = skin_roughness_height

        # Store centroid velocity at particle positions for diagnostics.
        self._update_particle_flow_field('centroid_particle_velocity', centroid_flow_field)

        # Use hydrodynamic flow for Q3D movement direction and depth-averaged
        # |U|. The MacDonald centroid velocity from the grid is retained only as
        # a diagnostic, because Q3D recomputes particle velocity at the live z_p.
        self._update_particle_flow_field('depth_avg_flow_velocity', hydrodynamic_flow_field)
        initial_particle_fields = {
            'bed_level': bed_level_field,
            'max_shear_velocity': max_shear_velocity,
            'selected_shear_velocity': selected_shear_velocity,
            'profile_roughness_height': profile_roughness_height,
            'total_transport_centroid_elevation': total_transport_centroid_elevation,
            'q3d_velocity_deficit_coefficient': q3d_velocity_deficit_coefficient,
            'q3d_vertical_velocity_gradient': q3d_vertical_velocity_gradient,
            'turbulent_shields_number': turbulent_shields_number,
            'critical_shields_number': critical_shields_number,
            'settling_velocity': settling_velocity,
            'water_depth': water_depth,
            'skin_roughness_height': skin_roughness_height,
            'q3d_entrainment_height_above_bed': entrainment_height_above_bed,
        }
        if rouse_number is not None:
            initial_particle_fields['rouse_number'] = rouse_number
        if q3d_entrainment_frequency is not None:
            initial_particle_fields['q3d_entrainment_frequency'] = q3d_entrainment_frequency
        if M_b is not None:
            initial_particle_fields['q3d_wave_breaking_factor'] = M_b
        self._update_particle_fields(initial_particle_fields)

        # Required fields after interpolation/broadcasting. Hydrodynamic flow is
        # required for direction; centroid velocity is diagnostics only.
        required_fields = (
            'bed_level',
            'max_shear_velocity',
            'selected_shear_velocity',
            'profile_roughness_height',
            'total_transport_centroid_elevation',
            'q3d_velocity_deficit_coefficient',
            'q3d_vertical_velocity_gradient',
            'turbulent_shields_number',
            'critical_shields_number',
            'settling_velocity',
            'water_depth',
            'skin_roughness_height',
            'q3d_entrainment_height_above_bed',
            'depth_avg_flow_velocity_u',
            'depth_avg_flow_velocity_v',
            'depth_avg_flow_velocity_magnitude',
        )
        if vertical_update_scheme == 'rouse_profile':
            required_fields = required_fields + ('rouse_number',)
        missing = [name for name in required_fields if name not in self.particles]
        if missing:
            raise KeyError(f'Missing fields for Q3D particle position update: {missing}')

        required_status_fields = (
            'status_domain',
            'status_left_domain',
            'status_alive',
            'status_released',
            'status_suspended',
            'status_deposited',
            'status_buried',
        )
        missing_status = [name for name in required_status_fields if name not in self.particles]
        if missing_status:
            raise KeyError(f'Missing status fields for Q3D particle position update: {missing_status}')

        # ------------------------------------------------------------------
        # 2. Read current particle state. z may still be NaN for particles
        #    that have not reached their release time yet.
        # ------------------------------------------------------------------
        n_particles = len(self.particles['x'])
        # Local bed elevation at each particle position at the start of this timestep.
        bed_level = np.asarray(self.particles['bed_level'], dtype=float)
        water_depth = np.maximum(np.nan_to_num(self.particles['water_depth'], nan=0.0), 0.0)
        skin_roughness = np.maximum(np.nan_to_num(self.particles['skin_roughness_height'], nan=0.0), 0.0)
        # z is the live absolute particle elevation. It starts as NaN because
        # the configured vertical position can only be resolved after bed_level,
        # water_depth, and possibly the Q3D centroid/entrainment height are known.
        if 'z' not in self.particles:
            self.particles['z'] = np.full(n_particles, np.nan, dtype=float)

        # Read the existing state arrays. On the first timestep these are mostly
        # defaults; on later timesteps they are the state left by the previous
        # Q3D update. The initializer below only changes particles that have not
        # been vertically initialized yet.
        z = np.asarray(self.particles['z'], dtype=float).copy()
        burial_depth = np.maximum(
            np.nan_to_num(self.particles.get('burial_depth', np.zeros(n_particles)), nan=0.0),
            0.0,
        )
        is_inside = (
            np.asarray(self.particles['status_domain'], dtype=bool)
            & ~np.asarray(self.particles['status_left_domain'], dtype=bool)
        )
        is_alive = np.asarray(self.particles['status_alive'], dtype=bool)
        is_released = np.asarray(self.particles['status_released'], dtype=bool)
        is_suspended = np.asarray(self.particles['status_suspended'], dtype=bool)
        is_deposited = np.asarray(self.particles['status_deposited'], dtype=bool)
        is_buried = np.asarray(self.particles['status_buried'], dtype=bool)

        # ------------------------------------------------------------------
        # 3. Resolve initial vertical position for newly released particles.
        # ------------------------------------------------------------------
        turbulent_shields = np.nan_to_num(self.particles['turbulent_shields_number'], nan=0.0)
        critical_shields = np.nan_to_num(self.particles['critical_shields_number'], nan=np.inf)
        entrainment_height = np.clip(
            np.nan_to_num(self.particles['q3d_entrainment_height_above_bed'], nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            water_depth,
        )

        # This applies seeding.vertical_position exactly once per particle.
        # Already initialized particles keep their current z, burial_depth, and
        # suspension/deposition/burial status.
        z, burial_depth, is_suspended, is_deposited, is_buried = self._initialize_vertical_position(
            bed_level,
            water_depth,
            entrainment_height,
            z,
            burial_depth,
            is_released,
            is_suspended,
            is_deposited,
            is_buried,
        )

        # ------------------------------------------------------------------
        # 4. Decide which bed particles entrain this timestep.
        #    The stochastic mode uses MacDonald's pickup/entrainment frequency
        #    f_e (Eqs. 57 and 61) as a timestep probability (Eq. 68). The
        #    threshold mode uses the turbulent Shields number from Eq. 59.
        # ------------------------------------------------------------------
        eligible = is_inside & is_alive & is_released
        available = eligible & ~is_buried & ~is_suspended
        entrainment_frequency = self.particles.get('q3d_entrainment_frequency')
        entrained_now, entrainment_probability = self._select_q3d_entrainment(
            available,
            turbulent_shields,
            critical_shields,
            dt,
            mode=q3d_entrainment_mode,
            entrainment_frequency=entrainment_frequency,
            probability_law=q3d_entrainment_probability_law,
            rng=rng,
        )
        not_entrained = available & ~entrained_now
        bed_waiting = eligible & ~available & ~is_suspended

        # Entrainment is sampled once for the outer particle timestep, so the
        # probability uses dt here. Q3D motion substeps below use dt_sub only for
        # advection, diffusion, and vertical position updates.

        # Bed-state particles that are not available for entrainment remain
        # deposited. Their burial depth is left unchanged because burial is
        # updated elsewhere.
        is_deposited[bed_waiting] = True
        is_suspended[bed_waiting] = False

        # Entrained particles leave the bed/burial layer and enter the water column.
        z[entrained_now] = bed_level[entrained_now] + entrainment_height[entrained_now]
        burial_depth[entrained_now] = 0.0
        is_buried[entrained_now] = False
        is_deposited[entrained_now] = False
        is_suspended[entrained_now] = True

        # Available particles that are not entrained stay at/on the bed.
        is_suspended[not_entrained] = False
        is_deposited[not_entrained] = True
        is_buried[not_entrained] = burial_depth[not_entrained] > 0.0
        z[not_entrained] = np.where(
            is_buried[not_entrained],
            bed_level[not_entrained] - burial_depth[not_entrained],
            bed_level[not_entrained],
        )

        # ------------------------------------------------------------------
        # 5. Move suspended particles in q3d_motion_substeps smaller updates.
        #    Substepping does not change the outer timestep physics; it only
        #    evaluates the Q3D random walk and vertical update on smaller dt_sub
        #    intervals, 
        #    Substepping uses dt_sub in the random-walk velocity formulas from
        #    MacDonald Eqs. 51-52. This keeps each diffusive displacement proportional
        #    to sqrt(E * dt_sub), so the accumulated random walk has the correct
        #    timestep scaling over the full particle timestep.
        # ------------------------------------------------------------------
        random_source = np.random if rng is None else rng
        particle_u = np.zeros(n_particles, dtype=float)
        particle_v = np.zeros(n_particles, dtype=float)
        vertical_advection = np.zeros(n_particles, dtype=float)
        particle_w = np.zeros(n_particles, dtype=float)
        deposited_now = np.zeros(n_particles, dtype=bool)
        height_above_bed = np.maximum(np.nan_to_num(z - bed_level, nan=0.0), 0.0)
        first_substep_diagnostics_saved = False
        first_substep_z_p = height_above_bed.copy()
        first_substep_modified_u = np.zeros(n_particles, dtype=float)
        first_substep_modified_v = np.zeros(n_particles, dtype=float)
        first_substep_particle_u = np.zeros(n_particles, dtype=float)
        first_substep_particle_v = np.zeros(n_particles, dtype=float)
        first_substep_particle_w = np.zeros(n_particles, dtype=float)
        if save_first_substep_diagnostics:
            first_substep_horizontal_diffusion_velocity_x = np.zeros(n_particles, dtype=float)
            first_substep_horizontal_diffusion_velocity_y = np.zeros(n_particles, dtype=float)
            first_substep_vertical_diffusion_velocity = np.zeros(n_particles, dtype=float)
            first_substep_vertical_advection = np.zeros(n_particles, dtype=float)
            first_substep_horizontal_diffusion = np.zeros(n_particles, dtype=float)
            first_substep_vertical_diffusion = np.zeros(n_particles, dtype=float)
            first_substep_bed_level = bed_level.copy()
            first_substep_water_depth = water_depth.copy()
            first_substep_skin_roughness = skin_roughness.copy()
            first_substep_shear_velocity = np.zeros(n_particles, dtype=float)
            first_substep_profile_roughness = np.zeros(n_particles, dtype=float)
            first_substep_z_c = np.zeros(n_particles, dtype=float)
            first_substep_deficit = np.zeros(n_particles, dtype=float)
            first_substep_vertical_velocity_gradient = np.zeros(n_particles, dtype=float)
            first_substep_settling_velocity = np.zeros(n_particles, dtype=float)
            first_substep_flow_magnitude = np.zeros(n_particles, dtype=float)
            first_substep_rouse_number = np.zeros(n_particles, dtype=float)
        vertical_scheme_codes = {'geometric': 0, 'centroid_floor': 1, 'rouse_profile': 2}

        for substep_index in range(substeps):
            # Only particles that are currently suspended and not deposited are active
            # for this substep. Particles that deposit in one substep stop moving in
            # the next substeps of the same outer timestep.
            active = eligible & is_suspended & ~is_deposited
            active_indices = np.flatnonzero(active)
            active_count = active_indices.size
            if active_count == 0:
                break

            bed_level_current = np.asarray(self.particles['bed_level'], dtype=float)
            water_depth_current = np.asarray(self.particles['water_depth'], dtype=float)
            skin_roughness_current = np.asarray(self.particles['skin_roughness_height'], dtype=float)

            # z_p is the key Q3D state: current particle height above the local bed.
            # This is particle-resolved z_p, not the grid diagnostic value that
            # assumes z_p = z_c. MacDonald's velocity, diffusion, and vertical
            # advection terms are evaluated at this live particle height.
            bed_active = bed_level_current[active_indices]
            waterdepth_active = np.maximum(np.nan_to_num(water_depth_current[active_indices], nan=0.0), 0.0)
            skin_active = np.maximum(np.nan_to_num(skin_roughness_current[active_indices], nan=0.0), 0.0)
            z_p_active = np.clip(np.nan_to_num(z[active_indices] - bed_active, nan=0.0), 0.0, waterdepth_active)
            max_shear_velocity_active = np.nan_to_num(
                np.asarray(self.particles['max_shear_velocity'], dtype=float)[active_indices],
                nan=0.0,
            )
            selected_shear_velocity_active = np.nan_to_num(
                np.asarray(self.particles.get('selected_shear_velocity'), dtype=float)[active_indices],
                nan=0.0,
            )
            profile_roughness_active = np.maximum(
                np.nan_to_num(
                    np.asarray(self.particles['profile_roughness_height'], dtype=float)[active_indices],
                    nan=0.0,
                ),
                1e-12,
            )
            z_c_active = np.clip(
                np.nan_to_num(
                    np.asarray(self.particles['total_transport_centroid_elevation'], dtype=float)[active_indices],
                    nan=0.0,
                ),
                1e-12,
                np.maximum(waterdepth_active, 1e-12),
            )
            deficit_active = np.clip(
                np.nan_to_num(
                    np.asarray(self.particles['q3d_velocity_deficit_coefficient'], dtype=float)[active_indices],
                    nan=1.0,
                ),
                0.0,
                1.0,
            )

            # Compute reduced horizontal speed from the log-law velocity at
            # z_p and the reference height 1.4 z_c. The velocity deficit follows
            # MacDonald Eqs. 39-40 after the fall-time logic in Eqs. 36-38.
            u_zp_active = self._loglaw_velocity_at_z(selected_shear_velocity_active, z_p_active, profile_roughness_active)
            u_1p4zc_active = self._loglaw_velocity_at_z(
                selected_shear_velocity_active,
                1.4 * z_c_active,
                profile_roughness_active,
            )
            modified_particle_velocity_active = self._apply_q3d_velocity_deficit(
                u_zp_active,
                u_1p4zc_active,
                z_p_active,
                z_c_active,
                deficit_active,
            )

            # Convert scalar particle speed to x/y components using the local
            # depth-averaged hydrodynamic flow direction.
            da_velocity_u_active = np.nan_to_num(
                np.asarray(self.particles['depth_avg_flow_velocity_u'], dtype=float)[active_indices],
                nan=0.0,
            )
            da_velocity_v_active = np.nan_to_num(
                np.asarray(self.particles['depth_avg_flow_velocity_v'], dtype=float)[active_indices],
                nan=0.0,
            )
            da_velocity_magnitude_active = np.maximum(
                np.nan_to_num(np.asarray(self.particles['depth_avg_flow_velocity_magnitude'], dtype=float)[active_indices], nan=0.0),
                1e-12,
            )
            modified_u_active = modified_particle_velocity_active * _safe_divide(da_velocity_u_active, da_velocity_magnitude_active)
            modified_v_active = modified_particle_velocity_active * _safe_divide(da_velocity_v_active, da_velocity_magnitude_active)

            # Compute turbulent diffusivities and draw random-walk velocities for
            # this substep. E_t,h and E_t,v follow MacDonald Eqs. 45 and 49;
            # the random-walk velocities follow Eqs. 51-52. Using dt_sub keeps
            # random displacement velocity * dt_sub = O(sqrt(E * dt_sub)).
            M_b_active = None
            if M_b is not None and 'q3d_wave_breaking_factor' in self.particles:
                M_b_active = np.nan_to_num(
                    np.asarray(self.particles['q3d_wave_breaking_factor'], dtype=float)[active_indices],
                    nan=1.0,
                )
            use_horizontal_diffusion = bool(q3d_horizontal_diffusion_enabled)
            use_vertical_diffusion = vertical_update_scheme == 'geometric'
            horizontal_diffusion_active, vertical_diffusion_active = self._turbulent_diffusion_coefficients(
                waterdepth_active,
                z_p_active,
                da_velocity_magnitude_active,
                max_shear_velocity_active,
                K_Et=K_Et,  # scalar
                K_Ev=K_Ev,  # scalar
                M_b=M_b_active,
                E_turb_hor_min=E_turb_hor_min,  # scalar
                E_turb_vert_min=E_turb_vert_min,  # scalar
                compute_horizontal=use_horizontal_diffusion,
                compute_vertical=use_vertical_diffusion,
            )
            # Only draw random values for enabled components. Besides avoiding
            # unnecessary work, this keeps disabled components from advancing
            # the random-number stream.
            random_horizontal_x_active = np.zeros(active_count, dtype=float)
            random_horizontal_y_active = np.zeros(active_count, dtype=float)
            if use_horizontal_diffusion:
                random_horizontal_x_active = (
                    2.0
                    * (random_source.random(active_count) - 0.5)
                    * np.sqrt(6.0 * horizontal_diffusion_active / dt_sub)
                )
                random_horizontal_y_active = (
                    2.0
                    * (random_source.random(active_count) - 0.5)
                    * np.sqrt(6.0 * horizontal_diffusion_active / dt_sub)
                )

            random_vertical_active = np.zeros(active_count, dtype=float)
            if use_vertical_diffusion:
                random_vertical_active = (
                    2.0
                    * (random_source.random(active_count) - 0.5)
                    * np.sqrt(6.0 * vertical_diffusion_active / dt_sub)
                )

            particle_u_active = np.nan_to_num(
                modified_u_active + random_horizontal_x_active,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            particle_v_active = np.nan_to_num(
                modified_v_active + random_horizontal_y_active,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            particle_u.fill(0.0)
            particle_v.fill(0.0)
            particle_u[active_indices] = particle_u_active
            particle_v[active_indices] = particle_v_active

            # Vertical velocity combines continuity advection (MacDonald
            # Eqs. 41-42), particle settling, and the vertical random walk from
            # Eq. 52, all evaluated at the old position for this substep.
            vertical_gradient_active = np.nan_to_num(
                np.asarray(self.particles['q3d_vertical_velocity_gradient'], dtype=float)[active_indices],
                nan=0.0,
            )
            vertical_advection_active = vertical_gradient_active * (waterdepth_active - z_p_active)
            settling_active = np.nan_to_num(
                np.asarray(self.particles['settling_velocity'], dtype=float)[active_indices],
                nan=0.0,
            )
            particle_w_active = np.nan_to_num(
                vertical_advection_active - settling_active + random_vertical_active,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            particle_w.fill(0.0)
            vertical_advection.fill(0.0)
            particle_w[active_indices] = particle_w_active
            vertical_advection[active_indices] = vertical_advection_active

            # Snapshot first-substep diagnostics. Final x/y/z/status still
            # describe the end of the outer timestep.
            if not first_substep_diagnostics_saved:
                first_substep_z_p[active_indices] = z_p_active
                first_substep_modified_u[active_indices] = modified_u_active
                first_substep_modified_v[active_indices] = modified_v_active
                first_substep_particle_u[active_indices] = particle_u_active
                first_substep_particle_v[active_indices] = particle_v_active
                first_substep_particle_w[active_indices] = particle_w_active
                if save_first_substep_diagnostics:
                    first_substep_horizontal_diffusion_velocity_x[active_indices] = random_horizontal_x_active
                    first_substep_horizontal_diffusion_velocity_y[active_indices] = random_horizontal_y_active
                    first_substep_vertical_diffusion_velocity[active_indices] = random_vertical_active
                    first_substep_vertical_advection[active_indices] = vertical_advection_active
                    first_substep_horizontal_diffusion[active_indices] = horizontal_diffusion_active
                    first_substep_vertical_diffusion[active_indices] = vertical_diffusion_active
                    first_substep_bed_level[active_indices] = bed_active
                    first_substep_water_depth[active_indices] = waterdepth_active
                    first_substep_skin_roughness[active_indices] = skin_active
                    first_substep_shear_velocity[active_indices] = max_shear_velocity_active
                    first_substep_profile_roughness[active_indices] = profile_roughness_active
                    first_substep_z_c[active_indices] = z_c_active
                    first_substep_deficit[active_indices] = deficit_active
                    first_substep_vertical_velocity_gradient[active_indices] = vertical_gradient_active
                    first_substep_settling_velocity[active_indices] = settling_active
                    first_substep_flow_magnitude[active_indices] = da_velocity_magnitude_active
                    if 'rouse_number' in self.particles:
                        first_substep_rouse_number[active_indices] = np.nan_to_num(
                            np.asarray(self.particles['rouse_number'], dtype=float)[active_indices],
                            nan=0.0,
                        )
                first_substep_diagnostics_saved = True

            # Horizontal move first using the Q3D particle velocity from the
            # depth-averaged flow direction plus random walk, then re-interpolate
            # bed/depth at the new x/y location before the vertical update.
            bed_level_current_active = bed_active.copy()
            z_p_current_active = z_p_active.copy() # relative to the bed, not absolute z, we need to know this for the vertical update.
            z_current_active = z[active_indices].copy() # absolute z, we need to know this for the vertical update.

            #Advect particles horizontally using the modified particle velocities 9including random walk). check if any particles leave the domain or beach, and update their status accordingly.
            left_domain_indices, beached_indices = self._advect_particles_with_velocity(
                active,
                particle_u,
                particle_v,
                dt_sub,
            )
            stopped_indices = np.concatenate((left_domain_indices, beached_indices))
            if stopped_indices.size:
                is_alive[left_domain_indices] = False
                is_inside[left_domain_indices] = False
                eligible[left_domain_indices] = False
                is_suspended[stopped_indices] = False
                is_deposited[beached_indices] = True
                deposited_now[beached_indices] = True
                z[beached_indices] = bed_level[beached_indices]
                burial_depth[beached_indices] = 0.0

                continuing = ~np.isin(active_indices, stopped_indices)
                active_indices = active_indices[continuing]
                active_count = active_indices.size
                if active_count == 0:
                    continue
                # If some particles stopped, we need to filter the active arrays to only include the continuing particles for the vertical update.
                bed_level_current_active = bed_level_current_active[continuing]
                z_p_current_active = z_p_current_active[continuing]
                z_current_active = z_current_active[continuing]
                particle_w_active = particle_w_active[continuing]
                settling_active = settling_active[continuing]

            # After the particle has moved to a new x,y horizontally, it might now be over a newer bed level, water depth and skin roughness. 
            post_advection_fields = {
                'bed_level': bed_level_field, # this is needed to convert beween absolute z and relative z_p for the vertical update.
                'water_depth': water_depth_field, # this is needed to clip the new z_p to the water depth after the vertical update.
                'skin_roughness_height': skin_roughness_height_field, # this is needed to compute the deposition threshold for the vertical update.
            }

            #If there is another substep we will need to update the following fields for the next substep, 
            # so we store them in the particle fields. If this is the last substep, we don't need to store them because they won't be used again.
            needs_next_substep_fields = substep_index < substeps - 1
            if vertical_update_scheme in {'centroid_floor', 'rouse_profile'} or needs_next_substep_fields:
                post_advection_fields['total_transport_centroid_elevation'] = total_transport_centroid_elevation
            if vertical_update_scheme == 'rouse_profile':
                post_advection_fields['rouse_number'] = rouse_number
            if needs_next_substep_fields:
                self._update_particle_flow_field('depth_avg_flow_velocity', hydrodynamic_flow_field, indices=active_indices)
                post_advection_fields.update(
                    {
                        'max_shear_velocity': max_shear_velocity,
                        'profile_roughness_height': profile_roughness_height,
                        'q3d_velocity_deficit_coefficient': q3d_velocity_deficit_coefficient,
                        'q3d_vertical_velocity_gradient': q3d_vertical_velocity_gradient,
                    }
                )
                if M_b is not None:
                    post_advection_fields['q3d_wave_breaking_factor'] = M_b
            # We use the updaed post advection fields to update the particle fields for the next substep or for the vertical advection
            self._update_particle_fields(post_advection_fields, indices=active_indices)
            
            # Next we read the new local bed level, water depth, and skin roughness at the new x,y position for the vertical update.
            bed_level_new_active = np.asarray(self.particles['bed_level'], dtype=float)[active_indices]
            water_depth_new_active = np.maximum(
                np.nan_to_num(np.asarray(self.particles['water_depth'], dtype=float)[active_indices], nan=0.0),
                0.0,
            )
            skin_roughness_new_active = np.maximum(
                np.nan_to_num(
                    np.asarray(self.particles['skin_roughness_height'], dtype=float)[active_indices],
                    nan=0.0,
                ),
                0.0,
            )
            deposition_threshold_active = 0.25 * skin_roughness_new_active

            # Prepare verical scheme inputs.
            if vertical_update_scheme in {'centroid_floor', 'rouse_profile'}:
                z_c_new_active = np.nan_to_num(
                    np.asarray(self.particles['total_transport_centroid_elevation'], dtype=float)[active_indices],
                    nan=0.0,
                )
            else: # z_c_new_acive is only useful for the sceme that uses i (centroid floor and rouse profile). 
                  # For the geometric scheme, we don't need it, so we just set it to zero, so that the function call has consistent arguments
                z_c_new_active = np.zeros(active_count, dtype=float)

            if vertical_update_scheme == 'rouse_profile':
                rouse_number_new_active = np.nan_to_num(
                    np.asarray(self.particles['rouse_number'], dtype=float)[active_indices],
                    nan=0.0,
                )
            else: # rouse_number_new_active is only useful for the sceme that uses it. 
                  # For the other schemes, we don't need it, so we just set it to zero, so that the function call has consistent arguments
                rouse_number_new_active = np.zeros(active_count, dtype=float)

            z_p_after_vertical_advection = self._q3d_height_after_vertical_update(
                vertical_update_scheme,
                z_p_old=z_p_current_active,
                bed_level_old=bed_level_current_active,
                bed_level_new=bed_level_new_active,
                water_depth_new=water_depth_new_active,
                particle_w=particle_w_active,
                settling_velocity=settling_active,
                transport_centroid_elevation_new=z_c_new_active,
                rouse_number_new=rouse_number_new_active,
                dt=dt_sub,
                rng=random_source,
            )
            z[active_indices] = bed_level_new_active + z_p_after_vertical_advection

            # Deposit particles that return to the near-bed roughness threshold.
            # MacDonald treats near-bed particles as available or buried within
            # the active layer; this threshold is the local numerical criterion
            # for returning a particle from suspension to the bed.
            deposited_active = np.isfinite(z_p_after_vertical_advection) & (z_p_after_vertical_advection <= deposition_threshold_active)
            deposited_indices = active_indices[deposited_active]

            if deposited_indices.size:
                z[deposited_indices] = bed_level_new_active[deposited_active]
                z_p_after_vertical_advection[deposited_active] = 0.0
                is_deposited[deposited_indices] = True
                is_suspended[deposited_indices] = False
                deposited_now[deposited_indices] = True

            z_p_after_vertical_advection = np.maximum(z_p_after_vertical_advection, 0.0)
            burial_depth[active_indices] = 0.0
            is_buried[active_indices] = False
            is_suspended[active_indices] = z_p_after_vertical_advection > deposition_threshold_active
            is_suspended[deposited_indices] = False
            is_deposited[active_indices] = ~is_suspended[active_indices]
            # TO implement - somewhere here we should also eventually update the burial depth of particles that are deposited but not buried, and set their is_buried flag to False. This will allow them to be entrained again in the future without needing a separate burial update step to reset their state.

        # Enforce the Q3D vertical-state invariant before storing statuses:
        # suspended and deposited are mutually exclusive, and buried particles
        # are deposited bed-state particles that cannot be suspended.
        is_suspended = np.asarray(is_suspended, dtype=bool)
        is_deposited = np.asarray(is_deposited, dtype=bool)
        is_buried = np.asarray(is_buried, dtype=bool)
        is_suspended[is_buried] = False
        is_deposited[is_buried] = True
        is_deposited[is_suspended] = False

        bed_level = np.asarray(self.particles['bed_level'], dtype=float)
        height_above_bed = np.maximum(np.nan_to_num(z - bed_level, nan=0.0), 0.0)

        # ------------------------------------------------------------------
        # 6. Store updated particle state and diagnostics.
        # ------------------------------------------------------------------
        centroid_u = np.nan_to_num(
            self.particles.get('centroid_particle_velocity_u', np.zeros(n_particles, dtype=float)),
            nan=0.0,
        )
        centroid_v = np.nan_to_num(
            self.particles.get('centroid_particle_velocity_v', np.zeros(n_particles, dtype=float)),
            nan=0.0,
        )
        centroid_magnitude = np.nan_to_num(
            self.particles.get('centroid_particle_velocity_magnitude', np.zeros(n_particles, dtype=float)),
            nan=0.0,
        )
        self.particles['centroid_particle_velocity_x'] = centroid_u
        self.particles['centroid_particle_velocity_y'] = centroid_v
        self.particles['centroid_particle_velocity'] = centroid_magnitude
        self.particles['z'] = z
        self.particles['z_p'] = height_above_bed
        self.particles['q3d_first_substep_z_p'] = first_substep_z_p
        self.particles['first_substep_modified_centroid_particle_velocity_x'] = first_substep_modified_u
        self.particles['first_substep_modified_centroid_particle_velocity_y'] = first_substep_modified_v
        self.particles['first_substep_modified_centroid_particle_velocity'] = np.hypot(
            first_substep_modified_u,
            first_substep_modified_v,
        )
        self.particles['first_substep_horizontal_particle_velocity_x'] = first_substep_particle_u
        self.particles['first_substep_horizontal_particle_velocity_y'] = first_substep_particle_v
        self.particles['first_substep_horizontal_particle_velocity'] = np.hypot(first_substep_particle_u, first_substep_particle_v)
        self.particles['first_substep_vertical_particle_velocity'] = first_substep_particle_w
        if save_first_substep_diagnostics:
            self.particles['first_substep_horizontal_diffusion_velocity_x'] = first_substep_horizontal_diffusion_velocity_x
            self.particles['first_substep_horizontal_diffusion_velocity_y'] = first_substep_horizontal_diffusion_velocity_y
            self.particles['first_substep_horizontal_diffusion_velocity'] = np.hypot(
                first_substep_horizontal_diffusion_velocity_x,
                first_substep_horizontal_diffusion_velocity_y,
            )
            self.particles['first_substep_vertical_advection_velocity'] = first_substep_vertical_advection
            self.particles['first_substep_vertical_diffusion_velocity'] = first_substep_vertical_diffusion_velocity
            self.particles['first_substep_vertical_diffusion_coefficient'] = first_substep_vertical_diffusion
            self.particles['first_substep_horizontal_diffusion_coefficient'] = first_substep_horizontal_diffusion
            self.particles['first_substep_bed_level'] = first_substep_bed_level
            self.particles['first_substep_water_depth'] = first_substep_water_depth
            self.particles['first_substep_skin_roughness_height'] = first_substep_skin_roughness
            self.particles['first_substep_max_shear_velocity'] = first_substep_shear_velocity
            self.particles['first_substep_profile_roughness_height'] = first_substep_profile_roughness
            self.particles['first_substep_total_transport_centroid_elevation'] = first_substep_z_c
            self.particles['first_substep_q3d_velocity_deficit_coefficient'] = first_substep_deficit
            self.particles['first_substep_q3d_vertical_velocity_gradient'] = first_substep_vertical_velocity_gradient
            self.particles['first_substep_settling_velocity'] = first_substep_settling_velocity
            self.particles['first_substep_depth_avg_flow_velocity_magnitude'] = first_substep_flow_magnitude
            self.particles['first_substep_rouse_number'] = first_substep_rouse_number
        else:
            for field_name in (
                'first_substep_horizontal_diffusion_velocity_x',
                'first_substep_horizontal_diffusion_velocity_y',
                'first_substep_horizontal_diffusion_velocity',
                'first_substep_vertical_advection_velocity',
                'first_substep_vertical_diffusion_velocity',
                'first_substep_vertical_diffusion_coefficient',
                'first_substep_horizontal_diffusion_coefficient',
                'first_substep_bed_level',
                'first_substep_water_depth',
                'first_substep_skin_roughness_height',
                'first_substep_max_shear_velocity',
                'first_substep_profile_roughness_height',
                'first_substep_total_transport_centroid_elevation',
                'first_substep_q3d_velocity_deficit_coefficient',
                'first_substep_q3d_vertical_velocity_gradient',
                'first_substep_settling_velocity',
                'first_substep_depth_avg_flow_velocity_magnitude',
                'first_substep_rouse_number',
            ):
                self.particles.pop(field_name, None)
        self.particles['q3d_vertical_update_scheme_code'] = np.full(
            n_particles,
            vertical_scheme_codes[vertical_update_scheme],
            dtype=int,
        )
        self.particles['q3d_motion_substeps'] = np.full(n_particles, substeps, dtype=int)
        self.particles['status_suspended'] = is_suspended
        self.particles['status_deposited'] = is_deposited
        self.particles['status_buried'] = is_buried
        self.particles['status_available_for_entrainment'] = available
        self.particles['q3d_entrainment_probability'] = entrainment_probability
        self.particles['status_entrained_now'] = entrained_now
        self.particles['status_deposited_now'] = deposited_now
        self.particles['burial_depth'] = burial_depth
        self.particles['z_burial'] = bed_level - burial_depth

        is_mobile = eligible & is_suspended & ~is_deposited
        self.particles['status_mobile'] = is_mobile

    def update_burial_depth(self) -> None:
        """Update burial depth for temporal bed-level accretion or erosion.

        ``bed_level_previous`` is the bed level at the particle's current
        position during the previous timestep. The difference from the current
        bed level is therefore the local temporal bed change.
        """
        if len(self.particles['x']) == 0:
            return

        bed_level_change = self.particles['bed_level'] - self.particles['bed_level_previous']
        self.particles['burial_depth'] += bed_level_change
        self.particles['burial_depth'] = np.maximum(self.particles['burial_depth'], 0.0)
        self.particles['z'] = self.particles['bed_level'] - self.particles['burial_depth']

    def update_bed_level_change_after_movement(self, bed_level) -> None:
        """Resample bed level after movement and update absolute particle z.

        This makes the bed level at the particle's new position the reference
        used by ``update_burial_depth`` during the next timestep.
        """
        if len(self.particles['x']) == 0:
            return

        self._update_particle_field('bed_level', bed_level)
        self.particles['z'] = self.particles['bed_level'] - self.particles['burial_depth']
        
    def update_status(self) -> None:
        """
        updates status of particles in the population.
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

        transport_probability_method = self.population_config.population_config.get('transport_probability', 'no_probability')
        if transport_probability_method == 'no_probability':
            self.particles['status_transported'] = np.ones(n_particles, dtype=bool)
        else:
            self.particles['status_transported'] = np.random.rand(n_particles) < self.particles[
                'transport_probability'
            ]

        if not self._particle_simplices_match_positions():
            self._refresh_particle_simplices()
        self._particle_simplices[left_domain] = -1
        self._mark_particle_simplices_current()
        self.particles['status_domain'] = (self._particle_simplices >= 0) & ~left_domain

        # New conditional logic based on transport_probability_method
        if transport_probability_method == 'no_probability':
            # For no_probability method, all particles are considered exposed (not buried)
            self.particles['status_buried'] = np.zeros(n_particles, dtype=bool)
        else:
            # For stochastic_transport and reduced_velocity methods, use burial_depth vs mixing_depth

            # if van westen method:
            # a particle is considered buried if it is deeper than or equal to the mixing depth
            self.particles['status_buried'] = self.particles['burial_depth'] >= self.particles['mixing_depth']

            # if soulsby method:
            # self.particles['status_buried'] = (this is where we implement Soulsby's F based on a and b)

        # Compute whether particles are released (or retained)
        self.particles['status_released'] = self._current_time >= self.particles['release_time']

        # Compute whether particles are alive (or dead) (still TODO)
        self.particles['status_alive'] = ~left_domain

        # Compute whether particles are mobile (or static) - combination of all status flags
        self.particles['status_mobile'] = (
            self.particles['status_domain']
            & self.particles['status_alive']
            & ~self.particles['status_buried']
            & self.particles['status_released']
            & self.particles['status_transported']
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

        if len(self.particles['x']) == 0:
            return

        ix = np.asarray(self.particles['status_mobile'], dtype=bool).copy()  # Freeze current mobile-particle mask.
        particle_indices = np.flatnonzero(ix)
        if particle_indices.size == 0:
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


