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
    burial_depth: float = field(init=False, default=0.0)  # burial depth of the particles
    strategy_settings: Dict = field(init=False, default_factory=dict)
    remove_permanently_buried: bool = field(init=False, default=False)

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
        _burial_depth = find_value(self.population_config, 'seeding.burial_depth', {})
        if not _burial_depth:
            raise MissingConfigurationParameter('"burial_depth" is not defined in the population configuration.')
        self.burial_depth = _burial_depth.get('constant', 0.0)  # TODO: support other types of burial depth

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

        particle_values = self._field_interpolator(field_array, self.particles['x'], self.particles['y'])
        if np.isnan(particle_values).all():
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

        # Horizontal diffusion coefficient, equation 46 and 48 in MacDonald et al. (2006)
        horizontal = M_b * K_Et * water_depth * shear_velocity
        horizontal = np.maximum(np.nan_to_num(horizontal, nan=0.0), E_turb_hor_min)

        # Vertical diffusion coefficient, equation 49 and 50 in MacDonald et al. (2006)
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
            is_deposited[idx] = ~is_suspended[idx]
            burial_depth[idx] = np.where(is_buried[idx], -height_above_bed, 0.0)
            initialized[idx] = True

        self.particles['vertical_position_initialized'] = initialized
        return z, burial_depth, is_suspended, is_deposited, is_buried

    def _advect_particles_with_velocity(self, active, velocity_x, velocity_y, dt):
        """Move active particles with particle-level horizontal velocities."""
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            return

        self.particles['x'][active] += velocity_x[active] * dt
        self.particles['y'][active] += velocity_y[active] * dt
        self._particle_simplices[active_indices] = self.grid_geometry.locate_points(
            self.particles['x'][active],
            self.particles['y'][active],
            start_simplices=self._particle_simplices[active_indices],
        )

    @staticmethod
    def _select_q3d_entrainment(
        available,
        turbulent_shields,
        critical_shields,
        dt,
        *,
        mode='shields_threshold',
        entrainment_frequency=None,
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
            - entrainment_frequency: stochastic entrainment with probability
              1 - exp(-entrainment_frequency * dt).
        entrainment_frequency : array or None
            Entrainment frequency [1/s] for entrainment_frequency mode,
            usually from the MacDonald q3d_entrainment_frequency field.
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
            probability = np.clip(-np.expm1(-frequency * dt), 0.0, 1.0)
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
        z_old,
        bed_level_old,
        bed_level_new,
        water_depth_new,
        vertical_velocity,
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
            height = z_p_old + vertical_velocity * dt + (bed_level_old - bed_level_new)
        elif scheme == 'centroid_floor':
            z_c_new = np.clip(np.nan_to_num(transport_centroid_elevation_new, nan=0.0), 0.0, water_depth_new)
            candidate_z = z_old - np.maximum(np.nan_to_num(settling_velocity, nan=0.0), 0.0) * dt
            centroid_floor_z = bed_level_new + z_c_new
            height = np.maximum(candidate_z, centroid_floor_z) - bed_level_new
        elif scheme == 'rouse_profile':
            height = self._sample_rouse_profile_height(water_depth_new, rouse_number_new, rng=rng)
        else:
            raise ValueError(f'Unsupported q3d vertical update scheme {scheme!r}')

        return np.clip(np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0), 0.0, water_depth_new)

    def update_q3d_particle_motion(
        self,
        current_timestep: float,
        centroid_flow_field: Dict,
        hydrodynamic_flow_field: Dict,
        bed_level_field: Any,
        max_shear_velocity: Any,
        total_roughness_height: Any,
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
        q3d_entrainment_mode: str = 'shields_threshold',
        q3d_entrainment_frequency: Any = None,
        q3d_vertical_update_scheme: str = 'geometric',
        q3d_motion_substeps: int = 1,
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
        total_roughness_height : array or temporal field
            Total roughness height k_s [m].
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
        q3d_entrainment_mode : str
            Rule used to decide which available bed particles enter suspension.
            Supported values are shields_threshold, non_zero_particle_velocity,
            and entrainment_frequency.
        q3d_entrainment_frequency : float or array, optional
            MacDonald q3d_entrainment_frequency field [1/s] used when
            q3d_entrainment_mode is entrainment_frequency.
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
        q3d_diagnostics : str
            Q3D particle diagnostics level. minimal stores the essential Q3D
            particle velocities/statuses and exported centroid velocity; full
            also stores intermediate diffusion and hydraulic diagnostic arrays.
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
        diagnostics_level = self._normalize_q3d_diagnostics_level(q3d_diagnostics)
        save_full_diagnostics = diagnostics_level == 'full'
        if vertical_update_scheme == 'rouse_profile' and rouse_number is None:
            raise ValueError('rouse_number is required when q3d_vertical_update_scheme is rouse_profile.')

        if hydrodynamic_flow_field is None:
            raise ValueError('hydrodynamic_flow_field is required for Q3D particle motion.')

        water_depth_field = water_depth
        skin_roughness_height_field = skin_roughness_height

        # Store centroid velocity at particle positions for diagnostics.
        self._update_particle_flow_field('centroid_particle_velocity', centroid_flow_field)

        # Use hydrodynamic flow for movement direction and depth-averaged |U|.
        self._update_particle_flow_field('q3d_flow', hydrodynamic_flow_field)
        initial_particle_fields = {
            'bed_level': bed_level_field,
            'max_shear_velocity': max_shear_velocity,
            'total_roughness_height': total_roughness_height,
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
            'total_roughness_height',
            'total_transport_centroid_elevation',
            'q3d_velocity_deficit_coefficient',
            'q3d_vertical_velocity_gradient',
            'turbulent_shields_number',
            'critical_shields_number',
            'settling_velocity',
            'water_depth',
            'skin_roughness_height',
            'q3d_entrainment_height_above_bed',
            'q3d_flow_u',
            'q3d_flow_v',
            'q3d_flow_magnitude',
        )
        if vertical_update_scheme == 'rouse_profile':
            required_fields = required_fields + ('rouse_number',)
        missing = [name for name in required_fields if name not in self.particles]
        if missing:
            raise KeyError(f'Missing fields for Q3D particle motion update: {missing}')

        # ------------------------------------------------------------------
        # 2. Read current particle state. z may still be NaN for particles
        #    that have not reached their release time yet.
        # ------------------------------------------------------------------
        n_particles = len(self.particles['x'])
        # Local bed elevation at each particle position at the start of this timestep.
        bed_level = np.asarray(self.particles['bed_level'], dtype=float)
        water_depth = np.maximum(np.nan_to_num(self.particles['water_depth'], nan=0.0), 0.0)
        skin_roughness = np.maximum(np.nan_to_num(self.particles['skin_roughness_height'], nan=0.0), 0.0)
        deposition_threshold = 0.25 * skin_roughness

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
        is_inside = np.asarray(self.particles.get('is_inside', np.ones(n_particles, dtype=bool)), dtype=bool)
        is_alive = np.asarray(self.particles.get('is_alive', np.ones(n_particles, dtype=bool)), dtype=bool)
        is_released = np.asarray(self.particles.get('is_released', np.ones(n_particles, dtype=bool)), dtype=bool)
        is_exposed = np.asarray(self.particles.get('is_exposed', np.ones(n_particles, dtype=bool)), dtype=bool)
        is_suspended = np.asarray(self.particles.get('is_suspended', np.zeros(n_particles, dtype=bool)), dtype=bool)
        is_deposited = np.asarray(self.particles.get('is_deposited', np.zeros(n_particles, dtype=bool)), dtype=bool)
        is_buried = np.asarray(
            self.particles.get('is_buried', burial_depth > 0.0),
            dtype=bool,
        )

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
        # ------------------------------------------------------------------
        eligible = is_inside & is_alive & is_released
        available = eligible & is_exposed & ~is_suspended
        entrainment_frequency = self.particles.get('q3d_entrainment_frequency')
        entrained_now, entrainment_probability = self._select_q3d_entrainment(
            available,
            turbulent_shields,
            critical_shields,
            dt,
            mode=q3d_entrainment_mode,
            entrainment_frequency=entrainment_frequency,
            rng=rng,
        )
        not_entrained = available & ~entrained_now

        # Particles that are not currently suspended and are not available stay on/in
        # the bed; their burial depth is left untouched because burial is handled elsewhere.
        bed_waiting = ~available & ~is_suspended
        is_deposited[bed_waiting] = True
        is_suspended[bed_waiting] = False

        # Entrained particles leave the bed/burial layer and enter the water column.
        z[entrained_now] = bed_level[entrained_now] + entrainment_height[entrained_now]
        burial_depth[entrained_now] = 0.0
        is_buried[entrained_now] = False
        is_deposited[entrained_now] = False
        is_suspended[entrained_now] = True

        # Not entrained particles stay where they were.
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
        # ------------------------------------------------------------------
        random_source = np.random if rng is None else rng
        particle_u = np.zeros(n_particles, dtype=float)
        particle_v = np.zeros(n_particles, dtype=float)
        vertical_advection = np.zeros(n_particles, dtype=float)
        vertical_velocity = np.zeros(n_particles, dtype=float)
        deposited_now = np.zeros(n_particles, dtype=bool)
        height_above_bed = np.maximum(np.nan_to_num(z - bed_level, nan=0.0), 0.0)
        first_substep_diagnostics_saved = False
        diagnostic_z_p = height_above_bed.copy()
        diagnostic_modified_u = np.zeros(n_particles, dtype=float)
        diagnostic_modified_v = np.zeros(n_particles, dtype=float)
        diagnostic_particle_u = np.zeros(n_particles, dtype=float)
        diagnostic_particle_v = np.zeros(n_particles, dtype=float)
        diagnostic_vertical_velocity = np.zeros(n_particles, dtype=float)
        if save_full_diagnostics:
            diagnostic_horizontal_diffusion_velocity_x = np.zeros(n_particles, dtype=float)
            diagnostic_horizontal_diffusion_velocity_y = np.zeros(n_particles, dtype=float)
            diagnostic_vertical_diffusion_velocity = np.zeros(n_particles, dtype=float)
            diagnostic_vertical_advection = np.zeros(n_particles, dtype=float)
            diagnostic_horizontal_diffusion = np.zeros(n_particles, dtype=float)
            diagnostic_vertical_diffusion = np.zeros(n_particles, dtype=float)
            diagnostic_bed_level = bed_level.copy()
            diagnostic_water_depth = water_depth.copy()
            diagnostic_skin_roughness = skin_roughness.copy()
            diagnostic_shear_velocity = np.zeros(n_particles, dtype=float)
            diagnostic_total_roughness = np.zeros(n_particles, dtype=float)
            diagnostic_z_c = np.zeros(n_particles, dtype=float)
            diagnostic_deficit = np.zeros(n_particles, dtype=float)
            diagnostic_vertical_velocity_gradient = np.zeros(n_particles, dtype=float)
            diagnostic_settling_velocity = np.zeros(n_particles, dtype=float)
            diagnostic_flow_magnitude = np.zeros(n_particles, dtype=float)
            diagnostic_rouse_number = np.zeros(n_particles, dtype=float)
        vertical_scheme_codes = {'geometric': 0, 'centroid_floor': 1, 'rouse_profile': 2}

        for substep_index in range(substeps):
            # Only particles that are currently suspended and not deposited are active
            # for this substep. Particles that deposit in one substep stop moving in
            # the next substep of the same outer timestep.
            active = eligible & is_suspended & ~is_deposited
            active_indices = np.flatnonzero(active)
            active_count = active_indices.size
            if active_count == 0:
                break

            bed_level_current = np.asarray(self.particles['bed_level'], dtype=float)
            water_depth_current = np.asarray(self.particles['water_depth'], dtype=float)
            skin_roughness_current = np.asarray(self.particles['skin_roughness_height'], dtype=float)

            # z_p is the key Q3D state: current particle height above the local bed.
            bed_active = bed_level_current[active_indices]
            water_active = np.maximum(np.nan_to_num(water_depth_current[active_indices], nan=0.0), 0.0)
            skin_active = np.maximum(np.nan_to_num(skin_roughness_current[active_indices], nan=0.0), 0.0)
            z_p_active = np.clip(np.nan_to_num(z[active_indices] - bed_active, nan=0.0), 0.0, water_active)
            shear_velocity_active = np.nan_to_num(
                np.asarray(self.particles['max_shear_velocity'], dtype=float)[active_indices],
                nan=0.0,
            )
            total_roughness_active = np.maximum(
                np.nan_to_num(np.asarray(self.particles['total_roughness_height'], dtype=float)[active_indices], nan=0.0),
                1e-12,
            )
            z_c_active = np.clip(
                np.nan_to_num(
                    np.asarray(self.particles['total_transport_centroid_elevation'], dtype=float)[active_indices],
                    nan=0.0,
                ),
                1e-12,
                np.maximum(water_active, 1e-12),
            )
            deficit_active = np.clip(
                np.nan_to_num(
                    np.asarray(self.particles['q3d_velocity_deficit_coefficient'], dtype=float)[active_indices],
                    nan=1.0,
                ),
                0.0,
                1.0,
            )

            # Compute reduced horizontal speed from the log-law velocity at z_p.
            u_zp_active = self._loglaw_velocity_at_z(shear_velocity_active, z_p_active, total_roughness_active)
            u_1p4zc_active = self._loglaw_velocity_at_z(
                shear_velocity_active,
                1.4 * z_c_active,
                total_roughness_active,
            )
            horizontal_speed_active = self._apply_q3d_velocity_deficit(
                u_zp_active,
                u_1p4zc_active,
                z_p_active,
                z_c_active,
                deficit_active,
            )

            # Convert scalar particle speed to x/y components using the local
            # depth-averaged hydrodynamic flow direction.
            flow_u_active = np.nan_to_num(
                np.asarray(self.particles['q3d_flow_u'], dtype=float)[active_indices],
                nan=0.0,
            )
            flow_v_active = np.nan_to_num(
                np.asarray(self.particles['q3d_flow_v'], dtype=float)[active_indices],
                nan=0.0,
            )
            flow_magnitude_active = np.maximum(
                np.nan_to_num(np.asarray(self.particles['q3d_flow_magnitude'], dtype=float)[active_indices], nan=0.0),
                1e-12,
            )
            modified_u_active = horizontal_speed_active * _safe_divide(flow_u_active, flow_magnitude_active)
            modified_v_active = horizontal_speed_active * _safe_divide(flow_v_active, flow_magnitude_active)

            # Compute turbulent diffusivities and draw random-walk velocities for
            # this substep. Using dt_sub here preserves the diffusion scaling:
            # random displacement is velocity * dt_sub = O(sqrt(E * dt_sub)).
            M_b_active = None
            if M_b is not None and 'q3d_wave_breaking_factor' in self.particles:
                M_b_active = np.nan_to_num(
                    np.asarray(self.particles['q3d_wave_breaking_factor'], dtype=float)[active_indices],
                    nan=1.0,
                )
            horizontal_diffusion_active, vertical_diffusion_active = self._turbulent_diffusion_coefficients(
                water_active,
                z_p_active,
                flow_magnitude_active,
                shear_velocity_active,
                K_Et=K_Et,
                K_Ev=K_Ev,
                M_b=M_b_active,
                E_turb_hor_min=E_turb_hor_min,
                E_turb_vert_min=E_turb_vert_min,
            )
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

            # Vertical velocity combines continuity advection, settling, and
            # vertical random walk at the old position for this substep.
            vertical_gradient_active = np.nan_to_num(
                np.asarray(self.particles['q3d_vertical_velocity_gradient'], dtype=float)[active_indices],
                nan=0.0,
            )
            vertical_advection_active = vertical_gradient_active * (water_active - z_p_active)
            settling_active = np.nan_to_num(
                np.asarray(self.particles['settling_velocity'], dtype=float)[active_indices],
                nan=0.0,
            )
            vertical_velocity_active = np.nan_to_num(
                vertical_advection_active - settling_active + random_vertical_active,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            vertical_velocity.fill(0.0)
            vertical_advection.fill(0.0)
            vertical_velocity[active_indices] = vertical_velocity_active
            vertical_advection[active_indices] = vertical_advection_active

            # Snapshot first-substep diagnostics. Final x/y/z/status still
            # describe the end of the outer timestep.
            if not first_substep_diagnostics_saved:
                diagnostic_z_p[active_indices] = z_p_active
                diagnostic_modified_u[active_indices] = modified_u_active
                diagnostic_modified_v[active_indices] = modified_v_active
                diagnostic_particle_u[active_indices] = particle_u_active
                diagnostic_particle_v[active_indices] = particle_v_active
                diagnostic_vertical_velocity[active_indices] = vertical_velocity_active
                if save_full_diagnostics:
                    diagnostic_horizontal_diffusion_velocity_x[active_indices] = random_horizontal_x_active
                    diagnostic_horizontal_diffusion_velocity_y[active_indices] = random_horizontal_y_active
                    diagnostic_vertical_diffusion_velocity[active_indices] = random_vertical_active
                    diagnostic_vertical_advection[active_indices] = vertical_advection_active
                    diagnostic_horizontal_diffusion[active_indices] = horizontal_diffusion_active
                    diagnostic_vertical_diffusion[active_indices] = vertical_diffusion_active
                    diagnostic_bed_level[active_indices] = bed_active
                    diagnostic_water_depth[active_indices] = water_active
                    diagnostic_skin_roughness[active_indices] = skin_active
                    diagnostic_shear_velocity[active_indices] = shear_velocity_active
                    diagnostic_total_roughness[active_indices] = total_roughness_active
                    diagnostic_z_c[active_indices] = z_c_active
                    diagnostic_deficit[active_indices] = deficit_active
                    diagnostic_vertical_velocity_gradient[active_indices] = vertical_gradient_active
                    diagnostic_settling_velocity[active_indices] = settling_active
                    diagnostic_flow_magnitude[active_indices] = flow_magnitude_active
                    if 'rouse_number' in self.particles:
                        diagnostic_rouse_number[active_indices] = np.nan_to_num(
                            np.asarray(self.particles['rouse_number'], dtype=float)[active_indices],
                            nan=0.0,
                        )
                first_substep_diagnostics_saved = True

            # Horizontal move first, then re-interpolate bed/depth at the new
            # x/y location before applying the selected vertical update scheme.
            bed_level_old_active = bed_active.copy()
            z_p_old_active = z_p_active.copy()
            z_old_active = z[active_indices].copy()
            self._advect_particles_with_velocity(active, particle_u, particle_v, dt_sub)

            post_advection_fields = {
                'bed_level': bed_level_field,
                'water_depth': water_depth_field,
                'skin_roughness_height': skin_roughness_height_field,
            }
            needs_next_substep_fields = substep_index < substeps - 1
            if vertical_update_scheme in {'centroid_floor', 'rouse_profile'} or needs_next_substep_fields:
                post_advection_fields['total_transport_centroid_elevation'] = total_transport_centroid_elevation
            if vertical_update_scheme == 'rouse_profile':
                post_advection_fields['rouse_number'] = rouse_number
            if needs_next_substep_fields:
                self._update_particle_flow_field('q3d_flow', hydrodynamic_flow_field, indices=active_indices)
                post_advection_fields.update(
                    {
                        'max_shear_velocity': max_shear_velocity,
                        'total_roughness_height': total_roughness_height,
                        'q3d_velocity_deficit_coefficient': q3d_velocity_deficit_coefficient,
                        'q3d_vertical_velocity_gradient': q3d_vertical_velocity_gradient,
                    }
                )
                if M_b is not None:
                    post_advection_fields['q3d_wave_breaking_factor'] = M_b
            self._update_particle_fields(post_advection_fields, indices=active_indices)

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

            if vertical_update_scheme in {'centroid_floor', 'rouse_profile'}:
                z_c_new_active = np.nan_to_num(
                    np.asarray(self.particles['total_transport_centroid_elevation'], dtype=float)[active_indices],
                    nan=0.0,
                )
            else:
                z_c_new_active = np.zeros(active_count, dtype=float)
            if vertical_update_scheme == 'rouse_profile':
                rouse_number_new_active = np.nan_to_num(
                    np.asarray(self.particles['rouse_number'], dtype=float)[active_indices],
                    nan=0.0,
                )
            else:
                rouse_number_new_active = np.zeros(active_count, dtype=float)

            height_after_vertical = self._q3d_height_after_vertical_update(
                vertical_update_scheme,
                z_p_old=z_p_old_active,
                z_old=z_old_active,
                bed_level_old=bed_level_old_active,
                bed_level_new=bed_level_new_active,
                water_depth_new=water_depth_new_active,
                vertical_velocity=vertical_velocity_active,
                settling_velocity=settling_active,
                transport_centroid_elevation_new=z_c_new_active,
                rouse_number_new=rouse_number_new_active,
                dt=dt_sub,
                rng=random_source,
            )
            z[active_indices] = bed_level_new_active + height_after_vertical

            # Deposit particles that return to the near-bed roughness threshold.
            height_active = z[active_indices] - bed_level_new_active
            deposited_active = np.isfinite(height_active) & (height_active <= deposition_threshold_active)
            deposited_indices = active_indices[deposited_active]

            if deposited_indices.size:
                z[deposited_indices] = bed_level_new_active[deposited_active]
                height_active[deposited_active] = 0.0
                is_deposited[deposited_indices] = True
                is_suspended[deposited_indices] = False
                deposited_now[deposited_indices] = True

            height_active = np.maximum(height_active, 0.0)
            burial_depth[active_indices] = 0.0
            is_buried[active_indices] = False
            is_suspended[active_indices] = height_active > deposition_threshold_active
            is_suspended[deposited_indices] = False
            height_above_bed[active_indices] = height_active

            # TO implement - somewhere here we should also eventually update the burial depth of particles that are deposited but not buried, and set their is_buried flag to False. This will allow them to be entrained again in the future without needing a separate burial update step to reset their state.

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
        self.particles['q3d_diagnostic_z_p_first_substep'] = diagnostic_z_p
        self.particles['modified_centroid_particle_velocity_x'] = diagnostic_modified_u
        self.particles['modified_centroid_particle_velocity_y'] = diagnostic_modified_v
        self.particles['modified_centroid_particle_velocity'] = np.hypot(
            diagnostic_modified_u,
            diagnostic_modified_v,
        )
        self.particles['horizontal_particle_velocity_x'] = diagnostic_particle_u
        self.particles['horizontal_particle_velocity_y'] = diagnostic_particle_v
        self.particles['horizontal_particle_velocity'] = np.hypot(diagnostic_particle_u, diagnostic_particle_v)
        self.particles['vertical_particle_velocity'] = diagnostic_vertical_velocity
        if save_full_diagnostics:
            self.particles['horizontal_diffusion_velocity_x'] = diagnostic_horizontal_diffusion_velocity_x
            self.particles['horizontal_diffusion_velocity_y'] = diagnostic_horizontal_diffusion_velocity_y
            self.particles['horizontal_diffusion_velocity'] = np.hypot(
                diagnostic_horizontal_diffusion_velocity_x,
                diagnostic_horizontal_diffusion_velocity_y,
            )
            self.particles['vertical_advection_velocity'] = diagnostic_vertical_advection
            self.particles['vertical_diffusion_velocity'] = diagnostic_vertical_diffusion_velocity
            self.particles['vertical_diffusion_coefficient'] = diagnostic_vertical_diffusion
            self.particles['horizontal_diffusion_coefficient'] = diagnostic_horizontal_diffusion
            self.particles['diagnostic_bed_level'] = diagnostic_bed_level
            self.particles['diagnostic_water_depth'] = diagnostic_water_depth
            self.particles['diagnostic_skin_roughness_height'] = diagnostic_skin_roughness
            self.particles['diagnostic_max_shear_velocity'] = diagnostic_shear_velocity
            self.particles['diagnostic_total_roughness_height'] = diagnostic_total_roughness
            self.particles['diagnostic_total_transport_centroid_elevation'] = diagnostic_z_c
            self.particles['diagnostic_q3d_velocity_deficit_coefficient'] = diagnostic_deficit
            self.particles['diagnostic_q3d_vertical_velocity_gradient'] = diagnostic_vertical_velocity_gradient
            self.particles['diagnostic_settling_velocity'] = diagnostic_settling_velocity
            self.particles['diagnostic_q3d_flow_magnitude'] = diagnostic_flow_magnitude
            self.particles['diagnostic_rouse_number'] = diagnostic_rouse_number
        else:
            for field_name in (
                'horizontal_diffusion_velocity_x',
                'horizontal_diffusion_velocity_y',
                'horizontal_diffusion_velocity',
                'vertical_advection_velocity',
                'vertical_diffusion_velocity',
                'vertical_diffusion_coefficient',
                'horizontal_diffusion_coefficient',
                'diagnostic_bed_level',
                'diagnostic_water_depth',
                'diagnostic_skin_roughness_height',
                'diagnostic_max_shear_velocity',
                'diagnostic_total_roughness_height',
                'diagnostic_total_transport_centroid_elevation',
                'diagnostic_q3d_velocity_deficit_coefficient',
                'diagnostic_q3d_vertical_velocity_gradient',
                'diagnostic_settling_velocity',
                'diagnostic_q3d_flow_magnitude',
                'diagnostic_rouse_number',
            ):
                self.particles.pop(field_name, None)
        self.particles['q3d_vertical_update_scheme_code'] = np.full(
            n_particles,
            vertical_scheme_codes[vertical_update_scheme],
            dtype=int,
        )
        self.particles['q3d_motion_substeps'] = np.full(n_particles, substeps, dtype=int)
        self.particles['is_suspended'] = is_suspended
        self.particles['is_deposited'] = is_deposited
        self.particles['is_buried'] = is_buried
        self.particles['is_available_for_entrainment'] = available
        self.particles['q3d_entrainment_probability'] = entrainment_probability
        self.particles['entrained_now'] = entrained_now
        self.particles['deposited_now'] = deposited_now
        self.particles['burial_depth'] = burial_depth
        self.particles['z_burial'] = bed_level - burial_depth

        self.particles['is_mobile'] = eligible & is_suspended & ~is_deposited

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
