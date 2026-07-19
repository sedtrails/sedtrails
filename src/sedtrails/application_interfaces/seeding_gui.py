"""Small desktop helper for choosing SedTRAILS seed points from input data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import yaml

from sedtrails.application_interfaces.validator import SedtrailsYamlLoader, YAMLConfigValidator

SEEDING_MODES = ('points', 'transect', 'random', 'grid')
RANDOM_CANDIDATE_BATCH_SIZE = 16_384
GRID_CANDIDATE_BATCH_SIZE = 100_000
DEFAULT_GRID_MAX_CANDIDATES = 2_000_000
MAP_ZOOM_IN_FACTOR = 0.5
MAP_ZOOM_OUT_FACTOR = 2.0


class GuiInputFormatSpec(TypedDict):
    coordinate_candidates: tuple[tuple[str, str], ...]
    bathymetry_variables: tuple[str, ...]
    variable_multipliers: dict[str, float]


SUPPORTED_GUI_INPUT_FORMATS: dict[str, GuiInputFormatSpec] = {
    'fm_netcdf': {
        'coordinate_candidates': (
            ('net_xcc', 'net_ycc'),
            ('mesh2d_face_x', 'mesh2d_face_y'),
        ),
        'bathymetry_variables': ('bedlevel', 'bed_level'),
        'variable_multipliers': {},
    },
    'd3d4_netcdf': {
        'coordinate_candidates': (
            ('XZ', 'YZ'),
            ('XCOR', 'YCOR'),
        ),
        'bathymetry_variables': ('DPS0', 'DP0', 'bedlevel', 'bed_level'),
        'variable_multipliers': {'DPS0': -1.0, 'DP0': -1.0},
    },
    'xbeach': {
        'coordinate_candidates': (
            ('globalx', 'globaly'),
            ('x', 'y'),
        ),
        'bathymetry_variables': ('zb_mean', 'zb'),
        'variable_multipliers': {},
    },
    'sfincs': {
        'coordinate_candidates': (
            ('mesh2d_node_x', 'mesh2d_node_y'),
            ('mesh2d_face_x', 'mesh2d_face_y'),
            ('x', 'y'),
        ),
        'bathymetry_variables': ('zb', 'bedlevel', 'bed_level'),
        'variable_multipliers': {},
    },
}


class SeedingGuiError(RuntimeError):
    """Raised for user-facing setup GUI errors."""


@dataclass(frozen=True)
class BathymetryViewData:
    """Spatial data needed by the seeding GUI map."""

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    variable: str
    input_file: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a SedTRAILS YAML file without converting timestamp strings.

    Parameters
    ----------
    config_path : str | Path
        Path to the SedTRAILS configuration file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the requested values.
    """

    path = Path(config_path)
    if not path.exists():
        raise SeedingGuiError(f'Configuration file not found: {path}')

    try:
        with path.open('r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=SedtrailsYamlLoader)
    except Exception as exc:
        raise SeedingGuiError(f'Could not read configuration: {exc}') from exc

    if not isinstance(config, dict):
        raise SeedingGuiError('Configuration file did not contain a YAML object.')

    return config


def get_population_names(config: dict[str, Any]) -> list[str]:
    """
    Return configured population names in file order.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration mapping used by the operation.

    Returns
    -------
    list[str]
        String result of the conversion.
    """

    populations = _get_populations(config)
    return [str(pop.get('name', f'population_{idx + 1}')) for idx, pop in enumerate(populations)]


def add_population_from_existing(
    config: dict[str, Any],
    *,
    source_population_name: str | None,
    new_population_name: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Return a copied config with a new population cloned from an existing one.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration mapping used by the operation.
    source_population_name : str | None
        Name of the population to copy.
    new_population_name : str | None
        Name for the copied population.

    Returns
    -------
    tuple[dict[str, Any], str]
        Dictionary containing the requested values.
    """

    updated = deepcopy(config)
    populations = _get_populations(updated)
    source_population = _select_population(populations, source_population_name)
    source_index = populations.index(source_population)
    resolved_name = _unique_population_name(
        get_population_names(updated),
        new_population_name or f"{source_population.get('name', 'population')}_copy",
    )
    new_population = deepcopy(source_population)
    new_population['name'] = resolved_name
    populations.insert(source_index + 1, new_population)
    return updated, resolved_name


def rename_population(config: dict[str, Any], *, old_name: str, new_name: str) -> dict[str, Any]:
    """
    Return a copied config with one population renamed.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration mapping used by the operation.
    old_name : str
        Existing population name to replace.
    new_name : str
        New population name to assign.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the requested values.
    """

    cleaned_name = new_name.strip()
    if not cleaned_name:
        raise SeedingGuiError('Population name cannot be empty.')

    updated = deepcopy(config)
    populations = _get_populations(updated)
    existing = [str(pop.get('name')) for pop in populations if pop.get('name') != old_name]
    if cleaned_name in existing:
        raise SeedingGuiError(f"Population '{cleaned_name}' already exists.")
    population = _select_population(populations, old_name)
    population['name'] = cleaned_name
    return updated


def remove_population(config: dict[str, Any], *, population_name: str) -> dict[str, Any]:
    """
    Return a copied config with one population removed.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration mapping used by the operation.
    population_name : str
        Name of the particle population.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the requested values.
    """

    updated = deepcopy(config)
    populations = _get_populations(updated)
    if len(populations) <= 1:
        raise SeedingGuiError('Cannot remove the only population.')
    population = _select_population(populations, population_name)
    populations.remove(population)
    return updated


def default_seeded_config_path(config_path: str | Path) -> Path:
    """
    Return the default copied-config path beside the source YAML file.

    Parameters
    ----------
    config_path : str | Path
        Path to the SedTRAILS configuration file.

    Returns
    -------
    Path
        Computed value returned by the function.
    """

    source = Path(config_path)
    suffix = source.suffix or '.yaml'
    return source.with_name(f'{source.stem}-seeded{suffix}')


def load_bathymetry_view_data(
    config_path: str | Path,
    *,
    format_override: str | None = None,
    variable: str | None = None,
) -> BathymetryViewData:
    """
    Load first-timestep map data for the GUI from a SedTRAILS config.

    Parameters
    ----------
    config_path : str | Path
        Path to the SedTRAILS configuration file.
    format_override : str | None
        Optional input format override.
    variable : str | None
        Name of the variable to inspect or sample.

    Returns
    -------
    BathymetryViewData
        Computed value returned by the function.
    """

    config_file = Path(config_path)
    config = load_config(config_file)
    input_model = config.get('general', {}).get('input_model', {})
    input_format = str(format_override or input_model.get('format') or '').strip().lower()
    format_spec = SUPPORTED_GUI_INPUT_FORMATS.get(input_format)
    if format_spec is None:
        supported = ', '.join(f"'{name}'" for name in SUPPORTED_GUI_INPUT_FORMATS)
        raise SeedingGuiError(
            f'Unsupported input format for setup GUI: {input_format!r}. Only {supported} are supported.'
        )

    data_path = config.get('inputs', {}).get('data')
    if not data_path:
        raise SeedingGuiError('Configuration is missing inputs.data.')

    input_file = _resolve_relative_path(data_path, config_file.parent)
    if not input_file.exists():
        raise SeedingGuiError(f'Input data file not found: {input_file}')

    dataset = _open_netcdf_dataset(input_file)

    variable_name = _resolve_bathymetry_variable(
        dataset,
        variable,
        default_candidates=format_spec['bathymetry_variables'],
    )
    try:
        values = _first_timestep_values(dataset[variable_name])
        value_multiplier = float(format_spec.get('variable_multipliers', {}).get(variable_name, 1.0))
        if value_multiplier != 1.0:
            values = values * value_multiplier
        values = np.asarray(values, dtype=float).reshape(-1)

        x, y = _resolve_coordinate_arrays(
            dataset,
            format_spec['coordinate_candidates'],
            expected_size=len(values),
        )
    except Exception as exc:
        raise SeedingGuiError(f'Could not extract map data: {exc}') from exc
    finally:
        close = getattr(dataset, 'close', None)
        if close is not None:
            close()

    if not (len(x) == len(y) == len(values)):
        raise SeedingGuiError(
            f'Map arrays have incompatible lengths: x={len(x)}, y={len(y)}, {variable_name}={len(values)}.'
        )
    x, y, values = _filter_finite_map_points(x, y, values, variable_name)

    return BathymetryViewData(x=x, y=y, values=values, variable=variable_name, input_file=input_file)


def _open_netcdf_dataset(input_file: Path) -> Any:
    try:
        import xarray as xr
    except Exception as exc:
        raise SeedingGuiError(f'Could not import xarray: {exc}') from exc

    errors: list[str] = []
    for engine in ('netcdf4', 'h5netcdf', 'scipy'):
        try:
            return xr.open_dataset(input_file, engine=engine, decode_timedelta=True)
        except Exception as exc:
            errors.append(f'{engine}: {exc}')

    details = '; '.join(errors)
    raise SeedingGuiError(f'Could not load input data with an explicit NetCDF engine: {details}')


_MAX_GLOBAL_DEDUPLICATION_POINTS = 100_000


def _filter_finite_map_points(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    variable_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite coordinate/value triples for GUI map operations."""
    x_array = np.asarray(x, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    value_array = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(value_array)
    if not np.any(finite):
        raise SeedingGuiError(f"No finite map points found for bathymetry variable '{variable_name}'.")

    x_finite = x_array[finite]
    y_finite = y_array[finite]
    values_finite = value_array[finite]

    # Some source files contain repeated/fill coordinates (often at origin), which
    # can distort triangulation. Global duplicate detection is expensive for large
    # maps, so only use it when the point count is bounded.
    if x_finite.size > _MAX_GLOBAL_DEDUPLICATION_POINTS:
        origin = (x_finite == 0.0) & (y_finite == 0.0)
        origin_count = np.count_nonzero(origin)
        if origin_count <= 1:
            return x_finite, y_finite, values_finite

        first_origin = int(np.argmax(origin))
        keep = ~origin
        keep[first_origin] = True
        values_collapsed = values_finite[keep]
        values_collapsed[first_origin] = np.mean(values_finite[origin])
        return x_finite[keep], y_finite[keep], values_collapsed

    xy = np.column_stack((x_finite, y_finite))
    unique_xy, inverse = np.unique(xy, axis=0, return_inverse=True)
    if unique_xy.shape[0] == xy.shape[0]:
        return x_finite, y_finite, values_finite

    value_sums = np.bincount(inverse, weights=values_finite)
    value_counts = np.bincount(inverse)
    unique_values = value_sums / value_counts
    return unique_xy[:, 0], unique_xy[:, 1], unique_values


def update_config_for_file_points(
    config: dict[str, Any],
    *,
    population_name: str | None,
    points_path: str,
) -> dict[str, Any]:
    """
    Return a copied config with one population using the file_points strategy.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration mapping used by the operation.
    population_name : str | None
        Name of the particle population.
    points_path : str
        Path to the seed-point file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the requested values.
    """

    updated = deepcopy(config)
    populations = _get_populations(updated)
    population = _select_population(populations, population_name)
    seeding = population.setdefault('seeding', {})
    seeding['strategy'] = {
        'file_points': {
            'path': points_path,
            'has_header': False,
            'x_col': 0,
            'y_col': 1,
            'deduplicate': False,
            'dropna': False,
            'stride': 1,
        }
    }
    return updated


def write_points_file(points: list[tuple[float, float]], output_path: str | Path) -> Path:
    """
    Write selected seed points as two whitespace-separated columns.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Point coordinates to process.
    output_path : str | Path
        Path where generated output is written.

    Returns
    -------
    Path
        Computed value returned by the function.
    """

    if not points:
        raise SeedingGuiError('No seed points selected.')

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for x, y in points:
            f.write(f'{float(x):.12g} {float(y):.12g}\n')
    return path


def generate_transect_points(vertices: list[tuple[float, float]], points_per_segment: int) -> list[tuple[float, float]]:
    """
    Generate equally spaced points from clicked endpoint pairs.

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        Polyline vertices used to generate points.
    points_per_segment : int
        Number of points generated per line segment.

    Returns
    -------
    list[tuple[float, float]]
        Tuple containing the computed values.
    """

    if points_per_segment < 1:
        raise SeedingGuiError('Transect points per segment must be at least 1.')
    if len(vertices) < 2 or len(vertices) % 2 != 0:
        raise SeedingGuiError('Transect mode needs an even number of clicked endpoints.')

    points: list[tuple[float, float]] = []
    for start, end in zip(vertices[0::2], vertices[1::2], strict=True):
        x1, y1 = start
        x2, y2 = end
        if points_per_segment == 1:
            points.append((float(x1), float(y1)))
            continue
        for idx in range(points_per_segment):
            fraction = idx / (points_per_segment - 1)
            points.append((float(x1 + fraction * (x2 - x1)), float(y1 + fraction * (y2 - y1))))
    return points


def generate_random_points_in_polygon(
    polygon: list[tuple[float, float]],
    *,
    nlocations: int,
    seed: int,
) -> list[tuple[float, float]]:
    """Generate uniformly sampled random points inside a polygon.

    Parameters
    ----------
    polygon : list of tuple of float
        Polygon vertices as ``(x, y)`` coordinate pairs.
    nlocations : int
        Number of seed points to generate.
    seed : int
        Seed for the NumPy random number generator.

    Returns
    -------
    list of tuple of float
        Random points clipped to the polygon.

    Raises
    ------
    SeedingGuiError
        If the requested point count is invalid or cannot be generated within
        the attempt limit.
    """

    if nlocations < 1:
        raise SeedingGuiError('Random nlocations must be at least 1.')
    path = _polygon_path(polygon)
    vertices = np.asarray(polygon, dtype=float)
    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)
    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = max(1000, nlocations * 500)
    batch_size = min(RANDOM_CANDIDATE_BATCH_SIZE, max_attempts)
    while len(points) < nlocations and attempts < max_attempts:
        candidate_count = min(batch_size, max_attempts - attempts)
        candidates = rng.uniform((xmin, ymin), (xmax, ymax), size=(candidate_count, 2))
        attempts += candidate_count
        accepted = candidates[path.contains_points(candidates)]
        if accepted.size == 0:
            continue
        needed = nlocations - len(points)
        points.extend((float(x), float(y)) for x, y in accepted[:needed])

    if len(points) < nlocations:
        raise SeedingGuiError(f'Only generated {len(points)} of {nlocations} random points before attempt limit.')
    return points


def generate_grid_points_in_polygon(
    polygon: list[tuple[float, float]],
    *,
    dx: float,
    dy: float,
    max_candidates: int | None = DEFAULT_GRID_MAX_CANDIDATES,
) -> list[tuple[float, float]]:
    """Generate a regular dx/dy grid clipped to a polygon.

    Parameters
    ----------
    polygon : list of tuple of float
        Polygon vertices as ``(x, y)`` coordinate pairs.
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    max_candidates : int or None, default=2_000_000
        Maximum number of candidate grid points to test before clipping.
        Set to ``None`` to disable this guard.

    Returns
    -------
    list of tuple of float
        Grid points clipped to the polygon.

    Raises
    ------
    SeedingGuiError
        If spacing is invalid or the candidate grid exceeds ``max_candidates``.
    """

    if dx <= 0 or dy <= 0:
        raise SeedingGuiError('Grid dx and dy must be positive.')
    path = _polygon_path(polygon)
    vertices = np.asarray(polygon, dtype=float)
    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)
    xs = np.arange(xmin, xmax + dx * 0.5, dx)
    ys = np.arange(ymin, ymax + dy * 0.5, dy)
    candidate_count = int(len(xs) * len(ys))
    if candidate_count == 0:
        return []
    if max_candidates is not None and candidate_count > max_candidates:
        raise SeedingGuiError(
            f'Grid spacing creates {candidate_count:,} candidate points, which exceeds the '
            f'{max_candidates:,} point safety limit. Increase dx/dy or pass a larger max_candidates value.'
        )

    points: list[tuple[float, float]] = []
    rows_per_batch = max(1, GRID_CANDIDATE_BATCH_SIZE // max(1, len(ys)))
    for start in range(0, len(xs), rows_per_batch):
        x_batch = xs[start : start + rows_per_batch]
        xx, yy = np.meshgrid(x_batch, ys, indexing='ij')
        candidates = np.column_stack((xx.ravel(), yy.ravel()))
        accepted = candidates[path.contains_points(candidates)]
        points.extend((float(x), float(y)) for x, y in accepted)
    return points


def clip_points_by_elevation(
    points: list[tuple[float, float]],
    *,
    field_x: np.ndarray,
    field_y: np.ndarray,
    field_values: np.ndarray,
    threshold: float,
    delete: str,
) -> list[tuple[float, float]]:
    """
    Delete selected points above or below an elevation using nearest source cell values.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Point coordinates to process.
    field_x : np.ndarray
        Field sample x coordinates.
    field_y : np.ndarray
        Field sample y coordinates.
    field_values : np.ndarray
        Field values sampled at the field coordinates.
    threshold : float
        Threshold used to filter points.
    delete : str
        Whether points meeting the threshold are removed.

    Returns
    -------
    list[tuple[float, float]]
        Tuple containing the computed values.
    """

    if delete not in {'above', 'below'}:
        raise SeedingGuiError("Clipping delete mode must be 'above' or 'below'.")
    if not points:
        return []

    from scipy.spatial import cKDTree

    source_xy = np.column_stack((np.asarray(field_x, dtype=float).reshape(-1), np.asarray(field_y, dtype=float).reshape(-1)))
    values = np.asarray(field_values, dtype=float).reshape(-1)
    tree = cKDTree(source_xy)
    return _clip_points_with_elevation_index(
        points,
        tree=tree,
        values=values,
        threshold=threshold,
        delete=delete,
    )


def _clip_points_with_elevation_index(
    points: list[tuple[float, float]],
    *,
    tree: Any,
    values: np.ndarray,
    threshold: float,
    delete: str,
) -> list[tuple[float, float]]:
    _, indices = tree.query(np.asarray(points, dtype=float))
    point_values = values[indices]
    if delete == 'above':
        keep = point_values <= threshold
    else:
        keep = point_values >= threshold
    return [(float(x), float(y)) for (x, y), keep_point in zip(points, keep, strict=True) if keep_point]


def _scaled_axis_limits(limits: tuple[float, float], *, factor: float) -> tuple[float, float]:
    center = (limits[0] + limits[1]) * 0.5
    half_span = (limits[1] - limits[0]) * factor * 0.5
    return center - half_span, center + half_span


def _panned_axis_limits(
    limits: tuple[float, float],
    *,
    pixel_delta: float,
    pixel_span: float,
) -> tuple[float, float]:
    if pixel_span <= 0:
        return limits
    data_delta = pixel_delta * (limits[1] - limits[0]) / pixel_span
    return limits[0] - data_delta, limits[1] - data_delta


class _Dropdown:
    """Small Matplotlib dropdown widget for compact in-figure selections."""

    def __init__(
        self,
        fig: Any,
        rect: tuple[float, float, float, float],
        *,
        label: str,
        options: tuple[str, ...] | list[str],
        selected: str,
        on_select: Any,
        option_height: float = 0.035,
    ) -> None:
        self.fig = fig
        self.rect = rect
        self.label = label
        self.on_select = on_select
        self.option_height = option_height
        self.options = [str(option) for option in options]
        self.selected = str(selected)
        self._open = False
        self._option_axes: list[Any] = []

        x, y, width, height = rect
        self.label_artist = fig.text(x - 0.010, y + height * 0.5, f'{label}:', ha='right', va='center', fontsize=9)
        self.ax = fig.add_axes(rect)
        self.ax.set_zorder(20)
        self.ax.set_facecolor('#f2f2f2')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.value_artist = self.ax.text(0.05, 0.5, self.selected, ha='left', va='center', fontsize=9)
        self.arrow_artist = self.ax.text(0.95, 0.5, 'v', ha='right', va='center', fontsize=8)
        self._rebuild_option_axes()
        self._connection_id = fig.canvas.mpl_connect('button_press_event', self._on_click)

    def set_options(self, options: tuple[str, ...] | list[str], *, selected: str) -> None:
        self.options = [str(option) for option in options]
        self.selected = str(selected)
        self.value_artist.set_text(self.selected)
        self._close()
        self._remove_option_axes()
        self._rebuild_option_axes()

    def set_selected(self, selected: str) -> None:
        self.selected = str(selected)
        self.value_artist.set_text(self.selected)
        self.fig.canvas.draw_idle()

    def _rebuild_option_axes(self) -> None:
        x, y, width, height = self.rect
        option_count = len(self.options)
        open_upward = y - option_count * self.option_height < 0.02
        for idx, option in enumerate(self.options):
            option_y = y + height + idx * self.option_height if open_upward else y - (idx + 1) * self.option_height
            option_ax = self.fig.add_axes((x, option_y, width, self.option_height))
            option_ax.set_zorder(30)
            option_ax.set_facecolor('#ffffff')
            option_ax.set_xticks([])
            option_ax.set_yticks([])
            option_ax.set_xlim(0, 1)
            option_ax.set_ylim(0, 1)
            option_ax.text(0.05, 0.5, option, ha='left', va='center', fontsize=9)
            option_ax.set_visible(False)
            self._option_axes.append(option_ax)

    def _remove_option_axes(self) -> None:
        for option_ax in self._option_axes:
            option_ax.remove()
        self._option_axes = []

    def _on_click(self, event: Any) -> None:
        if event.inaxes is self.ax:
            self._toggle()
            return

        if not self._open:
            return

        for idx, option_ax in enumerate(self._option_axes):
            if event.inaxes is option_ax:
                self.selected = self.options[idx]
                self.value_artist.set_text(self.selected)
                self._close()
                self.on_select(self.selected)
                return

        self._close()

    def _toggle(self) -> None:
        if self._open:
            self._close()
        else:
            self._open = True
            for option_ax in self._option_axes:
                option_ax.set_visible(True)
            self.fig.canvas.draw_idle()

    def _close(self) -> None:
        self._open = False
        for option_ax in self._option_axes:
            option_ax.set_visible(False)
        self.fig.canvas.draw_idle()


def save_seeded_config(
    *,
    source_config_path: str | Path,
    output_config_path: str | Path,
    points_output_path: str | Path,
    points: list[tuple[float, float]],
    population_name: str | None = None,
    config_data: dict[str, Any] | None = None,
    population_points: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[Path, Path]:
    """
    Write the point file and copied YAML config, then validate the YAML.

    Parameters
    ----------
    source_config_path : str | Path
        Path to the source configuration file.
    output_config_path : str | Path
        Path where the updated configuration file is written.
    points_output_path : str | Path
        Path where selected seed points are written.
    points : list[tuple[float, float]]
        Point coordinates to process.
    population_name : str | None
        Name of the particle population.
    config_data : dict[str, Any] | None
        Configuration data to update.
    population_points : dict[str, list[tuple[float, float]]] | None
        Seed points grouped by population.

    Returns
    -------
    tuple[Path, Path]
        Tuple containing the computed values.
    """

    source_path = Path(source_config_path)
    output_path = Path(output_config_path)
    points_path = Path(points_output_path)
    if not points_path.is_absolute() and points_path.parent == Path('.'):
        points_path = output_path.parent / points_path.name

    config = deepcopy(config_data) if config_data is not None else load_config(source_path)
    nonempty_population_points = {
        name: pop_points for name, pop_points in (population_points or {}).items() if pop_points
    }
    if nonempty_population_points:
        updated_config = deepcopy(config)
        single_population = len(nonempty_population_points) == 1
        last_points_path = points_path
        for point_population_name, point_values in nonempty_population_points.items():
            pop_points_path = points_path if single_population else _population_points_path(points_path, point_population_name)
            write_points_file(point_values, pop_points_path)
            last_points_path = pop_points_path
            updated_config = update_config_for_file_points(
                updated_config,
                population_name=point_population_name,
                points_path=_yaml_path_value(pop_points_path, output_path.parent),
            )
        points_path = last_points_path
    else:
        write_points_file(points, points_path)
        yaml_points_path = _yaml_path_value(points_path, output_path.parent)
        updated_config = update_config_for_file_points(
            config,
            population_name=population_name,
            points_path=yaml_points_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(updated_config, f, sort_keys=False, default_flow_style=False)

    try:
        YAMLConfigValidator().validate_yaml(str(output_path))
    except Exception as exc:
        raise SeedingGuiError(f'Generated YAML did not validate: {exc}') from exc

    return output_path, points_path


def launch_seeding_gui(
    *,
    config_path: str | Path,
    output_path: str | Path | None = None,
    points_output_path: str | Path | None = None,
    population_name: str | None = None,
    format_override: str | None = None,
    variable: str | None = None,
) -> None:
    """
    Open the Matplotlib seed-point selection GUI.

    Parameters
    ----------
    config_path : str | Path
        Path to the SedTRAILS configuration file.
    output_path : str | Path | None
        Path where generated output is written.
    points_output_path : str | Path | None
        Path where selected seed points are written.
    population_name : str | None
        Name of the particle population.
    format_override : str | None
        Optional input format override.
    variable : str | None
        Name of the variable to inspect or sample.
    """

    app = SeedingGuiApp(
        config_path=Path(config_path),
        output_path=Path(output_path) if output_path is not None else None,
        points_output_path=Path(points_output_path) if points_output_path is not None else None,
        population_name=population_name,
        format_override=format_override,
        variable=variable,
    )
    app.show()


class SeedingGuiApp:
    """Interactive Matplotlib application for selecting SedTRAILS seed points.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to the SedTRAILS configuration file to edit.
    output_path : pathlib.Path or None
        Destination for the seeded configuration. When None, a default sibling
        configuration path is used.
    points_output_path : pathlib.Path or None
        Destination for generated point coordinates. When None, a path derived
        from `output_path` is used.
    population_name : str or None
        Population selected for editing. When None, the first population is
        selected.
    format_override : str or None
        Optional input-model format override used to load bathymetry.
    variable : str or None
        Optional bathymetry variable name.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        output_path: Path | None,
        points_output_path: Path | None,
        population_name: str | None,
        format_override: str | None,
        variable: str | None,
    ) -> None:
        """Initialize the interactive seeding application.

        Parameters
        ----------
        config_path : pathlib.Path
            Path to the SedTRAILS configuration file to edit.
        output_path : pathlib.Path or None
            Destination for the seeded configuration. When None, a default
            sibling configuration path is used.
        points_output_path : pathlib.Path or None
            Destination for generated point coordinates. When None, a path
            derived from `output_path` is used.
        population_name : str or None
            Population selected for editing. When None, the first population
            is selected.
        format_override : str or None
            Optional input-model format override used to load bathymetry.
        variable : str or None
            Optional bathymetry variable name.

        Raises
        ------
        SeedingGuiError
            If `population_name` does not identify a configured population.
        """
        self.config_path = config_path
        self.output_path = output_path or default_seeded_config_path(config_path)
        self.points_output_path = points_output_path or self.output_path.with_suffix('.points.txt')
        self.population_name = population_name
        self.format_override = format_override
        self.variable = variable
        self.draft_vertices: list[tuple[float, float]] = []
        self.strategy_mode = 'points'
        self.polygon_closed = False
        self._nearest_remove_pixels = 12.0
        self.colormap_name = 'SEAWAD'
        self._bathymetry_vmin = -20.0
        self._bathymetry_vmax = 10.0
        self._max_display_points = 500_000
        self._clip_tree: Any | None = None
        self._clip_values: np.ndarray | None = None
        self._pan_active = False
        self._pan_start: dict[str, Any] | None = None

        self.config = load_config(self.config_path)
        self.population_names = get_population_names(self.config)
        if self.population_name is None and self.population_names:
            self.population_name = self.population_names[0]
        elif self.population_name not in self.population_names:
            raise SeedingGuiError(f'Population not found: {self.population_name}')
        self.population_points: dict[str, list[tuple[float, float]]] = {name: [] for name in self.population_names}
        self.points = self.population_points[self.population_name]

        self.view_data = load_bathymetry_view_data(
            self.config_path,
            format_override=self.format_override,
            variable=self.variable,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.widgets import Button, TextBox
        from sedtrails.pathway_visualizer.colormaps import bathymetry_colormap

        self.fig, self.ax = plt.subplots(figsize=(12.4, 7.0))
        self.fig.subplots_adjust(left=0.07, right=0.66, bottom=0.30, top=0.90)

        self.triangulation = _create_masked_triangulation(self.view_data.x, self.view_data.y, mtri_module=mtri)
        self.bathymetry_cmap, self.bathymetry_norm = bathymetry_colormap(
            self.colormap_name,
            vmin=self._bathymetry_vmin,
            vmax=self._bathymetry_vmax,
        )
        self.bathymetry_field = self.ax.tricontourf(
            self.triangulation,
            self.view_data.values,
            levels=32,
            cmap=self.bathymetry_cmap,
            norm=self.bathymetry_norm,
        )
        self.colorbar = self.fig.colorbar(self.bathymetry_field, ax=self.ax, label=self.view_data.variable)
        self.point_artist = self.ax.scatter([], [], marker='x', c='red', s=42, linewidths=1.8, label='seed points')
        (self.draft_artist,) = self.ax.plot(
            [],
            [],
            '-o',
            color='#f4a261',
            markerfacecolor='white',
            markeredgecolor='#f4a261',
            linewidth=1.5,
            markersize=5,
            label='draft geometry',
        )

        self.ax.set_title(f'{self.view_data.variable} at first timestep')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.legend(loc='upper right')

        self.fig.canvas.mpl_connect('button_press_event', self._on_map_button_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_map_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_map_button_release)

        axes = {
            'save': self.fig.add_axes((0.07, 0.055, 0.105, 0.055)),
            'save_as': self.fig.add_axes((0.19, 0.055, 0.105, 0.055)),
            'validate': self.fig.add_axes((0.31, 0.055, 0.105, 0.055)),
            'undo': self.fig.add_axes((0.43, 0.055, 0.105, 0.055)),
            'clear': self.fig.add_axes((0.55, 0.055, 0.105, 0.055)),
        }
        self._buttons = [
            Button(axes['save'], 'Save'),
            Button(axes['save_as'], 'Save as'),
            Button(axes['validate'], 'Validate'),
            Button(axes['undo'], 'Undo'),
            Button(axes['clear'], 'Clear'),
        ]
        self._buttons[0].on_clicked(self._save)
        self._buttons[1].on_clicked(self._save_as)
        self._buttons[2].on_clicked(self._validate)
        self._buttons[3].on_clicked(self._undo)
        self._buttons[4].on_clicked(self._clear)

        map_axes = {
            'zoom_in': self.fig.add_axes((0.07, 0.155, 0.070, 0.040)),
            'zoom_out': self.fig.add_axes((0.150, 0.155, 0.070, 0.040)),
            'pan': self.fig.add_axes((0.230, 0.155, 0.070, 0.040)),
        }
        self._zoom_in_button = Button(map_axes['zoom_in'], 'Zoom +')
        self._zoom_out_button = Button(map_axes['zoom_out'], 'Zoom -')
        self._pan_button = Button(map_axes['pan'], 'Pan')
        self._map_buttons = [self._zoom_in_button, self._zoom_out_button, self._pan_button]
        self._zoom_in_button.on_clicked(self._zoom_in)
        self._zoom_out_button.on_clicked(self._zoom_out)
        self._pan_button.on_clicked(self._toggle_pan_mode)
        self._update_pan_button_style()

        panel_x = 0.79
        panel_w = 0.17
        small_w = 0.060
        right_x = 0.915
        right_w = 0.045
        field_h = 0.040

        self._mode_dropdown = _Dropdown(
            self.fig,
            (panel_x, 0.855, panel_w, field_h),
            label='mode',
            options=SEEDING_MODES,
            selected=self.strategy_mode,
            on_select=self._set_strategy_mode,
        )
        self._population_dropdown = None
        self._rebuild_population_selector()
        self._colormap_dropdown = _Dropdown(
            self.fig,
            (panel_x, 0.735, panel_w, field_h),
            label='colors',
            options=('SEAWAD', 'Vintage'),
            selected=self.colormap_name,
            on_select=self._set_colormap,
        )

        self._population_name_box = TextBox(
            self.fig.add_axes((panel_x, 0.660, panel_w, field_h)),
            '',
            initial=self.population_name,
        )
        self.fig.text(panel_x - 0.010, 0.680, 'edit pop:', ha='right', va='center', fontsize=9)
        self._add_population_button = Button(self.fig.add_axes((0.735, 0.600, 0.065, field_h)), 'Add')
        self._rename_population_button = Button(self.fig.add_axes((0.812, 0.600, 0.070, field_h)), 'Rename')
        self._remove_population_button = Button(self.fig.add_axes((0.895, 0.600, 0.065, field_h)), 'Remove')
        self._add_population_button.on_clicked(self._add_population)
        self._rename_population_button.on_clicked(self._rename_population)
        self._remove_population_button.on_clicked(self._remove_population)

        self._n_box = TextBox(self.fig.add_axes((panel_x, 0.520, small_w, field_h)), '', initial='20')
        self._n_label = self.fig.text(panel_x - 0.010, 0.540, 'n:', ha='right', va='center', fontsize=9)
        self._seed_box = TextBox(self.fig.add_axes((right_x, 0.460, right_w, field_h)), '', initial='42')
        self._seed_label = self.fig.text(right_x - 0.010, 0.480, 'seed:', ha='right', va='center', fontsize=9)
        self._dx_box = TextBox(self.fig.add_axes((panel_x, 0.400, small_w, field_h)), '', initial='100')
        self._dx_label = self.fig.text(panel_x - 0.010, 0.420, 'dx:', ha='right', va='center', fontsize=9)
        self._dy_box = TextBox(self.fig.add_axes((right_x, 0.400, right_w, field_h)), '', initial='100')
        self._dy_label = self.fig.text(right_x - 0.010, 0.420, 'dy:', ha='right', va='center', fontsize=9)

        self._generate_button = Button(self.fig.add_axes((panel_x, 0.335, panel_w, 0.045)), 'Generate')
        self._generate_button.on_clicked(self._generate_from_strategy)

        self._cmin_box = TextBox(self.fig.add_axes((panel_x, 0.255, small_w, field_h)), '', initial=str(self._bathymetry_vmin))
        self.fig.text(panel_x - 0.010, 0.275, 'cmin:', ha='right', va='center', fontsize=9)
        self._cmax_box = TextBox(self.fig.add_axes((right_x, 0.255, right_w, field_h)), '', initial=str(self._bathymetry_vmax))
        self.fig.text(right_x - 0.010, 0.275, 'cmax:', ha='right', va='center', fontsize=9)
        self._color_button = Button(self.fig.add_axes((panel_x, 0.190, panel_w, 0.045)), 'Apply color limits')
        self._color_button.on_clicked(self._apply_color_limits)

        self._clip_mode = 'above'
        self._clip_dropdown = _Dropdown(
            self.fig,
            (panel_x, 0.125, 0.080, field_h),
            label='clip',
            options=('above', 'below'),
            selected=self._clip_mode,
            on_select=self._set_clip_mode,
        )
        self._clip_elevation_box = TextBox(self.fig.add_axes((right_x, 0.125, right_w, field_h)), '', initial='0')
        self.fig.text(right_x - 0.010, 0.145, 'z:', ha='right', va='center', fontsize=9)
        self._clip_button = Button(self.fig.add_axes((panel_x, 0.060, panel_w, 0.045)), 'Clip points')
        self._clip_button.on_clicked(self._clip_selected_points)

        self._set_n_input_state()
        self._set_random_seed_state()
        self._set_grid_input_state()
        self.status_text = self.fig.text(0.07, 0.125, self._status_message(), fontsize=8)

    def show(self) -> None:
        """Display the seeding GUI.

        Notes
        -----
        This method blocks until the Matplotlib window is closed.
        """
        import matplotlib.pyplot as plt

        plt.show()

    def _on_map_click(self, event: Any) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return

        if self.strategy_mode == 'points':
            self._on_points_click(event)
            return

        if self.strategy_mode == 'transect':
            self._on_transect_click(event)
            return

        if self.strategy_mode in {'random', 'grid'}:
            self._on_polygon_click(event)
            return

    def _on_map_button_press(self, event: Any) -> None:
        if self._pan_active:
            self._start_pan(event)
            return

        self._on_map_click(event)

    def _on_map_motion(self, event: Any) -> None:
        if self._pan_start is None or event.x is None or event.y is None:
            return

        bbox_width = self._pan_start['bbox_width']
        bbox_height = self._pan_start['bbox_height']
        xlim = _panned_axis_limits(
            self._pan_start['xlim'],
            pixel_delta=float(event.x) - self._pan_start['x'],
            pixel_span=bbox_width,
        )
        ylim = _panned_axis_limits(
            self._pan_start['ylim'],
            pixel_delta=float(event.y) - self._pan_start['y'],
            pixel_span=bbox_height,
        )
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.fig.canvas.draw_idle()

    def _on_map_button_release(self, _event: Any) -> None:
        self._pan_start = None

    def _start_pan(self, event: Any) -> None:
        if event.inaxes is not self.ax or event.button != 1 or event.x is None or event.y is None:
            return

        bbox = self.ax.bbox
        self._pan_start = {
            'x': float(event.x),
            'y': float(event.y),
            'xlim': self.ax.get_xlim(),
            'ylim': self.ax.get_ylim(),
            'bbox_width': float(bbox.width),
            'bbox_height': float(bbox.height),
        }

    def _zoom_in(self, _event: Any = None) -> None:
        self._zoom_map(MAP_ZOOM_IN_FACTOR)

    def _zoom_out(self, _event: Any = None) -> None:
        self._zoom_map(MAP_ZOOM_OUT_FACTOR)

    def _zoom_map(self, factor: float) -> None:
        self.ax.set_xlim(_scaled_axis_limits(self.ax.get_xlim(), factor=factor))
        self.ax.set_ylim(_scaled_axis_limits(self.ax.get_ylim(), factor=factor))
        self.fig.canvas.draw_idle()

    def _toggle_pan_mode(self, _event: Any = None) -> None:
        self._pan_active = not self._pan_active
        self._pan_start = None
        self._update_pan_button_style()
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _update_pan_button_style(self) -> None:
        facecolor = '0.70' if self._pan_active else '0.85'
        hovercolor = '0.78' if self._pan_active else '0.95'
        self._pan_button.color = facecolor
        self._pan_button.hovercolor = hovercolor
        self._pan_button.ax.set_facecolor(facecolor)

    def _on_points_click(self, event: Any) -> None:
        if event.button == 1:
            self.points.append((float(event.xdata), float(event.ydata)))
            self._refresh_points()
            return

        if event.button == 3:
            self._remove_nearest(event)

    def _on_transect_click(self, event: Any) -> None:
        if event.button == 1:
            self.draft_vertices.append((float(event.xdata), float(event.ydata)))
            self._refresh_draft()
            return

        if event.button == 3 and self.draft_vertices:
            self.draft_vertices.pop()
            self._refresh_draft()

    def _on_polygon_click(self, event: Any) -> None:
        if event.button == 1:
            if self.polygon_closed:
                self.draft_vertices.clear()
                self.polygon_closed = False
            self.draft_vertices.append((float(event.xdata), float(event.ydata)))
            self._refresh_draft()
            return

        if event.button == 3 and len(self.draft_vertices) >= 3:
            self.polygon_closed = True
            self._refresh_draft()

    def _remove_nearest(self, event: Any) -> None:
        if not self.points:
            return

        point_pixels = self.ax.transData.transform(np.asarray(self.points))
        click_pixel = np.asarray([event.x, event.y])
        distances = np.linalg.norm(point_pixels - click_pixel, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= self._nearest_remove_pixels:
            self.points.pop(nearest)
            self._refresh_points()

    def _refresh_points(self) -> None:
        if self._should_display_points() and self.points:
            self.point_artist.set_offsets(np.asarray(self.points))
        else:
            self.point_artist.set_offsets(np.empty((0, 2)))
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _refresh_draft(self) -> None:
        vertices = list(self.draft_vertices)
        if self.polygon_closed and len(vertices) >= 3:
            vertices.append(vertices[0])
        if vertices:
            xs, ys = zip(*vertices, strict=True)
            self.draft_artist.set_data(xs, ys)
        else:
            self.draft_artist.set_data([], [])
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _set_points(self, points: list[tuple[float, float]]) -> None:
        self.points = points
        self.population_points[self.population_name] = self.points
        self._refresh_points()

    def _rebuild_population_selector(self) -> None:
        self.population_names = get_population_names(self.config)
        for population_name in self.population_names:
            self.population_points.setdefault(population_name, [])
        if self.population_name not in self.population_names:
            self.population_name = self.population_names[0]
        if self._population_dropdown is None:
            self._population_dropdown = _Dropdown(
                self.fig,
                (0.79, 0.795, 0.17, 0.040),
                label='population',
                options=self.population_names,
                selected=self.population_name,
                on_select=self._set_population,
            )
        else:
            self._population_dropdown.set_options(self.population_names, selected=self.population_name)
        self.fig.canvas.draw_idle()

    def _set_strategy_mode(self, label: str) -> None:
        self.strategy_mode = label
        self.draft_vertices.clear()
        self.polygon_closed = False
        self._set_n_input_state()
        self._set_random_seed_state()
        self._set_grid_input_state()
        self._refresh_draft()

    def _set_textbox_state(self, textbox: Any, label: Any, *, active: bool) -> None:
        textbox.set_active(active)
        color = 'black' if active else '0.6'
        label.set_color(color)
        textbox.ax.patch.set_edgecolor(color)
        for spine in textbox.ax.spines.values():
            spine.set_color(color)
        text_artist = getattr(textbox, 'text_disp', None) or getattr(textbox, 'text', None)
        if text_artist is not None:
            text_artist.set_color(color)

    def _set_n_input_state(self) -> None:
        active = self.strategy_mode in {'transect', 'random'}
        self._set_textbox_state(self._n_box, self._n_label, active=active)

    def _set_random_seed_state(self) -> None:
        active = self.strategy_mode == 'random'
        self._set_textbox_state(self._seed_box, self._seed_label, active=active)

    def _set_grid_input_state(self) -> None:
        active = self.strategy_mode == 'grid'
        self._set_textbox_state(self._dx_box, self._dx_label, active=active)
        self._set_textbox_state(self._dy_box, self._dy_label, active=active)

    def _set_population(self, label: str) -> None:
        if label not in self.population_points:
            self.population_points[label] = []
        self.population_name = label
        self.points = self.population_points[label]
        self._population_name_box.set_val(label)
        if self._population_dropdown is not None:
            self._population_dropdown.set_selected(label)
        self._refresh_points()
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _add_population(self, _event: Any = None) -> None:
        source_name = self.population_name
        try:
            self.config, new_name = add_population_from_existing(
                self.config,
                source_population_name=source_name,
                new_population_name=self._population_name_box.text,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.population_points.setdefault(new_name, [])
        self.population_name = new_name
        self.points = self.population_points[new_name]
        self._population_name_box.set_val(new_name)
        self._rebuild_population_selector()
        self._refresh_points()
        self._show_info(
            f"Added population '{new_name}' by copying '{source_name}'.\n\n"
            'Check the generated YAML before running to confirm the copied particle settings are correct.'
        )

    def _rename_population(self, _event: Any = None) -> None:
        old_name = self.population_name
        new_name = self._population_name_box.text.strip()
        try:
            self.config = rename_population(self.config, old_name=old_name, new_name=new_name)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.population_points[new_name] = self.population_points.pop(old_name, [])
        self.population_name = new_name
        self.points = self.population_points[new_name]
        self._rebuild_population_selector()
        self._refresh_points()

    def _remove_population(self, _event: Any = None) -> None:
        removed_name = self.population_name
        try:
            self.config = remove_population(self.config, population_name=removed_name)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.population_points.pop(removed_name, None)
        self.population_names = get_population_names(self.config)
        self.population_name = self.population_names[0]
        self.points = self.population_points.setdefault(self.population_name, [])
        self._population_name_box.set_val(self.population_name)
        self._rebuild_population_selector()
        self._refresh_points()

    def _set_clip_mode(self, label: str) -> None:
        self._clip_mode = label

    def _set_colormap(self, label: str) -> None:
        self.colormap_name = label
        try:
            self._redraw_bathymetry()
        except Exception as exc:
            self._show_error(str(exc))

    def _generate_from_strategy(self, _event: Any = None) -> None:
        try:
            if self.strategy_mode == 'points':
                raise SeedingGuiError('Points mode does not need generation; left click to add points.')
            if self.strategy_mode == 'transect':
                generated = generate_transect_points(
                    self.draft_vertices,
                    self._parse_int_box(self._n_box, 'n'),
                )
            elif self.strategy_mode == 'random':
                if not self.polygon_closed:
                    raise SeedingGuiError('Right click to close the random polygon before generating points.')
                generated = generate_random_points_in_polygon(
                    self.draft_vertices,
                    nlocations=self._parse_int_box(self._n_box, 'n'),
                    seed=self._parse_int_box(self._seed_box, 'random seed'),
                )
            elif self.strategy_mode == 'grid':
                if not self.polygon_closed:
                    raise SeedingGuiError('Right click to close the grid polygon before generating points.')
                generated = generate_grid_points_in_polygon(
                    self.draft_vertices,
                    dx=self._parse_float_box(self._dx_box, 'grid dx'),
                    dy=self._parse_float_box(self._dy_box, 'grid dy'),
                )
            else:
                raise SeedingGuiError(f'Unknown seeding mode: {self.strategy_mode}')
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._set_points(generated)

    def _clip_selected_points(self, _event: Any = None) -> None:
        try:
            tree, values = self._get_clip_index()
            clipped_points = _clip_points_with_elevation_index(
                self.points,
                tree=tree,
                values=values,
                threshold=self._parse_float_box(self._clip_elevation_box, 'clip elevation'),
                delete=self._clip_mode,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._set_points(clipped_points)

    def _get_clip_index(self) -> tuple[Any, np.ndarray]:
        if self._clip_tree is None or self._clip_values is None:
            from scipy.spatial import cKDTree

            source_xy = np.column_stack(
                (
                    np.asarray(self.view_data.x, dtype=float).reshape(-1),
                    np.asarray(self.view_data.y, dtype=float).reshape(-1),
                )
            )
            self._clip_values = np.asarray(self.view_data.values, dtype=float).reshape(-1)
            self._clip_tree = cKDTree(source_xy)
        return self._clip_tree, self._clip_values

    def _apply_color_limits(self, _event: Any = None) -> None:
        try:
            vmin = self._parse_float_box(self._cmin_box, 'color minimum')
            vmax = self._parse_float_box(self._cmax_box, 'color maximum')
            if vmin >= vmax:
                raise SeedingGuiError('Color minimum must be less than color maximum.')
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._bathymetry_vmin = vmin
        self._bathymetry_vmax = vmax
        try:
            self._redraw_bathymetry()
        except Exception as exc:
            self._show_error(str(exc))

    def _redraw_bathymetry(self) -> None:
        from sedtrails.pathway_visualizer.colormaps import bathymetry_colormap

        self.bathymetry_cmap, self.bathymetry_norm = bathymetry_colormap(
            self.colormap_name,
            vmin=self._bathymetry_vmin,
            vmax=self._bathymetry_vmax,
        )
        if hasattr(self.bathymetry_field, 'remove'):
            self.bathymetry_field.remove()
        else:
            for collection in getattr(self.bathymetry_field, 'collections', []):
                collection.remove()
        levels = np.linspace(self._bathymetry_vmin, self._bathymetry_vmax, 32)
        self.bathymetry_field = self.ax.tricontourf(
            self.triangulation,
            self.view_data.values,
            levels=levels,
            cmap=self.bathymetry_cmap,
            norm=self.bathymetry_norm,
        )
        self.colorbar.update_normal(self.bathymetry_field)
        self.fig.canvas.draw_idle()

    def _parse_int_box(self, box: Any, label: str) -> int:
        value = self._parse_float_box(box, label)
        if int(value) != value:
            raise SeedingGuiError(f'{label} must be an integer.')
        return int(value)

    def _parse_float_box(self, box: Any, label: str) -> float:
        try:
            return float(str(box.text).strip())
        except ValueError as exc:
            raise SeedingGuiError(f'{label} must be numeric.') from exc

    def _save(self, _event: Any = None) -> None:
        try:
            config_path, points_path = save_seeded_config(
                source_config_path=self.config_path,
                output_config_path=self.output_path,
                points_output_path=self.points_output_path,
                points=self.points,
                population_name=self.population_name,
                config_data=self.config,
                population_points=self.population_points,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._show_info(f'Saved config:\n{config_path}\n\nSaved points:\n{points_path}')
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _save_as(self, _event: Any = None) -> None:
        try:
            from tkinter import filedialog

            selected = filedialog.asksaveasfilename(
                title='Save SedTRAILS config as',
                defaultextension='.yaml',
                filetypes=[('YAML files', '*.yaml *.yml'), ('All files', '*.*')],
                initialdir=str(self.config_path.parent),
                initialfile=self.output_path.name,
            )
        except Exception as exc:
            self._show_error(f'Could not open save dialog: {exc}')
            return

        if selected:
            self.output_path = Path(selected)
            self.points_output_path = self.output_path.with_suffix('.points.txt')
            self._save()

    def _validate(self, _event: Any = None) -> None:
        try:
            YAMLConfigValidator().validate_yaml(str(self.output_path))
        except Exception as exc:
            self._show_error(f'Validation failed:\n{exc}')
            return
        self._show_info(f'Configuration validates:\n{self.output_path}')

    def _undo(self, _event: Any = None) -> None:
        if self.points:
            self.points.pop()
            self._refresh_points()

    def _clear(self, _event: Any = None) -> None:
        self.points.clear()
        self.draft_vertices.clear()
        self.polygon_closed = False
        self._refresh_points()
        self._refresh_draft()

    def _status_message(self) -> str:
        hints = {
            'points': 'left add, right remove',
            'transect': 'click endpoint pairs, then Generate',
            'random': 'click polygon, right close, then Generate',
            'grid': 'click polygon, right close, then Generate',
        }
        total_points = sum(len(points) for points in self.population_points.values())
        warning = ''
        if not self._should_display_points() and self.points:
            warning = f' >{self._max_display_points:,} not displayed.'
        navigation = ' | Pan: on, drag map' if self._pan_active else ''
        return (
            f'Mode: {self.strategy_mode} | Pop: {self.population_name} | Seeds: {len(self.points)} | Total: {total_points} | '
            f'Draft: {len(self.draft_vertices)} | {hints.get(self.strategy_mode, "")}.{warning}{navigation}'
        )

    def _should_display_points(self) -> bool:
        return len(self.points) <= self._max_display_points

    def _show_error(self, message: str) -> None:
        self._show_dialog('SedTRAILS seeding setup error', message, error=True)

    def _show_info(self, message: str) -> None:
        self._show_dialog('SedTRAILS seeding setup', message, error=False)

    def _show_dialog(self, title: str, message: str, *, error: bool) -> None:
        try:
            from tkinter import messagebox

            if error:
                messagebox.showerror(title, message)
            else:
                messagebox.showinfo(title, message)
        except Exception:
            print(f'{title}: {message}')


def _first_timestep_values(variable: Any) -> np.ndarray:
    data = variable
    dims = getattr(data, 'dims', ())
    if 'layer' in dims:
        data = data.isel(layer=0)
    dims = getattr(data, 'dims', ())
    for time_dim in ('time', 'meantime', 'globaltime'):
        if time_dim in dims:
            data = data.isel({time_dim: 0})
            break
    return np.asarray(data.values)


def _create_masked_triangulation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mtri_module: Any,
) -> Any:
    """Build a triangulation and mask triangles likely to be rendering artifacts."""

    triangulation = mtri_module.Triangulation(x, y)
    triangles = getattr(triangulation, 'triangles', None)
    if triangles is None or len(triangles) == 0:
        return triangulation

    points = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    triangle_points = points[triangles]
    edge_01 = np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 1], axis=1)
    edge_12 = np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 2], axis=1)
    edge_20 = np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 0], axis=1)
    max_edge = np.maximum.reduce((edge_01, edge_12, edge_20))

    q1, q3 = np.percentile(max_edge, (25.0, 75.0))
    iqr = q3 - q1
    if iqr > 0:
        long_edge_threshold = q3 + 10.0 * iqr
    else:
        q95 = np.percentile(max_edge, 95.0)
        if q3 > 0 and q95 > (3.0 * q3):
            long_edge_threshold = 3.0 * q3
        else:
            long_edge_threshold = float('inf')

    long_edge_mask = max_edge > long_edge_threshold
    flat_mask = mtri_module.TriAnalyzer(triangulation).get_flat_tri_mask(min_circle_ratio=0.01)
    combined_mask = np.asarray(flat_mask, dtype=bool) | np.asarray(long_edge_mask, dtype=bool)
    if np.any(combined_mask):
        triangulation.set_mask(combined_mask)
    return triangulation


def _polygon_path(polygon: list[tuple[float, float]]) -> Any:
    if len(polygon) < 3:
        raise SeedingGuiError('A polygon needs at least three clicked vertices.')
    from matplotlib.path import Path as MplPath

    return MplPath(np.asarray(polygon, dtype=float))


def _resolve_bathymetry_variable(
    dataset: Any,
    requested: str | None,
    *,
    default_candidates: tuple[str, ...] = ('bedlevel', 'bed_level'),
) -> str:
    candidates = [requested] if requested else list(default_candidates)
    for candidate in candidates:
        if candidate and candidate in dataset:
            return candidate
    available = ', '.join(str(name) for name in dataset.data_vars)
    if requested:
        raise SeedingGuiError(f"Requested bathymetry variable '{requested}' was not found. Available variables: {available}")
    defaults = ' and '.join(f"'{name}'" for name in default_candidates)
    raise SeedingGuiError(f'No bathymetry variable found. Tried {defaults}. Available variables: {available}')


def _resolve_coordinate_arrays(
    dataset: Any,
    coordinate_candidates: tuple[tuple[str, str], ...],
    *,
    expected_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    available_pairs: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for x_name, y_name in coordinate_candidates:
        if x_name in dataset and y_name in dataset:
            x_values = np.asarray(dataset[x_name].values, dtype=float).reshape(-1)
            y_values = np.asarray(dataset[y_name].values, dtype=float).reshape(-1)
            available_pairs.append((x_name, y_name, x_values, y_values))
            if expected_size is None:
                return x_values, y_values

            if x_values.size == y_values.size == expected_size:
                return x_values, y_values

    # SFINCS fallback: derive face centroids from node coordinates + face connectivity.
    if (
        expected_size is not None
        and 'mesh2d_node_x' in dataset
        and 'mesh2d_node_y' in dataset
        and 'mesh2d_face_nodes' in dataset
    ):
        node_x = np.asarray(dataset['mesh2d_node_x'].values, dtype=float).reshape(-1)
        node_y = np.asarray(dataset['mesh2d_node_y'].values, dtype=float).reshape(-1)
        face_nodes_var = dataset['mesh2d_face_nodes']
        face_nodes = np.asarray(face_nodes_var.values)
        start_index = int(face_nodes_var.attrs.get('start_index', 0))
        fill_value = face_nodes_var.encoding.get('_FillValue', face_nodes_var.attrs.get('_FillValue', -1))
        face_x, face_y = _compute_face_centroids(
            node_x,
            node_y,
            face_nodes,
            start_index=start_index,
            fill_value=fill_value,
        )
        if face_x.size == face_y.size == expected_size:
            return face_x, face_y

    if available_pairs:
        return available_pairs[0][2], available_pairs[0][3]

    tried = ', '.join(f"'{x_name}/{y_name}'" for x_name, y_name in coordinate_candidates)
    available = ', '.join(str(name) for name in dataset.variables)
    raise SeedingGuiError(f'Missing coordinate variable pair. Tried {tried}. Available variables: {available}')


def _compute_face_centroids(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    *,
    start_index: int,
    fill_value: int | float | None,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.asarray(face_nodes, dtype=np.int64) - int(start_index)
    invalid = normalized < 0
    if fill_value is not None:
        invalid |= np.asarray(face_nodes) == fill_value
    normalized[invalid] = -1

    face_x = np.empty(normalized.shape[0], dtype=float)
    face_y = np.empty(normalized.shape[0], dtype=float)
    for idx, nodes in enumerate(normalized):
        valid = nodes[nodes >= 0]
        if valid.size == 0:
            face_x[idx] = np.nan
            face_y[idx] = np.nan
            continue
        face_x[idx] = float(np.mean(node_x[valid]))
        face_y[idx] = float(np.mean(node_y[valid]))
    return face_x, face_y


def _get_populations(config: dict[str, Any]) -> list[dict[str, Any]]:
    populations = config.get('particles', {}).get('populations')
    if not isinstance(populations, list) or not populations:
        raise SeedingGuiError('Configuration has no particles.populations entries.')
    return populations


def _select_population(populations: list[dict[str, Any]], population_name: str | None) -> dict[str, Any]:
    if population_name is None:
        return populations[0]

    for population in populations:
        if population.get('name') == population_name:
            return population

    names = ', '.join(str(pop.get('name')) for pop in populations)
    raise SeedingGuiError(f"Population '{population_name}' was not found. Available populations: {names}")


def _unique_population_name(existing_names: list[str], requested_name: str) -> str:
    base_name = requested_name.strip() or 'population'
    if base_name not in existing_names:
        return base_name
    suffix = 2
    while f'{base_name}_{suffix}' in existing_names:
        suffix += 1
    return f'{base_name}_{suffix}'


def _population_points_path(base_path: Path, population_name: str) -> Path:
    safe_name = ''.join(char if char.isalnum() or char in {'-', '_'} else '_' for char in population_name).strip('_')
    if not safe_name:
        safe_name = 'population'
    return base_path.with_name(f'{base_path.stem}.{safe_name}{base_path.suffix}')


def _resolve_relative_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    config_relative = (base_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return candidate.resolve()


# def _yaml_path_value(points_path: Path, output_config_dir: Path) -> str:
#     try:
#         cwd = Path.cwd().resolve()
#         path_value = Path(points_path).resolve().relative_to(cwd)
#         normalized = path_value.as_posix()
#     except ValueError:
#         try:
#             path_value = Path(points_path).resolve().relative_to(output_config_dir.resolve())
#             normalized = path_value.as_posix()
#         except ValueError:
#             normalized = Path(points_path).as_posix()

#     if not normalized.startswith(('.', '/')) and ':' not in normalized:
#         normalized = f'./{normalized}'
#     return normalized

def _yaml_path_value(points_path: Path, output_config_dir: Path) -> str:
    try:
        normalized = Path(points_path).resolve().relative_to(output_config_dir.resolve()).as_posix()
    except ValueError:
        normalized = Path(points_path).as_posix()

    if not normalized.startswith(('.', '/')) and ':' not in normalized:
        normalized = f'./{normalized}'
    return normalized
