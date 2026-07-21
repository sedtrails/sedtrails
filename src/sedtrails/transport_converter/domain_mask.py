"""Domain masking and boundary classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay

from sedtrails.transport_converter.tekal import read_tekal_polygons


@dataclass(frozen=True)
class ConnectivityMaskResult:
    """Filtered connectivity and diagnostics for a polygon mask operation."""

    connectivity: np.ndarray
    active_mask: np.ndarray
    removed_count: int


@dataclass(frozen=True)
class BoundaryEdgeClassification:
    """Boundary edge classes and diagnostics from polygon overrides."""

    edges: np.ndarray
    midpoints: np.ndarray
    classes: np.ndarray
    class_sources: tuple[tuple[str, ...], ...]
    class_counts: dict[str, int]
    polygon_counts: dict[str, int]
    class_files: dict[str, list[str]]

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata-friendly boundary edge classification table.

        Returns
        -------
        dict[str, Any]
            Dictionary containing edge node indices, edge midpoint
            coordinates, edge classes, class source diagnostics, counts, and
            configured polygon files. Arrays are converted to plain Python
            containers so they can be stored in ``SedtrailsMetadata``.
        """
        return {
            'edge_nodes': self.edges.tolist(),
            'edge_midpoints': self.midpoints.tolist(),
            'edge_classes': self.classes.tolist(),
            'edge_class_sources': [list(sources) for sources in self.class_sources],
            'class_counts': dict(self.class_counts),
            'polygon_counts': dict(self.polygon_counts),
            'class_pol_files': {name: list(files) for name, files in self.class_files.items()},
        }


BOUNDARY_CLASS_NAMES = ('open', 'land')


def inner_boundary_files_from_config(domain_config: Mapping[str, Any] | None) -> list[str]:
    """Return configured inner-boundary Tekal files as strings.

    Parameters
    ----------
    domain_config : Mapping[str, Any] or None
        Domain configuration mapping. The ``inner_boundary_pol_files`` entry
        may be missing, ``None``, a single path-like value, or an iterable of
        path-like values.

    Returns
    -------
    list[str]
        Configured inner-boundary polygon file paths. Missing configuration
        yields an empty list.
    """

    if not domain_config:
        return []

    pol_files = domain_config.get('inner_boundary_pol_files', [])
    if pol_files is None:
        return []
    if isinstance(pol_files, (str, Path)):
        return [str(pol_files)]
    return [str(pol_file) for pol_file in pol_files]


def load_inner_boundary_polygons(domain_config: Mapping[str, Any] | None) -> list[np.ndarray]:
    """Load inner-boundary polygons from a domain configuration mapping.

    Parameters
    ----------
    domain_config : Mapping[str, Any] or None
        Domain configuration mapping containing optional
        ``inner_boundary_pol_files``.

    Returns
    -------
    list[np.ndarray]
        Polygon coordinate arrays with shape ``(n_vertices, 2)``.

    Raises
    ------
    FileNotFoundError
        If any configured polygon file does not exist.
    ValueError
        If a configured Tekal polygon block is malformed.
    """

    return read_tekal_polygons(inner_boundary_files_from_config(domain_config))


def boundary_class_files_from_config(domain_config: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """Return configured boundary-class Tekal files by class.

    Parameters
    ----------
    domain_config : Mapping[str, Any] or None
        Domain configuration mapping. The ``boundary_class_pol_files`` entry
        may contain ``open`` and ``land`` lists.

    Returns
    -------
    dict[str, list[str]]
        Mapping with keys ``"open"`` and ``"land"``. Each value is a list of
        configured polygon file paths. Missing classes are returned with empty
        lists.
    """

    class_files = {class_name: [] for class_name in BOUNDARY_CLASS_NAMES}
    if not domain_config:
        return class_files

    configured = domain_config.get('boundary_class_pol_files', {}) or {}
    for class_name in BOUNDARY_CLASS_NAMES:
        pol_files = configured.get(class_name, [])
        if pol_files is None:
            continue
        if isinstance(pol_files, (str, Path)):
            class_files[class_name] = [str(pol_files)]
        else:
            class_files[class_name] = [str(pol_file) for pol_file in pol_files]
    return class_files


def load_boundary_class_polygons(domain_config: Mapping[str, Any] | None) -> dict[str, list[np.ndarray]]:
    """Load boundary-class override polygons by class from domain configuration.

    Parameters
    ----------
    domain_config : Mapping[str, Any] or None
        Domain configuration mapping containing optional
        ``boundary_class_pol_files``.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Mapping from boundary class name to Tekal polygon coordinate arrays.

    Raises
    ------
    FileNotFoundError
        If any configured polygon file does not exist.
    ValueError
        If a configured Tekal polygon block is malformed.
    """

    return {
        class_name: read_tekal_polygons(pol_files)
        for class_name, pol_files in boundary_class_files_from_config(domain_config).items()
    }


def delaunay_connectivity(node_x: np.ndarray, node_y: np.ndarray) -> np.ndarray:
    """Build triangular candidate connectivity from node coordinates.

    Parameters
    ----------
    node_x, node_y : np.ndarray
        One-dimensional or flattenable arrays with node x and y coordinates.

    Returns
    -------
    np.ndarray
        Zero-based triangular connectivity with shape ``(n_triangles, 3)``.
        If fewer than three points are provided, an empty ``(0, 3)`` array is
        returned.

    Raises
    ------
    ValueError
        If ``node_x`` and ``node_y`` do not have matching shapes.
    scipy.spatial.QhullError
        If SciPy cannot construct a Delaunay triangulation for the supplied
        coordinates.
    """

    x = np.asarray(node_x, dtype=float).ravel()
    y = np.asarray(node_y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f'node_x and node_y must have the same shape, got {x.shape} and {y.shape}')
    if x.size < 3:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(Delaunay(np.column_stack((x, y))).simplices, dtype=np.int64)


def filter_connectivity_by_inner_polygons(
    node_x: np.ndarray,
    node_y: np.ndarray,
    connectivity: np.ndarray,
    polygons: Iterable[np.ndarray],
) -> ConnectivityMaskResult:
    """Remove faces or triangles whose centroid falls inside any polygon.

    Parameters
    ----------
    node_x, node_y : np.ndarray
        Node coordinate arrays used by ``connectivity``.
    connectivity : np.ndarray
        Face-node or triangle connectivity. Invalid/padded node indices must
        be negative.
    polygons : Iterable[np.ndarray]
        Polygon coordinate arrays. Polygons with fewer than three vertices are
        ignored.

    Returns
    -------
    ConnectivityMaskResult
        Filtered connectivity, boolean active mask for the input connectivity,
        and the number of removed faces or triangles.
    """

    faces = np.asarray(connectivity, dtype=np.int64)
    if faces.size == 0:
        empty_mask = np.zeros(faces.shape[0], dtype=bool)
        return ConnectivityMaskResult(connectivity=faces, active_mask=empty_mask, removed_count=0)

    polygon_list = [np.asarray(polygon, dtype=float) for polygon in polygons if np.asarray(polygon).shape[0] >= 3]
    if not polygon_list:
        return ConnectivityMaskResult(
            connectivity=faces.copy(),
            active_mask=np.ones(faces.shape[0], dtype=bool),
            removed_count=0,
        )

    centroids = face_centroids(node_x, node_y, faces)
    inside = points_inside_any_polygon(centroids, polygon_list)
    active_mask = ~inside
    return ConnectivityMaskResult(
        connectivity=faces[active_mask],
        active_mask=active_mask,
        removed_count=int(np.count_nonzero(inside)),
    )


def triangulate_face_connectivity(connectivity: np.ndarray) -> np.ndarray:
    """Convert padded polygon face-node connectivity to triangles by fan split.

    Parameters
    ----------
    connectivity : np.ndarray
        Face-node connectivity with shape ``(n_faces, max_nodes_per_face)``.
        Negative entries are treated as invalid padding.

    Returns
    -------
    np.ndarray
        Triangular connectivity with shape ``(n_triangles, 3)``. Original node
        indices are preserved and nodes are not reindexed.
    """

    faces = np.asarray(connectivity, dtype=np.int64)
    triangles: list[list[int]] = []
    for face in faces:
        valid = face[face >= 0]
        if valid.size < 3:
            continue
        if valid.size == 3:
            triangles.append([int(valid[0]), int(valid[1]), int(valid[2])])
            continue
        for index in range(1, valid.size - 1):
            triangles.append([int(valid[0]), int(valid[index]), int(valid[index + 1])])

    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def extract_boundary_edges(connectivity: np.ndarray) -> np.ndarray:
    """Return edges used by exactly one face or triangle.

    Parameters
    ----------
    connectivity : np.ndarray
        Face-node or triangle connectivity. Negative entries are treated as
        invalid padding.

    Returns
    -------
    np.ndarray
        Boundary edge node indices with shape ``(n_edges, 2)``. Edge
        orientation follows the first face in which the edge appears.
    """

    faces = np.asarray(connectivity, dtype=np.int64)
    oriented_edges = _oriented_edges_from_uniform_connectivity(faces)
    if oriented_edges is None:
        oriented_edges = _oriented_edges_from_ragged_connectivity(faces)

    if oriented_edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    sorted_edges = np.sort(oriented_edges, axis=1)
    _, first_indices, counts = np.unique(
        sorted_edges,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_first_indices = np.sort(first_indices[counts == 1])
    if boundary_first_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return oriented_edges[boundary_first_indices]


def _oriented_edges_from_uniform_connectivity(faces: np.ndarray) -> np.ndarray | None:
    """Return oriented edges for fully populated face connectivity."""
    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 2:
        return np.empty((0, 2), dtype=np.int64)
    if np.any(faces < 0):
        return None

    starts = faces
    ends = np.roll(faces, shift=-1, axis=1)
    edges = np.stack((starts, ends), axis=2).reshape(-1, 2)
    return edges[edges[:, 0] != edges[:, 1]]


def _oriented_edges_from_ragged_connectivity(faces: np.ndarray) -> np.ndarray:
    """Return oriented edges for padded variable-width face connectivity."""
    edge_blocks = []
    for face in faces:
        valid = face[face >= 0]
        if valid.size < 2:
            continue
        starts = valid
        ends = np.roll(valid, shift=-1)
        edges = np.column_stack((starts, ends))
        edge_blocks.append(edges[edges[:, 0] != edges[:, 1]])

    if not edge_blocks:
        return np.empty((0, 2), dtype=np.int64)
    return np.vstack(edge_blocks).astype(np.int64, copy=False)


def classify_boundary_edges_from_config(
    node_x: np.ndarray,
    node_y: np.ndarray,
    connectivity: np.ndarray,
    domain_config: Mapping[str, Any] | None,
) -> BoundaryEdgeClassification | None:
    """Classify active boundary edges using configured polygon overrides.

    Parameters
    ----------
    node_x, node_y : np.ndarray
        Coordinate arrays used by ``connectivity``.
    connectivity : np.ndarray
        Active face-node or triangle connectivity.
    domain_config : Mapping[str, Any] or None
        Domain configuration mapping containing optional
        ``boundary_class_pol_files`` entries for ``open`` and ``land``.

    Returns
    -------
    BoundaryEdgeClassification or None
        Classification table for active boundary edges. ``None`` is returned
        when no boundary class polygon files are configured.

    Raises
    ------
    FileNotFoundError
        If any configured boundary-class polygon file does not exist.
    ValueError
        If a configured Tekal polygon block is malformed.
    """

    class_files = boundary_class_files_from_config(domain_config)
    if not any(class_files.values()):
        return None
    class_polygons = load_boundary_class_polygons(domain_config)
    return classify_boundary_edges(node_x, node_y, connectivity, class_polygons, class_files=class_files)


def classify_boundary_edges(
    node_x: np.ndarray,
    node_y: np.ndarray,
    connectivity: np.ndarray,
    class_polygons: Mapping[str, Iterable[np.ndarray]],
    class_files: Mapping[str, list[str]] | None = None,
) -> BoundaryEdgeClassification:
    """Classify active boundary edges by polygon-contained edge midpoints.

    Parameters
    ----------
    node_x, node_y : np.ndarray
        Coordinate arrays used by ``connectivity``.
    connectivity : np.ndarray
        Active face-node or triangle connectivity.
    class_polygons : Mapping[str, Iterable[np.ndarray]]
        Mapping from boundary class name to polygon coordinate arrays.
        Supported classes are ``"open"`` and ``"land"``.
    class_files : Mapping[str, list[str]], optional
        Configured polygon file paths by class, carried through for
        diagnostics.

    Returns
    -------
    BoundaryEdgeClassification
        Boundary edge node indices, edge midpoints, assigned class labels,
        class source matches, and diagnostic counts.

    Notes
    -----
    Edge classes are assigned from midpoint containment. If both ``open`` and
    ``land`` polygons select the same edge, ``land`` wins while both sources
    remain recorded in ``class_sources``.
    """

    x = np.asarray(node_x, dtype=float).ravel()
    y = np.asarray(node_y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f'node_x and node_y must have the same shape, got {x.shape} and {y.shape}')
    edges = extract_boundary_edges(connectivity)
    midpoints = np.full((edges.shape[0], 2), np.nan, dtype=float)
    if edges.size:
        valid = (edges >= 0) & (edges < x.size) & (edges < y.size)
        valid_edges = valid.all(axis=1)
        midpoints[valid_edges, 0] = np.mean(x[edges[valid_edges]], axis=1)
        midpoints[valid_edges, 1] = np.mean(y[edges[valid_edges]], axis=1)

    matches_by_class = {
        class_name: points_inside_any_polygon(midpoints, polygons)
        for class_name, polygons in class_polygons.items()
        if class_name in BOUNDARY_CLASS_NAMES
    }

    open_matches = matches_by_class.get('open', np.zeros(edges.shape[0], dtype=bool))
    land_matches = matches_by_class.get('land', np.zeros(edges.shape[0], dtype=bool))
    class_array = np.full(edges.shape[0], 'unclassified', dtype=object)
    class_array[open_matches] = 'open'
    class_array[land_matches] = 'land'
    class_sources = tuple(
        tuple(
            class_name
            for class_name, matches in (('open', open_matches), ('land', land_matches))
            if matches[edge_index]
        )
        for edge_index in range(edges.shape[0])
    )
    class_counts = {class_name: int(np.count_nonzero(class_array == class_name)) for class_name in BOUNDARY_CLASS_NAMES}
    class_counts['unclassified'] = int(np.count_nonzero(class_array == 'unclassified'))
    class_counts['ambiguous'] = int(np.count_nonzero(class_array == 'ambiguous'))
    polygon_counts = {
        class_name: len([polygon for polygon in polygons if np.asarray(polygon).shape[0] >= 3])
        for class_name, polygons in class_polygons.items()
        if class_name in BOUNDARY_CLASS_NAMES
    }
    for class_name in BOUNDARY_CLASS_NAMES:
        polygon_counts.setdefault(class_name, 0)

    return BoundaryEdgeClassification(
        edges=edges,
        midpoints=midpoints,
        classes=class_array,
        class_sources=tuple(class_sources),
        class_counts=class_counts,
        polygon_counts=polygon_counts,
        class_files={class_name: list(files) for class_name, files in (class_files or {}).items()},
    )


def face_centroids(node_x: np.ndarray, node_y: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Compute centroids for padded face-node connectivity.

    Parameters
    ----------
    node_x, node_y : np.ndarray
        Node coordinate arrays used by ``connectivity``.
    connectivity : np.ndarray
        Face-node connectivity. Negative and out-of-range entries are ignored.

    Returns
    -------
    np.ndarray
        Centroid coordinates with shape ``(n_faces, 2)``. Faces without any
        valid nodes receive ``NaN`` centroid coordinates.
    """

    x = np.asarray(node_x, dtype=float).ravel()
    y = np.asarray(node_y, dtype=float).ravel()
    faces = np.asarray(connectivity, dtype=np.int64)
    centroids = np.full((faces.shape[0], 2), np.nan, dtype=float)

    if faces.size == 0 or x.size == 0 or y.size == 0:
        return centroids

    valid = (faces >= 0) & (faces < x.size) & (faces < y.size)
    counts = np.count_nonzero(valid, axis=1)
    has_valid_nodes = counts > 0
    if not np.any(has_valid_nodes):
        return centroids

    clipped = np.clip(faces, 0, min(x.size, y.size) - 1)
    face_x = np.where(valid, x[clipped], 0.0)
    face_y = np.where(valid, y[clipped], 0.0)
    centroids[has_valid_nodes, 0] = np.sum(face_x[has_valid_nodes], axis=1) / counts[has_valid_nodes]
    centroids[has_valid_nodes, 1] = np.sum(face_y[has_valid_nodes], axis=1) / counts[has_valid_nodes]

    return centroids


def points_inside_any_polygon(points: np.ndarray, polygons: Iterable[np.ndarray]) -> np.ndarray:
    """Return whether points fall inside at least one polygon.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates with shape ``(n_points, 2)``.
    polygons : Iterable[np.ndarray]
        Polygon coordinate arrays. Polygons with fewer than three vertices are
        ignored.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(n_points,)``. Non-finite points are always
        marked ``False``.
    """

    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] < 2:
        raise ValueError(f'points must have shape (n_points, 2), got {points_array.shape}')
    points_array = points_array[:, :2]
    inside = np.zeros(points_array.shape[0], dtype=bool)
    finite = np.isfinite(points_array).all(axis=1)
    if not np.any(finite):
        return inside

    finite_points = points_array[finite]
    finite_inside = np.zeros(finite_points.shape[0], dtype=bool)
    for polygon in polygons:
        polygon_array = np.asarray(polygon, dtype=float)
        if polygon_array.shape[0] < 3:
            continue
        polygon_xy = polygon_array[:, :2]
        min_x, min_y = np.min(polygon_xy, axis=0)
        max_x, max_y = np.max(polygon_xy, axis=0)
        candidate = (
            (finite_points[:, 0] >= min_x)
            & (finite_points[:, 0] <= max_x)
            & (finite_points[:, 1] >= min_y)
            & (finite_points[:, 1] <= max_y)
        )
        if not np.any(candidate):
            continue
        finite_inside[candidate] |= MplPath(polygon_xy).contains_points(finite_points[candidate])

    inside[finite] = finite_inside
    return inside
