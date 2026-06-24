"""
Particle position calculation on an unstructured triangular grid.

The public factory name is kept for compatibility with existing callers, but
the implementation now caches SciPy Delaunay geometry and uses its simplex
search instead of scanning every triangle for every interpolation point when
connectivity is not supplied explicitly.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit, prange
from scipy.spatial import ConvexHull, Delaunay


TRIANGLE_TOLERANCE = 1e-10
MAX_SIMPLEX_WALK_STEPS = 128
BOUNDARY_CLASS_UNCLASSIFIED = np.int8(0)
BOUNDARY_CLASS_OPEN = np.int8(1)
BOUNDARY_CLASS_LAND = np.int8(2)


@dataclass
class GridGeometry:
    """Cached spatial geometry shared by particle populations on the same grid."""

    grid_x: np.ndarray
    grid_y: np.ndarray
    triangulation: Any
    triangles: np.ndarray
    outer_envelope: np.ndarray
    triangle_neighbors: np.ndarray
    p0_x: np.ndarray
    p0_y: np.ndarray
    inv00: np.ndarray
    inv01: np.ndarray
    inv10: np.ndarray
    inv11: np.ndarray
    triangle_finder: Any = None
    boundary_edges: np.ndarray | None = None
    boundary_edge_class_codes: np.ndarray | None = None
    triangle_edge_class_codes: np.ndarray | None = None

    @classmethod
    def from_points(cls, grid_x, grid_y, triangles=None, boundary_edge_classification=None):
        """
        Build cached grid geometry from point coordinates.

        Parameters
        ----------
        grid_x : object
            Grid node x coordinates.
        grid_y : object
            Grid node y coordinates.
        triangles : object
            Triangle connectivity array.

        Returns
        -------
        GridGeometry
            Cached grid geometry built from the supplied coordinates.
        """
        x = np.asarray(grid_x, dtype=np.float64).ravel()
        y = np.asarray(grid_y, dtype=np.float64).ravel()

        if x.shape != y.shape:
            raise ValueError(f'grid_x and grid_y must have the same shape, got {x.shape} and {y.shape}')

        points = np.column_stack((x, y))
        finite_points = points[np.isfinite(points).all(axis=1)]
        if finite_points.shape[0] < 3:
            raise ValueError('At least three finite grid points are required for particle interpolation')
        if finite_points.shape[0] != points.shape[0]:
            raise ValueError('grid_x and grid_y must contain only finite coordinates')

        triangle_finder = None
        if triangles is None:
            # Delaunay owns the simplex search structure and affine transforms.
            triangulation = Delaunay(points)
            triangle_array = np.asarray(triangulation.simplices, dtype=np.int64)
            triangle_neighbors = np.asarray(triangulation.neighbors, dtype=np.int64)
        else:
            import matplotlib.tri as mtri

            triangle_array = np.asarray(triangles, dtype=np.int64)
            if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
                raise ValueError('triangles must be an array with shape (n_triangles, 3)')
            triangulation = None
            if triangle_array.shape[0] == 0:
                triangle_finder = None
            else:
                triangle_finder = mtri.Triangulation(x, y, triangle_array).get_trifinder()
            triangle_neighbors = _compute_triangle_neighbors(triangle_array)

        unique_points = np.unique(finite_points, axis=0)
        if unique_points.shape[0] >= 3:
            try:
                outer_envelope = unique_points[ConvexHull(unique_points).vertices]
            except Exception:
                outer_envelope = _bounding_box(unique_points)
        else:
            outer_envelope = _bounding_box(unique_points)

        p0_x, p0_y, inv00, inv01, inv10, inv11 = _triangle_inverse_matrices(x, y, triangle_array)

        boundary_edges, boundary_edge_class_codes = _parse_boundary_edge_classification(boundary_edge_classification)
        boundary_edges, boundary_edge_class_codes = _prepare_boundary_edge_geometry(
            x, y, boundary_edges, boundary_edge_class_codes
        )
        triangle_edge_class_codes = _build_triangle_edge_class_codes(
            triangle_array,
            boundary_edges,
            boundary_edge_class_codes,
        )

        return cls(
            grid_x=x,
            grid_y=y,
            triangulation=triangulation,
            triangles=triangle_array,
            outer_envelope=outer_envelope,
            triangle_neighbors=triangle_neighbors,
            p0_x=p0_x,
            p0_y=p0_y,
            inv00=inv00,
            inv01=inv01,
            inv10=inv10,
            inv11=inv11,
            triangle_finder=triangle_finder,
            boundary_edges=boundary_edges,
            boundary_edge_class_codes=boundary_edge_class_codes,
            triangle_edge_class_codes=triangle_edge_class_codes,
        )

    def find_triangle(self, x, y) -> int:
        """
        Return the containing simplex index, or -1 outside the triangulation.

        Parameters
        ----------
        x : object
            X coordinate value or array.
        y : object
            Y coordinate value or array.

        Returns
        -------
        int
            Integer result of the calculation.
        """
        if self.triangles.shape[0] == 0:
            return -1
        if self.triangulation is not None:
            simplex = self.triangulation.find_simplex(np.array([[x, y]], dtype=np.float64), tol=TRIANGLE_TOLERANCE)
            return int(simplex[0])
        return int(self.triangle_finder([x], [y])[0])

    def locate_points(self, x_points, y_points, start_simplices=None):
        """
        Locate points, using cached simplices first and global search only for misses.

        Parameters
        ----------
        x_points : object
            Point x coordinates to sample.
        y_points : object
            Point y coordinates to sample.
        start_simplices : object
            Initial simplex ids used to seed local point searches.

        Returns
        -------
        np.ndarray
            Simplex index for each input point, or -1 outside the triangulation.
        """
        points = _points_array(x_points, y_points)
        if points.size == 0:
            return np.empty(0, dtype=np.int64)
        if self.triangles.shape[0] == 0:
            return np.full(points.shape[0], -1, dtype=np.int64)

        if start_simplices is None:
            simplices = np.full(points.shape[0], -1, dtype=np.int64)
        else:
            simplices = np.asarray(start_simplices, dtype=np.int64).copy()
            if simplices.shape != (points.shape[0],):
                raise ValueError(
                    f'start_simplices must have shape {(points.shape[0],)}, got {simplices.shape}'
                )
            simplices = _locate_points_walk_numba(
                points[:, 0],
                points[:, 1],
                simplices,
                self.triangle_neighbors,
                self.p0_x,
                self.p0_y,
                self.inv00,
                self.inv01,
                self.inv10,
                self.inv11,
                MAX_SIMPLEX_WALK_STEPS,
                TRIANGLE_TOLERANCE,
            )

        missing = simplices < 0
        if not np.any(missing):
            return simplices

        if self.triangulation is not None:
            simplices[missing] = self.triangulation.find_simplex(points[missing], tol=TRIANGLE_TOLERANCE)
        else:
            simplices[missing] = np.asarray(
                self.triangle_finder(points[missing, 0], points[missing, 1]),
                dtype=np.int64,
            )

        return simplices

    def barycentric_weights(self, x_points, y_points):
        """
        Return simplex indices and barycentric weights for points.

        Parameters
        ----------
        x_points : object
            Point x coordinates to sample.
        y_points : object
            Point y coordinates to sample.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Simplex indices and barycentric weights for each point.
        """
        points = _points_array(x_points, y_points)
        if self.triangles.shape[0] == 0:
            return np.full(points.shape[0], -1, dtype=np.int64), np.zeros((points.shape[0], 3), dtype=np.float64)

        weights = np.zeros((points.shape[0], 3), dtype=np.float64)

        if self.triangulation is not None:
            simplices = self.triangulation.find_simplex(points, tol=TRIANGLE_TOLERANCE)
            valid = simplices >= 0
            if not np.any(valid):
                return simplices, weights

            transform = self.triangulation.transform[simplices[valid]]
            delta = points[valid] - transform[:, 2, :]
            first_two = np.einsum('ijk,ik->ij', transform[:, :2, :], delta)
            weights[valid, :2] = first_two
            weights[valid, 2] = 1.0 - first_two.sum(axis=1)
            return simplices, weights

        simplices = np.asarray(self.triangle_finder(points[:, 0], points[:, 1]), dtype=np.int64)
        valid_indices = np.flatnonzero(simplices >= 0)
        if valid_indices.size:
            vertices = self.triangles[simplices[valid_indices]]
            local_weights, nondegenerate = _triangle_barycentric_weights(
                self.grid_x,
                self.grid_y,
                vertices,
                points[valid_indices],
            )
            weights[valid_indices[nondegenerate]] = local_weights[nondegenerate]
            simplices[valid_indices[~nondegenerate]] = -1

        return simplices, weights

    def barycentric_weights_with_simplex(self, x_points, y_points, simplex_ids=None):
        """
        Return simplex indices and barycentric weights using cached simplex ids.

        Parameters
        ----------
        x_points : object
            Point x coordinates to sample.
        y_points : object
            Point y coordinates to sample.
        simplex_ids : object
            Cached simplex ids for the point coordinates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Simplex indices and barycentric weights for each point.
        """
        if simplex_ids is None:
            return self.barycentric_weights(x_points, y_points)

        points = _points_array(x_points, y_points)
        simplices = self.locate_points(x_points, y_points, simplex_ids)
        weights = np.zeros((points.shape[0], 3), dtype=np.float64)

        valid_indices = np.flatnonzero(simplices >= 0)
        if valid_indices.size:
            valid_simplices = simplices[valid_indices]
            dx = points[valid_indices, 0] - self.p0_x[valid_simplices]
            dy = points[valid_indices, 1] - self.p0_y[valid_simplices]
            w1 = self.inv00[valid_simplices] * dx + self.inv01[valid_simplices] * dy
            w2 = self.inv10[valid_simplices] * dx + self.inv11[valid_simplices] * dy
            nondegenerate = (
                (self.inv00[valid_simplices] != 0.0)
                | (self.inv01[valid_simplices] != 0.0)
                | (self.inv10[valid_simplices] != 0.0)
                | (self.inv11[valid_simplices] != 0.0)
            )
            local_weights = np.column_stack((1.0 - w1 - w2, w1, w2))
            weights[valid_indices[nondegenerate]] = local_weights[nondegenerate]
            simplices[valid_indices[~nondegenerate]] = -1

        return simplices, weights

    def interpolate_field(self, field, x_points, y_points):
        """
        Barycentrically interpolate a nodal field at point coordinates.

        Parameters
        ----------
        field : object
            Scalar field values defined on grid nodes.
        x_points : object
            Point x coordinates to sample.
        y_points : object
            Point y coordinates to sample.

        Returns
        -------
        np.ndarray
            Interpolated scalar field values at the requested points.
        """
        return self.interpolate_fields((field,), x_points, y_points)[0]

    def interpolate_fields(self, fields, x_points, y_points):
        """
        Interpolate multiple nodal fields using one point-location pass.

        Parameters
        ----------
        fields : object
            Scalar field arrays defined on grid nodes.
        x_points : object
            Point x coordinates to sample.
        y_points : object
            Point y coordinates to sample.

        Returns
        -------
        tuple[np.ndarray, ...]
            Interpolated field values for each supplied field.
        """
        fields = tuple(fields)
        simplices, weights = self.barycentric_weights(x_points, y_points)
        outputs = [np.full(len(simplices), np.nan, dtype=np.float64) for _ in fields]

        valid = simplices >= 0
        if np.any(valid):
            vertices = self.triangles[simplices[valid]]
            valid_weights = weights[valid]
            for output, field in zip(outputs, fields, strict=True):
                values = np.asarray(field).ravel()
                output[valid] = np.einsum('ij,ij->i', values[vertices], valid_weights)

        return tuple(outputs)

    def interpolate_fields_with_simplex(self, fields, x_points, y_points, simplex_ids=None):
        """
        Interpolate nodal fields and return refreshed simplex ids.

        Parameters
        ----------
        fields : object
            Scalar field arrays defined on grid nodes.
        x_points : object
            Point x coordinates to sample.
        y_points : object
            Point y coordinates to sample.
        simplex_ids : object
            Cached simplex ids for the point coordinates.

        Returns
        -------
        tuple[tuple[np.ndarray, ...], np.ndarray]
            Interpolated field values and the containing simplex index for each point.
        """
        fields = tuple(fields)
        if not fields:
            simplices = self.locate_points(x_points, y_points, simplex_ids)
            return (), simplices

        x_values = np.asarray(x_points, dtype=np.float64)
        y_values = np.asarray(y_points, dtype=np.float64)
        if y_values.shape != x_values.shape:
            raise ValueError(
                f'x_points and y_points must have the same shape, got {x_values.shape} and {y_values.shape}'
            )

        field_values = []
        n_nodes = self.grid_x.size
        for field in fields:
            values = np.asarray(field, dtype=np.float64).ravel()
            if values.size != n_nodes:
                raise ValueError(f'field arrays must have {n_nodes} values, got {values.size}')
            field_values.append(values)

        simplices = self.locate_points(x_values, y_values, simplex_ids)
        stacked_fields = np.vstack(field_values)
        output_values, refreshed_simplices = _interpolate_fields_at_simplices_numba(
            stacked_fields,
            simplices,
            x_values.ravel(),
            y_values.ravel(),
            self.triangles,
            self.p0_x,
            self.p0_y,
            self.inv00,
            self.inv01,
            self.inv10,
            self.inv11,
            TRIANGLE_TOLERANCE,
        )
        outputs = tuple(output_values[i].copy() for i in range(output_values.shape[0]))
        return outputs, refreshed_simplices

    def update_particles(self, x0, y0, grid_u, grid_v, dt, igeo=0):
        """
        Advance particle positions one RK4 step using barycentric velocity sampling.

        Parameters
        ----------
        x0 : object
            Initial particle x coordinates.
        y0 : object
            Initial particle y coordinates.
        grid_u : object
            Grid-aligned x velocity component.
        grid_v : object
            Grid-aligned y velocity component.
        dt : object
            Integration timestep in seconds.
        igeo : object
            Coordinate-system flag used by the legacy kernels.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated x and y particle positions.
        """
        return self.update_particles_temporal(x0, y0, grid_u, grid_v, grid_u, grid_v, 0.0, dt, igeo)

    def update_particles_temporal(self, x0, y0, lower_u, lower_v, upper_u, upper_v, weight, dt, igeo=0):
        """
        Advance particles using lower/upper time-slice velocities blended by weight.

        Parameters
        ----------
        x0 : object
            Initial particle x coordinates.
        y0 : object
            Initial particle y coordinates.
        lower_u : object
            Lower-time x velocity component.
        lower_v : object
            Lower-time y velocity component.
        upper_u : object
            Upper-time x velocity component.
        upper_v : object
            Upper-time y velocity component.
        weight : object
            Temporal interpolation weight between lower and upper fields.
        dt : object
            Integration timestep in seconds.
        igeo : object
            Coordinate-system flag used by the legacy kernels.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated x and y particle positions.
        """
        x_new, y_new, _, _ = self.update_particles_temporal_with_boundary_class(
            x0,
            y0,
            lower_u,
            lower_v,
            upper_u,
            upper_v,
            weight,
            dt,
            simplex_ids=None,
            igeo=igeo,
        )
        return x_new, y_new

    def update_particles_with_simplex(self, x0, y0, grid_u, grid_v, dt, simplex_ids=None, igeo=0):
        """
        Advance particles and return updated simplex ids for the new positions.

        Parameters
        ----------
        x0 : object
            Initial particle x coordinates.
        y0 : object
            Initial particle y coordinates.
        grid_u : object
            Grid-aligned x velocity component.
        grid_v : object
            Grid-aligned y velocity component.
        dt : object
            Integration timestep in seconds.
        simplex_ids : object
            Cached simplex ids for particle positions.
        igeo : object
            Coordinate-system flag used by the legacy kernels.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Updated x positions, y positions, and simplex ids.
        """
        x_new, y_new, new_simplices, _ = self.update_particles_with_boundary_class(
            x0,
            y0,
            grid_u,
            grid_v,
            dt,
            simplex_ids=simplex_ids,
            igeo=igeo,
        )
        return x_new, y_new, new_simplices

    def update_particles_with_boundary_class(self, x0, y0, grid_u, grid_v, dt, simplex_ids=None, igeo=0):
        """
        Advance particles and return updated simplex ids plus exit boundary classes.

        Parameters are the same as ``update_particles_with_simplex``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Updated x positions, y positions, simplex ids, and boundary class
            codes for particles that leave the triangulation.
        """
        return self.update_particles_temporal_with_boundary_class(
            x0,
            y0,
            grid_u,
            grid_v,
            grid_u,
            grid_v,
            0.0,
            dt,
            simplex_ids=simplex_ids,
            igeo=igeo,
        )

    def update_particles_temporal_with_simplex(
        self,
        x0,
        y0,
        lower_u,
        lower_v,
        upper_u,
        upper_v,
        weight,
        dt,
        simplex_ids=None,
        igeo=0,
    ):
        """
        Advance particles with a Numba RK4 kernel and cached simplex ids.

        Parameters
        ----------
        x0 : object
            Initial particle x coordinates.
        y0 : object
            Initial particle y coordinates.
        lower_u : object
            Lower-time x velocity component.
        lower_v : object
            Lower-time y velocity component.
        upper_u : object
            Upper-time x velocity component.
        upper_v : object
            Upper-time y velocity component.
        weight : object
            Temporal interpolation weight between lower and upper fields.
        dt : object
            Integration timestep in seconds.
        simplex_ids : object
            Cached simplex ids for particle positions.
        igeo : object
            Coordinate-system flag used by the legacy kernels.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Updated x positions, y positions, and simplex ids.
        """
        x_new, y_new, new_simplices, _ = self.update_particles_temporal_with_boundary_class(
            x0,
            y0,
            lower_u,
            lower_v,
            upper_u,
            upper_v,
            weight,
            dt,
            simplex_ids=simplex_ids,
            igeo=igeo,
        )
        return x_new, y_new, new_simplices

    def update_particles_temporal_with_boundary_class(
        self,
        x0,
        y0,
        lower_u,
        lower_v,
        upper_u,
        upper_v,
        weight,
        dt,
        simplex_ids=None,
        igeo=0,
    ):
        """
        Advance particles and return exit boundary class codes.

        Parameters are the same as ``update_particles_temporal_with_simplex``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Updated x positions, y positions, simplex ids, and exit boundary
            class codes.
        """
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64)
        particle_shape = x0.shape
        if x0.size == 0:
            return (
                x0.copy(),
                y0.copy(),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int8),
            )

        lower_u_adj, lower_v_adj = self._velocity_arrays(lower_u, lower_v, igeo)
        if weight <= 0.0:
            upper_u_adj = lower_u_adj
            upper_v_adj = lower_v_adj
        else:
            upper_u_adj, upper_v_adj = self._velocity_arrays(upper_u, upper_v, igeo)

        starts = self.locate_points(x0, y0, simplex_ids)
        x_new, y_new, new_simplices, boundary_class_codes = _update_particles_temporal_numba(
            x0.ravel(),
            y0.ravel(),
            np.asarray(lower_u_adj).ravel(),
            np.asarray(lower_v_adj).ravel(),
            np.asarray(upper_u_adj).ravel(),
            np.asarray(upper_v_adj).ravel(),
            float(weight),
            float(dt),
            starts,
            self.triangles,
            self.triangle_neighbors,
            self.p0_x,
            self.p0_y,
            self.inv00,
            self.inv01,
            self.inv10,
            self.inv11,
            self.triangle_edge_class_codes,
            MAX_SIMPLEX_WALK_STEPS,
            TRIANGLE_TOLERANCE,
        )
        return (
            x_new.reshape(particle_shape),
            y_new.reshape(particle_shape),
            new_simplices,
            boundary_class_codes.reshape(particle_shape),
        )

    def interpolate_vector(self, grid_u, grid_v, x_points, y_points):
        """
        Interpolate vector components at point coordinates.

        Parameters
        ----------
        grid_u : object
            Grid-aligned x velocity component.
        grid_v : object
            Grid-aligned y velocity component.
        x_points : object
            Point x coordinates to sample.
        y_points : object
            Point y coordinates to sample.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Interpolated vector components at the requested points.
        """
        return (
            self.interpolate_field(grid_u, x_points, y_points),
            self.interpolate_field(grid_v, x_points, y_points),
        )

    def interpolate_temporal_vector(self, lower_u, lower_v, upper_u, upper_v, weight, x_points, y_points):
        """
        Interpolate lower/upper vector fields spatially, then blend in time.

        Parameters
        ----------
        lower_u : object
            Lower-time x velocity component.
        lower_v : object
            Lower-time y velocity component.
        upper_u : object
            Upper-time x velocity component.
        upper_v : object
            Upper-time y velocity component.
        weight : object
            Temporal interpolation weight between lower and upper fields.
        x_points : object
            Point x coordinates to sample.
        y_points : object
            Point y coordinates to sample.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Temporally blended vector components at the requested points.
        """
        if weight <= 0.0:
            return self.interpolate_fields((lower_u, lower_v), x_points, y_points)

        lower_u_values, lower_v_values, upper_u_values, upper_v_values = self.interpolate_fields(
            (lower_u, lower_v, upper_u, upper_v),
            x_points,
            y_points,
        )
        return (
            lower_u_values + weight * (upper_u_values - lower_u_values),
            lower_v_values + weight * (upper_v_values - lower_v_values),
        )

    def _velocity_arrays(self, grid_u, grid_v, igeo):
        grid_u = np.asarray(grid_u).ravel()
        grid_v = np.asarray(grid_v).ravel()

        if igeo != 1:
            return grid_u, grid_v

        geofac = 6378137.0
        cos_lat = np.cos(np.deg2rad(self.grid_y))
        return grid_u.astype(np.float64, copy=False) / (geofac * cos_lat), grid_v.astype(np.float64, copy=False) / geofac

    def classify_boundary_crossings(self, x0, y0, x1, y1) -> np.ndarray:
        """Return nearest boundary-edge class for particle movement segments.

        Parameters
        ----------
        x0, y0 : array-like
            Starting particle coordinates.
        x1, y1 : array-like
            Ending particle coordinates. Shapes must match ``x0`` and ``y0``.

        Returns
        -------
        np.ndarray
            Boundary class labels with the same shape as ``x0``. Values are
            drawn from the boundary-edge classification attached to this grid
            geometry, or ``"unclassified"`` when no compatible boundary table
            is available.

        Notes
        -----
        The method selects the classified boundary edge with the smallest
        segment-to-segment distance to each particle movement segment. Exact
        segment intersections have zero distance.
        """
        start_points = _points_array(x0, y0)
        end_points = _points_array(x1, y1)
        if start_points.shape != end_points.shape:
            raise ValueError(
                f'start and end coordinates must have the same flattened shape, '
                f'got {start_points.shape} and {end_points.shape}'
            )
        classes = np.full(start_points.shape[0], 'unclassified', dtype=object)

        if (
            self.boundary_edges is None
            or self.boundary_edge_class_codes is None
            or self.boundary_edges.size == 0
            or self.boundary_edge_class_codes.size == 0
        ):
            return classes.reshape(np.asarray(x0).shape)

        edge_start_x = np.asarray(self.grid_x[self.boundary_edges[:, 0]], dtype=np.float64)
        edge_start_y = np.asarray(self.grid_y[self.boundary_edges[:, 0]], dtype=np.float64)
        edge_end_x = np.asarray(self.grid_x[self.boundary_edges[:, 1]], dtype=np.float64)
        edge_end_y = np.asarray(self.grid_y[self.boundary_edges[:, 1]], dtype=np.float64)
        edge_indices = _nearest_boundary_edge_indices_numba(
            start_points[:, 0],
            start_points[:, 1],
            end_points[:, 0],
            end_points[:, 1],
            edge_start_x,
            edge_start_y,
            edge_end_x,
            edge_end_y,
            TRIANGLE_TOLERANCE,
        )
        valid = edge_indices >= 0
        classes[valid] = _boundary_class_labels(self.boundary_edge_class_codes[edge_indices[valid]])

        return classes.reshape(np.asarray(x0).shape)


def _bounding_box(points):
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    min_x = float(np.min(points[:, 0]))
    max_x = float(np.max(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    return np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]], dtype=np.float64)


def _points_array(x_points, y_points):
    x = np.asarray(x_points, dtype=np.float64)
    y = np.asarray(y_points, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f'x_points and y_points must have the same shape, got {x.shape} and {y.shape}')
    return np.column_stack((x.ravel(), y.ravel()))


def _triangle_barycentric_weights(grid_x, grid_y, vertices, points):
    x0 = grid_x[vertices[:, 0]]
    y0 = grid_y[vertices[:, 0]]
    x1 = grid_x[vertices[:, 1]]
    y1 = grid_y[vertices[:, 1]]
    x2 = grid_x[vertices[:, 2]]
    y2 = grid_y[vertices[:, 2]]

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegenerate = np.abs(denom) >= TRIANGLE_TOLERANCE

    weights = np.zeros((points.shape[0], 3), dtype=np.float64)
    if np.any(nondegenerate):
        x = points[nondegenerate, 0]
        y = points[nondegenerate, 1]
        denom_valid = denom[nondegenerate]
        weights[nondegenerate, 0] = (
            (y1[nondegenerate] - y2[nondegenerate]) * (x - x2[nondegenerate])
            + (x2[nondegenerate] - x1[nondegenerate]) * (y - y2[nondegenerate])
        ) / denom_valid
        weights[nondegenerate, 1] = (
            (y2[nondegenerate] - y0[nondegenerate]) * (x - x2[nondegenerate])
            + (x0[nondegenerate] - x2[nondegenerate]) * (y - y2[nondegenerate])
        ) / denom_valid
        weights[nondegenerate, 2] = 1.0 - weights[nondegenerate, 0] - weights[nondegenerate, 1]

    return weights, nondegenerate


def _triangle_inverse_matrices(grid_x, grid_y, triangles):
    x0 = grid_x[triangles[:, 0]]
    y0 = grid_y[triangles[:, 0]]
    x1 = grid_x[triangles[:, 1]]
    y1 = grid_y[triangles[:, 1]]
    x2 = grid_x[triangles[:, 2]]
    y2 = grid_y[triangles[:, 2]]

    m00 = x1 - x0
    m01 = x2 - x0
    m10 = y1 - y0
    m11 = y2 - y0
    det = m00 * m11 - m01 * m10

    inv00 = np.zeros_like(det, dtype=np.float64)
    inv01 = np.zeros_like(det, dtype=np.float64)
    inv10 = np.zeros_like(det, dtype=np.float64)
    inv11 = np.zeros_like(det, dtype=np.float64)
    valid = np.abs(det) >= TRIANGLE_TOLERANCE
    inv00[valid] = m11[valid] / det[valid]
    inv01[valid] = -m01[valid] / det[valid]
    inv10[valid] = -m10[valid] / det[valid]
    inv11[valid] = m00[valid] / det[valid]

    return x0, y0, inv00, inv01, inv10, inv11


def _compute_triangle_neighbors(triangles):
    neighbors = np.full((triangles.shape[0], 3), -1, dtype=np.int64)
    edge_owner = {}
    opposite_edges = ((1, 2), (0, 2), (0, 1))
    for tri_index, triangle in enumerate(triangles):
        for vertex_index, edge_vertices in enumerate(opposite_edges):
            edge = tuple(sorted((int(triangle[edge_vertices[0]]), int(triangle[edge_vertices[1]]))))
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = (tri_index, vertex_index)
            else:
                other_tri, other_vertex = previous
                neighbors[tri_index, vertex_index] = other_tri
                neighbors[other_tri, other_vertex] = tri_index
    return neighbors


def _parse_boundary_edge_classification(boundary_edge_classification):
    if not boundary_edge_classification:
        return None, None

    edge_nodes = boundary_edge_classification.get('edge_nodes')
    edge_classes = boundary_edge_classification.get('edge_classes')
    if edge_nodes is None or edge_classes is None:
        return None, None

    edges = np.asarray(edge_nodes, dtype=np.int64)
    class_codes = _boundary_class_codes(edge_classes)
    if edges.ndim != 2 or edges.shape[1] != 2 or class_codes.shape != (edges.shape[0],):
        return None, None
    return edges, class_codes


def _prepare_boundary_edge_geometry(grid_x, grid_y, boundary_edges, boundary_edge_class_codes):
    if boundary_edges is None or boundary_edge_class_codes is None or boundary_edges.size == 0:
        return None, None

    valid_edges = (
        (boundary_edges >= 0)
        & (boundary_edges < grid_x.size)
        & (boundary_edges < grid_y.size)
    ).all(axis=1)
    if not np.any(valid_edges):
        return None, None

    edges = np.asarray(boundary_edges[valid_edges], dtype=np.int64)
    class_codes = np.asarray(boundary_edge_class_codes[valid_edges], dtype=np.int8)
    return edges, class_codes


def _boundary_class_codes(labels) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=object)
    codes = np.zeros(labels_array.shape, dtype=np.int8)
    codes[labels_array == 'open'] = BOUNDARY_CLASS_OPEN
    codes[labels_array == 'land'] = BOUNDARY_CLASS_LAND
    return codes


def _boundary_class_labels(codes) -> np.ndarray:
    codes_array = np.asarray(codes, dtype=np.int8)
    labels = np.full(codes_array.shape, 'unclassified', dtype=object)
    labels[codes_array == BOUNDARY_CLASS_OPEN] = 'open'
    labels[codes_array == BOUNDARY_CLASS_LAND] = 'land'
    return labels


def _build_triangle_edge_class_codes(triangles, boundary_edges, boundary_edge_class_codes) -> np.ndarray:
    triangle_edge_class_codes = np.zeros((triangles.shape[0], 3), dtype=np.int8)
    if boundary_edges is None or boundary_edge_class_codes is None or boundary_edges.size == 0:
        return triangle_edge_class_codes

    triangle_array = np.asarray(triangles, dtype=np.int64)
    if triangle_array.size == 0:
        return triangle_edge_class_codes
    boundary_edge_array = np.asarray(boundary_edges, dtype=np.int64)
    max_node = int(max(np.max(triangle_array), np.max(boundary_edge_array))) + 1
    if max_node <= 0:
        return triangle_edge_class_codes

    triangle_edges = np.stack(
        (
            triangle_array[:, [1, 2]],
            triangle_array[:, [0, 2]],
            triangle_array[:, [0, 1]],
        ),
        axis=1,
    )
    triangle_edges = np.sort(triangle_edges, axis=2)
    triangle_keys = triangle_edges[:, :, 0] * max_node + triangle_edges[:, :, 1]

    sorted_boundary_edges = np.sort(boundary_edge_array, axis=1)
    boundary_keys = sorted_boundary_edges[:, 0] * max_node + sorted_boundary_edges[:, 1]
    order = np.argsort(boundary_keys)
    sorted_keys = boundary_keys[order]
    sorted_codes = np.asarray(boundary_edge_class_codes, dtype=np.int8)[order]

    flat_keys = triangle_keys.ravel()
    positions = np.searchsorted(sorted_keys, flat_keys)
    in_range = positions < sorted_keys.size
    safe_positions = np.minimum(positions, sorted_keys.size - 1)
    matches = in_range & (sorted_keys[safe_positions] == flat_keys)
    flat_codes = triangle_edge_class_codes.ravel()
    flat_codes[matches] = sorted_codes[safe_positions[matches]]
    return triangle_edge_class_codes


def _segment_distance_squared(p0, p1, q0, q1) -> float:
    """Return squared distance between 2-D line segments."""
    if _segments_intersect(p0, p1, q0, q1):
        return 0.0
    return float(
        min(
            _point_to_segment_distance_squared(p0, q0, q1),
            _point_to_segment_distance_squared(p1, q0, q1),
            _point_to_segment_distance_squared(q0, p0, p1),
            _point_to_segment_distance_squared(q1, p0, p1),
        )
    )


def _point_to_segment_distance_squared(point, seg_start, seg_end) -> float:
    segment = seg_end - seg_start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 0.0:
        delta = point - seg_start
        return float(np.dot(delta, delta))
    projection = float(np.dot(point - seg_start, segment) / length_squared)
    projection = min(1.0, max(0.0, projection))
    closest = seg_start + projection * segment
    delta = point - closest
    return float(np.dot(delta, delta))


def _segments_intersect(p0, p1, q0, q1) -> bool:
    def orientation(a, b, c):
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(value) <= TRIANGLE_TOLERANCE:
            return 0
        return 1 if value > 0 else -1

    def on_segment(a, b, c):
        return (
            min(a[0], c[0]) - TRIANGLE_TOLERANCE <= b[0] <= max(a[0], c[0]) + TRIANGLE_TOLERANCE
            and min(a[1], c[1]) - TRIANGLE_TOLERANCE <= b[1] <= max(a[1], c[1]) + TRIANGLE_TOLERANCE
        )

    o1 = orientation(p0, p1, q0)
    o2 = orientation(p0, p1, q1)
    o3 = orientation(q0, q1, p0)
    o4 = orientation(q0, q1, p1)

    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(p0, q0, p1))
        or (o2 == 0 and on_segment(p0, q1, p1))
        or (o3 == 0 and on_segment(q0, p0, q1))
        or (o4 == 0 and on_segment(q0, p1, q1))
    )


@njit(cache=True, parallel=True)
def _nearest_boundary_edge_indices_numba(
    x0,
    y0,
    x1,
    y1,
    edge_start_x,
    edge_start_y,
    edge_end_x,
    edge_end_y,
    tolerance,
):
    nearest = np.empty(x0.shape[0], dtype=np.int64)
    for i in prange(x0.shape[0]):
        best_index = -1
        best_distance = np.inf
        for edge_index in range(edge_start_x.shape[0]):
            distance = _segment_distance_squared_numba(
                x0[i],
                y0[i],
                x1[i],
                y1[i],
                edge_start_x[edge_index],
                edge_start_y[edge_index],
                edge_end_x[edge_index],
                edge_end_y[edge_index],
                tolerance,
            )
            if best_index < 0 or distance < best_distance:
                best_index = edge_index
                best_distance = distance
        nearest[i] = best_index
    return nearest


@njit(cache=True)
def _segment_distance_squared_numba(p0_x, p0_y, p1_x, p1_y, q0_x, q0_y, q1_x, q1_y, tolerance):
    if _segments_intersect_numba(p0_x, p0_y, p1_x, p1_y, q0_x, q0_y, q1_x, q1_y, tolerance):
        return 0.0
    d0 = _point_to_segment_distance_squared_numba(p0_x, p0_y, q0_x, q0_y, q1_x, q1_y)
    d1 = _point_to_segment_distance_squared_numba(p1_x, p1_y, q0_x, q0_y, q1_x, q1_y)
    d2 = _point_to_segment_distance_squared_numba(q0_x, q0_y, p0_x, p0_y, p1_x, p1_y)
    d3 = _point_to_segment_distance_squared_numba(q1_x, q1_y, p0_x, p0_y, p1_x, p1_y)
    return min(d0, d1, d2, d3)


@njit(cache=True)
def _point_to_segment_distance_squared_numba(point_x, point_y, seg_start_x, seg_start_y, seg_end_x, seg_end_y):
    segment_x = seg_end_x - seg_start_x
    segment_y = seg_end_y - seg_start_y
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared <= 0.0:
        delta_x = point_x - seg_start_x
        delta_y = point_y - seg_start_y
        return delta_x * delta_x + delta_y * delta_y

    projection = ((point_x - seg_start_x) * segment_x + (point_y - seg_start_y) * segment_y) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest_x = seg_start_x + projection * segment_x
    closest_y = seg_start_y + projection * segment_y
    delta_x = point_x - closest_x
    delta_y = point_y - closest_y
    return delta_x * delta_x + delta_y * delta_y


@njit(cache=True)
def _segments_intersect_numba(p0_x, p0_y, p1_x, p1_y, q0_x, q0_y, q1_x, q1_y, tolerance):
    o1 = _orientation_numba(p0_x, p0_y, p1_x, p1_y, q0_x, q0_y, tolerance)
    o2 = _orientation_numba(p0_x, p0_y, p1_x, p1_y, q1_x, q1_y, tolerance)
    o3 = _orientation_numba(q0_x, q0_y, q1_x, q1_y, p0_x, p0_y, tolerance)
    o4 = _orientation_numba(q0_x, q0_y, q1_x, q1_y, p1_x, p1_y, tolerance)

    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment_numba(p0_x, p0_y, q0_x, q0_y, p1_x, p1_y, tolerance))
        or (o2 == 0 and _on_segment_numba(p0_x, p0_y, q1_x, q1_y, p1_x, p1_y, tolerance))
        or (o3 == 0 and _on_segment_numba(q0_x, q0_y, p0_x, p0_y, q1_x, q1_y, tolerance))
        or (o4 == 0 and _on_segment_numba(q0_x, q0_y, p1_x, p1_y, q1_x, q1_y, tolerance))
    )


@njit(cache=True)
def _orientation_numba(a_x, a_y, b_x, b_y, c_x, c_y, tolerance):
    value = (b_x - a_x) * (c_y - a_y) - (b_y - a_y) * (c_x - a_x)
    if abs(value) <= tolerance:
        return 0
    if value > 0.0:
        return 1
    return -1


@njit(cache=True)
def _on_segment_numba(a_x, a_y, b_x, b_y, c_x, c_y, tolerance):
    return (
        min(a_x, c_x) - tolerance <= b_x <= max(a_x, c_x) + tolerance
        and min(a_y, c_y) - tolerance <= b_y <= max(a_y, c_y) + tolerance
    )


@njit(cache=True)
def _weights_in_simplex(simplex, x, y, p0_x, p0_y, inv00, inv01, inv10, inv11):
    dx = x - p0_x[simplex]
    dy = y - p0_y[simplex]
    w1 = inv00[simplex] * dx + inv01[simplex] * dy
    w2 = inv10[simplex] * dx + inv11[simplex] * dy
    w0 = 1.0 - w1 - w2
    return w0, w1, w2


@njit(cache=True)
def _walk_simplex(start, x, y, neighbors, p0_x, p0_y, inv00, inv01, inv10, inv11, max_steps, tolerance):
    n_triangles = neighbors.shape[0]
    if start < 0 or start >= n_triangles:
        return -1, 0.0, 0.0, 0.0

    simplex = start
    for _ in range(max_steps):
        w0, w1, w2 = _weights_in_simplex(simplex, x, y, p0_x, p0_y, inv00, inv01, inv10, inv11)
        if w0 >= -tolerance and w1 >= -tolerance and w2 >= -tolerance:
            return simplex, w0, w1, w2

        edge_index = 0
        min_weight = w0
        if w1 < min_weight:
            edge_index = 1
            min_weight = w1
        if w2 < min_weight:
            edge_index = 2

        next_simplex = neighbors[simplex, edge_index]
        if next_simplex < 0 or next_simplex == simplex:
            break
        simplex = next_simplex

    return -1, 0.0, 0.0, 0.0


@njit(cache=True)
def _walk_simplex_with_exit_class(
    start,
    x,
    y,
    neighbors,
    triangle_edge_class_codes,
    p0_x,
    p0_y,
    inv00,
    inv01,
    inv10,
    inv11,
    max_steps,
    tolerance,
):
    n_triangles = neighbors.shape[0]
    if start < 0 or start >= n_triangles:
        return -1, 0.0, 0.0, 0.0, np.int8(0)

    simplex = start
    for _ in range(max_steps):
        w0, w1, w2 = _weights_in_simplex(simplex, x, y, p0_x, p0_y, inv00, inv01, inv10, inv11)
        if w0 >= -tolerance and w1 >= -tolerance and w2 >= -tolerance:
            return simplex, w0, w1, w2, np.int8(0)

        edge_index = 0
        min_weight = w0
        if w1 < min_weight:
            edge_index = 1
            min_weight = w1
        if w2 < min_weight:
            edge_index = 2

        next_simplex = neighbors[simplex, edge_index]
        if next_simplex < 0 or next_simplex == simplex:
            return -1, 0.0, 0.0, 0.0, triangle_edge_class_codes[simplex, edge_index]
        simplex = next_simplex

    return -1, 0.0, 0.0, 0.0, np.int8(0)


@njit(cache=True, parallel=True)
def _locate_points_walk_numba(
    x_points,
    y_points,
    start_simplices,
    neighbors,
    p0_x,
    p0_y,
    inv00,
    inv01,
    inv10,
    inv11,
    max_steps,
    tolerance,
):
    out = np.empty(start_simplices.shape[0], dtype=np.int64)
    for i in prange(start_simplices.shape[0]):
        simplex, _, _, _ = _walk_simplex(
            start_simplices[i],
            x_points[i],
            y_points[i],
            neighbors,
            p0_x,
            p0_y,
            inv00,
            inv01,
            inv10,
            inv11,
            max_steps,
            tolerance,
        )
        out[i] = simplex
    return out


@njit(cache=True, parallel=True)
def _interpolate_fields_at_simplices_numba(
    fields,
    simplices,
    x_points,
    y_points,
    triangles,
    p0_x,
    p0_y,
    inv00,
    inv01,
    inv10,
    inv11,
    tolerance,
):
    n_fields = fields.shape[0]
    n_points = simplices.shape[0]
    outputs = np.empty((n_fields, n_points), dtype=np.float64)
    refreshed_simplices = simplices.copy()

    for i in prange(n_points):
        simplex = simplices[i]
        if simplex < 0:
            for field_index in range(n_fields):
                outputs[field_index, i] = np.nan
            continue

        if (
            inv00[simplex] == 0.0
            and inv01[simplex] == 0.0
            and inv10[simplex] == 0.0
            and inv11[simplex] == 0.0
        ):
            refreshed_simplices[i] = -1
            for field_index in range(n_fields):
                outputs[field_index, i] = np.nan
            continue

        dx = x_points[i] - p0_x[simplex]
        dy = y_points[i] - p0_y[simplex]
        w1 = inv00[simplex] * dx + inv01[simplex] * dy
        w2 = inv10[simplex] * dx + inv11[simplex] * dy
        w0 = 1.0 - w1 - w2
        if w0 < -tolerance or w1 < -tolerance or w2 < -tolerance:
            refreshed_simplices[i] = -1
            for field_index in range(n_fields):
                outputs[field_index, i] = np.nan
            continue

        v0 = triangles[simplex, 0]
        v1 = triangles[simplex, 1]
        v2 = triangles[simplex, 2]
        for field_index in range(n_fields):
            outputs[field_index, i] = (
                fields[field_index, v0] * w0
                + fields[field_index, v1] * w1
                + fields[field_index, v2] * w2
            )

    return outputs, refreshed_simplices


@njit(cache=True)
def _interpolate_temporal_velocity(
    x,
    y,
    start_simplex,
    lower_u,
    lower_v,
    upper_u,
    upper_v,
    weight,
    triangles,
    neighbors,
    p0_x,
    p0_y,
    inv00,
    inv01,
    inv10,
    inv11,
    max_steps,
    tolerance,
):
    simplex, w0, w1, w2 = _walk_simplex(
        start_simplex,
        x,
        y,
        neighbors,
        p0_x,
        p0_y,
        inv00,
        inv01,
        inv10,
        inv11,
        max_steps,
        tolerance,
    )
    if simplex < 0:
        return 0.0, 0.0, -1

    i0 = triangles[simplex, 0]
    i1 = triangles[simplex, 1]
    i2 = triangles[simplex, 2]
    lower_u_value = w0 * lower_u[i0] + w1 * lower_u[i1] + w2 * lower_u[i2]
    lower_v_value = w0 * lower_v[i0] + w1 * lower_v[i1] + w2 * lower_v[i2]
    if weight <= 0.0:
        return lower_u_value, lower_v_value, simplex

    upper_u_value = w0 * upper_u[i0] + w1 * upper_u[i1] + w2 * upper_u[i2]
    upper_v_value = w0 * upper_v[i0] + w1 * upper_v[i1] + w2 * upper_v[i2]
    return (
        lower_u_value + weight * (upper_u_value - lower_u_value),
        lower_v_value + weight * (upper_v_value - lower_v_value),
        simplex,
    )


@njit(cache=True, parallel=True)
def _update_particles_temporal_numba(
    x0,
    y0,
    lower_u,
    lower_v,
    upper_u,
    upper_v,
    weight,
    dt,
    start_simplices,
    triangles,
    neighbors,
    p0_x,
    p0_y,
    inv00,
    inv01,
    inv10,
    inv11,
    triangle_edge_class_codes,
    max_steps,
    tolerance,
):
    x_new = np.empty_like(x0, dtype=np.float64)
    y_new = np.empty_like(y0, dtype=np.float64)
    simplex_new = np.empty(start_simplices.shape[0], dtype=np.int64)
    boundary_class_codes = np.zeros(start_simplices.shape[0], dtype=np.int8)

    for i in prange(x0.shape[0]):
        start = start_simplices[i]
        u1, v1, s1 = _interpolate_temporal_velocity(
            x0[i],
            y0[i],
            start,
            lower_u,
            lower_v,
            upper_u,
            upper_v,
            weight,
            triangles,
            neighbors,
            p0_x,
            p0_y,
            inv00,
            inv01,
            inv10,
            inv11,
            max_steps,
            tolerance,
        )
        x1 = x0[i] + 0.5 * u1 * dt
        y1 = y0[i] + 0.5 * v1 * dt

        u2, v2, s2 = _interpolate_temporal_velocity(
            x1,
            y1,
            s1 if s1 >= 0 else start,
            lower_u,
            lower_v,
            upper_u,
            upper_v,
            weight,
            triangles,
            neighbors,
            p0_x,
            p0_y,
            inv00,
            inv01,
            inv10,
            inv11,
            max_steps,
            tolerance,
        )
        x2 = x0[i] + 0.5 * u2 * dt
        y2 = y0[i] + 0.5 * v2 * dt

        u3, v3, s3 = _interpolate_temporal_velocity(
            x2,
            y2,
            s2 if s2 >= 0 else s1,
            lower_u,
            lower_v,
            upper_u,
            upper_v,
            weight,
            triangles,
            neighbors,
            p0_x,
            p0_y,
            inv00,
            inv01,
            inv10,
            inv11,
            max_steps,
            tolerance,
        )
        x3 = x0[i] + u3 * dt
        y3 = y0[i] + v3 * dt

        u4, v4, s4 = _interpolate_temporal_velocity(
            x3,
            y3,
            s3 if s3 >= 0 else s2,
            lower_u,
            lower_v,
            upper_u,
            upper_v,
            weight,
            triangles,
            neighbors,
            p0_x,
            p0_y,
            inv00,
            inv01,
            inv10,
            inv11,
            max_steps,
            tolerance,
        )

        x_out = x0[i] + dt / 6.0 * (u1 + 2.0 * u2 + 2.0 * u3 + u4)
        y_out = y0[i] + dt / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
        x_new[i] = x_out
        y_new[i] = y_out

        final_start = s4
        if final_start < 0:
            final_start = s3
        if final_start < 0:
            final_start = s2
        if final_start < 0:
            final_start = s1
        if final_start < 0:
            final_start = start
        final_simplex, _, _, _, exit_class_code = _walk_simplex_with_exit_class(
            final_start,
            x_out,
            y_out,
            neighbors,
            triangle_edge_class_codes,
            p0_x,
            p0_y,
            inv00,
            inv01,
            inv10,
            inv11,
            max_steps,
            tolerance,
        )
        simplex_new[i] = final_simplex
        if final_simplex < 0:
            boundary_class_codes[i] = exit_class_code

    return x_new, y_new, simplex_new, boundary_class_codes


def create_grid_geometry(grid_x, grid_y, triangles=None, boundary_edge_classification=None) -> GridGeometry:
    """
    Create cached grid geometry for repeated particle interpolation.

    Parameters
    ----------
    grid_x : object
        Grid node x coordinates.
    grid_y : object
        Grid node y coordinates.
    triangles : object
        Triangle connectivity array.

    Returns
    -------
    GridGeometry
        Cached grid geometry instance.
    """
    return GridGeometry.from_points(
        grid_x,
        grid_y,
        triangles=triangles,
        boundary_edge_classification=boundary_edge_classification,
    )


def create_numba_particle_calculator(
    grid_x,
    grid_y,
    triangles=None,
    grid_geometry=None,
    boundary_edge_classification=None,
):
    """
    Create particle interpolation/update callables.

    The returned dictionary keeps the historical keys used by ParticlePopulation.

    Parameters
    ----------
    grid_x : object
        Grid node x coordinates.
    grid_y : object
        Grid node y coordinates.
    triangles : object
        Triangle connectivity array.
    grid_geometry : object
        The grid geometry value.

    Returns
    -------
    dict[str, object]
        Dictionary of geometry and particle interpolation/update callables.
    """
    geometry = (
        grid_geometry
        if grid_geometry is not None
        else create_grid_geometry(
            grid_x,
            grid_y,
            triangles=triangles,
            boundary_edge_classification=boundary_edge_classification,
        )
    )

    return {
        'geometry': geometry,
        'triangles': geometry.triangles,
        'find_triangle': geometry.find_triangle,
        'interpolate_field': geometry.interpolate_field,
        'interpolate_fields': geometry.interpolate_fields,
        'interpolate_fields_with_simplex': geometry.interpolate_fields_with_simplex,
        'update_particles': geometry.update_particles,
        'update_particles_temporal': geometry.update_particles_temporal,
        'update_particles_with_simplex': geometry.update_particles_with_simplex,
        'update_particles_temporal_with_simplex': geometry.update_particles_temporal_with_simplex,
        'update_particles_with_boundary_class': geometry.update_particles_with_boundary_class,
        'update_particles_temporal_with_boundary_class': geometry.update_particles_temporal_with_boundary_class,
        'update_particles_parallel': geometry.update_particles,
    }


def find_triangle(x, y, grid_x, grid_y, triangles=None):
    """
    Compatibility wrapper for one-off triangle lookup.

    Parameters
    ----------
    x : object
        X coordinate value or array.
    y : object
        Y coordinate value or array.
    grid_x : object
        Grid node x coordinates.
    grid_y : object
        Grid node y coordinates.
    triangles : object
        Triangle connectivity array.

    Returns
    -------
    int
        Index of the containing triangle, or -1 outside the grid.
    """
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).find_triangle(x, y)


def interpolate_field(field, x_points, y_points, grid_x, grid_y, triangles=None):
    """
    Compatibility wrapper for one-off scalar interpolation.

    Parameters
    ----------
    field : object
        Scalar field values defined on grid nodes.
    x_points : object
        Point x coordinates to sample.
    y_points : object
        Point y coordinates to sample.
    grid_x : object
        Grid node x coordinates.
    grid_y : object
        Grid node y coordinates.
    triangles : object
        Triangle connectivity array.

    Returns
    -------
    np.ndarray
        Interpolated scalar field values at the requested points.
    """
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).interpolate_field(field, x_points, y_points)


def update_particles_rk4(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0):
    """
    Compatibility wrapper for one-off RK4 particle updates.

    Parameters
    ----------
    x0 : object
        Initial particle x coordinates.
    y0 : object
        Initial particle y coordinates.
    grid_u : object
        Grid-aligned x velocity component.
    grid_v : object
        Grid-aligned y velocity component.
    grid_x : object
        Grid node x coordinates.
    grid_y : object
        Grid node y coordinates.
    triangles : object
        Triangle connectivity array.
    dt : object
        Integration timestep in seconds.
    igeo : object
        Coordinate-system flag used by the legacy kernels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated x and y particle positions.
    """
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).update_particles(x0, y0, grid_u, grid_v, dt, igeo)


def update_particles_rk4_parallel(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0):
    """
    Compatibility wrapper for the historical parallel update function.

    Parameters
    ----------
    x0 : object
        Initial particle x coordinates.
    y0 : object
        Initial particle y coordinates.
    grid_u : object
        Grid-aligned x velocity component.
    grid_v : object
        Grid-aligned y velocity component.
    grid_x : object
        Grid node x coordinates.
    grid_y : object
        Grid node y coordinates.
    triangles : object
        Triangle connectivity array.
    dt : object
        Integration timestep in seconds.
    igeo : object
        Coordinate-system flag used by the legacy kernels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated x and y particle positions.
    """
    return update_particles_rk4(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo)
