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

    @classmethod
    def from_points(cls, grid_x, grid_y, triangles=None):
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
        )

    def find_triangle(self, x, y) -> int:
        """Return the containing simplex index, or -1 outside the triangulation."""
        if self.triangulation is not None:
            simplex = self.triangulation.find_simplex(np.array([[x, y]], dtype=np.float64), tol=TRIANGLE_TOLERANCE)
            return int(simplex[0])
        return int(self.triangle_finder([x], [y])[0])

    def locate_points(self, x_points, y_points, start_simplices=None):
        """Locate points, using cached simplices first and global search only for misses."""
        points = _points_array(x_points, y_points)
        if points.size == 0:
            return np.empty(0, dtype=np.int64)

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
        """Return simplex indices and barycentric weights for points."""
        points = _points_array(x_points, y_points)
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

    def interpolate_field(self, field, x_points, y_points):
        """Barycentrically interpolate a nodal field at point coordinates."""
        return self.interpolate_fields((field,), x_points, y_points)[0]

    def interpolate_fields(self, fields, x_points, y_points):
        """Interpolate multiple nodal fields using one point-location pass."""
        simplices, weights = self.barycentric_weights(x_points, y_points)
        outputs = [np.zeros(len(simplices), dtype=np.float64) for _ in fields]

        valid = simplices >= 0
        if np.any(valid):
            vertices = self.triangles[simplices[valid]]
            valid_weights = weights[valid]
            for output, field in zip(outputs, fields, strict=True):
                values = np.asarray(field).ravel()
                output[valid] = np.einsum('ij,ij->i', values[vertices], valid_weights)

        return tuple(outputs)

    def update_particles(self, x0, y0, grid_u, grid_v, dt, igeo=0):
        """Advance particle positions one RK4 step using barycentric velocity sampling."""
        return self.update_particles_temporal(x0, y0, grid_u, grid_v, grid_u, grid_v, 0.0, dt, igeo)

    def update_particles_temporal(self, x0, y0, lower_u, lower_v, upper_u, upper_v, weight, dt, igeo=0):
        """Advance particles using lower/upper time-slice velocities blended by weight."""
        x_new, y_new, _ = self.update_particles_temporal_with_simplex(
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
        """Advance particles and return updated simplex ids for the new positions."""
        return self.update_particles_temporal_with_simplex(
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
        """Advance particles with a Numba RK4 kernel and cached simplex ids."""
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64)
        particle_shape = x0.shape
        if x0.size == 0:
            return x0.copy(), y0.copy(), np.empty(0, dtype=np.int64)

        lower_u_adj, lower_v_adj = self._velocity_arrays(lower_u, lower_v, igeo)
        if weight <= 0.0:
            upper_u_adj = lower_u_adj
            upper_v_adj = lower_v_adj
        else:
            upper_u_adj, upper_v_adj = self._velocity_arrays(upper_u, upper_v, igeo)

        starts = self.locate_points(x0, y0, simplex_ids)
        x_new, y_new, new_simplices = _update_particles_temporal_numba(
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
            MAX_SIMPLEX_WALK_STEPS,
            TRIANGLE_TOLERANCE,
        )
        return x_new.reshape(particle_shape), y_new.reshape(particle_shape), new_simplices

    def interpolate_vector(self, grid_u, grid_v, x_points, y_points):
        """Interpolate vector components at point coordinates."""
        return (
            self.interpolate_field(grid_u, x_points, y_points),
            self.interpolate_field(grid_v, x_points, y_points),
        )

    def interpolate_temporal_vector(self, lower_u, lower_v, upper_u, upper_v, weight, x_points, y_points):
        """Interpolate lower/upper vector fields spatially, then blend in time."""
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
    max_steps,
    tolerance,
):
    x_new = np.empty_like(x0, dtype=np.float64)
    y_new = np.empty_like(y0, dtype=np.float64)
    simplex_new = np.empty(start_simplices.shape[0], dtype=np.int64)

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
        final_simplex, _, _, _ = _walk_simplex(
            final_start,
            x_out,
            y_out,
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
        simplex_new[i] = final_simplex

    return x_new, y_new, simplex_new


def create_grid_geometry(grid_x, grid_y, triangles=None) -> GridGeometry:
    """Create cached grid geometry for repeated particle interpolation."""
    return GridGeometry.from_points(grid_x, grid_y, triangles=triangles)


def create_numba_particle_calculator(grid_x, grid_y, triangles=None, grid_geometry=None):
    """
    Create particle interpolation/update callables.

    The returned dictionary keeps the historical keys used by ParticlePopulation.
    """
    geometry = grid_geometry if grid_geometry is not None else create_grid_geometry(grid_x, grid_y, triangles)

    return {
        'geometry': geometry,
        'triangles': geometry.triangles,
        'find_triangle': geometry.find_triangle,
        'interpolate_field': geometry.interpolate_field,
        'interpolate_fields': geometry.interpolate_fields,
        'update_particles': geometry.update_particles,
        'update_particles_temporal': geometry.update_particles_temporal,
        'update_particles_with_simplex': geometry.update_particles_with_simplex,
        'update_particles_temporal_with_simplex': geometry.update_particles_temporal_with_simplex,
        'update_particles_parallel': geometry.update_particles,
    }


def find_triangle(x, y, grid_x, grid_y, triangles=None):
    """Compatibility wrapper for one-off triangle lookup."""
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).find_triangle(x, y)


def interpolate_field(field, x_points, y_points, grid_x, grid_y, triangles=None):
    """Compatibility wrapper for one-off scalar interpolation."""
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).interpolate_field(field, x_points, y_points)


def update_particles_rk4(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0):
    """Compatibility wrapper for one-off RK4 particle updates."""
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).update_particles(x0, y0, grid_u, grid_v, dt, igeo)


def update_particles_rk4_parallel(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0):
    """Compatibility wrapper for the historical parallel update function."""
    return update_particles_rk4(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo)
