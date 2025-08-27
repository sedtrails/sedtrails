"""
Unit tests for the ParticlePositionCalculator class, covering field interpolation,
particle updates with and without diffusion, parallel vs serial consistency,
geographic adjustments, and mixed network interpolation.
"""

import numpy as np
import pytest

from sedtrails.particle_tracer.position_calculator import (
    ParticlePositionCalculator,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def simple_interpolator():
    """
    Create a simple ParticlePositionCalculator instance based on a single triangle
    with a constant velocity field.
    """
    # Define a simple nondegenerate triangle.
    grid_x = np.array([0.0, 1.0, 0.0])
    grid_y = np.array([0.0, 0.0, 1.0])
    # Constant velocity field.
    grid_u = np.array([1.0, 1.0, 1.0])
    grid_v = np.array([0.0, 0.0, 0.0])

    return ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, igeo=0)


@pytest.fixture
def mixed_network_interpolator():
    """
    Creates a structured grid ParticlePositionCalculator with consistent triangle
    connectivity, suitable for interpolation accuracy tests with linear fields.
    """

    # Create a 4x4 structured grid: (x,y) ∈ [0, 3] × [0, 3]
    nx, ny = 4, 4
    x = np.linspace(0, 3, nx)
    y = np.linspace(0, 3, ny)
    xx, yy = np.meshgrid(x, y)
    grid_x = xx.ravel()
    grid_y = yy.ravel()

    # Constant velocity field (not used in the interpolation test)
    grid_u = np.ones_like(grid_x)
    grid_v = np.full_like(grid_y, 0.5)

    # Triangulate the structured grid into two triangles per quad cell
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            triangles.append([a, b, d])  # lower-right triangle
            triangles.append([a, d, c])  # upper-left triangle
    triangles = np.array(triangles)

    return ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, triangles, igeo=0)


@pytest.fixture
def simple_grid():
    # 2x2 square grid, forming two triangles
    grid_x = np.array([0.0, 1.0, 1.0, 0.0])
    grid_y = np.array([0.0, 0.0, 1.0, 1.0])
    grid_u = np.array([1.0, 1.0, 1.0, 1.0])  # uniform velocity (u=1)
    grid_v = np.array([0.0, 0.0, 0.0, 0.0])  # no v velocity
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    return grid_x, grid_y, grid_u, grid_v, triangles


def test_interpolation_on_uniform_field(simple_grid):
    grid_x, grid_y, grid_u, grid_v, triangles = simple_grid
    calc = ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, triangles)

    # Try interpolating inside the square
    test_points_x = np.array([0.5, 0.75])
    test_points_y = np.array([0.5, 0.25])
    interpolated = calc.interpolate_field(grid_u, test_points_x, test_points_y)

    # All values should be close to 1.0
    np.testing.assert_allclose(interpolated, 1.0, atol=1e-6)


def test_particle_rk4_motion_straight_line(simple_grid):
    grid_x, grid_y, grid_u, grid_v, triangles = simple_grid
    calc = ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, triangles)

    x0 = np.array([0.1])
    y0 = np.array([0.1])
    dt = np.float32(1.0)

    x_new, y_new = calc.update_particles(x0, y0, dt)

    # Account for RK4 over triangle mesh: expected ~0.9333
    expected_x = 0.93333333
    expected_y = 0.1

    assert np.allclose(x_new, expected_x, atol=1e-6)
    assert np.allclose(y_new, expected_y, atol=1e-6)


def test_particle_parallel_and_serial_match(simple_grid):
    grid_x, grid_y, grid_u, grid_v, triangles = simple_grid
    calc = ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, triangles)

    x0 = np.linspace(0.1, 0.9, 100)
    y0 = np.full_like(x0, 0.1)
    dt = np.float32(0.5)

    x_serial, y_serial = calc.update_particles(x0, y0, dt, parallel=False)
    x_parallel, y_parallel = calc.update_particles(x0, y0, dt, parallel=True)

    np.testing.assert_allclose(x_serial, x_parallel, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(y_serial, y_parallel, rtol=1e-6, atol=1e-8)


def test_igeo_scaling_applies_correctly():
    triangles = np.array([[0, 1, 2]])
    grid_x = np.array([0.0, 1.0, 0.0])
    grid_y = np.array([0.0, 0.0, 1.0])  # 3 points forming a right triangle
    grid_u = np.array([1.0, 1.0, 1.0])
    grid_v = np.array([0.0, 0.0, 0.0])

    calc = ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, triangles, igeo=1)

    x0 = np.array([0.0])
    y0 = np.array([0.0])
    dt = np.float32(1.0)

    x_new, y_new = calc.update_particles(x0, y0, dt)
    # Expect scaled movement: u = 1 / geofac
    expected_dx = 1.0 / calc.geofac
    assert np.allclose(x_new[0], x0[0] + expected_dx, atol=1e-8)
    assert np.allclose(y_new[0], y0[0], atol=1e-8)


# -----------------------------------------------------------------------------
# Tests for field interpolation
# -----------------------------------------------------------------------------


def test_interpolate_field_constant(simple_interpolator):
    """
    When the grid field is constant, the interpolated value at any particle
    position inside the triangle should equal that constant.
    """
    constant_field = np.full_like(simple_interpolator.grid_x, 5.0)
    part_x = np.array([0.2, 0.4, 0.1])
    part_y = np.array([0.2, 0.1, 0.4])
    interp_vals = simple_interpolator.interpolate_field(constant_field, part_x, part_y)
    np.testing.assert_allclose(interp_vals, 5.0)


# -----------------------------------------------------------------------------
# Tests for particle update without diffusion
# -----------------------------------------------------------------------------


def test_update_particles_no_velocity(simple_interpolator):
    """
    If grid velocities are zero, the particle positions should remain unchanged.
    """
    simple_interpolator.grid_u = np.zeros_like(simple_interpolator.grid_u)
    simple_interpolator.grid_v = np.zeros_like(simple_interpolator.grid_v)
    part_x = np.array([0.2, 0.4, 0.1])
    part_y = np.array([0.2, 0.1, 0.4])
    dt = 0.1
    x_new, y_new = simple_interpolator.update_particles(part_x, part_y, dt, parallel=False)
    np.testing.assert_allclose(x_new, part_x)
    np.testing.assert_allclose(y_new, part_y)


# -----------------------------------------------------------------------------
# Tests for consistency between parallel and serial execution
# -----------------------------------------------------------------------------


def test_parallel_vs_serial(simple_interpolator):
    """
    With diffusion disabled, the serial and parallel versions should produce
    nearly identical particle updates.
    """
    part_x = np.linspace(0.1, 0.9, 50)
    part_y = np.linspace(0.1, 0.9, 50)
    dt = 0.1
    x_serial, y_serial = simple_interpolator.update_particles(part_x, part_y, dt, parallel=False)
    x_parallel, y_parallel = simple_interpolator.update_particles(part_x, part_y, dt, parallel=True, num_workers=2)
    np.testing.assert_allclose(x_serial, x_parallel)
    np.testing.assert_allclose(y_serial, y_parallel)


# -----------------------------------------------------------------------------
# Test for geographic adjustment (igeo==1)
# -----------------------------------------------------------------------------


def test_geographic_adjustment():
    """
    For geographic coordinates (igeo==1), grid_u values should be scaled by
    1/(cos(latitude)*geofac) and grid_v by 1/geofac. For a constant field,
    the RK4 update should yield x_new = x0 + dt * adjusted_u.
    """
    # Define a nondegenerate triangle with constant latitude.
    grid_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    grid_y = np.array([45.0, 45.0, 46.0], dtype=np.float64)
    grid_u = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    grid_v = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    interpolator = ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, igeo=1)
    dt = 1.0
    part_x = np.array([0.2], dtype=np.float64)
    part_y = np.array([0.2], dtype=np.float64)
    expected_u = 1.0 / (np.cos(np.deg2rad(45.0)) * interpolator.geofac)
    expected_x_new = part_x + dt * expected_u
    x_new, y_new = interpolator.update_particles(part_x, part_y, dt, parallel=False)
    np.testing.assert_allclose(x_new, expected_x_new, rtol=1e-5)
    np.testing.assert_allclose(y_new, part_y, rtol=1e-5)


# -----------------------------------------------------------------------------
# Mixed network interpolation tests
# -----------------------------------------------------------------------------
def test_mixed_network_linear_interpolation(mixed_network_interpolator):
    """
    If the grid field is f(x,y) = 2*x + 3*y, barycentric interpolation should
    recover the exact value (to within numerical tolerance).

    The test first draws a large pool of random points, then keeps only those
    for which the interpolator returns a finite value (i.e. they truly lie
    inside some triangle *according to the same barycentric test* used by the
    production code).  This eliminates spurious NaNs arising from tiny
    tolerance differences.
    """

    # --- define linear field on the grid ------------------------------------
    def linear_func(x, y):
        return 2 * x + 3 * y

    field = linear_func(
        mixed_network_interpolator.grid_x,
        mixed_network_interpolator.grid_y,
    )

    print('Triangles:')
    print(mixed_network_interpolator.triangles)

    # Generate 1000 points, filter to inside triangles
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 3, 1000)
    y = rng.uniform(0, 3, 1000)

    interp = mixed_network_interpolator.interpolate_field(field, x, y)
    mask = np.isfinite(interp)

    assert mask.sum() >= 20
    x_valid = x[mask][:20]
    y_valid = y[mask][:20]

    interp_vals = mixed_network_interpolator.interpolate_field(field, x_valid, y_valid)
    expected_vals = linear_func(x_valid, y_valid)

    np.testing.assert_allclose(interp_vals, expected_vals, rtol=1e-6, atol=1e-10)


def test_mixed_network_update_particles_serial(mixed_network_interpolator):
    """
    Test that updating particles with constant grid velocities produces a simple shift.
    With constant u = 1.0 and v = 0.5, the RK4 integration should yield:
         x_new = x0 + dt * 1.0   and   y_new = y0 + dt * 0.5.
    """
    np.random.seed(42)
    # Use particles strictly inside the domain to avoid boundary issues.
    part_x = np.random.uniform(0.1, 1.9, 2)
    part_y = np.random.uniform(0.1, 2.9, 2)
    dt = 0.2

    expected_x = part_x + dt * 1.0
    expected_y = part_y + dt * 0.5

    x_new, y_new = mixed_network_interpolator.update_particles(part_x, part_y, dt, parallel=False)

    np.testing.assert_allclose(x_new, expected_x, rtol=1e-5)
    np.testing.assert_allclose(y_new, expected_y, rtol=1e-5)


def test_mixed_network_update_particles_parallel(mixed_network_interpolator):
    """
    Test that updating particles with constant grid velocities produces a simple shift.
    With constant u = 1.0 and v = 0.5, the RK4 integration should yield:
         x_new = x0 + dt * 1.0   and   y_new = y0 + dt * 0.5.
    """
    np.random.seed(42)
    # Use particles strictly inside the domain to avoid boundary issues.
    part_x = np.random.uniform(0.1, 1.9, 2)
    part_y = np.random.uniform(0.1, 2.9, 2)
    dt = 0.2

    expected_x = part_x + dt * 1.0
    expected_y = part_y + dt * 0.5

    x_new, y_new = mixed_network_interpolator.update_particles(part_x, part_y, dt, parallel=True, num_workers=3)

    np.testing.assert_allclose(x_new, expected_x, rtol=1e-5)
    np.testing.assert_allclose(y_new, expected_y, rtol=1e-5)
