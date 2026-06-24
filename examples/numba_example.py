"""
SedTRAILS reusable-grid particle update example.

This small example uses a synthetic square grid so it can be read without any
external NetCDF forcing file. It demonstrates the current
create_numba_particle_calculator API and the dictionary of callables it returns.
"""

import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator


# ===== STEP 1: Define a tiny triangular grid =====

# Four nodes and two triangles define a unit square.
grid_x = np.array([0.0, 1.0, 1.0, 0.0])
grid_y = np.array([0.0, 0.0, 1.0, 1.0])
triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


# ===== STEP 2: Define a velocity field on that grid =====

# Constant eastward velocity, expressed at the grid nodes.
grid_u = np.full(grid_x.shape, 0.1)
grid_v = np.zeros_like(grid_u)


# ===== STEP 3: Create the current reusable grid calculator =====

# Current signature: create_numba_particle_calculator(grid_x, grid_y, triangles=None)
calculator = create_numba_particle_calculator(grid_x, grid_y, triangles=triangles)

print('Returned calculator keys:')
for key in calculator:
    print(f'  {key}')


# ===== STEP 4: Update several particles =====

x = np.array([0.25, 0.50, 0.75])
y = np.array([0.25, 0.50, 0.75])
dt = 1.0

# Current signature: calculator['update_particles'](x, y, grid_u, grid_v, dt)
x_next, y_next = calculator['update_particles'](x, y, grid_u, grid_v, dt)

print('\nInitial x:', x)
print('Initial y:', y)
print('Updated x:', x_next)
print('Updated y:', y_next)
