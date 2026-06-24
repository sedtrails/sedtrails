"""
SedTRAILS particle simulation example.

This script demonstrates the lower-level API workflow for particle tracking:

1. Convert a hydrodynamic model file to SedTRAILS' internal data object.
2. Retrieve a named flow field at requested simulation times.
3. Move one particle through that flow field with ParticlePositionCalculator.
4. Plot the resulting trajectory.

The full SedTRAILS simulation manager does more than this example: it handles
configuration files, population seeding, status flags, physics runtime plans,
output writing, and dashboards. This script keeps those pieces out of the way
so the data flow through the particle-update API is visible.
"""

from pathlib import Path

import numpy as np

from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.particle_tracer.position_calculator import ParticlePositionCalculator
from sedtrails.transport_converter import FormatConverter

from visualization_utils import plot_particle_trajectory


# ===== CONFIGURATION =====

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent

FILE_PATH = REPO_ROOT / 'sample-data' / 'inlet_sedtrails.nc'
INPUT_FORMAT = 'fm_netcdf'
REFERENCE_DATE = '1970-01-01'
FLOW_FIELD_NAME = 'depth_avg_flow_velocity'

START_X = 40000.0
START_Y = 17000.0
TIMESTEP = 30.0  # seconds
TIMESTEP_INDEX = 2
NUM_STEPS = 30

OUTPUT_PATH = EXAMPLES_DIR / 'results' / 'particle_trajectory.png'


def main() -> int:
    """Run the example."""
    if not FILE_PATH.exists():
        raise FileNotFoundError(
            f'Input file not found: {FILE_PATH}. '
            'Download/create the forcing file, or edit FILE_PATH in this example.'
        )

    # ===== STEP 1: Load and convert flow-field data =====
    print('\n=== STEP 1: Loading and converting flow-field data ===')

    format_config = {
        'input_file': str(FILE_PATH),
        'input_format': INPUT_FORMAT,
        'reference_date': REFERENCE_DATE,
        'morfac': 1.0,
    }
    converter = FormatConverter(format_config)
    sedtrails_data = converter.convert_to_sedtrails()

    times = np.asarray(sedtrails_data.times, dtype=float)
    print(f'Converted {FILE_PATH}')
    print(f'Input times: {times[0]:.0f}s to {times[-1]:.0f}s ({times.size} timestamps)')

    # ===== STEP 2: Retrieve the first flow field =====
    print('\n=== STEP 2: Creating a field retriever ===')
    retriever = FieldDataRetriever(sedtrails_data)

    initial_time = float(times[TIMESTEP_INDEX])
    initial_flow = retriever.get_flow_field(initial_time, FLOW_FIELD_NAME)
    print(f'Retrieved {FLOW_FIELD_NAME!r} at t={initial_time:.0f}s')
    print(f'Flow field contains {initial_flow["x"].size} spatial points')

    # The calculator API operates on NumPy arrays of particle positions. Here we
    # track one particle, so the arrays have length one.
    particle_x = np.array([START_X], dtype=float)
    particle_y = np.array([START_Y], dtype=float)

    trajectory_x = [particle_x[0]]
    trajectory_y = [particle_y[0]]
    trajectory_time = [initial_time]
    print(f'Created one particle at ({particle_x[0]:.2f}, {particle_y[0]:.2f})')

    # ===== STEP 3: Move the particle through time =====
    print('\n=== STEP 3: Updating the particle position ===')
    current_time = initial_time

    for step in range(1, NUM_STEPS + 1):
        current_time = initial_time + step * TIMESTEP

        # Current signature: get_flow_field(time, flow_field_name)
        flow_data = retriever.get_flow_field(current_time, FLOW_FIELD_NAME)

        # Current signature: ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v)
        calculator = ParticlePositionCalculator(
            grid_x=flow_data['x'],
            grid_y=flow_data['y'],
            grid_u=flow_data['u'],
            grid_v=flow_data['v'],
        )

        # Current signature: update_particles(x0, y0, dt)
        particle_x, particle_y = calculator.update_particles(particle_x, particle_y, TIMESTEP)

        trajectory_x.append(particle_x[0])
        trajectory_y.append(particle_y[0])
        trajectory_time.append(current_time)

        print(
            f'Step {step:02d}/{NUM_STEPS}: '
            f't={current_time:.0f}s, position=({particle_x[0]:.2f}, {particle_y[0]:.2f})'
        )

    trajectory_x = np.asarray(trajectory_x)
    trajectory_y = np.asarray(trajectory_y)
    trajectory_time = np.asarray(trajectory_time)

    # ===== STEP 4: Plot and summarize the trajectory =====
    print('\n=== STEP 4: Visualizing and summarizing results ===')
    final_flow = retriever.get_flow_field(float(trajectory_time[-1]), FLOW_FIELD_NAME)

    plot_particle_trajectory(
        flow_data=final_flow,
        trajectory_x=trajectory_x,
        trajectory_y=trajectory_y,
        title=f'Particle trajectory through {FLOW_FIELD_NAME}',
        save_path=OUTPUT_PATH,
    )
    print(f'Trajectory plot saved to {OUTPUT_PATH}')

    step_distances = np.hypot(np.diff(trajectory_x), np.diff(trajectory_y))
    total_distance = float(np.sum(step_distances))
    displacement = float(np.hypot(trajectory_x[-1] - trajectory_x[0], trajectory_y[-1] - trajectory_y[0]))

    print('\n=== Simulation statistics ===')
    print(f'Starting position: ({trajectory_x[0]:.2f}, {trajectory_y[0]:.2f})')
    print(f'Final position:    ({trajectory_x[-1]:.2f}, {trajectory_y[-1]:.2f})')
    print(f'Total distance:    {total_distance:.2f} m')
    print(f'Displacement:      {displacement:.2f} m')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
