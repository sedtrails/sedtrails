"""
SedTRAILS particle-update API comparison.

This script walks through two current ways to update particle positions:

1. ParticlePositionCalculator
2. The reusable grid calculator returned by create_numba_particle_calculator

Both paths use the same grid, the same velocity field, and the same initial
particle position. The printed positions show how the two APIs express the same
particle-update problem.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.particle_tracer.position_calculator import ParticlePositionCalculator
from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator
from sedtrails.transport_converter import FormatConverter

from visualization_utils import plot_flow_field, plot_particle_trajectory


# ===== CONFIGURATION =====

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent

FILE_PATH = REPO_ROOT / 'sample-data' / 'inlet_sedtrails.nc'
INPUT_FORMAT = 'fm_netcdf'
REFERENCE_DATE = '1970-01-01'
FLOW_FIELD_NAME = 'depth_avg_flow_velocity'

START_X = 40000.0
START_Y = 17000.0
TIMESTEP = 30.0
TIMESTEP_INDEX = 2
NUM_STEPS = 20

OUTPUT_DIR = EXAMPLES_DIR / 'results' / 'particle_update_comparison'


def run_particle_position_calculator(flow_sequence, x0, y0):
    """Advance one particle with ParticlePositionCalculator."""
    x = np.array([x0], dtype=float)
    y = np.array([y0], dtype=float)
    trajectory_x = [x[0]]
    trajectory_y = [y[0]]

    for flow_data in flow_sequence:
        calculator = ParticlePositionCalculator(
            grid_x=flow_data['x'],
            grid_y=flow_data['y'],
            grid_u=flow_data['u'],
            grid_v=flow_data['v'],
        )
        x, y = calculator.update_particles(x, y, TIMESTEP)
        trajectory_x.append(x[0])
        trajectory_y.append(y[0])

    return np.asarray(trajectory_x), np.asarray(trajectory_y)


def run_reusable_grid_calculator(flow_sequence, x0, y0):
    """Advance one particle with create_numba_particle_calculator."""
    first_flow = flow_sequence[0]

    # Current signature: create_numba_particle_calculator(grid_x, grid_y, triangles=None)
    calculator = create_numba_particle_calculator(first_flow['x'], first_flow['y'])

    x = np.array([x0], dtype=float)
    y = np.array([y0], dtype=float)
    trajectory_x = [x[0]]
    trajectory_y = [y[0]]

    for flow_data in flow_sequence:
        # Current signature: calculator['update_particles'](x, y, grid_u, grid_v, dt)
        x, y = calculator['update_particles'](x, y, flow_data['u'], flow_data['v'], TIMESTEP)
        trajectory_x.append(x[0])
        trajectory_y.append(y[0])

    return np.asarray(trajectory_x), np.asarray(trajectory_y)


def main() -> int:
    """Run the example."""
    if not FILE_PATH.exists():
        raise FileNotFoundError(
            f'Input file not found: {FILE_PATH}. '
            'Download/create the forcing file, or edit FILE_PATH in this example.'
        )

    # ===== STEP 1: Load and convert flow-field data =====
    print('\n=== STEP 1: Loading and converting flow-field data ===')
    converter = FormatConverter(
        {
            'input_file': str(FILE_PATH),
            'input_format': INPUT_FORMAT,
            'reference_date': REFERENCE_DATE,
            'morfac': 1.0,
        }
    )
    sedtrails_data = converter.convert_to_sedtrails()
    times = np.asarray(sedtrails_data.times, dtype=float)
    retriever = FieldDataRetriever(sedtrails_data)

    initial_time = float(times[TIMESTEP_INDEX])
    print(f'Using {FLOW_FIELD_NAME!r} from t={initial_time:.0f}s')

    # ===== STEP 2: Build a short sequence of flow fields =====
    print('\n=== STEP 2: Retrieving the flow fields used for particle updates ===')
    flow_sequence = []
    for step in range(NUM_STEPS):
        current_time = initial_time + step * TIMESTEP
        flow_data = retriever.get_flow_field(current_time, FLOW_FIELD_NAME)
        flow_sequence.append(flow_data)
        print(f'Loaded flow field for step {step + 1:02d}: t={current_time:.0f}s')

    # ===== STEP 3: Update particles with both APIs =====
    print('\n=== STEP 3: Updating one particle with both APIs ===')
    class_x, class_y = run_particle_position_calculator(flow_sequence, START_X, START_Y)
    reusable_x, reusable_y = run_reusable_grid_calculator(flow_sequence, START_X, START_Y)

    print(f'ParticlePositionCalculator final position: ({class_x[-1]:.4f}, {class_y[-1]:.4f})')
    print(f'create_numba_particle_calculator final position: ({reusable_x[-1]:.4f}, {reusable_y[-1]:.4f})')
    print(f'Final position difference: {np.hypot(class_x[-1] - reusable_x[-1], class_y[-1] - reusable_y[-1]):.6g} m')

    # ===== STEP 4: Visualize both trajectories =====
    print('\n=== STEP 4: Plotting the two trajectories ===')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_flow = flow_sequence[-1]

    plot_particle_trajectory(
        final_flow,
        class_x,
        class_y,
        title='ParticlePositionCalculator trajectory',
        save_path=OUTPUT_DIR / 'particle_position_calculator.png',
    )
    plot_particle_trajectory(
        final_flow,
        reusable_x,
        reusable_y,
        title='create_numba_particle_calculator trajectory',
        trajectory_color='blue',
        save_path=OUTPUT_DIR / 'create_numba_particle_calculator.png',
    )

    comparison_path = OUTPUT_DIR / 'trajectory_comparison.png'
    fig, ax = plot_flow_field(final_flow, title='Current particle-update APIs')
    ax.plot(class_x, class_y, color='red', linewidth=2, label='ParticlePositionCalculator')
    ax.plot(reusable_x, reusable_y, color='blue', linestyle='--', linewidth=2, label='create_numba_particle_calculator')
    ax.legend(loc='upper right')
    fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison plot saved to {comparison_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
