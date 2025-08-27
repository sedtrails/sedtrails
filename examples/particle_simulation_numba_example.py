"""
SedTRAILS Performance Comparison: Original vs Numba-optimized ParticlePositionCalculator

This script compares performance between the original and Numba-optimized implementations:
1. Loading the same flow field data as the example
2. Running identical particle tracking simulation with both implementations
3. Measuring execution time differences
4. Checking calculation consistency
5. Visualizing and comparing the trajectories
"""

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import SedTRAILS modules
from sedtrails.transport_converter.format_converter import FormatConverter
from sedtrails.transport_converter.physics_converter import PhysicsConverter
from sedtrails.particle_tracer.data_retriever import FlowFieldDataRetriever
from sedtrails.particle_tracer.particle import Sand
from sedtrails.particle_tracer.position_calculator import ParticlePositionCalculator
from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator

# Import visualization utilities for plotting
from visualization_utils import plot_flow_field, plot_particle_trajectory

# ===== CONFIGURATION =====
# Adjust these parameters as needed
FILE_PATH = Path('../sample-data/inlet_sedtrails.nc')
TIMESTEP = 30.0  # seconds
OUTPUT_DIR = Path('../sample-data/output')

# Custom starting position
START_X = 40000.0
START_Y = 17000.0

# Start at the 3rd timestep (index 2)
TIMESTEP_INDEX = 2

# Duration: 6.333 hours = 6 hours and 20 minutes = 380 minutes = 22,800 seconds
DURATION_HOURS = 6.333
DURATION_SECONDS = DURATION_HOURS * 3600

# Calculate number of steps
NUM_STEPS = int(DURATION_SECONDS / TIMESTEP)

# ===== STEP 1: Load and Convert Flow Field Data =====
print('\n=== STEP 1: Loading and Converting Flow Field Data ===')
start_time = time.time()

# Convert to SedtrailsData format
format_config = {  # this should come from a config file or similar
    'input_file': FILE_PATH,
    'input_format': 'fm_netcdf',  # Specify the input format
    'reference_date': '1970-01-01',  # Default reference date
}

format_converter = FormatConverter(format_config)
sedtrails_data = format_converter.convert_to_sedtrails()
print(f'Data conversion completed in {time.time() - start_time:.2f} seconds')

# Add physics calculations to the SedtrailsData
physics_converter = PhysicsConverter()
physics_converter.convert_physics(sedtrails_data)

# ===== STEP 2: Initialize Particle and Position Calculator =====
print('\n=== STEP 2: Initializing Particle and Flow Field ===')

# Create flow field data retriever
retriever = FlowFieldDataRetriever(sedtrails_data)
retriever.flow_field_name = (
    'suspended_velocity'  # Options: "depth_avg_flow_velocity", "bed_load_velocity", "suspended_velocity"
)

# Get the initial flow field at specified timestep
initial_time = sedtrails_data.times[TIMESTEP_INDEX]
initial_flow = retriever.get_flow_field(initial_time)

# Create a particle at the specified position
particle = Sand(id=1, _x=START_X, _y=START_Y, name='Test Particle')
print(f'Created particle at position ({particle.x:.2f}, {particle.y:.2f})')
print(f'Starting time at index {TIMESTEP_INDEX}: {initial_time} seconds')
print(f'Will run for {DURATION_HOURS:.3f} hours ({NUM_STEPS} steps) with timestep {TIMESTEP} seconds')

# ===== STEP 3: Simulation with Original Implementation =====
print('\n=== STEP 3: Running Simulation with Original Implementation ===')

# Reset particle position
particle_orig = Sand(id=1, _x=START_X, _y=START_Y, name='Original Test Particle')

# Store trajectory
trajectory_orig_x = [particle_orig.x]
trajectory_orig_y = [particle_orig.y]

simulation_start = time.time()
current_time = initial_time

for step in range(1, NUM_STEPS + 1):
    # Update current time
    current_time = initial_time + step * TIMESTEP

    # Get flow field at current time
    flow_data = retriever.get_flow_field(current_time)

    # Create position calculator with current flow field
    calculator = ParticlePositionCalculator(
        grid_x=flow_data['x'], grid_y=flow_data['y'], grid_u=flow_data['u'], grid_v=flow_data['v']
    )

    # Update particle position
    new_x, new_y = calculator.update_particles(
        x0=np.array([particle_orig.x]), y0=np.array([particle_orig.y]), dt=TIMESTEP
    )

    # Update particle with new position
    particle_orig.x = new_x[0]
    particle_orig.y = new_y[0]

    # Store trajectory
    trajectory_orig_x.append(particle_orig.x)
    trajectory_orig_y.append(particle_orig.y)

    # Print progress every 20%
    if step % max(1, NUM_STEPS // 5) == 0:
        percent_complete = (step / NUM_STEPS) * 100
        elapsed_hours = step * TIMESTEP / 3600
        print(
            f'Step {step}/{NUM_STEPS} ({percent_complete:.1f}%) - '
            f'Time: {elapsed_hours:.2f} hours - '
            f'Position: ({particle_orig.x:.2f}, {particle_orig.y:.2f})'
        )

original_time = time.time() - simulation_start
print(f'Original implementation completed in {original_time:.4f} seconds')
print(f'Average speed: {NUM_STEPS / original_time:.1f} steps/second')
print(f'Final position: ({particle_orig.x:.4f}, {particle_orig.y:.4f})')

# ===== STEP 4: Simulation with Numba Implementation =====
print('\n=== STEP 4: Running Simulation with Numba Implementation ===')

# Reset particle position
particle_numba = Sand(id=2, _x=START_X, _y=START_Y, name='Numba Test Particle')

# Store trajectory
trajectory_numba_x = [particle_numba.x]
trajectory_numba_y = [particle_numba.y]

# First get the initial flow data to create calculator
flow_data = retriever.get_flow_field(initial_time)

# Create and compile the numba calculator - this will include compilation time
print('Creating and compiling Numba calculator...')
compile_start = time.time()
numba_calc = create_numba_particle_calculator(grid_x=flow_data['x'], grid_y=flow_data['y'])
compile_time = time.time() - compile_start
print(f'Numba calculator compiled in {compile_time:.4f} seconds')

# Warm up with one calculation to trigger JIT compilation
print('Warming up JIT compilation...')
warmup_start = time.time()
_ = numba_calc['update_particles'](
    np.array([particle_numba.x]), np.array([particle_numba.y]), flow_data['u'], flow_data['v'], TIMESTEP
)
warmup_time = time.time() - warmup_start
print(f'JIT warm-up completed in {warmup_time:.4f} seconds')

# Start timer after compilation
simulation_start = time.time()
current_time = initial_time

for step in range(1, NUM_STEPS + 1):
    # Update current time
    current_time = initial_time + step * TIMESTEP

    # Get flow field at current time
    flow_data = retriever.get_flow_field(current_time)

    # Update particle position using Numba calculator
    new_x, new_y = numba_calc['update_particles'](
        np.array([particle_numba.x]), np.array([particle_numba.y]), flow_data['u'], flow_data['v'], TIMESTEP
    )

    # Update particle with new position
    particle_numba.x = new_x[0]
    particle_numba.y = new_y[0]

    # Store trajectory
    trajectory_numba_x.append(particle_numba.x)
    trajectory_numba_y.append(particle_numba.y)

    # Print progress every 20%
    if step % max(1, NUM_STEPS // 5) == 0:
        percent_complete = (step / NUM_STEPS) * 100
        elapsed_hours = step * TIMESTEP / 3600
        print(
            f'Step {step}/{NUM_STEPS} ({percent_complete:.1f}%) - '
            f'Time: {elapsed_hours:.2f} hours - '
            f'Position: ({particle_numba.x:.2f}, {particle_numba.y:.2f})'
        )

numba_time = time.time() - simulation_start
print(f'Numba implementation completed in {numba_time:.4f} seconds')
print(f'Average speed: {NUM_STEPS / numba_time:.1f} steps/second')
print(f'Final position: ({particle_numba.x:.4f}, {particle_numba.y:.4f})')

# ===== STEP 5: Performance Comparison =====
print('\n=== STEP 5: Performance Comparison ===')

# Calculate speedup
speedup = original_time / numba_time
print('Execution time comparison:')
print(f'  Original: {original_time:.4f} seconds (wall clock)')
print(f'  Numba:    {numba_time:.4f} seconds (wall clock)')
print(f'  Speedup:  {speedup:.2f}x')
print(f'  Simulation time: {DURATION_HOURS:.3f} hours ({NUM_STEPS} timesteps)')

# Check result consistency
position_diff = np.sqrt((particle_orig.x - particle_numba.x) ** 2 + (particle_orig.y - particle_numba.y) ** 2)
print('\nResult comparison:')
print(f'  Position difference: {position_diff:.8f} meters')

# Calculate trajectory differences
trajectory_orig_x = np.array(trajectory_orig_x)
trajectory_orig_y = np.array(trajectory_orig_y)
trajectory_numba_x = np.array(trajectory_numba_x)
trajectory_numba_y = np.array(trajectory_numba_y)

trajectory_diffs = np.sqrt(
    (trajectory_orig_x - trajectory_numba_x) ** 2 + (trajectory_orig_y - trajectory_numba_y) ** 2
)

print(f'  Mean trajectory difference: {np.mean(trajectory_diffs):.8f} meters')
print(f'  Max trajectory difference:  {np.max(trajectory_diffs):.8f} meters')

# ===== STEP 6: Visualize Trajectories =====
print('\n=== STEP 6: Visualizing Trajectories ===')

# Get final flow field for visualization
final_flow = retriever.get_flow_field(current_time)

# Plot the original trajectory
orig_plot_path = OUTPUT_DIR / 'original_trajectory.png'
plot_particle_trajectory(
    flow_data=final_flow,
    trajectory_x=trajectory_orig_x,
    trajectory_y=trajectory_orig_y,
    title=f'Original Implementation - {DURATION_HOURS:.2f} hours, {NUM_STEPS} steps',
    save_path=orig_plot_path,
)
print(f'Original trajectory plot saved to {orig_plot_path}')

# Plot the Numba trajectory
numba_plot_path = OUTPUT_DIR / 'numba_trajectory.png'
plot_particle_trajectory(
    flow_data=final_flow,
    trajectory_x=trajectory_numba_x,
    trajectory_y=trajectory_numba_y,
    title=f'Numba Implementation - {DURATION_HOURS:.2f} hours, {NUM_STEPS} steps',
    trajectory_color='blue',
    save_path=numba_plot_path,
)
print(f'Numba trajectory plot saved to {numba_plot_path}')

# Plot comparison (both trajectories on the same flow field)
# Since the plot_particle_trajectory function only plots one trajectory,
# we'll create a custom comparison plot

# First create the flow field plot
comparison_plot_path = OUTPUT_DIR / 'trajectory_comparison.png'
fig, ax = plot_flow_field(
    flow_data=final_flow, title=f'Trajectory Comparison - {DURATION_HOURS:.2f} hours, {NUM_STEPS} steps', downsample=10
)

# Add original trajectory
ax.plot(trajectory_orig_x, trajectory_orig_y, '-', color='red', linewidth=2, label='Original Implementation')
ax.plot(trajectory_orig_x[0], trajectory_orig_y[0], 'go', markersize=8, label='Start Position')
ax.plot(trajectory_orig_x[-1], trajectory_orig_y[-1], 'ro', markersize=8, label='Original End Position')

# Add Numba trajectory
ax.plot(trajectory_numba_x, trajectory_numba_y, '-', color='blue', linewidth=2, label='Numba Implementation')
ax.plot(trajectory_numba_x[-1], trajectory_numba_y[-1], 'bo', markersize=8, label='Numba End Position')

# Add legend
ax.legend(loc='upper right')

# Save comparison plot
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Comparison plot saved to {comparison_plot_path}')

print('\nPerformance comparison and visualization completed!')
