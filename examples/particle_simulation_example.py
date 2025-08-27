"""
SedTRAILS Particle Simulation Example

This script demonstrates the complete workflow for particle tracking in a flow field:
1. Loading and converting flow data from NetCDF files
2. Creating and initializing particles
3. Tracking particle positions over time using the ParticlePositionCalculator
4. Visualizing results

The example uses a 30-second timestep and simulates for 6.333 hours.
"""

import time
from pathlib import Path
import numpy as np

# Import SedTRAILS modules
from sedtrails.transport_converter.format_converter import FormatConverter, InputType
from sedtrails.particle_tracer.data_retriever import FlowFieldDataRetriever
from sedtrails.particle_tracer.position_calculator import ParticlePositionCalculator
from sedtrails.particle_tracer.particle import Sand

# Import visualization utilities
from visualization_utils import plot_particle_trajectory

# ===== CONFIGURATION =====
# Specific parameters
FILE_PATH = Path("/Users/mgarciaalvarez/devel/sedtrails/sample-data/inlet_sedtrails.nc")
START_X = 40000.0
START_Y = 17000.0
TIMESTEP = 30.0  # seconds
TIMESTEP_INDEX = 2  # Start at the 3rd timestep (index 2)
DURATION_HOURS = 6.333
DURATION_SECONDS = DURATION_HOURS * 3600
OUTPUT_PATH = Path("../sedtrails_data/particle_trajectory.png")

# ===== STEP 1: Load and Convert Flow Field Data =====
print("\n=== STEP 1: Loading and Converting Flow Field Data ===")
start_time = time.time()

print(f"Processing file: {FILE_PATH}")

# Load and convert the data
converter = FormatConverter(FILE_PATH, input_type=InputType.NETCDF_DFM)
converter.read_data()
time_info = converter.get_time_info()
print(f"Time range: {time_info['time_start']} to {time_info['time_end']}")
print(f"Total timestamps: {time_info['num_times']}")

# Convert to SedtrailsData format
sedtrails_data = converter.convert_to_sedtrails_data()
print(f"Data conversion completed in {time.time() - start_time:.2f} seconds")

# ===== STEP 2: Initialize Particle and Position Calculator =====
print("\n=== STEP 2: Initializing Particle and Position Calculator ===")

# Create flow field data retriever
retriever = FlowFieldDataRetriever(sedtrails_data)

# Get the initial flow field at specified timestep
initial_time = sedtrails_data.times[TIMESTEP_INDEX]
initial_flow = retriever.get_flow_field(initial_time)

# Create a particle at the specified position
particle = Sand(id=1, _x=START_X, _y=START_Y, name="Test Particle")
print(f"Created particle at position ({particle.x:.2f}, {particle.y:.2f})")
print(f"Starting at timestep index {TIMESTEP_INDEX}: {initial_time} seconds")

# Initialize storage for particle trajectory
trajectory_x = [particle.x]
trajectory_y = [particle.y]
trajectory_times = [initial_time]

# ===== STEP 3: Simulation Loop =====
print("\n=== STEP 3: Running Particle Simulation ===")
simulation_start = time.time()

# Calculate number of simulation steps
num_steps = int(DURATION_SECONDS / TIMESTEP)
print(f"Simulation duration: {DURATION_HOURS:.3f} hours ({DURATION_SECONDS} seconds)")
print(f"Number of simulation steps: {num_steps}")

# Initialize progress tracking
progress_interval = max(1, num_steps // 20)  # Show progress ~20 times
last_progress_time = time.time()

current_time = initial_time
for step in range(1, num_steps + 1):
    # Update current time
    current_time = initial_time + step * TIMESTEP
    
    # Get flow field at current time
    flow_data = retriever.get_flow_field(current_time)
    
    # Create position calculator with current flow field
    calculator = ParticlePositionCalculator(
        grid_x=flow_data['x'],
        grid_y=flow_data['y'],
        grid_u=flow_data['u'],
        grid_v=flow_data['v']
    )
    
    # Update particle position
    new_x, new_y = calculator.update_particles(
        x0=np.array([particle.x]),
        y0=np.array([particle.y]),
        dt=TIMESTEP
    )
    
    # Update particle with new position
    particle.x = new_x[0]
    particle.y = new_y[0]
    
    # Store trajectory
    trajectory_x.append(particle.x)
    trajectory_y.append(particle.y)
    trajectory_times.append(current_time)
    
    # Print progress
    if step % progress_interval == 0 or step == num_steps:
        current_progress_time = time.time()
        elapsed = current_progress_time - last_progress_time
        steps_per_second = progress_interval / elapsed if elapsed > 0 else 0
        percent_complete = (step / num_steps) * 100
        elapsed_hours = step * TIMESTEP / 3600
        
        print(f"Step {step}/{num_steps} ({percent_complete:.1f}%) - " 
              f"Simulated time: {elapsed_hours:.2f} hours - "
              f"Position: ({particle.x:.2f}, {particle.y:.2f}) - "
              f"Speed: {steps_per_second:.1f} steps/s")
        
        last_progress_time = current_progress_time

simulation_time = time.time() - simulation_start
print(f"\nSimulation completed in {simulation_time:.2f} seconds")
print(f"Average speed: {num_steps / simulation_time:.1f} steps/second")

# ===== STEP 4: Visualize Results =====
print("\n=== STEP 4: Visualizing Results ===")

# Convert trajectory to numpy arrays
trajectory_x = np.array(trajectory_x)
trajectory_y = np.array(trajectory_y)

# Plot flow field with particle trajectory using the function
final_flow = retriever.get_flow_field(current_time)

plot_particle_trajectory(
    flow_data=final_flow,
    trajectory_x=trajectory_x,
    trajectory_y=trajectory_y,
    title=f'Particle Trajectory - {DURATION_HOURS:.2f} hours, {num_steps} steps',
    save_path=OUTPUT_PATH
)
print(f"Trajectory plot saved to {OUTPUT_PATH}")

# Show simulation statistics
print("\n=== Simulation Statistics ===")
total_distance = np.sum(np.sqrt(np.diff(trajectory_x)**2 + np.diff(trajectory_y)**2))
displacement = np.sqrt((trajectory_x[-1] - trajectory_x[0])**2 + (trajectory_y[-1] - trajectory_y[0])**2)
avg_velocity = total_distance / DURATION_SECONDS

print(f"Starting position: ({trajectory_x[0]:.2f}, {trajectory_y[0]:.2f})")
print(f"Final position: ({trajectory_x[-1]:.2f}, {trajectory_y[-1]:.2f})")
print(f"Total distance traveled: {total_distance:.2f} m")
print(f"Displacement from start: {displacement:.2f} m")
print(f"Average velocity: {avg_velocity:.2f} m/s")
print(f"Average velocity: {avg_velocity * 3.6:.2f} km/h")

print("\nParticle Simulation Example completed successfully!")