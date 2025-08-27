"""
Comprehensive example of SedTRAILS functionality.

This script demonstrates the complete workflow for visualizing flow fields:
1. Loading and converting flow data from NetCDF files
2. Creating a flow field data retriever
3. Processing and visualizing flow fields at specified time steps
"""

from pathlib import Path
import matplotlib.pyplot as plt

# Import Sedtrails modules
from sedtrails.transport_converter.format_converter import FormatConverter, InputType
from sedtrails.particle_tracer.data_retriever import FlowFieldDataRetriever

# Import visualization utilities
from visualization_utils import plot_flow_field

# Set up the file path for the NetCDF file
file_path = Path("../sedtrails_data/inlet_sedtrails.nc")
print(f"Processing file: {file_path}")

# Number of time steps to process (default: 5)
num_timesteps = 10

# ===== STEP 1: Load and Convert NetCDF File =====
print("\n=== STEP 1: Loading and Converting Data ===")
converter = FormatConverter(file_path, input_type=InputType.NETCDF_DFM)
converter.read_data()
time_info = converter.get_time_info()
print(f"Time range: {time_info['time_start']} to {time_info['time_end']}, with total timestamps: {time_info['num_times']}")

# Convert to SedtrailsData
sedtrails_data = converter.convert_to_sedtrails_data()

# ===== STEP 2: Create Flow Field Data Retriever =====
print("\n=== STEP 2: Creating Flow Field Data Retriever ===")
retriever = FlowFieldDataRetriever(sedtrails_data)
print(f"Retriever created for flow field: {retriever.flow_field_name}")

# ===== STEP 3: Process Each Time Step =====
print("\n=== STEP 3: Processing Time Steps ===")
indices = [int(1 + i * (num_timesteps - 2) / (num_timesteps - 1)) for i in range(num_timesteps)]
for i, idx in enumerate(indices):

    current_time = sedtrails_data.times[idx]
    print(f"Time step {i+1}/{num_timesteps} - Time: {current_time:.2f} s")
    
    # Get flow field data for the current time
    flow_data = retriever.get_flow_field(current_time)
    
    # Create individual plot for current time step
    fig, ax = plot_flow_field(flow_data)
    fig.savefig(f"../sedtrails_data/flow_field_{i+1:03d}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to save memory

print("\nExample completed successfully!")