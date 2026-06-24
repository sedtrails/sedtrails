# SedTRAILS Examples

This directory contains tracked example inputs and scripts for local testing and exploration.

## Simulation Configurations

- `sedtrails-example.yaml`: Basic D-Flow FM/van Westen simulation using randomly seeded sand populations.
- `sedtrails-example-multisource.yaml`: D-Flow FM/van Westen simulation using seed points from `sources_xy_inlet.txt`.
- `config.example_soulsby.yaml`: D-Flow FM/Soulsby simulation with two sand populations.
- `config.example_sfincs.yaml`: SFINCS/passive-tracer simulation.

The NetCDF forcing files referenced by these examples are not stored in this directory. Download or create the required files locally and update each configuration's `inputs.data` path before running.

## Python Examples

- `particle_simulation_example.py`: Shows the lower-level flow conversion, retrieval, particle update, and plotting workflow.
- `reading_flowfields_example.py`: Converts model input with `FormatConverter` and samples a flow field with `FieldDataRetriever`.
- `seeder_example.py`: Demonstrates `ParticleSeeder` with in-memory field coordinates.
- `particle_simulation_numba_example.py`: Teaches the two current particle-update APIs side by side on one converted flow field.
- `numba_example.py`: Demonstrates the reusable grid calculator on a synthetic grid.
- `visualization_utils.py`: Shared plotting helpers used by the examples.

## Data Files

- `sources_xy_inlet.txt`: Example x/y release points for `sedtrails-example-multisource.yaml`.
