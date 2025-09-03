# SedTRAILS User Manual

## Hydrodynamic & Sediment Transport Input

### General

Before running SedTRAILS, determine the modeling goal: real-time simulation, gross/net transport, or specific timescales. Typically, SedTRAILS is run in cyclic mode using a representative tidal cycle, enabling long simulations with limited input. This avoids assigning specific timescales to transport patterns.

A "representative" tidal cycle should have nearly identical velocities at start and end to avoid artificial net transport. The selection depends on project needs and domain size. See Lambregts (2021) for details.

The cycle may come from a morphological tide (Latteux, 1995; Lesser, 2009) or a longer time series (Lambregts, 2021; Meijers, 2021).

### Delft3D-4

Similar setup to FM but with format differences.

### D-Flow FM

Ensure the `.mor` file has:

```
[Output]
AverageAtEachOutputTime= 1
```

To directly specify FM output for SedTRAILS, use a compiled FM version and add to `.mdu`:

```
[sedtrails]
SedtrailsGrid=sedtrails_net.nc
SedtrailsAnalysis=all
SedtrailsInterval=3600 3600 10200
SedtrailsOutputFile=
```

Output types:

* `all`: all SedTRAILS types
* `soulsby`: for Soulsby analysis
* `transport`: transport flux only
* `flowvelocity`: flow velocity only

## Sediment Transport Vector Field Computation

Scripts:

* `sedtrails_preprocess_d3d.m`: converts Delft3D-4 output
* `sedtrails_preprocess_dfm_unstruct.m`: processes FM netCDFs

Output files depend on analysis type (e.g., Soulsby, TRANSP, FLOWVEL). Resulting `.mat` files are used in SedTRAILS.

`sedtrails_reversifier.m` flips vectors for backward tracking.

## Particle Tracking

Specify particle sources using:

* `sedtrails_setup_source_clusters.m`: clusters bathymetry using k-means
* `sedtrails_setup_source_transects.m`: transect-based release
* `sedtrails_setup_source_ftle.m`: grid release for FTLE

To run:

* `sedtrails_run_batch.m`: main launcher script
* `sedtrails_io.m`: sets properties, calls tracker
* `sedtrails_particletracking_cyclic.m`: runs tracking loop
* `sedtrails_particles.m`: core RK4 integration

Output options include:

* Full trajectories
* Final/end positions
* Sticky depth simulation
* Particle density
* Diffusion metrics
* Fluxes across sections

## Post-processing

Use `sedtrails_plot_multiscenario_individual.m` for visual comparisons across conditions. Reads wave condition metadata and loops through output directories.

## Connectivity

Convert SedTRAILS output into connectivity matrices:

1. Define Voronoi polygons for source cells
2. Count particle destinations
3. Populate adjacency matrix

Use Brain Connectivity Toolbox (`connectivity/brainConnectivityToolbox`) for further analysis.

Scripts and examples in `connectivity/unorganized/`.
