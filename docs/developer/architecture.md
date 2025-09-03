# SedTRAILS Architecture

## Component Diagrams

.. tip:: General format to describe components
  Each component has the following attributes
  - Purpose: what role the component plays
  - Function: what the component actually does


## Component Diagram For the Simulation Configuration Interface

### User Interaction

#### Modeler
- **Role**: End user who aims to track sand particles in a coastal system.
- **Function**: Provides simulation configurations via:
  - Terminal commands (CLI)
  - YAML configuration files

---

### Simulation Configuration Interface (container)
This container abstracts the workflow needed to prepare simulation configurations for SedTRAILS. It contains the following components:

#### Simulation CLI (component)
- **Technology**: `Python`, `Typer`
- **Purpose**: Presents the commands to the user via a Command Line Interface.
- **Function**:
  - Collects user input for simulation setup.
  - Passes configuration data and instructions to the **Configuration Controller**.

#### Configuration Controller (component)
- **Technology**: `Python`
- **Purpose**: Central logic handler for configuration (i.e., model input parameters) management.
- **Function**:
  - Reads and interprets YAML configuration files.
  - Applies default configuration values when needed.
  - Distributes complete configuration data to downstream components.

#### Validator (component)
- **Technology**: `YAML`
- **Purpose**: Validates input configuration files against the SedTRAILS schema.
- **Interaction**: Operates on file content provided by the **Configuration Controller**.

---

### External Components

These components receive configuration data from the Simulation Configuration Interface and are responsible for simulation execution and data transformation.

#### Lagrangian Particle Tracer (component)
- **Technology**: `Python`
- **Description**: Core simulation engine.
- **Function**: Computes particle positions \((x, y, t)\) in a coastal system using provided configuration.
- **Input**: Receives validated configurations from the **Configuration Controller**.

#### Transport Converter (component)
- **Technology**: `Python`
- **Description**: Data adapter for external flow-field datasets.
- **Function**:
  - Reads and converts 'transport flow–field data' into SedTRAILS-compatible formats and physical units.
  - Interfaces with the **Configuration Controller** to ensure compatibility with the rest of the pipeline.

---

### Data Flow Summary

1. **Modeler** initiates the simulation via CLI or configuration file.
2. **Simulation CLI** exposes commands and forwards user inputs.
3. **Configuration Controller**:
   - Reads and interprets configuration files.
   - Sends content to **Validator** for schema validation.
   - Distributes validated configurations (i.e., model parameter settings) to **Lagrangian Particle Tracer** and **Transport Converter**.
4. **Lagrangian Particle Tracer** and **Transport Converter** execute tasks based on configurations (i.e., model parameter settings).

## Component Diagram for Transport Converter

The **Transport Converter** is a core component of the SedTRAILS architecture that enables the use of external Eulerian flow modeling systems by transforming their output into SedTRAILS-compatible formats and physical units. It acts as an adapter layer that standardizes, configures, and converts transport flow–field data for use in sediment particle simulations.

---

### User Interaction

#### Modeler
- **Role**: End user who defines the simulation parameters and output configuration.
- **Function**: Provides a YAML configuration file that specifies source data and conversion preferences.

---

### Transport Converter (container)
This container provides the functionality for importing and converting external flow–field data. It includes the following components:

#### Format Converter Interface (component)
- **Technology**: `Python`
- **Purpose**: Reads flow–field data from external Eulerian modeling systems.
- **Function**:
  - Reads flow–field input (e.g., in `NetCDF` format).
  - Converts the custom format to the internal **SedTRAILS** standard data structure.
  - Provides data to the **Physics Converter**.

#### Physics Converter (component)
- **Technology**: `Python`
- **Purpose**: Transforms flow–field data into physically consistent units as configured by the user.
- **Function**:
  - Accesses user-specified conversion methods from the **Simulation Configuration Interface**.
  - Converts flow-field physical units using methods available in the **Physics Conversion Library**.
  - Provides converted transport flow-field data to **Lagrangian Particle Tracer** and **Data Management Module**.

#### Physics Conversion Library (component)
- **Technology**: `Python`
- **Purpose**: Provides a suite of conversion methods.
- **Function**:
  - Defines reusable functions for physical unit conversion.
  - Provides conversion methods to the **Physics Converter**.

---

### Connected Components

These components work alongside or consume output from the Transport Converter:

#### Simulation Configuration Interface (component)
- **Technology**: `Python`
- **Purpose**: Reads the user’s configuration file and provides setup parameters.
- **Function**: Provides configuration details to the **Physics Converter**.

#### Lagrangian Particle Tracer (component)
- **Technology**: `Python`
- **Purpose**: Executes the core simulation to compute particle trajectories.
- **Input**: Converted flow–field data for particle transport simulation from the **Physics Converter**.

#### Data Management Module (component)
- **Technology**: `Python`
- **Purpose**: Handles data and file I/O.
- **Function**:
  - Receives converted flow–field data.
  - Writes data to disk using the local file system in formats like `NetCDF`.

#### SedTRAILS Logger (component)
- **Technology**: `Python`
- **Purpose**: Logs simulation states, intermediate outputs, errors and performance metrics.
- **Function**: Tracks execution start/end times of routines used in the **Physics Converter**.

---

### External Systems

#### Eulerian Flow Modeling System (external system)
- **Technology**: `[Any]`
- **Description**: External system that simulates water/wind dynamics and outputs results in a custom structure.
- **Interaction**: Source of the flow–field input data to the **Format Converter Interface**.

#### Local File System (external system)
- **Technology**: `Windows`, `Linux`, `MacOS`
- **Purpose**: Stores converted flow–field data.
- **Interaction**: Used by the **Data Management Module** for reading/writing data in `NetCDF` or other supported formats.

---

### Data Flow Summary

1. **Modeler** defines simulation and conversion preferences via a YAML file.
2. **Simulation Configuration Interface** parses configuration and passes relevant parameters to the **Physics Converter**.
3. **Format Converter Interface** reads flow–field input data from an external Eulerian model (e.g., in NetCDF).
4. **Physics Converter**:
   - Receives the standard-formatted data.
   - Converts units using methods defined in the **Physics Conversion Library**.
   - Produces SedTRAILS-compatible transport flow–field data.
5. Converted data is used by:
   - **Lagrangian Particle Tracer** for simulation,
   - **Data Management Module** for storage,
   - **SedTRAILS Logger** for tracking computation metrics.

## Component Diagram for Lagrangian Particle Tracer

The **Lagrangian Particle Tracer** is the core engine of the SedTRAILS system. It computes particle trajectories in coastal environments based on time-varying flow fields and user-defined release conditions. It is built to support parallel particle tracing, modular physics, and comprehensive state tracking and logging.

---

### User Interaction

#### Modeler
- **Role**: End user who tracks particles in a coastal system.
- **Function**: Provides seeding configuration (e.g., time, location, number of particles) via a YAML configuration file.

---

### Parallel Lagrangian Particle Tracer (container)

This container handles particle release, position calculation, status updates, and property interpolation during the simulation. It supports parallel execution through modular orchestration and shared data buffers.

---

### Components

#### Particle Seeding Tool
- **Technology**: `Python`
- **Purpose**: Defines particle release strategies and initial conditions.
- **Function**:
  - Computes initial particle positions.
  - Applies various particle release strategies based on user configuration.

#### Flow Field Data Retriever
- **Technology**: `Python`
- **Purpose**: Loads flow–field input data required for particle tracing.
- **Function**:
  - Retrieves converted flow–field data from the **Transport Converter** or the **Simulation Cache and Recovery** module.

#### Processing Time Interface
- **Technology**: `Python`
- **Purpose**: Controls simulation time loop and scheduling.
- **Function**: Provides access to current time step and manages time iteration during the simulation.

#### Particle Result Collector
- **Technology**: `Python`
- **Purpose**: Stores final particle positions and states.
- **Function**: Collects final outputs at the end of the simulation run.

#### Particle Tracing Orchestrator
- **Technology**: `Python`
- **Purpose**: Manages parallel execution of particle tracing processes.
- **Function**:
  - Assigns a worker to each particle set.
  - Coordinates execution using multiprocessing.

#### Flow Field Data Buffer
- **Technology**: `Python`
- **Purpose**: Provides fast access to time-sliced flow–field data.
- **Function**:
  - Stores flow–field input for each tracer during a time step.
  - Interfaces with the **Flow Field Data Retriever**.

#### Particle Status Checker
- **Technology**: `Python`
- **Purpose**: Updates particle status and applies rules to determine end-of-life or availability for transport.
- **Function**:
  - Uses velocity or domain thresholds to identify particles that are inactive or buried.
  - Retrieves methods from the **Particle Status Library**.

#### Particle Status Library
- **Technology**: `Python`
- **Purpose**: Houses logic for particle termination or changes in availability for transport.
- **Function**: Provides reusable methods to check depth, burial, and exit conditions.

#### Particle Position Calculator
- **Technology**: `Numba`, `Python`
- **Purpose**: Computes new particle positions at each time step.
- **Function**:
  - Uses numerical solvers like Runge-Kutta 4.
  - Accesses interpolated velocity from the **Flow Field Data Buffer**.

#### Particle Property Interpolator
- **Technology**: `Python`
- **Purpose**: Interpolates scalar environmental properties (e.g., salinity, depth, bed level).
- **Function**: Updates each particle with values of scalar fields based on position.

#### Diffusion Library
- **Technology**: `Python`
- **Purpose**: Implements diffusion behavior in particle tracking.
- **Function**: Provides diffusion calculations based on stochastic or deterministic rules (e.g., random walk).

#### Particle Pathway Interface
- **Technology**: `Python`
- **Purpose**: Gives access to particle histories.
- **Function**: Collects positions, properties, and (if requested) full time series.

---

### Connected Components

#### Simulation Configuration Interface
- **Technology**: `Python`
- **Purpose**: Provides model parameter settings.
- **Function**: Supplies physical parameters, seeding information, numerical methods, and time settings.

#### Transport Converter
- **Technology**: `Python`
- **Purpose**: Supplies processed flow–field data (e.g., sediment particle velocities).
- **Function**: Feeds converted transport field data into the **Flow Field Data Retriever**.

#### Simulation Cache and Recovery
- **Technology**: `Python`
- **Purpose**: Supports recovery and resume of simulations.
- **Function**: Provides previously cached seeding and simulation inputs when restarting.

#### SedTRAILS Logger
- **Technology**: `Python`
- **Purpose**: Logs simulation events and timings.
- **Function**: Tracks start/end time of all major simulation components and subprocesses.

#### Data Management Module
- **Technology**: `Python`
- **Purpose**: Handles simulation output persistence.
- **Function**: Stores output files (e.g., final particle positions) to the local file system.

---

### External Systems

#### Local File System
- **Technology**: `Windows`, `Linux`, `MacOS`
- **Purpose**: Storage backend.
- **Function**: Used by the **Data Management Module** to write simulation output and intermediate files (e.g., `NetCDF`, `.csv`).

---

### Data Flow Summary

1. **Modeler** defines particle seeding strategy and model configuration via YAML.
2. **Simulation Configuration Interface** parses settings and passes them to the Lagrangian Particle Tracer.
3. **Flow Field Data Retriever** loads processed flow–field input from the **Transport Converter**.
4. **Particle Seeding Tool** releases particles using specified strategy.
5. **Particle Tracing Orchestrator** distributes the simulation task across processors.
6. At each time step:
   - **Flow Field Data Buffer** provides velocity input.
   - **Particle Position Calculator** updates particle locations.
   - **Particle Status Checker** evaluates burial/death criteria.
   - **Particle Property Interpolator** enriches particles with field values.
   - **Diffusion Library** applies diffusion if configured.
7. **Particle Result Collector** aggregates final results.
8. **Data Management Module** stores simulation outputs.
9. **SedTRAILS Logger** records runtime information and diagnostics.
10. Optionally, **Simulation Cache and Recovery** enables resuming interrupted runs.

