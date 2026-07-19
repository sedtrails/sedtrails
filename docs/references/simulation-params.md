# Simulation Parameters

This document provides a complete reference for all parameters that can be configured in a SedTRAILS simulation. Parameters are defined in YAML configuration files and control everything from the simulation domain to particle behavior and output settings.

:::warning
In the current version, not all parameters may be fully implemented. Please refer to the latest release notes for updates on supported features.
:::

## Table of Contents

1. [General Settings](#general-settings)
2. [Input Configuration](#input-configuration)
3. [Domain Definition](#domain-definition)
4. [Time Configuration](#time-configuration)
5. [Physics Parameters](#physics-parameters)
6. [Particle Populations](#particle-populations)
7. [Output Settings](#output-settings)
8. [Visualization](#visualization)

---

(general-settings)=
## General Settings

Controls basic simulation behavior and model configuration.

| Parameter                | Type    | Required | Default | Description                                                                                          |
| ------------------------ | ------- | -------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `preprocess`             | boolean | Optional | `true`  | Enable preprocessing of input data. Set to `false` to use existing processed files.                  |
| `input_model`            | object  | Optional | -       | Configuration for the input flow model. See [Input Model Configuration](#input-model-configuration). |
| `n_runs`                 | integer | Optional | `1`     | Number of simulation runs to execute.                                                                |
| `display_input_metadata` | boolean | Optional | `false` | Display all metadata from input files during loading.                                                |
| `report_domain_exit_updates` | boolean | Optional | `false` | Log per-timestep messages when particles newly leave the domain or beach on land. Final totals remain controlled by the CLI domain-exit reporting option. |
| `numerical_scheme`       | string  | Optional | `rk4`   | Numerical integration method. Options: `rk4` (Runge-Kutta 4th order), `euler` (Euler method).        |

(input-model-configuration)=
### Input Model Configuration

Nested under `general.input_model`:

When configuration defaults are applied, `general.input_model` is created if it
is omitted and populated with the nested defaults below.

| Parameter        | Type   | Required | Default      | Description                                                                                                          |
| ---------------- | ------ | -------- | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `format`         | string | Optional | `fm_netcdf`  | Input model format. Options: `fm_netcdf` (D-Flow FM), `xbeach` (XBeach), `sfincs` (SFINCS).                          |
| `reference_date` | string | Optional | `1970-01-01` | Reference date for time series in input data. Accepted formats include `YYYY-MM-DD` and `YYYY-MM-DD HH:MM:SS`. Used as the time origin for simulation and particle `release_start`. |
| `morfac`         | number | Optional | `1`          | Morphological acceleration factor for time decompression. Value of 1 means no acceleration.                          |

**Example:**

```yaml
general:
  preprocess: true
  report_domain_exit_updates: false
  numerical_scheme: rk4
  input_model:
    format: fm_netcdf
    reference_date: "2020-01-01 00:00:00"
    morfac: 1.0
```

---

(input-configuration)=
## Input Configuration

Specifies paths to input data and reading parameters.

| Parameter                 | Type    | Required     | Default       | Description                                                                                                                                   |
| ------------------------- | ------- | ------------ | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`                    | string  | **Required** | -             | Path to the flow field data file (e.g., D-Flow FM NetCDF output).                                                                             |
| `read_interval`           | string  | Optional     | `30D12H25M0S` | Time chunk size for reading input data. Format: `DDdHHhMMmSSs` (e.g., `30D` for 30 days, `12H` for 12 hours).                                 |
| `repeat_eulerian_fields`  | boolean | Optional     | `false`       | Repeat Eulerian flow fields only after a valid simulation start inside the forcing window. If `false`, reuse the final fields after the forcing period is exhausted. |
| `comp_dir`                | string  | Optional     | -             | Path to directory containing complementary validation data.                                                                                   |

**Example:**

```yaml
inputs:
  data: /path/to/flow_model_output.nc
  read_interval: "15D"
  repeat_eulerian_fields: true
  comp_dir: /path/to/validation_data/
```

---

(domain-definition)=
## Domain Definition

Defines optional custom domain controls. If the `domain` section is omitted, SedTRAILS uses the active grid from the input model without extra cutouts or boundary-class overrides.

If the `domain` section is present, you must specify **exactly one** of the following methods to define the active extent:

⚠️ **Mutually Exclusive Options**: Choose only ONE method from the following:
- **Method 1**: `pol_file` - Use a polygon file
- **Method 2**: `subset_x` and `subset_y` - Use coordinate ranges

Do not include an empty `domain: {}` block. `inner_boundary_pol_files` and `boundary_class_pol_files` do not define the simulation extent by themselves. They can be used with either domain method above. This means `pol_file` can be omitted when `subset_x` and `subset_y` are present, and boundary classes will still be applied. If no custom extent, inner boundaries, or boundary classes are needed, omit the `domain` section entirely.

| Parameter                  | Type    | Required     | Default | Description                                                                                               |
| -------------------------- | ------- | ------------ | ------- | --------------------------------------------------------------------------------------------------------- |
| `pol_file`                 | string  | Conditional* | -       | Path to Deltares `.pol` file containing domain boundary polygon.                                          |
| `subset_x`                 | string  | Conditional* | -       | X-coordinate range as `min:max` (e.g., `35000:52000`).                                                    |
| `subset_y`                 | string  | Conditional* | -       | Y-coordinate range as `min:max` (e.g., `12000:150000`).                                                   |
| `inner_boundary_pol_files` | array   | Optional     | `[]`    | Tekal `.pol` files with island or cutout polygons to remove from the active particle-tracking mesh.       |
| `boundary_class_pol_files` | object  | Optional     | `{}`    | User override Tekal `.pol` files that classify active boundary edges as `open` or `land`.                 |
| `flow_field_data`          | object  | Optional     | -       | Flow field format-specific settings. See [Flow Field Data Configuration](#flow-field-data-configuration). |

*Conditional: if `domain` is present, one extent method must be specified. Use either `pol_file` or both `subset_x` and `subset_y`; omit the entire `domain` section when no custom domain controls are needed.

### Inner Boundaries and Boundary Actions

For FM and SFINCS inputs, SedTRAILS can use Tekal polygon files to mask islands/cutouts. For FM, SFINCS, and XBeach inputs, SedTRAILS can use Tekal polygon files to distinguish open offshore boundaries from land boundaries.

`inner_boundary_pol_files` removes candidate faces or triangles whose centroids fall inside any configured polygon. This creates holes in the active particle mesh. Particles inside those holes are not treated as valid in-domain particles.

`boundary_class_pol_files` classifies active mesh boundary edges. Each class can point to one or more Tekal `.pol` files, and each file may contain multiple polygon blocks. Boundary edge classification uses the midpoint of each active boundary edge:

- `open`: particles crossing this edge are marked as having left the model domain and are removed from later movement calculations.
- `land`: particles crossing this edge are marked as beached for that timestep, remain at their last valid in-domain position, and can become mobile again on a later timestep if hydrodynamic and transport conditions permit.

If an edge midpoint is selected by both `open` and `land` override polygons, `land` takes priority. The source match is still kept in diagnostics.

Boundary-class polygons therefore do not need to be thin lines that exactly trace the boundary. A wider swath is allowed as long as it selects only the intended boundary-edge midpoints. Avoid polygons that are so wide they also contain midpoints from neighboring or unrelated open/land edges.

**Example:**

```yaml
domain:
  pol_file: ./outer_domain.pol
  inner_boundary_pol_files:
    - ./islands.pol
    - ./harbour_cutouts.pol
  boundary_class_pol_files:
    open:
      - ./offshore_open_edges.pol
      - ./lateral_open_edges.pol
    land:
      - ./coastline_edges.pol
      - ./island_edges.pol
```

The same boundary-class configuration can be used with a rectangular subset instead of `pol_file`:

```yaml
domain:
  subset_x: "35000:65000"
  subset_y: "12000:45000"
  boundary_class_pol_files:
    open:
      - ./offshore_open_edges.pol
    land:
      - ./coastline_edges.pol
```

Relative polygon paths are resolved relative to the YAML configuration file.

The converted flow field metadata stores diagnostics under:

- `inner_boundary_pol_files`
- `inner_boundary_polygon_count`
- `inner_boundary_masked_face_count`
- `inner_boundary_active_face_count`
- `boundary_edge_classification`

The `boundary_edge_classification` metadata contains edge node ids, edge midpoints, assigned edge classes, source matches, polygon counts, and class counts.

(flow-field-data-configuration)=
### Flow Field Data Configuration

Nested under `domain.flow_field_data`. This section currently exposes D-Flow FM-specific settings.

#### D-Flow FM Settings (`fm`)

| Parameter | Type   | Required    | Default        | Description                                                             |
| --------- | ------ | ----------- | -------------- | ----------------------------------------------------------------------- |
| `fm`      | string | Conditional | `sedtrails_nc` | Format of D-Flow FM output files. Options: `sedtrails_nc`, `merged_nc`. |

**Example:**

```yaml
domain:
  subset_x: "35000:52000"
  subset_y: "12000:150000"
  flow_field_data:
    fm: sedtrails_nc
```

---

(time-configuration)=
## Time Configuration

Controls simulation timing and time-stepping.

| Parameter       | Type   | Required     | Default      | Description                                                                                                    |
| --------------- | ------ | ------------ | ------------ | -------------------------------------------------------------------------------------------------------------- |
| `start`         | string | Optional     | -            | Start time for simulation (format: `YYYY-MM-DD HH:MM:SS`). If omitted, starts at beginning of flow field data. |
| `duration`      | string | Optional     | `0D12H25M0S` | Total simulation duration (format: `DDdHHhMMmSSs`). If omitted, runs until end of flow field data.             |
| `timestep`      | string | **Required** | `30S`        | Base time step for particle tracking (format: `DDdHHhMMmSSs`). Example: `30S` for 30 seconds.                  |
| `cfl_condition` | number | Optional     | `0.7`        | CFL condition for adaptive time-stepping (0-1). Set to `0` to disable adaptive time-stepping.                  |

**Duration Format Examples:**
- `30S` - 30 seconds
- `5M` - 5 minutes
- `2H` - 2 hours
- `1D` - 1 day
- `1D12H30M` - 1 day, 12 hours, 30 minutes

**Example:**

```yaml
time:
  start: "2020-01-01 00:00:00"
  duration: "30D"
  timestep: "30S"
  cfl_condition: 0.7
```

---

(physics-parameters)=
## Physics Parameters

Defines physical constants and sediment transport parameters.

### Physical Constants

Nested under `physics.constants`:

| Parameter               | Type   | Required | Default   | Description                                                                      |
| ----------------------- | ------ | -------- | --------- | -------------------------------------------------------------------------------- |
| `grain_diameter`        | number | Optional | `2.5e-4`  | Representative grain diameter [m]. Default is 0.25 mm (fine sand).               |
| `morphology_factor`     | number | Optional | `1.0`     | Morphological acceleration factor. Value of 1 means no acceleration.             |
| `porosity`              | number | Optional | `0.4`     | Sediment bed porosity (0-1). Default is 40%.                                     |
| `g`                     | number | Optional | `9.81`    | Gravitational acceleration [m/s²].                                               |
| `von_karman`            | number | Optional | `0.4`     | Von Kármán's constant for turbulent flow.                                        |
| `kinematic_viscosity`   | number | Optional | `1.36e-6` | Kinematic viscosity of water [m²/s]. Default valid for 10°C and 35 ppt salinity. |
| `rho_w`                 | number | Optional | `1027.0`  | Water density [kg/m³]. Default valid for 10°C and 35 ppt salinity.               |
| `rho_s`                 | number | Optional | `2650.0`  | Sediment particle density [kg/m³]. Default is quartz density.                    |
| `friction_angle`        | number | Optional | `30.0`    | Friction angle of sediment [degrees].                                            |
| `diffusion_coefficient` | number | Optional | `0.1`     | Random walk diffusion coefficient (horizontal diffusivity $K_h$) for turbulent dispersion. |

### Bed Shear Stress

Nested under `physics.bed_shear_stress`:

| Parameter  | Type   | Required | Default | Description                                                                                         |
| ---------- | ------ | -------- | ------- | --------------------------------------------------------------------------------------------------- |
| `mod_file` | string | Optional | -       | Path to file containing additional bed shear stress time series (spatially constant, time-varying). |

### Bed Slope Effects

Nested under `physics.bed_slope`:

| Parameter               | Type    | Required | Default | Description                                               |
| ----------------------- | ------- | -------- | ------- | --------------------------------------------------------- |
| `calculate`             | boolean | Optional | `false` | Calculate local bed slope in x and y directions.          |
| `resolution`            | number  | Optional | `100.0` | Resolution [m] for bed slope calculation grid.            |
| `bedslope_check_plot`   | boolean | Optional | `false` | Create diagnostic plot for bed slopes.                    |
| `bedslope_caxis_factor` | number  | Optional | `100.0` | Color axis scaling factor for bed slope plots (1/factor). |
| `bedslope_alpha_bs`     | number  | Optional | `1.0`   | Longitudinal bed slope transport effect tuning factor.    |
| `bedslope_alpha_bn`     | number  | Optional | `1.5`   | Transverse bed slope transport effect tuning factor.      |

**Example:**

```yaml
physics:
  constants:
    grain_diameter: 2.5e-4
    rho_s: 2650.0
    porosity: 0.4
  bed_slope:
    calculate: true
    resolution: 50.0
    bedslope_alpha_bs: 1.0
    bedslope_alpha_bn: 1.5
```

---

(particle-populations)=
## Particle Populations

Defines one or more groups of particles with different characteristics. At least one population is required.

The `particles` section contains an array of `populations`, where each population has the following structure:

### Population Definition

| Parameter               | Type   | Required     | Default          | Description                                                                                                |
| ----------------------- | ------ | ------------ | ---------------- | ---------------------------------------------------------------------------------------------------------- |
| `name`                  | string | **Required** | `particle`       | Unique name for this population (e.g., `sediment-fine`, `sand-01`).                                        |
| `particle_type`         | string | **Required** | `passive`        | Type of particle. Options: `passive`, `sand`, `mud`.                                                       |
| `characteristics`       | object | **Required** | -                | Type-specific particle properties. See [Particle Characteristics](#particle-characteristics).              |
| `diffusion`             | object | Optional     | `brownian`, `0.0` | Method, coefficient, and optional seed for horizontal diffusion. See [Per-population diffusion](#per-population-diffusion). |
| `tracer_methods`        | object | **Required** | -                | Transport calculation method(s). See [Tracer Methods](#tracer-methods).                                    |
| `transport_probability` | string | Optional     | `no_probability` | How to apply transport probability. Options: `no_probability`, `stochastic_transport`, `reduced_velocity`. |
| `seeding`               | object | **Required** | -                | Particle release configuration. See [Particle Seeding](#particle-seeding).                                 |
| `barriers`              | object | Optional     | -                | Thin dams or permeable structures. See [Barriers](#barriers).                                              |

(particle-characteristics)=
### Particle Characteristics

The `characteristics` object varies by `particle_type`:

#### Passive Particles

| Parameter               | Type   | Required     | Default | Description                        |
| ----------------------- | ------ | ------------ | ------- | ---------------------------------- |
| `diffusion_coefficient` | number | Optional (legacy) | `0.0` | Legacy fallback for the top-level `diffusion.coefficient`; prefer the top-level configuration. |

#### Per-population diffusion

The optional `diffusion` object applies to every particle type and tracer method. `method` is `brownian` (default) or `none`; `coefficient` is a non-negative horizontal diffusivity in `m^2/s` and defaults to `0.0`; `seed` is an optional integer that makes draws reproducible for that population. A zero coefficient and `method: none` both disable diffusion.

```yaml
diffusion:
  method: brownian
  coefficient: 0.05
  seed: 1234
```

#### Sand Particles

| Parameter    | Type   | Required     | Default  | Description                            |
| ------------ | ------ | ------------ | -------- | -------------------------------------- |
| `density`    | number | **Required** | `2650.0` | Particle density [kg/m³].              |
| `grain_size` | number | **Required** | `0.0005` | Grain diameter [m]. Default is 0.5 mm. |

#### Mud Particles

| Parameter | Type   | Required     | Default   | Description                                     |
| --------- | ------ | ------------ | --------- | ----------------------------------------------- |
| `density` | number | **Required** | `2000.0`  | Particle density [kg/m³].                       |
| `size`    | number | **Required** | `0.00005` | Grain diameter [m]. Default is 0.05 mm (50 μm). |

(tracer-methods)=
### Tracer Methods

Exactly one tracer method must be specified per population. Supported method keys are `vanwesten`, `soulsby`, and `passive_tracer`.

For `passive_tracer` populations:

- `particle_type` must be `passive`.
- `transport_probability` must be `no_probability`.
- `seeding.burial_depth` is not supported and must be omitted.

#### Van Westen Method

| Parameter         | Type   | Required | Default | Description                                                                           |
| ----------------- | ------ | -------- | ------- | ------------------------------------------------------------------------------------- |
| `flow_field_name` | array  | **Required** | -       | List of flow field names to use (e.g., `["bed_load_velocity", "suspended_velocity"]`). |
| `beta`            | number | Optional | `0.2`   | Beta parameter for Van Westen formulation.                                            |

#### Soulsby Method

| Parameter                 | Type   | Required     | Default  | Description                                                      |
| ------------------------- | ------ | ------------ | -------- | ---------------------------------------------------------------- |
| `flow_field_name`         | array  | **Required** | -        | List of flow field names to use, commonly `["grain_velocity"]`.  |
| `tracer_grain_size`       | number | Optional     | `0.0002` | Grain size of tracer sediment [m].                               |
| `background_grain_size`   | number | Optional     | `0.0002` | Grain size of background sediment [m].                           |
| `soulsby_b_e`             | number | Optional     | `1.7e-7` | Maximum free-to-trapped transition probability per second [1/s]. |
| `soulsby_theta_s`         | number | Optional     | `0.1`    | Transition scale value [-].                                      |
| `soulsby_gamma_e`         | number | Optional     | `0.1`    | Long-term equilibrium proportion of free particles [-].          |
| `soulsby_mu_d`            | number | Optional     | `0.5`    | Dynamic friction coefficient [-].                                |
| `soulsby_freedom_factor`  | number | Optional     | `1`      | Initial freedom factor (`0` = trapped, `1` = free).              |

#### Passive Tracer Method

| Parameter         | Type  | Required | Default                       | Description                                           |
| ----------------- | ----- | -------- | ----------------------------- | ----------------------------------------------------- |
| `flow_field_name` | array | Optional | `["depth_avg_flow_velocity"]` | List of flow field names to use for passive tracers. |

Sediment fraction selection is handled separately via `sediment_fraction_index` or `sediment_fraction_name`. For multi-fraction Delft3D-4 NetCDF inputs, named selection is resolved against the `NAMCON` labels in the file. If the input only contains a single sediment fraction, NAMCON is not used for fraction selection. Name-based fraction selection is not yet available for FM or XBeach inputs, because their transport variable names differ and need separate support.
A selection in `general.input_model` is the default for populations that do not set either field. A population-level selection replaces that default entirely, and `sediment_fraction_name` takes precedence when both fields are set in the same scope.

(particle-seeding)=
### Particle Seeding

Controls where, when, and how particles are released.

| Parameter       | Type    | Required     | Default         | Description                                                                                   |
| --------------- | ------- | ------------ | --------------- | --------------------------------------------------------------------------------------------- |
| `quantity`      | number  | **Required** | `1`             | Number of particles to release per release location.                                          |
| `class`         | string  | Optional     | `clusters`      | Seeding class. Options: `clusters`, `pointSource`, `lineSource`, `regularGrid`, `densePatch`. |
| `per_timestep`  | boolean | Optional     | `false`         | Release particles every time step (continuous release).                                       |
| `release_type`  | string  | Optional     | `instantaneous` | Release timing. Options: `instantaneous`, `continuous`.                                       |
| `lifespan`      | number  | Optional     | `9e+99`         | Maximum particle lifetime [seconds]. Use very large value for unlimited.                      |
| `release_start` | string  | Optional     | simulation start | Release start time for the population (format: `YYYY-MM-DD HH:MM:SS`). Converted to seconds relative to `general.input_model.reference_date`. |
| `release_stop`  | string  | Optional     | -               | Release stop time for continuous release. Defaults to immediate stop after first release.     |
| `burial_depth`  | object  | Optional     | -               | Initial burial depth configuration (not allowed for `passive_tracer`). See [Burial Depth](#burial-depth). |
| `strategy`      | object  | **Required** | -               | Spatial release strategy. See [Release Strategies](#release-strategies).                      |

(burial-depth)=
#### Burial Depth

⚠️ **Mutually Exclusive**: Choose either `random` OR `constant`.

| Parameter  | Type   | Required    | Default | Description                                                 |
| ---------- | ------ | ----------- | ------- | ----------------------------------------------------------- |
| `random`   | number | Conditional | `1.0`   | Maximum depth [m] for random burial depth from bed surface. |
| `constant` | number | Conditional | `1.0`   | Fixed burial depth [m] below bed surface.                   |

(release-strategies)=
#### Release Strategies

Choose **ONE** strategy. Each strategy has different required parameters.

For random and grid release areas, use `poly` for polygon files or inline polygon vertices, and `bbox` for rectangular areas.

##### Point Release

Releases particles at specific point locations.

⚠️ **Mutually Exclusive**: Specify either `pol_file` OR `locations`.

| Parameter          | Type    | Required    | Default | Description                                                    |
| ------------------ | ------- | ----------- | ------- | -------------------------------------------------------------- |
| `pol_file`         | string  | Conditional | -       | Path to `.pol` file with release point locations.              |
| `locations`        | array   | Conditional | -       | List of coordinate pairs (e.g., `["1000,2000", "3000,4000"]`). |
| `show_check_plots` | boolean | Optional    | `false` | Display diagnostic plots of release locations.                 |
| `save_check_plots` | boolean | Optional    | `false` | Save diagnostic plots to files.                                |

##### Transect Release

Releases particles along line segments (transects).

⚠️ **Mutually Exclusive**: Specify either `pol_file` OR `segments`.

| Parameter          | Type    | Required    | Default | Description                                              |
| ------------------ | ------- | ----------- | ------- | -------------------------------------------------------- |
| `pol_file`         | string  | Conditional | -       | Path to `.pol` file with transect line segments.         |
| `segments`         | array   | Conditional | -       | List of line segments (e.g., `["1000,2000 3000,4000"]`). |
| `k`                | number  | Optional    | `100.0` | Number of release points to create along each segment.   |
| `show_check_plots` | boolean | Optional    | `false` | Display diagnostic plots.                                |
| `save_check_plots` | boolean | Optional    | `false` | Save diagnostic plots.                                   |

##### Random Release

Releases particles at random locations within a rectangular or polygonal area.

⚠️ **Area Required**: Specify `poly` or `bbox`. For polygon files, use `poly: ./release_area.pol`. If both `poly` and `bbox` are supplied, `poly` takes precedence.

| Parameter          | Type            | Required     | Default | Description                                                                                         |
| ------------------ | --------------- | ------------ | ------- | --------------------------------------------------------------------------------------------------- |
| `poly`             | string or array | Conditional  | -       | Polygon boundary as a path to a `.pol`, CSV, or text file, or an inline list of coordinate strings. |
| `bbox`             | string          | Conditional  | -       | Bounding box as `xmin,ymin xmax,ymax`.                                                              |
| `nlocations`       | integer         | **Required** | `1`     | Number of random points to generate.                                                                |
| `seed`             | number          | Optional     | `42`    | Random number generator seed for reproducibility.                                                   |
| `show_check_plots` | boolean         | Optional     | `false` | Display diagnostic plots.                                                                           |
| `save_check_plots` | boolean         | Optional     | `false` | Save diagnostic plots.                                                                              |

##### Grid Release

Releases particles on a regular grid within a rectangular or polygonal area.

⚠️ **Area Required**: Specify `poly` or `bbox`. For polygon files, use `poly: ./release_area.pol`. If both `poly` and `bbox` are supplied, `poly` takes precedence.

| Parameter          | Type            | Required     | Default | Description                                                                                         |
| ------------------ | --------------- | ------------ | ------- | --------------------------------------------------------------------------------------------------- |
| `poly`             | string or array | Conditional  | -       | Polygon boundary as a path to a `.pol`, CSV, or text file, or an inline list of coordinate strings. |
| `bbox`             | string          | Conditional  | -       | Bounding box as `xmin,ymin xmax,ymax`.                                                              |
| `separation.dx`    | number          | **Required** | `100.0` | Grid spacing in x-direction [m].                                                                    |
| `separation.dy`    | number          | **Required** | `100.0` | Grid spacing in y-direction [m].                                                                    |
| `show_check_plots` | boolean         | Optional     | `false` | Display diagnostic plots.                                                                           |
| `save_check_plots` | boolean         | Optional     | `false` | Save diagnostic plots.                                                                              |

##### File Points Release

Releases particles at an arbitrary set of points specified in a text/CSV file.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | Yes | - | Path to the text/CSV/TSV file containing x,y coordinates |
| `x_col` | integer or string | No | - | Column index (0-based) or column name for x-coordinates |
| `y_col` | integer or string | No | - | Column index (0-based) or column name for y-coordinates |
| `has_header` | boolean | No | `true` | Whether the file contains a header row |
| `deduplicate` | boolean | No | `true` | Remove duplicate coordinate pairs |
| `dropna` | boolean | No | `true` | Remove rows with missing x or y values |
| `stride` | integer | No | `1` | Keep every Nth point (useful for subsampling dense datasets) |
| `bbox` | string or object | No | - | Optional bounding box to filter points |

(barriers)=
### Barriers

| Parameter | Type   | Required | Default | Description                                                       |
| --------- | ------ | -------- | ------- | ----------------------------------------------------------------- |
| `po_file` | string | Optional | -       | Path to `.pol` file containing thin dams or permeable structures. |

### Population Example

```yaml
particles:
  populations:
    - name: fine-sand
      particle_type: sand
      characteristics:
        density: 2650.0
        grain_size: 0.00025  # 0.25 mm
      tracer_methods:
        vanwesten:
          flow_field_name: ["bed_load_velocity"]
          beta: 0.2
      transport_probability: reduced_velocity
      seeding:
        quantity: 100
        release_type: instantaneous
        burial_depth:
          constant: 0.0
        strategy:
          point:
            locations:
              - "50000,25000"
              - "51000,26000"

    - name: coarse-sand
      particle_type: sand
      characteristics:
        density: 2650.0
        grain_size: 0.0005  # 0.5 mm
      tracer_methods:
        soulsby:
          flow_field_name: ["grain_velocity"]
          tracer_grain_size: 0.0005
          background_grain_size: 0.0005
      seeding:
        quantity: 50
        strategy:
          grid:
            poly: ./domain_boundary.pol
            separation:
              dx: 500
              dy: 500
```

---

(output-settings)=
## Output Settings

Controls what results are saved and where.

| Parameter             | Type    | Required    | Default    | Description                                                                                             |
| --------------------- | ------- | ----------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| `directory`           | string  | Optional    | `./output` | Path to directory for storing simulation results.                                                       |
| `save_interval`       | string  | Optional    | `1H`       | How often to store trajectory samples. CFL integration can use shorter internal steps; output stores the initial sample, scheduled samples, and final sample. |
| `sync_interval`       | string  | Optional    | `save_interval` | Compatibility setting for how often to flush streaming NetCDF output to disk. Prefer `outputs.netcdf.sync_interval` for new configs. |
| `store_tracks`        | boolean | Optional    | `true`     | Store complete particle trajectories over time. Set to `false` for compact final-state output. |
| `store_end_positions` | boolean | Optional    | `false`    | Store only final particle positions in `sedtrails_results.nc`. Creates a compact one-state file and skips full trajectory output. |

**Example:**

```yaml
outputs:
  directory: ./results/simulation_001
  save_interval: "30M"
  sync_interval: "2H"
  store_tracks: true
  netcdf:
    compression: auto
    compression_auto_threshold_mb: 1024
```

### NetCDF Compression

Nested under `outputs.netcdf`, `compression` accepts `true`, `false`, or `auto` and defaults to `auto`. In `auto` mode, SedTRAILS estimates the uncompressed particle output payload before writing. Compression is enabled when the estimate is greater than or equal to `compression_auto_threshold_mb` MiB, which defaults to `1024`. Full-track output estimates all saved trajectory slots; `store_end_positions` and restart checkpoints estimate one particle snapshot.

---

(visualization)=
## Visualization

Controls plotting, animation, and real-time visualization during and after simulations.

### Dashboard Settings

Real-time simulation monitoring.

Nested under `visualization.dashboard`:

| Parameter         | Type    | Required | Default | Description                                                                |
| ----------------- | ------- | -------- | ------- | -------------------------------------------------------------------------- |
| `enable`          | boolean | Optional | `false` | Enable real-time dashboard during simulation.                              |
| `update_interval` | string  | Optional | `1H`    | How often to update dashboard in simulation time (format: `DDdHHhMMmSSs`). |

### General Plotting Settings

:::warning
Not implemented yet.
:::

Nested under `visualization.plotting.general_settings`:

| Parameter                        | Type    | Required | Default                    | Description                                                                      |
| -------------------------------- | ------- | -------- | -------------------------- | -------------------------------------------------------------------------------- |
| `animation`                      | boolean | Optional | `false`                    | Create animated visualizations of particle movement.                             |
| `overview_age`                   | boolean | Optional | `true`                     | Color particles by age in overview plots.                                        |
| `overview_origin.plot`           | boolean | Optional | `true`                     | Color particles by distance from origin.                                         |
| `overview_origin.rotationtheta`  | number  | Optional | `-90.0`                    | Rotation angle [degrees] for origin-based coloring.                              |
| `static_mask`                    | boolean | Optional | `false`                    | Apply static masking to particles.                                               |
| `random_mask`                    | boolean | Optional | `false`                    | Plot only a random fraction of particles.                                        |
| `random_threshold`               | number  | Optional | `0`                        | Fraction of particles to plot when using random mask (0-1).                      |
| `elevation_mask`                 | boolean | Optional | `true`                     | Mask particles based on elevation.                                               |
| `elevation_threshold`            | number  | Optional | `2.0`                      | Elevation threshold [m] below which to plot particles (avoids plotting on land). |
| `point_size`                     | number  | Optional | `8.0`                      | Size of particle markers in plots.                                               |
| `print_scaling`                  | number  | Optional | `1.0`                      | Scaling factor when saving figures.                                              |
| `print_dimensions`               | string  | Optional | `[29.7 21.0]`              | Figure dimensions [cm] as `[width height]`. Default is A4 paper.                 |
| `dx`                             | number  | Optional | `100.0`                    | Grid spacing [m] for bed interpolation in plots.                                 |
| `mlw`                            | number  | Optional | `-1.0`                     | Mean low water level [m] for reference lines.                                    |
| `mhw`                            | number  | Optional | `1.0`                      | Mean high water level [m] for reference lines.                                   |
| `depth_contours`                 | array   | Optional | `[2.5, 5, 10, 15, 20, 25]` | Depth contours [m] to plot (positive up).                                        |
| `land_pol_file`                  | string  | Optional | -                          | Path to `.pol` file with land and island polygons.                               |
| `xlim`                           | string  | Optional | -                          | X-axis limits as `[left right]`.                                                 |
| `ylim`                           | string  | Optional | -                          | Y-axis limits as `[bottom top]`.                                                 |
| `pathways`                       | boolean | Optional | `false`                    | Plot particle pathways (trajectories).                                           |
| `monitoring_point`               | number  | Optional | `0`                        | Point index for monitoring time series (e.g., tidal signal).                     |
| `monitoring_point_vel_component` | string  | Optional | `E`                        | Velocity component to monitor. Options: `E` (x-direction), `N` (y-direction).    |
| `monitoring_point_signal`        | string  | Optional | `velocity`                 | Signal type to monitor. Options: `velocity`, `waterlevel`.                       |
| `timeseries_ylim`                | string  | Optional | -                          | Y-axis limits for time series plots as `[bottom top]`.                           |

### Validation Plotting


:::warning
Not implemented yet.
:::


Nested under `visualization.plotting.validation`:

| Parameter                      | Type    | Required | Default | Description                                                                      |
| ------------------------------ | ------- | -------- | ------- | -------------------------------------------------------------------------------- |
| `tracer_validation_data`       | string  | Optional | -       | Path to tracer validation data file. If omitted, no tracer validation plotted.   |
| `drifter_validation_data_path` | string  | Optional | -       | Path to drifter validation data file. If omitted, no drifter validation plotted. |
| `drifter_track_comparison`     | boolean | Optional | `false` | Create comparison plots between simulated and observed drifter tracks.           |

**Example:**

```yaml
visualization:
  dashboard:
    enable: true
    update_interval: "30M"
  plotting:
    general_settings:
      animation: false
      overview_age: true
      elevation_mask: true
      elevation_threshold: 2.0
      depth_contours: [5, 10, 15, 20]
      land_pol_file: ./coastline.pol
      pathways: true
    validation:
      drifter_validation_data_path: ./validation/drifters.csv
      drifter_track_comparison: true
```

---

## Complete Example Configuration

Here's a complete example showing how all sections work together:

```yaml
general:
  preprocess: true
  numerical_scheme: rk4
  input_model:
    format: fm_netcdf
    reference_date: "2020-06-01 00"
    morfac: 1.0

inputs:
  data: /data/flow_model/output.nc
  read_interval: "15D"

domain:
  subset_x: "35000:65000"
  subset_y: "12000:45000"
  inner_boundary_pol_files:
    - ./islands.pol
  boundary_class_pol_files:
    open:
      - ./offshore_boundary_edges.pol
    land:
      - ./coastline_boundary_edges.pol
  flow_field_data:
    fm: sedtrails_nc

time:
  start: "2020-06-01 00:00:00"
  duration: "30D"
  timestep: "30S"
  cfl_condition: 0.7

physics:
  constants:
    grain_diameter: 2.5e-4
    rho_s: 2650.0
    porosity: 0.4
    diffusion_coefficient: 0.1
  bed_slope:
    calculate: true
    resolution: 100.0

particles:
  populations:
    - name: fine-sand
      particle_type: sand
      characteristics:
        density: 2650.0
        grain_size: 0.00025
      tracer_methods:
        vanwesten:
          flow_field_name: ["bed_load_velocity"]
          beta: 0.2
      transport_probability: reduced_velocity
      seeding:
        quantity: 500
        release_type: instantaneous
        burial_depth:
          constant: 0.0
        strategy:
          grid:
            poly: ./release_area.pol
            separation:
              dx: 200
              dy: 200

outputs:
  directory: ./results/june_2020
  save_interval: "1H"
  store_tracks: true

visualization:
  dashboard:
    enable: true
    update_interval: "1H"
  plotting:
    general_settings:
      overview_age: true
      elevation_mask: true
      elevation_threshold: 2.0
      depth_contours: [5, 10, 15, 20]
      pathways: true
```

---

## Tips and Best Practices

### Time Step Selection

- Start with `30S` (30 seconds) for most coastal applications
- Use smaller time steps (`10S`-`15S`) for:
  - Fine-resolution grids (< 50 m)
  - Strong tidal currents (> 1 m/s)
  - Complex geometries
- Enable `cfl_condition: 0.7` for automatic adaptive time-stepping

### Domain Definition

- Omit the `domain` section entirely when the input model's active grid is the intended simulation extent and no cutout or boundary-class overrides are needed
- Use `domain.pol_file` for complex, irregular simulation domains
- Use `subset_x` and `subset_y` for simple rectangular domains
- Use `inner_boundary_pol_files` when the flow grid contains island or cutout regions that should not be valid water for particle tracking
- Use `boundary_class_pol_files.open` for offshore boundaries where particles should leave the model
- Use `boundary_class_pol_files.land` for coastlines, islands, and cutouts where particles should beach temporarily and remain available for later remobilization
- Keep open and land override polygons narrow enough to select the intended boundary-edge midpoints only
- Always visualize your domain boundary before running long simulations


### Output Management

- Adjust `save_interval` based on simulation duration:
  - Short runs (< 7 days): `30M` or `1H`
  - Long runs (> 30 days): `6H` or `1D`

### Performance Optimization

- Set `read_interval` to balance memory usage and I/O:
  - Large domains: `7D` to `15D`
  - Small domains: `30D` to `60D`
- Increase `timestep` if simulations are too slow (but check CFL condition)

### Validation

- Enable `elevation_mask` to avoid unrealistic particle behavior on land

---

## Schema Validation

SedTRAILS automatically validates your configuration file against the schemas before running. If you see validation errors:

1. Check that all **required** parameters are present
2. Verify that mutually exclusive options aren't both specified
3. Ensure data types match (string vs. number vs. boolean)
4. Check that enum values match exactly (case-sensitive)

Refer to the [Simulation Guide](../user/simulations.md) for an example  of a YAML file with correct syntax and structure.
