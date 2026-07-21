# Seeding Strategies

:::warning
Explain the different strategies available for seeding particles to the end user.
Focus on what the user needs to know, not on how it is implemented. Long explanations should be avoided. Only explain what is necessary for the user to know. If a longer explanation is needed, a different section should be created with explains the soundings of the strategies from the point of view of an expert in sediment transport.
:::



## Particle seeding in SedTRAILS

:::warning
This section needs to be rethought and rewritten. Currently, it sounds more like explaning than instructing the user. 
:::

Seeding is the process of adding the particle(s) to the model. In SedTRAILS, the type and initial position(s) of the sediment(s) are set as separate populations. The options for particle types are sand, mud, and passive tracer. Multiple strategies are available to locate the paticles as initial conditions. They can be added as either a single particle (point strategy) or more. In case of adding many particles, they can be located along a spcified line (transect strategy), or within a polygon (grid and random strategies). 

### Release timing and `release_start`

Particle release timing is controlled by `seeding.release_start` per population.

- `release_start` is optional.
- If omitted, particles are released from simulation start.
- If provided, it must use format `YYYY-MM-DD HH:MM:SS` (for example: `2020-01-01 00:10:00`).
- The configured value is converted to seconds relative to `general.input_model.reference_date`.
- A particle only becomes mobile once simulation time is greater than or equal to its converted release time.

This means you can define multiple populations with different delayed release times in one run.

Example:

```yaml
general:
  input_model:
    reference_date: "2020-01-01 00:00:00"

particles:
  populations:
    - name: early-release
      particle_type: sand
      seeding:
        quantity: 10
        release_start: "2020-01-01 00:05:00"
        strategy:
          point:
            locations: ["1000,2000"]
    - name: delayed-release
      particle_type: sand
      seeding:
        quantity: 10
        release_start: "2020-01-01 02:00:00"
        strategy:
          point:
            locations: ["1000,2000"]
```

Inputs required for particle types:
| Sand       | Mud        | Passive    |
| ---------- | ---------- | ---------- |
| grain size | grain size | grain size |
| ???        | ???        | ???        |


Inputs required for seeding strategies:
| Point | Transect | Grid (uniform)    | Grid (random) |
| ----- | -------- | ----------------- | ------------- |
| x, y  | x1, y1   | x1, y1            | ???           |
| ???   | x2, y2   | x2, y2            | ???           |
| ???   | ???      | particle distance | ???           |
| ???   | ???      | ???               | ???           |

## Seeding setup GUI

SedTRAILS includes a lightweight graphical helper for setting up particle release locations from an existing YAML configuration file. The GUI is intended for quick validation-case setup: it loads the configured input model data, displays first-timestep bathymetry, lets you create seed points interactively, and writes a copied YAML file plus point-file inputs.

Start the GUI from the command line:

```bash
sedtrails config gui --config examples/sedtrails-example-multisource.yaml
```

Common options:

```bash
sedtrails config gui \
  --config examples/sedtrails-example-multisource.yaml \
  --output examples/my-case.yaml \
  --points-output examples/my-case.points.txt \
  --population populaton_1 \
  --variable bedlevel
```

Useful keywords:

- `--config` or `-c`: source SedTRAILS YAML file to load.
- `--output` or `-o`: copied YAML file to write. If omitted, the GUI writes next to the source YAML using `*-seeded.yaml`.
- `--points-output`: text file for generated x/y release locations.
- `--population` or `-p`: initial population to edit. If omitted, the first configured population is selected.
- `--format`: input format override. Supported values are `fm_netcdf`, `d3d4_netcdf`, `xbeach`, and `sfincs`.
- `--variable`: bathymetry variable to display. Format defaults are:
  `bedlevel`/`bed_level` for `fm_netcdf`,
  `DPS0`/`DP0` for `d3d4_netcdf`,
  `zb_mean`/`zb` for `xbeach`,
  and `zb` for `sfincs`.

### GUI workflow

1. Open an existing YAML config with `sedtrails config gui`.
2. Select a particle population in the population panel. The first population is selected by default.
3. Use **Zoom +**, **Zoom -**, and **Pan** to inspect the map (toggle **Pan** off again to add points), then choose a seeding mode and create points.
4. Optionally clip generated points by bathymetry elevation.
5. Save. The original YAML is not edited; a copied YAML and one or more point files are written.

The GUI saves release locations through the existing `file_points` strategy. This means points created with manual clicks, transects, random polygons, or grids all become ordinary text files referenced from the generated YAML.

### GUI seeding modes

- **Points**: left click on the map to add release points. Right click near an existing point to remove it.
- **Transect**: left click endpoint pairs, set `k`, then press **Generate**. Each pair becomes a straight transect with `k` generated release locations.
- **Random**: left click polygon vertices, right click to close the polygon, set `n` and `seed`, then press **Generate**. The GUI generates `n` random points inside the polygon.
- **Grid**: left click polygon vertices, right click to close the polygon, set `dx` and `dy`, then press **Generate**. The GUI creates a regular grid clipped to the polygon.

### Populations in the GUI

The GUI can select, add, rename, and remove particle populations. Adding a population copies the currently selected population's particle settings and characteristics, then gives the copy a new name. Always inspect the generated YAML before running the simulation, especially after adding populations, because copied tracer methods, particle type, grain-size settings, and release timing may need adjustment.

When multiple populations have generated points, the GUI writes one point file per population and updates each population's `seeding.strategy.file_points` entry.

### Elevation clipping

The clipping controls remove generated points using nearest-cell bathymetry values from the displayed input data. Choose whether to delete points above or below a threshold elevation, enter the threshold, then press **Clip points**. This is useful for quickly removing points from dry areas, channels, shoals, or other elevation-defined zones.


Here I copy paste Manuel's comments in Github issue#267. I will revise this part of the documentation while the seeding factory is finalised.

"""
The code in this PR follows this reasoning:

The seeding tool generates n-number of particles for a population using the seeding strategy defined in the configuration for each population.
The number of particle per location are defined by quantity and the number of seeding locations depend on the strategy used. e.g., above only one seed location is specified. The number of seeding locations for some of the strategies are computed at runtime (e.g., grid).
The total number of particle in a population is the product of quantity * seeding locations
When seeding happens a list with the total number of particles in a population is created. If more than one population is specified in the configuration file, than particle creation should be repeated for each one (I will work on automating this). The creation of particle is manage by a ParticleFactory, which takes the configuration for a population and generates the particles, assigns the initial x,y coordinates, release_time (release_start) and quantity.
"""

## File Points Seeding Strategy

The `file_points` seeding strategy allows you to specify particle release locations by reading coordinates from external text, CSV, or TSV files. This is useful when you have pre-defined release points from field surveys, model outputs, or other data sources.

### Configuration Parameters

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

### Column Specification

- **By index**: Use 0-based column indices (e.g., `x_col: 0`, `y_col: 1`)
- **By name**: Use column names if the file has headers (e.g., `x_col: "longitude"`, `y_col: "latitude"`)
- **Default behavior**: If not specified, the first two columns are assumed to be x and y coordinates

### Bounding Box Filtering

The optional `bbox` parameter allows you to filter points to a specific geographic area:

**String format:**
```yaml
bbox: "xmin,ymin xmax,ymax"  # e.g., "1000,2000 5000,6000"
```

**Basic Usage with CSV file:**
```yaml
seeding:
  strategy:
    file_points:
      path: "release_points.csv"
      x_col: "x"
      y_col: "y"
      has_header: true
```

**Using column indices with tab-separated file:**
```yaml
seeding:
  strategy:
    file_points:
      path: "coordinates.tsv"
      x_col: 0
      y_col: 1
      has_header: false
```

**Subsampling with stride and bounding box:**
```yaml
seeding:
  strategy:
    file_points:
      path: "dense_grid.txt"
      stride: 5  # Keep every 5th point
      bbox: "1000,2000 5000,6000"
      deduplicate: true
```

### Supported File Formats

- **Text files (.txt)**: Space or tab-separated values
- **CSV files (.csv)**: Comma-separated values
- **TSV files (.tsv)**: Tab-separated values

### Example Text File
Here is a simple comma-separated text file with five points (``examples\sources_xy_inlet.txt``), for use in ``examples\sedtrails-example-multisource.yaml``.

```
39400,16800
39400,17800
40000,17300 
40600,16800 
40600,17800
```

### Data Processing Options

- **Deduplication**: Automatically removes duplicate coordinate pairs when `deduplicate: true`
- **Missing data handling**: Removes rows with NaN or missing values when `dropna: true`
- **Subsampling**: Use `stride` parameter to reduce point density (e.g., `stride: 10` keeps every 10th point)
- **Spatial filtering**: Apply bounding box to restrict release locations to specific areas

### Use Cases

This strategy is particularly useful for:

- Field survey locations
- Existing monitoring station coordinates
- Output from other modeling systems
- Pre-processed release point datasets
- Integration with GIS-derived coordinate lists
