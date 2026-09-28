# Visualization

This page is a placeholder for SedTRAILS visualization guidance beyond the live monitoring dashboard. It can be expanded with examples for plotting model input fields, output trajectories, particle statistics, and publication-style figures.

For the real-time run dashboard, see [Monitoring Dashboard](dashboard.md). For output file inspection and trajectory plotting commands, see [Output](output.md).

## Basic Visualization

Use ``sedtrails viz trajectories`` to plot the saved ``x``, ``y``, and ``time`` trajectories:

```text
sedtrails viz trajectories -f C:\your-filepath-here\sedtrails_results.nc
```

By default, the command draws only the spatial trajectory panel so it remains usable for large particle sets. Write directly to a file and plot a deterministic sample for the fastest workflow:

```text
sedtrails viz trajectories -f C:\your-filepath-here\sedtrails_results.nc --output trajectories.png --max-particles 2000 --markers end
```

Use ``--panels all`` to draw the previous four-panel summary, or request selected optional panels:

```text
sedtrails viz trajectories -f C:\your-filepath-here\sedtrails_results.nc --panels spatial,population --output trajectories.png
```

To also create an animated GIF of the sampled trajectories, use ``--animate-gif``. The GIF shows moving particle positions and leaves each particle's trail behind it:

```text
sedtrails viz trajectories -f C:\your-filepath-here\sedtrails_results.nc --output trajectories.png --animate-gif --max-particles 500 --gif-fps 8
```

Use ``--gif-frame-stride`` to skip rendered particle-position frames while still including the first and last saved timestep. For example, ``--gif-frame-stride 20`` renders frames ``0, 20, 40, ...`` and appends the final frame when needed. The trajectory trails still use all saved timesteps up to each rendered frame.

GIF playback is chronological by default, so reverse-tracking output is animated from earlier timestamps toward later timestamps even though the NetCDF stores those samples in reverse integration order. Use ``--gif-time-order simulation`` to preserve the stored integration order instead.

By default, SedTRAILS writes bare output filenames next to the NetCDF file, so ``--output trajectories.png --animate-gif`` creates ``trajectories.png`` and ``trajectories.gif`` in the results file directory. If the projected GIF frame buffer is larger than ``20`` MiB, SedTRAILS asks for confirmation and defaults to ``N``. Use ``--yes-large-gif`` to skip that prompt for batch scripts.

![SedTRAILS trajectory example](../_static/img/example-trajectory-plots.png)

The built-in trajectory plotter reads the saved particle coordinates directly. Its distance panel is computed from each particle's saved ``x`` and ``y`` positions relative to that particle's first valid position; it does not use ``covered_distance``.

The following options can be used to save and customize the plot:

- ``--file`` or ``-f``: Path to the SedTRAILS NetCDF file to visualize. By default, it expects ``sedtrails_results.nc`` in the current directory.
- ``--output`` or ``-o``: Output target. If this points to an existing directory, SedTRAILS writes ``particle_trajectories.png`` inside it. A bare filename is written next to the NetCDF file. If omitted, SedTRAILS writes ``particle_trajectories.png`` next to the NetCDF file.
- ``--max-particles``: Maximum number of particles to plot. The default is ``10000``. The sample is deterministic and stratified when static population metadata is available.
- ``--max-plot-points``: Maximum selected particle-time coordinates to render. The default is ``2000000``; SedTRAILS evenly decimates saved times before loading coordinates to stay within the budget.
- ``--sample-fraction``: Fraction of particles to plot. This cannot be combined with ``--max-particles``.
- ``--sample-seed``: Seed for deterministic sampling. The default is ``0``.
- ``--markers``: Endpoint markers to draw: ``none``, ``end``, or ``start-end``. The default is ``end``.
- ``--marker-size``: Marker size for start/end points. The default is ``3``.
- ``--panels``: Panels to draw. The default is ``spatial``. Use ``all`` for a four-panel plot, or a comma-separated subset of ``spatial``, ``distance``, ``population``, and ``population-distance``.
- ``--show`` or ``--no-show``: Override whether an interactive figure window is displayed. By default, plots are not shown.
- ``--animate-gif``: Also save an animated GIF of moving particles with trajectory trails.
- ``--gif-output``: Exact GIF output path. If omitted, SedTRAILS uses the static plot path with a ``.gif`` suffix.
- ``--gif-fps``: Frames per second for the GIF. The default is ``10``.
- ``--gif-dpi``: DPI used to render GIF frames. The default is ``120``.
- ``--gif-frame-stride``: Save one GIF frame every N output timesteps. The default is ``1``. The first and last timesteps are always included, and trails still use all saved timesteps up to each rendered frame.
- ``--gif-time-order``: GIF playback order: ``chronological`` or ``simulation``. The default is ``chronological``.
- ``--gif-max-size-warning-mb``: Prompt threshold for the projected GIF frame buffer. The default is ``20``.
- ``--yes-large-gif`` or ``--no-large-gif``: Override the large-GIF prompt. By default, SedTRAILS asks and defaults to no.
- ``--help`` or ``-h``: Show the command help.

:::warning
More advanced visualization functions will be added in future releases, but for now we encourage users to build custom visualizations directly from the output data.
:::

## Bathymetry colormaps

SedTRAILS has reusable bathymetry colormaps for input-data checks and visualization tools. These are currently used by the seeding setup GUI and can also be reused from Python plotting scripts.

Available named colormaps:

- `SEAWAD`: a blue-white-brown-green bathymetry palette. Copyright: (c) Stuart G. Pearson, 2022. (CC BY 4.0).
- `Vintage`: a vintage Wadden Sea map-inspired bathymetry palette. Copyright: (c) Stuart G. Pearson, 2022. (CC BY 4.0), based on Rijkswaterstaat Studiedienst Hoorn (1943); see Elias et al. 2019.

In the seeding setup GUI, use the colormap selector to switch between `SEAWAD` and `Vintage`. The color-limit boxes can be used to change the displayed bathymetry range without changing the underlying input data.

Python placeholder:

```python
from sedtrails.pathway_visualizer.colormaps import bathymetry_colormap

cmap, norm = bathymetry_colormap("Vintage", vmin=-12, vmax=6)
```

Planned additions for this page include examples for applying these colormaps to model input fields, dashboard plots, and output trajectory maps.
