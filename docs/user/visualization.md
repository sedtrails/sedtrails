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

![SedTRAILS trajectory example](../_static/img/example-trajectory-plots.png)

The built-in trajectory plotter reads the saved particle coordinates directly. Its distance panel is computed from each particle's saved ``x`` and ``y`` positions relative to that particle's first valid position; it does not use ``covered_distance``.

The following options can be used to save and customize the plot:

- ``--file`` or ``-f``: Path to the SedTRAILS NetCDF file to visualize. By default, it expects ``sedtrails_results.nc`` in the current directory.
- ``--output`` or ``-o``: Output target. If this points to an existing directory, SedTRAILS writes ``particle_trajectories.png`` inside it. Otherwise, SedTRAILS treats it as a filename. If omitted, SedTRAILS writes ``particle_trajectories.png`` next to the NetCDF file.
- ``--max-particles``: Maximum number of particles to plot. The sample is deterministic and stratified by population where population IDs are available.
- ``--sample-fraction``: Fraction of particles to plot. This cannot be combined with ``--max-particles``.
- ``--sample-seed``: Seed for deterministic sampling. The default is ``0``.
- ``--markers``: Endpoint markers to draw: ``none``, ``end``, or ``start-end``. The default is ``end``.
- ``--marker-size``: Marker size for start/end points. The default is ``3``.
- ``--panels``: Panels to draw. The default is ``spatial``. Use ``all`` for a four-panel plot, or a comma-separated subset of ``spatial``, ``distance``, ``population``, and ``population-distance``.
- ``--show`` or ``--no-show``: Override whether an interactive figure window is displayed. By default, plots are not shown.
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
