# Visualization

This page is a placeholder for SedTRAILS visualization guidance beyond the live monitoring dashboard. It can be expanded with examples for plotting model input fields, output trajectories, particle statistics, and publication-style figures.

For the real-time run dashboard, see [Monitoring Dashboard](dashboard.md). For output file inspection and trajectory plotting commands, see [Output](output.md).

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
