"""Shared colormap helpers for SedTRAILS visualizations.

SEAWAD: (c) Stuart G. Pearson, 2022. (CC BY 4.0)
Vintage: (c) Stuart G. Pearson, 2022. (CC BY 4.0), based on
Rijkswaterstaat Studiedienst Hoorn (1943); see Elias et al. 2019.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


# SEAWAD: (c) Stuart G. Pearson, 2022. (CC BY 4.0)
SEAWAD_BATHYMETRY_LEVELS = np.array([-20.0, -10.0, -5.0, -2.0, 2.0, 3.0, 10.0])
SEAWAD_BATHYMETRY_COLORS = np.array(
    [
        [0, 67, 143],
        [13, 182, 255],
        [255, 255, 255],
        [199, 181, 181],
        [158, 144, 144],
        [29, 89, 74],
        [29, 89, 74],
    ],
    dtype=float,
) / 255.0

# Vintage: (c) Stuart G. Pearson, 2022. (CC BY 4.0), based on
# Rijkswaterstaat Studiedienst Hoorn (1943); see Elias et al. 2019.
VINTAGE_BATHYMETRY_LEVELS = np.array([-20.0, -16.0, -12.0, -8.0, -5.0, -3.0, -1.2, 0.0, 1.4, 10.0])
VINTAGE_BATHYMETRY_COLORS = np.array(
    [
        [27, 126, 129],
        [41, 155, 151],
        [56, 170, 164],
        [130, 199, 180],
        [220, 231, 194],
        [255, 240, 196],
        [244, 214, 176],
        [217, 188, 146],
        [255, 221, 146],
        [226, 129, 61],
    ],
    dtype=float,
) / 255.0

BATHYMETRY_COLORMAPS = {
    'SEAWAD': (SEAWAD_BATHYMETRY_LEVELS, SEAWAD_BATHYMETRY_COLORS),
    'Vintage': (VINTAGE_BATHYMETRY_LEVELS, VINTAGE_BATHYMETRY_COLORS),
}

# Backward-compatible aliases for callers/tests that imported the original names.
SEDTRAILS_BATHYMETRY_LEVELS = SEAWAD_BATHYMETRY_LEVELS
SEDTRAILS_BATHYMETRY_COLORS = SEAWAD_BATHYMETRY_COLORS


def available_bathymetry_colormaps() -> tuple[str, ...]:
    """
    Return available named SedTRAILS bathymetry palettes.

    Returns
    -------
    tuple[str, ...]
        Tuple containing the computed values.
    """

    return tuple(BATHYMETRY_COLORMAPS)


def bathymetry_colormap(
    colormap_name: str = 'SEAWAD',
    *,
    n_colors: int = 256,
    vmin: float | None = None,
    vmax: float | None = None,
    clip: bool = True,
) -> tuple[LinearSegmentedColormap, Normalize]:
    """
    Return a named SedTRAILS bathymetry colormap and normalization.

    ``SEAWAD`` and ``Vintage`` are both (c) Stuart G. Pearson, 2022
    (CC BY 4.0). ``Vintage`` is based on Rijkswaterstaat Studiedienst
    Hoorn (1943); see Elias et al. 2019.

    Parameters
    ----------
    colormap_name : str
        Name of the colormap to use.
    n_colors : int
        Number of colors to sample from the colormap.
    vmin : float | None
        Lower bound of the normalized color range.
    vmax : float | None
        Upper bound of the normalized color range.
    clip : bool
        Whether to clip values outside the color range.

    Returns
    -------
    tuple[LinearSegmentedColormap, Normalize]
        Tuple containing the computed values.
    """

    canonical_name = _resolve_colormap_name(colormap_name)
    levels, colors = BATHYMETRY_COLORMAPS[canonical_name]
    positions = (levels - levels[0]) / (levels[-1] - levels[0])
    cmap = LinearSegmentedColormap.from_list(
        canonical_name,
        list(zip(positions, colors, strict=True)),
        N=n_colors,
    )
    norm = Normalize(
        vmin=float(levels[0] if vmin is None else vmin),
        vmax=float(levels[-1] if vmax is None else vmax),
        clip=clip,
    )
    return cmap, norm


def sedtrails_bathymetry_colormap(
    *,
    name: str = 'SEAWAD',
    n_colors: int = 256,
    vmin: float | None = None,
    vmax: float | None = None,
    clip: bool = True,
) -> tuple[LinearSegmentedColormap, Normalize]:
    """
    Return the default SedTRAILS bathymetry colormap and value normalization.

    The default ``SEAWAD`` anchors are based on the historical MATLAB bathymetry palette:
    dark blue at -20 m NAP, light blue at -10 m NAP, white at -5 m NAP,
    light/muddy browns around the intertidal range, and dark green above 3 m NAP.

    Parameters
    ----------
    name : str
        Name of the requested object.
    n_colors : int
        Number of colors to sample from the colormap.
    vmin : float | None
        Lower bound of the normalized color range.
    vmax : float | None
        Upper bound of the normalized color range.
    clip : bool
        Whether to clip values outside the color range.

    Returns
    -------
    tuple[LinearSegmentedColormap, Normalize]
        Tuple containing the computed values.
    """

    return bathymetry_colormap(
        name,
        n_colors=n_colors,
        vmin=vmin,
        vmax=vmax,
        clip=clip,
    )


def _resolve_colormap_name(colormap_name: str) -> str:
    for available_name in BATHYMETRY_COLORMAPS:
        if available_name.lower() == colormap_name.lower():
            return available_name
    names = ', '.join(BATHYMETRY_COLORMAPS)
    raise ValueError(f"Unknown bathymetry colormap '{colormap_name}'. Available: {names}")
