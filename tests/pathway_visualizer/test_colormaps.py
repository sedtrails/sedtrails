import numpy as np

from sedtrails.pathway_visualizer.colormaps import (
    VINTAGE_BATHYMETRY_COLORS,
    VINTAGE_BATHYMETRY_LEVELS,
    available_bathymetry_colormaps,
    bathymetry_colormap,
    SEDTRAILS_BATHYMETRY_COLORS,
    SEDTRAILS_BATHYMETRY_LEVELS,
    SEAWAD_BATHYMETRY_COLORS,
    SEAWAD_BATHYMETRY_LEVELS,
    sedtrails_bathymetry_colormap,
)


def test_sedtrails_bathymetry_colormap_uses_matlab_anchors():
    """SEAWAD keeps the original MATLAB bathymetry levels and endpoint colors."""

    cmap, norm = sedtrails_bathymetry_colormap(n_colors=256)

    assert np.array_equal(SEAWAD_BATHYMETRY_LEVELS, np.array([-20.0, -10.0, -5.0, -2.0, 2.0, 3.0, 10.0]))
    assert np.array_equal(SEDTRAILS_BATHYMETRY_LEVELS, SEAWAD_BATHYMETRY_LEVELS)
    assert np.allclose(SEAWAD_BATHYMETRY_COLORS[0], np.array([0, 67, 143]) / 255.0)
    assert np.allclose(SEAWAD_BATHYMETRY_COLORS[-1], np.array([29, 89, 74]) / 255.0)
    assert np.array_equal(SEDTRAILS_BATHYMETRY_COLORS, SEAWAD_BATHYMETRY_COLORS)
    assert norm.vmin == -20.0
    assert norm.vmax == 10.0
    assert norm.clip is True
    assert np.allclose(cmap(norm(-20.0))[:3], SEAWAD_BATHYMETRY_COLORS[0])
    assert np.allclose(cmap(norm(-30.0))[:3], SEAWAD_BATHYMETRY_COLORS[0])


def test_named_bathymetry_colormap_registry_includes_seawad_and_vintage():
    """The named palette registry exposes SEAWAD and Vintage with expected anchors."""

    assert available_bathymetry_colormaps() == ('SEAWAD', 'Vintage')

    _, seawad_norm = bathymetry_colormap('seawad')
    vintage_cmap, vintage_norm = bathymetry_colormap('Vintage')

    assert seawad_norm.vmin == -20.0
    assert seawad_norm.vmax == 10.0
    assert np.array_equal(
        VINTAGE_BATHYMETRY_LEVELS,
        np.array([-20.0, -16.0, -12.0, -8.0, -5.0, -3.0, -1.2, 0.0, 1.4, 10.0]),
    )
    assert np.allclose(VINTAGE_BATHYMETRY_COLORS[0], np.array([27, 126, 129]) / 255.0)
    assert np.allclose(VINTAGE_BATHYMETRY_COLORS[-1], np.array([226, 129, 61]) / 255.0)
    assert vintage_norm.vmin == -20.0
    assert vintage_norm.vmax == 10.0
    assert np.allclose(vintage_cmap(vintage_norm(-20.0))[:3], VINTAGE_BATHYMETRY_COLORS[0])


def test_sedtrails_bathymetry_colormap_accepts_custom_limits():
    """Custom bathymetry color limits override the palette default normalization range."""

    _, norm = sedtrails_bathymetry_colormap(vmin=-8.0, vmax=4.0)

    assert norm.vmin == -8.0
    assert norm.vmax == 4.0
