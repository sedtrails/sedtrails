"""Tests for the Delft3D-FM NetCDF format plugin."""

import numpy as np
import pytest
import xarray as xr

from sedtrails.transport_converter.plugins.format.fm_netcdf import FormatPlugin


def _plugin_with_dataset(tmp_path, monkeypatch, dataset):
    """Create an FM plugin whose load method exposes the provided dataset."""
    input_file = tmp_path / 'input.nc'
    input_file.touch()
    plugin = FormatPlugin(str(input_file))

    def fake_load():
        """Attach the synthetic dataset instead of reading NetCDF from disk."""
        plugin.input_data = dataset

    monkeypatch.setattr(plugin, 'load', fake_load)
    return plugin


def test_get_max_exposure_depth_fields_uses_dynamic_bed_and_shear(tmp_path, monkeypatch):
    """Maximum exposure fields should use full-period erosion and bed shear maxima."""
    dataset = xr.Dataset(
        {
            'bedlevel': (('time', 'mesh2d_nFaces'), np.array([[10.0, 5.0], [8.0, 6.0], [9.0, 3.0]])),
            'max_bss_magnitude': (('time', 'mesh2d_nFaces'), np.array([[1.0, 2.0], [5.0, 1.0], [4.0, 3.0]])),
        }
    )
    plugin = _plugin_with_dataset(tmp_path, monkeypatch, dataset)

    max_erosion, max_bss = plugin.get_max_exposure_depth_fields()

    np.testing.assert_allclose(max_erosion, [2.0, 2.0])
    np.testing.assert_allclose(max_bss, [5.0, 3.0])


def test_get_max_exposure_depth_fields_handles_static_fields(tmp_path, monkeypatch):
    """Static bed levels should produce zero erosion and static max BSS."""
    dataset = xr.Dataset(
        {
            'bedlevel': ('mesh2d_nFaces', np.array([10.0, 5.0])),
            'max_bss_magnitude': ('mesh2d_nFaces', np.array([2.0, 4.0])),
        }
    )
    plugin = _plugin_with_dataset(tmp_path, monkeypatch, dataset)

    max_erosion, max_bss = plugin.get_max_exposure_depth_fields()

    np.testing.assert_allclose(max_erosion, [0.0, 0.0])
    np.testing.assert_allclose(max_bss, [2.0, 4.0])


def test_map_dfm_variables_rejects_layer_velocity_as_depth_average(tmp_path, monkeypatch):
    """A single selected layer must not be reported as depth-averaged velocity."""
    dataset = xr.Dataset(
        {
            'net_xcc': ('mesh2d_nFaces', np.array([0.0, 1.0])),
            'net_ycc': ('mesh2d_nFaces', np.array([0.0, 0.0])),
            'sea_water_x_velocity': (
                ('time', 'layer', 'mesh2d_nFaces'),
                np.ones((1, 2, 2)),
            ),
        }
    )
    plugin = _plugin_with_dataset(tmp_path, monkeypatch, dataset)
    plugin.load()
    monkeypatch.setattr(plugin, '_active_triangular_connectivity', lambda x, y: np.array([[0, 1, 1]]))

    with pytest.raises(ValueError, match='requires depth-averaged flow velocity'):
        plugin._map_dfm_variables({'num_times': 1}, None, None)


@pytest.mark.parametrize('missing_variable', ['bedlevel', 'max_bss_magnitude'])
def test_get_max_exposure_depth_fields_requires_inputs(tmp_path, monkeypatch, missing_variable):
    """Missing exposure-input variables should fail with clear KeyErrors."""
    data_vars = {
        'bedlevel': ('mesh2d_nFaces', np.array([10.0, 5.0])),
        'max_bss_magnitude': ('mesh2d_nFaces', np.array([2.0, 4.0])),
    }
    del data_vars[missing_variable]
    plugin = _plugin_with_dataset(tmp_path, monkeypatch, xr.Dataset(data_vars))

    with pytest.raises(KeyError, match=missing_variable):
        plugin.get_max_exposure_depth_fields()


def test_convert_squeezes_leading_singleton_axis_only(tmp_path, monkeypatch):
    """DFM singleton chunk axes should be removed without flattening time/spatial axes."""
    input_file = tmp_path / 'input.nc'
    input_file.touch()
    plugin = FormatPlugin(str(input_file))
    monkeypatch.setattr(plugin, 'load', lambda: None)
    monkeypatch.setattr(
        plugin,
        '_get_time_info',
        lambda dataset, reference_date: {
            'seconds_since_reference': np.array([0.0]),
            'reference_date': np.datetime64('1970-01-01T00:00:00'),
        },
    )
    monkeypatch.setattr(plugin, '_decompress_time', lambda time_info: time_info)
    monkeypatch.setattr(plugin, '_calculate_time_slice', lambda current_time, reading_interval, time_info: (None, None))

    one_by_time_by_node = np.arange(2.0).reshape(1, 1, 2)
    mapped_data = {
        'x': np.array([0.0, 1.0]),
        'y': np.array([0.0, 0.0]),
        'bed_level': one_by_time_by_node + 10.0,
        'flow_velocity_x': one_by_time_by_node + 1.0,
        'flow_velocity_y': one_by_time_by_node,
        'bed_load_transport_x': one_by_time_by_node + 0.1,
        'bed_load_transport_y': one_by_time_by_node,
        'suspended_transport_x': one_by_time_by_node + 0.2,
        'suspended_transport_y': one_by_time_by_node,
        'water_depth': one_by_time_by_node + 2.0,
        'mean_bed_shear_stress': one_by_time_by_node + 0.5,
        'max_bed_shear_stress': one_by_time_by_node + 0.8,
        'sediment_concentration': one_by_time_by_node + 0.01,
    }
    monkeypatch.setattr(plugin, '_map_dfm_variables', lambda time_info, time_start_idx, time_end_idx: mapped_data)

    sedtrails_data = plugin.convert()

    assert sedtrails_data.depth_avg_flow_velocity['x'].shape == (1, 2)
    assert sedtrails_data.bed_level.shape == (1, 2)
