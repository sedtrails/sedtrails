import numpy as np
import pytest

from sedtrails.transport_converter.plugins.format.fm_netcdf import FormatPlugin as FmNetcdfFormatPlugin
from sedtrails.transport_converter.plugins.format.sfincs import FormatPlugin as SfincsFormatPlugin


@pytest.mark.parametrize('plugin_factory', [FmNetcdfFormatPlugin, SfincsFormatPlugin])
def test_epoch_based_forcing_axis_uses_span_for_read_interval(tmp_path, plugin_factory):
    """A read interval longer than the forcing span should load the full file."""
    input_file = tmp_path / 'dummy.nc'
    input_file.touch()
    plugin = plugin_factory(str(input_file))
    time_info = {
        'seconds_since_reference': np.array(
            [
                1474485600.0,
                1474486200.0,
                1474486800.0,
            ]
        )
    }

    assert plugin._calculate_time_slice(
        current_time=1474485600.0,
        reading_interval=3600.0,
        time_info=time_info,
    ) == (None, None)


@pytest.mark.parametrize('plugin_factory', [FmNetcdfFormatPlugin, SfincsFormatPlugin])
def test_short_read_interval_still_chunks_epoch_based_forcing_axis(tmp_path, plugin_factory):
    """Read intervals shorter than the forcing span should still request a chunk."""
    input_file = tmp_path / 'dummy.nc'
    input_file.touch()
    plugin = plugin_factory(str(input_file))
    time_info = {
        'seconds_since_reference': np.array(
            [
                1474485600.0,
                1474486200.0,
                1474486800.0,
            ]
        )
    }

    assert plugin._calculate_time_slice(
        current_time=1474485600.0,
        reading_interval=600.0,
        time_info=time_info,
    ) == (0, 3)
