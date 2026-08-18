"""Tests for the Delft3D4 NetCDF format converter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sedtrails.transport_converter.plugins.format import d3d4_netcdf


SAMPLE_FILE = Path('sample-data/trim-inlet.nc')


def _plot_if_requested(
    data,
    output_dir: Path,
    time_idx: int,
    mask: np.ndarray | None,
) -> None:
    """Write optional diagnostic plots when SEDTRAILS_PLOT_DIR is configured."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(data.x)
    y = np.asarray(data.y)
    scalar_x, scalar_y = _resolve_scalar_coordinates(data)
    time_label = _format_time_label(data, time_idx)

    if x.ndim == 2 and y.ndim == 2:
        invalid_mask = (x == 0) & (y == 0)
        if mask is not None:
            invalid_mask = invalid_mask | mask
        stride_x = max(1, x.shape[0] // 25)
        stride_y = max(1, x.shape[1] // 25)
        xs = x[::stride_x, ::stride_y]
        ys = y[::stride_x, ::stride_y]
        u = data.depth_avg_flow_velocity['x'][time_idx][::stride_x, ::stride_y]
        v = data.depth_avg_flow_velocity['y'][time_idx][::stride_x, ::stride_y]
        subsampled_mask = invalid_mask[::stride_x, ::stride_y]
        u = np.where(subsampled_mask, np.nan, u)
        v = np.where(subsampled_mask, np.nan, v)
        plt.figure(figsize=(7, 6))
        plt.quiver(xs, ys, u, v, scale=10)
        plt.title(f'Depth-averaged flow velocity ({time_label})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(output_dir / 'flow_velocity_vectors.png', dpi=150)
        plt.close()
    elif x.ndim == 1 and y.ndim == 1:
        u = data.depth_avg_flow_velocity['x'][time_idx]
        v = data.depth_avg_flow_velocity['y'][time_idx]
        stride = max(1, x.size // 2500)
        xs = x[::stride]
        ys = y[::stride]
        u = u[::stride]
        v = v[::stride]
        plt.figure(figsize=(7, 6))
        plt.quiver(xs, ys, u, v, scale=10)
        plt.title(f'Depth-averaged flow velocity ({time_label})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(output_dir / 'flow_velocity_vectors.png', dpi=150)
        plt.close()

    _plot_scalar_field(
        scalar_x,
        scalar_y,
        data.mean_bed_shear_stress[time_idx],
        output_dir / 'mean_bed_shear_stress.png',
        title=f'Mean bed shear stress ({time_label})',
        colorbar_label='Pa',
        mask=mask,
    )

    _plot_scalar_field(
        scalar_x,
        scalar_y,
        data.max_bed_shear_stress[time_idx],
        output_dir / 'max_bed_shear_stress.png',
        title=f'Max bed shear stress ({time_label})',
        colorbar_label='Pa',
        mask=mask,
    )

    if data.bed_load_transport is not None:
        _plot_scalar_field(
            scalar_x,
            scalar_y,
            data.bed_load_transport['magnitude'][time_idx],
            output_dir / 'bed_load_transport.png',
            title=f'Bed load transport magnitude ({time_label})',
            colorbar_label='m3/(s m)',
            mask=mask,
        )

    if data.suspended_transport is not None:
        _plot_scalar_field(
            scalar_x,
            scalar_y,
            data.suspended_transport['magnitude'][time_idx],
            output_dir / 'suspended_transport.png',
            title=f'Suspended transport magnitude ({time_label})',
            colorbar_label='m3/(s m)',
            mask=mask,
        )


def _plot_scalar_field(
    x,
    y,
    values,
    output_path: Path,
    title: str,
    colorbar_label: str,
    mask: np.ndarray | None = None,
) -> None:
    """Plot a scalar field using provided x/y coordinates when available."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    if np.asarray(x).ndim == 2 and np.asarray(y).ndim == 2:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        invalid_mask = (x_arr == 0) & (y_arr == 0)
        if mask is not None:
            invalid_mask = invalid_mask | mask
        masked_values = np.where(invalid_mask, np.nan, values)
        valid = np.isfinite(masked_values)
        plt.scatter(
            x_arr[valid],
            y_arr[valid],
            c=masked_values[valid],
            s=36,
            marker='s',
            linewidths=0,
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
    elif np.asarray(x).ndim == 1 and np.asarray(y).ndim == 1:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        invalid_mask = (x_arr == 0) & (y_arr == 0)
        if mask is not None:
            invalid_mask = invalid_mask | mask
        masked_values = np.where(invalid_mask, np.nan, values)
        valid = np.isfinite(masked_values)
        plt.scatter(
            x_arr[valid],
            y_arr[valid],
            c=masked_values[valid],
            s=36,
            marker='s',
            linewidths=0,
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
    else:
        plt.imshow(values, origin='lower')
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _format_time_label(data, time_idx: int) -> str:
    """Return a timestamp label for plot titles."""
    try:
        timestamp = data.reference_date + np.timedelta64(int(data.times[time_idx]), 's')
        return str(np.datetime_as_string(timestamp, unit='s'))
    except Exception:
        return f'index {time_idx}'


def _resolve_plot_time_index(data) -> int:
    """Resolve plot time index from environment, with safe fallback."""
    raw_value = os.environ.get('SEDTRAILS_PLOT_TIME_INDEX')
    if raw_value is None:
        return 0
    try:
        idx = int(raw_value)
    except ValueError:
        return 0
    if idx < 0:
        idx = 0
    if idx >= data.times.size:
        idx = data.times.size - 1
    return idx


def _resolve_scalar_coordinates(data) -> tuple[np.ndarray, np.ndarray]:
    """Return XZ/YZ coordinates for scalar plotting when available."""
    import xarray as xr

    if np.asarray(data.x).ndim == 1:
        return np.asarray(data.x), np.asarray(data.y)

    if SAMPLE_FILE.exists():
        try:
            ds = xr.open_dataset(SAMPLE_FILE, decode_times=False, decode_timedelta=False)
        except Exception:
            ds = None
        else:
            if 'XZ' in ds and 'YZ' in ds:
                return np.asarray(ds['XZ'].values), np.asarray(ds['YZ'].values)

    return np.asarray(data.x), np.asarray(data.y)


def _resolve_plot_mask() -> str | None:
    """Return optional plot mask selector from environment."""
    value = os.environ.get('SEDTRAILS_PLOT_MASK')
    if value is None:
        return None
    value = value.strip().lower()
    return value if value else None


def _build_kcs_mask() -> np.ndarray | None:
    """Build a mask from KCS values in the sample dataset when available."""
    import xarray as xr

    if not SAMPLE_FILE.exists():
        return None

    try:
        ds = xr.open_dataset(SAMPLE_FILE, decode_times=False, decode_timedelta=False)
    except Exception:
        return None

    if 'KCS' not in ds:
        return None

    kcs = np.asarray(ds['KCS'].values)
    return kcs == 0


@pytest.mark.integration
@pytest.mark.skipif(not SAMPLE_FILE.exists(), reason='Sample Delft3D4 NetCDF file not available')
def test_delft3d4_netcdf_conversion() -> None:
    """Check Delft3D4 NetCDF conversion produces a valid SedtrailsData object."""
    plugin = d3d4_netcdf.FormatPlugin(str(SAMPLE_FILE))
    data = plugin.convert(reference_date=np.datetime64('1970-01-01T00:00:00'))

    assert data.times.size > 0
    assert data.x.shape == data.y.shape
    assert data.bed_level.shape == data.x.shape

    assert data.depth_avg_flow_velocity is not None
    assert data.depth_avg_flow_velocity['x'].shape[0] == data.times.size
    assert data.depth_avg_flow_velocity['y'].shape == data.depth_avg_flow_velocity['x'].shape

    assert data.water_depth.shape[0] == data.times.size
    assert data.mean_bed_shear_stress.shape == data.max_bed_shear_stress.shape

    assert np.isfinite(data.x).any()
    assert np.isfinite(data.y).any()

    domain = data.metadata.flowfield_domain
    for key in ['x_min', 'x_max', 'y_min', 'y_max']:
        assert key in domain

    plot_dir = os.environ.get('SEDTRAILS_PLOT_DIR')
    if plot_dir:
        time_idx = _resolve_plot_time_index(data)
        mask_type = _resolve_plot_mask()
        mask = _build_kcs_mask() if mask_type == 'kcs' else None
        _plot_if_requested(data, Path(plot_dir), time_idx, mask)


def test_centers_and_rotates_all_delft3d_local_vector_pairs(tmp_path: Path) -> None:
    """Delft3D U/V pairs use dynamic wet masks and ALFAS before SedTRAILS mapping."""
    input_file = tmp_path / 'local_vectors.nc'
    input_file.write_text('')

    u_values = np.array(
        [
            [[2.0, 4.0, 6.0], [10.0, 12.0, 14.0]],
            [[102.0, 104.0, 106.0], [110.0, 112.0, 114.0]],
        ]
    )
    v_values = np.array(
        [
            [[3.0, 9.0, 15.0], [21.0, 27.0, 33.0]],
            [[103.0, 109.0, 115.0], [121.0, 127.0, 133.0]],
        ]
    )
    u_masks = np.array(
        [
            [[1, 1, 1], [1, 0, 1]],
            [[1, 1, 1], [1, 1, 1]],
        ]
    )
    v_masks = np.array(
        [
            [[1, 0, 1], [1, 1, 0]],
            [[1, 1, 1], [1, 1, 1]],
        ]
    )
    expected_u = np.array(
        [
            [[2.0, 4.0, 6.0], [6.0, 4.0, 10.0]],
            [[102.0, 104.0, 106.0], [106.0, 108.0, 110.0]],
        ]
    )
    expected_v = np.array(
        [
            [[3.0, 3.0, 15.0], [21.0, 24.0, 27.0]],
            [[103.0, 106.0, 112.0], [121.0, 124.0, 130.0]],
        ]
    )

    sediment_u = np.stack([u_values, 3.0 * u_values], axis=1)
    sediment_v = np.stack([v_values, 3.0 * v_values], axis=1)
    dataset = xr.Dataset(
        data_vars={
            'ALFAS': (('M', 'N'), np.full((2, 3), 90.0)),
            'KFU': (('time', 'MC', 'N'), u_masks),
            'KFV': (('time', 'M', 'NC'), v_masks),
            'U1': (('time', 'MC', 'N'), u_values),
            'V1': (('time', 'M', 'NC'), v_values),
            'TAUKSI': (('time', 'MC', 'N'), 2.0 * u_values),
            'TAUETA': (('time', 'M', 'NC'), 2.0 * v_values),
            'SBUU': (('time', 'LSED', 'MC', 'N'), sediment_u),
            'SBVV': (('time', 'LSED', 'M', 'NC'), sediment_v),
            'SSUU': (('time', 'LSED', 'MC', 'N'), 4.0 * sediment_u),
            'SSVV': (('time', 'LSED', 'M', 'NC'), 4.0 * sediment_v),
        },
        coords={'LSED': ['fine', 'coarse']},
    )

    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = dataset

    flow_x, flow_y = plugin._map_vector_pair(
        u_variable='U1',
        v_variable='V1',
        grid_shape=(2, 3),
        time_slice=slice(None),
        num_times=2,
        select_fraction_dims=True,
    )
    stress_x, stress_y = plugin._map_vector_pair(
        u_variable='TAUKSI',
        v_variable='TAUETA',
        grid_shape=(2, 3),
        time_slice=slice(None),
        num_times=2,
        select_fraction_dims=True,
    )
    bed_x, bed_y = plugin._map_vector_pair(
        u_variable='SBUU',
        v_variable='SBVV',
        grid_shape=(2, 3),
        time_slice=slice(None),
        num_times=2,
        select_fraction_dims=False,
    )
    suspended_x, suspended_y = plugin._map_vector_pair(
        u_variable='SSUU',
        v_variable='SSVV',
        grid_shape=(2, 3),
        time_slice=slice(None),
        num_times=2,
        select_fraction_dims=False,
    )

    np.testing.assert_allclose(flow_x, -expected_v)
    np.testing.assert_allclose(flow_y, expected_u)
    np.testing.assert_allclose(stress_x, -2.0 * expected_v)
    np.testing.assert_allclose(stress_y, 2.0 * expected_u)
    np.testing.assert_allclose(bed_x, -np.stack([expected_v, 3.0 * expected_v], axis=1))
    np.testing.assert_allclose(bed_y, np.stack([expected_u, 3.0 * expected_u], axis=1))
    np.testing.assert_allclose(suspended_x, -4.0 * np.stack([expected_v, 3.0 * expected_v], axis=1))
    np.testing.assert_allclose(suspended_y, 4.0 * np.stack([expected_u, 3.0 * expected_u], axis=1))


def test_rotates_local_vector_pairs_with_positive_alfas(tmp_path: Path) -> None:
    """A positive ALFAS angle rotates xi/eta components counter-clockwise into x/y."""
    input_file = tmp_path / 'single_cell.nc'
    input_file.write_text('')
    dataset = xr.Dataset(
        data_vars={
            'ALFAS': (('M', 'N'), np.array([[45.0]])),
            'KFU': (('time', 'MC', 'N'), np.ones((1, 1, 1))),
            'KFV': (('time', 'M', 'NC'), np.ones((1, 1, 1))),
            'U1': (('time', 'MC', 'N'), np.array([[[2.0]]])),
            'V1': (('time', 'M', 'NC'), np.array([[[1.0]]])),
        }
    )
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = dataset

    x_component, y_component = plugin._map_vector_pair(
        u_variable='U1',
        v_variable='V1',
        grid_shape=(1, 1),
        time_slice=slice(None),
        num_times=1,
        select_fraction_dims=True,
    )

    np.testing.assert_allclose(x_component, np.array([[[1.0 / np.sqrt(2.0)]]]))
    np.testing.assert_allclose(y_component, np.array([[[3.0 / np.sqrt(2.0)]]]))


def test_selects_configured_fraction_index_from_lsed_dimension(tmp_path: Path) -> None:
    """The plugin should select the configured sediment fraction index for LSED data."""
    input_file = tmp_path / 'dummy.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.sediment_fraction_index = 2

    var = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=('LSED', 'M'),
        coords={'LSED': ['sediment100_nat', 'sediment200_nat', 'sediment_mud']},
    )

    selected = plugin._select_first_dims(var)
    np.testing.assert_array_equal(selected.values, var.values[2])


def test_selects_configured_fraction_name_from_lsed_dimension(tmp_path: Path) -> None:
    """The plugin should resolve sediment fraction by coordinate name when configured."""
    input_file = tmp_path / 'dummy.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.sediment_fraction_index = 0
    plugin.sediment_fraction_name = 'sediment300_nat'

    var = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=('LSED', 'M'),
        coords={'LSED': ['sediment100_nat', 'sediment200_nat', 'sediment300_nat']},
    )

    selected = plugin._select_first_dims(var)
    np.testing.assert_array_equal(selected.values, var.values[2])


def test_converts_namcon_labels_into_metadata(tmp_path: Path) -> None:
    """The converter should expose Delft3D-4 NAMCON labels as sediment fraction metadata."""
    input_file = tmp_path / 'namcon.nc'

    labels = ['sediment100_nat', 'sediment200_nat', 'sediment300_nat']
    namcon = np.full((len(labels), 20), b' ', dtype='S1')
    for row_index, label in enumerate(labels):
        encoded = np.array(list(label.ljust(20)), dtype='S1')
        namcon[row_index, :] = encoded

    x_coords = np.array([[0.0, 1.0], [0.0, 1.0]])
    y_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    dataset = xr.Dataset(
        data_vars={
            'NAMCON': (('LSTSCI', 'strlen20'), namcon),
            'XCOR': (('y', 'x'), x_coords),
            'YCOR': (('y', 'x'), y_coords),
            'DP0': (('y', 'x'), np.array([[1.0, 1.0], [1.0, 1.0]])),
            'time': (('time',), np.array(['2010-01-01T00:00:00'], dtype='datetime64[ns]')),
            'U1': (('time', 'y', 'x'), np.array([[[1.0, 1.0], [1.0, 1.0]]])),
            'V1': (('time', 'y', 'x'), np.array([[[0.0, 0.0], [0.0, 0.0]]])),
            'TAUKSI': (('time', 'y', 'x'), np.array([[[0.1, 0.1], [0.1, 0.1]]])),
            'TAUETA': (('time', 'y', 'x'), np.array([[[0.2, 0.2], [0.2, 0.2]]])),
            'TAUMAX': (('time', 'y', 'x'), np.array([[[0.3, 0.3], [0.3, 0.3]]])),
            'SBUU': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
            'SBVV': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
            'SSUU': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
            'SSVV': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
            'R1': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
        }
    )
    dataset.to_netcdf(input_file)

    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    data = plugin.convert(reference_date=np.datetime64('1970-01-01T00:00:00'))

    assert data.metadata.get('sediment_fraction_labels') == labels


def test_raises_for_out_of_bounds_fraction_index(tmp_path: Path) -> None:
    """An invalid fraction index should fail fast with a clear error."""
    input_file = tmp_path / 'dummy.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.sediment_fraction_index = 99

    var = xr.DataArray(np.arange(8).reshape(2, 4), dims=('LSED', 'M'))

    with pytest.raises(ValueError, match='out of bounds'):
        plugin._select_first_dims(var)


def test_uses_static_kcu_kcv_masks_when_dynamic_masks_are_absent(tmp_path: Path) -> None:
    """Static KCU/KCV masks must prevent dry-face values entering vector centers."""
    input_file = tmp_path / 'static_masks.nc'
    input_file.write_text('')
    u_values = np.array([[[2.0, 4.0], [6.0, 8.0]]])
    v_values = np.array([[[10.0, 20.0], [30.0, 40.0]]])
    dataset = xr.Dataset(
        data_vars={
            'ALFAS': (('M', 'N'), np.zeros((2, 2))),
            'KCU': (('M', 'N'), np.array([[1, 0], [0, 1]])),
            'KCV': (('M', 'N'), np.array([[1, 0], [0, 1]])),
            'U1': (('time', 'MC', 'N'), u_values),
            'V1': (('time', 'M', 'NC'), v_values),
        }
    )
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = dataset

    x_component, y_component = plugin._map_vector_pair(
        u_variable='U1',
        v_variable='V1',
        grid_shape=(2, 2),
        time_slice=slice(None),
        num_times=1,
        select_fraction_dims=True,
    )

    np.testing.assert_allclose(x_component, np.array([[[2.0, 0.0], [2.0, 8.0]]]))
    np.testing.assert_allclose(y_component, np.array([[[10.0, 10.0], [0.0, 40.0]]]))


def test_rejects_an_incomplete_local_vector_pair(tmp_path: Path) -> None:
    """A local component without its matching pair must fail before conversion."""
    input_file = tmp_path / 'incomplete_vector.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = xr.Dataset(
        data_vars={
            'U1': (('time', 'MC', 'N'), np.ones((1, 1, 1))),
        }
    )

    with pytest.raises(KeyError, match='incomplete'):
        plugin._map_vector_pair(
            u_variable='U1',
            v_variable='V1',
            grid_shape=(1, 1),
            time_slice=slice(None),
            num_times=1,
            select_fraction_dims=True,
        )


def test_reuses_vector_masks_and_static_grid_rotation_per_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each conversion reads wet masks and static ALFAS geometry only once."""
    input_file = tmp_path / 'cached_geometry.nc'
    input_file.write_text('')
    pair_values = np.ones((2, 1, 1))
    dataset = xr.Dataset(
        data_vars={
            'XZ': (('M', 'N'), np.array([[10.0]])),
            'YZ': (('M', 'N'), np.array([[20.0]])),
            'DP0': (('M', 'N'), np.ones((1, 1))),
            'ALFAS': (('M', 'N'), np.zeros((1, 1))),
            'KFU': (('time', 'MC', 'N'), np.ones((2, 1, 1))),
            'KFV': (('time', 'M', 'NC'), np.ones((2, 1, 1))),
            'U1': (('time', 'MC', 'N'), pair_values),
            'V1': (('time', 'M', 'NC'), pair_values),
            'TAUKSI': (('time', 'MC', 'N'), pair_values),
            'TAUETA': (('time', 'M', 'NC'), pair_values),
            'SBUU': (('time', 'MC', 'N'), pair_values),
            'SBVV': (('time', 'M', 'NC'), pair_values),
            'SSUU': (('time', 'MC', 'N'), pair_values),
            'SSVV': (('time', 'M', 'NC'), pair_values),
        }
    )
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = dataset
    read_calls: list[str] = []
    angle_calls = 0
    original_read = plugin._read_values
    original_angle = plugin._local_grid_angle

    def track_read(variable_name: str, **kwargs) -> np.ndarray:
        read_calls.append(variable_name)
        return original_read(variable_name, **kwargs)

    def track_angle(grid_shape: tuple[int, int]) -> np.ndarray:
        nonlocal angle_calls
        angle_calls += 1
        return original_angle(grid_shape)

    monkeypatch.setattr(plugin, '_read_values', track_read)
    monkeypatch.setattr(plugin, '_local_grid_angle', track_angle)

    data = plugin._map_delft3d4_variables({'num_times': 2})

    assert read_calls.count('KFU') == 1
    assert read_calls.count('KFV') == 1
    assert angle_calls == 1
    assert data['flow_velocity_x'].shape == (2, 1)


def test_filters_kcs_inactive_faces_from_conversion_and_seeding(tmp_path: Path) -> None:
    """Inactive KCS=0 faces must not become duplicate SedTRAILS nodes."""
    input_file = tmp_path / 'inactive_faces.nc'
    dataset = xr.Dataset(
        data_vars={
            'XZ': (('M', 'N'), np.array([[0.0, 1.0], [0.0, 0.0]])),
            'YZ': (('M', 'N'), np.array([[0.0, 0.0], [1.0, 0.0]])),
            'KCS': (('M', 'N'), np.array([[1, 1], [1, 0]])),
            'DP0': (('M', 'N'), np.ones((2, 2))),
            'ALFAS': (('M', 'N'), np.zeros((2, 2))),
            'KFU': (('time', 'MC', 'N'), np.ones((1, 2, 2))),
            'KFV': (('time', 'M', 'NC'), np.ones((1, 2, 2))),
            'U1': (('time', 'MC', 'N'), np.ones((1, 2, 2))),
            'V1': (('time', 'M', 'NC'), np.ones((1, 2, 2))),
            'TAUKSI': (('time', 'MC', 'N'), np.ones((1, 2, 2))),
            'TAUETA': (('time', 'M', 'NC'), np.ones((1, 2, 2))),
            'TAUMAX': (('time', 'M', 'N'), np.ones((1, 2, 2))),
        },
        coords={'time': np.array(['2010-01-01T00:00:00'], dtype='datetime64[ns]')},
    )
    dataset.to_netcdf(input_file)

    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    data = plugin.convert(reference_date=np.datetime64('2010-01-01T00:00:00'))
    seed_x, seed_y = plugin.get_seeding_coordinates()

    assert data.x.size == 3
    assert data.bed_level.shape == (3,)
    assert data.depth_avg_flow_velocity['x'].shape == (1, 3)
    assert np.count_nonzero((data.x == 0.0) & (data.y == 0.0)) == 1
    np.testing.assert_array_equal(seed_x, data.x)
    np.testing.assert_array_equal(seed_y, data.y)


def test_map_vector_pair_rejects_layer_velocity_as_depth_average(tmp_path: Path) -> None:
    """Delft3D layer zero must not be reported as depth-averaged velocity."""
    input_file = tmp_path / 'layered_velocity.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = xr.Dataset(
        {
            'U1': (('time', 'KMAXOUT_RESTR', 'M', 'N'), np.ones((1, 2, 1, 1))),
            'V1': (('time', 'KMAXOUT_RESTR', 'M', 'N'), np.ones((1, 2, 1, 1))),
        }
    )

    with pytest.raises(ValueError, match='requires depth-averaged flow velocity'):
        plugin._map_vector_pair(
            u_variable='U1',
            v_variable='V1',
            grid_shape=(1, 1),
            time_slice=slice(None),
            num_times=1,
            select_fraction_dims=True,
            require_depth_averaged=True,
        )


@pytest.mark.parametrize('metadata_store', ['attrs', 'encoding'])
def test_decompresses_numeric_cf_time_using_shared_seconds_utility(tmp_path: Path, metadata_store: str) -> None:
    """Morfac decompression must not mix numeric source time with decoded datetimes."""
    input_file = tmp_path / 'numeric_cf_time.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file), morfac=2.0)
    plugin.input_data = xr.Dataset(
        data_vars={'time': (('time',), np.array([100.0, 160.0]))},
    )
    metadata = getattr(plugin.input_data['time'], metadata_store)
    metadata['units'] = 'seconds since 2000-01-01 00:00:00'
    metadata['calendar'] = 'proleptic_gregorian'
    reference_date = np.datetime64('1970-01-01T00:00:00')

    time_info = plugin._get_time_info(plugin.input_data, reference_date)
    decompressed = plugin._decompress_time(time_info)

    assert time_info['original_units'] == metadata['units']
    assert time_info['original_calendar'] == metadata['calendar']
    np.testing.assert_array_equal(decompressed['seconds_since_reference'], np.array([946684900.0, 946685020.0]))
    np.testing.assert_array_equal(
        decompressed['time_values'].astype('datetime64[s]'),
        np.array(['2000-01-01T00:01:40', '2000-01-01T00:03:40'], dtype='datetime64[s]'),
    )
    assert plugin.get_time_bounds(reference_date=reference_date) == (946684900.0, 946685020.0)


def test_cf_time_metadata_attributes_override_encoding(tmp_path: Path) -> None:
    """CF metadata in attrs takes precedence over conflicting encoding metadata."""
    input_file = tmp_path / 'cf_time_metadata_precedence.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = xr.Dataset(
        data_vars={'time': (('time',), np.array([100.0, 160.0]))},
    )
    plugin.input_data['time'].attrs.update(
        units='seconds since 2000-01-01 00:00:00',
        calendar='proleptic_gregorian',
    )
    plugin.input_data['time'].encoding.update(
        units='seconds since 1970-01-01 00:00:00',
        calendar='standard',
    )

    time_info = plugin._get_time_info(plugin.input_data, np.datetime64('1970-01-01T00:00:00'))

    assert time_info['original_units'] == 'seconds since 2000-01-01 00:00:00'
    assert time_info['original_calendar'] == 'proleptic_gregorian'
    np.testing.assert_array_equal(
        time_info['time_values'].astype('datetime64[s]'),
        np.array(['2000-01-01T00:01:40', '2000-01-01T00:02:40'], dtype='datetime64[s]'),
    )


def test_converts_positive_down_bottom_depth_to_bed_elevation(tmp_path: Path) -> None:
    """DPS0 becomes elevation while the S1 fallback remains physical water depth."""
    input_file = tmp_path / 'positive_down_depth.nc'
    input_file.write_text('')
    dps0 = np.array([[4.0, 6.0]])
    dataset = xr.Dataset(
        data_vars={
            'XZ': (('M', 'N'), np.array([[10.0, 20.0]])),
            'YZ': (('M', 'N'), np.array([[30.0, 30.0]])),
            'DPS0': (('M', 'N'), dps0),
            'DP0': (('MC', 'NC'), np.array([[99.0, 99.0]])),
            'S1': (('time', 'M', 'N'), np.array([[[1.0, -2.0]]])),
        }
    )
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = dataset

    mapped = plugin._map_delft3d4_variables({'num_times': 1})

    np.testing.assert_array_equal(mapped['bed_level'], np.array([-4.0, -6.0]))
    np.testing.assert_array_equal(mapped['water_depth'], np.array([[5.0, 4.0]]))


def test_uses_face_aligned_dp0_as_bottom_depth_fallback(tmp_path: Path) -> None:
    """A face-aligned DP0 fallback must retain the depth-to-elevation conversion."""
    input_file = tmp_path / 'face_aligned_dp0.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = xr.Dataset(
        data_vars={
            'XZ': (('M', 'N'), np.array([[10.0, 20.0]])),
            'YZ': (('M', 'N'), np.array([[30.0, 30.0]])),
            'DP0': (('M', 'N'), np.array([[3.0, 5.0]])),
            'S1': (('time', 'M', 'N'), np.array([[[2.0, -1.0]]])),
        }
    )

    mapped = plugin._map_delft3d4_variables({'num_times': 1})

    np.testing.assert_array_equal(mapped['bed_level'], np.array([-3.0, -5.0]))
    np.testing.assert_array_equal(mapped['water_depth'], np.array([[5.0, 4.0]]))


def test_rejects_node_located_dp0_without_dps0(tmp_path: Path) -> None:
    """A node DP0 field cannot be silently registered to face coordinates."""
    input_file = tmp_path / 'node_dp0.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = xr.Dataset(
        data_vars={
            'XZ': (('M', 'N'), np.array([[10.0, 20.0], [10.0, 20.0]])),
            'YZ': (('M', 'N'), np.array([[30.0, 30.0], [40.0, 40.0]])),
            'DP0': (('MC', 'NC'), np.array([[1.0, 2.0], [3.0, 4.0]])),
        }
    )

    with pytest.raises(ValueError, match='DP0 must be face-located'):
        plugin._map_delft3d4_variables({'num_times': 1})


def test_rejects_node_dp0_with_only_node_coordinates(tmp_path: Path) -> None:
    """Node coordinates do not make a node DP0 field face-aligned."""
    input_file = tmp_path / 'node_coordinates_node_dp0.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = xr.Dataset(
        data_vars={
            'XCOR': (('MC', 'NC'), np.array([[10.0, 20.0], [10.0, 20.0]])),
            'YCOR': (('MC', 'NC'), np.array([[30.0, 30.0], [40.0, 40.0]])),
            'DP0': (('MC', 'NC'), np.array([[1.0, 2.0], [3.0, 4.0]])),
        }
    )

    with pytest.raises(ValueError, match='DP0 must be face-located'):
        plugin._map_delft3d4_variables({'num_times': 1})


def test_rejects_non_face_dp0_metadata(tmp_path: Path) -> None:
    """A DP0 field explicitly marked as an edge cannot represent face depth."""
    input_file = tmp_path / 'edge_dp0.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.input_data = xr.Dataset(
        data_vars={
            'XZ': (('M', 'N'), np.array([[10.0, 20.0]])),
            'YZ': (('M', 'N'), np.array([[30.0, 30.0]])),
            'DP0': (
                ('M', 'N'),
                np.array([[1.0, 2.0]]),
                {'location': 'edge1'},
            ),
        }
    )

    with pytest.raises(ValueError, match='DP0 must be face-located'):
        plugin._map_delft3d4_variables({'num_times': 1})


def test_preserves_fraction_dimension_when_requested(tmp_path: Path) -> None:
    """Sediment variables should keep LSED when selection is deferred to population level."""
    input_file = tmp_path / 'dummy.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))

    var = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=('LSED', 'M'),
    )

    selected = plugin._select_first_dims(var, select_fraction_dims=False)
    assert selected.dims == ('LSED', 'M')
    np.testing.assert_array_equal(selected.values, var.values)
