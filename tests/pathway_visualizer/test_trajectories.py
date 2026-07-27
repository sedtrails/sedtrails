import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from sedtrails.pathway_visualizer.trajectories import (
    _sample_dataset,
    _select_sample_indices,
    _trajectory_arrays,
    plot_trajectories,
    read_netcdf,
)


def test_plot_trajectories_accepts_fixed_width_population_names(monkeypatch, tmp_path):
    n_particles = 2
    n_timesteps = 3

    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [1.0, 1.5], [2.0, 3.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
            # 1D fixed-width byte strings as produced by current writer path
            'population_name': (('n_populations',), np.array([b'pop_A', b'pop_B'], dtype='S24')),
        },
        coords={
            'n_particles': np.arange(n_particles),
            'n_timesteps': np.arange(n_timesteps),
            'n_populations': np.arange(2),
        },
    )

    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    # This previously raised: "too many indices"
    plot_trajectories(ds, output=tmp_path / 'fixed_width_population_names.png')


def test_plot_trajectories_decodes_object_wrapped_byte_names(monkeypatch, tmp_path):
    """Legend labels decode object-wrapped NumPy byte scalars."""
    n_particles = 2
    n_timesteps = 3
    population_names = np.empty(2, dtype=object)
    population_names[0] = np.array(np.bytes_(b'population_0\x00   '), dtype=object)
    population_names[1] = np.array(np.bytes_(b'population_1\x00   '), dtype=object)
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [1.0, 1.5], [2.0, 3.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
            'population_name': (('n_populations',), population_names),
        },
        coords={
            'n_particles': np.arange(n_particles),
            'n_timesteps': np.arange(n_timesteps),
            'n_populations': np.arange(2),
        },
    )
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    fig, axes_by_panel = plot_trajectories(
        ds,
        output=tmp_path / 'population_legend.png',
        panels='population-distance',
        show=False,
    )

    legend = axes_by_panel['population-distance'].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    plt.close(fig)
    assert labels == [
        'population_0 (mean)',
        'population_0 (+/-1 std)',
        'population_1 (mean)',
        'population_1 (+/-1 std)',
    ]


def test_plot_trajectories_accepts_time_major_layout(monkeypatch, tmp_path):
    n_particles = 2
    n_timesteps = 3

    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [1.0, 1.5], [2.0, 3.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
            'population_name': (('n_populations',), np.array([b'pop_A', b'pop_B'], dtype='S24')),
        },
        coords={
            'n_particles': np.arange(n_particles),
            'n_timesteps': np.arange(n_timesteps),
            'n_populations': np.arange(2),
        },
        attrs={'trajectory_layout': 'time_particle'},
    )

    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    plot_trajectories(ds, output=tmp_path / 'time_major_layout.png')


def test_read_netcdf_keeps_cf_time_values_as_seconds(tmp_path):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0], [2.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [0.5], [1.0]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
        },
        attrs={'reference_date': '2020-01-01 00:00:00'},
    )
    ds['time'].attrs['units'] = 'seconds since 2020-01-01 00:00:00'
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    loaded = read_netcdf(netcdf_file)
    try:
        _, _, time_data = _trajectory_arrays(loaded)
    finally:
        loaded.close()

    np.testing.assert_array_equal(time_data[0], np.array([0.0, 60.0, 120.0]))


def test_trajectory_arrays_converts_decoded_datetime_time_to_seconds():
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0], [2.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [0.5], [1.0]])),
            'time': (
                ('n_timesteps',),
                np.array(
                    [
                        '2020-01-01T00:00:00',
                        '2020-01-01T00:01:00',
                        '2020-01-01T00:02:00',
                    ],
                    dtype='datetime64[ns]',
                ),
            ),
        },
        attrs={'reference_date': '2020-01-01 00:00:00'},
    )

    _, _, time_data = _trajectory_arrays(ds)

    np.testing.assert_array_equal(time_data[0], np.array([0.0, 60.0, 120.0]))


def test_plot_trajectories_rejects_particle_major_layout(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'y': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'time': (('n_particles', 'n_timesteps'), np.array([[0.0, 60.0]])),
        }
    )

    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    with pytest.raises(ValueError, match='time-major trajectory arrays'):
        plot_trajectories(ds)


def test_plot_trajectories_output_file_skips_show_by_default(tmp_path, monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [1.0, 1.5], [2.0, 3.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
        },
        coords={
            'n_particles': np.arange(2),
            'n_timesteps': np.arange(3),
            'n_populations': np.arange(2),
        },
    )
    show_called = False

    def _show():
        nonlocal show_called
        show_called = True

    monkeypatch.setattr('matplotlib.pyplot.show', _show)

    output_file = tmp_path / 'trajectories.png'
    plot_trajectories(ds, output=output_file)

    assert output_file.exists()
    assert not show_called


def test_plot_trajectories_saves_next_to_source_file_by_default(tmp_path, monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [1.0, 1.5], [2.0, 3.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
        },
        coords={
            'n_particles': np.arange(2),
            'n_timesteps': np.arange(3),
        },
    )
    source_file = tmp_path / 'results.nc'
    ds.encoding['source'] = str(source_file)

    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    plot_trajectories(ds)

    assert (tmp_path / 'particle_trajectories.png').exists()


def test_plot_trajectories_output_dot_uses_current_working_directory(tmp_path, monkeypatch):
    """An explicit dot output should not fall back to the NetCDF source directory."""
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [0.5]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0])),
        },
        coords={
            'n_particles': np.arange(1),
            'n_timesteps': np.arange(2),
        },
    )
    source_dir = tmp_path / 'source'
    current_dir = tmp_path / 'current'
    source_dir.mkdir()
    current_dir.mkdir()
    ds.encoding['source'] = str(source_dir / 'results.nc')
    monkeypatch.chdir(current_dir)
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    fig, _ = plot_trajectories(ds, output='.')
    try:
        assert (current_dir / 'particle_trajectories.png').exists()
        assert not (source_dir / 'particle_trajectories.png').exists()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_trajectories_output_existing_directory_uses_default_filename(tmp_path, monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0])),
        },
        coords={
            'n_particles': np.arange(1),
            'n_timesteps': np.arange(2),
        },
    )
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    output_dir = tmp_path / 'plots'
    output_dir.mkdir()

    plot_trajectories(ds, output=output_dir)

    assert (output_dir / 'particle_trajectories.png').exists()


def test_plot_trajectories_draws_spatial_panel_by_default(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [1.0, 1.5], [2.0, 3.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
        },
        coords={
            'n_particles': np.arange(2),
            'n_timesteps': np.arange(3),
        },
    )
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    fig, axes = plot_trajectories(ds)
    try:
        assert list(axes) == ['spatial']
        assert len(fig.axes) == 1
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_trajectories_can_draw_all_panels(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [1.0, 1.5], [2.0, 3.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
            'population_name': (('n_populations',), np.array([b'pop_A', b'pop_B'], dtype='S24')),
        },
        coords={
            'n_particles': np.arange(2),
            'n_timesteps': np.arange(3),
            'n_populations': np.arange(2),
        },
    )
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    fig, axes = plot_trajectories(ds, panels='all', marker_size=8)
    try:
        assert list(axes) == ['spatial', 'distance', 'population', 'population-distance']
        assert len(fig.axes) == 4
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_trajectories_show_can_be_forced_with_output_file(tmp_path, monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0])),
        },
        coords={
            'n_particles': np.arange(1),
            'n_timesteps': np.arange(2),
        },
    )
    show_called = False

    def _show():
        nonlocal show_called
        show_called = True

    monkeypatch.setattr('matplotlib.pyplot.show', _show)

    plot_trajectories(ds, output=tmp_path / 'trajectories.png', show=True)

    assert show_called


def test_select_sample_indices_is_deterministic_and_population_stratified():
    population_ids = np.array([0] * 8 + [1] * 2)

    selected = _select_sample_indices(10, population_ids, max_particles=5, sample_seed=123)
    repeated = _select_sample_indices(10, population_ids, max_particles=5, sample_seed=123)

    np.testing.assert_array_equal(selected, repeated)
    assert len(selected) == 5
    assert np.any(population_ids[selected] == 0)
    assert np.any(population_ids[selected] == 1)


def test_sample_dataset_selects_particles_before_array_conversion():
    n_particles = 10
    n_timesteps = 2
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.arange(n_timesteps * n_particles).reshape(n_timesteps, n_particles)),
            'y': (('n_timesteps', 'n_particles'), np.arange(n_timesteps * n_particles).reshape(n_timesteps, n_particles)),
            'time': (('n_timesteps',), np.array([0.0, 60.0])),
            'population_id': (('n_particles',), np.array([0] * 5 + [1] * 5, dtype=int)),
        },
        coords={
            'n_particles': np.arange(n_particles),
            'n_timesteps': np.arange(n_timesteps),
            'n_populations': np.arange(2),
        },
    )

    sampled, total, selected = _sample_dataset(ds, max_particles=4, sample_seed=42)

    assert total == 10
    assert selected == 4
    assert sampled.sizes['n_particles'] == 4


def test_plot_trajectories_validates_sampling_options(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0])),
        }
    )
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    with pytest.raises(ValueError, match='mutually exclusive'):
        plot_trajectories(ds, max_particles=1, sample_fraction=0.5)

    with pytest.raises(ValueError, match='markers'):
        plot_trajectories(ds, markers='both')

    with pytest.raises(ValueError, match='panels'):
        plot_trajectories(ds, panels='bad-panel')

    with pytest.raises(ValueError, match='marker_size'):
        plot_trajectories(ds, marker_size=0)
