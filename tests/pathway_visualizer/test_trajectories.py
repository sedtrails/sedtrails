import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from sedtrails.pathway_visualizer.trajectories import (
    _confirm_large_gif,
    _distance_line_segments,
    _gif_frame_indices,
    _gif_playback_indices,
    _gif_trail_indices,
    _line_segments,
    _resolve_gif_output_file,
    _sample_dataset,
    _select_sample_indices,
    _trajectory_arrays,
    plot_trajectories,
    read_netcdf,
)


def test_geographic_trajectory_segments_split_at_antimeridian():
    """A map line must not draw the long way across the longitude seam."""
    segments, indices, _, _, _ = _line_segments(
        np.array([[179.0, 179.8, -179.8, -179.0]]),
        np.array([[0.0, 0.0, 0.0, 0.0]]),
        geographic=True,
    )

    assert len(segments) == 2
    np.testing.assert_array_equal(indices, [0, 0])


def test_geographic_trajectory_distance_is_reported_in_metres():
    """Geographic distance panels use great-circle metres."""
    _, _, particle_distances = _distance_line_segments(
        np.array([[179.9, -179.9]]),
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 60.0]]),
        0.0,
        geographic=True,
    )

    _, distance = particle_distances[0]
    assert distance[-1] == pytest.approx(22_239.0, rel=2.0e-4)


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


def test_confirm_large_gif_defaults_to_no(monkeypatch, tmp_path):
    monkeypatch.setattr('builtins.input', lambda _: '')

    assert not _confirm_large_gif(tmp_path / 'large.gif', 21.0, 20.0, None)


def test_confirm_large_gif_accepts_yes_response(monkeypatch, tmp_path):
    monkeypatch.setattr('builtins.input', lambda _: 'y')

    assert _confirm_large_gif(tmp_path / 'large.gif', 21.0, 20.0, None)


def test_gif_frame_indices_respect_stride_and_include_last():
    indices = _gif_frame_indices(289, frame_stride=20)

    np.testing.assert_array_equal(indices, np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 288]))


def test_gif_frame_indices_can_play_reverse_output_chronologically():
    indices = _gif_frame_indices(
        5,
        frame_stride=2,
        time_values=np.array([40.0, 30.0, 20.0, 10.0, 0.0]),
    )

    np.testing.assert_array_equal(indices, np.array([4, 2, 0]))


def test_gif_trails_use_all_available_timesteps_between_strided_reverse_frames():
    playback_indices = _gif_playback_indices(
        5,
        time_values=np.array([40.0, 30.0, 20.0, 10.0, 0.0]),
    )
    frame_indices = _gif_frame_indices(
        5,
        frame_stride=2,
        time_values=np.array([40.0, 30.0, 20.0, 10.0, 0.0]),
    )

    np.testing.assert_array_equal(frame_indices, np.array([4, 2, 0]))
    np.testing.assert_array_equal(_gif_trail_indices(playback_indices, frame_indices[1]), np.array([4, 3, 2]))


def test_gif_frame_indices_can_preserve_simulation_order():
    indices = _gif_frame_indices(
        5,
        frame_stride=2,
        time_values=np.array([40.0, 30.0, 20.0, 10.0, 0.0]),
        time_order='simulation',
    )

    np.testing.assert_array_equal(indices, np.array([0, 2, 4]))


def test_gif_trails_use_all_available_timesteps_between_strided_simulation_frames():
    playback_indices = _gif_playback_indices(
        5,
        time_values=np.array([40.0, 30.0, 20.0, 10.0, 0.0]),
        time_order='simulation',
    )
    frame_indices = _gif_frame_indices(
        5,
        frame_stride=2,
        time_values=np.array([40.0, 30.0, 20.0, 10.0, 0.0]),
        time_order='simulation',
    )

    np.testing.assert_array_equal(frame_indices, np.array([0, 2, 4]))
    np.testing.assert_array_equal(_gif_trail_indices(playback_indices, frame_indices[1]), np.array([0, 1, 2]))


def test_gif_frame_indices_validate_stride():
    with pytest.raises(ValueError, match='gif_frame_stride'):
        _gif_frame_indices(3, frame_stride=0)


def test_plot_trajectories_can_save_optional_gif(tmp_path, monkeypatch):
    import sedtrails.pathway_visualizer.trajectories as trajectory_module

    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0], [2.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [0.5], [1.0]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
        },
        coords={
            'n_particles': np.arange(1),
            'n_timesteps': np.arange(3),
        },
    )
    observed = {}

    def fake_save_gif(sampled_ds, output, **kwargs):
        observed['sampled_ds'] = sampled_ds
        observed['output'] = output
        observed['kwargs'] = kwargs
        return output

    monkeypatch.setattr(trajectory_module, 'save_trajectory_gif', fake_save_gif)

    plot_trajectories(
        ds,
        output=tmp_path / 'trajectories.png',
        animate_gif=True,
        gif_fps=6,
        gif_dpi=90,
        gif_frame_stride=2,
        gif_time_order='simulation',
        gif_confirm_large=True,
    )

    assert observed['sampled_ds'].sizes['n_timesteps'] == 3
    assert observed['output'] == tmp_path / 'trajectories.gif'
    assert observed['kwargs']['fps'] == 6
    assert observed['kwargs']['dpi'] == 90
    assert observed['kwargs']['frame_stride'] == 2
    assert observed['kwargs']['time_order'] == 'simulation'
    assert observed['kwargs']['confirm_large_gif'] is True


def test_plot_trajectories_bare_output_filename_resolves_next_to_source(tmp_path, monkeypatch):
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
    source_dir = tmp_path / 'source'
    current_dir = tmp_path / 'current'
    source_dir.mkdir()
    current_dir.mkdir()
    ds.encoding['source'] = str(source_dir / 'results.nc')
    monkeypatch.chdir(current_dir)
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    plot_trajectories(ds, output='trajectories.png')

    assert (source_dir / 'trajectories.png').exists()
    assert not (current_dir / 'trajectories.png').exists()


def test_resolve_gif_output_bare_filename_uses_static_output_directory(tmp_path):
    gif_path = _resolve_gif_output_file(tmp_path / 'source' / 'trajectories.png', 'custom.gif')

    assert gif_path == tmp_path / 'source' / 'custom.gif'


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


def test_plot_trajectories_draws_spatial_panel_by_default(monkeypatch, tmp_path):
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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    fig, axes = plot_trajectories(ds)
    try:
        assert list(axes) == ['spatial']
        assert len(fig.axes) == 1
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_trajectories_can_draw_all_panels(monkeypatch, tmp_path):
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
    monkeypatch.chdir(tmp_path)
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
def test_sample_dataset_applies_default_particle_and_point_budgets():
    """The low-level API is bounded unless callers explicitly opt out."""
    import sedtrails.pathway_visualizer.trajectories as trajectory_module

    n_particles = trajectory_module.DEFAULT_MAX_PLOT_PARTICLES + 1
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.zeros((2, n_particles))),
            'y': (('n_timesteps', 'n_particles'), np.zeros((2, n_particles))),
            'time': (('n_timesteps',), np.array([0.0, 60.0])),
        }
    )

    sampled, total, selected = _sample_dataset(ds)

    assert total == n_particles
    assert selected == trajectory_module.DEFAULT_MAX_PLOT_PARTICLES
    assert sampled.sizes['n_particles'] == trajectory_module.DEFAULT_MAX_PLOT_PARTICLES

    time_ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.zeros((11, 5))),
            'y': (('n_timesteps', 'n_particles'), np.zeros((11, 5))),
            'time': (('n_timesteps',), np.arange(11, dtype=float)),
        }
    )
    decimated, _, _ = _sample_dataset(
        time_ds,
        max_particles=None,
        max_plot_points=20,
    )
    np.testing.assert_array_equal(decimated['time'].values, np.array([0.0, 3.0, 6.0, 10.0]))

    full_resolution, _, _ = _sample_dataset(
        time_ds,
        max_particles=None,
        max_plot_points=None,
    )
    assert full_resolution.sizes['n_timesteps'] == 11


def test_plot_trajectories_renders_selected_particles_in_bounded_batches(tmp_path, monkeypatch):
    """Rendering must not materialize one coordinate or segment batch for all tracks."""
    import sedtrails.pathway_visualizer.trajectories as trajectory_module

    n_particles = trajectory_module.DEFAULT_RENDER_PARTICLE_CHUNK + 1
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.zeros((2, n_particles))),
            'y': (('n_timesteps', 'n_particles'), np.zeros((2, n_particles))),
            'time': (('n_timesteps',), np.array([0.0, 60.0])),
            'population_id': (('n_particles',), np.zeros(n_particles, dtype=int)),
        },
        coords={'n_populations': np.arange(1)},
    )
    original_arrays = trajectory_module._trajectory_arrays
    observed_batch_sizes = []

    def recording_arrays(chunk_ds):
        arrays = original_arrays(chunk_ds)
        observed_batch_sizes.append(arrays[0].shape[0])
        return arrays

    monkeypatch.setattr(trajectory_module, '_trajectory_arrays', recording_arrays)
    fig, _ = plot_trajectories(
        ds,
        output=tmp_path / 'batched.png',
        max_particles=None,
        max_plot_points=None,
        panels='all',
        markers='none',
    )
    try:
        assert max(observed_batch_sizes) <= trajectory_module.DEFAULT_RENDER_PARTICLE_CHUNK
        assert len(observed_batch_sizes) >= 4
    finally:
        plt.close(fig)

def test_sample_dataset_caps_particle_selection_to_the_point_budget():
    """A point budget also bounds explicitly unbounded particle requests."""
    time_ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.zeros((11, 5))),
            'y': (('n_timesteps', 'n_particles'), np.zeros((11, 5))),
            'time': (('n_timesteps',), np.arange(11, dtype=float)),
        }
    )

    sampled, total, selected = _sample_dataset(
        time_ds,
        max_particles=None,
        max_plot_points=4,
    )

    assert total == 5
    assert selected == 2
    assert sampled.sizes['n_particles'] * sampled.sizes['n_timesteps'] <= 4
    np.testing.assert_array_equal(sampled['time'].values, np.array([0.0, 10.0]))


def test_plotting_public_entrypoints_preserve_existing_positional_arguments(monkeypatch, tmp_path):
    """Adding the point budget must not shift established positional parameters."""
    from sedtrails.application_interfaces import api as api_module
    import sedtrails.pathway_visualizer as visualizer_module

    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0])),
        }
    )
    monkeypatch.chdir(tmp_path)
    fig, _ = plot_trajectories(ds, None, 1, None, 0, 'none', 7.0, 'spatial', False)
    plt.close(fig)

    observed = {}

    def fake_plot(*args, **kwargs):
        observed['args'] = args
        observed['kwargs'] = kwargs

    monkeypatch.setattr(visualizer_module, 'read_netcdf', lambda _: ds)
    monkeypatch.setattr(visualizer_module, 'plot_trajectories', fake_plot)
    api_module.plot_trajectories('results.nc', None, 1, None, 0, 'none', 7.0, 'spatial', False)

    assert observed['args'] == (ds,)
    assert observed['kwargs']['markers'] == 'none'
    assert observed['kwargs']['marker_size'] == 7.0
    assert observed['kwargs']['panels'] == 'spatial'
    assert observed['kwargs']['show'] is False
