import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest
import xarray as xr

from sedtrails.pathway_visualizer.trajectories import _trajectory_arrays, plot_trajectories, read_netcdf

def test_plot_trajectories_accepts_fixed_width_population_names(monkeypatch):
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
    plot_trajectories(ds)


def test_plot_trajectories_accepts_time_major_layout(monkeypatch):
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

    plot_trajectories(ds)


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
