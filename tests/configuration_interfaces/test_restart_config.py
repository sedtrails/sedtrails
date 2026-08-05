from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

import sedtrails.application_interfaces.restart as restart_module
from sedtrails.application_interfaces.restart import create_restart_from_netcdf


def _trajectory_dataset(
    *,
    x,
    y,
    time,
    population_id=None,
    status_alive=None,
    status_domain=None,
    attrs=None,
):
    """Create a SedTRAILS v2 time-major trajectory fixture."""
    time_values = np.asarray(time, dtype=float)
    data_vars = {
        'x': (('n_timesteps', 'n_particles'), np.asarray(x)),
        'y': (('n_timesteps', 'n_particles'), np.asarray(y)),
        'time': (('n_timesteps',), time_values),
    }
    if population_id is not None:
        data_vars['population_id'] = (('n_particles',), np.asarray(population_id, dtype=int))
    if status_alive is not None:
        data_vars['status_alive'] = (('n_timesteps', 'n_particles'), np.asarray(status_alive))
    if status_domain is not None:
        data_vars['status_domain'] = (('n_timesteps', 'n_particles'), np.asarray(status_domain))

    output_attrs = {'trajectory_layout': 'time_particle', 'written_slots': int(time_values.size)}
    if attrs:
        output_attrs.update(attrs)

    return xr.Dataset(data_vars=data_vars, attrs=output_attrs)


def test_open_restart_dataset_uses_netcdf4_engine(monkeypatch, tmp_path):
    observed = {}

    def fake_open_dataset(path, **kwargs):
        observed['path'] = path
        observed['kwargs'] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(restart_module.xr, 'open_dataset', fake_open_dataset)
    netcdf_file = tmp_path / 'results.nc'

    ds = restart_module._open_restart_dataset(netcdf_file)

    assert isinstance(ds, xr.Dataset)
    assert observed['path'] == netcdf_file
    assert observed['kwargs']['engine'] == 'netcdf4'
    assert observed['kwargs']['decode_times'] is False


def test_create_restart_rejects_forcing_file_with_clear_message(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }
    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    forcing = xr.Dataset(
        data_vars={
            'bedlevel': (('time', 'mesh2d_nFaces'), np.zeros((1, 2))),
            'sea_water_x_velocity': (('time', 'mesh2d_nFaces'), np.ones((1, 2))),
        }
    )
    forcing_file = tmp_path / 'forcing.nc'
    forcing.to_netcdf(forcing_file)

    with pytest.raises(ValueError, match='not a SedTRAILS trajectory output file.*Eulerian forcing files'):
        create_restart_from_netcdf(
            netcdf_file=str(forcing_file),
            base_config_file=str(config_file),
            output_config_file=str(tmp_path / 'restart.yaml'),
        )


def test_create_restart_from_netcdf_generates_yaml_and_seed_files(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
                {
                    'name': 'population_2',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array(
            [
                [0.0, 5.0, 10.0],
                [1.0, 6.0, 11.0],
                [2.0, np.nan, 12.0],
            ]
        ),
        y=np.array(
            [
                [0.0, 4.0, 7.0],
                [0.5, 4.5, 7.5],
                [1.0, np.nan, 8.0],
            ]
        ),
        time=np.array([0.0, 60.0, 120.0]),
        population_id=np.array([0, 0, 1], dtype=int),
        status_alive=np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        ),
        status_domain=np.ones((3, 3), dtype=float),
    )

    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    out_config = tmp_path / 'restart.yaml'
    out_seed_dir = tmp_path / 'restart_points'

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(out_config),
        seed_points_dir=str(out_seed_dir),
    )

    assert summary.output_config == out_config
    assert summary.restart_time == '2020-01-01 00:02:00'
    assert summary.retained_particles == 2

    assert (out_seed_dir / 'population_1.restart_points.csv').exists()
    assert (out_seed_dir / 'population_2.restart_points.csv').exists()

    with open(out_config, 'r', encoding='utf-8') as handle:
        restart_cfg = yaml.safe_load(handle)

    assert restart_cfg['time']['start'] == '2020-01-01 00:02:00'
    assert restart_cfg['time']['duration'] == '23H58M'

    pop1 = restart_cfg['particles']['populations'][0]
    pop2 = restart_cfg['particles']['populations'][1]

    assert pop1['seeding']['release_start'] == '2020-01-01 00:02:00'
    assert pop1['seeding']['quantity'] == 1
    assert 'file_points' in pop1['seeding']['strategy']

    assert pop2['seeding']['release_start'] == '2020-01-01 00:02:00'
    assert pop2['seeding']['quantity'] == 1
    assert 'file_points' in pop2['seeding']['strategy']

    pop1_file = Path(pop1['seeding']['strategy']['file_points']['path'])
    pop2_file = Path(pop2['seeding']['strategy']['file_points']['path'])

    assert pop1_file.name == 'population_1.restart_points.csv'
    assert pop2_file.name == 'population_2.restart_points.csv'


def test_create_restart_from_time_major_netcdf_reads_final_written_slot(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
                {
                    'name': 'population_2',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = xr.Dataset(
        data_vars={
            'x': (
                ('n_timesteps', 'n_particles'),
                np.array(
                    [
                        [0.0, 5.0, 10.0],
                        [1.0, 6.0, 11.0],
                        [2.0, 7.0, 12.0],
                        [np.nan, np.nan, np.nan],
                    ],
                    dtype=np.float32,
                ),
            ),
            'y': (
                ('n_timesteps', 'n_particles'),
                np.array(
                    [
                        [0.0, 4.0, 7.0],
                        [0.5, 4.5, 7.5],
                        [1.0, 5.0, 8.0],
                        [np.nan, np.nan, np.nan],
                    ],
                    dtype=np.float32,
                ),
            ),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0, np.nan])),
            'population_id': (('n_particles',), np.array([0, 0, 1], dtype=int)),
            'status_alive': (
                ('n_timesteps', 'n_particles'),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 0, 1],
                        [0, 0, 0],
                    ],
                    dtype=np.uint8,
                ),
            ),
            'status_domain': (
                ('n_timesteps', 'n_particles'),
                np.ones((4, 3), dtype=np.uint8),
            ),
        },
        attrs={'trajectory_layout': 'time_particle', 'written_slots': 3},
    )

    netcdf_file = tmp_path / 'results_v2.nc'
    ds.to_netcdf(netcdf_file)

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert summary.restart_time == '2020-01-01 00:02:00'
    assert summary.retained_particles == 2


def test_create_restart_from_checkpoint_netcdf(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
                {
                    'name': 'population_2',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles',), np.array([2.0, 12.0], dtype=np.float32)),
            'y': (('n_particles',), np.array([1.0, 8.0], dtype=np.float32)),
            'time': ((), 180.0),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
            'status_alive': (('n_particles',), np.array([1, 1], dtype=np.uint8)),
            'status_domain': (('n_particles',), np.array([1, 1], dtype=np.uint8)),
            'z': (('n_particles',), np.array([-1.2, -2.3])),
            'z_p': (('n_particles',), np.array([0.4, 0.5])),
            'burial_depth': (('n_particles',), np.array([0.0, 0.0])),
            'status_suspended': (('n_particles',), np.array([1, 1], dtype=np.uint8)),
            'status_deposited': (('n_particles',), np.array([0, 0], dtype=np.uint8)),
            'status_buried': (('n_particles',), np.array([0, 0], dtype=np.uint8)),
            'vertical_position_initialized': (('n_particles',), np.array([1, 1], dtype=np.uint8)),
        },
        attrs={'sedtrails_file_kind': 'checkpoint'},
    )

    netcdf_file = tmp_path / 'checkpoint.nc'
    ds.to_netcdf(netcdf_file)

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert summary.restart_time == '2020-01-01 00:03:00'
    assert summary.retained_particles == 2
    seed_text = summary.seed_files['population_1'].read_text(encoding='utf-8')
    assert seed_text.splitlines()[0].startswith('x,y,z,z_p,burial_depth')
    assert '-1.2,0.4' in seed_text


def test_restart_state_rejects_particle_major_legacy_output():
    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'y': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'time': (('n_particles', 'n_timesteps'), np.array([[0.0, 60.0]])),
            'population_id': (('n_particles',), np.array([0], dtype=int)),
        }
    )

    with pytest.raises(ValueError, match='Legacy particle-major trajectory files are not supported'):
        restart_module._extract_restart_state(ds)


def test_create_restart_uses_reference_date_for_netcdf_time(tmp_path):
    base_config = {
        'general': {'input_model': {'reference_date': '1970-01-01'}},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:30:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        # 2016-09-21 19:40:28 UTC in epoch seconds
        time=np.array([1474485628.0, 1474486828.0]),
        population_id=np.array([0], dtype=int),
    )

    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    out_config = tmp_path / 'restart.yaml'

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(out_config),
    )

    assert summary.restart_time == '2016-09-21 19:40:28'

    with open(out_config, 'r', encoding='utf-8') as handle:
        restart_cfg = yaml.safe_load(handle)
    assert restart_cfg['time']['duration'] == '23H39M32S'
    assert restart_cfg['general']['input_model']['reference_date'] == '1970-01-01'


def test_create_restart_keeps_cf_time_values_as_seconds(tmp_path):
    base_config = {
        'general': {'input_model': {'reference_date': '1970-01-01'}},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:20:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        time=np.array([1474485600.0, 1474489200.0]),
        population_id=np.array([0], dtype=int),
        attrs={'reference_date': '1970-01-01', 'time_units': 'seconds since 1970-01-01'},
    )
    ds['time'].attrs['units'] = 'seconds since 1970-01-01'
    ds['time'].attrs['reference_date'] = '1970-01-01'
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert summary.restart_time == '2016-09-21 20:20:00'


def test_create_restart_validates_generated_time_against_original_forcing(tmp_path, monkeypatch):
    """A 1970 restart generated for 2016 forcing should fail before writing a bad config."""

    class FakeFormatConverter:
        def __init__(self, config):
            self.config = config

        def get_time_bounds(self):
            return 1474485600.0, 1474489200.0

    monkeypatch.setattr(restart_module, 'FormatConverter', FakeFormatConverter)

    base_config = {
        'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
        'inputs': {'data': 'forcing.nc'},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:20:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0]]),
        y=np.array([[0.0]]),
        time=np.array([15400.0]),
        population_id=np.array([0], dtype=int),
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    out_config = tmp_path / 'restart.yaml'
    with pytest.raises(ValueError, match='outside the original input forcing window'):
        create_restart_from_netcdf(
            netcdf_file=str(netcdf_file),
            base_config_file=str(config_file),
            output_config_file=str(out_config),
        )

    assert not out_config.exists()


def test_create_restart_with_valid_forcing_preserves_original_reference_date(tmp_path, monkeypatch):
    """Valid restart generation should keep the base input-model reference date unchanged."""

    class FakeFormatConverter:
        def __init__(self, config):
            self.config = config

        def get_time_bounds(self):
            return 1474485600.0, 1474489200.0

    monkeypatch.setattr(restart_module, 'FormatConverter', FakeFormatConverter)

    base_config = {
        'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
        'inputs': {'data': 'forcing.nc'},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:20:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        time=np.array([1474485600.0, 1474486828.0]),
        population_id=np.array([0], dtype=int),
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    out_config = tmp_path / 'restart.yaml'
    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(out_config),
    )

    assert summary.restart_time == '2016-09-21 19:40:28'

    with open(out_config, 'r', encoding='utf-8') as handle:
        restart_cfg = yaml.safe_load(handle)
    assert restart_cfg['time']['start'] == '2016-09-21 19:40:28'
    assert restart_cfg['general']['input_model']['reference_date'] == '1970-01-01'


def test_create_restart_prefers_output_time_units_reference_date(tmp_path, monkeypatch):
    """Future output metadata can define the time axis reference date explicitly."""

    class FakeFormatConverter:
        def __init__(self, config):
            self.config = config

        def get_time_bounds(self):
            return 1474485600.0, 1474489200.0

    monkeypatch.setattr(restart_module, 'FormatConverter', FakeFormatConverter)

    base_config = {
        'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
        'inputs': {'data': 'forcing.nc'},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:20:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        time=np.array([0.0, 1228.0]),
        population_id=np.array([0], dtype=int),
        attrs={'time_units': 'seconds since 2016-09-21 19:20:00'},
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert summary.restart_time == '2016-09-21 19:40:28'


def test_create_restart_writes_seed_paths_relative_to_cwd(tmp_path, monkeypatch):
    examples_dir = tmp_path / 'examples'
    examples_dir.mkdir(parents=True)

    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'populaton_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = examples_dir / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        time=np.array([0.0, 60.0]),
        population_id=np.array([0], dtype=int),
    )
    netcdf_file = examples_dir / 'results.nc'
    ds.to_netcdf(netcdf_file)

    monkeypatch.chdir(tmp_path)
    out_config = examples_dir / 'restart.yaml'

    create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(out_config),
    )

    with open(out_config, 'r', encoding='utf-8') as handle:
        restart_cfg = yaml.safe_load(handle)

    seed_path = restart_cfg['particles']['populations'][0]['seeding']['strategy']['file_points']['path']
    assert seed_path == './examples/restart_seeds/populaton_1.restart_points.csv'


def test_create_restart_raises_if_no_remaining_duration(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        # Last available time is exactly at end of original duration.
        time=np.array([0.0, 86400.0]),
        population_id=np.array([0], dtype=int),
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    with pytest.raises(ValueError, match='No remaining duration'):
        create_restart_from_netcdf(
            netcdf_file=str(netcdf_file),
            base_config_file=str(config_file),
            output_config_file=str(tmp_path / 'restart.yaml'),
        )
