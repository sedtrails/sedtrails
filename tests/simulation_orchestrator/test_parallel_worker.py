import logging
from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc4
import numpy as np
import pytest
import yaml

from sedtrails.application_interfaces.restart import create_restart_from_netcdf
from sedtrails.data_manager.netcdf_writer import NetCDFWriter
from sedtrails.simulation_orchestrator import parallel_worker as _worker_module
from sedtrails.simulation_orchestrator.parallel_worker import _merge_outputs, _worker_fn


class _Population:
    """Small population double for NetCDF shard tests."""

    def __init__(self, name: str, x_values, pop_type: int = 0):
        """Create a population with deterministic particle arrays."""
        x_values = np.asarray(x_values, dtype=float)
        n_particles = x_values.size
        self.name = name
        self.particle_type = pop_type
        self.repr_volume = 10.0
        self.particles = {
            'x': x_values,
            'y': x_values + 100.0,
            'z': np.zeros(n_particles, dtype=float),
            'burial_depth': np.full(n_particles, 0.1, dtype=float),
            'mixing_depth': np.full(n_particles, 0.5, dtype=float),
            'status_alive': np.ones(n_particles, dtype=np.uint8),
            'status_buried': np.zeros(n_particles, dtype=np.uint8),
            'status_domain': np.ones(n_particles, dtype=np.uint8),
            'status_transported': np.ones(n_particles, dtype=np.uint8),
            'status_released': np.ones(n_particles, dtype=np.uint8),
            'status_mobile': np.ones(n_particles, dtype=np.uint8),
        }


def _write_track_shard(base: Path, task_id: int, populations: list[_Population], times=(0.0, 10.0)) -> None:
    """Write one trajectory_v2 worker result shard."""
    output_dir = base / f'task_{task_id:04d}'
    writer = NetCDFWriter(output_dir)
    total_particles = sum(len(pop.particles['x']) for pop in populations)
    handle = writer.open_output(
        'sedtrails_results.nc',
        len(times),
        total_particles,
        len(populations),
        1,
        populations,
        ['velocity'],
        compression=False,
    )
    handle.reference_date = '2020-01-01 00:00:00'
    handle.time_units = 'seconds since 2020-01-01 00:00:00'
    handle['time'].units = handle.time_units
    for slot, time_value in enumerate(times):
        writer.record_output(handle, populations, slot, time_value)
    writer.close_output(handle)


def _write_end_position_shard(base: Path, task_id: int, populations: list[_Population]) -> None:
    """Write one end_positions_v1 worker result shard."""
    output_dir = base / f'task_{task_id:04d}'
    writer = NetCDFWriter(output_dir)
    writer.write_end_positions(
        'sedtrails_results.nc',
        populations,
        current_time=99.0,
        reference_date='2020-01-01 00:00:00',
        time_units='seconds since 2020-01-01 00:00:00',
        compression=False,
    )


def _write_checkpoint_shard(base: Path, task_id: int, populations: list[_Population]) -> None:
    """Write one checkpoint_v1 worker shard."""
    output_dir = base / f'task_{task_id:04d}'
    writer = NetCDFWriter(output_dir)
    writer.write_checkpoint(
        'sedtrails_checkpoint.nc',
        populations,
        current_time=123.0,
        reference_date='2020-01-01 00:00:00',
        time_units='seconds since 2020-01-01 00:00:00',
        compression=False,
    )


def _write_restart_base_config(path: Path) -> None:
    """Write a minimal two-population config for restart generation tests."""
    config = {
        'time': {
            'start': '2020-01-01 00:00:00',
            'timestep': '10S',
            'duration': '1H',
        },
        'particles': {
            'populations': [
                {
                    'name': 'fine',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'burial_depth': {'constant': 0.0},
                        'strategy': {
                            'random': {
                                'bbox': '0,0 1,1',
                                'nlocations': 1,
                            }
                        },
                    },
                },
                {
                    'name': 'coarse',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'burial_depth': {'constant': 0.0},
                        'strategy': {
                            'random': {
                                'bbox': '0,0 1,1',
                                'nlocations': 1,
                            }
                        },
                    },
                },
            ]
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')


def _read_restart_points(path: Path) -> np.ndarray:
    """Read restart seed coordinates from a generated file_points CSV."""
    return np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)


def test_merge_outputs_preserves_time_major_numeric_ids(tmp_path):
    """Merge trajectory shards without reverting numeric trajectory IDs."""
    _write_track_shard(
        tmp_path,
        0,
        [_Population('fine', [1.0, 2.0]), _Population('coarse', [10.0])],
    )
    _write_track_shard(
        tmp_path,
        1,
        [_Population('fine', [3.0]), _Population('coarse', [11.0, 12.0])],
    )

    _merge_outputs(tmp_path, logging.getLogger(__name__))

    with nc4.Dataset(tmp_path / 'sedtrails_results.nc') as ds:
        assert ds.trajectory_layout == 'time_particle'
        assert ds.sedtrails_output_schema == 'trajectory_v2'
        assert ds.dimensions['n_particles'].size == 6
        assert ds['x'].dimensions == ('n_timesteps', 'n_particles')
        assert ds['trajectory_id'].dimensions == ('n_particles',)
        assert ds['trajectory_id'].dtype == np.dtype('int64')
        np.testing.assert_array_equal(ds['trajectory_id'][:], np.arange(6))
        np.testing.assert_array_equal(ds['population_count'][:], np.array([3, 3]))
        np.testing.assert_array_equal(ds['population_start_idx'][:], np.array([0, 3]))
        np.testing.assert_allclose(ds['x'][0, :], np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0]))

    assert not list(tmp_path.glob('task_*'))


def test_merge_outputs_merges_checkpoint_shards(tmp_path):
    """Merge checkpoint shards alongside worker result files."""
    task0_populations = [_Population('fine', [1.0]), _Population('coarse', [10.0, 11.0])]
    task1_populations = [_Population('fine', [2.0, 3.0]), _Population('coarse', [12.0])]
    _write_track_shard(tmp_path, 0, task0_populations)
    _write_track_shard(tmp_path, 1, task1_populations)
    _write_checkpoint_shard(tmp_path, 0, task0_populations)
    _write_checkpoint_shard(tmp_path, 1, task1_populations)

    _merge_outputs(tmp_path, logging.getLogger(__name__))

    with nc4.Dataset(tmp_path / 'sedtrails_checkpoint.nc') as ds:
        assert ds.sedtrails_file_kind == 'checkpoint'
        assert ds.sedtrails_output_schema == 'checkpoint_v1'
        assert ds.trajectory_layout == 'checkpoint'
        assert ds.dimensions['n_particles'].size == 6
        assert ds['x'].dimensions == ('n_particles',)
        np.testing.assert_array_equal(ds['trajectory_id'][:], np.arange(6))
        np.testing.assert_array_equal(ds['population_count'][:], np.array([3, 3]))
        np.testing.assert_array_equal(ds['population_start_idx'][:], np.array([0, 3]))
        np.testing.assert_allclose(ds['x'][:], np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0]))


def test_restart_config_from_parallel_merged_results(tmp_path):
    """Create restart seed files from merged parallel trajectory output."""
    _write_track_shard(
        tmp_path,
        0,
        [_Population('fine', [1.0, 2.0]), _Population('coarse', [10.0])],
    )
    _write_track_shard(
        tmp_path,
        1,
        [_Population('fine', [3.0]), _Population('coarse', [11.0, 12.0])],
    )
    _merge_outputs(tmp_path, logging.getLogger(__name__))

    config_path = tmp_path / 'base.yaml'
    output_config_path = tmp_path / 'restart.yaml'
    seeds_dir = tmp_path / 'restart_points'
    _write_restart_base_config(config_path)

    summary = create_restart_from_netcdf(
        str(tmp_path / 'sedtrails_results.nc'),
        str(config_path),
        str(output_config_path),
        seed_points_dir=str(seeds_dir),
    )

    assert summary.restart_time == '2020-01-01 00:00:10'
    assert summary.retained_particles == 6
    assert set(summary.seed_files) == {'fine', 'coarse'}
    np.testing.assert_allclose(
        _read_restart_points(seeds_dir / 'fine.restart_points.csv'),
        np.array([[1.0, 101.0], [2.0, 102.0], [3.0, 103.0]]),
    )
    np.testing.assert_allclose(
        _read_restart_points(seeds_dir / 'coarse.restart_points.csv'),
        np.array([[10.0, 110.0], [11.0, 111.0], [12.0, 112.0]]),
    )

    restart_config = yaml.safe_load(output_config_path.read_text(encoding='utf-8'))
    assert restart_config['time']['start'] == '2020-01-01 00:00:10'
    assert restart_config['time']['duration'] == '59M50S'
    for population in restart_config['particles']['populations']:
        assert population['seeding']['quantity'] == 1
        assert 'file_points' in population['seeding']['strategy']


def test_restart_config_from_parallel_merged_checkpoint(tmp_path):
    """Create restart seed files from merged parallel checkpoint output."""
    task0_populations = [_Population('fine', [1.0]), _Population('coarse', [10.0, 11.0])]
    task1_populations = [_Population('fine', [2.0, 3.0]), _Population('coarse', [12.0])]
    _write_track_shard(tmp_path, 0, task0_populations)
    _write_track_shard(tmp_path, 1, task1_populations)
    _write_checkpoint_shard(tmp_path, 0, task0_populations)
    _write_checkpoint_shard(tmp_path, 1, task1_populations)
    _merge_outputs(tmp_path, logging.getLogger(__name__))

    config_path = tmp_path / 'base.yaml'
    output_config_path = tmp_path / 'restart.yaml'
    seeds_dir = tmp_path / 'restart_points'
    _write_restart_base_config(config_path)

    summary = create_restart_from_netcdf(
        str(tmp_path / 'sedtrails_checkpoint.nc'),
        str(config_path),
        str(output_config_path),
        seed_points_dir=str(seeds_dir),
    )

    assert summary.restart_time == '2020-01-01 00:02:03'
    assert summary.retained_particles == 6
    assert set(summary.seed_files) == {'fine', 'coarse'}
    np.testing.assert_allclose(
        _read_restart_points(seeds_dir / 'fine.restart_points.csv'),
        np.array([[1.0, 101.0], [2.0, 102.0], [3.0, 103.0]]),
    )
    np.testing.assert_allclose(
        _read_restart_points(seeds_dir / 'coarse.restart_points.csv'),
        np.array([[10.0, 110.0], [11.0, 111.0], [12.0, 112.0]]),
    )

    restart_config = yaml.safe_load(output_config_path.read_text(encoding='utf-8'))
    assert restart_config['time']['start'] == '2020-01-01 00:02:03'
    assert restart_config['time']['duration'] == '57M57S'


def test_merge_outputs_handles_end_position_shards(tmp_path):
    """Merge compact end-position shards without assuming an n_timesteps dimension."""
    _write_end_position_shard(
        tmp_path,
        0,
        [_Population('fine', [1.0]), _Population('coarse', [10.0, 11.0])],
    )
    _write_end_position_shard(
        tmp_path,
        1,
        [_Population('fine', [2.0, 3.0]), _Population('coarse', [12.0])],
    )

    _merge_outputs(tmp_path, logging.getLogger(__name__))

    with nc4.Dataset(tmp_path / 'sedtrails_results.nc') as ds:
        assert ds.sedtrails_file_kind == 'end_positions'
        assert ds.trajectory_layout == 'end_positions'
        assert 'n_timesteps' not in ds.dimensions
        assert ds['time'].dimensions == ()
        assert ds['x'].dimensions == ('n_particles',)
        assert ds['trajectory_id'].dimensions == ('n_particles',)
        np.testing.assert_array_equal(ds['trajectory_id'][:], np.arange(6))
        np.testing.assert_array_equal(ds['population_count'][:], np.array([3, 3]))
        np.testing.assert_array_equal(ds['population_start_idx'][:], np.array([0, 3]))
        np.testing.assert_allclose(ds['x'][:], np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0]))


def test_restart_config_from_parallel_merged_end_positions(tmp_path):
    """Create restart seed files from merged compact parallel result output."""
    _write_end_position_shard(
        tmp_path,
        0,
        [_Population('fine', [1.0]), _Population('coarse', [10.0, 11.0])],
    )
    _write_end_position_shard(
        tmp_path,
        1,
        [_Population('fine', [2.0, 3.0]), _Population('coarse', [12.0])],
    )
    _merge_outputs(tmp_path, logging.getLogger(__name__))

    config_path = tmp_path / 'base.yaml'
    output_config_path = tmp_path / 'restart.yaml'
    seeds_dir = tmp_path / 'restart_points'
    _write_restart_base_config(config_path)

    summary = create_restart_from_netcdf(
        str(tmp_path / 'sedtrails_results.nc'),
        str(config_path),
        str(output_config_path),
        seed_points_dir=str(seeds_dir),
    )

    assert summary.restart_time == '2020-01-01 00:01:39'
    assert summary.retained_particles == 6
    np.testing.assert_allclose(
        _read_restart_points(seeds_dir / 'fine.restart_points.csv'),
        np.array([[1.0, 101.0], [2.0, 102.0], [3.0, 103.0]]),
    )
    np.testing.assert_allclose(
        _read_restart_points(seeds_dir / 'coarse.restart_points.csv'),
        np.array([[10.0, 110.0], [11.0, 111.0], [12.0, 112.0]]),
    )


def test_merge_outputs_rejects_incompatible_time_vectors(tmp_path):
    """Fail clearly when trajectory shards do not describe the same output times."""
    _write_track_shard(tmp_path, 0, [_Population('fine', [1.0])], times=(0.0, 10.0))
    _write_track_shard(tmp_path, 1, [_Population('fine', [2.0])], times=(0.0, 20.0))

    with pytest.raises(ValueError, match='time values'):
        _merge_outputs(tmp_path, logging.getLogger(__name__))


def test_worker_fn_forces_dashboard_off(monkeypatch):
    """Parallel workers should not create GUI dashboards."""
    from sedtrails.simulation_orchestrator import simulation_manager

    calls = []

    class DummySimulation:
        """Capture worker simulation construction."""

        def __init__(self, config_file, enable_dashboard=None):
            """Store construction arguments."""
            calls.append(('init', config_file, enable_dashboard))
            self.format_converter = SimpleNamespace(
                format_plugin=SimpleNamespace(input_data=None)
            )

        def run(self):
            """Capture worker run invocation."""
            calls.append(('run',))

    monkeypatch.setattr(simulation_manager, 'Simulation', DummySimulation)
    monkeypatch.setattr(_worker_module, '_set_worker_memory_limit', lambda n_tasks: None)
    monkeypatch.setattr(_worker_module, '_PRELOADED_INPUT_DATA', None)

    _worker_fn('config.yml', task_id=1, n_tasks=3)

    assert calls == [('init', 'config.yml', False), ('run',)]
