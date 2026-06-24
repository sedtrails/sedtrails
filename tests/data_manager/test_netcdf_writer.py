import numpy as np
import pytest
import xarray as xr

from sedtrails.data_manager.netcdf_writer import NetCDFWriter


@pytest.fixture
def tmp_output_dir(tmp_path):
    """
    Pytest fixture to provide a temporary output directory for NetCDF files.
    """
    return tmp_path / 'output'


class MockPopulation:
    """Mock population class for testing."""

    def __init__(self, name, particle_type=0):
        self.name = name
        self.particle_type = particle_type
        self.particles = {
            'x': np.array([1.0, 2.0, 3.0]),
            'y': np.array([1.5, 2.5, 3.5]),
            'burial_depth': np.array([0.1, 0.2, 0.0]),
            'mixing_depth': np.array([0.5, 0.6, 0.4]),
            'status_mobile': np.array([1, 0, 1], dtype=np.int32),
            'status_alive': np.array([1, 1, 1], dtype=np.int32),
            'status_buried': np.array([0, 0, 0], dtype=np.int32),
            'status_domain': np.array([1, 1, 1], dtype=np.int32),
            'status_transported': np.array([0, 1, 0], dtype=np.int32),
            'status_released': np.array([1, 1, 1], dtype=np.int32),
        }


# ---------------------------------------------------------------------------
# Streaming output tests
# ---------------------------------------------------------------------------

class TestNetCDFWriterStreaming:
    """Tests for open_output / record_output / close_output."""

    N_PARTICLES = 3
    N_SLOTS = 4
    N_POPULATIONS = 1
    N_FLOWFIELDS = 1

    @pytest.fixture
    def writer(self, tmp_path):
        return NetCDFWriter(tmp_path / 'output')

    @pytest.fixture
    def population(self):
        return MockPopulation('test_pop')

    @pytest.fixture
    def open_handle(self, writer, population, tmp_path):
        """Open a streaming file and yield the handle; close in teardown."""
        handle = writer.open_output(
            'stream.nc',
            self.N_SLOTS,
            self.N_PARTICLES,
            self.N_POPULATIONS,
            self.N_FLOWFIELDS,
            [population],
            ['water_velocity'],
        )
        yield handle
        if handle.isopen():
            handle.close()

    def test_open_creates_file_with_correct_dimensions(self, open_handle):
        ds = open_handle
        assert ds.dimensions['n_particles'].size == self.N_PARTICLES
        assert ds.dimensions['n_timesteps'].size == self.N_SLOTS
        assert ds.dimensions['n_populations'].size == self.N_POPULATIONS
        assert ds.dimensions['n_flowfields'].size == self.N_FLOWFIELDS
        assert ds.trajectory_layout == 'time_particle'

    def test_open_creates_all_trajectory_variables(self, open_handle):
        expected = {'x', 'y', 'z', 'time', 'burial_depth', 'mixing_depth',
                    'status_alive', 'status_buried', 'status_domain',
                    'status_transported', 'status_released', 'status_mobile'}
        assert expected.issubset(set(open_handle.variables))
        assert 'covered_distance' not in open_handle.variables
        assert open_handle['time'].dimensions == ('n_timesteps',)
        assert open_handle['x'].dimensions == ('n_timesteps', 'n_particles')
        assert open_handle['x'].dtype == np.dtype('float32')
        assert open_handle['status_mobile'].dtype == np.dtype('uint8')
        assert open_handle['trajectory_id'].dimensions == ('n_particles',)
        np.testing.assert_array_equal(open_handle['trajectory_id'][:], np.arange(self.N_PARTICLES))
        assert open_handle['x'].chunking() == [1, self.N_PARTICLES]

    def test_open_writes_population_metadata(self, open_handle):
        assert open_handle['population_count'][0] == self.N_PARTICLES
        assert open_handle['population_start_idx'][0] == 0

    def test_open_writes_flowfield_metadata(self, open_handle):
        name_chars = open_handle['flowfield_name'][0, :].data
        name = b''.join(name_chars).decode('ascii').strip()
        assert name == 'water_velocity'

    def test_record_writes_coordinates_to_correct_slot(self, writer, population):
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )
        writer.record_output(handle, [population], slot_idx=0, current_time=100.0)
        writer.record_output(handle, [population], slot_idx=1, current_time=200.0)

        np.testing.assert_array_almost_equal(handle['x'][0, :], population.particles['x'])
        np.testing.assert_array_almost_equal(handle['x'][1, :], population.particles['x'])
        assert handle['time'][0] == pytest.approx(100.0)
        assert handle['time'][1] == pytest.approx(200.0)
        handle.close()

    def test_record_writes_status_fields(self, writer, population):
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )
        writer.record_output(handle, [population], slot_idx=0, current_time=0.0)

        np.testing.assert_array_equal(
            handle['status_mobile'][0, :], population.particles['status_mobile']
        )
        handle.close()

    def test_record_writes_minimal_q3d_fields(self, writer, population):
        population.particles['z_p'] = np.array([0.2, 0.3, 0.4])
        population.particles['z_burial'] = np.array([0.9, 0.8, 1.0])
        population.particles['horizontal_particle_velocity'] = np.array([1.0, 2.0, 3.0])
        population.particles['q3d_motion_substeps'] = np.array([2, 2, 2])
        population.particles['status_suspended'] = np.array([True, False, True])
        population.particles['status_deposited'] = np.array([False, True, False])
        population.particles['status_available_for_entrainment'] = np.array([False, True, False])
        population.particles['status_entrained_now'] = np.array([True, False, False])
        population.particles['status_deposited_now'] = np.array([False, True, False])
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )

        writer.record_output(handle, [population], slot_idx=0, current_time=0.0)

        np.testing.assert_allclose(handle['z_p'][0, :], population.particles['z_p'])
        np.testing.assert_allclose(handle['z_burial'][0, :], population.particles['z_burial'])
        np.testing.assert_allclose(
            handle['horizontal_particle_velocity'][0, :],
            population.particles['horizontal_particle_velocity'],
        )
        np.testing.assert_array_equal(
            handle['q3d_motion_substeps'][0, :],
            population.particles['q3d_motion_substeps'],
        )
        for status_name in (
            'status_suspended',
            'status_deposited',
            'status_available_for_entrainment',
            'status_entrained_now',
            'status_deposited_now',
        ):
            np.testing.assert_array_equal(
                handle[status_name][0, :],
                population.particles[status_name],
            )
        assert not any(name.startswith('is_') for name in handle.variables)
        assert 'diagnostic_water_depth' not in handle.variables
        handle.close()

    def test_full_q3d_diagnostics_are_opt_in(self, writer, population):
        population.particles['diagnostic_water_depth'] = np.array([4.0, 5.0, 6.0])
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
            q3d_diagnostics='full',
        )

        writer.record_output(handle, [population], slot_idx=0, current_time=0.0)

        np.testing.assert_allclose(
            handle['diagnostic_water_depth'][0, :],
            population.particles['diagnostic_water_depth'],
        )
        handle.close()

    def test_unwritten_slots_are_fill_values(self, writer, population):
        """Slots not yet written should contain the declared fill value, not zeros."""
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )
        writer.record_output(handle, [population], slot_idx=0, current_time=0.0)
        # slot 1 is unwritten; returned as a masked array (fill_value=NaN)
        slot1 = handle['x'][1, :]
        assert np.all(np.ma.getmaskarray(slot1))
        handle.close()

    def test_close_returns_correct_path(self, writer, population):
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )
        path = writer.close_output(handle)
        assert path.name == 'stream.nc'
        assert path.exists()

    def test_streaming_round_trip(self, writer, population, tmp_path):
        """Full open-record-close cycle produces a valid, readable NetCDF file."""
        times = [0.0, 3600.0, 7200.0]
        handle = writer.open_output(
            'round_trip.nc', len(times), self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['water_velocity'],
        )
        for slot, t in enumerate(times):
            writer.record_output(handle, [population], slot_idx=slot, current_time=t)
        path = writer.close_output(handle)

        # Read back with xarray and verify
        ds = xr.open_dataset(path, engine='netcdf4')
        assert ds.sizes['n_timesteps'] == len(times)
        assert ds.sizes['n_particles'] == self.N_PARTICLES
        assert ds.attrs['trajectory_layout'] == 'time_particle'
        np.testing.assert_array_almost_equal(ds['time'].values, times)
        np.testing.assert_array_almost_equal(
            ds['x'].values[0, :], population.particles['x']  # all particles at slot 0
        )
        ds.close()

    def test_two_populations_particle_offsets(self, writer, tmp_path):
        """Particles from separate populations must land in consecutive index ranges."""
        pop_a = MockPopulation('pop_a')
        pop_b = MockPopulation('pop_b')
        pop_b.particles['x'] = np.array([10.0, 20.0, 30.0])
        n_total = len(pop_a.particles['x']) + len(pop_b.particles['x'])

        handle = writer.open_output(
            'two_pops.nc', 2, n_total, 2, 1, [pop_a, pop_b], ['vel'],
        )
        writer.record_output(handle, [pop_a, pop_b], slot_idx=0, current_time=0.0)

        np.testing.assert_array_almost_equal(handle['x'][0, :3], pop_a.particles['x'])
        np.testing.assert_array_almost_equal(handle['x'][0, 3:], pop_b.particles['x'])
        handle.close()

    def test_write_checkpoint_stores_current_particle_state(self, writer, population):
        path = writer.write_checkpoint(
            'sedtrails_checkpoint.nc',
            [population],
            current_time=123.0,
            reference_date='2020-01-01 00:00:00',
            time_units='seconds since 2020-01-01 00:00:00',
        )

        ds = xr.open_dataset(path, engine='netcdf4')
        assert ds.attrs['sedtrails_file_kind'] == 'checkpoint'
        assert ds.attrs['reference_date'] == '2020-01-01 00:00:00'
        assert ds.sizes['n_particles'] == self.N_PARTICLES
        assert ds['x'].dims == ('n_particles',)
        assert float(ds['time'].values) == pytest.approx(123.0)
        np.testing.assert_array_almost_equal(ds['x'].values, population.particles['x'])
        np.testing.assert_array_equal(ds['population_id'].values, np.zeros(self.N_PARTICLES, dtype=int))
        ds.close()

    def test_write_end_positions_stores_compact_result_state(self, writer, population):
        """End-position results should use one particle dimension and no trajectory cube."""
        path = writer.write_end_positions(
            'sedtrails_results.nc',
            [population],
            current_time=456.0,
            reference_date='2020-01-01 00:00:00',
            time_units='seconds since 2020-01-01 00:00:00',
        )

        ds = xr.open_dataset(path, engine='netcdf4')
        assert ds.attrs['sedtrails_file_kind'] == 'end_positions'
        assert ds.attrs['sedtrails_output_schema'] == 'end_positions_v1'
        assert ds.attrs['trajectory_layout'] == 'end_positions'
        assert ds.sizes['n_particles'] == self.N_PARTICLES
        assert 'n_timesteps' not in ds.sizes
        assert ds['x'].dims == ('n_particles',)
        assert float(ds['time'].values) == pytest.approx(456.0)
        np.testing.assert_array_almost_equal(ds['x'].values, population.particles['x'])
        ds.close()
