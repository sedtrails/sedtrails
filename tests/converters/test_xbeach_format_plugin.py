"""Tests for the XBeach format plugin."""

import numpy as np
import pytest
import xarray as xr

from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.particle_tracer.particle_seeder import ParticleSeeder
from sedtrails.transport_converter.format_converter import FormatConverter
from sedtrails.transport_converter.plugins.format import xbeach


def _existing_input_path():
    """Returns a guaranteed-existing file path for plugin constructor inputs."""
    return __file__


def _write_square_pol(path, xmin, ymin, xmax, ymax):
    """Write a single-block Tekal polygon around a square."""
    path.write_text(
        f"""
selector
4 2
{xmin} {ymin}
{xmax} {ymin}
{xmax} {ymax}
{xmin} {ymax}
""".strip()
    )


def test_xbeach_convert_uses_mean_variables_and_flattens_spatial_dims(monkeypatch):
    """Checks XBeach mean output maps to SedTRAILS dimensions."""
    globalx = np.array([[0.0, 1.0], [0.0, 1.0]])
    globaly = np.array([[0.0, 0.0], [1.0, 1.0]])
    scalar = np.arange(8, dtype=float).reshape(2, 2, 2)
    frac_x = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    frac_y = frac_x + 10.0

    ds = xr.Dataset(
        data_vars={
            'zb_mean': (('meantime', 'ny', 'nx'), scalar),
            'hh_mean': (('meantime', 'ny', 'nx'), scalar + 1.0),
            'ue_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
            've_mean': (('meantime', 'ny', 'nx'), np.full((2, 2, 2), 2.0)),
            'taubx_mean': (('meantime', 'ny', 'nx'), np.full((2, 2, 2), 3.0)),
            'tauby_mean': (('meantime', 'ny', 'nx'), np.full((2, 2, 2), 4.0)),
            'Subg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), frac_x),
            'Svbg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), frac_y),
            'Susg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), frac_x + 20.0),
            'Svsg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), frac_y + 20.0),
            'cctot_mean': (('meantime', 'ny', 'nx'), scalar + 2.0),
            'ua_mean': (('meantime', 'ny', 'nx'), scalar + 3.0),
            'thetamean_mean': (('meantime', 'ny', 'nx'), np.zeros((2, 2, 2))),
        },
        coords={
            'globalx': (('ny', 'nx'), globalx),
            'globaly': (('ny', 'nx'), globaly),
            'meantime': (('meantime',), np.array([10.0, 20.0])),
            'globaltime': (('globaltime',), np.array([0.0, 10.0, 20.0])),
        },
    )
    ds['meantime'].attrs['units'] = 's'

    def fake_load(self):
        """Injects synthetic XBeach mean-output dataset."""
        self.input_data = ds
        return ds

    def fail_delaunay(*_args, **_kwargs):
        """Fails if structured XBeach grids fall back to Delaunay connectivity."""
        raise AssertionError('Structured XBeach grids should not use Delaunay connectivity')

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    monkeypatch.setattr(xbeach, 'delaunay_connectivity', fail_delaunay)

    plugin = xbeach.FormatPlugin(_existing_input_path())
    sedtrails_data = plugin.convert(reference_date=np.datetime64('1970-01-01T00:00:00'))

    expected_connectivity = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int64)
    np.testing.assert_array_equal(sedtrails_data.times, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(sedtrails_data.x, np.array([0.0, 1.0, 0.0, 1.0]))
    assert sedtrails_data.bed_level.shape == (2, 4)
    assert sedtrails_data.water_depth.shape == (2, 4)
    assert sedtrails_data.bed_load_transport['x'].shape == (2, 4)
    assert sedtrails_data.suspended_transport['y'].shape == (2, 4)
    assert sedtrails_data.fractions == 1
    assert sedtrails_data.metadata.source_sediment_classes == 2
    assert sedtrails_data.metadata.sediment_transport_fraction_handling == 'sum_over_source_sediment_classes'
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, expected_connectivity)
    np.testing.assert_array_equal(sedtrails_data.particle_face_connectivity, expected_connectivity)
    np.testing.assert_array_equal(sedtrails_data.bed_load_transport['x'], np.array([[4.0, 6.0, 8.0, 10.0], [20.0, 22.0, 24.0, 26.0]]))
    np.testing.assert_array_equal(sedtrails_data.bed_load_transport['y'], np.array([[24.0, 26.0, 28.0, 30.0], [40.0, 42.0, 44.0, 46.0]]))
    np.testing.assert_allclose(sedtrails_data.mean_bed_shear_stress, 5.0)
    np.testing.assert_allclose(sedtrails_data.max_bed_shear_stress, 5.0)
    np.testing.assert_array_equal(sedtrails_data.nonlinear_wave_velocity['x'], (scalar + 3.0).reshape(2, 4))


def test_xbeach_cutout_coordinates_filter_fields_and_connectivity(monkeypatch):
    """Checks XBeach cutout points are removed from fields and mesh connectivity."""
    globalx = np.array([[0.0, np.nan, 2.0], [0.0, 1.0, 2.0]])
    globaly = np.array([[0.0, np.nan, 0.0], [1.0, 1.0, 1.0]])
    scalar = np.arange(6, dtype=float).reshape(1, 2, 3)
    transport = np.ones((1, 1, 2, 3), dtype=float)

    ds = xr.Dataset(
        data_vars={
            'zb_mean': (('meantime', 'ny', 'nx'), scalar),
            'hh_mean': (('meantime', 'ny', 'nx'), scalar + 1.0),
            'ue_mean': (('meantime', 'ny', 'nx'), scalar + 2.0),
            've_mean': (('meantime', 'ny', 'nx'), scalar + 3.0),
            'taubx_mean': (('meantime', 'ny', 'nx'), np.full((1, 2, 3), 3.0)),
            'tauby_mean': (('meantime', 'ny', 'nx'), np.full((1, 2, 3), 4.0)),
            'Subg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport),
            'Svbg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 1.0),
            'Susg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 2.0),
            'Svsg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 3.0),
            'cctot_mean': (('meantime', 'ny', 'nx'), scalar + 4.0),
        },
        coords={
            'globalx': (('ny', 'nx'), globalx),
            'globaly': (('ny', 'nx'), globaly),
            'meantime': (('meantime',), np.array([0.0])),
        },
    )

    def fake_load(self):
        """Injects synthetic XBeach grid coordinates with a cutout point."""
        self.input_data = ds
        return ds

    def fail_delaunay(*_args, **_kwargs):
        """Fails if structured XBeach grids fall back to Delaunay connectivity."""
        raise AssertionError('Structured XBeach grids should not use Delaunay connectivity')

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    monkeypatch.setattr(xbeach, 'delaunay_connectivity', fail_delaunay)

    plugin = xbeach.FormatPlugin(_existing_input_path())
    sedtrails_data = plugin.convert()
    field_data = plugin.get_seeding_field_data()

    expected_x = np.array([0.0, 2.0, 0.0, 1.0, 2.0])
    expected_y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    expected_connectivity = np.array([[0, 3, 2]], dtype=np.int64)

    np.testing.assert_array_equal(sedtrails_data.x, expected_x)
    np.testing.assert_array_equal(sedtrails_data.y, expected_y)
    np.testing.assert_array_equal(sedtrails_data.bed_level, np.array([[0.0, 2.0, 3.0, 4.0, 5.0]]))
    np.testing.assert_array_equal(sedtrails_data.depth_avg_flow_velocity['x'], np.array([[2.0, 4.0, 5.0, 6.0, 7.0]]))
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, expected_connectivity)
    np.testing.assert_array_equal(field_data.x, expected_x)
    np.testing.assert_array_equal(field_data.y, expected_y)
    np.testing.assert_array_equal(field_data.face_node_connectivity, expected_connectivity)
    assert np.isfinite(field_data.x).all()
    assert np.isfinite(field_data.y).all()


def test_xbeach_empty_cutout_connectivity_does_not_fall_back_to_delaunay(monkeypatch):
    """Checks cutout-only cells stay outside instead of being Delaunay-filled."""
    ds = xr.Dataset(
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, np.nan]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, np.nan]])),
        },
    )

    def fake_load(self):
        """Injects a grid where no complete structured triangle survives."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    field_data = xbeach.FormatPlugin(_existing_input_path()).get_seeding_field_data()

    np.testing.assert_array_equal(field_data.particle_face_connectivity, np.empty((0, 3), dtype=np.int64))
    config = {
        'name': 'cutout-only population',
        'particle_type': 'sand',
        'transport_probability': 'no_probability',
        'seeding': {
            'strategy': {'point': {'locations': ['0.5,0.5']}},
            'quantity': 1,
            'release_start': '0',
            'burial_depth': {'constant': 0.0},
        },
    }
    with pytest.raises(ConfigurationError, match='outside the input field domain'):
        ParticleSeeder([config]).seed(field_data)


def test_format_converter_get_seeding_field_data_uses_xbeach_geometry_reader(monkeypatch):
    """Checks FormatConverter uses the XBeach geometry fast path."""
    ds = xr.Dataset(
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, 1.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, 1.0]])),
        },
    )

    def fake_load(self):
        """Injects synthetic XBeach grid coordinates."""
        self.input_data = ds
        return ds

    def fail_convert(self, *_args, **_kwargs):
        """Fails if seeding falls back to full conversion."""
        raise RuntimeError('XBeach seeding should use get_seeding_coordinates')

    def fail_delaunay(*_args, **_kwargs):
        """Fails if structured XBeach grids fall back to Delaunay connectivity."""
        raise AssertionError('Structured XBeach grids should not use Delaunay connectivity')

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    monkeypatch.setattr(xbeach.FormatPlugin, 'convert', fail_convert)
    monkeypatch.setattr(xbeach, 'delaunay_connectivity', fail_delaunay)

    converter = FormatConverter(
        {'input_file': _existing_input_path(), 'input_format': 'xbeach', 'reference_date': '2000-01-01'}
    )
    field_data = converter.get_seeding_field_data()

    expected_connectivity = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int64)
    np.testing.assert_array_equal(field_data.x, np.array([0.0, 1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(field_data.y, np.array([0.0, 0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(field_data.face_node_connectivity, expected_connectivity)
    np.testing.assert_array_equal(field_data.particle_face_connectivity, expected_connectivity)
    assert field_data.reference_date == np.datetime64('2000-01-01')


def test_xbeach_open_boundary_classification_reaches_particle_updates(monkeypatch, tmp_path):
    """XBeach particles crossing configured open edges should leave the domain."""

    open_file = tmp_path / 'open.pol'
    _write_square_pol(open_file, 0.25, -0.10, 0.75, 0.10)
    ds = xr.Dataset(
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, 1.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, 1.0]])),
        },
    )

    def fake_load(self):
        """Injects synthetic XBeach grid coordinates."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    monkeypatch.setattr(np.random, 'rand', lambda n_particles: np.zeros(n_particles))
    converter = FormatConverter(
        {
            'input_file': _existing_input_path(),
            'input_format': 'xbeach',
            'reference_date': '2000-01-01',
            'domain_config': {'boundary_class_pol_files': {'open': [str(open_file)]}},
        }
    )
    field_data = converter.get_seeding_field_data()
    config = {
        'name': 'XBeach open boundary population',
        'particle_type': 'sand',
        'transport_probability': 'no_probability',
        'seeding': {
            'strategy': {'point': {'locations': ['0.5,0.2']}},
            'quantity': 1,
            'release_start': '0',
            'burial_depth': {'constant': 0.0},
        },
    }

    population = ParticleSeeder([config]).seed(field_data)[0]
    population._current_time = 0.0
    population.particles['transport_probability'] = np.ones(1)
    population.update_status()
    population.update_position(flow_field={'u': np.zeros(4), 'v': -np.ones(4)}, current_timestep=0.5)

    assert field_data.boundary_edge_classification['class_counts']['open'] == 1
    assert population.particles['status_left_domain'].tolist() == [True]
    assert population.particles['status_alive'].tolist() == [False]


def test_xbeach_get_scalar_field_slices_time_dependent_bed_level(monkeypatch):
    """Verifies dynamic XBeach bed levels are exposed as time slices."""
    ds = xr.Dataset(
        data_vars={
            'zb_mean': (('meantime', 'ny', 'nx'), np.arange(8, dtype=float).reshape(2, 2, 2)),
            'hh_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
            'ue_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
            've_mean': (('meantime', 'ny', 'nx'), np.zeros((2, 2, 2))),
            'taubx_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
            'tauby_mean': (('meantime', 'ny', 'nx'), np.zeros((2, 2, 2))),
            'Subg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), np.ones((2, 1, 2, 2))),
            'Svbg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), np.zeros((2, 1, 2, 2))),
            'Susg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), np.ones((2, 1, 2, 2))),
            'Svsg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), np.zeros((2, 1, 2, 2))),
            'cctot_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
        },
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, 1.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, 1.0]])),
            'meantime': (('meantime',), np.array([0.0, 10.0])),
        },
    )

    def fake_load(self):
        """Injects synthetic XBeach dataset with changing bed levels."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    sedtrails_data = xbeach.FormatPlugin(_existing_input_path()).convert()
    retriever = FieldDataRetriever(sedtrails_data)

    field = retriever.get_scalar_field(10.0, 'bed_level')

    np.testing.assert_array_equal(field['magnitude'], np.array([4.0, 5.0, 6.0, 7.0]))


def test_xbeach_transport_component_without_fraction_dim_is_not_summed(monkeypatch):
    """Checks transport data shaped (time, y, x) is flattened, not summed over y."""
    transport = np.arange(8, dtype=float).reshape(2, 2, 2)
    ds = xr.Dataset(
        data_vars={'Subg_mean': (('meantime', 'ny', 'nx'), transport)},
        coords={'meantime': (('meantime',), np.array([0.0, 10.0]))},
    )

    def fake_load(self):
        """Injects a synthetic transport dataset without a fraction dimension."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.load()
    total_transport, source_sediment_classes = plugin._mean_transport_component(
        'Subg_mean',
        slice(None),
        num_times=2,
        grid_size=4,
    )

    assert source_sediment_classes == 1
    np.testing.assert_array_equal(total_transport, transport.reshape(2, 4))
