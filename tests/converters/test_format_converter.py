"""Tests for the format converter."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from sedtrails.particle_tracer.position_calculator_numba import create_grid_geometry
from sedtrails.transport_converter.format_converter import FormatConverter
from sedtrails.transport_converter.plugins.format import fm_netcdf, sfincs


def _existing_input_path():
    """Returns a guaranteed-existing file path for plugin constructor inputs."""
    return __file__


def _write_square_pol(path, xmin, ymin, xmax, ymax):
    """Write a single-block Tekal polygon around a square."""
    path.write_text(
        f"""
island
4 2
{xmin} {ymin}
{xmax} {ymin}
{xmax} {ymax}
{xmin} {ymax}
""".strip()
    )


def _sfincs_dataset_from_face_centers(face_centers):
    """Build a small SFINCS-like UGRID dataset with quad faces at requested centroids."""
    centers = np.asarray(face_centers, dtype=float)
    node_x = []
    node_y = []
    faces = []
    half_size = 0.05
    for center_x, center_y in centers:
        start = len(node_x)
        node_x.extend([center_x - half_size, center_x + half_size, center_x + half_size, center_x - half_size])
        node_y.extend([center_y - half_size, center_y - half_size, center_y + half_size, center_y + half_size])
        faces.append([start, start + 1, start + 2, start + 3])

    n_faces = centers.shape[0]
    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.asarray(node_x)),
            'mesh2d_node_y': (('node',), np.asarray(node_y)),
            'mesh2d_face_nodes': (('face', 'nmax'), np.asarray(faces, dtype=np.int64)),
            'zb': (('face',), -np.arange(1, n_faces + 1, dtype=float)),
            'h': (('time', 'face'), np.vstack((np.arange(n_faces), np.arange(n_faces) + 10.0))),
            'u': (('time', 'face'), np.full((2, n_faces), 0.1)),
            'v': (('time', 'face'), np.zeros((2, n_faces))),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 0
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -1
    return ds


class _PluginWithCoordinateReader:
    """Test double exposing get_seeding_coordinates to bypass conversion."""

    def __init__(self):
        """Tracks whether convert was called unexpectedly."""
        self.convert_called = False

    def get_seeding_coordinates(self):
        """Provides deterministic coordinate arrays for seeding tests."""
        return [1.0, 2.0], [3.0, 4.0]

    def convert(self, *_args, **_kwargs):
        """Fails if fallback conversion is invoked when direct coordinates exist."""
        self.convert_called = True
        raise RuntimeError('convert should not be called when get_seeding_coordinates is available')


class _PluginWithSeedingFieldData:
    """Test double exposing the rich seeding field-data fast path."""

    def __init__(self):
        """Tracks whether less specific fallbacks were called unexpectedly."""
        self.coordinate_reader_called = False
        self.convert_called = False

    def get_seeding_field_data(self):
        """Provides deterministic field geometry for seeding tests."""
        return SimpleNamespace(
            x=[1.0, 2.0, 3.0],
            y=[4.0, 5.0, 6.0],
            face_node_connectivity=[[0, 1, 2]],
            particle_face_connectivity=[[0, 2, 1]],
            boundary_edge_classification={'edge_nodes': [[0, 1]], 'edge_classes': ['open']},
            face_node_fill_value=-99,
        )

    def get_seeding_coordinates(self):
        """Fails if the richer field-data path is skipped."""
        self.coordinate_reader_called = True
        raise RuntimeError('get_seeding_coordinates should not be called when get_seeding_field_data is available')

    def convert(self, *_args, **_kwargs):
        """Fails if fallback conversion is invoked when direct field data exists."""
        self.convert_called = True
        raise RuntimeError('convert should not be called when get_seeding_field_data is available')


class _PluginWithConvertOnly:
    """Test double that only supports convert-based seeding field retrieval."""

    class _SedtrailsLike:
        """Minimal Sedtrails-like object exposing x/y coordinates."""

        def __init__(self, x, y):
            """Stores coordinate arrays returned by the fake conversion step."""
            self.x = x
            self.y = y

    def convert(self, *_args, **_kwargs):
        """Returns a minimal converted object for fallback seeding logic."""
        return self._SedtrailsLike(x=(10.0, 20.0), y=(30.0, 40.0))


def test_get_seeding_field_data_prefers_field_data_reader():
    """Ensures rich seeding field-data plugins preserve mesh metadata."""
    converter = FormatConverter({'input_file': 'dummy.nc', 'input_format': 'dummy', 'reference_date': '1999-12-31'})
    plugin = _PluginWithSeedingFieldData()
    converter._format_plugin = plugin

    field_data = converter.get_seeding_field_data()

    np.testing.assert_array_equal(field_data.x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(field_data.y, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(field_data.face_node_connectivity, np.array([[0, 1, 2]]))
    np.testing.assert_array_equal(field_data.particle_face_connectivity, np.array([[0, 2, 1]]))
    assert field_data.boundary_edge_classification == {'edge_nodes': [[0, 1]], 'edge_classes': ['open']}
    assert field_data.face_node_fill_value == -99
    assert field_data.reference_date == np.datetime64('1999-12-31')
    assert not plugin.coordinate_reader_called
    assert not plugin.convert_called


def test_get_seeding_field_data_prefers_coordinate_reader():
    """Ensures coordinate-reader plugins are used without calling convert."""
    converter = FormatConverter({'input_file': 'dummy.nc', 'input_format': 'dummy', 'reference_date': '2000-01-01'})
    plugin = _PluginWithCoordinateReader()
    converter._format_plugin = plugin

    field_data = converter.get_seeding_field_data()

    np.testing.assert_array_equal(field_data.x, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(field_data.y, np.array([3.0, 4.0]))
    assert field_data.reference_date == np.datetime64('2000-01-01')
    assert not plugin.convert_called


def test_get_seeding_field_data_falls_back_to_convert():
    """Ensures convert fallback is used when no coordinate reader is available."""
    converter = FormatConverter({'input_file': 'dummy.nc', 'input_format': 'dummy', 'reference_date': '2001-02-03'})
    converter._format_plugin = _PluginWithConvertOnly()

    field_data = converter.get_seeding_field_data()

    np.testing.assert_array_equal(field_data.x, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(field_data.y, np.array([30.0, 40.0]))
    assert field_data.reference_date == np.datetime64('2001-02-03')


def test_fm_convert_masks_inner_boundary_faces_from_domain_config(monkeypatch, tmp_path):
    """FM conversion excludes source faces whose centroids fall inside island polygons."""

    island_file = tmp_path / 'island.pol'
    island_file.write_text(
        """
island
4 2
9 9
12 9
12 12
9 12
""".strip()
    )
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('node',), np.array([0.0, 1.0, 0.0, 10.0, 11.0, 10.0])),
            'net_ycc': (('node',), np.array([0.0, 0.0, 1.0, 10.0, 10.0, 11.0])),
            'NetElemNode': (('face', 'nmax'), np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['NetElemNode'].attrs['start_index'] = 0

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(fm_netcdf.FormatPlugin, 'load', fake_load)
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'inner_boundary_pol_files': [str(island_file)]}

    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))
    seeding_data = plugin.get_seeding_field_data()

    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, np.array([[0, 1, 2]], dtype=np.int64))
    np.testing.assert_array_equal(seeding_data.face_node_connectivity, np.array([[0, 1, 2]], dtype=np.int64))
    assert sedtrails_data.metadata.inner_boundary_polygon_count == 1
    assert sedtrails_data.metadata.inner_boundary_masked_face_count == 1


def test_fm_convert_stores_boundary_edge_class_overrides(monkeypatch, tmp_path):
    """FM conversion stores user override classes for active boundary edges."""

    open_file = tmp_path / 'open.pol'
    land_file = tmp_path / 'land.pol'
    _write_square_pol(open_file, 0.25, -0.10, 0.75, 0.10)
    _write_square_pol(land_file, -0.10, 0.25, 0.10, 0.75)
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('node',), np.array([0.0, 1.0, 0.0])),
            'net_ycc': (('node',), np.array([0.0, 0.0, 1.0])),
            'NetElemNode': (('face', 'nmax'), np.array([[0, 1, 2]], dtype=np.int64)),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['NetElemNode'].attrs['start_index'] = 0

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(fm_netcdf.FormatPlugin, 'load', fake_load)
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'boundary_class_pol_files': {'open': [str(open_file)], 'land': [str(land_file)]}}

    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    edge_classes = sedtrails_data.metadata.boundary_edge_classification
    assert edge_classes['class_counts'] == {'open': 1, 'land': 1, 'unclassified': 1, 'ambiguous': 0}
    assert edge_classes['polygon_counts'] == {'open': 1, 'land': 1}
    assert edge_classes['class_pol_files']['open'] == [str(open_file)]


def test_format_converter_applies_domain_config_to_fm_plugin(monkeypatch, tmp_path):
    """FormatConverter must pass domain polygons to plugin seeding geometry."""

    open_file = tmp_path / 'open.pol'
    _write_square_pol(open_file, 0.25, -0.10, 0.75, 0.10)
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('node',), np.array([0.0, 1.0, 0.0])),
            'net_ycc': (('node',), np.array([0.0, 0.0, 1.0])),
            'NetElemNode': (('face', 'nmax'), np.array([[0, 1, 2]], dtype=np.int64)),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['NetElemNode'].attrs['start_index'] = 0

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(fm_netcdf.FormatPlugin, 'load', fake_load)
    converter = FormatConverter(
        {
            'input_file': _existing_input_path(),
            'input_format': 'fm_netcdf',
            'reference_date': '2024-01-01T00:00:00',
            'domain_config': {'boundary_class_pol_files': {'open': [str(open_file)]}},
        }
    )

    field_data = converter.get_seeding_field_data()

    edge_classes = field_data.boundary_edge_classification
    assert edge_classes['class_counts']['open'] == 1
    assert edge_classes['polygon_counts']['open'] == 1
    assert edge_classes['class_pol_files']['open'] == [str(open_file)]


def test_fm_uses_delaunay_when_source_connectivity_does_not_index_cell_centers(monkeypatch, tmp_path):
    """FM native node connectivity should not be used with cell-center coordinates."""

    open_file = tmp_path / 'open.pol'
    _write_square_pol(open_file, 0.25, -0.10, 0.75, 0.10)
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('face',), np.array([0.0, 1.0, 1.0, 0.0])),
            'net_ycc': (('face',), np.array([0.0, 0.0, 1.0, 1.0])),
            'NetElemNode': (('elem', 'nmax'), np.array([[10, 11, 12]], dtype=np.int64)),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['NetElemNode'].attrs['start_index'] = 0

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(fm_netcdf.FormatPlugin, 'load', fake_load)
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'boundary_class_pol_files': {'open': [str(open_file)]}}

    field_data = plugin.get_seeding_field_data()

    assert np.max(field_data.face_node_connectivity) < field_data.x.size
    assert field_data.boundary_edge_classification['class_counts']['open'] == 1


def test_fm_active_geometry_and_boundary_classification_are_cached(monkeypatch):
    """Repeated FM geometry requests should reuse static connectivity and edge classes."""
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.input_data = xr.Dataset()
    node_x = np.array([0.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0])
    connectivity = np.array([[0, 1, 2]], dtype=np.int64)
    call_counts = {'delaunay': 0, 'classify': 0}

    def fake_delaunay(_node_x, _node_y):
        call_counts['delaunay'] += 1
        return connectivity

    class FakeClassification:
        def to_metadata(self):
            return {'edge_nodes': [[0, 1]], 'edge_classes': ['open']}

    def fake_classify(*_args, **_kwargs):
        call_counts['classify'] += 1
        return FakeClassification()

    monkeypatch.setattr(fm_netcdf, 'delaunay_connectivity', fake_delaunay)
    monkeypatch.setattr(fm_netcdf, 'classify_boundary_edges_from_config', fake_classify)

    first_connectivity = plugin._active_triangular_connectivity(node_x, node_y)
    second_connectivity = plugin._active_triangular_connectivity(node_x, node_y)
    first_classes = plugin._boundary_edge_classification(node_x, node_y, first_connectivity)
    second_classes = plugin._boundary_edge_classification(node_x, node_y, second_connectivity)

    assert call_counts == {'delaunay': 1, 'classify': 1}
    np.testing.assert_array_equal(first_connectivity, connectivity)
    np.testing.assert_array_equal(second_connectivity, connectivity)
    assert first_classes == second_classes == {'edge_nodes': [[0, 1]], 'edge_classes': ['open']}


def test_sfincs_get_seeding_coordinates_uses_ugrid_face_coordinates(monkeypatch):
    """Checks SFINCS seeding coordinates come directly from UGRID face centers."""
    class FakeUgrid2d:
        """Minimal UGRID-like grid with predefined face coordinates."""

        def __init__(self):
            self.face_x = np.array([1.0, 2.0, 3.0])
            self.face_y = np.array([4.0, 5.0, 6.0])

    class FakeUgridDataset:
        """Dataset wrapper exposing a single UGRID-like grid object."""

        def __init__(self):
            self.grid = FakeUgrid2d()

    def fake_load(self):
        """Injects synthetic UGRID input data into the plugin."""
        self.input_data = FakeUgridDataset()

    monkeypatch.setattr(sfincs.xu, 'UgridDataset', FakeUgridDataset)
    monkeypatch.setattr(sfincs.xu, 'Ugrid2d', FakeUgrid2d)
    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    x, y = plugin.get_seeding_coordinates()

    np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(y, np.array([4.0, 5.0, 6.0]))


def test_sfincs_get_seeding_coordinates_fallback_computes_centroids(monkeypatch):
    """Verifies centroid fallback from face-node connectivity when UGRID is unavailable."""
    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.array([0.0, 2.0, 2.0, 0.0])),
            'mesh2d_node_y': (('node',), np.array([0.0, 0.0, 2.0, 2.0])),
            'mesh2d_face_nodes': (('face', 'nmax'), np.array([[1, 2, 3, 4]], dtype=np.int64)),
        }
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 1
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -999

    def fake_load(self):
        """Injects synthetic mesh dataset into the plugin."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    x, y = plugin.get_seeding_coordinates()

    np.testing.assert_allclose(x, np.array([1.0]))
    np.testing.assert_allclose(y, np.array([1.0]))


def test_sfincs_convert_stores_face_node_mesh_geometry(monkeypatch):
    """Checks convert stores node coordinates and normalized face connectivity."""
    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.array([0.0, 2.0, 2.0, 0.0])),
            'mesh2d_node_y': (('node',), np.array([0.0, 0.0, 2.0, 2.0])),
            'mesh2d_face_nodes': (('face', 'nmax'), np.array([[1, 2, 3, 4]], dtype=np.int64)),
            'zb': (('face',), np.array([-1.0])),
            'h': (('time', 'face'), np.array([[1.0], [2.0]])),
            'u': (('time', 'face'), np.array([[0.1], [0.2]])),
            'v': (('time', 'face'), np.array([[0.0], [0.0]])),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 1
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -999

    def fake_load(self):
        """Injects synthetic SFINCS dataset for conversion testing."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    np.testing.assert_array_equal(sedtrails_data.node_x, np.array([0.0, 2.0, 2.0, 0.0]))
    np.testing.assert_array_equal(sedtrails_data.node_y, np.array([0.0, 0.0, 2.0, 2.0]))
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, np.array([[0, 1, 2, 3]]))
    assert sedtrails_data.face_node_fill_value == -1
    assert sedtrails_data.mesh_geometry()['face_node_connectivity'].shape == (1, 4)


def test_sfincs_convert_masks_inner_boundary_faces_from_domain_config(monkeypatch, tmp_path):
    """SFINCS conversion excludes faces whose centroids fall inside island polygons."""
    island_file = tmp_path / 'island.pol'
    _write_square_pol(island_file, 0.8, 0.8, 1.2, 1.2)
    ds = _sfincs_dataset_from_face_centers([(0.0, 0.0), (1.0, 1.0)])

    def fake_load(self):
        """Injects synthetic SFINCS dataset for conversion testing."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'inner_boundary_pol_files': [str(island_file)]}
    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    np.testing.assert_allclose(sedtrails_data.x, np.array([0.0]))
    np.testing.assert_allclose(sedtrails_data.y, np.array([0.0]))
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, np.array([[0, 1, 2, 3]]))
    np.testing.assert_array_equal(sedtrails_data.water_depth, np.array([[0.0], [10.0]]))
    np.testing.assert_allclose(sedtrails_data.depth_avg_flow_velocity['x'], np.full((2, 1), 0.1))
    assert sedtrails_data.node_x.shape == (8,)
    assert sedtrails_data.metadata.inner_boundary_polygon_count == 1
    assert sedtrails_data.metadata.inner_boundary_masked_face_count == 1
    assert sedtrails_data.metadata.inner_boundary_active_face_count == 1


def test_sfincs_seeding_geometry_respects_inner_boundary_holes(monkeypatch, tmp_path):
    """SFINCS seeding triangles should leave configured island interiors outside the mesh."""
    island_file = tmp_path / 'island.pol'
    _write_square_pol(island_file, 0.85, 0.85, 1.15, 1.15)
    face_centers = [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
        (0.8, 0.8),
        (1.2, 0.8),
        (1.2, 1.2),
        (0.8, 1.2),
        (1.0, 1.0),
    ]
    ds = _sfincs_dataset_from_face_centers(face_centers)

    def fake_load(self):
        """Injects synthetic SFINCS dataset for seeding-geometry testing."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'inner_boundary_pol_files': [str(island_file)]}
    field_data = plugin.get_seeding_field_data()
    grid_geometry = create_grid_geometry(
        field_data.x,
        field_data.y,
        triangles=field_data.particle_face_connectivity,
    )

    assert field_data.x.shape == (8,)
    assert field_data.particle_face_connectivity.shape[1] == 3
    simplices = grid_geometry.locate_points(
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 0.4, 1.6]),
    )
    assert simplices[0] == -1
    assert simplices[1] >= 0
    assert simplices[2] >= 0


def test_sfincs_convert_stores_boundary_edge_class_overrides(monkeypatch, tmp_path):
    """SFINCS conversion stores user override classes for active particle-domain edges."""
    open_file = tmp_path / 'open.pol'
    land_file = tmp_path / 'land.pol'
    _write_square_pol(open_file, 0.75, -0.10, 1.25, 0.10)
    _write_square_pol(land_file, -0.10, 0.75, 0.10, 1.25)
    ds = _sfincs_dataset_from_face_centers([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])

    def fake_load(self):
        """Injects synthetic SFINCS dataset for boundary-class testing."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'boundary_class_pol_files': {'open': [str(open_file)], 'land': [str(land_file)]}}
    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    edge_classes = sedtrails_data.metadata.boundary_edge_classification
    assert edge_classes['class_counts']['open'] == 1
    assert edge_classes['class_counts']['land'] == 1
    assert edge_classes['class_counts']['ambiguous'] == 0
    assert edge_classes['polygon_counts'] == {'open': 1, 'land': 1}


def test_sfincs_active_face_mask_is_cached(monkeypatch, tmp_path):
    """Repeated SFINCS active-mask requests should reuse polygon containment results."""
    island_file = tmp_path / 'island.pol'
    _write_square_pol(island_file, 0.8, 0.8, 1.2, 1.2)
    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'inner_boundary_pol_files': [str(island_file)]}
    call_count = 0

    def fake_points_inside_any_polygon(_points, _polygons):
        nonlocal call_count
        call_count += 1
        return np.array([False, True])

    monkeypatch.setattr(sfincs, 'points_inside_any_polygon', fake_points_inside_any_polygon)

    first_mask = plugin._active_face_mask(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    second_mask = plugin._active_face_mask(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    assert call_count == 1
    np.testing.assert_array_equal(first_mask, np.array([True, False]))
    np.testing.assert_array_equal(second_mask, np.array([True, False]))


def test_sfincs_get_seeding_coordinates_raises_on_non_ugrid2d(monkeypatch):
    """Ensures non-UGRID grids raise a clear TypeError in coordinate retrieval."""
    class FakeNonUgrid2d:
        """Placeholder grid type that intentionally does not match Ugrid2d."""

        pass

    class FakeUgridDataset:
        """Dataset wrapper exposing an invalid grid type."""

        def __init__(self):
            self.grid = FakeNonUgrid2d()

    def fake_load(self):
        """Injects non-UGRID dataset into the plugin to trigger validation."""
        self.input_data = FakeUgridDataset()

    monkeypatch.setattr(sfincs.xu, 'UgridDataset', FakeUgridDataset)
    monkeypatch.setattr(sfincs.xu, 'Ugrid2d', type('OtherGrid', (), {}))
    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())

    with pytest.raises(TypeError, match='Expected Ugrid2d'):
        plugin.get_seeding_coordinates()
