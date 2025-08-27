import pytest
import xugrid as xu
import numpy as np
from sedtrails.data_manager.netcdf_writer import NetCDFWriter


@pytest.fixture
def tmp_output_dir(tmp_path):
    """
    Pytest fixture to provide a temporary output directory for NetCDF files.
    """
    return tmp_path / 'output'


def test_netcdf_writer_creates_file(tmp_output_dir):
    """
    Test that NetCDFWriter successfully writes a UgridDataset to a NetCDF file.
    """
    # Create a minimal SedTRAILS-inspired UgridDataset
    node_x = np.array([0, 1, 1, 0])
    node_y = np.array([0, 0, 1, 1])
    face_node_connectivity = np.array([[0, 1, 2, 3]])
    fill_value = -1

    mesh = xu.Ugrid2d(
        node_x=node_x, node_y=node_y, face_node_connectivity=face_node_connectivity, fill_value=fill_value
    )
    # Add SedTRAILS-like variables directly to the mesh's dataset
    mesh_ds = mesh.to_dataset()
    mesh_ds['bed_level'] = (('mesh2d_face',), np.array([0.5]))
    mesh_ds['water_depth'] = (('mesh2d_face',), np.array([2.0]))
    mesh_ds['depth_avg_flow_velocity'] = (('mesh2d_face',), np.array([0.1]))
    mesh_ds['sediment_concentration'] = (('mesh2d_face',), np.array([0.01]))

    # NetCDFWriter expects a UgridDataset, so we convert it
    mesh = xu.UgridDataset(mesh_ds)

    writer = NetCDFWriter(tmp_output_dir)
    filename = 'test_output.nc'
    writer.write(mesh, filename)
    output_file = tmp_output_dir / filename
    assert output_file.exists()
