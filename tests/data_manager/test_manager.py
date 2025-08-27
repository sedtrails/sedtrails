import numpy as np
import xugrid as xu
from sedtrails.data_manager.manager import DataManager


def create_test_mesh():
    """
    Test mesh creation function for DataManager tests.
    """
    node_x = np.array([0, 1, 1, 0])
    node_y = np.array([0, 0, 1, 1])
    face_node_connectivity = np.array([[0, 1, 2, 3]])
    fill_value = -1
    return node_x, node_y, face_node_connectivity, fill_value


def test_add_data_and_write(tmp_path):
    """
    Test that DataManager buffers data and writes to disk when requested.
    """
    dm = DataManager(tmp_path)
    node_x, node_y, face_node_connectivity, fill_value = create_test_mesh()
    dm.set_mesh(node_x, node_y, face_node_connectivity, fill_value)
    # Add some data
    dm.add_data(1, 0.0, 10.0, 20.0)
    dm.add_data(2, 1.0, 11.0, 21.0)
    # Write to disk
    dm.write('test_manager_output.nc')
    output_dir = dm.writer.output_dir
    output_file = output_dir / 'test_manager_output.nc'
    assert output_file.exists()
    ds = xu.open_dataset(output_file)
    assert 'particle_id' in ds
    assert 'x' in ds
    assert 'y' in ds
    assert len(ds.time) == 2


def test_buffer_limit_triggers_write(tmp_path):
    """
    Test that DataManager writes to disk automatically when buffer exceeds memory limit.
    """
    # Set a very low memory limit to force a write
    dm = DataManager(tmp_path, max_bytes=1)
    node_x, node_y, face_node_connectivity, fill_value = create_test_mesh()
    dm.set_mesh(node_x, node_y, face_node_connectivity, fill_value)
    # Add enough data to exceed the limit
    for i in range(100):
        dm.add_data(i, float(i), float(i), float(i))
    # The first file should have been written
    output_dir = dm.writer.output_dir
    expected_file = output_dir / '.sim_buffer_0.nc'
    assert expected_file.exists()


def test_merge(tmp_path):
    """
    Test that DataManager.merge merges chunked files into a single file.
    """
    dm = DataManager(tmp_path)
    node_x, node_y, face_node_connectivity, fill_value = create_test_mesh()
    dm.set_mesh(node_x, node_y, face_node_connectivity, fill_value)
    # Write two chunks
    for i in range(5):
        dm.add_data(i, float(i), float(i), float(i))
    dm.write()  # .sim_buffer_0.nc
    for i in range(5, 10):
        dm.add_data(i, float(i), float(i), float(i))
    dm.write()  # .sim_buffer_1.nc
    # Get the output directory from the writer
    output_dir = dm.writer.output_dir
    # Merge
    dm.merge('merged_manager.nc')
    merged_file = output_dir / 'merged_manager.nc'
    assert merged_file.exists()
    ds = xu.open_dataset(merged_file)
    assert len(ds.time) == 10
