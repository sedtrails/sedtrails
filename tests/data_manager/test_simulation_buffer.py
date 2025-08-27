import pytest
import numpy as np
import xugrid as xu
from sedtrails.data_manager.simulation_buffer import SimulationDataBuffer
from sedtrails.data_manager.netcdf_writer import NetCDFWriter


def test_add_and_get_data():
    """
    Test adding data to the buffer and retrieving it as numpy arrays.
    """
    buffer = SimulationDataBuffer()
    buffer.add(1, 0.0, 10.0, 20.0)
    buffer.add(2, 1.0, 11.0, 21.0)
    data = buffer.get_data()
    assert np.array_equal(data['particle_id'], np.array([1, 2]))
    assert np.array_equal(data['time'], np.array([0.0, 1.0]))
    assert np.array_equal(data['x'], np.array([10.0, 11.0]))
    assert np.array_equal(data['y'], np.array([20.0, 21.0]))


def test_clear():
    """
    Test clearing the buffer.
    """
    buffer = SimulationDataBuffer()
    buffer.add(1, 0.0, 10.0, 20.0)
    buffer.clear()
    data = buffer.get_data()
    for arr in data.values():
        assert arr.size == 0


def test_to_ugrid_dataset():
    """
    Test converting the buffer to a xu.UgridDataset.
    """
    buffer = SimulationDataBuffer()
    buffer.add(1, 0.0, 10.0, 20.0)
    buffer.add(2, 1.0, 11.0, 21.0)
    node_x = np.array([0, 1])
    node_y = np.array([0, 1])
    face_node_connectivity = np.array([[0, 1]])
    fill_value = -1
    ugrid_ds = buffer.to_ugrid_dataset(node_x, node_y, face_node_connectivity, fill_value)
    assert isinstance(ugrid_ds, xu.UgridDataset)
    assert 'particle_id' in ugrid_ds
    assert 'x' in ugrid_ds
    assert 'y' in ugrid_ds
    assert 'time' in ugrid_ds


def test_write_to_disk(tmp_path):
    """
    Test writing buffer to disk using the write_to_disk method.
    """
    # Create buffer with test data
    buffer = SimulationDataBuffer()
    buffer.add(1, 0.0, 10.0, 20.0)
    buffer.add(2, 1.0, 11.0, 21.0)

    # Create test mesh
    node_x = np.array([0, 1, 1, 0])
    node_y = np.array([0, 0, 1, 1])
    face_node_connectivity = np.array([[0, 1, 2, 3]])
    fill_value = -1

    # Create writer and filename
    writer = NetCDFWriter(tmp_path)
    filename = 'test_output.nc'
    output_dir = writer.output_dir

    # Get data before write to check clearing
    before_data = buffer.get_data().copy()
    assert len(before_data['particle_id']) == 2  # Verify data exists

    # Write to disk
    buffer.write_to_disk(node_x, node_y, face_node_connectivity, fill_value, writer, filename)

    # Check file exists
    output_file = output_dir / filename
    assert output_file.exists()

    # Check file contents
    ds = xu.open_dataset(output_file)
    assert 'particle_id' in ds
    assert 'x' in ds
    assert 'y' in ds
    assert len(ds.time) == 2
    assert ds.particle_id.values.tolist() == [1, 2]

    # Check buffer was cleared
    after_data = buffer.get_data()
    for arr in after_data.values():
        assert arr.size == 0


def test_merge_output_files(tmp_path):
    """
    Test that merge_output_files correctly merges multiple NetCDF chunk files into one.
    """
    # Create test mesh
    node_x = np.array([0, 1, 1, 0])
    node_y = np.array([0, 0, 1, 1])
    face_node_connectivity = np.array([[0, 1, 2, 3]])
    fill_value = -1

    writer = NetCDFWriter(tmp_path)

    # Create and write first chunk - KEEP the dot prefix
    buffer1 = SimulationDataBuffer()
    for i in range(5):
        buffer1.add(i, float(i), float(i), float(i))
    buffer1.write_to_disk(node_x, node_y, face_node_connectivity, fill_value, writer, '.sim_buffer_0.nc')

    # Create and write second chunk
    buffer2 = SimulationDataBuffer()
    for i in range(5, 10):
        buffer2.add(i, float(i), float(i), float(i))
    buffer2.write_to_disk(node_x, node_y, face_node_connectivity, fill_value, writer, '.sim_buffer_1.nc')

    # Make sure we get the correct output directory
    output_dir = writer.output_dir
    # Merge the files
    SimulationDataBuffer.merge_output_files(output_dir, 'merged_test.nc')
    # Check that merged file exists
    merged_file = output_dir / 'merged_test.nc'
    assert merged_file.exists()

    # Open merged file and verify data
    ds = xu.open_dataset(merged_file)
    assert 'x' in ds
    assert 'y' in ds
    assert 'particle_id' in ds
    assert len(ds.time) == 10  # Should have 10 time steps total


def test_merge_output_files_no_files(tmp_path):
    """
    Test that merge_output_files raises an error when no files are found.
    """
    # Use the EXACT error message including the period at the end
    with pytest.raises(FileNotFoundError, match='No .sim_buffer_\\*.nc files found to merge\\.'):
        SimulationDataBuffer.merge_output_files(tmp_path, 'merged_test.nc')
