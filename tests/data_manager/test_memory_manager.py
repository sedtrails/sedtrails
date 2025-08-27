import numpy as np
from sedtrails.data_manager.memory_manager import MemoryManager


def create_test_mesh():
    """
    Create a minimal mesh for testing purposes.

    Returns
    -------
    tuple
        node_x, node_y, face_node_connectivity, fill_value
    """
    node_x = np.array([0, 1, 1, 0])
    node_y = np.array([0, 0, 1, 1])
    face_node_connectivity = np.array([[0, 1, 2, 3]])
    fill_value = -1
    return node_x, node_y, face_node_connectivity, fill_value


def test_buffer_size_bytes_counts_numpy_and_lists():
    """
    Test that buffer_size_bytes correctly counts memory for numpy arrays and lists.
    """
    mm = MemoryManager()
    buffer = {'a': np.zeros(100, dtype=np.float64), 'b': [1] * 100}
    size = mm.buffer_size_bytes(buffer)
    assert size > 0


def test_is_limit_exceeded():
    """
    Test that is_limit_exceeded correctly identifies when a buffer exceeds the memory limit.
    """
    mm = MemoryManager(max_bytes=100)
    small_buffer = {'data': np.zeros(10)}
    large_buffer = {'data': np.zeros(1000)}

    assert not mm.is_limit_exceeded(small_buffer)  # Should be under limit
    assert mm.is_limit_exceeded(large_buffer)  # Should exceed limit
