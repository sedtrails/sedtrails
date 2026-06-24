from sedtrails.data_manager.manager import DataManager
from sedtrails.data_manager.netcdf_writer import NetCDFWriter


def test_data_manager_exposes_streaming_writer(tmp_path):
    """DataManager should be a lightweight owner for the streaming writer."""
    manager = DataManager(tmp_path)

    assert isinstance(manager.writer, NetCDFWriter)
    assert manager.output_dir == manager.writer.output_dir
    assert manager.output_dir.exists()


def test_data_manager_accepts_legacy_max_bytes_argument(tmp_path):
    """The removed xarray buffer size argument should not break construction."""
    manager = DataManager(tmp_path, max_bytes=1)

    assert isinstance(manager.writer, NetCDFWriter)
