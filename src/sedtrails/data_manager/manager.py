from pathlib import Path
from typing import Union

from sedtrails.data_manager.netcdf_writer import NetCDFWriter


class DataManager:
    """
    Owns SedTRAILS output writers and output-directory state.

    The previous in-memory xarray buffering path has been removed. Simulation
    output should use ``writer.open_output()``, ``writer.record_output()``, and
    ``writer.close_output()`` so particle tracks are streamed directly to disk.
    """

    def __init__(self, output_dir: Union[str, Path], max_bytes=None):
        """
        Initialize the data manager.

        Parameters
        ----------
        output_dir : str or Path
            Path to the output directory where data will be stored.
        max_bytes : object, optional
            Deprecated compatibility argument from the removed xarray buffer
            path. It is accepted but ignored.
        """
        self.writer = NetCDFWriter(output_dir)
        self.output_dir = self.writer.output_dir
