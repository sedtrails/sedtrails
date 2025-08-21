"""
NetCDF Reader

Reads NetCDF files produced by the SedTrails Particle Tracer System.
"""

import xugrid as xu
from pathlib import Path


class NetCDFReader:
    """
    A class for reading NetCDF files produced by the SedTrails Particle Tracer System.

    This class validates and loads NetCDF files using xugrid's lazy loading functionality.
    The file is automatically validated and loaded upon instantiation.

    Parameters:
    -----------
    file_path : str
        The path to the CF compliant NetCDF file.

    Attributes:
    -----------
    file_path : str
        The path to the CF compliant NetCDF file.
    data : xu.UgridDataset
        The loaded NetCDF dataset as an xugrid UgridDataset object.

    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file exists but is not a valid NetCDF file or has an invalid extension.

    Examples:
    ---------
    >>> reader = NetCDFReader('sample-data/example_output.nc')
    >>> print(reader.data)
    >>> particle_id = reader.data['particle_id']
    """

    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self._read_file()
            
    def _validate_file(self) -> bool:
        """
        Validates the file path to ensure it exists and is a NetCDF file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file exists but is not a valid NetCDF file.
        
        Returns:
            bool: True if the file is valid.
        """
        FILE_EXTENSIONS = ['.nc', '.nc4', '.netcdf']  # Added more common NetCDF extensions
        path = Path(self.file_path)
        
        # Check if file exists
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Check if file has valid NetCDF extension
        if path.suffix.lower() not in FILE_EXTENSIONS:
            raise ValueError(f"Invalid file type. Expected NetCDF file with extensions {FILE_EXTENSIONS}, got: {path.suffix}")
        
        # Optional: Check if file is actually readable as NetCDF
        try:
            xu.open_dataset(self.file_path)
        except Exception as e:
            raise ValueError(f"File exists but is not a valid NetCDF file: {e}") from e
        
        return True

    def _read_file(self):
        """
        Reads the NetCDF file using the lazy loader from xugrid.
        """
        if not self._validate_file():
            raise ValueError("Invalid file path. Ensure the file is a NetCDF file.")
        else:
            self.data = xu.open_dataset(self.file_path)


if __name__ == "__main__":
    reader = NetCDFReader('sample-data/example_output.nc')
    print(reader)

    # # dfm dataset:
    # fm = 'sample-data/dfm_sedtrails.nc'
    # d3 = 'sample-data/d3d4.dat'
    # # 
    # uds = xu.open_mfdataset(fm)  # return Ugrid dataset

    # uds = xu.open_dataset(fm)  # return Ugrid dataset


    # print(uds)

    # elev = uds['elevation']
    # print(elev)

    # elev.plot()
