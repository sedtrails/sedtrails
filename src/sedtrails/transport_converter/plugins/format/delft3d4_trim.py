"""A plugin for converting Delft3D4 TRIM format to SedTRAILS format."""

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData


class FormatPlugin(BaseFormatPlugin):
    """
    Plugin for converting Delft3D4 TRIM format to SedTRAILS format.
    """

    def convert(self, *args, **kwargs) -> SedtrailsData:
        """
        Converts  from Delft3D4 TRIM format.

        Parameters:
        -----------
        *args : tuple
            Positional arguments for conversion.
        **kwargs : dict
            Keyword arguments for conversion.
        """
        raise NotImplementedError('NetCDF DFM conversion is not yet implemented.')
