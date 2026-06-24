"""Legacy Delft3D4 TRIM converter stub.

The public configuration schema does not expose Delft3D4 as a supported input
format. This module is retained as a placeholder and raises
``NotImplementedError`` if used directly.
"""

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData


class FormatPlugin(BaseFormatPlugin):
    """Placeholder for a future Delft3D4 TRIM converter."""

    def convert(self, *args, **kwargs) -> SedtrailsData:
        """Convert Delft3D4 TRIM input.

        Parameters
        ----------
        *args : tuple
            Positional arguments for conversion.
        **kwargs : dict
            Keyword arguments for conversion.

        Raises
        ------
        NotImplementedError
            Always raised because Delft3D4 TRIM conversion is not implemented.
        """
        raise NotImplementedError('Delft3D trim file conversion is not yet implemented.')
