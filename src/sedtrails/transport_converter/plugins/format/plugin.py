"""
A common interface for format plugins.
Plugins must inherit from this class and implement the convert method.
"""

from abc import ABC, abstractmethod
from sedtrails.transport_converter.sedtrails_data import SedtrailsData


class BaseFormatPlugin(ABC):
    """
    Abstract base class for format plugins.
    """

    @abstractmethod
    def convert(self, *args, **kwargs) -> SedtrailsData:
        """
        Converts flow-field data from varios formats to the sedtrails format.

        Returns
        -------
        SedtrailsData
            The converted SedtrailsData object.

        Parameters
        ----------
        *args : object
            Additional positional arguments passed through to the implementation.
        **kwargs : object
            Additional keyword arguments passed through to the implementation.
        """
        pass
