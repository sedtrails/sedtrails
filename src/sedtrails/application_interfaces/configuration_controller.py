"""
Configuration Controller
========================

Reads simulation configuration files, applies default configuration values,
and provides configurations to other components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from sedtrails.application_interfaces.find import find_value
from sedtrails.application_interfaces.validator import YAMLConfigValidator


class Controller(ABC):
    """
    Abstract base class for configuration controllers.
    """

    # TODO: Something to keep in mind in this class:
    # if we can and it is convenient to have separated methods to retrieve parts of the configuration
    # that go to the particle tracer and the transport converter. For example, if it is convenient to
    # have a method that returns the configuration values that are only relevant for the particle tracer.

    @abstractmethod
    def load_config(self, config_file: str) -> None:
        """
        Reads the configuration file and applies default values.

        Parameters
        ----------
        config_file : str
            Path to the SedTRAILS configuration file.
        """

        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Returns the current configuration.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the requested values.
        """

        pass

    @abstractmethod
    def get(self, keys: str, default=None) -> Any:
        """
        Retrieves a value from the configuration data using a dot-separated key.

        Parameters
        ----------
        keys : str
            Dot-separated configuration key path to retrieve.
        default : object
            Fallback value used when the requested key is absent.

        Returns
        -------
        Any
            Requested value.
        """

        pass


class ConfigurationController(Controller):
    """
    A controller for managing the configuration of the simulation.

    Attributes
    ----------

    config : str
        The path to the configuration file.
    config_data : dict
        The configuration data loaded from the file.
    """

    def __init__(self, config_file: str) -> None:
        """
        Initializes the ConfigurationController with a configuration file.
        Parameters
        ----------

        config_file : str
            The path to the configuration file to read.
        """

        self.config: str = config_file
        self.config_data = {}
        self.raw_config_data = {}

    def load_config(self, config_file: str) -> None:
        """
        Updates the configuraton based on a configuration file, validates its contents and sets it.

        Parameters
        ----------

        config_file : str
            The path to the configuration file to read.

        Returns
        -------
        None
            This method returns None after updating the controller state.
        """

        if self.config is None and config_file is None:
            raise ValueError('Configuration file path must be provided to the ConfigurationController.')
        else:
            self.config = config_file
            validator = YAMLConfigValidator()
            self.config_data = validator.validate_yaml(config_file)
            self.raw_config_data = validator.raw_config

        return None

    def get_config(self) -> dict:
        """
        Returns the current configuration.

        Returns
        -------

        dict
            The current configuration.
        """

        if not self.config_data:
            self.load_config(self.config)

        return self.config_data

    def get(self, keys: str, default=None) -> Any:
        """
        Retrieves a value from the configuration data using a dot-separated key.

        Parameters
        ----------

        key : str
            The dot-separated key to retrieve from the configuration.
        default : any, optional
            The default value to return if the key is not found.

        Returns
        -------

        any
            The value associated with the key or the default value. None if the key is not found.

        Example
        --------

        >>> controller = ConfigurationController()
        >>> controller.read_config('path/to/config.yaml')
        >>> value = controller.get('some.nested.key')
        >>> print(value)  # Outputs the value for 'some.nested.key'
        """

        from copy import deepcopy  # Lazy import

        config_data = deepcopy(self.get_config())

        return find_value(config_data, keys, default)

    def has_configured_value(self, keys: str) -> bool:
        """
        Return whether a dot-separated key was present in the source configuration.

        Parameters
        ----------
        keys : str
            Dot-separated configuration key path to test.

        Returns
        -------
        bool
            True when every key segment exists in the raw user configuration,
            before schema defaults are applied.
        """

        if not hasattr(self, 'raw_config_data') or self.raw_config_data is None:
            self.get_config()

        current = self.raw_config_data
        for key in keys.split('.'):
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
        return True
