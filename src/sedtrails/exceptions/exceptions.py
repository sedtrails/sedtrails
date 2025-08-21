"""
Custom exceptions for the Sedtrails package.
This module defines custom exceptions that can be raised during the execution of the Sedtrails package.
"""


class SedtrailsException(Exception):
    """
    Base class for all exceptions in the Sedtrails package.
    This class inherits from the built-in Exception class.
    """

    pass


# === YAML/Configuration Exceptions ===


class YamlParsingError(SedtrailsException):
    """
    Exception raised when there is an error parsing a YAML file.
    """

    pass


class YamlValidationError(SedtrailsException):
    """
    Exception raised when a YAML file does not conform to the expected schema.
    """

    pass


class YamlOutputError(SedtrailsException):
    """
    Exception raised when there is an error writing to a YAML file.
    """

    pass


class ConfigurationError(SedtrailsException):
    """
    Exception raised when configuration is invalid or cannot be loaded.
    """

    pass


# === Time/Date Exceptions ===


class DateFormatError(ValueError):
    """
    Exception raised when the date string does not match the required
    format 'YYYY-MM-DD 00:00:00'.
    """

    pass


class MissingConfigurationParameter(SedtrailsException):
    """
    Exception raised when a required configuration parameter is missing.
    This can occur if a necessary parameter is not provided in the configuration file,
    or an operation fails to fetch a required configuration value.
    """

    pass


class ZeroDuration(SedtrailsException):
    """
    Exception raised when a time representing a duration is zero.
    A simulation cannot run with a zero duration or time step.
    """

    pass


class DurationFormatError(ValueError):
    """
    Exception raised when the duration string does not match the required format '3D 2H1M3S'.
    """

    pass


# === Simulation Exceptions ===


class DataConversionError(SedtrailsException):
    """
    Exception raised when data conversion or processing fails.
    """

    pass


class ParticleInitializationError(SedtrailsException):
    """
    Exception raised when particle initialization fails.
    """

    pass


class NumbaCompilationError(SedtrailsException):
    """
    Exception raised when Numba JIT compilation fails.
    """

    pass


class SimulationExecutionError(SedtrailsException):
    """
    Exception raised during simulation loop execution.
    """

    pass


class VisualizationError(SedtrailsException):
    """
    Exception raised when visualization generation fails (typically non-critical).
    """

    pass


class OutputError(SedtrailsException):
    """
    Exception raised when output file generation fails.
    """

    pass
