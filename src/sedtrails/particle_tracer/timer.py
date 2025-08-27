"""
Time related classes used internally to represent simulation time.
"""

from dataclasses import dataclass, field
import numpy as np
import re

from sedtrails.exceptions.exceptions import DateFormatError, DurationFormatError, ZeroDuration


def convert_duration_string_to_seconds(duration_str: str) -> int:
    """
    Parse a duration string ('3D 2H1M3S') into the total number of seconds.

    The duration string may include days (D), hours (H), minutes (M), and seconds (S)
    in any combination and order, separated by optional spaces. Missing units are treated as zero.

    Parameters
    ----------
    duration_str : str
        Duration string to parse ('3D 2H1M3S', '45S', '1H 30M').

    Returns
    -------
    int
        Total duration in seconds.

    Raises
    ------
    DurationFormatError
        If the input string does not match the expected format.
    """
    pattern = r'(?:(\d+)D)?\s*(?:(\d+)H)?\s*(?:(\d+)M)?\s*(?:(\d+)S)?'
    try:
        match = re.fullmatch(pattern, duration_str.strip())
        if not match:
            raise DurationFormatError(f"Invalid duration format: '{duration_str}' (expected e.g. '3D 2H1M3S')")
    except Exception as e:
        raise DurationFormatError(f"Invalid duration format: '{duration_str}' (expected e.g. '3D 2H1M3S')") from e

    days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def convert_datetime_string_to_datetime64(datetime_str: str) -> np.datetime64:
    """
    Convert date in str format to numpy.datetime64 enforcing 'YYYY-MM-DD hh:mm:ss'.

    Parameters
    ----------
    datetime_str : str
        Date string in format 'YYYY-MM-DD hh:mm:ss'.

    Returns
    -------
    np.datetime64
        The datetime as a numpy.datetime64 object with second precision.

    Raises
    ------
    DateFormatError
        If the input string does not match the required format.
    """
    if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', datetime_str):
        raise DateFormatError(f"date string '{datetime_str}' does not match required format 'YYYY-MM-DD hh:mm:ss'")
    return np.datetime64(datetime_str, 's')


class Duration:
    """
    Class representing a duration in the simulation.

    Attributes
    ----------
    _duration : str
        The duration as a string in format '3D 2H1M3S'.

    Properties
    ----------
    string : str
        Returns the duration string.
    seconds : int
        Returns the duration in seconds.
    deltatime : np.timedelta64
        Returns the duration as a numpy.timedelta64 object.
    """

    def __init__(self, duration: str):
        """
        Initialize a Duration object.

        Parameters
        ----------
        duration : str
            Duration string in format '3D 2H1M3S'.
        """
        self._duration = duration

    @property
    def string(self) -> str:
        """
        Returns the duration string.

        Returns
        -------
        str
            The original duration string in format '3D 2H1M3S'.
        """
        return self._duration

    @property
    def seconds(self) -> int:
        """
        Returns the duration in seconds.

        Returns
        -------
        int
            Total duration converted to seconds.

        Raises
        ------
        DurationFormatError
            If the duration string format is invalid.
        """
        return convert_duration_string_to_seconds(self._duration)

    @property
    def deltatime(self) -> np.timedelta64:
        """
        Converts the duration string to a numpy.timedelta64 object.

        Returns
        -------
        numpy.timedelta64
            The duration as a Numpy timedelta64 object.
        """
        return np.timedelta64(self.seconds, 's')


@dataclass
class Time:
    """
    Class representing time parameters in the simulation.

    Attributes
    ----------
    _start : str
        The simulation start time as string in format 'YYYY-MM-DD hh:mm:ss'.
    time_step : Duration
        The simulation time step as a Duration object. Defaults to Duration('1H').
    duration : Duration
        The simulation duration as a Duration object. Defaults to Duration('3D 2H1M3S').
    read_input_timestep : Duration
        The timestep for reading input data in chunks as a Duration object.
        Defaults to Duration('30D12H25M0S').
    cfl_condition : float
        The Courant-Friedrichs-Lewy (CFL) condition for the simulation.
        Defaults to 0.7
    reference_date : str
        The reference date as string in format 'YYYY-MM-DD hh:mm:ss'.
        Defaults to UTC epoch '1970-01-01 00:00:00'.
    _start_time_np : np.datetime64
        Internal numpy datetime64 representation of the start time.
    """

    # reference_date and start_time should be a str,
    # the class will do the transformation into a numpy.datetime64 object
    _start: str
    time_step: Duration = field(default_factory=lambda: Duration('1H'), init=True)
    duration: Duration = field(default_factory=lambda: Duration('3D 2H1M3S'), init=True)
    read_input_timestep: Duration = field(default_factory=lambda: Duration('30D12H25M0S'), init=True)
    reference_date: str = field(default='1970-01-01 00:00:00')
    cfl_condition: float = field(default=0.7)
    _start_time_np: np.datetime64 = field(init=False)

    def __post_init__(self):
        """
        Validates time_step, duration, and read_input_timestep aren't zero and initializes internal datetime representation.

        Raises
        ------
        ZeroDuration
            If time_step, duration, or read_input_timestep is zero length.
        DateFormatError
            If the start time string format is invalid.
        """
        if self.time_step.seconds == 0:
            raise ZeroDuration('time_step cannot be of length zero')
        if self.duration.seconds == 0:
            raise ZeroDuration('duration cannot be of length zero')
        if self.read_input_timestep.seconds == 0:
            raise ZeroDuration('read_input_timestep cannot be of length zero')

        # Convert start time to numpy.datetime64
        self._start_time_np = convert_datetime_string_to_datetime64(self._start)

    @property
    def start(self) -> int:
        """Returns the simulation start time as an integer representing seconds since the reference date."""
        start_datetime = convert_datetime_string_to_datetime64(self._start)
        reference_datetime = convert_datetime_string_to_datetime64(self.reference_date)
        # Calculate the difference in seconds
        delta_seconds = (start_datetime - reference_datetime).astype('timedelta64[s]').astype(int)
        return delta_seconds

    @start.setter
    def start(self, value: str) -> None:
        """
        Sets the simulation start time from a string in format 'YYYY-MM-DD hh:mm:ss'.

        Parameters
        ----------
        value : str
            The new start time as a string.

        Raises
        ------
        DateFormatError
            If the date string does not match the required format.
        """
        if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', value):
            raise DateFormatError(f"date string '{value}' does not match required format 'YYYY-MM-DD hh:mm:ss'")
        self._start = value

    @property
    def end(self) -> int:
        """
        Returns the simulation end-time in seconds since reference date.

        Returns
        -------
        int
            The simulation end time as seconds since the reference date.
        """
        return self.start + self.duration.seconds


@dataclass
class Timer:
    """
    Class representing a timer for simulation time management with variable timestep support.

    Manages the current simulation time and provides methods to advance
    through the simulation timeline with adaptive timesteps.

    Attributes
    ----------
    simulation_time : Time
        The Time object containing simulation time parameters.
    _current : int | float
        Current time in seconds since reference date (internal field).
    _current_timestep : int | float
        Current adaptive timestep in seconds.

    Properties
    ----------
    current : int | float
        Returns/sets the current time in seconds since reference time.
    current_timestep : int | float
        Returns/sets the current adaptive timestep in seconds.
    next : int | float
        Returns the next time step as current + current_timestep.
    step_count : int
        Returns the number of steps taken in the simulation.

    Methods
    -------
    advance()
        Advance the current time by current timestep.
    set_timestep(timestep)
        Set the current adaptive timestep.
    compute_cfl_timestep(flow_data, sedtrails_data, cfl_condition)
        Compute CFL-based timestep and update current timestep.
    """

    simulation_time: Time
    _current: int | float = field(init=False)  # in seconds
    _current_timestep: int | float = field(init=False)  # adaptive timestep in seconds
    cfl_condition: float = field(default=0.7)
    stop: bool = False

    def __post_init__(self):
        """
        Initialize current time from the simulation start time and set initial timestep.

        Sets the current time to the simulation start time in seconds
        since the reference date and initializes the adaptive timestep.
        """
        self._current = self.simulation_time.start
        self._current_timestep = self.simulation_time.time_step.seconds
        self.step_count = 0

    @property
    def current(self) -> int | float:
        """
        Returns the current time in seconds since reference time.
        """
        return self._current

    @current.setter
    def current(self, value: int | float) -> None:
        """
        Sets a value for the current time.

        Parameters
        ----------
        value : int | float
            The new current time in seconds since the reference date.

        Raises
        ------
        TypeError
            If value is not an integer or float.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected 'current' to be integer or float, got {type(value).__name__}")
        self._current = value

    @property
    def current_timestep(self) -> int | float:
        """
        Returns the current adaptive timestep in seconds.
        """
        return self._current_timestep

    @current_timestep.setter
    def current_timestep(self, value: int | float) -> None:
        """
        Sets the current adaptive timestep.

        Parameters
        ----------
        value : int | float
            The new timestep in seconds.
        """
        self._current_timestep = value

    @property
    def next(self) -> int | float:
        """
        Returns the next time as current + current_timestep in seconds.

        Returns
        -------
        int | float
            The next time step value in seconds since reference date.
        """
        return self._current + self._current_timestep

    def set_timestep(self, timestep: float) -> None:
        """
        Set the current adaptive timestep.

        Parameters
        ----------
        timestep : float
            The new timestep in seconds.
        """
        self._current_timestep = timestep

    def advance(self) -> None:
        """
        Advance the current time by current timestep.

        Moves the current time forward by the current adaptive timestep if it doesn't
        exceed the simulation end time.

        Raises
        ------
        ValueError
            If advancing would exceed the simulation end time.
        """

        if self.stop:
            raise RuntimeWarning('Timer has stopped. Cannot advance time.')

        if self.next <= self.simulation_time.end:
            self._current = self.next
            self.step_count += 1
        else:
            self.stop = True

    def compute_cfl_timestep(self, flow_data_list: list, sedtrails_data) -> float:
        """
        Compute CFL-based timestep from multiple flow fields and update current timestep.

        Parameters
        ----------
        flow_data_list : list
            List of flow field data dictionaries, each with 'magnitude' key
        sedtrails_data : SedtrailsData
            SedTRAILS data containing min_resolution

        Returns
        -------
        float
            Computed timestep in seconds
        """

        if self.cfl_condition > 0:
            # Find maximum velocity across all flow fields
            max_velocity = 0.0
            for flow_data in flow_data_list:
                magnitude = flow_data['magnitude']

                # Handle NaNs by setting them to 0
                magnitude_clean = np.nan_to_num(magnitude, nan=0.0)
                max_velocity = max(max_velocity, np.max(magnitude_clean))
                max_velocity = max(max_velocity, 1e-12)

            min_resolution = sedtrails_data.metadata.min_resolution
            cfl_timestep = self.cfl_condition * min_resolution / max_velocity

            # Ensure CFL timestep doesn't exceed sedtrails data timestep
            cfl_timestep = min(cfl_timestep, sedtrails_data.metadata.timestep)

            self.set_timestep(cfl_timestep)
