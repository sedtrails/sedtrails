"""
Unit tests for the Timer class in the timer.py module of the sedtrails package.
"""

import pytest
import numpy as np
from sedtrails.particle_tracer.timer import Duration, Timer, Time
from sedtrails.exceptions.exceptions import DateFormatError, DurationFormatError, ZeroDuration


class TestDuration:
    """Test cases for the Duration class."""

    def test_duration_init(self):
        """Test Duration initialization."""
        duration = Duration('3D 2H1M3S')
        assert duration.string == '3D 2H1M3S'

    def test_seconds_full_format(self):
        """Test conversion of full duration format to seconds."""
        duration = Duration('3D 2H1M3S')
        expected = 3 * 86400 + 2 * 3600 + 1 * 60 + 3  # 266463 seconds
        assert duration.seconds == expected

    def test_seconds_partial_formats(self):
        """Test conversion of partial duration formats to seconds."""
        test_cases = [
            ('1D', 86400),
            ('2H', 7200),
            ('30M', 1800),
            ('45S', 45),
            ('1H 30M', 5400),
            ('2D 5H', 2 * 86400 + 5 * 3600),
            ('1M 30S', 90),
        ]

        for duration_str, expected_seconds in test_cases:
            duration = Duration(duration_str)
            assert duration.seconds == expected_seconds

    def test_seconds_with_spaces(self):
        """Test duration parsing with various spacing."""
        test_cases = [
            ('1D 2H 3M 4S', 1 * 86400 + 2 * 3600 + 3 * 60 + 4),
            ('1D2H3M4S', 1 * 86400 + 2 * 3600 + 3 * 60 + 4),
            ('  1D  2H  3M  4S  ', 1 * 86400 + 2 * 3600 + 3 * 60 + 4),
        ]

        for duration_str, expected_seconds in test_cases:
            duration = Duration(duration_str)
            assert duration.seconds == expected_seconds

    def test_seconds_zero_duration(self):
        """Test conversion of zero/empty duration."""
        duration = Duration('')
        # Empty string should return 0 seconds as all groups are optional
        assert duration.seconds == 0

    def test_seconds_invalid_format(self):
        """Test invalid duration formats raise DurationFormatError."""
        invalid_durations = [
            'invalid',
            '1X',
            '1D 2X',
            '1.5D',
            '-1D',
        ]

        for invalid_duration in invalid_durations:
            duration = Duration(invalid_duration)
            with pytest.raises(DurationFormatError):
                _ = duration.seconds

    def test_seconds_valid_edge_cases(self):
        """Test edge cases that should be valid."""
        # These should work fine
        valid_cases = [
            ('1D 25H', 1 * 86400 + 25 * 3600),  # 25 hours is valid
            ('0D', 0),
            ('0H', 0),
            ('0M', 0),
            ('0S', 0),
        ]

        for duration_str, expected_seconds in valid_cases:
            duration = Duration(duration_str)
            assert duration.seconds == expected_seconds

    def test_deltatime(self):
        """Test conversion to numpy timedelta64."""
        duration = Duration('1D 2H3M4S')
        timedelta = duration.deltatime
        expected_seconds = 1 * 86400 + 2 * 3600 + 3 * 60 + 4
        expected_timedelta = np.timedelta64(expected_seconds, 's')
        assert timedelta == expected_timedelta


class TestTimer:
    """Test cases for the Timer class."""

    def test_timer_init(self):
        """Test Timer initialization."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1H'))
        timer = Timer(simulation_time=time)
        assert timer.simulation_time == time
        assert timer.current == time.start  # Should be seconds since reference date

    def test_timer_with_zero_duration_raises_error(self):
        """Test Timer with zero duration raises ZeroDuration error."""
        with pytest.raises(ZeroDuration):
            time = Time(_start='2023-01-01 12:00:00', time_step=Duration('0H'))
            Timer(simulation_time=time)

    def test_timer_with_zero_time_step_raises_error(self):
        """Test Timer with zero time_step raises ZeroDuration error."""
        with pytest.raises(ZeroDuration):
            time = Time(_start='2023-01-01 12:00:00', duration=Duration('0D'))
            Timer(simulation_time=time)

    def test_timer_invalid_date_format(self):
        """Test Timer with invalid date format raises DateFormatError."""
        invalid_dates = [
            '2023-1-1 12:00:00',  # Single digit month/day
            '23-01-01 12:00:00',  # Two digit year
            '2023-01-01 12:00',  # Missing seconds
            '2023/01/01 12:00:00',  # Wrong separator
            'invalid date',
        ]

        for invalid_date in invalid_dates:
            with pytest.raises(DateFormatError):
                time = Time(_start=invalid_date, time_step=Duration('1H'))
                Timer(simulation_time=time)

    def test_current_property_getter(self):
        """Test current property getter."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1H'))
        timer = Timer(simulation_time=time)
        # Current should be start time in seconds since reference date
        current_value = timer.current
        assert isinstance(current_value, (int, float, np.integer)), (
            f'Expected int/float/np.integer, got {type(current_value)}'
        )
        assert current_value == time.start

    def test_current_property_setter(self):
        """Test current property setter."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1H'))
        timer = Timer(simulation_time=time)
        new_time = 3600  # 1 hour in seconds
        timer.current = new_time
        assert timer.current == new_time

    def test_current_property_setter_invalid_type(self):
        """Test current property setter with invalid type raises TypeError."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1H'))
        timer = Timer(simulation_time=time)
        with pytest.raises(TypeError):
            timer.current = '2023-01-02 15:30:00'

    def test_next_property(self):
        """Test next property returns current + time_step."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1H'))
        timer = Timer(simulation_time=time)
        expected = timer.current + 3600  # 1 hour in seconds
        assert timer.next == expected

    def test_next_property_complex_time_step(self):
        """Test next property with complex time step."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1D 2H30M'))
        timer = Timer(simulation_time=time)
        # 1D 2H30M = 86400 + 7200 + 1800 = 95400 seconds
        expected = timer.current + 95400
        assert timer.next == expected

    def test_advance(self):
        """Test advance method updates current time."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1H'))
        timer = Timer(simulation_time=time)
        original_current = timer.current
        original_next = timer.next

        timer.advance()

        assert timer.current == original_next
        assert timer.current != original_current

    def test_multiple_advances(self):
        """Test multiple advances work correctly."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('30M'))
        timer = Timer(simulation_time=time)
        original_current = timer.current

        # First advance (30 minutes = 1800 seconds)
        timer.advance()
        assert timer.current == original_current + 1800

        # Second advance (another 30 minutes)
        timer.advance()
        assert timer.current == original_current + 3600

        # Third advance (another 30 minutes)
        timer.advance()
        assert timer.current == original_current + 5400

    def test_advance_beyond_simulation_end(self):
        """Test advance method when reaching simulation end."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1H'), duration=Duration('30M'))
        timer = Timer(simulation_time=time)

        # Try to advance beyond the end (30 minutes duration, but 1 hour step)
        timer.advance()

        # Timer should stop when it would exceed the end
        assert timer.stop

    def test_advance_when_stopped_raises_warning(self):
        """Test advance raises RuntimeWarning when timer is stopped."""
        time = Time(_start='2023-01-01 12:00:00', time_step=Duration('1H'), duration=Duration('30M'))
        timer = Timer(simulation_time=time)

        # Advance to stop the timer
        timer.advance()
        assert timer.stop

        # Try to advance again should raise RuntimeWarning
        with pytest.raises(RuntimeWarning):
            timer.advance()


class TestTime:
    """Test cases for the Time class."""

    def test_time_default_init(self):
        """Test Time initialization with default values."""
        time = Time(_start='1970-01-01 00:00:00')
        assert time.reference_date == '1970-01-01 00:00:00'
        assert isinstance(time.duration, Duration)
        assert time.duration.string == '3D 2H1M3S'

    def test_time_custom_init(self):
        """Test Time initialization with custom values."""
        custom_duration = Duration('5D 10H')
        time = Time(_start='2023-01-15 12:00:00', reference_date='2023-01-01 00:00:00', duration=custom_duration)
        assert time.reference_date == '2023-01-01 00:00:00'
        assert time.duration == custom_duration

    def test_start_property(self):
        """Test start property returns seconds since reference date."""
        time = Time(_start='2023-01-01 12:00:00', reference_date='2023-01-01 00:00:00')
        # 12 hours = 12 * 3600 = 43200 seconds
        assert time.start == 43200

    def test_end_property(self):
        """Test end property calculation."""
        duration = Duration('2D 12H')
        time = Time(_start='2023-01-01 12:00:00', duration=duration, reference_date='2023-01-01 00:00:00')
        # Start: 12 hours = 43200 seconds
        # Duration: 2D 12H = 2*86400 + 12*3600 = 216000 seconds
        # End: 43200 + 216000 = 259200 seconds
        expected_end = 43200 + 216000
        assert time.end == expected_end

    def test_end_property_complex_duration(self):
        """Test end property with complex duration."""
        duration = Duration('1D 2H30M45S')
        time = Time(_start='2023-01-01 00:00:00', duration=duration, reference_date='2023-01-01 00:00:00')
        # Start: 0 seconds (same as reference)
        # Duration: 1*86400 + 2*3600 + 30*60 + 45 = 95445 seconds
        expected_end = 95445
        assert time.end == expected_end

    def test_time_invalid_start_date(self):
        """Test Time with invalid start date format."""
        with pytest.raises(DateFormatError):
            Time(_start='invalid-date')

    def test_time_start_setter(self):
        """Test start setter with valid and invalid dates."""
        time = Time(_start='2023-01-01 12:00:00')

        # Valid date should work
        time.start = '2023-01-02 15:30:45'

        # Invalid date should raise error
        with pytest.raises(DateFormatError):
            time.start = 'invalid-date'

    def test_zero_duration_validation(self):
        """Test that zero duration raises ZeroDuration error."""
        with pytest.raises(ZeroDuration):
            Time(_start='2023-01-01 12:00:00', duration=Duration('0S'))

    def test_zero_time_step_validation(self):
        """Test that zero time_step raises ZeroDuration error."""
        with pytest.raises(ZeroDuration):
            Time(_start='2023-01-01 12:00:00', time_step=Duration('0S'))

    def test_duration_integration(self):
        """Test integration between Time and Duration classes."""
        duration = Duration('7D')
        time = Time(_start='2023-12-25 00:00:00', duration=duration, reference_date='2023-12-25 00:00:00')
        # 7 days = 7 * 86400 = 604800 seconds
        expected_end = 604800
        assert time.end == expected_end


class TestIntegration:
    """Integration tests for Timer and Time classes working together."""

    def test_timer_with_time_parameters(self):
        """Test Timer using parameters from Time class."""
        time = Time(_start='2023-01-01 00:00:00', duration=Duration('2D'), time_step=Duration('6H'))

        timer = Timer(simulation_time=time)

        assert timer.current == time.start
        initial_time = timer.current

        # Advance through the simulation (6H steps)
        timer.advance()  # +6H
        assert timer.current == initial_time + 6 * 3600

        timer.advance()  # +12H
        assert timer.current == initial_time + 12 * 3600

        timer.advance()  # +18H
        assert timer.current == initial_time + 18 * 3600

        timer.advance()  # +24H (1 day)
        assert timer.current == initial_time + 24 * 3600

        # Check if we're still within the simulation duration
        assert timer.current < time.end

    def test_timer_simulation_boundary(self):
        """Test timer advancing to simulation end boundary."""
        time = Time(_start='2023-01-01 00:00:00', duration=Duration('1D'), time_step=Duration('12H'))

        timer = Timer(simulation_time=time)
        initial_time = timer.current

        # Should be able to advance twice to reach the end
        timer.advance()  # +12H
        assert timer.current == initial_time + 12 * 3600
        assert timer.current < time.end
        assert not timer.stop

        timer.advance()  # +24H (exactly 1 day)
        assert timer.current == initial_time + 24 * 3600
        assert timer.current == time.end
        assert not timer.stop

        # Try to advance one more time - should stop the timer
        timer.advance()
        assert timer.stop
