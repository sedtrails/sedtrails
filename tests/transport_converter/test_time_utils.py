import numpy as np

from sedtrails.transport_converter.time_utils import decompress_time_info


def test_decompress_time_info_applies_morfac_to_seconds_and_datetimes():
    """Morfac decompression should stretch the forcing time axis from its first timestamp."""
    reference_date = np.datetime64('2024-01-01T00:00:00')
    time_info = {
        'time_values': np.array(['2024-01-01T00:01:40', '2024-01-01T00:02:40'], dtype='datetime64[ns]'),
        'time_start': np.datetime64('2024-01-01T00:01:40'),
        'time_end': np.datetime64('2024-01-01T00:02:40'),
        'seconds_since_reference': np.array([100.0, 160.0]),
        'reference_date': reference_date,
        'num_times': 2,
    }

    decompressed = decompress_time_info(time_info, morfac=2.0)

    np.testing.assert_array_equal(decompressed['seconds_since_reference'], np.array([100.0, 220.0]))
    assert decompressed['time_start'] == reference_date + np.timedelta64(100, 's')
    assert decompressed['time_end'] == reference_date + np.timedelta64(220, 's')


def test_decompress_time_info_keeps_empty_time_info_unchanged():
    """Empty time coordinates should pass through without synthetic bounds."""
    time_info = {
        'seconds_since_reference': np.array([], dtype=float),
        'reference_date': np.datetime64('2024-01-01T00:00:00'),
    }

    decompressed = decompress_time_info(time_info, morfac=3.0)

    np.testing.assert_array_equal(decompressed['seconds_since_reference'], np.array([], dtype=float))
    assert 'time_start' not in decompressed
    assert 'time_end' not in decompressed
