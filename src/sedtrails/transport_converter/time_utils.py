"""Utilities for transport-converter time handling."""

from typing import Any

import numpy as np


def decompress_time_info(time_info: dict[str, Any], morfac: float) -> dict[str, Any]:
    """
    Apply morfac decompression to converter time information.

    Parameters
    ----------
    time_info : dict
        Time metadata containing ``seconds_since_reference`` and
        ``reference_date`` entries.
    morfac : float
        Morphological acceleration factor. A value of 1.0 leaves the time
        coordinate unchanged.

    Returns
    -------
    dict
        Copy of ``time_info`` with decompressed ``time_values``,
        ``time_start``, ``time_end``, and ``seconds_since_reference`` entries.
    """
    decompressed_info = time_info.copy()
    seconds_since_reference = np.asarray(time_info['seconds_since_reference'], dtype=float)
    if seconds_since_reference.size == 0:
        return decompressed_info

    seconds_start = seconds_since_reference[0]
    decompressed_seconds = seconds_start + (seconds_since_reference - seconds_start) * float(morfac)
    reference_date = time_info['reference_date']

    decompressed_info['seconds_since_reference'] = decompressed_seconds
    decompressed_info['time_values'] = np.array(
        [reference_date + np.timedelta64(int(round(seconds * 1_000_000)), 'us') for seconds in decompressed_seconds]
    )
    decompressed_info['time_start'] = decompressed_info['time_values'][0]
    decompressed_info['time_end'] = decompressed_info['time_values'][-1]
    return decompressed_info
