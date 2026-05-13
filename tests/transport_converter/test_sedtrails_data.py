import numpy as np
import pytest

from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


def _build_sedtrails_data(x, y):
    """Builds a minimal SedtrailsData instance for metadata resolution tests."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    npoints = x.size
    ntimes = 2

    zeros_time_space = np.zeros((ntimes, npoints), dtype=float)
    zero_vector = {
        'x': zeros_time_space.copy(),
        'y': zeros_time_space.copy(),
        'magnitude': zeros_time_space.copy(),
    }

    metadata = SedtrailsMetadata(
        flowfield_domain={
            'x_min': float(np.min(x)),
            'x_max': float(np.max(x)),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
        }
    )

    # Keep all dynamic fields simple/zeroed so tests isolate spatial metadata behavior.
    return SedtrailsData(
        times=np.array([0.0, 1.0], dtype=float),
        reference_date=np.datetime64('1970-01-01T00:00:00'),
        x=x,
        y=y,
        bed_level=np.zeros(npoints, dtype=float),
        depth_avg_flow_velocity=zero_vector,
        fractions=1,
        bed_load_transport=zero_vector,
        suspended_transport=zero_vector,
        water_depth=zeros_time_space.copy(),
        mean_bed_shear_stress=zeros_time_space.copy(),
        max_bed_shear_stress=zeros_time_space.copy(),
        sediment_concentration=zeros_time_space.copy(),
        nonlinear_wave_velocity=zero_vector,
        metadata=metadata,
    )


def test_min_resolution_ignores_duplicate_coordinates():
    """Checks min resolution uses distinct points and ignores duplicates."""
    data = _build_sedtrails_data(
        x=[0.0, 1.0, 1.0, 2.0],
        y=[0.0, 0.0, 0.0, 0.0],
    )

    assert data.metadata.min_resolution == pytest.approx(1.0)


def test_min_resolution_none_when_no_distinct_points():
    """Checks degenerate coordinates produce no resolution/envelope metadata."""
    data = _build_sedtrails_data(
        x=[5.0, 5.0, 5.0],
        y=[7.0, 7.0, 7.0],
    )

    assert data.metadata.min_resolution is None
    assert data.metadata.outer_envelope == []
