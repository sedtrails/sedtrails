import numpy as np

from sedtrails.data_manager.xarray_dataset import collect_timestep_data, create_sedtrails_dataset


class MockPopulation:
    """Minimal population stub exposing particle arrays used by dataset export."""

    def __init__(self):
        """Provides deterministic particle state for a single-timestep test."""
        # Keys mirror the particle fields consumed by collect_timestep_data.
        self.particles = {
            'x': np.array([1.0, 2.0]),
            'y': np.array([3.0, 4.0]),
            'burial_depth': np.array([0.0, 0.1]),
            'is_mobile': np.array([True, False]),
        }


def test_collect_timestep_data_exports_is_mobile_status():
    """Checks boolean mobility flags are written as integer status values."""
    ds = create_sedtrails_dataset(N_particles=2, N_populations=1, N_timesteps=1, N_flowfields=1)

    collect_timestep_data(ds, [MockPopulation()], timestep=0, current_time=12.0)

    np.testing.assert_array_equal(ds['status_mobile'].values[:, 0], np.array([1, 0]))
