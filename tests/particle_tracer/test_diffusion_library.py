"""Unit tests for the supported Brownian diffusion strategy."""

import numpy as np

from sedtrails.particle_tracer.diffusion_library import BrownianDiffusionStrategy, DiffusionCalculator


def test_brownian_diffusion_with_seeded_generator_is_reproducible():
    """Independent calculators with the same seed generate identical walks."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    u = np.zeros_like(x)
    v = np.zeros_like(y)

    first = DiffusionCalculator(BrownianDiffusionStrategy(), rng=np.random.default_rng(1234))
    second = DiffusionCalculator(BrownianDiffusionStrategy(), rng=np.random.default_rng(1234))

    first_x, first_y = first.calc_diffusion(x, y, u, v, 0.5, 0.1)
    second_x, second_y = second.calc_diffusion(x, y, u, v, 0.5, 0.1)

    np.testing.assert_allclose(first_x, second_x)
    np.testing.assert_allclose(first_y, second_y)


def test_brownian_diffusion_zero_coefficient_is_a_noop():
    """A zero coefficient preserves coordinates without consuming randomness."""
    x = np.array([0.0, 1.0])
    y = np.array([2.0, 3.0])
    strategy = BrownianDiffusionStrategy()

    x_new, y_new = strategy.calculate(1.0, x, y, np.zeros_like(x), np.zeros_like(y), 0.0)

    np.testing.assert_array_equal(x_new, x)
    np.testing.assert_array_equal(y_new, y)
