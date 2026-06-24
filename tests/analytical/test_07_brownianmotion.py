import math
import numpy as np

from sedtrails.particle_tracer.diffusion_library import BrownianDiffusionStrategy

from ._helpers import (
    legend_style,
    maybe_save_artifact,
    maybe_save_plot,
    plot_fontdict,
    write_metrics,
)


def test_brownian_random_walk_moments():
    """Validate Brownian diffusion moments against the analytic Gaussian spread."""
    np.random.seed(123456)

    n_particles = 100_000

    x = np.zeros(n_particles, dtype=np.float64)
    y = np.zeros(n_particles, dtype=np.float64)
    dt = 300.0
    kh = 100.0
    total_time = 86_400.0
    steps = int(total_time / dt)

    # Random-walk diffusion with constant kh.
    diffusion = BrownianDiffusionStrategy()
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    for _ in range(steps):
        x, y = diffusion.calculate(dt=dt, x=x, y=y, u=u, v=v, kh=kh)

    dx = x
    dy = y

    # Analytic variance for Brownian motion in each dimension.
    expected_var = 2.0 * kh * total_time
    mean_tol = 3.0 * math.sqrt(expected_var / n_particles)

    maybe_save_artifact(
        '07_brownian_stats',
        np.array([dx.mean(), dy.mean(), dx.var(ddof=1), dy.var(ddof=1)]),
    )

    write_metrics(
        '07_brownian_metrics',
        {
            'mean_dx': float(dx.mean()),
            'mean_dy': float(dy.mean()),
            'var_dx': float(dx.var(ddof=1)),
            'var_dy': float(dy.var(ddof=1)),
            'expected_var': float(expected_var),
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fontdict = plot_fontdict()
        sigma_km = math.sqrt(expected_var) / 1000.0
        grid = np.linspace(-4.0 * sigma_km, 4.0 * sigma_km, 400)
        pdf = (1.0 / (sigma_km * math.sqrt(2.0 * math.pi))) * np.exp(-(grid**2) / (2.0 * sigma_km**2))
        ax.plot(grid, pdf, color='#0b2d5c', linewidth=1.6, linestyle='--', label='analytic PDF')
        ax.hist(dx / 1000.0, bins=60, alpha=0.5, density=True, color='#cfe3f7', label='dx')
        ax.hist(dy / 1000.0, bins=60, alpha=0.35, density=True, color='#a9c9ee', label='dy')
        ax.set_xlabel('displacement [km]', fontdict=fontdict)
        ax.set_ylabel('density', fontdict=fontdict)
        ax.set_title('Brownian motion: displacement histograms', fontdict=fontdict)
        ax.legend(**legend_style())
        return fig

    maybe_save_plot('07_brownian_histograms', _plot)

    assert abs(dx.mean()) < mean_tol
    assert abs(dy.mean()) < mean_tol
    np.testing.assert_allclose(dx.var(ddof=1), expected_var, rtol=0.1)
    np.testing.assert_allclose(dy.var(ddof=1), expected_var, rtol=0.1)
