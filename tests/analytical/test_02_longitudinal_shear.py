import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator

from ._helpers import (
    analytic_line_style,
    integrate_time_dependent,
    legend_style,
    liu_weisberg_skill,
    maybe_save_artifact,
    maybe_save_plot,
    plot_fontdict,
    rect_grid,
    sedtrails_line_style,
    write_metrics,
)


def test_longitudinal_shear_adapted_linear_shear():
    """Validate linear zonal flow against the analytic shear solution."""
    gx, gy = rect_grid(-15_000_000.0, 15_000_000.0, 121, -8_000_000.0, 8_000_000.0, 81)
    calculator = create_numba_particle_calculator(gx, gy)

    # Convert degrees to meters using a flat-Earth approximation.
    meters_per_degree = 111_000.0
    u_m_per_s = 1.0

    def velocity_fn(_t):
        return u_m_per_s * np.ones_like(gy), np.zeros_like(gy)

    x0 = np.zeros(31, dtype=np.float64)
    y0 = np.linspace(-30.0, 60.0, x0.size) * meters_per_degree

    total_time = 57.0 * 86_400.0
    # Integrate particle trajectories and sample a daily history.
    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=total_time,
        dt=300.0,
        return_history=True,
        history_stride=int(86_400.0 / 300.0),
    )

    # Analytic solution is uniform advection in x with constant u.
    u_particles = np.full_like(y0, u_m_per_s)
    x_true = x0 + u_particles * total_time
    y_true = y0

    maybe_save_artifact('02_longitudinal_shear_x_error', x_end - x_true)
    maybe_save_artifact('02_longitudinal_shear_y_error', y_end - y_true)

    analytic_x = np.stack([x0 + u_particles * t for t in times])
    analytic_y = np.stack([y0 for _ in times])
    skill_mean, skill_std, _ = liu_weisberg_skill(analytic_x, analytic_y, xs, ys)

    write_metrics(
        '02_longitudinal_shear_metrics',
        {
            'max_abs_x_error_m': float(np.max(np.abs(x_end - x_true))),
            'mean_abs_x_error_m': float(np.mean(np.abs(x_end - x_true))),
            'max_abs_y_error_m': float(np.max(np.abs(y_end - y_true))),
            'liu_weisberg_skill_mean': skill_mean,
            'liu_weisberg_skill_std': skill_std,
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fontdict = plot_fontdict()
        dense_times = np.linspace(0.0, total_time, 400)
        for idx in range(xs.shape[1]):
            lat_deg = y0[idx] / meters_per_degree
            lat_rad = np.deg2rad(lat_deg)
            deg_per_meter = 1.0 / (meters_per_degree * np.cos(lat_rad))
            analytic_x = x0[idx] + u_particles[idx] * dense_times
            analytic_y = np.full_like(dense_times, y0[idx])
            ax.plot(
                analytic_x * deg_per_meter,
                analytic_y / meters_per_degree,
                **analytic_line_style(),
                label='analytic' if idx == 0 else None,
            )
            ax.plot(
                xs[:, idx] * deg_per_meter,
                ys[:, idx] / meters_per_degree,
                **sedtrails_line_style(),
                label='sedtrails' if idx == 0 else None,
            )
        ax.set_xlabel('longitude [degrees]', fontdict=fontdict)
        ax.set_ylabel('latitude [degrees]', fontdict=fontdict)
        ax.set_title('Longitudinal shear: SedTRAILS vs analytic', fontdict=fontdict)
        ax.legend(**legend_style())
        return fig

    maybe_save_plot('02_longitudinal_shear_comparison', _plot)

    np.testing.assert_allclose(x_end, x_true, rtol=1e-2, atol=2.0)
    np.testing.assert_allclose(y_end, y_true, rtol=1e-6, atol=1e-6)
