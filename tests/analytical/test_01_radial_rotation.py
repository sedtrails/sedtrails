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


def test_radial_rotation_closed_orbits():
    """Validate closed orbits for solid-body rotation against an analytic solution."""
    domain = 20_000.0
    gx, gy = rect_grid(-domain / 2, domain / 2, 201, -domain / 2, domain / 2, 201)
    calculator = create_numba_particle_calculator(gx, gy)

    # Solid-body rotation rate for a 24-hour period.
    omega = 2.0 * np.pi / 86_400.0

    def velocity_fn(_t):
        return -omega * gy, omega * gx

    x0 = np.zeros(4, dtype=np.float64)
    y0 = np.array([1_000.0, 2_000.0, 3_000.0, 4_000.0])

    # Integrate trajectories and keep a history for comparison plots.
    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=86_400.0,
        dt=300.0,
        return_history=True,
        history_stride=12,
    )

    # Closure error after one full rotation should be small.
    closing_error = np.sqrt((x_end - x0) ** 2 + (y_end - y0) ** 2)
    maybe_save_artifact('01_radial_rotation_closing_error', closing_error)

    analytic_x = np.stack([x0 * np.cos(omega * t) - y0 * np.sin(omega * t) for t in times])
    analytic_y = np.stack([x0 * np.sin(omega * t) + y0 * np.cos(omega * t) for t in times])
    skill_mean, skill_std, _ = liu_weisberg_skill(analytic_x, analytic_y, xs, ys)

    write_metrics(
        '01_radial_rotation_metrics',
        {
            'max_closing_error_m': float(np.max(closing_error)),
            'mean_closing_error_m': float(np.mean(closing_error)),
            'liu_weisberg_skill_mean': skill_mean,
            'liu_weisberg_skill_std': skill_std,
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fontdict = plot_fontdict()
        dense_times = np.linspace(0.0, 86_400.0, 400)
        for idx in range(xs.shape[1]):
            r = np.hypot(x0[idx], y0[idx])
            theta = np.arctan2(y0[idx], x0[idx]) + omega * dense_times
            ax.plot(
                r * np.cos(theta),
                r * np.sin(theta),
                **analytic_line_style(),
                label='analytic' if idx == 0 else None,
            )
            ax.plot(
                xs[:, idx],
                ys[:, idx],
                **sedtrails_line_style(),
                label='sedtrails' if idx == 0 else None,
            )
        ax.set_xlabel('x [m]', fontdict=fontdict)
        ax.set_ylabel('y [m]', fontdict=fontdict)
        ax.set_title('Radial rotation: SedTRAILS vs analytic', fontdict=fontdict)
        ax.legend(**legend_style())
        ax.axis('equal')
        return fig

    maybe_save_plot('01_radial_rotation_comparison', _plot)

    assert np.max(closing_error) < 25.0
