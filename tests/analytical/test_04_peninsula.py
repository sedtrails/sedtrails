import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator

from ._helpers import (
    analytic_line_style,
    integrate_time_dependent,
    legend_style,
    maybe_save_artifact,
    maybe_save_plot,
    plot_fontdict,
    rect_grid,
    sedtrails_line_style,
    write_metrics,
)


def test_peninsula_streamfunction_conservation():
    """Validate streamfunction conservation around a peninsula flow."""
    gx, gy = rect_grid(0.0, 100.0, 100, 0.0, 50.0, 50)

    u0 = 0.001
    x_center = 50.0
    radius = 0.32 * 50.0

    def psi_fn(x, y):
        return u0 * radius**2 * y / ((x - x_center) ** 2 + y**2) - u0 * y

    # Velocity derived from the streamfunction, with a land mask.
    def velocity_fn(_t):
        denom = np.maximum(((gx - x_center) ** 2 + gy**2) ** 2, 1e-12)
        u = u0 - u0 * radius**2 * ((gx - x_center) ** 2 - gy**2) / denom
        v = -2.0 * u0 * radius**2 * ((gx - x_center) * gy) / denom
        land_mask = psi_fn(gx, gy) >= 0.0
        u = np.where(land_mask, 0.0, u)
        v = np.where(land_mask, 0.0, v)
        return u, v

    calculator = create_numba_particle_calculator(gx, gy)

    y0 = np.linspace(3.0, 47.0, 20)
    x0 = np.full(y0.size, 3.0)
    psi0 = psi_fn(x0, y0)

    # Integrate trajectories and stop if any particle reaches the open boundary.
    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=86_400.0,
        dt=300.0,
        return_history=True,
        stop_x=100.0,
        history_stride=int(3_600.0 / 300.0),
    )
    psi_end = psi_fn(x_end, y_end)

    # Relative streamfunction error provides a scale-aware tolerance.
    psi_rel = np.abs(psi_end - psi0) / np.maximum(np.abs(psi0), 1.0)
    maybe_save_artifact('04_peninsula_streamfunction_relative_error', psi_rel)

    write_metrics(
        '04_peninsula_metrics',
        {
            'max_rel_psi_error': float(np.max(psi_rel)),
            'mean_rel_psi_error': float(np.mean(psi_rel)),
        },
    )

    def _plot():
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots()
        fontdict = plot_fontdict()
        x_vals = np.linspace(np.min(gx), np.max(gx), 800)
        y_vals = np.linspace(np.min(gy), np.max(gy), 400)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
        psi_grid = psi_fn(xx, yy)
        ax.contourf(xx, yy, psi_grid >= 0.0, levels=[0.5, 1.5], colors=['#c8c8c8'])
        ax.contour(
            xx,
            yy,
            psi_grid,
            levels=np.sort(psi0),
            colors=analytic_line_style()['color'],
            linewidths=1.8,
            linestyles='--',
            alpha=0.85,
        )
        for idx in range(xs.shape[1]):
            ax.plot(
                xs[:, idx],
                ys[:, idx],
                **sedtrails_line_style(),
                label='sedtrails' if idx == 0 else None,
            )
        ax.set_xlabel('x [km]', fontdict=fontdict)
        ax.set_ylabel('y [km]', fontdict=fontdict)
        ax.set_ylim(0.0, 35.0)
        ax.set_title('Peninsula flow: SedTRAILS vs streamfunction', fontdict=fontdict)
        stream_handle = Line2D(
            [0],
            [0],
            color=analytic_line_style()['color'],
            linestyle='--',
            linewidth=1.6,
            label='streamfunction',
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([stream_handle, *handles], ['streamfunction', *labels], **legend_style())
        return fig

    def _plot_psi():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fontdict = plot_fontdict()
        for idx in range(xs.shape[1]):
            psi_hist = psi_fn(xs[:, idx], ys[:, idx])
            sed_style = sedtrails_line_style()
            sed_style.pop('marker', None)
            sed_style.pop('markersize', None)
            sed_style.pop('markevery', None)
            ax.plot(times, psi_hist - psi0[idx], **sed_style, label=f'particle {idx + 1}')
        ax.set_xlabel('time [s]', fontdict=fontdict)
        ax.set_ylabel('psi error [km^2/s]', fontdict=fontdict)
        ax.set_title('Peninsula flow: streamfunction error', fontdict=fontdict)
        return fig

    maybe_save_plot('04_peninsula_trajectories', _plot)
    maybe_save_plot('04_peninsula_psi_error', _plot_psi)

    assert np.max(psi_rel) < 1e-2
