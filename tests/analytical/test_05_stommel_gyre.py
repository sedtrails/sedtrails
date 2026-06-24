import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator

from ._helpers import (
    analytic_line_style,
    integrate_time_dependent,
    legend_style,
    maybe_save_artifact,
    maybe_save_plot,
    plot_fontdict,
    sedtrails_line_style,
    write_metrics,
)


def test_stommel_gyre_streamfunction_conservation():
    """Validate streamfunction conservation for a Stommel gyre."""
    a = b = 10_000.0
    eps = 0.05
    amplitude = 100.0

    x_vals = np.linspace(0.0, a, 200)
    y_vals = np.linspace(0.0, b, 200)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
    gx = xx.ravel()
    gy = yy.ravel()

    l1 = (-1.0 + np.sqrt(1.0 + 4.0 * np.pi**2 * eps**2)) / (2.0 * eps)
    l2 = (-1.0 - np.sqrt(1.0 + 4.0 * np.pi**2 * eps**2)) / (2.0 * eps)
    c1 = (1.0 - np.exp(l2)) / (np.exp(l2) - np.exp(l1))
    c2 = -(1.0 + c1)

    # Streamfunction for the analytic Stommel gyre solution.
    def psi_fn(x, y):
        xi = x / a
        yi = y / b
        return amplitude * (c1 * np.exp(l1 * xi) + c2 * np.exp(l2 * xi) + 1.0) * np.sin(np.pi * yi)

    psi_grid = psi_fn(xx, yy)
    dx = 2.0 * a / x_vals.size
    dy = 2.0 * b / y_vals.size
    u_grid = np.zeros_like(psi_grid)
    v_grid = np.zeros_like(psi_grid)
    v_grid[:, 1:-1] = (psi_grid[:, 2:] - psi_grid[:, :-2]) / dx
    u_grid[1:-1, :] = -(psi_grid[2:, :] - psi_grid[:-2, :]) / dy
    u_grid_flat = u_grid.ravel()
    v_grid_flat = v_grid.ravel()

    def velocity_fn(_t):
        return u_grid_flat, v_grid_flat

    calculator = create_numba_particle_calculator(gx, gy)

    x0 = np.linspace(100.0, 1000.0, 4)
    y0 = np.full_like(x0, 5_000.0)
    psi0 = psi_fn(x0, y0)

    total_time = 50.0 * 86_400.0
    dt = 300.0
    history_stride = int(86_400.0 / dt)

    # Integrate trajectories over a long window to check drift.
    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=total_time,
        dt=dt,
        return_history=True,
        history_stride=history_stride,
    )
    psi_end = psi_fn(x_end, y_end)

    # Relative streamfunction error provides a scale-aware tolerance.
    psi_rel = np.abs(psi_end - psi0) / np.maximum(np.abs(psi0), 1.0)
    maybe_save_artifact('05_stommel_streamfunction_relative_error', psi_rel)

    write_metrics(
        '05_stommel_metrics',
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
        x_vals = np.linspace(np.min(gx), np.max(gx), 400)
        y_vals = np.linspace(np.min(gy), np.max(gy), 400)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
        psi_grid = psi_fn(xx, yy)
        ax.contour(
            xx,
            yy,
            psi_grid,
            levels=np.sort(psi0),
            colors=analytic_line_style()['color'],
            linewidths=1.6,
            linestyles='--',
            alpha=0.6,
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
        ax.set_title('Stommel gyre: SedTRAILS vs streamfunction', fontdict=fontdict)
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
        ax.set_title('Stommel gyre: streamfunction error', fontdict=fontdict)
        return fig

    maybe_save_plot('05_stommel_trajectories', _plot)
    maybe_save_plot('05_stommel_psi_error', _plot_psi)

    assert np.max(psi_rel) < 1e-2
