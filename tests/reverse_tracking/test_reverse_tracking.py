from pathlib import Path

import numpy as np

from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.particle_tracer.particle_seeder import ParticlePopulation, PopulationConfig
from sedtrails.simulation_orchestrator.simulation_manager import Simulation
from sedtrails.simulation_orchestrator.runtime_plan import build_population_runtime_plans
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


def rect_grid(xmin, xmax, nx, ymin, ymax, ny):
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys, indexing='xy')
    return xg.ravel(), yg.ravel()


def _output_dir() -> Path:
    output_dir = Path(__file__).resolve().parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def maybe_save_artifact(name, array):
    np.save(_output_dir() / f'{name}.npy', np.asarray(array))


def write_metrics(name, metrics):
    lines = [f'{key}: {value}' for key, value in metrics.items()]
    (_output_dir() / f'{name}.txt').write_text('\n'.join(lines), encoding='utf-8')


def maybe_save_plot(name, plot_fn):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    original_rc = plt.rcParams.copy()
    try:
        plt.rcParams.update(
            {
                'font.family': 'Arial',
                'font.weight': 'bold',
                'font.style': 'italic',
            }
        )
        fig = plot_fn()
        fig.savefig(_output_dir() / f'{name}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    finally:
        plt.rcParams.update(original_rc)


def plot_fontdict():
    return {
        'family': 'Arial',
        'weight': 'bold',
        'style': 'italic',
    }


def sedtrails_line_style():
    return {
        'color': '#0b2d5c',
        'linewidth': 0.6,
        'marker': '.',
        'markersize': 4,
    }


def analytic_line_style():
    return {
        'color': '#b7d6f2',
        'linestyle': '--',
        'linewidth': 1.6,
    }


def legend_style():
    return {'prop': plot_fontdict()}


def make_sedtrails_data(times, gx, gy, u_time, v_time):
    return SedtrailsData(
        times=times,
        reference_date=np.datetime64('1970-01-01'),
        x=gx,
        y=gy,
        bed_level=np.zeros_like(gx),
        depth_avg_flow_velocity={
            'x': u_time,
            'y': v_time,
            'magnitude': np.sqrt(u_time**2 + v_time**2),
        },
        fractions=1,
        bed_load_transport={
            'x': np.zeros_like(u_time),
            'y': np.zeros_like(v_time),
            'magnitude': np.zeros_like(u_time),
        },
        suspended_transport={
            'x': np.zeros_like(u_time),
            'y': np.zeros_like(v_time),
            'magnitude': np.zeros_like(u_time),
        },
        water_depth=np.ones_like(u_time),
        mean_bed_shear_stress=np.zeros_like(u_time),
        max_bed_shear_stress=np.zeros_like(u_time),
        sediment_concentration=np.zeros_like(u_time),
        nonlinear_wave_velocity={
            'x': np.zeros_like(u_time),
            'y': np.zeros_like(v_time),
            'magnitude': np.zeros_like(u_time),
        },
        metadata=SedtrailsMetadata(
            flowfield_domain={
                'x_min': float(np.min(gx)),
                'x_max': float(np.max(gx)),
                'y_min': float(np.min(gy)),
                'y_max': float(np.max(gy)),
            }
        ),
    )


def _constant_velocity_field(times, grid_x, grid_y, u_value, v_value=0.0):
    u = np.full((times.size, grid_x.size), float(u_value), dtype=np.float64)
    v = np.full((times.size, grid_y.size), float(v_value), dtype=np.float64)
    return {'x': u, 'y': v, 'magnitude': np.hypot(u, v)}


def _static_velocity_data(gx, gy, u, v, total_time, dt):
    times = np.arange(0.0, total_time + dt, dt)
    u_time = np.repeat(np.asarray(u, dtype=np.float64)[None, :], times.size, axis=0)
    v_time = np.repeat(np.asarray(v, dtype=np.float64)[None, :], times.size, axis=0)
    return make_sedtrails_data(times, gx, gy, u_time, v_time)


def _make_passive_population(field_x, field_y, x0, y0, release_start=0.0):
    locations = [f'{float(x)},{float(y)}' for x, y in zip(x0, y0, strict=True)]
    population_config = {
        'name': 'passive_reverse_test',
        'particle_type': 'passive',
        'characteristics': {'diffusion_coefficient': 0.0},
        'diffusion': {'method': 'none', 'coefficient': 0.0},
        'transport_probability': 'no_probability',
        'tracer_methods': {'passive_tracer': {}},
        'seeding': {
            'quantity': 1,
            'release_start': release_start,
            'strategy': {'point': {'locations': locations}},
        },
    }
    population = ParticlePopulation(
        field_x=field_x,
        field_y=field_y,
        population_config=PopulationConfig(population_config),
    )
    runtime_plan = build_population_runtime_plans([population_config], [population], {})[0]
    assert runtime_plan.tracer.method_name == 'passive_tracer'
    assert runtime_plan.tracer.flow_field_names == ('depth_avg_flow_velocity',)
    return population, runtime_plan


def _run_passive_pipeline(data, initial_x, initial_y, total_time, dt, reverse=False, history_stride=1):
    retriever = FieldDataRetriever(data)
    release_start = total_time if reverse else 0.0
    population, runtime_plan = _make_passive_population(data.x, data.y, initial_x, initial_y, release_start)
    flow_field_name = runtime_plan.tracer.flow_field_names[0]
    xs = [population.particles['x'].copy()]
    ys = [population.particles['y'].copy()]
    ts = [total_time if reverse else 0.0]

    for step in range(int(total_time / dt)):
        elapsed = step * dt
        current_time = total_time - elapsed if reverse else elapsed
        field_time = current_time
        population.update_information(
            current_time=current_time,
            mixing_depth=1.0,
            transport_probability=1.0,
            bed_level=0.0,
        )
        population.update_status(time_direction=-1 if reverse else 1)
        flow = retriever.get_flow_field_bounds(field_time, flow_field_name)
        if reverse:
            flow = Simulation._reverse_flow_field(flow)
        population.update_position(flow_field=flow, current_timestep=dt)

        if (step + 1) % max(1, int(history_stride)) == 0:
            next_elapsed = (step + 1) * dt
            xs.append(population.particles['x'].copy())
            ys.append(population.particles['y'].copy())
            ts.append(total_time - next_elapsed if reverse else next_elapsed)

    return {
        'x': np.stack(xs),
        'y': np.stack(ys),
        'time': np.asarray(ts),
        'end_x': population.particles['x'].copy(),
        'end_y': population.particles['y'].copy(),
        'status_domain': population.particles['status_domain'].copy(),
        'status_mobile': population.particles['status_mobile'].copy(),
    }


def _write_closure_metrics(name, forward, reverse, start_x, start_y):
    forward_distance = np.hypot(forward['end_x'] - start_x, forward['end_y'] - start_y)
    closure_distance = np.hypot(reverse['end_x'] - start_x, reverse['end_y'] - start_y)
    maybe_save_artifact(f'{name}_forward_displacement', forward_distance)
    maybe_save_artifact(f'{name}_reverse_closure_distance', closure_distance)
    write_metrics(
        f'{name}_metrics',
        {
            'max_forward_displacement': float(np.max(forward_distance)),
            'mean_forward_displacement': float(np.mean(forward_distance)),
            'max_reverse_closure_distance': float(np.max(closure_distance)),
            'mean_reverse_closure_distance': float(np.mean(closure_distance)),
        },
    )
    return closure_distance


def _plot_forward_reverse_paths(name, forward, reverse, start_x, start_y, title, decorate_axes=None):
    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fontdict = plot_fontdict()
        if decorate_axes is not None:
            decorate_axes(ax)
        forward_style = sedtrails_line_style()
        reverse_style = analytic_line_style()
        reverse_style['color'] = '#4f8fc7'
        reverse_style['linestyle'] = ':'
        reverse_style['linewidth'] = 1.8
        for particle_index in range(start_x.size):
            ax.plot(
                forward['x'][:, particle_index],
                forward['y'][:, particle_index],
                **forward_style,
                label='passive forward' if particle_index == 0 else None,
            )
            ax.plot(
                reverse['x'][:, particle_index],
                reverse['y'][:, particle_index],
                **reverse_style,
                label='passive reverse' if particle_index == 0 else None,
            )
            ax.scatter(
                start_x[particle_index],
                start_y[particle_index],
                marker='o',
                s=18,
                color='black',
                zorder=3,
            )
            ax.scatter(
                forward['end_x'][particle_index],
                forward['end_y'][particle_index],
                marker='|',
                s=80,
                color='black',
                zorder=3,
            )
        ax.set_xlabel('x [m]', fontdict=fontdict)
        ax.set_ylabel('y [m]', fontdict=fontdict)
        ax.set_title(title, fontdict=fontdict)
        ax.legend(**legend_style())
        return fig

    maybe_save_plot(name, _plot)


def test_reverse_tracking_closes_passive_uniform_advection():
    """Reverse tracking should recover passive starts through the population pipeline."""
    total_time = 4.0 * 3600.0
    dt = 300.0
    times = np.arange(0.0, total_time + dt, dt)
    gx, gy = rect_grid(-5_000.0, 5_000.0, 3, -2_000.0, 2_000.0, 5)
    passive_speed = 0.20

    data = make_sedtrails_data(
        times,
        gx,
        gy,
        _constant_velocity_field(times, gx, gy, -passive_speed)['x'],
        _constant_velocity_field(times, gx, gy, -passive_speed)['y'],
    )

    start_x = np.array([3200.0, 3400.0, 3600.0], dtype=np.float64)
    start_y = np.array([-600.0, -450.0, -300.0], dtype=np.float64)

    forward = _run_passive_pipeline(data, start_x, start_y, total_time, dt, history_stride=6)
    expected_forward_x = start_x - passive_speed * total_time
    np.testing.assert_allclose(forward['end_x'], expected_forward_x, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(forward['end_y'], start_y, rtol=0.0, atol=1.0e-8)

    reverse = _run_passive_pipeline(data, forward['end_x'], forward['end_y'], total_time, dt, reverse=True, history_stride=6)
    np.testing.assert_allclose(reverse['end_x'], start_x, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(reverse['end_y'], start_y, rtol=0.0, atol=1.0e-8)

    passive_forward_error = forward['end_x'] - expected_forward_x
    passive_closure_error = reverse['end_x'] - start_x
    maybe_save_artifact('reverse_tracking_passive_forward_x_error', passive_forward_error)
    maybe_save_artifact('reverse_tracking_passive_closure_x_error', passive_closure_error)
    write_metrics(
        'reverse_tracking_uniform_passive_metrics',
        {
            'passive_forward_max_abs_x_error_m': float(np.max(np.abs(passive_forward_error))),
            'passive_reverse_closure_max_abs_x_error_m': float(np.max(np.abs(passive_closure_error))),
            'passive_displacement_m': float(passive_speed * total_time),
        },
    )

    _plot_forward_reverse_paths(
        'reverse_tracking_uniform_passive_comparison',
        forward,
        reverse,
        start_x,
        start_y,
        'Passive forward-to-reverse closure in uniform leftward flow',
    )


def test_reverse_tracking_closes_passive_peninsula_flow():
    """Reverse tracking should approximately close in the peninsula passive pipeline case."""
    gx, gy = rect_grid(0.0, 100.0, 100, 0.0, 50.0, 50)

    u0 = 0.001
    x_center = 50.0
    radius = 0.32 * 50.0

    def psi_fn(x, y):
        return u0 * radius**2 * y / ((x - x_center) ** 2 + y**2) - u0 * y

    denom = np.maximum(((gx - x_center) ** 2 + gy**2) ** 2, 1e-12)
    u = u0 - u0 * radius**2 * ((gx - x_center) ** 2 - gy**2) / denom
    v = -2.0 * u0 * radius**2 * ((gx - x_center) * gy) / denom
    land_mask = psi_fn(gx, gy) >= 0.0
    u = np.where(land_mask, 0.0, u)
    v = np.where(land_mask, 0.0, v)

    total_time = 12.0 * 3600.0
    dt = 300.0
    data = _static_velocity_data(gx, gy, u, v, total_time, dt)
    start_y = np.linspace(4.0, 26.0, 8)
    start_x = np.full(start_y.size, 3.0)

    forward = _run_passive_pipeline(data, start_x, start_y, total_time, dt, history_stride=6)
    assert np.all(forward['status_domain'])
    reverse = _run_passive_pipeline(data, forward['end_x'], forward['end_y'], total_time, dt, reverse=True, history_stride=6)
    assert np.all(reverse['status_domain'])

    closure_distance = _write_closure_metrics('reverse_tracking_peninsula_passive', forward, reverse, start_x, start_y)

    def _decorate(ax):
        x_vals = np.linspace(np.min(gx), np.max(gx), 800)
        y_vals = np.linspace(np.min(gy), np.max(gy), 400)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
        ax.contourf(xx, yy, psi_fn(xx, yy) >= 0.0, levels=[0.5, 1.5], colors=['#c8c8c8'])
        ax.set_ylim(0.0, 35.0)

    _plot_forward_reverse_paths(
        'reverse_tracking_peninsula_passive_comparison',
        forward,
        reverse,
        start_x,
        start_y,
        'Passive reverse closure in peninsula flow',
        decorate_axes=_decorate,
    )

    assert np.max(closure_distance) < 0.05


def test_reverse_tracking_closes_passive_stommel_flow():
    """Reverse tracking should approximately close in the Stommel passive pipeline case."""
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

    total_time = 10.0 * 86_400.0
    dt = 600.0
    data = _static_velocity_data(gx, gy, u_grid.ravel(), v_grid.ravel(), total_time, dt)
    start_x = np.linspace(150.0, 900.0, 4)
    start_y = np.full_like(start_x, 5_000.0)

    forward = _run_passive_pipeline(data, start_x, start_y, total_time, dt, history_stride=int(86_400.0 / dt))
    assert np.all(forward['status_domain'])
    reverse = _run_passive_pipeline(
        data,
        forward['end_x'],
        forward['end_y'],
        total_time,
        dt,
        reverse=True,
        history_stride=int(86_400.0 / dt),
    )
    assert np.all(reverse['status_domain'])

    closure_distance = _write_closure_metrics('reverse_tracking_stommel_passive', forward, reverse, start_x, start_y)

    def _decorate(ax):
        x_plot = np.linspace(np.min(gx), np.max(gx), 400)
        y_plot = np.linspace(np.min(gy), np.max(gy), 400)
        xx_plot, yy_plot = np.meshgrid(x_plot, y_plot, indexing='xy')
        ax.contour(
            xx_plot,
            yy_plot,
            psi_fn(xx_plot, yy_plot),
            levels=np.sort(psi_fn(start_x, start_y)),
            colors=analytic_line_style()['color'],
            linewidths=1.6,
            linestyles='--',
            alpha=0.6,
        )

    _plot_forward_reverse_paths(
        'reverse_tracking_stommel_passive_comparison',
        forward,
        reverse,
        start_x,
        start_y,
        'Passive reverse closure in Stommel gyre',
        decorate_axes=_decorate,
    )

    assert np.max(closure_distance) < 1.0
