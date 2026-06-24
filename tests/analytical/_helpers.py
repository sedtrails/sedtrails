from pathlib import Path

import numpy as np

from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


def rect_grid(xmin, xmax, nx, ymin, ymax, ny):
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys, indexing='xy')
    return xg.ravel(), yg.ravel()


def _output_dir() -> Path:
    output_dir = Path(__file__).resolve().parent / "output"
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


def liu_weisberg_skill(x_obs, y_obs, x_mod, y_mod, n_tolerance=1.0):
    """Compute Liu & Weisberg (2011) NCLS skill score for each particle.

    Inputs are arrays shaped (n_times, n_particles).
    Returns mean skill, std skill, and per-particle skill.
    """
    x_obs = np.asarray(x_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    x_mod = np.asarray(x_mod, dtype=np.float64)
    y_mod = np.asarray(y_mod, dtype=np.float64)

    if x_obs.shape != x_mod.shape or y_obs.shape != y_mod.shape:
        raise ValueError('Observed and modeled arrays must share the same shape')

    dxo = np.vstack([np.zeros((1, x_obs.shape[1])), np.diff(x_obs, axis=0)])
    dyo = np.vstack([np.zeros((1, y_obs.shape[1])), np.diff(y_obs, axis=0)])
    dlo = np.sqrt(dxo**2 + dyo**2)
    lo = np.cumsum(dlo, axis=0)

    dmo = np.sqrt((x_obs - x_mod) ** 2 + (y_obs - y_mod) ** 2)
    sncls = np.divide(dmo, lo, out=np.full_like(dmo, np.nan), where=lo > 0)

    ss = 1.0 - sncls / float(n_tolerance)
    ss[sncls > n_tolerance] = 0.0

    ss_mean = np.nanmean(ss, axis=0)
    return float(np.nanmean(ss_mean)), float(np.nanstd(ss_mean)), ss_mean


def integrate_time_dependent(
    calculator,
    x0,
    y0,
    velocity_fn,
    total_time,
    dt,
    use_temporal=False,
    return_history=False,
    stop_x=None,
    history_stride=1,
    stop_fn=None,
):
    x = np.asarray(x0, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64)

    nsteps = int(round(total_time / dt))
    t = 0.0
    simplex_ids = None

    if return_history:
        xs = [x.copy()]
        ys = [y.copy()]
        ts = [0.0]
        stride = max(1, int(history_stride))

    for step_idx in range(nsteps):
        if use_temporal:
            lower_u, lower_v = velocity_fn(t)
            upper_u, upper_v = velocity_fn(t + dt)
            x, y, simplex_ids = calculator['update_particles_temporal_with_simplex'](
                x,
                y,
                lower_u,
                lower_v,
                upper_u,
                upper_v,
                0.5,
                dt,
                simplex_ids,
                0,
            )
        else:
            u, v = velocity_fn(t)
            x, y, simplex_ids = calculator['update_particles_with_simplex'](
                x,
                y,
                u,
                v,
                dt,
                simplex_ids,
                0,
            )

        t += dt

        if return_history and ((step_idx + 1) % stride == 0):
            xs.append(x.copy())
            ys.append(y.copy())
            ts.append(t)

        if stop_x is not None and np.any(x >= stop_x):
            break

        if stop_fn is not None and stop_fn(step_idx, t, x, y):
            break

    if return_history:
        return x, y, np.asarray(ts), np.stack(xs), np.stack(ys)

    return x, y


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


def make_retriever(times, gx, gy, u_time, v_time):
    return FieldDataRetriever(make_sedtrails_data(times, gx, gy, u_time, v_time))
