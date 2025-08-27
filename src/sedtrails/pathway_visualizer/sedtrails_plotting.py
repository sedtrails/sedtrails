"""
sedtrails_plotting.py

Standalone plotting + selection utilities for SedTRAILS NetCDF outputs.

Design goals
------------
- Work directly with arrays (x, y, time, statuses), so callers can load
  using xarray or netCDF4 however they prefer.
- Produce the three core plots:
    (1) trajectories colored by particle age (time since release)
    (2) trajectories colored by distance from a rotated baseline
    (3) an animation of (x, y) at timestep t
- Export per-particle trajectory statistics
- Provide polygon-based selection helpers and an optional interactive
  polygon drawing tool (matplotlib.widgets.PolygonSelector).
- Keep dependencies light: numpy + matplotlib only. (matplotlib.path is used
  for point-in-polygon tests). Pandas is optional and only used if available
  for nicer CSV writing.

Expected SedTRAILS array layout
-------------------------------
Dimensions:
    n_timesteps = T
    n_particles = N
    n_populations = P
    n_flowfields = F

Variables (shapes shown in parentheses):
    time (N, T)                float64
    x, y, z (N, T)             float32
    status_alive (N, T)        int32  (0/1)
    status_domain (N, T)       int32  (0/1)
    status_released (N, T)     int32  (0/1)
    status_mobile (N, T)       int32  (0/1)
    trajectory_id (N,)         (optional char)
    population_id (N,)         int32  (optional)

Notes
-----
- All plotters accept `units_scale` (default=1.0). Use e.g. units_scale=1e-3
  to convert meters->kilometers on the axes without modifying inputs.
- Age is computed per-particle as time - time_of_release, where time_of_release
  is the first timestep where status_released==1. If status_released is missing,
  we fall back to the first finite (x, y) position.
- Baseline distance coloring uses the *initial* (at `first_stable_index`) rotated-x
  coordinate per-particle as a constant color for its whole trajectory. The rotated
  frame is defined by `rotation_deg` (counterclockwise) and the origin at the
  global min(x), min(y) of the domain unless overridden.

Copyright
---------
MIT (c) 2025 SedTRAILS contributors.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass as _dataclass


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.path import Path as MplPath

try:
    import pandas as _pd  # optional
except Exception:
    _pd = None

try:
    from matplotlib.widgets import PolygonSelector, Slider, Button
except Exception:
    PolygonSelector = None
    Slider = None
    Button = None


# -----------------------------------------------------------------------------
# Helper dataclass and utilities
# -----------------------------------------------------------------------------


@dataclass
class TrajectoryArrays:
    """Container for required arrays to plot & analyze SedTRAILS trajectories."""

    time: np.ndarray  # shape (N, T)
    x: np.ndarray  # shape (N, T)
    y: np.ndarray  # shape (N, T)

    # Optional status masks (0/1 or bool), all shape (N, T)
    status_alive: Optional[np.ndarray] = None
    status_domain: Optional[np.ndarray] = None
    status_released: Optional[np.ndarray] = None
    status_mobile: Optional[np.ndarray] = None

    # Optional meta
    population_id: Optional[np.ndarray] = None  # shape (N,)
    trajectory_id: Optional[Sequence[str]] = None

    def valid_mask(self) -> np.ndarray:
        """Points considered plottable by default.

        Returns
        -------
        mask : (N, T) bool
            True at points that are finite in x,y and have (if present):
            alive==1, in-domain==1, released==1.
        """
        mask = np.isfinite(self.x) & np.isfinite(self.y)
        if self.status_alive is not None:
            mask &= self.status_alive.astype(bool)
        if self.status_domain is not None:
            mask &= self.status_domain.astype(bool)
        if self.status_released is not None:
            mask &= self.status_released.astype(bool)
        return mask

    def release_time(self) -> np.ndarray:
        """Compute per-particle release time (shape (N,)).

        Prefers first index where status_released==1. If not provided,
        uses first finite (x, y) time. If a particle has no finite
        positions, falls back to time[:,0].
        """
        N, T = self.time.shape
        t0 = np.empty(N, dtype=float)
        t0[:] = np.nan

        if self.status_released is not None:
            rel = self.status_released.astype(bool)
            first_rel = np.argmax(rel, axis=1)  # zeros where all False
            no_rel = ~rel.any(axis=1)
            t0 = self.time[np.arange(N), first_rel]
            t0[no_rel] = np.nan
        # fallback to first finite x,y
        needs = np.isnan(t0)
        if np.any(needs):
            finite = np.isfinite(self.x) & np.isfinite(self.y)
            first_fin = np.argmax(finite, axis=1)
            no_fin = ~finite.any(axis=1)
            t0[needs] = self.time[np.arange(N), first_fin][needs]
            t0[no_fin] = self.time[no_fin, 0]
        return t0

    def age(self) -> np.ndarray:
        """Age matrix (N, T) = time - release_time, negatives masked to nan."""
        t0 = self.release_time()[:, None]  # (N,1)
        age = self.time - t0
        age[age < 0] = np.nan
        return age


# -----------------------------------------------------------------------------
# Plotting primitives
# -----------------------------------------------------------------------------


def _flatten_valid(
    x: np.ndarray, y: np.ndarray, c: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten arrays to 1D while removing nans and optionally a boolean mask."""
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    m = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    xf = x[m].ravel()
    yf = y[m].ravel()
    cf = c[m].ravel()
    return xf, yf, cf


def plot_trajectories_by_age(
    tr: TrajectoryArrays,
    units_scale: float = 1.0,
    point_size: float = 8.0,
    cmap: str = 'viridis',
    first_stable_index: int = 0,
    show_start: bool = True,
    show_end: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot all positions of all particles, colored by particle age."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    age = tr.age()
    mask = tr.valid_mask()
    mask[:, :first_stable_index] = False

    xf, yf, cf = _flatten_valid(tr.x * units_scale, tr.y * units_scale, age, mask)

    sc = ax.scatter(xf, yf, s=point_size, c=cf, cmap=cmap, edgecolor='none')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Time since release')

    if show_start:
        ax.scatter(
            tr.x[:, first_stable_index] * units_scale,
            tr.y[:, first_stable_index] * units_scale,
            s=point_size * 1.4,
            marker='o',
            facecolor='white',
            edgecolor='black',
            linewidths=0.6,
            zorder=3,
            label='start',
        )

    if show_end:
        last_idx = np.where(np.isfinite(tr.x), np.arange(tr.x.shape[1]), -1).max(axis=1)
        end_x = tr.x[np.arange(tr.x.shape[0]), last_idx] * units_scale
        end_y = tr.y[np.arange(tr.y.shape[0]), last_idx] * units_scale
        ax.scatter(
            end_x, end_y, s=point_size * 1.4, marker='x', edgecolor='black', linewidths=0.8, zorder=3, label='end'
        )

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Particle Trajectories — colored by age')
    ax.grid(True, alpha=0.3)
    return ax


def _rotate_points(
    x: np.ndarray, y: np.ndarray, rotation_deg: float, origin_xy: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate points by rotation_deg (CCW) around origin_xy."""
    theta = np.deg2rad(rotation_deg)
    ox, oy = origin_xy
    xr = x - ox
    yr = y - oy
    xr2 = xr * np.cos(theta) - yr * np.sin(theta)
    yr2 = xr * np.sin(theta) + yr * np.cos(theta)
    return xr2, yr2


def plot_trajectories_by_baseline(
    tr: TrajectoryArrays,
    rotation_deg: float = 0.0,
    first_stable_index: int = 0,
    origin_xy: Optional[Tuple[float, float]] = None,
    units_scale: float = 1.0,
    point_size: float = 8.0,
    cmap: str = 'viridis',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot all positions colored by per-particle distance from a rotated baseline.

    Mirrors the rotated-axis idea used in the MATLAB pathway plots. The color
    is computed from the *initial* rotated-X coordinate per particle and then
    broadcast to all timesteps. :contentReference[oaicite:2]{index=2}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    mask = tr.valid_mask()
    mask[:, :first_stable_index] = False

    x0 = tr.x[:, first_stable_index]
    y0 = tr.y[:, first_stable_index]

    if origin_xy is None:
        origin_xy = (np.nanmin(tr.x), np.nanmin(tr.y))

    rx0, ry0 = _rotate_points(x0, y0, rotation_deg, origin_xy)
    c_particle = rx0 - np.nanmin(rx0)
    c_full = np.broadcast_to(c_particle[:, None], tr.x.shape)

    xf, yf, cf = _flatten_valid(tr.x * units_scale, tr.y * units_scale, c_full, mask)
    sc = ax.scatter(xf, yf, s=point_size, c=cf, cmap=cmap, edgecolor='none')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Source distance from baseline')

    ax.scatter(
        tr.x[:, first_stable_index] * units_scale,
        tr.y[:, first_stable_index] * units_scale,
        s=point_size * 1.4,
        marker='o',
        facecolor='white',
        edgecolor='black',
        linewidths=0.6,
        zorder=3,
        label='start',
    )
    last_idx = np.where(np.isfinite(tr.x), np.arange(tr.x.shape[1]), -1).max(axis=1)
    end_x = tr.x[np.arange(tr.x.shape[0]), last_idx] * units_scale
    end_y = tr.y[np.arange(tr.y.shape[0]), last_idx] * units_scale
    ax.scatter(end_x, end_y, s=point_size * 1.4, marker='x', edgecolor='black', linewidths=0.8, zorder=3, label='end')

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Particle Trajectories — colored by baseline distance')
    ax.grid(True, alpha=0.3)
    return ax


def animate_particles(
    tr: TrajectoryArrays,
    rotation_deg: float = 0.0,
    first_stable_index: int = 0,
    origin_xy: Optional[Tuple[float, float]] = None,
    color_mode: str = 'baseline',  # "baseline" or "age"
    units_scale: float = 1.0,
    point_size: float = 12.0,
    interval_ms: int = 80,
    t_indices: Optional[Sequence[int]] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Animate particle positions through time (scatter by timestep)."""
    N, T = tr.x.shape
    if t_indices is None:
        t_indices = list(range(max(first_stable_index, 0), T))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    title = ax.set_title('Particle motion')

    mask_all = tr.valid_mask()

    # Precompute colors
    if color_mode == 'age':
        age = tr.age()
        color_arr = age
        cbar_label = 'Age'
    elif color_mode == 'baseline':
        x0 = tr.x[:, first_stable_index]
        y0 = tr.y[:, first_stable_index]
        if origin_xy is None:
            origin_xy = (np.nanmin(tr.x), np.nanmin(tr.y))
        rx0, _ = _rotate_points(x0, y0, rotation_deg, origin_xy)
        c_particle = rx0 - np.nanmin(rx0)
        color_arr = np.broadcast_to(c_particle[:, None], (N, T))
        cbar_label = 'Source distance from baseline'
    else:
        raise ValueError("color_mode must be 'baseline' or 'age'")

    t0 = t_indices[0]
    m0 = mask_all[:, t0]
    scat = ax.scatter(
        tr.x[m0, t0] * units_scale,
        tr.y[m0, t0] * units_scale,
        s=point_size,
        c=color_arr[m0, t0],
        cmap='viridis',
        edgecolor='none',
    )
    cb = plt.colorbar(scat, ax=ax)
    cb.set_label(cbar_label)

    def update(frame_idx: int):
        t = t_indices[frame_idx]
        m = mask_all[:, t]
        offsets = np.column_stack([tr.x[m, t] * units_scale, tr.y[m, t] * units_scale])
        scat.set_offsets(offsets)
        scat.set_array(color_arr[m, t])
        title.set_text(f'Particle motion — t={t}')
        return scat, title

    anim = FuncAnimation(fig, update, frames=len(t_indices), interval=interval_ms, blit=False)

    if save_path is not None:
        if save_path.lower().endswith('.gif'):
            try:
                anim.save(save_path, writer=PillowWriter(fps=max(1, int(1000 / interval_ms))), dpi=dpi)
            except Exception as e:
                warnings.warn(f'Failed to save GIF: {e}', stacklevel=1)
        elif save_path.lower().endswith('.mp4'):
            try:
                anim.save(save_path, writer='ffmpeg', dpi=dpi)
            except Exception as e:
                warnings.warn(f'Failed to save MP4 (ffmpeg missing?): {e}', stacklevel=1)

    return fig, anim


# -----------------------------------------------------------------------------
# Selection helpers (polygons)
# -----------------------------------------------------------------------------


def _points_in_poly(x: np.ndarray, y: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    """Return a boolean mask (T,) or (N,T) for points contained in polygon."""
    path = MplPath(poly_xy)
    pts = np.column_stack([x.ravel(), y.ravel()])
    inside = path.contains_points(pts)
    return inside.reshape(x.shape)


def particles_originating_in_polygon(
    tr: TrajectoryArrays, poly_xy: np.ndarray, first_stable_index: int = 0
) -> np.ndarray:
    """Mask (N,) for particles whose START point is inside polygon."""
    x0 = tr.x[:, first_stable_index]
    y0 = tr.y[:, first_stable_index]
    inside = _points_in_poly(x0[None, :], y0[None, :], poly_xy)[0]
    return inside


def particles_passing_through_polygon(
    tr: TrajectoryArrays, poly_xy: np.ndarray, first_stable_index: int = 0
) -> np.ndarray:
    """Mask (N,) for particles that enter polygon at any timestep."""
    mask_valid = tr.valid_mask()
    inside = _points_in_poly(tr.x, tr.y, poly_xy) & mask_valid
    return inside.any(axis=1)


def particles_between_two_polygons(
    tr: TrajectoryArrays,
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    order: Optional[str] = None,
    first_stable_index: int = 0,
) -> np.ndarray:
    """Particles that pass through both polygons (optionally in a given order)."""
    mask_valid = tr.valid_mask()
    in_a = _points_in_poly(tr.x, tr.y, poly_a) & mask_valid
    in_b = _points_in_poly(tr.x, tr.y, poly_b) & mask_valid

    hit_a = in_a.any(axis=1)
    hit_b = in_b.any(axis=1)
    both = hit_a & hit_b

    if order is None:
        return both

    idx_a = np.argmax(in_a, axis=1)
    idx_b = np.argmax(in_b, axis=1)
    idx_a[~hit_a] = np.iinfo(np.int32).max
    idx_b[~hit_b] = np.iinfo(np.int32).max

    if order == 'A->B':
        return both & (idx_a < idx_b)
    elif order == 'B->A':
        return both & (idx_b < idx_a)
    else:
        raise ValueError("order must be one of {'A->B','B->A', None}")


def particles_include_exclude(
    tr: TrajectoryArrays, include_polys: List[np.ndarray], exclude_polys: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """Particles that pass through *any* include polygon and *none* of the exclude polygons."""
    mask_valid = tr.valid_mask()
    include_any = np.zeros(tr.x.shape[0], dtype=bool)
    for poly in include_polys:
        include_any |= (_points_in_poly(tr.x, tr.y, poly) & mask_valid).any(axis=1)

    if exclude_polys:
        exclude_any = np.zeros(tr.x.shape[0], dtype=bool)
        for poly in exclude_polys:
            exclude_any |= (_points_in_poly(tr.x, tr.y, poly) & mask_valid).any(axis=1)
    else:
        exclude_any = np.zeros(tr.x.shape[0], dtype=bool)

    return include_any & (~exclude_any)


class InteractivePolygonTool:
    """Interactive polygon drawing to build selection masks in a matplotlib figure."""

    def __init__(self, ax: plt.Axes, on_done):
        if PolygonSelector is None:
            raise RuntimeError('matplotlib.widgets.PolygonSelector not available in this environment.')
        self.ax = ax
        self.on_done = on_done
        self.selector = PolygonSelector(ax, self._onselect, useblit=True)
        self._poly = None

    def _onselect(self, verts):
        self._poly = np.array(verts, dtype=float)

    def disconnect(self):
        if self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None

    @property
    def polygon(self) -> Optional[np.ndarray]:
        return self._poly


# -----------------------------------------------------------------------------
# Statistics (inspired by analyze_pathways.m)
# -----------------------------------------------------------------------------


@_dataclass
class ParticleStats:
    srcx: float
    srcy: float
    srct: float
    netDispX: float
    netDispY: float
    netDispMag: float
    duration: float
    uLRV: float
    vLRV: float
    grossDisp: float
    ratio: float


def compute_particle_stats(tr: TrajectoryArrays, first_stable_index: int = 0) -> List[ParticleStats]:
    """Compute per-particle trajectory statistics similar to analyze_pathways.m. :contentReference[oaicite:3]{index=3}"""
    N, T = tr.x.shape
    stats: List[ParticleStats] = []

    for i in range(N):
        x = tr.x[i, :]
        y = tr.y[i, :]
        t = tr.time[i, :]
        valid = np.isfinite(x) & np.isfinite(y)
        if tr.status_released is not None:
            valid &= tr.status_released[i, :].astype(bool)
        if tr.status_domain is not None:
            valid &= tr.status_domain[i, :].astype(bool)
        if tr.status_alive is not None:
            valid &= tr.status_alive[i, :].astype(bool)
        if valid.sum() < 2:
            stats.append(ParticleStats(*([np.nan] * 11)))
            continue

        idx0 = np.argmax(valid)
        idxN = np.where(valid)[0][-1]

        x0, y0, t0 = x[idx0], y[idx0], t[idx0]
        xN, yN, tN = x[idxN], y[idxN], t[idxN]

        net_dx = float(xN - x0)
        net_dy = float(yN - y0)
        net_mag = float(np.hypot(net_dx, net_dy))
        dur = float(tN - t0)

        if dur > 0:
            u_res = net_dx / dur
            v_res = net_dy / dur
        else:
            u_res = np.nan
            v_res = np.nan

        xv = x[valid]
        yv = y[valid]
        seg = np.hypot(np.diff(xv), np.diff(yv))
        gross = float(np.nansum(seg))
        ratio = gross / net_mag if (net_mag > 0 and np.isfinite(gross)) else np.nan

        stats.append(
            ParticleStats(
                srcx=float(x0),
                srcy=float(y0),
                srct=float(t0),
                netDispX=net_dx,
                netDispY=net_dy,
                netDispMag=net_mag,
                duration=dur,
                uLRV=float(u_res),
                vLRV=float(v_res),
                grossDisp=gross,
                ratio=float(ratio),
            )
        )

    return stats


def stats_to_csv(stats: List[ParticleStats], path: str) -> None:
    """Write particle stats to CSV. Pandas if available; csv otherwise."""
    header = [f for f in ParticleStats.__dataclass_fields__.keys()]
    if _pd is not None:
        _pd.DataFrame([{h: getattr(s, h) for h in header} for s in stats]).to_csv(path, index=False)
    else:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for s in stats:
                w.writerow([getattr(s, h) for h in header])


# -----------------------------------------------------------------------------
# Quick Explorer (slider + color-mode toggle)
# -----------------------------------------------------------------------------


def quick_explorer(
    tr: TrajectoryArrays,
    rotation_deg: float = 0.0,
    first_stable_index: int = 0,
    origin_xy: Optional[Tuple[float, float]] = None,
    units_scale: float = 1.0,
):
    """Quick interactive viewer with a time slider and color mode toggle."""
    if Slider is None or Button is None:
        warnings.warn('matplotlib widgets not available; quick_explorer requires Slider and Button.', stacklevel=1)
        return None

    N, T = tr.x.shape
    mask_all = tr.valid_mask()
    age = tr.age()

    x0 = tr.x[:, first_stable_index]
    y0 = tr.y[:, first_stable_index]
    if origin_xy is None:
        origin_xy = (np.nanmin(tr.x), np.nanmin(tr.y))
    rx0, _ = _rotate_points(x0, y0, rotation_deg, origin_xy)
    c_particle = rx0 - np.nanmin(rx0)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    title = ax.set_title('Quick Explorer — baseline colors')
    t0 = max(first_stable_index, 0)

    m0 = mask_all[:, t0]
    scat = ax.scatter(
        tr.x[m0, t0] * units_scale,
        tr.y[m0, t0] * units_scale,
        s=12.0,
        c=c_particle[m0],
        cmap='viridis',
        edgecolor='none',
    )
    cb = plt.colorbar(scat, ax=ax)
    cb.set_label('Source distance from baseline')

    ax_sl = plt.axes([0.15, 0.1, 0.7, 0.03])
    s_time = Slider(ax_sl, 't', valmin=0, valmax=T - 1, valinit=t0, valstep=1)

    ax_b1 = plt.axes([0.15, 0.05, 0.15, 0.04])
    ax_b2 = plt.axes([0.32, 0.05, 0.15, 0.04])
    b_baseline = Button(ax_b1, 'Baseline')
    b_age = Button(ax_b2, 'Age')

    state = {'mode': 'baseline'}

    def update_time(val):
        t = int(s_time.val)
        m = mask_all[:, t]
        offsets = np.column_stack([tr.x[m, t] * units_scale, tr.y[m, t] * units_scale])
        scat.set_offsets(offsets)
        if state['mode'] == 'baseline':
            scat.set_array(c_particle[m])
            cb.set_label('Source distance from baseline')
            title.set_text(f'Quick Explorer — baseline colors (t={t})')
        else:
            scat.set_array(age[m, t])
            cb.set_label('Age')
            title.set_text(f'Quick Explorer — age colors (t={t})')
        fig.canvas.draw_idle()

    def set_baseline(event):
        state['mode'] = 'baseline'
        update_time(None)

    def set_age(event):
        state['mode'] = 'age'
        update_time(None)

    s_time.on_changed(update_time)
    b_baseline.on_clicked(set_baseline)
    b_age.on_clicked(set_age)
    return fig


# -----------------------------------------------------------------------------
# 2D density heatmap (bonus)
# -----------------------------------------------------------------------------


def plot_density_heatmap(
    tr: TrajectoryArrays,
    bins: int = 200,
    units_scale: float = 1.0,
    first_stable_index: int = 0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a 2D histogram of trajectory visitation density (spatial heatmap)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    mask = tr.valid_mask()
    mask[:, :first_stable_index] = False
    x = tr.x * units_scale
    y = tr.y * units_scale
    xf = x[mask]
    yf = y[mask]

    H, xedges, yedges = np.histogram2d(xf, yf, bins=bins)
    H = H.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(H, origin='lower', extent=extent, aspect='equal')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Counts per bin')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectory visitation heatmap')
    return ax


# -----------------------------------------------------------------------------
# Minimal loader (optional), in case users want a helper to build TrajectoryArrays
# -----------------------------------------------------------------------------


def load_from_xarray(ds) -> TrajectoryArrays:
    """Build TrajectoryArrays from an xarray Dataset with SedTRAILS variables."""
    tr = TrajectoryArrays(
        time=np.asarray(ds['time']),
        x=np.asarray(ds['x']),
        y=np.asarray(ds['y']),
        status_alive=np.asarray(ds['status_alive']) if 'status_alive' in ds else None,
        status_domain=np.asarray(ds['status_domain']) if 'status_domain' in ds else None,
        status_released=np.asarray(ds['status_released']) if 'status_released' in ds else None,
        status_mobile=np.asarray(ds['status_mobile']) if 'status_mobile' in ds else None,
        population_id=np.asarray(ds['population_id']) if 'population_id' in ds else None,
        trajectory_id=list(map(str, ds['trajectory_id'].values)) if 'trajectory_id' in ds else None,
    )
    return tr
