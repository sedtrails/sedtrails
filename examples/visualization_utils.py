"""Small plotting helpers used by the SedTRAILS example scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _as_1d(values) -> np.ndarray:
    """Return values as a flattened float array."""
    return np.asarray(values, dtype=float).ravel()


def _save_if_requested(fig, save_path) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')


def plot_flow_field(
    flow_data,
    title=None,
    downsample=5,
    figsize=(12, 10),
    cmap='viridis',
    vector_color='white',
    save_path=None,
):
    """Plot a SedTRAILS flow field with velocity magnitude and vectors.

    Parameters
    ----------
    flow_data : dict
        Dictionary containing ``x``, ``y``, ``u``, ``v``, and ``magnitude`` arrays.
    title : str, optional
        Plot title.
    downsample : int, optional
        Keep every Nth vector in the quiver overlay.
    figsize : tuple, optional
        Matplotlib figure size.
    cmap : str, optional
        Colormap for the magnitude field.
    vector_color : str, optional
        Quiver vector color.
    save_path : str or pathlib.Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    tuple
        ``(fig, ax)`` Matplotlib figure and axis.
    """
    x = _as_1d(flow_data['x'])
    y = _as_1d(flow_data['y'])
    u = _as_1d(flow_data['u'])
    v = _as_1d(flow_data['v'])
    magnitude = _as_1d(flow_data['magnitude'])

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(magnitude)
    if finite.sum() < 3:
        raise ValueError('At least three finite flow-field points are required for plotting.')

    x = x[finite]
    y = y[finite]
    u = u[finite]
    v = v[finite]
    magnitude = magnitude[finite]

    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.tricontourf(x, y, magnitude, cmap=cmap, levels=20)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Flow magnitude [m/s]')

    stride = max(1, int(downsample))
    q = ax.quiver(
        x[::stride],
        y[::stride],
        u[::stride],
        v[::stride],
        color=vector_color,
        scale=15,
        width=0.002,
    )
    ax.quiverkey(q, 0.85, 0.02, 0.5, '0.5 m/s', labelpos='E', coordinates='figure', color=vector_color)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(title or 'Flow field')
    ax.set_aspect('equal', adjustable='box')

    _save_if_requested(fig, save_path)
    return fig, ax


def plot_particle_trajectory(
    flow_data,
    trajectory_x,
    trajectory_y,
    title=None,
    downsample=5,
    figsize=(12, 10),
    cmap='viridis',
    vector_color='white',
    trajectory_color='red',
    save_path=None,
):
    """Plot a SedTRAILS flow field with one particle trajectory overlay."""
    fig, ax = plot_flow_field(
        flow_data,
        title=title or 'Flow field with particle trajectory',
        downsample=downsample,
        figsize=figsize,
        cmap=cmap,
        vector_color=vector_color,
    )

    trajectory_x = _as_1d(trajectory_x)
    trajectory_y = _as_1d(trajectory_y)
    valid = np.isfinite(trajectory_x) & np.isfinite(trajectory_y)
    if valid.sum() == 0:
        raise ValueError('Trajectory contains no finite positions.')

    trajectory_x = trajectory_x[valid]
    trajectory_y = trajectory_y[valid]
    ax.plot(trajectory_x, trajectory_y, '-', color=trajectory_color, linewidth=2, label='Particle trajectory')
    ax.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=8, label='Start')
    ax.plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=8, label='End')
    ax.legend(loc='upper right')

    _save_if_requested(fig, save_path)
    return fig, ax
