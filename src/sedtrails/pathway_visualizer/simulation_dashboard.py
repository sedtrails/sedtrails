"""
Simulation Dashboard for real-time monitoring and testing simulations.

This module provides interactive dashboard capabilities for monitoring particle
simulations with spatial and temporal visualizations.
"""

import datetime
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, QhullError, cKDTree


@dataclass
class RasterInterpolationWeights:
    """Cached mapping from dashboard raster pixels to source grid faces."""

    geometry_key: tuple
    shape: tuple[int, int]
    extent: tuple[float, float, float, float]
    face_indices: np.ndarray
    weights: np.ndarray
    valid_mask: np.ndarray
    quiver_indices: np.ndarray


class SimulationDashboard:
    """Real-time visualization dashboard for particle simulations."""

    LARGE_GRID_POINT_LIMIT = 5_000
    LARGE_GRID_QUIVER_LIMIT = 100
    PARTICLE_RENDER_LIMIT = 5_000
    RASTER_MAX_SIDE = 700
    RASTER_K_NEIGHBORS = 4
    STRANDED_PARTICLE_COLOR = '#ffb3b3'

    def __init__(self, reference_date: str = '1970-01-01'):
        """Initialize the dashboard."""
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.data_store = defaultdict(list)
        self.trajectories = {'x': [], 'y': [], 'time': []}  # Store sampled initial positions for display
        self.time_stamps = []
        self.plot_initialized = False
        self.last_update_time = 0
        self.keep_open = True  # Control for keeping window open after simulation
        self._raster_weights: RasterInterpolationWeights | None = None
        self._raster_images = {}
        self._spatial_quivers = {}
        self._particle_artists = []
        self._particle_sample_indices = None
        self._particle_sample_count = None
        self._previous_particle_positions = None

        # Store reference date for time conversions
        self.reference_date = datetime.datetime.fromisoformat(reference_date)

        # Load SE colormap for bathymetry (relative to this file)
        colormap_path = Path(__file__).parent / 'SE_cmap.npy'
        colors = np.load(colormap_path)
        self.bathymetry_cmap = ListedColormap(colors)
        self.bathymetry_vmin = -12
        self.bathymetry_vmax = 6

    def initialize_dashboard(self, figsize: Tuple[float, float] = (16, 10)) -> None:
        """
        Initialize the dashboard with subplot layout.

        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size in inches.
        """
        self.fig = plt.figure(figsize=figsize)

        # Use mosaic layout: M1=flowfield, M2=bathymetry, T1-T4=timeseries, P=progress
        mosaic = [['M1', 'T1'], ['M1', 'T2'], ['M2', 'T3'], ['M2', 'T4'], ['P', 'P']]

        self.axes = self.fig.subplot_mosaic(mosaic, height_ratios=[1, 1, 1, 1, 0.15], width_ratios=[1, 3])

        # Map axes to meaningful names
        self.axes['flowfield'] = self.axes['M1']
        self.axes['bathymetry'] = self.axes['M2']
        self.axes['longshore_vel'] = self.axes['T1']
        self.axes['crossshore_vel'] = self.axes['T2']
        self.axes['distance'] = self.axes['T3']
        self.axes['burial_depth'] = self.axes['T4']
        self.axes['progress'] = self.axes['P']

        plt.ion()
        self._setup_time_series_plots()
        self._setup_progress_bar()
        self._set_titles()

        # Fix overlapping text
        plt.tight_layout()

        # Handle window closing event
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        self.plot_initialized = True

        # Force display and bring to front
        self._show_and_raise_window()

    def should_update(self, current_time: float, plot_interval: float) -> bool:
        """
        Return whether the dashboard should redraw for the given simulation time.

        Parameters
        ----------
        current_time : float
            Current simulation time in seconds.
        plot_interval : float
            Interval between plot updates.

        Returns
        -------
        bool
            Boolean result of the check.
        """
        return current_time - self.last_update_time >= plot_interval

    def _show_and_raise_window(self):
        """Show window and bring it to front (cross-platform)."""
        try:
            # Show the figure
            self.fig.show()

            # Draw and flush to ensure display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # Try to position and raise window
            backend = plt.get_backend().lower()

            if 'qt' in backend:
                # Qt backend (most common)
                try:
                    manager = self.fig.canvas.manager
                    manager.window.show()
                    manager.window.raise_()
                    manager.window.activateWindow()

                    # On macOS, also try to bring to front
                    import sys

                    if sys.platform == 'darwin':
                        manager.window.raise_()
                        # Additional macOS-specific activation
                        try:
                            import subprocess

                            subprocess.run(
                                ['osascript', '-e', 'tell application "Python" to activate'],
                                check=False,
                                capture_output=True,
                            )
                        except Exception:
                            pass

                except Exception:
                    pass

            elif 'tk' in backend:
                # Tkinter backend
                try:
                    manager = self.fig.canvas.manager
                    manager.window.lift()
                    manager.window.attributes('-topmost', True)
                    manager.window.after(100, lambda: manager.window.attributes('-topmost', False))
                except Exception:
                    pass

            # Force a small pause to let the window system catch up
            plt.pause(0.1)

        except Exception as e:
            print(f'Warning: Could not properly display dashboard window: {e}')

    def _on_close(self, event):
        """Handle window close event."""
        self.keep_open = False
        plt.ioff()  # Turn off interactive mode

    def keep_window_open(self):
        """Keep the window open after simulation ends."""
        if self.fig is not None and self.keep_open:
            print('Simulation complete. Dashboard window will remain open.')
            print('Close the window manually when done viewing results.')

            # Turn off interactive mode but keep window open
            plt.ioff()

            # Show blocking to keep window open
            try:
                # This will keep the window open until manually closed
                self.fig.show()

                # For different backends, we might need different approaches
                backend = plt.get_backend().lower()
                if 'qt' in backend:
                    # Qt backend - keep event loop running
                    try:
                        from matplotlib.backends.qt_compat import QtWidgets

                        app = QtWidgets.QApplication.instance()
                        if app is not None:
                            # Don't call app.exec_() as it would block everything
                            # Instead, just ensure the window stays visible
                            pass
                    except Exception:
                        pass

                # Keep matplotlib event loop alive
                while self.keep_open and plt.fignum_exists(self.fig.number):
                    plt.pause(0.1)

            except Exception as e:
                print(f'Could not keep window open: {e}')

    def _setup_time_series_plots(self) -> None:
        """Set up empty line plots for time series data."""
        # Set up date formatter for all time series plots
        # Use a compact format to fit better in the available space
        date_fmt = mdates.DateFormatter('%m-%d %H:%M')

        # Longshore velocity
        (self.lines['longshore_avg'],) = self.axes['longshore_vel'].plot([], [], 'b-', label='Average', linewidth=2)
        (self.lines['longshore_max'],) = self.axes['longshore_vel'].plot([], [], 'r-', label='Maximum', linewidth=2)
        self.axes['longshore_vel'].legend()
        self.axes['longshore_vel'].set_ylabel('Velocity (m/s)')
        self.axes['longshore_vel'].xaxis.set_major_formatter(date_fmt)
        self.axes['longshore_vel'].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        self.axes['longshore_vel'].grid(True, alpha=0.3)

        # Cross-shore velocity
        (self.lines['crossshore_avg'],) = self.axes['crossshore_vel'].plot([], [], 'b-', label='Average', linewidth=2)
        (self.lines['crossshore_max'],) = self.axes['crossshore_vel'].plot([], [], 'r-', label='Maximum', linewidth=2)
        self.axes['crossshore_vel'].legend()
        self.axes['crossshore_vel'].set_ylabel('Velocity (m/s)')
        self.axes['crossshore_vel'].xaxis.set_major_formatter(date_fmt)
        self.axes['crossshore_vel'].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        self.axes['crossshore_vel'].grid(True, alpha=0.3)

        # Distance
        (self.lines['distance'],) = self.axes['distance'].plot([], [], 'g-', linewidth=2)
        self.axes['distance'].set_ylabel('Distance (m)')
        self.axes['distance'].xaxis.set_major_formatter(date_fmt)
        self.axes['distance'].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        self.axes['distance'].grid(True, alpha=0.3)

        # Burial depth
        (self.lines['burial_avg'],) = self.axes['burial_depth'].plot([], [], 'darkorange', label='Average', linewidth=2)
        (self.lines['burial_max'],) = self.axes['burial_depth'].plot([], [], 'orangered', label='Maximum', linewidth=2)
        (self.lines['mixing_depth'],) = self.axes['burial_depth'].plot(
            [], [], 'gray', linestyle='--', label='Mixing Depth', linewidth=1
        )
        self.axes['burial_depth'].legend()
        self.axes['burial_depth'].set_ylabel('Burial Depth (m)')
        self.axes['burial_depth'].set_xlabel('Date and Time')
        self.axes['burial_depth'].xaxis.set_major_formatter(date_fmt)
        self.axes['burial_depth'].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        self.axes['burial_depth'].grid(True, alpha=0.3)

    def _setup_progress_bar(self) -> None:
        """Set up progress bar."""
        self.axes['progress'].set_xlim(0, 100)
        self.axes['progress'].set_ylim(-0.5, 0.5)
        self.axes['progress'].set_xlabel('')
        self.axes['progress'].set_yticks([])
        self.axes['progress'].margins(0, 0)  # Remove margins

        # Progress bar background
        self.progress_bg = self.axes['progress'].barh(0, 100, height=0.6, color='lightgray', alpha=0.3)
        # Progress bar fill
        self.progress_fill = self.axes['progress'].barh(0, 0, height=0.6, color='green', alpha=0.7)

        # Text labels for start, current, end times
        self.progress_text_start = self.axes['progress'].text(0, -0.2, '', ha='left', va='top', fontsize=8)
        self.progress_text_current = self.axes['progress'].text(
            50, 0.2, '', ha='center', va='bottom', fontsize=8, fontweight='bold'
        )
        self.progress_text_end = self.axes['progress'].text(100, -0.2, '', ha='right', va='top', fontsize=8)

    def _set_titles(self) -> None:
        """Set plot titles."""
        self.axes['flowfield'].set_title('(a) Flow Field (Latest)', fontsize=12, fontweight='bold')
        self.axes['bathymetry'].set_title('(b) Bathymetry + Particles', fontsize=12, fontweight='bold')
        self.axes['longshore_vel'].set_title('(c) Longshore Velocity', fontsize=12, fontweight='bold')
        self.axes['crossshore_vel'].set_title('(d) Cross-shore Velocity', fontsize=12, fontweight='bold')
        self.axes['distance'].set_title('(e) Average Distance per Output Timestep', fontsize=12, fontweight='bold')
        self.axes['burial_depth'].set_title('(f) Burial Depth', fontsize=12, fontweight='bold')

    def update(
        self,
        flow_field: Dict[str, np.ndarray],
        bathymetry: np.ndarray,
        particles: Dict[str, np.ndarray],
        current_time: float,
        timestep: float,
        plot_interval: float,
        simulation_start_time: float = 0,
        simulation_end_time: float | None = None,
        mesh_geometry: Dict[str, Any] | None = None,
    ) -> None:
        """
        Update dashboard with current simulation data.

        Parameters
        ----------
        flow_field : Dict[str, np.ndarray]
            Flow-field data to visualize.
        bathymetry : np.ndarray
            Bathymetry data used for plotting.
        particles : Dict[str, np.ndarray]
            Particle data used for plotting or output.
        current_time : float
            Current simulation time in seconds.
        timestep : float
            Current simulation timestep.
        plot_interval : float
            Interval between plot updates.
        simulation_start_time : float
            Simulation start time in seconds.
        simulation_end_time : float | None
            Simulation end time in seconds.
        mesh_geometry : Dict[str, Any] | None
            Optional mesh geometry used for plotting.
        """

        if not self.should_update(current_time, plot_interval):
            return

        dashboard_particles = self._sample_particle_payload(particles)
        self._ensure_initial_particle_snapshot(dashboard_particles, current_time)

        # Store data for time series analysis
        self._store_particle_data(dashboard_particles, current_time, timestep, flow_field)

        # Prepare particle data with initial positions from trajectories
        particle_data_with_initial = dashboard_particles.copy()
        if len(self.trajectories['x']) > 0:
            particle_data_with_initial['x_initial'] = self.trajectories['x'][0]
            particle_data_with_initial['y_initial'] = self.trajectories['y'][0]

        # Update all plots
        self._update_flowfield_plot(flow_field, bathymetry, mesh_geometry=mesh_geometry)
        self._update_bathymetry_plot(flow_field, bathymetry, particle_data_with_initial, mesh_geometry=mesh_geometry)
        self._update_time_series_plots()
        self._update_progress_bar(current_time, simulation_start_time, simulation_end_time)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.last_update_time = current_time

    def _reset_particle_history(self) -> None:
        """Reset sampled particle history when the particle population changes."""
        self.trajectories = {'x': [], 'y': [], 'time': []}
        self._previous_particle_positions = None

    def _particle_plot_indices(self, n_particles: int) -> slice | np.ndarray:
        """Return stable deterministic particle indices for dashboard plotting."""
        cached_count = getattr(self, '_particle_sample_count', None)
        cached_indices = getattr(self, '_particle_sample_indices', None)
        if cached_indices is not None and cached_count == n_particles:
            return cached_indices

        indices = self._sample_indices(n_particles, self.PARTICLE_RENDER_LIMIT)
        self._particle_sample_indices = indices
        self._particle_sample_count = n_particles
        self._reset_particle_history()
        return indices

    def _sample_particle_payload(self, particles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Return a bounded particle payload for dashboard-only rendering and summaries."""
        n_particles = len(particles['x'])
        indices = self._particle_plot_indices(n_particles)
        return {name: np.asarray(values)[indices] for name, values in particles.items()}

    def _ensure_initial_particle_snapshot(self, particles: Dict[str, np.ndarray], current_time: float) -> None:
        """Store only the first sampled particle positions used for displacement lines."""
        if len(self.trajectories['x']) > 0:
            return

        self.trajectories['x'].append(np.asarray(particles['x']).copy())
        self.trajectories['y'].append(np.asarray(particles['y']).copy())
        self.trajectories['time'].append(current_time)

    def _store_particle_data(
        self, particles: Dict[str, np.ndarray], current_time: float, timestep: float, flow_field: Dict[str, np.ndarray]
    ) -> None:
        """Store particle data for time series analysis."""
        # Convert time to datetime object
        current_datetime = self.reference_date + datetime.timedelta(seconds=float(current_time))
        self.time_stamps.append(current_datetime)

        # Interpolate flow velocities at particle positions
        particle_u = np.interp(particles['x'], flow_field['x'], flow_field['u'])
        particle_v = np.interp(particles['y'], flow_field['y'], flow_field['v'])

        # Store velocity statistics
        self.data_store['longshore_avg'].append(np.mean(np.abs(particle_u)))
        self.data_store['longshore_max'].append(np.max(np.abs(particle_u)))
        self.data_store['crossshore_avg'].append(np.mean(np.abs(particle_v)))
        self.data_store['crossshore_max'].append(np.max(np.abs(particle_v)))

        # Calculate average distance covered per output timestep
        previous_positions = getattr(self, '_previous_particle_positions', None)
        if previous_positions is not None:
            prev_x, prev_y = previous_positions
            distances = np.sqrt((particles['x'] - prev_x) ** 2 + (particles['y'] - prev_y) ** 2)
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0.0

        self.data_store['distance'].append(avg_distance)
        self._previous_particle_positions = (np.asarray(particles['x']).copy(), np.asarray(particles['y']).copy())

        # Store burial depth statistics (if available)
        if 'burial_depth' in particles:
            self.data_store['burial_avg'].append(np.mean(particles['burial_depth']))
            self.data_store['burial_max'].append(np.max(particles['burial_depth']))
        else:
            self.data_store['burial_avg'].append(0.0)
            self.data_store['burial_max'].append(0.0)

        # Store mixing depth (if available)
        if 'mixing_depth' in particles:
            self.data_store['mixing_depth'].append(np.mean(particles['mixing_depth']))
        else:
            self.data_store['mixing_depth'].append(0.0)

    @classmethod
    def _is_large_grid(cls, n_points: int) -> bool:
        """Return whether the spatial plot should use the sampled large-grid path."""
        return n_points > cls.LARGE_GRID_POINT_LIMIT

    @classmethod
    def _sample_indices(cls, n_points: int, limit: int) -> slice | np.ndarray:
        """Return deterministic point indices for large-grid plotting."""
        if n_points <= limit:
            return slice(None)
        return np.linspace(0, n_points - 1, limit, dtype=np.intp)

    @staticmethod
    def _flatten(values: np.ndarray) -> np.ndarray:
        """Flatten field data for plotting without copying when possible."""
        return np.asarray(values).ravel()

    @staticmethod
    def _particle_status_mask(
        particles: Dict[str, np.ndarray], status_name: str, n_particles: int, default: bool = False
    ) -> np.ndarray:
        """Return a boolean particle status mask with a safe fallback for older payloads."""
        status = particles.get(status_name)
        if status is None:
            return np.full(n_particles, default, dtype=bool)

        status = np.asarray(status, dtype=bool)
        if status.shape != (n_particles,):
            return np.full(n_particles, default, dtype=bool)
        return status

    def _particle_trajectory_history(self, n_particles: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return stored particle trajectory arrays when all snapshots match the current particle count."""
        trajectories = getattr(self, 'trajectories', {})
        x_history = trajectories.get('x', [])
        y_history = trajectories.get('y', [])
        if len(x_history) < 2 or len(x_history) != len(y_history):
            return None

        x_arrays = [np.asarray(values, dtype=float).ravel() for values in x_history]
        y_arrays = [np.asarray(values, dtype=float).ravel() for values in y_history]
        if any(values.shape != (n_particles,) for values in x_arrays + y_arrays):
            return None

        return np.vstack(x_arrays), np.vstack(y_arrays)

    @staticmethod
    def _particle_trail_segments(trail_x: np.ndarray, trail_y: np.ndarray, particle_indices: np.ndarray) -> np.ndarray:
        """Return line-collection segments for selected particle trajectory columns."""
        if particle_indices.size == 0:
            return np.empty((0, 0, 2), dtype=float)
        return np.stack((trail_x[:, particle_indices], trail_y[:, particle_indices]), axis=-1).transpose(1, 0, 2)

    def _geometry_key(self, x: np.ndarray, y: np.ndarray, mesh_geometry: Dict[str, Any] | None) -> tuple:
        extent = self._spatial_extent(x, y, mesh_geometry)
        if mesh_geometry is None:
            return (x.size, extent, None, None, self.RASTER_MAX_SIDE, self.RASTER_K_NEIGHBORS)

        node_x = mesh_geometry.get('node_x')
        face_node_connectivity = mesh_geometry.get('face_node_connectivity')
        node_count = 0 if node_x is None else np.asarray(node_x).size
        connectivity_shape = None if face_node_connectivity is None else np.asarray(face_node_connectivity).shape
        return (x.size, extent, node_count, connectivity_shape, self.RASTER_MAX_SIDE, self.RASTER_K_NEIGHBORS)

    def _spatial_extent(
        self, x: np.ndarray, y: np.ndarray, mesh_geometry: Dict[str, Any] | None
    ) -> tuple[float, float, float, float]:
        if mesh_geometry is not None and mesh_geometry.get('node_x') is not None and mesh_geometry.get('node_y') is not None:
            extent_x = self._flatten(mesh_geometry['node_x'])
            extent_y = self._flatten(mesh_geometry['node_y'])
        else:
            extent_x = x
            extent_y = y

        valid = np.isfinite(extent_x) & np.isfinite(extent_y)
        min_x = float(np.nanmin(extent_x[valid]))
        max_x = float(np.nanmax(extent_x[valid]))
        min_y = float(np.nanmin(extent_y[valid]))
        max_y = float(np.nanmax(extent_y[valid]))

        if min_x == max_x:
            min_x -= 0.5
            max_x += 0.5
        if min_y == max_y:
            min_y -= 0.5
            max_y += 0.5

        return (min_x, max_x, min_y, max_y)

    def _raster_shape(self, extent: tuple[float, float, float, float]) -> tuple[int, int]:
        min_x, max_x, min_y, max_y = extent
        width = max_x - min_x
        height = max_y - min_y
        if width >= height:
            n_cols = self.RASTER_MAX_SIDE
            n_rows = max(2, int(round(self.RASTER_MAX_SIDE * height / width)))
        else:
            n_rows = self.RASTER_MAX_SIDE
            n_cols = max(2, int(round(self.RASTER_MAX_SIDE * width / height)))
        return n_rows, n_cols

    def _get_raster_weights(
        self, x: np.ndarray, y: np.ndarray, mesh_geometry: Dict[str, Any] | None
    ) -> RasterInterpolationWeights:
        geometry_key = self._geometry_key(x, y, mesh_geometry)
        current_weights = getattr(self, '_raster_weights', None)
        if current_weights is not None and current_weights.geometry_key == geometry_key:
            return current_weights

        weights = self._build_raster_weights(x, y, mesh_geometry, geometry_key)
        self._raster_weights = weights
        return weights

    def _build_raster_weights(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mesh_geometry: Dict[str, Any] | None,
        geometry_key: tuple,
    ) -> RasterInterpolationWeights:
        extent = self._spatial_extent(x, y, mesh_geometry)
        n_rows, n_cols = self._raster_shape(extent)
        min_x, max_x, min_y, max_y = extent

        pixel_x = np.linspace(min_x, max_x, n_cols, endpoint=False) + (max_x - min_x) / (2 * n_cols)
        pixel_y = np.linspace(min_y, max_y, n_rows, endpoint=False) + (max_y - min_y) / (2 * n_rows)
        pixel_xx, pixel_yy = np.meshgrid(pixel_x, pixel_y)
        pixel_points = np.column_stack((pixel_xx.ravel(), pixel_yy.ravel()))

        valid_faces = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        face_points = np.column_stack((x[valid_faces], y[valid_faces]))
        tree = cKDTree(face_points)
        k_neighbors = min(self.RASTER_K_NEIGHBORS, valid_faces.size)
        distances, local_indices = self._query_tree(tree, pixel_points, k_neighbors)
        if k_neighbors == 1:
            distances = distances[:, None]
            local_indices = local_indices[:, None]

        face_indices = valid_faces[local_indices]
        interpolation_weights = self._inverse_distance_weights(distances)
        valid_mask = self._raster_valid_mask(pixel_points, distances[:, 0], extent, mesh_geometry).reshape(n_rows, n_cols)
        quiver_indices = self._sample_indices(x.size, self.LARGE_GRID_QUIVER_LIMIT)
        if isinstance(quiver_indices, slice):
            quiver_indices = np.arange(x.size)[quiver_indices]

        return RasterInterpolationWeights(
            geometry_key=geometry_key,
            shape=(n_rows, n_cols),
            extent=extent,
            face_indices=face_indices,
            weights=interpolation_weights,
            valid_mask=valid_mask,
            quiver_indices=quiver_indices,
        )

    @staticmethod
    def _query_tree(tree: cKDTree, pixel_points: np.ndarray, k_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            return tree.query(pixel_points, k=k_neighbors, workers=-1)
        except TypeError:
            return tree.query(pixel_points, k=k_neighbors)

    @staticmethod
    def _inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
        zero_distance = distances <= np.finfo(float).eps
        has_exact_match = np.any(zero_distance, axis=1)
        weights = np.zeros_like(distances, dtype=float)

        if np.any(has_exact_match):
            exact = zero_distance[has_exact_match]
            weights[has_exact_match] = exact / exact.sum(axis=1, keepdims=True)

        remaining = ~has_exact_match
        if np.any(remaining):
            inverse_distance = 1.0 / np.maximum(distances[remaining], np.finfo(float).eps) ** 2
            weights[remaining] = inverse_distance / inverse_distance.sum(axis=1, keepdims=True)

        return weights

    def _raster_valid_mask(
        self,
        pixel_points: np.ndarray,
        nearest_distances: np.ndarray,
        extent: tuple[float, float, float, float],
        mesh_geometry: Dict[str, Any] | None,
    ) -> np.ndarray:
        mask = np.ones(pixel_points.shape[0], dtype=bool)
        if mesh_geometry is not None:
            hull_mask = self._mesh_hull_mask(pixel_points, mesh_geometry)
            if hull_mask is not None:
                mask &= hull_mask

        cell_radius = self._representative_cell_radius(mesh_geometry)
        min_x, max_x, min_y, max_y = extent
        pixel_dx = (max_x - min_x) / max(1, self._raster_shape(extent)[1])
        pixel_dy = (max_y - min_y) / max(1, self._raster_shape(extent)[0])
        pixel_diagonal = float(np.hypot(pixel_dx, pixel_dy))
        if cell_radius is not None and np.isfinite(cell_radius):
            mask &= nearest_distances <= max(3.0 * cell_radius, 2.0 * pixel_diagonal)

        return mask

    def _mesh_hull_mask(self, pixel_points: np.ndarray, mesh_geometry: Dict[str, Any]) -> np.ndarray | None:
        node_x = mesh_geometry.get('node_x')
        node_y = mesh_geometry.get('node_y')
        if node_x is None or node_y is None:
            return None

        coords = np.column_stack((self._flatten(node_x), self._flatten(node_y)))
        finite = np.isfinite(coords).all(axis=1)
        coords = coords[finite]
        if coords.shape[0] < 3:
            return None

        try:
            hull = ConvexHull(coords)
        except QhullError:
            return None

        hull_path = MplPath(coords[hull.vertices])
        return hull_path.contains_points(pixel_points)

    def _representative_cell_radius(self, mesh_geometry: Dict[str, Any] | None) -> float | None:
        if mesh_geometry is None:
            return None

        node_x = mesh_geometry.get('node_x')
        node_y = mesh_geometry.get('node_y')
        connectivity = mesh_geometry.get('face_node_connectivity')
        if node_x is None or node_y is None or connectivity is None:
            return None

        node_x = self._flatten(node_x)
        node_y = self._flatten(node_y)
        connectivity = np.asarray(connectivity, dtype=np.int64)
        if connectivity.size == 0:
            return None

        sample_size = min(connectivity.shape[0], 100_000)
        sample_indices = np.linspace(0, connectivity.shape[0] - 1, sample_size, dtype=np.intp)
        sampled_connectivity = connectivity[sample_indices]
        valid = sampled_connectivity >= 0
        clipped = np.clip(sampled_connectivity, 0, max(0, node_x.size - 1))
        face_node_x = np.where(valid, node_x[clipped], np.nan)
        face_node_y = np.where(valid, node_y[clipped], np.nan)
        center_x = np.nanmean(face_node_x, axis=1)
        center_y = np.nanmean(face_node_y, axis=1)
        radius = np.nanmax(np.hypot(face_node_x - center_x[:, None], face_node_y - center_y[:, None]), axis=1)
        radius = radius[np.isfinite(radius) & (radius > 0)]
        if radius.size == 0:
            return None
        return float(np.nanmedian(radius))

    def _rasterize_field(self, values: np.ndarray, weights: RasterInterpolationWeights) -> np.ndarray:
        field = self._flatten(values)
        sampled_values = field[weights.face_indices]
        finite = np.isfinite(sampled_values)
        weighted_values = np.where(finite, sampled_values * weights.weights, 0.0)
        valid_weights = np.where(finite, weights.weights, 0.0)
        denominator = valid_weights.sum(axis=1)

        image = np.full(weights.face_indices.shape[0], np.nan, dtype=float)
        valid = denominator > 0.0
        image[valid] = weighted_values.sum(axis=1)[valid] / denominator[valid]
        image[~weights.valid_mask.ravel()] = np.nan
        return image.reshape(weights.shape)

    def _update_image_artist(
        self,
        axis_name: str,
        image: np.ndarray,
        weights: RasterInterpolationWeights,
        cmap,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        ax = self.axes[axis_name]
        images = getattr(self, '_raster_images', {})
        image_artist = images.get(axis_name)
        if image_artist is None or image_artist.axes is not ax:
            ax.clear()
            image_artist = ax.imshow(
                image,
                origin='lower',
                extent=weights.extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest',
            )
            images[axis_name] = image_artist
            self._raster_images = images
        else:
            image_artist.set_data(image)
            image_artist.set_extent(weights.extent)
            image_artist.set_clim(vmin=vmin, vmax=vmax)
        return image_artist

    def _remove_spatial_artist(self, artist_key: str) -> None:
        artists = getattr(self, artist_key, {})
        if isinstance(artists, dict):
            for artist in artists.values():
                try:
                    artist.remove()
                except (AttributeError, NotImplementedError, ValueError):
                    pass
            artists.clear()
        elif isinstance(artists, list):
            for artist in artists:
                try:
                    artist.remove()
                except (AttributeError, NotImplementedError, ValueError):
                    pass
            artists.clear()

    def _update_flowfield_plot(
        self,
        flow_field: Dict[str, np.ndarray],
        bathymetry: np.ndarray,
        mesh_geometry: Dict[str, Any] | None = None,
    ) -> None:
        """Update the flow field spatial plot."""
        ax = self.axes['flowfield']

        x = self._flatten(flow_field['x'])
        y = self._flatten(flow_field['y'])
        magnitude = self._flatten(flow_field['magnitude'])
        u = self._flatten(flow_field['u'])
        v = self._flatten(flow_field['v'])
        bathymetry = self._flatten(bathymetry)
        n_points = x.size

        self._remove_spatial_artist('_spatial_quivers')

        if self._is_large_grid(n_points):
            weights = self._get_raster_weights(x, y, mesh_geometry)
            image = self._rasterize_field(magnitude, weights)
            self._update_image_artist('flowfield', image, weights, cmap='viridis')
            quiver_sampled = weights.quiver_indices
        else:
            getattr(self, '_raster_images', {}).pop('flowfield', None)
            ax.clear()
            ax.tricontourf(x, y, magnitude, levels=15, cmap='viridis')
            ax.tricontour(x, y, bathymetry, levels=[0], colors='white', linewidths=2, linestyles='-')
            downsample = max(1, n_points // 20)
            quiver_sampled = slice(None, None, downsample)

        quiver = ax.quiver(
            x[quiver_sampled],
            y[quiver_sampled],
            u[quiver_sampled],
            v[quiver_sampled],
            color='white',
            scale=10,
            width=0.004,
            alpha=0.8,
        )
        self._spatial_quivers = {'flowfield': quiver}

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.set_title('(a) Flow Field (Latest)', fontsize=12, fontweight='bold')

    def _update_bathymetry_plot(
        self,
        flow_field: Dict[str, np.ndarray],
        bathymetry: np.ndarray,
        particles: Dict[str, np.ndarray],
        mesh_geometry: Dict[str, Any] | None = None,
    ) -> None:
        """Update the bathymetry and particles spatial plot."""
        ax = self.axes['bathymetry']

        x = self._flatten(flow_field['x'])
        y = self._flatten(flow_field['y'])
        bathymetry = self._flatten(bathymetry)

        self._remove_spatial_artist('_particle_artists')
        legend = ax.get_legend()
        if legend is not None:
            try:
                legend.remove()
            except (AttributeError, NotImplementedError, ValueError):
                pass

        if self._is_large_grid(x.size):
            weights = self._get_raster_weights(x, y, mesh_geometry)
            image = self._rasterize_field(bathymetry, weights)
            self._update_image_artist(
                'bathymetry',
                image,
                weights,
                cmap=self.bathymetry_cmap,
                vmin=self.bathymetry_vmin,
                vmax=self.bathymetry_vmax,
            )
        else:
            getattr(self, '_raster_images', {}).pop('bathymetry', None)
            ax.clear()
            ax.tricontourf(
                x,
                y,
                bathymetry,
                levels=20,
                cmap=self.bathymetry_cmap,
                vmin=self.bathymetry_vmin,
                vmax=self.bathymetry_vmax,
            )
            ax.tricontour(x, y, bathymetry, levels=[0], colors='black', linewidths=2, linestyles='-')

        # Plot particles
        particle_artists = getattr(self, '_particle_artists', [])
        particle_x = np.asarray(particles['x'])
        particle_y = np.asarray(particles['y'])
        n_particles = len(particle_x)
        if n_particles > 0:
            left_domain = self._particle_status_mask(particles, 'status_left_domain', n_particles)
            beached = self._particle_status_mask(particles, 'status_beached', n_particles)
            visible_particles = ~left_domain
            active_particles = visible_particles & ~beached
            stranded_particles = visible_particles & beached

            # Current in-domain positions (white circles)
            if np.any(active_particles):
                particle_artists.append(
                    ax.scatter(
                        particle_x[active_particles],
                        particle_y[active_particles],
                        color='white',
                        s=50,
                        marker='o',
                        edgecolors='black',
                        linewidth=1,
                        label='Current',
                        zorder=5,
                    )
                )

            # Current stranded/beached positions (light-red circles)
            if np.any(stranded_particles):
                particle_artists.append(
                    ax.scatter(
                        particle_x[stranded_particles],
                        particle_y[stranded_particles],
                        color=self.STRANDED_PARTICLE_COLOR,
                        s=50,
                        marker='o',
                        edgecolors='black',
                        linewidth=1,
                        label='Stranded',
                        zorder=6,
                    )
                )

            initial_x = particles.get('x_initial')
            initial_y = particles.get('y_initial')
            has_initial_positions = initial_x is not None and initial_y is not None
            if has_initial_positions:
                initial_x = np.asarray(initial_x)
                initial_y = np.asarray(initial_y)
                has_initial_positions = initial_x.shape == particle_x.shape and initial_y.shape == particle_y.shape

            # Initial positions (white crosses), excluding particles that left the domain.
            if has_initial_positions and np.any(visible_particles):
                visible_indices = np.flatnonzero(visible_particles)
                particle_artists.append(
                    ax.scatter(
                        initial_x[visible_particles],
                        initial_y[visible_particles],
                        color='white',
                        s=50,
                        marker='x',
                        linewidth=3,
                        label='Initial',
                        zorder=5,
                    )
                )

                trajectory_history = self._particle_trajectory_history(n_particles)
                if trajectory_history is None:
                    trail_x = np.vstack((initial_x, particle_x))
                    trail_y = np.vstack((initial_y, particle_y))
                else:
                    trail_x, trail_y = trajectory_history

                segments = self._particle_trail_segments(trail_x, trail_y, visible_indices)
                if segments.size:
                    trails = LineCollection(
                        segments,
                        colors='white',
                        linewidths=1,
                        alpha=0.7,
                        zorder=4,
                        label='_particle_trails',
                    )
                    ax.add_collection(trails)
                    particle_artists.append(trails)

            if particle_artists:
                ax.legend(loc='upper right')
        self._particle_artists = particle_artists

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.set_title('(b) Bathymetry + Particles', fontsize=12, fontweight='bold')

    @staticmethod
    def _particle_displacement_segments(particles: Dict[str, np.ndarray]) -> np.ndarray:
        """Build finite initial-to-current displacement line segments."""
        current = np.column_stack((particles['x'], particles['y']))
        initial = np.column_stack((particles['x_initial'], particles['y_initial']))
        finite = np.isfinite(current).all(axis=1) & np.isfinite(initial).all(axis=1)
        if not np.any(finite):
            return np.empty((0, 2, 2), dtype=float)
        return np.stack((initial[finite], current[finite]), axis=1)

    def _update_time_series_plots(self) -> None:
        """Update all time series plots."""
        if len(self.time_stamps) < 2:
            return

        times = self.time_stamps  # Already datetime objects

        # Dynamically adjust date format based on time span
        time_span = times[-1] - times[0]
        if time_span.days > 7:
            # For longer simulations, use date without time
            date_fmt = mdates.DateFormatter('%m-%d')
            locator = mdates.DayLocator(interval=1)
        elif time_span.days > 1:
            # For multi-day simulations, use date and time
            date_fmt = mdates.DateFormatter('%m-%d %H:%M')
            locator = mdates.HourLocator(interval=12)
        else:
            # For short simulations, use time only
            date_fmt = mdates.DateFormatter('%H:%M')
            locator = mdates.HourLocator(interval=2)

        # Longshore velocity
        self.lines['longshore_avg'].set_data(times, self.data_store['longshore_avg'])
        self.lines['longshore_max'].set_data(times, self.data_store['longshore_max'])
        self.axes['longshore_vel'].relim()
        self.axes['longshore_vel'].autoscale_view()
        self.axes['longshore_vel'].xaxis.set_major_formatter(date_fmt)
        self.axes['longshore_vel'].xaxis.set_major_locator(locator)
        # Rotate x-axis labels for better readability
        for label in self.axes['longshore_vel'].get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        # Cross-shore velocity
        self.lines['crossshore_avg'].set_data(times, self.data_store['crossshore_avg'])
        self.lines['crossshore_max'].set_data(times, self.data_store['crossshore_max'])
        self.axes['crossshore_vel'].relim()
        self.axes['crossshore_vel'].autoscale_view()
        self.axes['crossshore_vel'].xaxis.set_major_formatter(date_fmt)
        self.axes['crossshore_vel'].xaxis.set_major_locator(locator)
        # Rotate x-axis labels for better readability
        for label in self.axes['crossshore_vel'].get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        # Distance
        self.lines['distance'].set_data(times, self.data_store['distance'])
        self.axes['distance'].relim()
        self.axes['distance'].autoscale_view()
        self.axes['distance'].xaxis.set_major_formatter(date_fmt)
        self.axes['distance'].xaxis.set_major_locator(locator)
        # Rotate x-axis labels for better readability
        for label in self.axes['distance'].get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        # Burial depth
        self.lines['burial_avg'].set_data(times, self.data_store['burial_avg'])
        self.lines['burial_max'].set_data(times, self.data_store['burial_max'])
        self.lines['mixing_depth'].set_data(times, self.data_store['mixing_depth'])
        self.axes['burial_depth'].relim()
        self.axes['burial_depth'].autoscale_view()
        self.axes['burial_depth'].xaxis.set_major_formatter(date_fmt)
        self.axes['burial_depth'].xaxis.set_major_locator(locator)
        # Rotate x-axis labels for better readability
        for label in self.axes['burial_depth'].get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        # Apply tight layout again after data updates
        plt.tight_layout()

    def _update_progress_bar(self, current_time: float, start_time: float, end_time: float) -> None:
        """Update progress bar."""
        if end_time is None or end_time <= start_time:
            progress = 0
        else:
            progress = ((current_time - start_time) / (end_time - start_time)) * 100
            progress = max(0, min(100, progress))  # Clamp between 0 and 100

        # Update progress bar fill
        self.progress_fill.remove()
        self.progress_fill = self.axes['progress'].barh(0, progress, height=0.6, color='green', alpha=0.7)

        # Convert seconds to actual dates
        start_date = self.reference_date + datetime.timedelta(seconds=float(start_time))
        current_date = self.reference_date + datetime.timedelta(seconds=float(current_time))

        # Update text labels with dates
        self.progress_text_start.set_text(f'{start_date.strftime("%Y-%m-%d %H:%M")}')
        self.progress_text_current.set_text(f'{current_date.strftime("%Y-%m-%d %H:%M")} ({progress:.1f}%)')

        if end_time:
            end_date = self.reference_date + datetime.timedelta(seconds=float(end_time))
            self.progress_text_end.set_text(f'{end_date.strftime("%Y-%m-%d %H:%M")}')
        else:
            self.progress_text_end.set_text('End: Unknown')

    def _setup_window_position(self) -> None:
        """Try to position window on second screen and make it full-screen."""
        try:
            backend = plt.get_backend()

            if 'Qt' in backend:
                # Qt backend
                manager = self.fig.canvas.manager
                # Try to get screen information
                from matplotlib.backends.qt_compat import QtWidgets

                app = QtWidgets.QApplication.instance()
                if app is not None:
                    screens = app.screens()
                    if len(screens) > 1:
                        # Move to second screen
                        second_screen = screens[1]
                        geometry = second_screen.geometry()
                        manager.window.move(geometry.x(), geometry.y())
                        manager.window.showMaximized()
                    else:
                        manager.window.showMaximized()

            elif 'Tk' in backend:
                # Tkinter backend
                manager = self.fig.canvas.manager
                window = manager.window
                window.state('zoomed')  # Windows full-screen

        except Exception:
            # Fallback - just maximize on current screen
            try:
                manager = self.fig.canvas.manager
                manager.window.showMaximized()
            except Exception:
                pass  # Give up on window positioning

    def close(self) -> None:
        """Close the dashboard."""
        if self.fig is not None:
            plt.close(self.fig)

    def save(self, save_path: str) -> None:
        """
        Save current dashboard state.

        Parameters
        ----------
        save_path : str
            Path where the figure or output file is saved.
        """
        if self.fig is not None:
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
