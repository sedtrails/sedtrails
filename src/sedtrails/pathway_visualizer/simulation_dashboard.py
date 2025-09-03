"""
Simulation Dashboard for real-time monitoring and testing simulations.

This module provides interactive dashboard capabilities for monitoring particle
simulations with spatial and temporal visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
from pathlib import Path
import datetime
from typing import Dict, Tuple


class SimulationDashboard:
    """Real-time visualization dashboard for particle simulations."""

    def __init__(self, reference_date: str = '1970-01-01'):
        """Initialize the dashboard."""
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.data_store = defaultdict(list)
        self.trajectories = {'x': [], 'y': [], 'time': []}  # Store particle trajectories
        self.time_stamps = []
        self.plot_initialized = False
        self.last_update_time = 0
        self.keep_open = True  # Control for keeping window open after simulation

        # Store reference date for time conversions
        self.reference_date = datetime.datetime.fromisoformat(reference_date)

        # Load SE colormap for bathymetry (relative to this file)
        colormap_path = Path(__file__).parent / 'SE_cmap.npy'
        colors = np.load(colormap_path)
        self.bathymetry_cmap = ListedColormap(colors)
        self.bathymetry_vmin = -12
        self.bathymetry_vmax = 6

    def initialize_dashboard(self, figsize: Tuple[float, float] = (16, 10)) -> None:
        """Initialize the dashboard with subplot layout."""
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
        # Longshore velocity
        (self.lines['longshore_avg'],) = self.axes['longshore_vel'].plot([], [], 'b-', label='Average', linewidth=2)
        (self.lines['longshore_max'],) = self.axes['longshore_vel'].plot([], [], 'r-', label='Maximum', linewidth=2)
        self.axes['longshore_vel'].legend()
        self.axes['longshore_vel'].set_ylabel('Velocity (m/s)')
        self.axes['longshore_vel'].grid(True, alpha=0.3)

        # Cross-shore velocity
        (self.lines['crossshore_avg'],) = self.axes['crossshore_vel'].plot([], [], 'b-', label='Average', linewidth=2)
        (self.lines['crossshore_max'],) = self.axes['crossshore_vel'].plot([], [], 'r-', label='Maximum', linewidth=2)
        self.axes['crossshore_vel'].legend()
        self.axes['crossshore_vel'].set_ylabel('Velocity (m/s)')
        self.axes['crossshore_vel'].grid(True, alpha=0.3)

        # Distance
        (self.lines['distance'],) = self.axes['distance'].plot([], [], 'g-', linewidth=2)
        self.axes['distance'].set_ylabel('Distance (m)')
        self.axes['distance'].grid(True, alpha=0.3)

        # Burial depth
        (self.lines['burial_avg'],) = self.axes['burial_depth'].plot([], [], 'darkorange', label='Average', linewidth=2)
        (self.lines['burial_max'],) = self.axes['burial_depth'].plot([], [], 'orangered', label='Maximum', linewidth=2)
        (self.lines['mixing_depth'],) = self.axes['burial_depth'].plot(
            [], [], 'gray', linestyle='--', label='Mixing Depth', linewidth=1
        )
        self.axes['burial_depth'].legend()
        self.axes['burial_depth'].set_ylabel('Burial Depth (m)')
        self.axes['burial_depth'].set_xlabel('Simulation Time (hours)')
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
        self.axes['flowfield'].set_title('Flow Field (Latest)', fontsize=12, fontweight='bold')
        self.axes['bathymetry'].set_title('Bathymetry + Particles', fontsize=12, fontweight='bold')
        self.axes['longshore_vel'].set_title('Longshore Velocity', fontsize=12, fontweight='bold')
        self.axes['crossshore_vel'].set_title('Cross-shore Velocity', fontsize=12, fontweight='bold')
        self.axes['distance'].set_title('Average Distance per Timestep', fontsize=12, fontweight='bold')
        self.axes['burial_depth'].set_title('Burial Depth', fontsize=12, fontweight='bold')

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
    ) -> None:
        """Update dashboard with current simulation data."""

        # Check if we should update based on plot_interval
        if current_time - self.last_update_time < plot_interval:
            return

        # Store trajectory data at this plot_interval
        self.trajectories['x'].append(particles['x'].copy())
        self.trajectories['y'].append(particles['y'].copy())
        self.trajectories['time'].append(current_time)

        # Store data for time series analysis
        self._store_particle_data(particles, current_time, timestep, flow_field)

        # Prepare particle data with initial positions from trajectories
        particle_data_with_initial = particles.copy()
        if len(self.trajectories['x']) > 0:
            particle_data_with_initial['x_initial'] = self.trajectories['x'][0]
            particle_data_with_initial['y_initial'] = self.trajectories['y'][0]

        # Update all plots
        self._update_flowfield_plot(flow_field, bathymetry)
        self._update_bathymetry_plot(flow_field, bathymetry, particle_data_with_initial)
        self._update_time_series_plots()
        self._update_progress_bar(current_time, simulation_start_time, simulation_end_time)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.last_update_time = current_time

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

        # Calculate average distance covered per timestep
        if len(self.data_store['prev_positions']) > 0:
            prev_x, prev_y = self.data_store['prev_positions'][-1]
            distances = np.sqrt((particles['x'] - prev_x) ** 2 + (particles['y'] - prev_y) ** 2)
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0.0

        self.data_store['distance'].append(avg_distance)
        self.data_store['prev_positions'].append((particles['x'].copy(), particles['y'].copy()))

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

    def _update_flowfield_plot(self, flow_field: Dict[str, np.ndarray], bathymetry: np.ndarray) -> None:
        """Update the flow field spatial plot."""
        ax = self.axes['flowfield']
        ax.clear()

        x, y, magnitude = flow_field['x'], flow_field['y'], flow_field['magnitude']

        # Contour plot of magnitude
        ax.tricontourf(x, y, magnitude, levels=15, cmap='viridis')

        # Add z=0 bathymetry contour
        ax.tricontour(x, y, bathymetry, levels=[0], colors='white', linewidths=2, linestyles='-')

        # Add velocity vectors (more vectors - reduce downsample)
        downsample = max(1, len(x) // 20)  # Reduced from 50 to 20 for more vectors
        skip = slice(None, None, downsample)
        ax.quiver(
            x[skip],
            y[skip],
            flow_field['u'][skip],
            flow_field['v'][skip],
            color='white',
            scale=10,
            width=0.004,
            alpha=0.8,
        )  # Adjusted scale and width

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.set_title('Flow Field (Latest)', fontsize=12, fontweight='bold')

    def _update_bathymetry_plot(
        self, flow_field: Dict[str, np.ndarray], bathymetry: np.ndarray, particles: Dict[str, np.ndarray]
    ) -> None:
        """Update the bathymetry and particles spatial plot."""
        ax = self.axes['bathymetry']
        ax.clear()

        x, y = flow_field['x'], flow_field['y']

        # Bathymetry contour plot
        ax.tricontourf(
            x, y, bathymetry, levels=20, cmap=self.bathymetry_cmap, vmin=self.bathymetry_vmin, vmax=self.bathymetry_vmax
        )

        # Add z=0 bathymetry contour
        ax.tricontour(x, y, bathymetry, levels=[0], colors='black', linewidths=2, linestyles='-')

        # Plot particles
        if len(particles['x']) > 0:
            # Current positions (white circles)
            ax.scatter(
                particles['x'],
                particles['y'],
                color='white',
                s=50,
                marker='o',
                edgecolors='black',
                linewidth=1,
                label='Current',
                zorder=5,
            )

            # Initial positions (white crosses)
            if 'x_initial' in particles:
                ax.scatter(
                    particles['x_initial'],
                    particles['y_initial'],
                    color='white',
                    s=50,
                    marker='x',
                    linewidth=3,
                    label='Initial',
                    zorder=5,
                )

                # Connect with lines
                for i in range(len(particles['x'])):
                    ax.plot(
                        [particles['x_initial'][i], particles['x'][i]],
                        [particles['y_initial'][i], particles['y'][i]],
                        'w-',
                        alpha=0.7,
                        linewidth=1,
                        zorder=4,
                    )

            ax.legend(loc='upper right')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.set_title('Bathymetry + Particles', fontsize=12, fontweight='bold')

    def _update_time_series_plots(self) -> None:
        """Update all time series plots."""
        if len(self.time_stamps) < 2:
            return

        times = self.time_stamps  # Already datetime objects

        # Longshore velocity
        self.lines['longshore_avg'].set_data(times, self.data_store['longshore_avg'])
        self.lines['longshore_max'].set_data(times, self.data_store['longshore_max'])
        self.axes['longshore_vel'].relim()
        self.axes['longshore_vel'].autoscale_view()

        # Cross-shore velocity
        self.lines['crossshore_avg'].set_data(times, self.data_store['crossshore_avg'])
        self.lines['crossshore_max'].set_data(times, self.data_store['crossshore_max'])
        self.axes['crossshore_vel'].relim()
        self.axes['crossshore_vel'].autoscale_view()

        # Distance
        self.lines['distance'].set_data(times, self.data_store['distance'])
        self.axes['distance'].relim()
        self.axes['distance'].autoscale_view()

        # Burial depth
        self.lines['burial_avg'].set_data(times, self.data_store['burial_avg'])
        self.lines['burial_max'].set_data(times, self.data_store['burial_max'])
        self.lines['mixing_depth'].set_data(times, self.data_store['mixing_depth'])
        self.axes['burial_depth'].relim()
        self.axes['burial_depth'].autoscale_view()

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
        """Save current dashboard state."""
        if self.fig is not None:
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
