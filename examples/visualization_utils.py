"""
TEMPORARY visualization utilities for SedTRAILS.

This module provides visualization functions for SedTRAILS data,
particularly for flow field visualization.
"""
import matplotlib.pyplot as plt

def plot_flow_field(flow_data, title=None, downsample=5, figsize=(12, 10), 
                   cmap='viridis', vector_color='white', save_path=None):
    """
    Plot flow field with magnitude as contour and vectors for direction.
    
    Parameters:
    -----------
    flow_data : dict
        Dictionary containing 'x', 'y', 'u', 'v', and 'magnitude' arrays
    title : str, optional
        Title for the plot
    downsample : int, optional
        Factor to downsample vectors for clearer visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap for the contour plot
    vector_color : str, optional
        Color for the velocity vectors
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        
    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axis objects
    """
    # Extract data
    x = flow_data['x']
    y = flow_data['y']
    u = flow_data['u']
    v = flow_data['v']
    magnitude = flow_data['magnitude']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot magnitude as contour/heatmap
    contour = ax.tricontourf(x, y, magnitude, cmap=cmap, levels=20)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Flow Magnitude (m/s)')
    
    # Downsample for vectors to avoid cluttering
    skip = (slice(None, None, downsample), slice(None, None, downsample))
    if len(x.shape) > 1:  # 2D grid
        x_subset = x[skip]
        y_subset = y[skip]
        u_subset = u[skip]
        v_subset = v[skip]
    else:  # 1D arrays
        x_subset = x[::downsample]
        y_subset = y[::downsample]
        u_subset = u[::downsample]
        v_subset = v[::downsample]
    
    # Plot velocity vectors
    q = ax.quiver(x_subset, y_subset, u_subset, v_subset, 
                 color=vector_color, scale=15, width=0.002)
    ax.quiverkey(q, 0.85, 0.02, 0.5, "0.5 m/s", 
                labelpos='E', coordinates='figure', color=vector_color)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Flow Field Visualization')
    
    # Equal aspect ratio for geographic data
    ax.set_aspect('equal')
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig, ax


# Define a new function for plotting particle trajectories
def plot_particle_trajectory(flow_data, trajectory_x, trajectory_y, title=None, 
                           downsample=5, figsize=(12, 10), cmap='viridis',
                           vector_color='white', trajectory_color='red', 
                           save_path=None):
    """
    Plot flow field with particle trajectory overlay.
    
    Parameters:
    -----------
    flow_data : dict
        Dictionary containing 'x', 'y', 'u', 'v', and 'magnitude' arrays
    trajectory_x, trajectory_y : array_like
        Arrays containing the x and y coordinates of particle positions
    title : str, optional
        Title for the plot
    downsample : int, optional
        Factor to downsample vectors for clearer visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap for the contour plot
    vector_color : str, optional
        Color for the velocity vectors
    trajectory_color : str, optional
        Color for the particle trajectory
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        
    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axis objects
    """
    # First create the flow field plot using existing function
    fig, ax = plot_flow_field(flow_data, title=title, downsample=downsample,
                             figsize=figsize, cmap=cmap, vector_color=vector_color,
                             save_path=None)  # Don't save yet
    
    # Add particle trajectory
    ax.plot(trajectory_x, trajectory_y, '-', color=trajectory_color, 
           linewidth=2, label='Particle Trajectory')
    ax.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=8, label='Start Position')
    ax.plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=8, label='End Position')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Update title if not provided
    if not title:
        ax.set_title('Flow Field with Particle Trajectory')
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig, ax
