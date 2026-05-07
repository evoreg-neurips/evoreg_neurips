"""
Visualization utilities for point clouds.

Provides functions for visualizing point clouds, registration results,
and training progress using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple, Union
import torch


def visualize_point_cloud(
    points: Union[np.ndarray, torch.Tensor],
    title: str = "Point Cloud",
    color: Union[str, np.ndarray] = 'blue',
    size: int = 1,
    ax: Optional[Axes3D] = None,
    view_angles: Tuple[float, float] = (30, 45)
) -> Axes3D:
    """
    Visualizes a single point cloud in 3D.
    
    Args:
        points: Point cloud of shape (N, 3)
        title: Title for the plot
        color: Color specification (string or per-point colors)
        size: Point size for scatter plot
        ax: Existing 3D axis to plot on (creates new if None)
        view_angles: Elevation and azimuth angles for 3D view
        
    Returns:
        3D axis object with plotted point cloud
    """
    # Converts torch tensor to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # Creates new figure and axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plots the point cloud
    ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2],
        c=color,
        s=size,
        alpha=0.6
    )
    
    # Sets equal aspect ratio for all axes
    set_axes_equal(ax)
    
    # Sets viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Adds labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return ax


def visualize_registration(
    source: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    transformed: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: str = "Point Cloud Registration",
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Visualizes point cloud registration results.
    
    Shows source, target, and optionally transformed point clouds
    side by side for comparison.
    
    Args:
        source: Source point cloud (N, 3)
        target: Target point cloud (M, 3)
        transformed: Transformed source point cloud (N, 3)
        title: Overall figure title
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Converts tensors to numpy arrays
    if isinstance(source, torch.Tensor):
        source = source.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if transformed is not None and isinstance(transformed, torch.Tensor):
        transformed = transformed.detach().cpu().numpy()
    
    # Determines number of subplots based on whether transformed is provided
    n_plots = 3 if transformed is not None else 2
    
    # Creates figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Plots source point cloud
    ax1 = fig.add_subplot(1, n_plots, 1, projection='3d')
    visualize_point_cloud(source, "Source", color='red', ax=ax1)
    
    # Plots target point cloud
    ax2 = fig.add_subplot(1, n_plots, 2, projection='3d')
    visualize_point_cloud(target, "Target", color='blue', ax=ax2)
    
    # Plots transformed source if provided
    if transformed is not None:
        ax3 = fig.add_subplot(1, n_plots, 3, projection='3d')
        # Overlays transformed source on target
        visualize_point_cloud(target, "Overlay", color='blue', ax=ax3, size=1)
        visualize_point_cloud(transformed, "", color='green', ax=ax3, size=2)
        ax3.legend(['Target', 'Transformed Source'])
    
    plt.tight_layout()
    return fig


def visualize_correspondences(
    source: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    correspondences: Optional[np.ndarray] = None,
    max_lines: int = 100,
    title: str = "Point Correspondences"
) -> plt.Figure:
    """
    Visualizes correspondences between point clouds.
    
    Draws lines connecting corresponding points between
    source and target point clouds.
    
    Args:
        source: Source point cloud (N, 3)
        target: Target point cloud (M, 3)
        correspondences: Correspondence indices (N,) or None for 1-to-1
        max_lines: Maximum number of correspondence lines to draw
        title: Figure title
        
    Returns:
        Matplotlib figure object
    """
    # Converts tensors to numpy
    if isinstance(source, torch.Tensor):
        source = source.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Creates figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plots both point clouds
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], 
               c='red', s=20, alpha=0.5, label='Source')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], 
               c='blue', s=20, alpha=0.5, label='Target')
    
    # Determines correspondences
    if correspondences is None:
        # Assumes 1-to-1 correspondence if same size
        if source.shape[0] == target.shape[0]:
            correspondences = np.arange(source.shape[0])
        else:
            # Finds nearest neighbors
            from scipy.spatial import distance_matrix
            dist_matrix = distance_matrix(source, target)
            correspondences = np.argmin(dist_matrix, axis=1)
    
    # Draws correspondence lines
    n_lines = min(max_lines, len(correspondences))
    indices = np.random.choice(len(correspondences), n_lines, replace=False)
    
    for i in indices:
        # Gets corresponding points
        p1 = source[i]
        p2 = target[correspondences[i]]
        
        # Draws line between points
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    # Sets labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Sets equal aspect ratio
    set_axes_equal(ax)
    
    return fig


def plot_training_history(
    history: dict,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plots training history with loss curves.
    
    Args:
        history: Dictionary with 'loss', 'val_loss', etc.
        title: Overall figure title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Creates figure with subplots for different metrics
    n_metrics = len(history)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # Ensures axes is always a list
    if n_metrics == 1:
        axes = [axes]
    
    # Plots each metric
    for idx, (metric_name, values) in enumerate(history.items()):
        ax = axes[idx]
        
        # Plots the metric values
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, 'b-', label=metric_name)
        
        # Adds smoothed version if enough data points
        if len(values) > 10:
            # Computes moving average for smoothing
            window_size = min(10, len(values) // 5)
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            smooth_epochs = range(window_size//2 + 1, len(values) - window_size//2 + 1)
            ax.plot(smooth_epochs, smoothed, 'r-', alpha=0.5, label=f'{metric_name} (smoothed)')
        
        # Sets labels and grid
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Sets overall title
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def set_axes_equal(ax: Axes3D):
    """
    Makes axes of 3D plot have equal scale.
    
    This is important for visualizing point clouds without distortion.
    
    Args:
        ax: 3D matplotlib axis
    """
    # Gets current limits
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    # Calculates ranges
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    # Finds the largest range
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    # Sets new limits with equal scale
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def save_visualization(
    fig: plt.Figure,
    filename: str,
    dpi: int = 150,
    bbox_inches: str = 'tight'
):
    """
    Saves a matplotlib figure to file.
    
    Args:
        fig: Figure to save
        filename: Output filename
        dpi: Resolution in dots per inch
        bbox_inches: How to handle figure boundaries
    """
    # Saves the figure
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    
    # Provides feedback
    print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    # Tests visualization functions
    print("Testing visualization utilities...")
    
    # Creates synthetic test data
    n_points = 1000
    
    # Generates a sphere point cloud
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 1.0
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    sphere = np.stack([x, y, z], axis=1)
    
    # Generates a transformed version (scaled and translated)
    transformed = sphere * 0.8 + np.array([0.2, 0.1, 0.0])
    
    # Tests single point cloud visualization
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    visualize_point_cloud(sphere, "Test Sphere", ax=ax1)
    
    # Tests registration visualization
    fig2 = visualize_registration(sphere, transformed, transformed * 0.9)
    
    # Tests correspondence visualization
    fig3 = visualize_correspondences(sphere[:100], transformed[:100])
    
    # Tests training history plot
    history = {
        'loss': np.exp(-np.linspace(0, 2, 50)) + np.random.randn(50) * 0.05,
        'val_loss': np.exp(-np.linspace(0, 1.8, 50)) + np.random.randn(50) * 0.08
    }
    fig4 = plot_training_history(history)
    
    print("[OK] Visualization tests completed!")
    print("Close the plots to continue...")
    
    plt.show()