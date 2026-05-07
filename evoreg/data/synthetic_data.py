"""
Synthetic data generation for testing EvoReg.

Provides functions to generate synthetic point clouds and registration
pairs for development and testing purposes.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import torch


def generate_sphere(
    n_points: int = 1000,
    radius: float = 1.0,
    noise_std: float = 0.0,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generates a sphere point cloud.
    
    Creates points uniformly distributed on a sphere surface
    with optional Gaussian noise.
    
    Args:
        n_points: Number of points to generate
        radius: Sphere radius
        noise_std: Standard deviation of Gaussian noise
        center: Center position (default: origin)
        
    Returns:
        Point cloud of shape (n_points, 3)
    """
    # Uses spherical coordinates for uniform distribution
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n_points))
    
    # Converts to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    # Stacks into point cloud
    points = np.stack([x, y, z], axis=1)
    
    # Adds noise if specified
    if noise_std > 0:
        noise = np.random.randn(n_points, 3) * noise_std
        points = points + noise
    
    # Translates to specified center
    if center is not None:
        points = points + center.reshape(1, 3)
    
    return points.astype(np.float32)


def generate_cube(
    n_points: int = 1000,
    side_length: float = 2.0,
    noise_std: float = 0.0,
    center: Optional[np.ndarray] = None,
    surface_only: bool = True
) -> np.ndarray:
    """
    Generates a cube point cloud.
    
    Creates points on cube surfaces or within cube volume.
    
    Args:
        n_points: Number of points to generate
        side_length: Length of cube sides
        noise_std: Standard deviation of Gaussian noise
        center: Center position (default: origin)
        surface_only: If True, generates points only on surfaces
        
    Returns:
        Point cloud of shape (n_points, 3)
    """
    half_side = side_length / 2.0
    
    if surface_only:
        # Generates points on cube surfaces
        # Determines points per face (approximately)
        points_per_face = n_points // 6
        remaining = n_points - (points_per_face * 6)
        
        face_points = []
        
        # Generates points for each face
        for face in range(6):
            n_face_points = points_per_face + (1 if face < remaining else 0)
            
            # Generates random 2D points
            u = np.random.uniform(-half_side, half_side, n_face_points)
            v = np.random.uniform(-half_side, half_side, n_face_points)
            
            # Maps to 3D based on face
            if face == 0:  # +X face
                face_pts = np.stack([np.full(n_face_points, half_side), u, v], axis=1)
            elif face == 1:  # -X face
                face_pts = np.stack([np.full(n_face_points, -half_side), u, v], axis=1)
            elif face == 2:  # +Y face
                face_pts = np.stack([u, np.full(n_face_points, half_side), v], axis=1)
            elif face == 3:  # -Y face
                face_pts = np.stack([u, np.full(n_face_points, -half_side), v], axis=1)
            elif face == 4:  # +Z face
                face_pts = np.stack([u, v, np.full(n_face_points, half_side)], axis=1)
            else:  # -Z face
                face_pts = np.stack([u, v, np.full(n_face_points, -half_side)], axis=1)
            
            face_points.append(face_pts)
        
        # Concatenates all face points
        points = np.vstack(face_points)
    
    else:
        # Generates points within cube volume
        points = np.random.uniform(-half_side, half_side, (n_points, 3))
    
    # Adds noise if specified
    if noise_std > 0:
        noise = np.random.randn(n_points, 3) * noise_std
        points = points + noise
    
    # Translates to specified center
    if center is not None:
        points = points + center.reshape(1, 3)
    
    return points.astype(np.float32)


def generate_cylinder(
    n_points: int = 1000,
    radius: float = 1.0,
    height: float = 2.0,
    noise_std: float = 0.0,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generates a cylinder point cloud.
    
    Creates points on cylinder surface including top and bottom caps.
    
    Args:
        n_points: Number of points to generate
        radius: Cylinder radius
        height: Cylinder height
        noise_std: Standard deviation of Gaussian noise
        center: Center position (default: origin)
        
    Returns:
        Point cloud of shape (n_points, 3)
    """
    # Allocates points between surface and caps
    n_surface = int(n_points * 0.8)  # 80% on curved surface
    n_caps = n_points - n_surface
    n_top = n_caps // 2
    n_bottom = n_caps - n_top
    
    # Generates points on curved surface
    theta = np.random.uniform(0, 2 * np.pi, n_surface)
    z_surface = np.random.uniform(-height/2, height/2, n_surface)
    x_surface = radius * np.cos(theta)
    y_surface = radius * np.sin(theta)
    surface_points = np.stack([x_surface, y_surface, z_surface], axis=1)
    
    # Generates points on top cap
    r_top = np.sqrt(np.random.uniform(0, radius**2, n_top))
    theta_top = np.random.uniform(0, 2 * np.pi, n_top)
    x_top = r_top * np.cos(theta_top)
    y_top = r_top * np.sin(theta_top)
    z_top = np.full(n_top, height/2)
    top_points = np.stack([x_top, y_top, z_top], axis=1)
    
    # Generates points on bottom cap
    r_bottom = np.sqrt(np.random.uniform(0, radius**2, n_bottom))
    theta_bottom = np.random.uniform(0, 2 * np.pi, n_bottom)
    x_bottom = r_bottom * np.cos(theta_bottom)
    y_bottom = r_bottom * np.sin(theta_bottom)
    z_bottom = np.full(n_bottom, -height/2)
    bottom_points = np.stack([x_bottom, y_bottom, z_bottom], axis=1)
    
    # Concatenates all points
    points = np.vstack([surface_points, top_points, bottom_points])
    
    # Adds noise if specified
    if noise_std > 0:
        noise = np.random.randn(len(points), 3) * noise_std
        points = points + noise
    
    # Translates to specified center
    if center is not None:
        points = points + center.reshape(1, 3)
    
    return points.astype(np.float32)


def generate_torus(
    n_points: int = 1000,
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    noise_std: float = 0.0,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generates a torus (donut) point cloud.
    
    Args:
        n_points: Number of points to generate
        major_radius: Distance from torus center to tube center
        minor_radius: Radius of the tube
        noise_std: Standard deviation of Gaussian noise
        center: Center position (default: origin)
        
    Returns:
        Point cloud of shape (n_points, 3)
    """
    # Generates angles for torus parametrization
    theta = np.random.uniform(0, 2 * np.pi, n_points)  # Around major circle
    phi = np.random.uniform(0, 2 * np.pi, n_points)    # Around tube
    
    # Computes torus points
    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
    z = minor_radius * np.sin(phi)
    
    # Stacks into point cloud
    points = np.stack([x, y, z], axis=1)
    
    # Adds noise if specified
    if noise_std > 0:
        noise = np.random.randn(n_points, 3) * noise_std
        points = points + noise
    
    # Translates to specified center
    if center is not None:
        points = points + center.reshape(1, 3)
    
    return points.astype(np.float32)


def apply_transformation(
    points: np.ndarray,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    scale: float = 1.0
) -> np.ndarray:
    """
    Applies rigid transformation to point cloud.
    
    Args:
        points: Input point cloud (N, 3)
        rotation: 3x3 rotation matrix or None for identity
        translation: 3D translation vector or None for zero
        scale: Uniform scaling factor
        
    Returns:
        Transformed point cloud (N, 3)
    """
    # Applies scaling first
    transformed = points * scale
    
    # Applies rotation if provided
    if rotation is not None:
        transformed = transformed @ rotation.T
    
    # Applies translation if provided
    if translation is not None:
        transformed = transformed + translation.reshape(1, 3)
    
    return transformed


def generate_random_rotation() -> np.ndarray:
    """
    Generates a random 3D rotation matrix.
    
    Uses the approach from "Generating uniform random rotations"
    by James Arvo.
    
    Returns:
        3x3 rotation matrix
    """
    # Generates random quaternion
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    
    # Computes quaternion components
    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    
    # Converts quaternion to rotation matrix
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])
    
    return R


def generate_registration_pair(
    shape_type: str = 'sphere',
    n_points: int = 1000,
    noise_std: float = 0.01,
    outlier_ratio: float = 0.0,
    partial_overlap: float = 1.0,
    **shape_kwargs
) -> Dict[str, np.ndarray]:
    """
    Generates a pair of point clouds for registration testing.
    
    Creates a source point cloud and a transformed version as target,
    with optional noise, outliers, and partial overlap.
    
    Args:
        shape_type: Type of shape ('sphere', 'cube', 'cylinder', 'torus')
        n_points: Number of points in source
        noise_std: Noise standard deviation for target
        outlier_ratio: Fraction of outliers to add
        partial_overlap: Fraction of points that overlap (1.0 = full)
        **shape_kwargs: Additional arguments for shape generation
        
    Returns:
        Dictionary with 'source', 'target', 'transformation' entries
    """
    # Generates source shape based on type
    if shape_type == 'sphere':
        source = generate_sphere(n_points, **shape_kwargs)
    elif shape_type == 'cube':
        source = generate_cube(n_points, **shape_kwargs)
    elif shape_type == 'cylinder':
        source = generate_cylinder(n_points, **shape_kwargs)
    elif shape_type == 'torus':
        source = generate_torus(n_points, **shape_kwargs)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    # Generates random transformation
    rotation = generate_random_rotation()
    translation = np.random.randn(3) * 0.5
    scale = np.random.uniform(0.8, 1.2)
    
    # Applies transformation to create initial target
    target = apply_transformation(source, rotation, translation, scale)
    
    # Handles partial overlap
    if partial_overlap < 1.0:
        # Keeps only a subset of points for partial overlap
        n_keep = int(n_points * partial_overlap)
        keep_indices = np.random.choice(n_points, n_keep, replace=False)
        target = target[keep_indices]
        
        # Adds some points from a different region
        n_extra = n_points - n_keep
        extra_center = translation + np.random.randn(3)
        if shape_type == 'sphere':
            extra_points = generate_sphere(n_extra, center=extra_center, **shape_kwargs)
        else:
            extra_points = generate_sphere(n_extra, center=extra_center)  # Default to sphere
        
        target = np.vstack([target, extra_points])
    
    # Adds noise to target
    if noise_std > 0:
        noise = np.random.randn(len(target), 3) * noise_std
        target = target + noise
    
    # Adds outliers
    if outlier_ratio > 0:
        n_outliers = int(len(target) * outlier_ratio)
        outlier_indices = np.random.choice(len(target), n_outliers, replace=False)
        
        # Generates random outlier positions
        bbox_min = np.min(target, axis=0)
        bbox_max = np.max(target, axis=0)
        outliers = np.random.uniform(bbox_min * 2, bbox_max * 2, (n_outliers, 3))
        
        target[outlier_indices] = outliers
    
    # Creates transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation * scale
    transformation[:3, 3] = translation
    
    return {
        'source': source.astype(np.float32),
        'target': target.astype(np.float32),
        'transformation': transformation.astype(np.float32),
        'rotation': rotation.astype(np.float32),
        'translation': translation.astype(np.float32),
        'scale': scale
    }


def generate_dataset(
    n_samples: int = 100,
    save_dir: Optional[str] = None,
    **pair_kwargs
) -> List[Dict[str, np.ndarray]]:
    """
    Generates a dataset of registration pairs.
    
    Args:
        n_samples: Number of registration pairs to generate
        save_dir: Directory to save generated data (optional)
        **pair_kwargs: Arguments passed to generate_registration_pair
        
    Returns:
        List of registration pair dictionaries
    """
    # Generates all pairs
    dataset = []
    
    for i in range(n_samples):
        # Varies shape types
        shape_types = ['sphere', 'cube', 'cylinder', 'torus']
        shape_type = shape_types[i % len(shape_types)]
        
        # Generates pair
        pair = generate_registration_pair(shape_type=shape_type, **pair_kwargs)
        pair['shape_type'] = shape_type
        pair['index'] = i
        
        dataset.append(pair)
        
        # Saves to file if directory specified
        if save_dir is not None:
            from pathlib import Path
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Saves as npz file
            np.savez(
                save_path / f'pair_{i:04d}.npz',
                source=pair['source'],
                target=pair['target'],
                transformation=pair['transformation'],
                shape_type=shape_type
            )
    
    if save_dir is not None:
        print(f"Saved {n_samples} registration pairs to {save_dir}")
    
    return dataset


if __name__ == "__main__":
    # Tests synthetic data generation
    print("Testing synthetic data generation...")
    
    # Tests individual shape generation
    sphere = generate_sphere(1000, noise_std=0.01)
    cube = generate_cube(1000, surface_only=True)
    cylinder = generate_cylinder(1000)
    torus = generate_torus(1000)
    
    print(f"Generated shapes:")
    print(f"  Sphere: {sphere.shape}")
    print(f"  Cube: {cube.shape}")
    print(f"  Cylinder: {cylinder.shape}")
    print(f"  Torus: {torus.shape}")
    
    # Tests registration pair generation
    pair = generate_registration_pair(
        shape_type='sphere',
        n_points=500,
        noise_std=0.02,
        outlier_ratio=0.1,
        partial_overlap=0.8
    )
    
    print(f"\nGenerated registration pair:")
    print(f"  Source: {pair['source'].shape}")
    print(f"  Target: {pair['target'].shape}")
    print(f"  Transformation: {pair['transformation'].shape}")
    
    # Visualizes if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 4))
        
        # Plots source
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(sphere[:, 0], sphere[:, 1], sphere[:, 2], c='blue', s=1)
        ax1.set_title('Sphere')
        
        # Plots cube
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(cube[:, 0], cube[:, 1], cube[:, 2], c='green', s=1)
        ax2.set_title('Cube')
        
        # Plots registration pair
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(pair['source'][:, 0], pair['source'][:, 1], pair['source'][:, 2], 
                   c='red', s=1, alpha=0.5, label='Source')
        ax3.scatter(pair['target'][:, 0], pair['target'][:, 1], pair['target'][:, 2], 
                   c='blue', s=1, alpha=0.5, label='Target')
        ax3.set_title('Registration Pair')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    
    print("[OK] Synthetic data generation tests passed!")