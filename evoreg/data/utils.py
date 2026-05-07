import numpy as np
from typing import Tuple
import math

def generate_transformation(
    rotation_range: float = None,
    translation_range: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates random rotation, translation, and scaling.

    Args:
        rotation_range: If provided, constrains rotation to ±rotation_range degrees
                       using Euler angles. If None, uses full SO(3) uniform sampling.
        translation_range: Half-width of translation box (default: 0.2).

    Returns:
        Tuple of (rotation_matrix, translation_vector, scale, transformation_matrix)
    """
    if rotation_range is not None:
        rotation = random_euler_rotation(rotation_range)
    else:
        rotation = random_rotation()
    translation = random_translation(a=translation_range)
    scale = random_scaling()

    # 4x4 transformation matrix (includes rotation and translation, scale applied separately)
    transformation = np.eye(4, dtype=np.float32)
    transformation[:3, :3] = rotation * scale  # Apply scale to rotation
    transformation[:3, 3] = translation

    return rotation, translation, scale, transformation

def random_rotation() -> np.ndarray:
    """
    Generates a uniformly random 3x3 rotation matrix using the
    quaternion method.
    """
    # Draw three random numbers uniformly in [0, 1)
    u1, u2, u3 = np.random.rand(3)

    # Construct a random unit quaternion (x, y, z, w)
    q = np.array([
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1)     * math.sin(2 * math.pi * u3),
        math.sqrt(u1)     * math.cos(2 * math.pi * u3)
    ], dtype=np.float64)

    x, y, z, w = q

    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)

    return R

def random_euler_rotation(max_angle_deg: float = 45.0) -> np.ndarray:
    """
    Generates a random rotation matrix from bounded Euler angles.

    Each of roll, pitch, yaw is sampled uniformly from [-max_angle_deg, +max_angle_deg].
    This produces small-to-moderate rotations suitable for evaluating models
    trained on limited rotation ranges.

    Args:
        max_angle_deg: Maximum rotation angle in degrees for each Euler axis.

    Returns:
        3x3 rotation matrix as float32 numpy array.
    """
    angles_deg = np.random.uniform(-max_angle_deg, max_angle_deg, 3)
    roll, pitch, yaw = np.radians(angles_deg)

    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    return (Rz @ Ry @ Rx).astype(np.float32)

def random_translation(a: float = 0.2) -> np.ndarray:
    """
    Generates random translation vector uniformly distributed in [-a, a]^3.
    
    Translations are sampled from a zero-mean bounded distribution:
    t ~ Uniform([-a, a]^3)
    where a is chosen relative to the object's bounding radius (e.g., a = 0.2 for unit sphere).
    This avoids unreasonably large displacements that would destroy overlap.
    
    Args:
        a: Translation range (half the width of the translation box)
    
    Returns:
        3D translation vector as numpy array
    """
    return ((np.random.rand(3) * 2 - 1) * a).astype(np.float32)

def random_scaling(sigma_log: float = 0.03, anisotropic: bool = False) -> np.ndarray:
    """
    Sample object scaling factors using a log-normal distribution.
    
    To simulate realistic object resizing:
    log(s) ~ N(0, sigma_log^2)
    Thus s = exp(epsilon) with epsilon ~ N(0, sigma_log^2) ensuring multiplicative 
    perturbation around 1. With sigma_log^2 ≈ 0.03.
    
    Args:
        sigma_log: Standard deviation for the log scale perturbation (default: 0.03)
        anisotropic: If True, sample separate scale factors per axis (3 values).
                    If False, return a single isotropic scale factor.
    
    Returns:
        Scaling factor(s) as numpy array. Shape () for isotropic, shape (3,) for anisotropic.
    """
    size = 3 if anisotropic else 1
    epsilon = np.random.randn(size) * sigma_log 
    scaling_factor = np.exp(epsilon).astype(np.float32)
    
    return scaling_factor if anisotropic else scaling_factor.squeeze()

def partial_visibility_mask(points: np.ndarray, retention_ratio: float = 0.6) -> np.ndarray:
    """
    Creates a random half-space occlusion mask (planar cut).

    To emulate occlusion, we define a random half-space mask M with M(x) = 1 if (n, x) >= τ and 0 otherwise, where
    n ~ Uniform(S2) (random normal direction), S2 = {n ∈ R3 | |n| = 1}, the set of all 3D unit vectors.
    To sample uniformly from S2, draw u1, u2 ~ Uniform[0, 1]. Compute nz = 1 - 2u1, φ = 2πu2.
    Then nx = √(1 - nz^2)cos φ, ny = √(1 - nz^2)sin φ.
    Return (nx, ny, nz). τ = Quantile_{1-ρ}(<n, X>) ensuring fraction ρ is kept (you can choose ρ = 0.6).
    This creates uniformly distributed occlusions, as if a random planar cut removes part of the shape.

    Args:
        points: Point cloud array of shape (N, 3)
        retention_ratio: Fraction of points to keep (default: 0.6)

    Returns:
        Boolean mask indicating which points to keep (N,)
    """
    # Sample random direction uniformly on unit sphere
    u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
    nz = 1 - 2 * u1
    phi = 2 * np.pi * u2
    nx = np.sqrt(1 - nz**2) * np.cos(phi)
    ny = np.sqrt(1 - nz**2) * np.sin(phi)
    n = np.array([nx, ny, nz], dtype=np.float32)

    # Compute dot products of all points with normal
    projections = points @ n  # (N,)

    # Find threshold that keeps retention_ratio of points
    tau = np.quantile(projections, 1 - retention_ratio)

    # Create mask: keep points on one side of the plane
    mask = projections >= tau

    return n


def apply_point_dropout(points: np.ndarray, dropout_ratio: float = 0.2) -> np.ndarray:
    """
    Randomly drops a percentage of points (uniform random dropout).

    Simulates missing sensor data or sparse sampling.
    Inspired by PRNet's occlusion augmentation.

    Args:
        points: Point cloud array of shape (N, 3)
        dropout_ratio: Fraction of points to remove (default: 0.2)

    Returns:
        Point cloud with dropout applied (M, 3) where M < N
    """
    n_points = len(points)
    keep_ratio = 1.0 - dropout_ratio
    n_keep = max(1, int(n_points * keep_ratio))  # Keep at least 1 point

    # Randomly select points to keep
    keep_indices = np.random.choice(n_points, size=n_keep, replace=False)
    return points[keep_indices]


def apply_region_occlusion(points: np.ndarray, retention_ratio: float = 0.7) -> np.ndarray:
    """
    Applies region-based occlusion using random planar cut.

    Simulates object occlusion (one object behind another).
    More realistic than uniform dropout as real occlusions are spatial.

    Args:
        points: Point cloud array of shape (N, 3)
        retention_ratio: Fraction of points to keep (default: 0.7)

    Returns:
        Occluded point cloud (M, 3) where M ≈ retention_ratio * N
    """
    mask = partial_visibility_mask(points, retention_ratio)
    return points[mask]


def apply_gaussian_noise(points: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Adds Gaussian noise to point coordinates.

    Simulates sensor measurement error or surface roughness.

    Args:
        points: Point cloud array of shape (N, 3)
        noise_std: Standard deviation of Gaussian noise (default: 0.01)

    Returns:
        Noisy point cloud (N, 3)
    """
    noise = np.random.randn(*points.shape).astype(np.float32) * noise_std
    return points + noise


def apply_outlier_injection(points: np.ndarray, outlier_ratio: float = 0.1) -> np.ndarray:
    """
    Injects random outlier points into the point cloud.

    Simulates clutter, background points, or sensor artifacts.

    Args:
        points: Point cloud array of shape (N, 3)
        outlier_ratio: Ratio of outliers to add relative to N (default: 0.1)

    Returns:
        Point cloud with outliers (N + M, 3) where M = outlier_ratio * N
    """
    n_points = len(points)
    n_outliers = int(n_points * outlier_ratio)

    if n_outliers == 0:
        return points

    # Sample outliers from a larger volume (2x the point cloud bounds)
    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    point_center = (point_min + point_max) / 2
    point_range = (point_max - point_min) * 2

    outliers = np.random.rand(n_outliers, 3).astype(np.float32)
    outliers = point_center + (outliers - 0.5) * point_range

    return np.vstack([points, outliers])


def apply_occlusion_augmentation(
    points: np.ndarray,
    occlusion_type: str = 'dropout',
    occlusion_ratio: float = 0.2,
    noise_std: float = 0.01
) -> np.ndarray:
    """
    Main dispatcher for occlusion augmentation.

    Applies various types of occlusion/noise to simulate real-world conditions.
    Inspired by PRNet's data augmentation strategy.

    Args:
        points: Point cloud array of shape (N, 3)
        occlusion_type: Type of occlusion ('dropout', 'region', 'noise', 'outlier')
        occlusion_ratio: Ratio for dropout/region/outlier operations
        noise_std: Standard deviation for Gaussian noise

    Returns:
        Augmented point cloud
    """
    if occlusion_type == 'dropout':
        return apply_point_dropout(points, dropout_ratio=occlusion_ratio)
    elif occlusion_type == 'region':
        retention_ratio = 1.0 - occlusion_ratio
        return apply_region_occlusion(points, retention_ratio=retention_ratio)
    elif occlusion_type == 'noise':
        return apply_gaussian_noise(points, noise_std=noise_std)
    elif occlusion_type == 'outlier':
        return apply_outlier_injection(points, outlier_ratio=occlusion_ratio)
    else:
        raise ValueError(f"Unknown occlusion type: {occlusion_type}. "
                        f"Choose from: 'dropout', 'region', 'noise', 'outlier'")


def apply_nonrigid_deformation(
    points: np.ndarray,
    n_control_points: int = 8,
    deformation_scale: float = 0.05,
    rbf_sigma: float = 0.3
) -> np.ndarray:
    """
    Applies smooth non-rigid deformation using RBF interpolation.

    Samples random control points from the point cloud, generates random
    displacements for each, then interpolates smoothly across all points
    using Gaussian RBF weights. Produces realistic local deformations.

    Args:
        points: Point cloud array of shape (N, 3)
        n_control_points: Number of control points to sample (default: 8)
        deformation_scale: Maximum displacement magnitude (default: 0.05)
        rbf_sigma: Gaussian RBF bandwidth — controls locality of deformations.
                   Smaller = more local, larger = more global (default: 0.3)

    Returns:
        Deformed point cloud (N, 3)
    """
    n_points = len(points)
    n_ctrl = min(n_control_points, n_points)

    # Sample control point indices
    ctrl_indices = np.random.choice(n_points, n_ctrl, replace=False)
    ctrl_points = points[ctrl_indices]  # (K, 3)

    # Generate random displacements for control points
    ctrl_displacements = (np.random.randn(n_ctrl, 3) * deformation_scale).astype(np.float32)

    # Compute RBF weights: w_ij = exp(-||p_i - c_j||^2 / (2 * sigma^2))
    # points: (N, 3), ctrl_points: (K, 3)
    diff = points[:, np.newaxis, :] - ctrl_points[np.newaxis, :, :]  # (N, K, 3)
    sq_dists = (diff ** 2).sum(axis=2)  # (N, K)
    weights = np.exp(-sq_dists / (2 * rbf_sigma ** 2))  # (N, K)

    # Normalize weights so they sum to 1 per point
    weight_sums = weights.sum(axis=1, keepdims=True) + 1e-8
    weights = weights / weight_sums  # (N, K)

    # Interpolate displacements: displacement_i = sum_j(w_ij * d_j)
    displacements = weights @ ctrl_displacements  # (N, 3)

    return (points + displacements).astype(np.float32)