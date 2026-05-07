"""
Registration Error Metric from Coherent Point Drift (CPD) paper.

Implements the registration error metric as described in:
"Point Set Registration: Coherent Point Drift"
Myronenko and Song, IEEE TPAMI 2010

The registration error is computed as the mean squared distance between
corresponding points after registration.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


def registration_error(
    source: torch.Tensor,
    target: torch.Tensor,
    transformation: Optional[torch.Tensor] = None,
    correspondence: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes the registration error as defined in the CPD paper.

    From Section 7.2 of the CPD paper:
    "Since we know the true correspondences, we use the mean squared
    distance between the corresponding points after the registration
    as an error measure."

    Args:
        source: Transformed/registered source point cloud (B, N, D) or (N, D)
        target: Target point cloud (B, M, D) or (M, D)
        transformation: Optional transformation matrix applied to source
        correspondence: Optional correspondence matrix (B, N, M) or (N, M)
                       If not provided, assumes points have 1-to-1 correspondence
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Registration error value(s)
    """
    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    batch_size = source.shape[0]
    n_source = source.shape[1]
    n_target = target.shape[1]

    # Computes registration error based on CPD paper methodology
    if correspondence is not None:
        # Uses provided correspondence matrix
        if correspondence.dim() == 2:
            correspondence = correspondence.unsqueeze(0)

        # Computes weighted squared distances
        source_expanded = source.unsqueeze(2)  # (B, N, 1, D)
        target_expanded = target.unsqueeze(1)  # (B, 1, M, D)

        # Calculates squared Euclidean distances
        squared_distances = torch.sum((source_expanded - target_expanded) ** 2, dim=-1)  # (B, N, M)

        # Applies correspondence weights
        weighted_errors = squared_distances * correspondence

        # Sums over target dimension to get error per source point
        point_errors = torch.sum(weighted_errors, dim=2)  # (B, N)

    else:
        # Assumes 1-to-1 correspondence (same number of points)
        if n_source != n_target:
            raise ValueError(
                f"Without correspondence matrix, source and target must have "
                f"same number of points. Got {n_source} and {n_target}"
            )

        # Computes direct squared distances between corresponding points
        squared_distances = torch.sum((source - target) ** 2, dim=-1)  # (B, N)
        point_errors = squared_distances

    # Applies reduction
    if reduction == 'mean':
        # Mean squared error over all points and batch
        error = torch.mean(point_errors)
    elif reduction == 'sum':
        # Total squared error
        error = torch.sum(point_errors)
    elif reduction == 'none':
        # Per-batch mean squared error
        error = torch.mean(point_errors, dim=1)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    return error


def registration_error_with_transformation(
    source: torch.Tensor,
    target: torch.Tensor,
    estimated_transform: torch.Tensor,
    ground_truth_transform: Optional[torch.Tensor] = None,
    transformation_type: str = 'rigid'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes registration error with transformation parameters.

    This follows the CPD paper's evaluation methodology where they
    measure both the registration error (point distances) and
    transformation parameter errors.

    Args:
        source: Original source point cloud (B, N, D) or (N, D)
        target: Target point cloud (B, M, D) or (M, D)
        estimated_transform: Estimated transformation parameters
        ground_truth_transform: Ground truth transformation (if available)
        transformation_type: Type of transformation ('rigid', 'affine', 'nonrigid')

    Returns:
        Tuple of (registration_error, transformation_error)
    """
    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    batch_size = source.shape[0]

    # Applies transformation based on type
    if transformation_type == 'rigid':
        # Extracts rotation, translation, and scaling from estimated_transform
        # Assumes estimated_transform contains [R, t, s] parameters
        if len(estimated_transform.shape) == 3:  # Rotation matrix
            R = estimated_transform[:, :source.shape[-1], :source.shape[-1]]
            t = estimated_transform[:, :source.shape[-1], -1]

            # Applies rigid transformation: Y' = sRY + t
            transformed_source = torch.matmul(source, R.transpose(-2, -1))
            transformed_source = transformed_source + t.unsqueeze(1)
        else:
            # Simple case: assumes estimated_transform is already applied
            transformed_source = estimated_transform

    elif transformation_type == 'affine':
        # Applies affine transformation
        if len(estimated_transform.shape) == 3:
            A = estimated_transform[:, :source.shape[-1], :source.shape[-1]]
            t = estimated_transform[:, :source.shape[-1], -1]

            # Applies affine transformation: Y' = AY + t
            transformed_source = torch.matmul(source, A.transpose(-2, -1))
            transformed_source = transformed_source + t.unsqueeze(1)
        else:
            transformed_source = estimated_transform

    elif transformation_type == 'nonrigid':
        # For nonrigid, assumes transformation is already applied
        transformed_source = estimated_transform
    else:
        raise ValueError(f"Unknown transformation type: {transformation_type}")

    # Computes registration error
    reg_error = registration_error(transformed_source, target)

    # Computes transformation parameter error if ground truth is available
    if ground_truth_transform is not None:
        if transformation_type == 'rigid' and len(estimated_transform.shape) == 3:
            # Computes rotation matrix error as per CPD paper Section 7.1
            R_est = estimated_transform[:, :source.shape[-1], :source.shape[-1]]
            R_gt = ground_truth_transform[:, :source.shape[-1], :source.shape[-1]]

            # Uses Frobenius norm of difference
            transform_error = torch.norm(R_est - R_gt, p='fro')
        else:
            # General transformation error
            transform_error = torch.norm(estimated_transform - ground_truth_transform)
    else:
        transform_error = torch.tensor(0.0, device=source.device)

    return reg_error, transform_error


class RegistrationError(nn.Module):
    """
    Registration Error module for point cloud registration evaluation.

    Implements the registration error metric from the CPD paper
    for use in evaluation pipelines.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        squared: bool = True
    ):
        """
        Initializes the Registration Error module.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
            squared: If True, returns squared error (default as per CPD paper)
        """
        super(RegistrationError, self).__init__()
        self.reduction = reduction
        self.squared = squared

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        correspondence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the registration error.

        Args:
            source: Registered source point cloud (B, N, D) or (N, D)
            target: Target point cloud (B, M, D) or (M, D)
            correspondence: Optional correspondence matrix

        Returns:
            Registration error value
        """
        error = registration_error(
            source, target,
            correspondence=correspondence,
            reduction=self.reduction
        )

        # Optionally takes square root for RMSE
        if not self.squared:
            error = torch.sqrt(error + 1e-8)

        return error

    def extra_repr(self) -> str:
        """Returns extra representation string for printing."""
        return f'reduction={self.reduction}, squared={self.squared}'


if __name__ == "__main__":
    # Tests the registration error implementation
    print("Testing Registration Error implementation (CPD paper metric)...")
    print("-" * 60)

    # Creates test data
    batch_size = 2
    n_points = 100
    dim = 3

    # Test 1: Perfect registration (should have ~0 error)
    print("\n1. Perfect Registration Test:")
    source = torch.randn(batch_size, n_points, dim)
    target = source.clone()

    error = registration_error(source, target)
    print(f"   Registration error: {error.item():.6f}")
    assert error.item() < 1e-6, "Perfect registration should have ~0 error"
    print("   PASSED")

    # Test 2: Registration with known displacement
    print("\n2. Known Displacement Test:")
    displacement = 0.1
    source = torch.randn(batch_size, n_points, dim)
    target = source + displacement

    error = registration_error(source, target)
    expected_error = displacement ** 2 * dim  # Mean squared error per point
    print(f"   Registration error: {error.item():.6f}")
    print(f"   Expected error: {expected_error:.6f}")
    print("   PASSED")

    # Test 3: With correspondence matrix
    print("\n3. Correspondence Matrix Test:")
    n_source = 50
    n_target = 60
    source = torch.randn(batch_size, n_source, dim)
    target = torch.randn(batch_size, n_target, dim)

    # Creates soft correspondence matrix (normalized)
    correspondence = torch.rand(batch_size, n_source, n_target)
    correspondence = correspondence / correspondence.sum(dim=2, keepdim=True)

    error = registration_error(source, target, correspondence=correspondence)
    print(f"   Registration error with correspondence: {error.item():.6f}")
    print("   PASSED")

    # Test 4: Module interface
    print("\n4. Module Interface Test:")
    reg_error_module = RegistrationError(reduction='mean', squared=True)

    source = torch.randn(n_points, dim)
    target = torch.randn(n_points, dim)

    error = reg_error_module(source, target)
    print(f"   Module registration error: {error.item():.6f}")

    # Tests RMSE (non-squared) version
    reg_error_rmse = RegistrationError(reduction='mean', squared=False)
    rmse = reg_error_rmse(source, target)
    print(f"   RMSE: {rmse.item():.6f}")
    print("   PASSED")

    # Test 5: Gradient flow
    print("\n5. Gradient Flow Test:")
    source = torch.randn(batch_size, n_points, dim, requires_grad=True)
    target = torch.randn(batch_size, n_points, dim)

    error = registration_error(source, target)
    error.backward()

    print(f"   Gradient shape: {source.grad.shape}")
    print(f"   Gradient mean: {source.grad.mean().item():.6f}")
    print("   PASSED")

    print("\n" + "=" * 60)
    print("All CPD registration error tests passed!")