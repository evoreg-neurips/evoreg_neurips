"""
Sliced Wasserstein Distance (SWD) for point cloud evaluation.

Implements the Sliced Wasserstein Distance and its variants for comparing
point clouds. SWD projects point clouds onto random directions and computes
1D Wasserstein distances, providing an efficient approximation to EMD.

Based on the paper:
"Point-set Distances for Learning Representations of 3D Point Clouds"
Nguyen et al., ICCV 2021
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import numpy as np


def sliced_wasserstein_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    n_projections: int = 100,
    p: int = 2,
    reduction: str = 'mean',
    return_projections: bool = False
) -> torch.Tensor:
    """
    Computes the Sliced Wasserstein Distance between two point clouds.

    SWD averages 1D Wasserstein distances over random projections,
    providing an efficient alternative to EMD with O(N log N) complexity.

    Args:
        source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
        target: Target point cloud tensor of shape (B, M, 3) or (M, 3)
        n_projections: Number of random projections to use
        p: Order of the Wasserstein distance (1 or 2)
        reduction: Reduction method ('mean', 'sum', or 'none')
        return_projections: If True, also returns the projection directions

    Returns:
        SWD value(s). If return_projections is True, also returns
        the projection directions of shape (n_projections, 3)
    """
    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    batch_size = source.shape[0]
    n_source = source.shape[1]
    n_target = target.shape[1]
    dim = source.shape[2]

    # Generates random projection directions on unit sphere
    # Uses Gaussian sampling followed by normalization
    projections = torch.randn(n_projections, dim, device=source.device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)

    # Initializes SWD accumulator
    swd_values = torch.zeros(batch_size, device=source.device)

    for proj_idx in range(n_projections):
        # Gets current projection direction
        theta = projections[proj_idx]  # (3,)

        # Projects point clouds onto the direction
        # Computes dot product with projection vector
        source_proj = torch.matmul(source, theta)  # (B, N)
        target_proj = torch.matmul(target, theta)  # (B, M)

        # Sorts projected points for 1D Wasserstein computation
        source_sorted, _ = torch.sort(source_proj, dim=1)
        target_sorted, _ = torch.sort(target_proj, dim=1)

        # Handles different point cloud sizes via interpolation
        if n_source != n_target:
            # Interpolates to common size for comparison
            # Uses linear interpolation to match distributions
            if n_source > n_target:
                # Upsamples target to match source size
                indices = torch.linspace(0, n_target - 1, n_source, device=target.device)
                indices_floor = indices.floor().long()
                indices_ceil = (indices_floor + 1).clamp(max=n_target - 1)
                weights = indices - indices_floor.float()

                target_interp = (target_sorted[:, indices_floor] * (1 - weights) +
                                target_sorted[:, indices_ceil] * weights)
                source_interp = source_sorted
            else:
                # Upsamples source to match target size
                indices = torch.linspace(0, n_source - 1, n_target, device=source.device)
                indices_floor = indices.floor().long()
                indices_ceil = (indices_floor + 1).clamp(max=n_source - 1)
                weights = indices - indices_floor.float()

                source_interp = (source_sorted[:, indices_floor] * (1 - weights) +
                                source_sorted[:, indices_ceil] * weights)
                target_interp = target_sorted
        else:
            # Uses sorted points directly if same size
            source_interp = source_sorted
            target_interp = target_sorted

        # Computes 1D Wasserstein distance
        if p == 1:
            wd_1d = torch.abs(source_interp - target_interp).mean(dim=1)
        elif p == 2:
            wd_1d = torch.sqrt(torch.mean((source_interp - target_interp) ** 2, dim=1) + 1e-8)
        else:
            wd_1d = torch.mean(torch.abs(source_interp - target_interp) ** p, dim=1) ** (1.0 / p)

        # Accumulates the 1D Wasserstein distances
        swd_values += wd_1d

    # Averages over all projections
    swd_values = swd_values / n_projections

    # Applies reduction
    if reduction == 'mean':
        swd_values = swd_values.mean()
    elif reduction == 'sum':
        swd_values = swd_values.sum()
    elif reduction == 'none':
        pass  # Keeps per-batch values
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    if return_projections:
        return swd_values, projections
    else:
        return swd_values


def max_sliced_wasserstein_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    n_projections: int = 100,
    n_iterations: int = 10,
    p: int = 2,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes the Max-Sliced Wasserstein Distance between two point clouds.

    Max-SWD finds the projection that maximizes the Wasserstein distance,
    providing a more discriminative metric than standard SWD.

    Args:
        source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
        target: Target point cloud tensor of shape (B, M, 3) or (M, 3)
        n_projections: Number of random projections to sample
        n_iterations: Number of optimization iterations
        p: Order of the Wasserstein distance (1 or 2)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Max-SWD value(s)
    """
    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    batch_size = source.shape[0]
    n_source = source.shape[1]
    n_target = target.shape[1]
    dim = source.shape[2]

    # Initializes with random projections
    projections = torch.randn(n_projections, dim, device=source.device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)

    # Tracks maximum SWD across iterations
    max_swd = torch.zeros(batch_size, device=source.device)

    for iter_idx in range(n_iterations):
        # Computes SWD for all projections
        batch_swd = []

        for proj_idx in range(n_projections):
            theta = projections[proj_idx]

            # Projects point clouds
            source_proj = torch.matmul(source, theta)
            target_proj = torch.matmul(target, theta)

            # Sorts projected points
            source_sorted, _ = torch.sort(source_proj, dim=1)
            target_sorted, _ = torch.sort(target_proj, dim=1)

            # Handles size mismatch
            if n_source != n_target:
                if n_source > n_target:
                    indices = torch.linspace(0, n_target - 1, n_source, device=target.device)
                    indices_floor = indices.floor().long()
                    indices_ceil = (indices_floor + 1).clamp(max=n_target - 1)
                    weights = indices - indices_floor.float()

                    target_interp = (target_sorted[:, indices_floor] * (1 - weights) +
                                    target_sorted[:, indices_ceil] * weights)
                    source_interp = source_sorted
                else:
                    indices = torch.linspace(0, n_source - 1, n_target, device=source.device)
                    indices_floor = indices.floor().long()
                    indices_ceil = (indices_floor + 1).clamp(max=n_source - 1)
                    weights = indices - indices_floor.float()

                    source_interp = (source_sorted[:, indices_floor] * (1 - weights) +
                                    source_sorted[:, indices_ceil] * weights)
                    target_interp = target_sorted
            else:
                source_interp = source_sorted
                target_interp = target_sorted

            # Computes 1D Wasserstein distance
            if p == 1:
                wd_1d = torch.abs(source_interp - target_interp).mean(dim=1)
            elif p == 2:
                wd_1d = torch.sqrt(torch.mean((source_interp - target_interp) ** 2, dim=1) + 1e-8)
            else:
                wd_1d = torch.mean(torch.abs(source_interp - target_interp) ** p, dim=1) ** (1.0 / p)

            batch_swd.append(wd_1d)

        # Stacks all SWD values
        batch_swd = torch.stack(batch_swd, dim=0)  # (n_projections, B)

        # Finds maximum SWD for each batch
        current_max, max_indices = torch.max(batch_swd, dim=0)

        # Updates maximum SWD
        max_swd = torch.maximum(max_swd, current_max)

        # Refines projections around best directions if not last iteration
        if iter_idx < n_iterations - 1:
            # Generates new projections with perturbations around best ones
            projections = torch.randn(n_projections, dim, device=source.device)
            # Adds influence from best directions
            for b in range(batch_size):
                best_idx = max_indices[b].item()
                # Biases some projections toward the best one
                if b == 0:  # Uses first batch's best for refinement
                    projections[:n_projections//4] = projections[:n_projections//4] * 0.3 + projections[best_idx] * 0.7
            # Normalizes to unit sphere
            projections = projections / torch.norm(projections, dim=1, keepdim=True)

    # Applies reduction
    if reduction == 'mean':
        max_swd = max_swd.mean()
    elif reduction == 'sum':
        max_swd = max_swd.sum()
    elif reduction == 'none':
        pass  # Keeps per-batch values
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    return max_swd


def adaptive_sliced_wasserstein_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    initial_projections: int = 10,
    max_projections: int = 500,
    epsilon: float = 0.01,
    p: int = 2,
    reduction: str = 'mean'
) -> Tuple[torch.Tensor, int]:
    """
    Computes Adaptive Sliced Wasserstein Distance with automatic stopping.

    Dynamically determines the number of projections needed to achieve
    a specified accuracy, as described in the paper.

    Args:
        source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
        target: Target point cloud tensor of shape (B, M, 3) or (M, 3)
        initial_projections: Initial number of projections
        max_projections: Maximum number of projections allowed
        epsilon: Tolerance for convergence
        p: Order of the Wasserstein distance (1 or 2)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Tuple of (ASW distance, number of projections used)
    """
    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    batch_size = source.shape[0]
    n_source = source.shape[1]
    n_target = target.shape[1]
    dim = source.shape[2]

    # Initializes with initial projections
    n_proj = initial_projections
    swd_sum = torch.zeros(batch_size, device=source.device)
    swd_squared_sum = torch.zeros(batch_size, device=source.device)

    # Tracks convergence statistics
    total_projections = 0
    k = 2.0  # Corresponds to ~95% confidence interval

    while total_projections < max_projections:
        # Generates new batch of projections
        batch_size_proj = min(n_proj, max_projections - total_projections)
        projections = torch.randn(batch_size_proj, dim, device=source.device)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)

        # Computes SWD for new projections
        batch_swd = []

        for proj_idx in range(batch_size_proj):
            theta = projections[proj_idx]

            # Projects and sorts
            source_proj = torch.matmul(source, theta)
            target_proj = torch.matmul(target, theta)

            source_sorted, _ = torch.sort(source_proj, dim=1)
            target_sorted, _ = torch.sort(target_proj, dim=1)

            # Handles size mismatch
            if n_source != n_target:
                if n_source > n_target:
                    indices = torch.linspace(0, n_target - 1, n_source, device=target.device)
                    indices_floor = indices.floor().long()
                    indices_ceil = (indices_floor + 1).clamp(max=n_target - 1)
                    weights = indices - indices_floor.float()

                    target_interp = (target_sorted[:, indices_floor] * (1 - weights) +
                                    target_sorted[:, indices_ceil] * weights)
                    source_interp = source_sorted
                else:
                    indices = torch.linspace(0, n_source - 1, n_target, device=source.device)
                    indices_floor = indices.floor().long()
                    indices_ceil = (indices_floor + 1).clamp(max=n_source - 1)
                    weights = indices - indices_floor.float()

                    source_interp = (source_sorted[:, indices_floor] * (1 - weights) +
                                    source_sorted[:, indices_ceil] * weights)
                    target_interp = target_sorted
            else:
                source_interp = source_sorted
                target_interp = target_sorted

            # Computes 1D Wasserstein
            if p == 1:
                wd_1d = torch.abs(source_interp - target_interp).mean(dim=1)
            elif p == 2:
                wd_1d = torch.sqrt(torch.mean((source_interp - target_interp) ** 2, dim=1) + 1e-8)
            else:
                wd_1d = torch.mean(torch.abs(source_interp - target_interp) ** p, dim=1) ** (1.0 / p)

            batch_swd.append(wd_1d)

        # Updates running statistics
        batch_swd = torch.stack(batch_swd, dim=0)  # (batch_size_proj, B)
        swd_sum += batch_swd.sum(dim=0)
        swd_squared_sum += (batch_swd ** 2).sum(dim=0)
        total_projections += batch_size_proj

        # Computes mean and variance
        mean_swd = swd_sum / total_projections
        var_swd = (swd_squared_sum / total_projections) - (mean_swd ** 2)

        # Checks convergence criterion
        # Uses central limit theorem for confidence interval
        if total_projections >= initial_projections:
            std_error = torch.sqrt(var_swd / total_projections + 1e-8)
            confidence_width = k * std_error

            # Checks if confidence interval is within tolerance
            if torch.all(confidence_width < epsilon):
                break

        # Doubles the number of projections for next iteration
        n_proj = min(n_proj * 2, max_projections - total_projections)

    # Computes final SWD estimate
    aswd_values = swd_sum / total_projections

    # Applies reduction
    if reduction == 'mean':
        aswd_values = aswd_values.mean()
    elif reduction == 'sum':
        aswd_values = aswd_values.sum()
    elif reduction == 'none':
        pass  # Keeps per-batch values
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    return aswd_values, total_projections


class SlicedWassersteinDistance(nn.Module):
    """
    Sliced Wasserstein Distance module for point cloud evaluation.

    Provides a PyTorch module wrapper around SWD computation
    for use in evaluation pipelines.
    """

    def __init__(
        self,
        n_projections: int = 100,
        p: int = 2,
        reduction: str = 'mean',
        variant: str = 'standard'
    ):
        """
        Initializes the SWD module.

        Args:
            n_projections: Number of random projections
            p: Order of the Wasserstein distance (1 or 2)
            reduction: Reduction method ('mean', 'sum', or 'none')
            variant: SWD variant ('standard', 'max', or 'adaptive')
        """
        super(SlicedWassersteinDistance, self).__init__()
        self.n_projections = n_projections
        self.p = p
        self.reduction = reduction
        self.variant = variant

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """
        Computes the Sliced Wasserstein Distance.

        Args:
            source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
            target: Target point cloud tensor of shape (B, M, 3) or (M, 3)

        Returns:
            SWD value, or tuple of (SWD, n_projections) for adaptive variant
        """
        if self.variant == 'standard':
            return sliced_wasserstein_distance(
                source, target,
                self.n_projections,
                self.p,
                self.reduction
            )
        elif self.variant == 'max':
            return max_sliced_wasserstein_distance(
                source, target,
                self.n_projections,
                n_iterations=10,
                p=self.p,
                reduction=self.reduction
            )
        elif self.variant == 'adaptive':
            return adaptive_sliced_wasserstein_distance(
                source, target,
                initial_projections=10,
                max_projections=self.n_projections,
                epsilon=0.01,
                p=self.p,
                reduction=self.reduction
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def extra_repr(self) -> str:
        """Returns extra representation string for printing."""
        return f'n_projections={self.n_projections}, p={self.p}, variant={self.variant}'


if __name__ == "__main__":
    # Tests the SWD implementation
    print("Testing Sliced Wasserstein Distance implementation...")

    # Creates random test point clouds
    batch_size = 2
    n_points_source = 100
    n_points_target = 150

    source = torch.randn(batch_size, n_points_source, 3)
    target = torch.randn(batch_size, n_points_target, 3)

    # Tests standard SWD
    print("\n1. Testing standard SWD...")
    swd = sliced_wasserstein_distance(source, target, n_projections=100)
    print(f"Standard SWD: {swd.item():.4f}")

    # Tests with different p values
    print("\n2. Testing different p values...")
    swd_p1 = sliced_wasserstein_distance(source, target, n_projections=100, p=1)
    swd_p2 = sliced_wasserstein_distance(source, target, n_projections=100, p=2)
    print(f"SWD (p=1): {swd_p1.item():.4f}")
    print(f"SWD (p=2): {swd_p2.item():.4f}")

    # Tests max-sliced Wasserstein
    print("\n3. Testing max-sliced Wasserstein...")
    max_swd = max_sliced_wasserstein_distance(source, target, n_projections=50)
    print(f"Max-SWD: {max_swd.item():.4f}")

    # Tests adaptive SWD
    print("\n4. Testing adaptive SWD...")
    aswd, n_used = adaptive_sliced_wasserstein_distance(source, target, max_projections=200)
    print(f"Adaptive SWD: {aswd.item():.4f} (used {n_used} projections)")

    # Tests module interface
    print("\n5. Testing module interface...")
    swd_module = SlicedWassersteinDistance(n_projections=100, variant='standard')
    loss = swd_module(source, target)
    print(f"SWD (module): {loss.item():.4f}")

    # Tests with single point cloud (no batch dimension)
    print("\n6. Testing unbatched input...")
    source_single = torch.randn(n_points_source, 3)
    target_single = torch.randn(n_points_target, 3)
    swd_single = sliced_wasserstein_distance(source_single, target_single)
    print(f"SWD (single): {swd_single.item():.4f}")

    # Tests gradient flow
    print("\n7. Testing gradient flow...")
    source_grad = torch.randn(batch_size, n_points_source, 3, requires_grad=True)
    target_grad = torch.randn(batch_size, n_points_target, 3)

    loss_grad = sliced_wasserstein_distance(source_grad, target_grad)
    loss_grad.backward()

    print(f"Gradient shape: {source_grad.grad.shape}")
    print(f"Gradient mean: {source_grad.grad.mean().item():.6f}")

    # Compares all metrics
    print("\n8. Comparing different metrics...")
    from evoreg.losses.chamfer_distance import chamfer_distance
    from evoreg.evaluation.earth_movers_distance import approximate_emd_sinkhorn

    # Uses same size for EMD comparison
    source_same = torch.randn(batch_size, 100, 3)
    target_same = torch.randn(batch_size, 100, 3)

    chamfer = chamfer_distance(source_same, target_same)
    emd = approximate_emd_sinkhorn(source_same, target_same)
    swd = sliced_wasserstein_distance(source_same, target_same)

    print(f"Chamfer distance: {chamfer.item():.4f}")
    print(f"EMD (Sinkhorn): {emd.item():.4f}")
    print(f"SWD: {swd.item():.4f}")

    print("\nAll SWD tests passed!")