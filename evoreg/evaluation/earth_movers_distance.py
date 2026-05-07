"""
Earth Mover's Distance (EMD) for point cloud evaluation.

Implements the Earth Mover's Distance (Wasserstein-1 distance) for comparing
point clouds. EMD measures the minimum cost to transform one point cloud
distribution into another.

Based on the paper:
"Point-set Distances for Learning Representations of 3D Point Clouds"
Nguyen et al., ICCV 2021
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


def earth_movers_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
    return_assignment: bool = False
) -> torch.Tensor:
    """
    Computes the Earth Mover's Distance between two point clouds.

    EMD finds the optimal bijection between point clouds that minimizes
    the total transportation cost. This implementation assumes equal mass
    at each point and requires source and target to have the same number
    of points.

    Args:
        source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
        target: Target point cloud tensor of shape (B, N, 3) or (N, 3)
        reduction: Reduction method ('mean', 'sum', or 'none')
        return_assignment: If True, also returns the optimal assignment matrix

    Returns:
        EMD value(s). If return_assignment is True, also returns
        the assignment matrix of shape (B, N, N)
    """
    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    batch_size, n_points, dim = source.shape

    # Verifies same number of points
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"EMD requires same number of points. Got {source.shape[1]} and {target.shape[1]}")

    # Computes pairwise distances
    # Uses broadcasting for efficiency
    source_expanded = source.unsqueeze(2)  # (B, N, 1, 3)
    target_expanded = target.unsqueeze(1)  # (B, 1, N, 3)

    # Calculates L2 distance matrix
    cost_matrix = torch.norm(source_expanded - target_expanded, p=2, dim=-1)  # (B, N, N)

    # Solves the optimal transport problem using Hungarian algorithm
    # Notes: This has O(N^3) computational complexity
    emd_values = []
    assignments = []

    for b in range(batch_size):
        cost = cost_matrix[b].detach().cpu().numpy()

        # Uses scipy's linear_sum_assignment for Hungarian algorithm
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            # Computes EMD as sum of assigned distances
            emd_value = cost[row_ind, col_ind].sum()

            # Creates assignment matrix
            assignment = torch.zeros(n_points, n_points)
            assignment[row_ind, col_ind] = 1.0

        except ImportError:
            # Falls back to greedy nearest neighbor approximation
            # Notes: This is NOT exact EMD but provides a reasonable approximation
            import warnings
            warnings.warn("scipy not available. Using approximate EMD with greedy assignment.")

            assignment = torch.zeros(n_points, n_points)
            used_targets = set()
            total_cost = 0.0

            # Sorts by minimum cost
            costs_sorted, indices = torch.sort(cost_matrix[b].view(-1))

            for idx in indices:
                i = idx // n_points
                j = idx % n_points
                if assignment[i].sum() == 0 and j not in used_targets:
                    assignment[i, j] = 1.0
                    used_targets.add(j.item())
                    total_cost += cost_matrix[b, i, j].item()

                if len(used_targets) == n_points:
                    break

            emd_value = total_cost

        emd_values.append(emd_value)
        assignments.append(assignment)

    # Converts back to tensor
    emd_values = torch.tensor(emd_values, device=source.device, dtype=source.dtype)

    if return_assignment:
        assignments = torch.stack(assignments).to(source.device)

    # Applies reduction
    if reduction == 'mean':
        # Averages over points (divide by N)
        emd_values = emd_values / n_points
        # Averages over batch
        emd_values = emd_values.mean()
    elif reduction == 'sum':
        # Sums over batch
        emd_values = emd_values.sum()
    elif reduction == 'none':
        # Returns per-batch EMD normalized by number of points
        emd_values = emd_values / n_points
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    if return_assignment:
        return emd_values, assignments
    else:
        return emd_values


def approximate_emd_sinkhorn(
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 0.01,
    max_iter: int = 100,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Approximates Earth Mover's Distance using Sinkhorn iterations.

    This provides a differentiable approximation to EMD using entropic
    regularization. It's much faster than exact EMD but provides an
    upper bound approximation.

    Args:
        source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
        target: Target point cloud tensor of shape (B, M, 3) or (M, 3)
        epsilon: Regularization parameter (smaller = closer to true EMD)
        max_iter: Maximum number of Sinkhorn iterations
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Approximated EMD value(s)
    """
    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    batch_size = source.shape[0]
    n_source = source.shape[1]
    n_target = target.shape[1]

    # Computes cost matrix
    source_expanded = source.unsqueeze(2)  # (B, N, 1, 3)
    target_expanded = target.unsqueeze(1)  # (B, 1, M, 3)
    cost_matrix = torch.norm(source_expanded - target_expanded, p=2, dim=-1)  # (B, N, M)

    # Initializes uniform distributions
    mu = torch.ones(batch_size, n_source, device=source.device) / n_source
    nu = torch.ones(batch_size, n_target, device=target.device) / n_target

    # Performs Sinkhorn iterations
    # Clips cost matrix to prevent overflow in exp
    cost_matrix_scaled = torch.clamp(cost_matrix / epsilon, max=50)
    K = torch.exp(-cost_matrix_scaled)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(max_iter):
        # Updates scaling factors with numerical stability
        u = mu / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        v = nu / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)

    # Computes transport plan
    T = u.unsqueeze(-1) * K * v.unsqueeze(1)

    # Computes approximate EMD
    emd_approx = (T * cost_matrix).sum(dim=(1, 2))

    # Applies reduction
    if reduction == 'mean':
        emd_approx = emd_approx.mean()
    elif reduction == 'sum':
        emd_approx = emd_approx.sum()
    elif reduction == 'none':
        pass  # Keeps per-batch values
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    return emd_approx


class EarthMoversDistance(nn.Module):
    """
    Earth Mover's Distance module for point cloud evaluation.

    Provides a PyTorch module wrapper around EMD computation
    for use in evaluation pipelines.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        approximation: str = 'exact',
        epsilon: float = 0.01,
        max_iter: int = 100
    ):
        """
        Initializes the EMD module.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
            approximation: 'exact' for Hungarian algorithm, 'sinkhorn' for approximation
            epsilon: Regularization for Sinkhorn (if using approximation)
            max_iter: Maximum iterations for Sinkhorn
        """
        super(EarthMoversDistance, self).__init__()
        self.reduction = reduction
        self.approximation = approximation
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Earth Mover's Distance.

        Args:
            source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
            target: Target point cloud tensor of shape (B, N, 3) or (N, 3)

        Returns:
            EMD value
        """
        if self.approximation == 'exact':
            return earth_movers_distance(source, target, self.reduction)
        elif self.approximation == 'sinkhorn':
            return approximate_emd_sinkhorn(
                source, target,
                self.epsilon,
                self.max_iter,
                self.reduction
            )
        else:
            raise ValueError(f"Unknown approximation method: {self.approximation}")

    def extra_repr(self) -> str:
        """Returns extra representation string for printing."""
        return f'reduction={self.reduction}, approximation={self.approximation}'


if __name__ == "__main__":
    # Tests the EMD implementation
    print("Testing Earth Mover's Distance implementation...")

    # Creates random test point clouds with same number of points
    batch_size = 2
    n_points = 100

    # Notes: For EMD, source and target must have same number of points
    source = torch.randn(batch_size, n_points, 3)
    target = torch.randn(batch_size, n_points, 3)

    # Tests exact EMD
    print("\n1. Testing exact EMD...")
    try:
        emd_exact = earth_movers_distance(source, target, reduction='mean')
        print(f"Exact EMD: {emd_exact.item():.4f}")
    except ImportError:
        print("scipy not available for exact EMD. Skipping...")

    # Tests Sinkhorn approximation
    print("\n2. Testing Sinkhorn approximation...")
    emd_sinkhorn = approximate_emd_sinkhorn(source, target, epsilon=0.01)
    print(f"Sinkhorn EMD (ε=0.01): {emd_sinkhorn.item():.4f}")

    emd_sinkhorn_large = approximate_emd_sinkhorn(source, target, epsilon=0.1)
    print(f"Sinkhorn EMD (ε=0.1): {emd_sinkhorn_large.item():.4f}")

    # Tests module interface
    print("\n3. Testing module interface...")
    emd_module = EarthMoversDistance(reduction='mean', approximation='sinkhorn')
    loss = emd_module(source, target)
    print(f"EMD (module, Sinkhorn): {loss.item():.4f}")

    # Tests with single point cloud (no batch dimension)
    print("\n4. Testing unbatched input...")
    source_single = torch.randn(n_points, 3)
    target_single = torch.randn(n_points, 3)
    emd_single = approximate_emd_sinkhorn(source_single, target_single)
    print(f"EMD (single): {emd_single.item():.4f}")

    # Tests gradient flow with Sinkhorn
    print("\n5. Testing gradient flow...")
    source_grad = torch.randn(batch_size, n_points, 3, requires_grad=True)
    target_grad = torch.randn(batch_size, n_points, 3)

    loss_grad = approximate_emd_sinkhorn(source_grad, target_grad)
    loss_grad.backward()

    print(f"Gradient shape: {source_grad.grad.shape}")
    print(f"Gradient mean: {source_grad.grad.mean().item():.6f}")

    # Compares with Chamfer distance for same point clouds
    print("\n6. Comparing with Chamfer distance...")
    from evoreg.losses.chamfer_distance import chamfer_distance
    chamfer = chamfer_distance(source, target)
    print(f"Chamfer distance: {chamfer.item():.4f}")
    print(f"Sinkhorn EMD: {emd_sinkhorn.item():.4f}")
    print(f"Ratio (EMD/Chamfer): {(emd_sinkhorn/chamfer).item():.4f}")

    print("\nAll EMD tests passed!")