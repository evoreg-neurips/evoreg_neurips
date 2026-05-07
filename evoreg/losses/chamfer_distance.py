"""
Chamfer distance loss for point cloud registration.

Implements the Chamfer distance metric which measures the similarity between
two point clouds by computing nearest neighbor distances in both directions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def chamfer_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
    return_indices: bool = False
) -> torch.Tensor:
    """
    Computes the Chamfer distance between two point clouds.
    
    The Chamfer distance is the sum of the average nearest neighbor
    distances from source to target and from target to source.
    
    Args:
        source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
        target: Target point cloud tensor of shape (B, M, 3) or (M, 3)
        reduction: Reduction method ('mean', 'sum', or 'none')
        return_indices: If True, also returns nearest neighbor indices
        
    Returns:
        Chamfer distance value(s). If return_indices is True, also returns
        tuple of (source_to_target_indices, target_to_source_indices)
    """
    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    
    batch_size = source.shape[0]
    n_source = source.shape[1]
    n_target = target.shape[1]
    
    # Computes pairwise squared distances using broadcasting
    # Expands dimensions for broadcasting
    source_expanded = source.unsqueeze(2)  # (B, N, 1, 3)
    target_expanded = target.unsqueeze(1)  # (B, 1, M, 3)
    
    # Calculates squared Euclidean distances
    distances = torch.sum((source_expanded - target_expanded) ** 2, dim=-1)  # (B, N, M)
    
    # Finds nearest neighbors from source to target
    source_to_target_dists, source_to_target_idx = torch.min(distances, dim=2)  # (B, N)
    
    # Finds nearest neighbors from target to source
    target_to_source_dists, target_to_source_idx = torch.min(distances, dim=1)  # (B, M)
    
    # Computes the Chamfer distance
    if reduction == 'mean':
        # Averages over points and batch
        chamfer_dist = (torch.mean(source_to_target_dists) + 
                       torch.mean(target_to_source_dists))
    elif reduction == 'sum':
        # Sums over all points and batch
        chamfer_dist = (torch.sum(source_to_target_dists) + 
                       torch.sum(target_to_source_dists))
    elif reduction == 'none':
        # Returns per-batch distances
        chamfer_dist = (torch.mean(source_to_target_dists, dim=1) + 
                       torch.mean(target_to_source_dists, dim=1))
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    
    if return_indices:
        return chamfer_dist, (source_to_target_idx, target_to_source_idx)
    else:
        return chamfer_dist


class ChamferDistance(nn.Module):
    """
    Chamfer distance loss module for point cloud registration.
    
    Provides a PyTorch module wrapper around the chamfer_distance function
    for use in neural network training pipelines.
    """
    
    def __init__(
        self, 
        reduction: str = 'mean',
        squared: bool = False,
        symmetric: bool = True
    ):
        """
        Initializes the Chamfer distance module.
        
        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
            squared: If True, returns squared distances (no square root)
            symmetric: If True, computes bidirectional distance
        """
        super(ChamferDistance, self).__init__()
        self.reduction = reduction
        self.squared = squared
        self.symmetric = symmetric
    
    def forward(
        self, 
        source: torch.Tensor, 
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the Chamfer distance loss.
        
        Args:
            source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
            target: Target point cloud tensor of shape (B, M, 3) or (M, 3)
            weights: Optional point weights of shape (B, N) or (N,)
            
        Returns:
            Chamfer distance loss value
        """
        # Handles both batched and unbatched inputs
        if source.dim() == 2:
            source = source.unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0)
        
        batch_size = source.shape[0]
        n_source = source.shape[1]
        n_target = target.shape[1]
        
        # Computes pairwise squared distances
        source_expanded = source.unsqueeze(2)  # (B, N, 1, 3)
        target_expanded = target.unsqueeze(1)  # (B, 1, M, 3)
        
        # Calculates squared Euclidean distances
        sq_distances = torch.sum((source_expanded - target_expanded) ** 2, dim=-1)
        
        # Applies square root if not using squared distances
        if not self.squared:
            distances = torch.sqrt(sq_distances + 1e-8)  # Add epsilon for stability
        else:
            distances = sq_distances
        
        # Computes forward distance (source to target)
        source_to_target_dists, _ = torch.min(distances, dim=2)  # (B, N)
        
        # Applies weights if provided
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(0)
            source_to_target_dists = source_to_target_dists * weights
        
        # Computes forward loss component
        if self.reduction == 'mean':
            forward_loss = torch.mean(source_to_target_dists)
        elif self.reduction == 'sum':
            forward_loss = torch.sum(source_to_target_dists)
        elif self.reduction == 'none':
            forward_loss = torch.mean(source_to_target_dists, dim=1)
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")
        
        # Adds backward distance if symmetric
        if self.symmetric:
            # Computes backward distance (target to source)
            target_to_source_dists, _ = torch.min(distances, dim=1)  # (B, M)
            
            # Computes backward loss component
            if self.reduction == 'mean':
                backward_loss = torch.mean(target_to_source_dists)
            elif self.reduction == 'sum':
                backward_loss = torch.sum(target_to_source_dists)
            elif self.reduction == 'none':
                backward_loss = torch.mean(target_to_source_dists, dim=1)
            
            # Combines forward and backward losses
            total_loss = forward_loss + backward_loss
        else:
            # Uses only forward distance
            total_loss = forward_loss
        
        return total_loss
    
    def extra_repr(self) -> str:
        """
        Returns extra representation string for printing.
        
        Returns:
            String with module parameters
        """
        return f'reduction={self.reduction}, squared={self.squared}, symmetric={self.symmetric}'


def chamfer_distance_with_normals(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_normals: Optional[torch.Tensor] = None,
    target_normals: Optional[torch.Tensor] = None,
    normal_weight: float = 0.1
) -> torch.Tensor:
    """
    Computes Chamfer distance with optional normal consistency term.
    
    Extends standard Chamfer distance with a term that penalizes
    misalignment of surface normals between nearest neighbors.
    
    Args:
        source_points: Source point cloud (B, N, 3) or (N, 3)
        target_points: Target point cloud (B, M, 3) or (M, 3)
        source_normals: Source normals (B, N, 3) or (N, 3)
        target_normals: Target normals (B, M, 3) or (M, 3)
        normal_weight: Weight for normal consistency term
        
    Returns:
        Combined Chamfer distance and normal consistency loss
    """
    # Computes standard Chamfer distance and gets nearest neighbor indices
    chamfer_dist, (s2t_idx, t2s_idx) = chamfer_distance(
        source_points, target_points, return_indices=True
    )
    
    # Adds normal consistency if normals are provided
    if source_normals is not None and target_normals is not None:
        # Ensures proper dimensions
        if source_normals.dim() == 2:
            source_normals = source_normals.unsqueeze(0)
        if target_normals.dim() == 2:
            target_normals = target_normals.unsqueeze(0)
        
        batch_size = source_points.shape[0] if source_points.dim() == 3 else 1
        
        # Gathers corresponding normals using nearest neighbor indices
        # For source to target
        target_normals_expanded = target_normals.unsqueeze(1).expand(-1, s2t_idx.shape[1], -1, -1)
        s2t_idx_expanded = s2t_idx.unsqueeze(-1).expand(-1, -1, 3)
        corresponding_target_normals = torch.gather(target_normals_expanded, 2, s2t_idx_expanded.unsqueeze(2)).squeeze(2)
        
        # Computes normal consistency as 1 - dot product (0 for aligned, 2 for opposite)
        normal_consistency_s2t = 1.0 - torch.sum(source_normals * corresponding_target_normals, dim=-1)
        
        # For target to source
        source_normals_expanded = source_normals.unsqueeze(1).expand(-1, t2s_idx.shape[1], -1, -1)
        t2s_idx_expanded = t2s_idx.unsqueeze(-1).expand(-1, -1, 3)
        corresponding_source_normals = torch.gather(source_normals_expanded, 2, t2s_idx_expanded.unsqueeze(2)).squeeze(2)
        
        normal_consistency_t2s = 1.0 - torch.sum(target_normals * corresponding_source_normals, dim=-1)
        
        # Combines normal consistency losses
        normal_loss = (torch.mean(normal_consistency_s2t) + 
                      torch.mean(normal_consistency_t2s)) * normal_weight
        
        # Adds to Chamfer distance
        total_loss = chamfer_dist + normal_loss
    else:
        total_loss = chamfer_dist
    
    return total_loss


if __name__ == "__main__":
    # Tests the Chamfer distance implementation
    print("Testing Chamfer distance implementation...")
    
    # Creates random test point clouds
    batch_size = 2
    n_points_source = 100
    n_points_target = 150
    
    source = torch.randn(batch_size, n_points_source, 3)
    target = torch.randn(batch_size, n_points_target, 3)
    
    # Tests functional interface
    dist = chamfer_distance(source, target)
    print(f"Chamfer distance (functional): {dist.item():.4f}")
    
    # Tests module interface
    chamfer_loss = ChamferDistance(reduction='mean', squared=False, symmetric=True)
    loss = chamfer_loss(source, target)
    print(f"Chamfer distance (module): {loss.item():.4f}")
    
    # Tests with single point cloud (no batch dimension)
    source_single = torch.randn(n_points_source, 3)
    target_single = torch.randn(n_points_target, 3)
    dist_single = chamfer_distance(source_single, target_single)
    print(f"Chamfer distance (single): {dist_single.item():.4f}")
    
    # Tests gradient flow
    source_grad = torch.randn(batch_size, n_points_source, 3, requires_grad=True)
    target_grad = torch.randn(batch_size, n_points_target, 3)
    
    loss_grad = chamfer_loss(source_grad, target_grad)
    loss_grad.backward()
    
    print(f"Gradient shape: {source_grad.grad.shape}")
    print(f"Gradient mean: {source_grad.grad.mean().item():.6f}")
    
    print("\nAll Chamfer distance tests passed!")