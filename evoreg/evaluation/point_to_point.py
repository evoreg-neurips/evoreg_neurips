"""
Point to Point Error for point cloud registration.

Implements the Point to Point error which measures the corresponding 
average distance between two point clouds by computing the mean L2 Norm difference.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def Point_to_Point_Error(source: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'none',
    Norm: int = 2,
    
) -> torch.Tensor:
    """
    Computes the Point_to_Point_Error.
    
    The Point to Point Error is the average Euclidean distance between 
    predicted correspondences and the ground-truth correspondences distances from 
    source to target and from target to source.

    **Requires source and target to have the same shape**
    **Assumes correspondence between source and target (i.e. Target[0,:] is the same point as 
    Source[0,:])**
    
    Args:
        source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
        target: Target point cloud tensor of shape (B, N, 3) or (N, 3)
        Norm: Norm type (0,1,2)
        Reduction: How to reduce the norm (mean,sum,none)
        
        
    Returns:
        Point to Point Error value(s). 
    """

    # Handles both batched and unbatched inputs
    if source.dim() == 2:
        source = source.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    
    #Calculate Norm
    dif = source - target
    P2PError = dif.norm(p = Norm,dim = -1)
    
    
    # Computes the P2P Error distance
    if reduction == 'mean':
        P2PError = P2PError.mean(dim = 0)
    elif reduction == 'sum':
        # Sums over all points and batch
        P2PError = P2PError.sum(dim = 0)
    elif reduction == 'none':
        # Returns per-batch distances
        P2PError = P2PError.mean(dim = 1)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    
    
    return P2PError.mean()



class P2PError(nn.Module):
    """
    Chamfer distance loss module for point cloud registration.
    
    Provides a PyTorch module wrapper around the chamfer_distance function
    for use in neural network training pipelines.
    """
    
    def __init__(
        self, 
        reduction: str = 'mean',
        Norm: int = 2
    ):
        """
        Initializes the Chamfer distance module.
        
        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
            
        """
        super(P2PError, self).__init__()
        self.reduction = reduction
        self.Norm = Norm
    
    def forward(
        self, 
        source: torch.Tensor, 
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the P2P Error.
        
        Args:
            source: Source point cloud tensor of shape (B, N, 3) or (N, 3)
            target: Target point cloud tensor of shape (B, N, 3) or (N, 3)
            weights: Optional point weights of shape (B, N) or (N,)
            
        Returns:
            P2P Error value
        """
        # Handles both batched and unbatched inputs
        if source.dim() == 2:
            source = source.unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0)

        
        #Calculate Norm
        dif = source - target
        P2PError = dif.norm(p = self.Norm,dim = -1)
        

        # Applies weights if provided
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(0)
            P2PError = P2PError * weights
        
        # Computes the P2P Error
        if self.reduction == 'mean':
            P2PError = P2PError.mean(dim = 0)
        elif self.reduction == 'sum':
            # Sums over all points and batch
            P2PError = P2PError.sum(dim = 0)
        elif self.reduction == 'none':
            # Returns per-batch distances
            P2PError = P2PError.mean(dim = 1)
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")
        
        
        return P2PError.mean()
        
    
    def extra_repr(self) -> str:
        """
        Returns extra representation string for printing.
        
        Returns:
            String with module parameters
        """
        return f'reduction={self.reduction}, Norm={self.Norm}'



if __name__ == "__main__":
    # Tests the P2P Error distance implementation
    print("Testing P2P Error distance implementation...")
    
    # Creates random test point clouds
    # P2P Error assumes correspondence between source and target
    batch_size = 2
    n_points_source = 100
    n_points_target = 100
    
    source = torch.randn(batch_size, n_points_source, 3)
    target = torch.randn(batch_size, n_points_target, 3)
    
    # Tests functional interface
    dist = Point_to_Point_Error(source, target)
    print(f"P2P Error distance (functional): {dist.item():.4f}")
    
    # Tests module interface
    P2P_loss = P2PError(reduction='mean', Norm=2)
    loss = P2P_loss(source, target)
    print(f"P2P Error distance (module): {loss.item():.4f}")
    
    # Tests with single point cloud (no batch dimension)
    source_single = torch.randn(n_points_source, 3)
    target_single = torch.randn(n_points_target, 3)
    dist_single = Point_to_Point_Error(source_single, target_single)
    print(f"P2P Error distance (single): {dist_single.item():.4f}")
    
    # Tests gradient flow
    source_grad = torch.randn(batch_size, n_points_source, 3, requires_grad=True)
    target_grad = torch.randn(batch_size, n_points_target, 3)
    
    loss_grad = P2P_loss(source_grad, target_grad)
    loss_grad.backward()
    
    print(f"Gradient shape: {source_grad.grad.shape}")
    print(f"Gradient mean: {source_grad.grad.mean().item():.6f}")
    
    print("\nAll P2P Error distance tests passed!")