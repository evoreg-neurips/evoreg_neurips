"""
PointNet encoder for processing point clouds.

Implements a simplified PointNet architecture that extracts global features
from point clouds while maintaining permutation invariance.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PointNetEncoder(nn.Module):
    """
    PointNet encoder for extracting features from point clouds.
    
    Processes point clouds through MLPs and max pooling to create
    permutation-invariant global features suitable for downstream tasks.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        output_dim: int = 1024,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initializes the PointNet encoder.
        
        Args:
            input_dim: Input dimension per point (3 for XYZ)
            hidden_dims: Dimensions of hidden layers in the MLP
            output_dim: Output feature dimension
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
        """
        super(PointNetEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Builds the point-wise MLP (processes each point independently)
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Adds linear layer
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            # Adds batch normalization if specified
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Adds ReLU activation
            layers.append(nn.ReLU())
            
            # Adds dropout if specified
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_dim = hidden_dim
        
        # Creates the point-wise feature extractor
        self.point_features = nn.Sequential(*layers)
        
        # Creates the global feature MLP (after pooling)
        self.global_features = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Processes point cloud through the encoder.
        
        Args:
            points: Input point cloud tensor of shape (B, N, D) or (N, D)
                    where B is batch size, N is number of points, D is dimension
        
        Returns:
            Global feature vector of shape (B, output_dim) or (output_dim,)
        """
        # Handles both batched and unbatched input
        if points.dim() == 2:
            # Adds batch dimension if not present
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = points.shape[0]
        n_points = points.shape[1]
        
        # Reshapes for batch normalization
        # From (B, N, D) to (B*N, D) for processing
        points_flat = points.view(-1, self.input_dim)
        
        # Applies point-wise MLP
        point_features = self.point_features(points_flat)
        
        # Reshapes back to (B, N, F) where F is feature dimension
        point_features = point_features.view(batch_size, n_points, -1)
        
        # Applies max pooling to get global features
        # Max pooling ensures permutation invariance
        global_features, _ = torch.max(point_features, dim=1)  # (B, F)
        
        # Processes global features through final MLP
        # Sets model to eval mode for single samples to avoid batch norm issues
        if batch_size == 1 and self.training:
            self.eval()
            output = self.global_features(global_features)
            self.train()
        else:
            output = self.global_features(global_features)
        
        # Removes batch dimension if input was unbatched
        if squeeze_output:
            output = output.squeeze(0)

        return output

    def forward_with_point_features(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Same as forward() but also returns per-point features before max-pool.

        Args:
            points: (B, N, D) or (N, D)

        Returns:
            global_features:    (B, output_dim)
            per_point_features: (B, N, hidden_dims[-1])
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)

        batch_size = points.shape[0]
        n_points = points.shape[1]

        points_flat = points.view(-1, self.input_dim)
        point_features = self.point_features(points_flat)
        point_features = point_features.view(batch_size, n_points, -1)  # (B, N, 1024)

        global_features, _ = torch.max(point_features, dim=1)  # (B, 1024)

        if batch_size == 1 and self.training:
            self.eval()
            output = self.global_features(global_features)
            self.train()
        else:
            output = self.global_features(global_features)

        return output, point_features

class PointNetEncoderWithAttention(nn.Module):
    """
    Enhanced PointNet encoder with self-attention mechanism.
    
    Adds attention layers to capture relationships between points
    before aggregation, allowing for more context-aware features.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
        attention_dim: int = 256,
        output_dim: int = 512,
        n_heads: int = 4,
        use_batch_norm: bool = True
    ):
        """
        Initializes the PointNet encoder with attention.
        
        Args:
            input_dim: Input dimension per point
            hidden_dims: Dimensions of hidden layers before attention
            attention_dim: Dimension for attention mechanism
            output_dim: Output feature dimension
            n_heads: Number of attention heads
            use_batch_norm: Whether to use batch normalization
        """
        super(PointNetEncoderWithAttention, self).__init__()
        
        # Builds initial point-wise MLP
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.point_features = nn.Sequential(*layers)
        
        # Adds self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=n_heads,
            batch_first=True
        )
        
        # Projects to attention dimension
        self.attention_proj = nn.Linear(hidden_dims[-1], attention_dim)
        
        # Global feature processing
        self.global_features = nn.Sequential(
            nn.Linear(attention_dim, output_dim),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Processes point cloud through encoder with attention.
        
        Args:
            points: Input point cloud (B, N, D) or (N, D)
        
        Returns:
            Global feature vector (B, output_dim) or (output_dim,)
        """
        # Handles batching
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = points.shape[0]
        n_points = points.shape[1]
        
        # Processes through initial MLP
        points_flat = points.view(-1, points.shape[-1])
        point_features = self.point_features(points_flat)
        point_features = point_features.view(batch_size, n_points, -1)
        
        # Applies self-attention
        attended_features, _ = self.attention(
            point_features, point_features, point_features
        )
        
        # Projects to attention dimension
        attended_features = self.attention_proj(attended_features)
        
        # Max pools to get global features
        global_features, _ = torch.max(attended_features, dim=1)
        
        # Processes through final MLP
        output = self.global_features(global_features)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


def create_pointnet_encoder(
    input_dim: int = 3,
    output_dim: int = 1024,
    use_attention: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create PointNet encoder variants.
    
    Args:
        input_dim: Input dimension per point
        output_dim: Output feature dimension
        use_attention: Whether to use attention-enhanced version
        **kwargs: Additional arguments for the encoder
    
    Returns:
        PointNet encoder module
    """
    if use_attention:
        return PointNetEncoderWithAttention(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        return PointNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )


if __name__ == "__main__":
    # Tests the PointNet encoder
    print("Testing PointNet encoder...")
    
    # Creates test data
    batch_size = 4
    n_points = 1000
    points = torch.randn(batch_size, n_points, 3)
    
    # Tests basic encoder
    encoder = PointNetEncoder(input_dim=3, output_dim=512)
    features = encoder(points)
    print(f"Input shape: {points.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output mean: {features.mean().item():.4f}")
    print(f"Output std: {features.std().item():.4f}")
    
    # Tests with single point cloud (no batch)
    single_points = torch.randn(500, 3)
    single_features = encoder(single_points)
    print(f"\nSingle point cloud input: {single_points.shape}")
    print(f"Single point cloud output: {single_features.shape}")
    
    # Tests attention version
    encoder_attn = PointNetEncoderWithAttention(
        input_dim=3, 
        output_dim=256,
        n_heads=4
    )
    features_attn = encoder_attn(points)
    print(f"\nAttention encoder output: {features_attn.shape}")
    
    # Tests gradient flow
    loss = features.sum()
    loss.backward()
    print(f"\nGradient check passed!")
    
    # Counts parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    print("\nAll PointNet encoder tests passed!")