"""
Generator network for point cloud deformation.

Implements the generator component that takes a source point cloud and
latent deformation code to produce a registered/deformed point cloud.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PointCloudGenerator(nn.Module):
    """
    Generator network for deforming point clouds.
    
    Takes a source point cloud and latent deformation code, then
    generates per-point displacements to create the registered output.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        point_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 512, 512, 256),
        use_residual: bool = True,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        output_activation: Optional[str] = None
    ):
        """
        Initializes the generator network.
        
        Args:
            latent_dim: Dimension of latent deformation code
            point_dim: Dimension of each point (3 for 3D)
            hidden_dims: Dimensions of hidden layers
            use_residual: Whether to add residual connection
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            output_activation: Activation for output ('tanh', 'sigmoid', or None)
        """
        super(PointCloudGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.point_dim = point_dim
        self.use_residual = use_residual
        
        # Input dimension is point coordinates + latent code
        input_dim = point_dim + latent_dim
        
        # Builds the MLP for processing each point
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Adds linear layer
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            # Adds batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Adds activation
            layers.append(nn.ReLU())
            
            # Adds dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_dim = hidden_dim
        
        # Final layer outputs displacement/position
        layers.append(nn.Linear(hidden_dims[-1], point_dim))
        
        # Adds output activation if specified
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        # Creates the generator network
        self.generator = nn.Sequential(*layers)
        
        # Optional: Global latent processing
        # Processes latent code before concatenating with points
        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(
        self,
        points: torch.Tensor,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates deformed point cloud from source and latent code.
        
        Args:
            points: Source point cloud (B, N, 3) or (N, 3)
            latent: Latent deformation code (B, latent_dim) or (latent_dim,)
        
        Returns:
            Deformed point cloud of same shape as input
        """
        # Handles single point cloud
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        
        batch_size = points.shape[0]
        n_points = points.shape[1]
        
        # Processes latent code
        latent_processed = self.latent_processor(latent)
        
        # Expands latent code to match number of points
        # From (B, latent_dim) to (B, N, latent_dim)
        latent_expanded = latent_processed.unsqueeze(1).expand(
            batch_size, n_points, self.latent_dim
        )
        
        # Concatenates point coordinates with latent code
        # Each point gets the same latent code but different coordinates
        point_features = torch.cat([points, latent_expanded], dim=-1)
        
        # Reshapes for batch processing
        point_features_flat = point_features.view(-1, self.point_dim + self.latent_dim)
        
        # Generates displacement or new positions
        output_flat = self.generator(point_features_flat)
        
        # Reshapes back to point cloud
        output = output_flat.view(batch_size, n_points, self.point_dim)
        
        # Applies residual connection if specified
        # Output is displacement that gets added to original points
        if self.use_residual:
            output = points + output
        
        # Removes batch dimension if needed
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


class AttentionGenerator(nn.Module):
    """
    Generator with attention mechanism for context-aware deformation.
    
    Uses self-attention to allow points to influence each other's
    deformation based on the latent code.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        point_dim: int = 3,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        use_residual: bool = True
    ):
        """
        Initializes the attention-based generator.
        
        Args:
            latent_dim: Dimension of latent code
            point_dim: Dimension of points
            hidden_dim: Hidden dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            use_residual: Whether to use residual connection
        """
        super(AttentionGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
        # Projects points and latent to hidden dimension
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Creates transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, point_dim)
        )
    
    def forward(
        self,
        points: torch.Tensor,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates deformed points using attention mechanism.
        
        Args:
            points: Source point cloud (B, N, 3)
            latent: Latent code (B, latent_dim)
        
        Returns:
            Deformed point cloud
        """
        # Handles dimensions
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        
        batch_size = points.shape[0]
        n_points = points.shape[1]
        
        # Projects points to hidden dimension
        point_features = self.point_proj(points)
        
        # Projects and expands latent code
        latent_features = self.latent_proj(latent)
        latent_features = latent_features.unsqueeze(1)
        
        # Adds latent features to each point
        # This conditions the attention on the deformation code
        point_features = point_features + latent_features
        
        # Applies transformer
        transformed = self.transformer(point_features)
        
        # Projects to output dimension
        output = self.output_proj(transformed)
        
        # Applies residual connection
        if self.use_residual:
            output = points + output
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


def create_generator(
    generator_type: str = 'mlp',
    latent_dim: int = 128,
    **kwargs
) -> nn.Module:
    """
    Factory function to create generator variants.
    
    Args:
        generator_type: Type of generator ('mlp' or 'attention')
        latent_dim: Dimension of latent space
        **kwargs: Additional arguments for the generator
    
    Returns:
        Generator module
    """
    if generator_type == 'mlp':
        return PointCloudGenerator(latent_dim=latent_dim, **kwargs)
    elif generator_type == 'attention':
        return AttentionGenerator(latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


if __name__ == "__main__":
    # Tests the generator
    print("Testing Generator networks...")
    
    # Test parameters
    batch_size = 4
    n_points = 1000
    latent_dim = 128
    
    # Creates test data
    source_points = torch.randn(batch_size, n_points, 3)
    latent_code = torch.randn(batch_size, latent_dim)
    
    # Tests MLP generator
    print("\n1. Testing MLP Generator:")
    generator = PointCloudGenerator(
        latent_dim=latent_dim,
        use_residual=True
    )
    
    output = generator(source_points, latent_code)
    print(f"Source shape: {source_points.shape}")
    print(f"Latent shape: {latent_code.shape}")
    print(f"Output shape: {output.shape}")
    
    # Computes deformation statistics
    deformation = output - source_points
    print(f"Mean deformation: {deformation.mean().item():.4f}")
    print(f"Std deformation: {deformation.std().item():.4f}")
    
    # Tests without residual
    print("\n2. Testing without residual connection:")
    generator_no_res = PointCloudGenerator(
        latent_dim=latent_dim,
        use_residual=False
    )
    
    output_no_res = generator_no_res(source_points, latent_code)
    print(f"Output mean: {output_no_res.mean().item():.4f}")
    print(f"Output std: {output_no_res.std().item():.4f}")
    
    # Tests attention generator
    print("\n3. Testing Attention Generator:")
    attention_gen = AttentionGenerator(
        latent_dim=latent_dim,
        n_heads=4,
        n_layers=2
    )
    
    output_attn = attention_gen(source_points, latent_code)
    print(f"Attention output shape: {output_attn.shape}")
    
    # Tests gradient flow
    loss = output.sum()
    loss.backward()
    print("\nGradient check passed!")
    
    # Counts parameters
    n_params_mlp = sum(p.numel() for p in generator.parameters())
    n_params_attn = sum(p.numel() for p in attention_gen.parameters())
    print(f"\nMLP Generator parameters: {n_params_mlp:,}")
    print(f"Attention Generator parameters: {n_params_attn:,}")
    
    # Tests single sample
    print("\n4. Testing single sample:")
    single_points = torch.randn(500, 3)
    single_latent = torch.randn(latent_dim)
    single_output = generator(single_points, single_latent)
    print(f"Single output shape: {single_output.shape}")
    
    print("\nAll Generator tests passed!")