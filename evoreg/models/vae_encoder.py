"""
VAE encoder for learning latent deformation codes.

Implements the variational autoencoder component that maps combined
source-target features to a latent distribution for deformation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .pointnet_encoder import PointNetEncoder


class VAEEncoder(nn.Module):
    """
    VAE encoder that learns latent deformation distribution.
    
    Takes combined features from source and target point clouds and
    outputs parameters for a Gaussian distribution in latent space.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: Tuple[int, ...] = (512, 256),
        latent_dim: int = 128,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initializes the VAE encoder.
        
        Args:
            input_dim: Input dimension (combined features)
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of latent space
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
        """
        super(VAEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Builds the MLP for processing combined features
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Adds linear layer
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            # Adds batch normalization if specified
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Adds activation
            layers.append(nn.ReLU())
            
            # Adds dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_dim = hidden_dim
        
        # Creates the feature processor
        self.feature_processor = nn.Sequential(*layers)
        
        # Creates separate heads for mean and log variance
        # This is crucial for VAE - we need two outputs
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var_head = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes features to latent distribution parameters.
        
        Args:
            features: Combined features tensor of shape (B, input_dim)
                     or (input_dim,) for single sample
        
        Returns:
            Tuple of (mu, log_var) each of shape (B, latent_dim)
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        """
        # Handles single sample
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Processes features through MLP
        hidden = self.feature_processor(features)
        
        # Computes mean and log variance
        mu = self.mu_head(hidden)
        log_var = self.log_var_head(hidden)
        
        # Clamps log variance for numerical stability
        # Prevents variance from becoming too small or too large
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # Removes batch dimension if needed
        if squeeze_output:
            mu = mu.squeeze(0)
            log_var = log_var.squeeze(0)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for backpropagation through sampling.
        
        Samples from N(mu, sigma^2) by computing mu + sigma * epsilon
        where epsilon ~ N(0, I).
        
        Args:
            mu: Mean of the distribution
            log_var: Log variance of the distribution
        
        Returns:
            Sampled latent code
        """
        # Computes standard deviation
        std = torch.exp(0.5 * log_var)
        
        # Samples epsilon from standard normal
        eps = torch.randn_like(std)
        
        # Reparameterization: z = mu + sigma * epsilon
        z = mu + std * eps
        
        return z


class ConditionalVAEEncoder(nn.Module):
    """
    Conditional VAE encoder using separate point cloud encoders.
    
    Processes source and target point clouds separately before
    combining their features for latent distribution learning.
    """
    
    def __init__(
        self,
        point_encoder: Optional[nn.Module] = None,
        feature_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (512, 256),
        latent_dim: int = 128,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        share_encoder: bool = True
    ):
        """
        Initializes the conditional VAE encoder.
        
        Args:
            point_encoder: PointNet encoder for processing point clouds
            feature_dim: Dimension of point cloud features
            hidden_dims: Dimensions of hidden layers after combining
            latent_dim: Dimension of latent space
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            share_encoder: Whether to share encoder for source and target
        """
        super(ConditionalVAEEncoder, self).__init__()
        
        # Creates or uses provided point encoder
        if point_encoder is None:
            self.source_encoder = PointNetEncoder(output_dim=feature_dim)
        else:
            self.source_encoder = point_encoder
        
        # Shares encoder or creates separate one for target
        if share_encoder:
            self.target_encoder = self.source_encoder
        else:
            if point_encoder is None:
                self.target_encoder = PointNetEncoder(output_dim=feature_dim)
            else:
                # Creates a copy of the encoder architecture
                self.target_encoder = type(point_encoder)(output_dim=feature_dim)
        
        self.share_encoder = share_encoder
        
        # Creates the VAE encoder for combined features
        self.vae_encoder = VAEEncoder(
            input_dim=feature_dim * 2,  # Concatenated features
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
    
    def forward(
        self, 
        source: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes source and target point clouds to latent distribution.
        
        Args:
            source: Source point cloud (B, N, 3) or (N, 3)
            target: Target point cloud (B, M, 3) or (M, 3)
        
        Returns:
            Tuple of (mu, log_var) for the latent distribution
        """
        # Encodes source and target point clouds
        source_features = self.source_encoder(source)
        target_features = self.target_encoder(target)
        
        # Concatenates features
        combined_features = torch.cat([source_features, target_features], dim=-1)
        
        # Passes through VAE encoder
        mu, log_var = self.vae_encoder(combined_features)
        
        return mu, log_var
    
    def encode_and_sample(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes point clouds and samples from latent distribution.
        
        Args:
            source: Source point cloud
            target: Target point cloud
        
        Returns:
            Tuple of (z, mu, log_var) where z is sampled latent code
        """
        # Gets distribution parameters
        mu, log_var = self.forward(source, target)
        
        # Samples using reparameterization trick
        z = self.vae_encoder.reparameterize(mu, log_var)
        
        return z, mu, log_var


def create_vae_encoder(
    encoder_type: str = 'conditional',
    latent_dim: int = 128,
    **kwargs
) -> nn.Module:
    """
    Factory function to create VAE encoder variants.
    
    Args:
        encoder_type: Type of encoder ('simple' or 'conditional')
        latent_dim: Dimension of latent space
        **kwargs: Additional arguments for the encoder
    
    Returns:
        VAE encoder module
    """
    if encoder_type == 'simple':
        return VAEEncoder(latent_dim=latent_dim, **kwargs)
    elif encoder_type == 'conditional':
        return ConditionalVAEEncoder(latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    # Tests the VAE encoder
    print("Testing VAE encoder...")
    
    # Tests simple VAE encoder
    print("\n1. Testing simple VAE encoder:")
    vae = VAEEncoder(input_dim=1024, latent_dim=128)
    
    # Creates test combined features
    batch_size = 4
    combined_features = torch.randn(batch_size, 1024)
    
    # Encodes to distribution
    mu, log_var = vae(combined_features)
    print(f"Input shape: {combined_features.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log var shape: {log_var.shape}")
    
    # Tests reparameterization
    z = vae.reparameterize(mu, log_var)
    print(f"Sampled z shape: {z.shape}")
    print(f"Z mean: {z.mean().item():.4f}")
    print(f"Z std: {z.std().item():.4f}")
    
    # Tests conditional VAE encoder
    print("\n2. Testing conditional VAE encoder:")
    conditional_vae = ConditionalVAEEncoder(
        feature_dim=512,
        latent_dim=128
    )
    
    # Creates test point clouds
    n_points = 1000
    source = torch.randn(batch_size, n_points, 3)
    target = torch.randn(batch_size, n_points, 3)
    
    # Encodes point clouds
    mu, log_var = conditional_vae(source, target)
    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log var shape: {log_var.shape}")
    
    # Tests encode and sample
    z, mu, log_var = conditional_vae.encode_and_sample(source, target)
    print(f"Sampled z shape: {z.shape}")
    
    # Tests gradient flow
    loss = z.sum()
    loss.backward()
    print("\nGradient check passed!")
    
    # Counts parameters
    n_params = sum(p.numel() for p in conditional_vae.parameters())
    print(f"\nTotal parameters in conditional VAE: {n_params:,}")
    
    print("\nAll VAE encoder tests passed!")