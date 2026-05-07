"""
Score network for diffusion-based point cloud refinement.

Learns to predict the score (gradient of log probability) at different
noise levels for iterative denoising during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps.
    
    Encodes the timestep into a high-dimensional representation
    that helps the network understand the current noise level.
    """
    
    def __init__(self, embed_dim: int = 128):
        """
        Initializes time embedding module.
        
        Args:
            embed_dim: Dimension of time embedding
        """
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        
        # Creates embedding layers
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim * 4)
        self.activation = nn.SiLU()  # Smooth activation for time embeddings
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Creates sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: Batch of timesteps (B,)
        
        Returns:
            Time embeddings (B, embed_dim * 4)
        """
        # Creates sinusoidal features (similar to positional encoding)
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        
        # Projects to higher dimension
        embeddings = self.linear1(embeddings)
        embeddings = self.activation(embeddings)
        embeddings = self.linear2(embeddings)
        
        return embeddings


class PointwiseNet(nn.Module):
    """
    Pointwise network that processes each point independently.
    
    Takes point features and global conditioning to predict
    per-point score values.
    """
    
    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        time_embed_dim: int = 512,
        n_layers: int = 4
    ):
        """
        Initializes pointwise network.
        
        Args:
            point_dim: Dimension of points (3 for 3D)
            hidden_dim: Hidden layer dimension
            time_embed_dim: Dimension of time embedding
            n_layers: Number of processing layers
        """
        super(PointwiseNet, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(point_dim * 2, hidden_dim)  # Concat noisy + target
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        
        # Processing layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Use LayerNorm instead of GroupNorm for point clouds
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, point_dim)
        )
    
    def forward(
        self,
        noisy_points: torch.Tensor,
        target_points: torch.Tensor,
        time_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts score for each point.
        
        Args:
            noisy_points: Noisy point positions (B, N, 3)
            target_points: Target point cloud (B, M, 3)
            time_embed: Time embedding (B, time_embed_dim)
        
        Returns:
            Predicted score for each point (B, N, 3)
        """
        batch_size, n_points = noisy_points.shape[:2]
        
        # Computes nearest neighbor features from target
        # For each noisy point, find its nearest target point
        distances = torch.cdist(noisy_points, target_points)  # (B, N, M)
        nearest_idx = distances.argmin(dim=-1)  # (B, N)
        
        # Gathers nearest target points
        nearest_target = torch.gather(
            target_points,
            1,
            nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, N, 3)
        
        # Concatenates noisy points with nearest target
        point_features = torch.cat([noisy_points, nearest_target], dim=-1)  # (B, N, 6)
        
        # Projects to hidden dimension
        x = self.input_proj(point_features)  # (B, N, hidden_dim)
        
        # Adds time conditioning
        time_features = self.time_proj(time_embed).unsqueeze(1)  # (B, 1, hidden_dim)
        x = x + time_features  # Broadcasts to all points
        
        # Processes through layers with residual connections
        for layer in self.layers:
            x = x + layer(x)
        
        # Projects to score
        score = self.output_proj(x)  # (B, N, 3)
        
        return score


class ScoreNetwork(nn.Module):
    """
    Complete score network for diffusion-based point cloud denoising.
    
    Combines time embedding and pointwise processing to predict
    the score (gradient) for denoising at different timesteps.
    """
    
    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        n_layers: int = 4,
        max_timesteps: int = 1000
    ):
        """
        Initializes score network.
        
        Args:
            point_dim: Dimension of points
            hidden_dim: Hidden layer dimension
            time_embed_dim: Base dimension for time embedding
            n_layers: Number of processing layers
            max_timesteps: Maximum number of diffusion timesteps
        """
        super(ScoreNetwork, self).__init__()
        
        self.point_dim = point_dim
        self.max_timesteps = max_timesteps
        
        # Time embedding module
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Pointwise score prediction
        self.pointwise_net = PointwiseNet(
            point_dim=point_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim * 4,  # After projection
            n_layers=n_layers
        )
        
        # Optional: Global context encoder for target
        self.target_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        noisy_points: torch.Tensor,
        target_points: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts score for denoising.
        
        Args:
            noisy_points: Noisy point cloud (B, N, 3)
            target_points: Target point cloud (B, M, 3)
            timesteps: Current timestep for each sample (B,)
        
        Returns:
            Score (gradient) for denoising (B, N, 3)
        """
        # Embeds timesteps
        time_embed = self.time_embedding(timesteps)
        
        # Predicts score for each point
        score = self.pointwise_net(noisy_points, target_points, time_embed)
        
        return score
    
    def denoise_step(
        self,
        noisy_points: torch.Tensor,
        target_points: torch.Tensor,
        timestep: int,
        noise_scale: float = 0.1
    ) -> torch.Tensor:
        """
        Performs one denoising step (for inference).
        
        Args:
            noisy_points: Current noisy points (B, N, 3)
            target_points: Target point cloud (B, M, 3)
            timestep: Current timestep
            noise_scale: Scale factor for the denoising step
        
        Returns:
            Denoised points (B, N, 3)
        """
        batch_size = noisy_points.shape[0]
        
        # Creates timestep tensor
        t = torch.full((batch_size,), timestep, device=noisy_points.device)
        
        # Predicts score
        score = self.forward(noisy_points, target_points, t)
        
        # Updates points in the direction of the score
        # This is a simplified Langevin dynamics step
        denoised = noisy_points + noise_scale * score
        
        # Optional: Add small noise for stochasticity (except at t=0)
        if timestep > 0:
            noise = torch.randn_like(noisy_points) * (noise_scale * 0.1)
            denoised = denoised + noise
        
        return denoised


class UNetScoreNetwork(nn.Module):
    """
    U-Net style score network with skip connections.
    
    More sophisticated architecture for complex denoising tasks.
    Includes encoder-decoder structure with skip connections.
    """
    
    def __init__(
        self,
        point_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 256, 512, 256, 128),
        time_embed_dim: int = 128
    ):
        """
        Initializes U-Net score network.
        
        Args:
            point_dim: Dimension of points
            hidden_dims: Dimensions for each layer
            time_embed_dim: Dimension of time embedding
        """
        super(UNetScoreNetwork, self).__init__()
        
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Encoder layers
        self.encoders = nn.ModuleList()
        in_dim = point_dim * 2  # Noisy + target features
        
        for hidden_dim in hidden_dims[:len(hidden_dims)//2 + 1]:
            self.encoders.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ))
            in_dim = hidden_dim
        
        # Middle layer with time conditioning
        self.middle = nn.Sequential(
            nn.Linear(hidden_dims[len(hidden_dims)//2] + time_embed_dim * 4, 
                     hidden_dims[len(hidden_dims)//2]),
            nn.LayerNorm(hidden_dims[len(hidden_dims)//2]),
            nn.SiLU()
        )
        
        # Decoder layers with skip connections
        self.decoders = nn.ModuleList()
        decoder_dims = hidden_dims[len(hidden_dims)//2 + 1:]
        
        for i, hidden_dim in enumerate(decoder_dims):
            # Account for skip connection
            skip_dim = hidden_dims[len(hidden_dims)//2 - i - 1]
            self.decoders.append(nn.Sequential(
                nn.Linear(hidden_dims[len(hidden_dims)//2] + skip_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ))
        
        # Output layer
        self.output_proj = nn.Linear(decoder_dims[-1] if decoder_dims else hidden_dims[-1], point_dim)
    
    def forward(
        self,
        noisy_points: torch.Tensor,
        target_points: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with U-Net architecture.
        
        Args:
            noisy_points: Noisy points (B, N, 3)
            target_points: Target points (B, M, 3)
            timesteps: Timesteps (B,)
        
        Returns:
            Predicted score (B, N, 3)
        """
        batch_size, n_points = noisy_points.shape[:2]
        
        # Computes nearest neighbor features
        distances = torch.cdist(noisy_points, target_points)
        nearest_idx = distances.argmin(dim=-1)
        nearest_target = torch.gather(
            target_points,
            1,
            nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        )
        
        # Initial features
        x = torch.cat([noisy_points, nearest_target], dim=-1)
        
        # Encoder with skip connections saved
        skip_connections = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skip_connections.append(x)
        
        # Last encoder
        x = self.encoders[-1](x)
        
        # Middle layer with time conditioning
        time_embed = self.time_embedding(timesteps)
        time_features = time_embed.unsqueeze(1).expand(-1, n_points, -1)
        x = torch.cat([x, time_features], dim=-1)
        x = self.middle(x)
        
        # Decoder with skip connections
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=-1)
            x = decoder(x)
        
        # Output projection
        score = self.output_proj(x)
        
        return score


if __name__ == "__main__":
    # Tests the score network
    print("Testing Score Networks...")
    
    # Test parameters
    batch_size = 2
    n_points = 1000
    m_points = 1200
    
    # Creates test data
    noisy_points = torch.randn(batch_size, n_points, 3)
    target_points = torch.randn(batch_size, m_points, 3)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # Tests basic score network
    print("\n1. Testing Basic Score Network:")
    score_net = ScoreNetwork(
        point_dim=3,
        hidden_dim=256,
        time_embed_dim=128,
        n_layers=4
    )
    
    score = score_net(noisy_points, target_points, timesteps)
    print(f"Input shapes: noisy={noisy_points.shape}, target={target_points.shape}")
    print(f"Timesteps: {timesteps}")
    print(f"Score shape: {score.shape}")
    print(f"Score stats: mean={score.mean().item():.4f}, std={score.std().item():.4f}")
    
    # Tests denoising step
    print("\n2. Testing Denoising Step:")
    denoised = score_net.denoise_step(noisy_points, target_points, timestep=500)
    print(f"Denoised shape: {denoised.shape}")
    movement = (denoised - noisy_points).norm(dim=-1).mean()
    print(f"Average point movement: {movement.item():.4f}")
    
    # Tests U-Net version
    print("\n3. Testing U-Net Score Network:")
    unet_score = UNetScoreNetwork(
        point_dim=3,
        hidden_dims=(128, 256, 512, 256, 128),
        time_embed_dim=128
    )
    
    unet_output = unet_score(noisy_points, target_points, timesteps)
    print(f"U-Net score shape: {unet_output.shape}")
    
    # Tests gradient flow
    loss = score.sum()
    loss.backward()
    print("\n4. Gradient check passed!")
    
    # Counts parameters
    n_params = sum(p.numel() for p in score_net.parameters())
    n_params_unet = sum(p.numel() for p in unet_score.parameters())
    print(f"\nBasic Score Network parameters: {n_params:,}")
    print(f"U-Net Score Network parameters: {n_params_unet:,}")
    
    # Tests with single timestep
    print("\n5. Testing single timestep:")
    single_timestep = torch.tensor([250])
    single_score = score_net(
        noisy_points[:1],
        target_points[:1],
        single_timestep
    )
    print(f"Single score shape: {single_score.shape}")
    
    print("\nAll Score Network tests passed!")