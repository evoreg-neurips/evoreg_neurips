"""
Local feature enrichment via k-NN edge convolution.

Adds local geometry awareness to per-point features by aggregating
information from k nearest neighbors using edge convolution (DGCNN-style).

Memory-efficient: projects high-dim features down before edge convolution,
then projects back up. This avoids creating huge (B*N*k, 2*feat_dim) tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalFeatureEnrichment(nn.Module):
    """
    Enriches per-point features with local neighborhood information.

    For each point, finds k nearest neighbors in 3D space, computes
    edge features in a low-dimensional projected space, and maps back.
    Output is added to original features (residual connection).

    Memory budget: edge_feats is (B*N*k, 2*proj_dim) = (16*1024*16, 512) ≈ 128MB
    vs original (16*1024*20, 2048) ≈ 1.3GB
    """

    def __init__(self, feat_dim: int = 1024, k: int = 16, proj_dim: int = 256):
        super().__init__()
        self.k = k
        self.feat_dim = feat_dim

        # Project down: 1024 -> 256 (saves 8x memory in edge conv)
        self.proj_down = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
        )

        # Edge convolution in low-dim space: 512 -> 256 -> 256
        self.edge_conv = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Project back up: 256 -> 1024
        self.proj_up = nn.Linear(proj_dim, feat_dim)

    def forward(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) point coordinates for k-NN
            features: (B, N, D) per-point features to enrich

        Returns:
            (B, N, D) enriched features (residual added)
        """
        B, N, D = features.shape

        # Project to low-dim space
        feats_low = self.proj_down(features)  # (B, N, proj_dim)
        proj_dim = feats_low.shape[-1]

        # k-NN on 3D coordinates
        dists = torch.cdist(points, points)  # (B, N, N)
        _, idx = dists.topk(self.k, dim=-1, largest=False)  # (B, N, k)

        # Gather neighbor features efficiently using flat indexing
        batch_idx = torch.arange(B, device=features.device).view(B, 1, 1).expand(-1, N, self.k)
        neighbor_feats = feats_low[batch_idx, idx]  # (B, N, k, proj_dim)

        # Edge features: concat(feat_i, feat_j - feat_i)
        center_feats = feats_low.unsqueeze(2).expand_as(neighbor_feats)  # (B, N, k, proj_dim)
        edge_feats = torch.cat([center_feats, neighbor_feats - center_feats], dim=-1)  # (B, N, k, 2*proj_dim)

        # MLP on edge features
        edge_feats = edge_feats.reshape(B * N * self.k, -1)  # (B*N*k, 2*proj_dim)
        edge_feats = self.edge_conv(edge_feats)               # (B*N*k, proj_dim)
        edge_feats = edge_feats.view(B, N, self.k, proj_dim)

        # Max-pool over neighbors
        local_feats = edge_feats.max(dim=2).values  # (B, N, proj_dim)

        # Project back to original dim
        local_feats = self.proj_up(local_feats)  # (B, N, D)

        # Residual connection
        return features + local_feats
