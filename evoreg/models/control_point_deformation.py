"""
Control-point deformation module for Stage 3.

Alternative to VAE+Generator: selects K control points via farthest-point
sampling, predicts per-control-point displacements from local features,
and interpolates to all N points via Gaussian RBF weighting.
"""

import torch
import torch.nn as nn
from typing import Dict


class ControlPointDeformation(nn.Module):
    """
    Control-point-based non-rigid deformation head.

    1. Farthest-point sampling selects K control points from source.
    2. Per-control-point features (gathered from encoder) are concatenated
       with target global features and fed through an MLP to predict
       3D displacements.
    3. Gaussian RBF weights interpolate control displacements to all N points.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        target_feature_dim: int = 1024,
        hidden_dim: int = 512,
        n_control_points: int = 128,
        rbf_sigma: float = 0.5,
    ):
        super().__init__()
        self.n_control_points = n_control_points
        self.rbf_sigma = rbf_sigma

        # MLP: per-control-point features + target global → 3D displacement
        self.displacement_mlp = nn.Sequential(
            nn.Linear(feature_dim + target_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        # Zero-init final layer so model starts predicting no deformation
        nn.init.zeros_(self.displacement_mlp[-1].weight)
        nn.init.zeros_(self.displacement_mlp[-1].bias)

    @staticmethod
    def farthest_point_sampling(
        points: torch.Tensor, K: int
    ) -> torch.Tensor:
        """
        Iterative farthest-point sampling.

        Args:
            points: (B, N, 3)
            K: number of points to select

        Returns:
            indices: (B, K) long tensor of selected point indices
        """
        B, N, _ = points.shape
        device = points.device

        indices = torch.zeros(B, K, dtype=torch.long, device=device)
        # Distance from each point to the nearest already-selected point
        distances = torch.full((B, N), float('inf'), device=device)

        # Start from a random point (deterministic seed via first point)
        farthest = torch.zeros(B, dtype=torch.long, device=device)
        indices[:, 0] = farthest

        for i in range(1, K):
            # Gather the farthest point's coords: (B, 3)
            centroid = points[
                torch.arange(B, device=device), farthest
            ].unsqueeze(1)  # (B, 1, 3)
            # Squared distance from this centroid to all points
            dist = torch.sum((points - centroid) ** 2, dim=-1)  # (B, N)
            # Update minimum distances
            distances = torch.min(distances, dist)
            # Next farthest point
            farthest = distances.argmax(dim=-1)  # (B,)
            indices[:, i] = farthest

        return indices

    def forward(
        self,
        source_rigid_2: torch.Tensor,
        per_point_feats: torch.Tensor,
        tgt_points: torch.Tensor,
        tgt_point_feats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            source_rigid_2: (B, N, 3) source points after rigid alignment
            per_point_feats: (B, N, F) per-point features from encoder
            tgt_points: (B, M, 3) target point cloud coordinates
            tgt_point_feats: (B, M, F) per-point features from target encoder

        Returns:
            Dict with 'output', 'delta', 'control_points', 'control_displacements'
        """
        B, N, _ = source_rigid_2.shape
        K = min(self.n_control_points, N)

        # 1. FPS to select control point indices
        cp_indices = self.farthest_point_sampling(source_rigid_2, K)  # (B, K)

        # 2. Gather control point coords and features
        # Expand indices for gathering: (B, K, 1) → broadcast over last dim
        idx_expand_3 = cp_indices.unsqueeze(-1).expand(-1, -1, 3)
        control_points = torch.gather(source_rigid_2, 1, idx_expand_3)  # (B, K, 3)

        F = per_point_feats.shape[-1]
        idx_expand_f = cp_indices.unsqueeze(-1).expand(-1, -1, F)
        control_feats = torch.gather(per_point_feats, 1, idx_expand_f)  # (B, K, F)

        # 3. For each control point, find nearest target point in XYZ and gather its features
        dists = torch.cdist(control_points, tgt_points)  # (B, K, M)
        nn_indices = dists.argmin(dim=-1)  # (B, K)

        idx_expand_tf = nn_indices.unsqueeze(-1).expand(-1, -1, tgt_point_feats.shape[-1])
        tgt_local_feats = torch.gather(tgt_point_feats, 1, idx_expand_tf)  # (B, K, F)

        mlp_input = torch.cat([control_feats, tgt_local_feats], dim=-1)  # (B, K, 2F)

        # 4. MLP → control displacements
        # Reshape for BatchNorm: (B*K, F+D) → (B*K, 3) → (B, K, 3)
        mlp_flat = mlp_input.reshape(B * K, -1)
        w = self.displacement_mlp(mlp_flat).reshape(B, K, 3)  # (B, K, 3)

        # 5. Gaussian RBF interpolation
        # Squared distances: (B, N, K)
        # source_rigid_2: (B, N, 3), control_points: (B, K, 3)
        diff = source_rigid_2.unsqueeze(2) - control_points.unsqueeze(1)  # (B, N, K, 3)
        sq_dists = (diff ** 2).sum(dim=-1)  # (B, N, K)

        # Normalized RBF weights via softmax for numerical stability
        weights = torch.softmax(-sq_dists / (2 * self.rbf_sigma ** 2), dim=-1)  # (B, N, K)

        # 6. Interpolate: delta = weights @ w → (B, N, 3)
        delta = torch.bmm(weights, w)  # (B, N, 3)

        # 7. Output
        output = source_rigid_2 + delta

        return {
            'output': output,
            'delta': delta,
            'control_points': control_points,
            'control_displacements': w,
        }
