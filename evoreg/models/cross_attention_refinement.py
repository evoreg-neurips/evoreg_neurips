"""
Cross-attention refinement for Stage 2a of coarse-to-fine registration.

Replaces the MLP rigid head with cross-attention-based correspondences + SVD,
preserving per-point spatial information that max-pool destroys.

Pipeline:
  1. Project per-point features (1024 -> 256) with LayerNorm
  2. Cross-attention: source attends to target (with inter-cloud geo bias)
  3. Soft correspondences (cosine similarity -> Sinkhorn) -> assignment matrix
  4. Kabsch SVD: weighted rigid alignment from correspondences
  5. Global features for VAE (max-pool enriched source + cached tgt_global)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .geometric_attention import GaussianSmearing
from .soft_correspondence import FeatureProjector, DifferentiableKabsch


class InterCloudGeometricBias(nn.Module):
    """RBF-encoded distances between source and target points -> per-head attention bias.

    Unlike GeometricBiasComputer (intra-cloud, NxN), this computes INTER-cloud
    distances (NxM) for cross-attention bias.
    """

    def __init__(self, num_heads: int = 4, num_rbf: int = 16, cutoff: float = 2.0):
        super().__init__()
        self.rbf = GaussianSmearing(num_rbf, cutoff)
        self.proj = nn.Linear(num_rbf, num_heads, bias=False)

    def forward(self, src_points: torch.Tensor, tgt_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_points: (B, N, 3)
            tgt_points: (B, M, 3)
        Returns:
            bias: (B, num_heads, N, M)
        """
        with torch.no_grad():
            distances = torch.cdist(src_points, tgt_points)  # (B, N, M)
            rbf_values = self.rbf(distances)  # (B, N, M, K)
        bias = self.proj(rbf_values)  # (B, N, M, H)
        return bias.permute(0, 3, 1, 2)  # (B, H, N, M)


class CrossAttentionWithGeoBias(nn.Module):
    """Multi-head cross-attention with geometric distance bias.

    Q from source features, K/V from target features.
    Adds geo_bias to QK^T attention logits.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x_query: torch.Tensor,
        x_context: torch.Tensor,
        geo_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x_query:   (B, N, dim) source features
            x_context: (B, M, dim) target features
            geo_bias:  (B, H, N, M) geometric attention bias (optional)
        Returns:
            (B, N, dim)
        """
        B, N, _ = x_query.shape
        M = x_context.shape[1]
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x_query).reshape(B, N, H, d).transpose(1, 2)   # (B, H, N, d)
        k = self.k_proj(x_context).reshape(B, M, H, d).transpose(1, 2)  # (B, H, M, d)
        v = self.v_proj(x_context).reshape(B, M, H, d).transpose(1, 2)  # (B, H, M, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)
        if geo_bias is not None:
            attn = attn + geo_bias
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        return self.out_proj(out)


class CrossAttentionRefinement(nn.Module):
    """
    Stage 2a replacement: cross-attention + soft correspondences + SVD.

    Replaces the MLP rigid head that operates on max-pooled global features
    with a per-point cross-attention mechanism that preserves spatial info.
    """

    def __init__(
        self,
        point_feat_dim: int = 1024,
        attn_dim: int = 256,
        corr_proj_dim: int = 256,
        global_feat_dim: int = 512,
        num_heads: int = 4,
        num_rbf: int = 16,
        rbf_cutoff: float = 2.0,
        dropout: float = 0.1,
        temperature: float = 0.05,
    ):
        super().__init__()

        # Feature projectors (1024 -> 256)
        self.src_feat_proj = nn.Sequential(
            nn.Linear(point_feat_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
        )
        self.tgt_feat_proj = nn.Sequential(
            nn.Linear(point_feat_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
        )

        # Cross-attention with geometric bias
        self.geo_bias = InterCloudGeometricBias(num_heads, num_rbf, rbf_cutoff)
        self.cross_attn = CrossAttentionWithGeoBias(attn_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 4, attn_dim),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(attn_dim)

        # Correspondence projectors (separate from Stage 1)
        self.src_corr_proj = FeatureProjector(attn_dim, corr_proj_dim)
        self.tgt_corr_proj = FeatureProjector(attn_dim, corr_proj_dim)

        # Learnable temperature (sharp, init 0.05)
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Kabsch SVD
        self.kabsch = DifferentiableKabsch()

        # Global feature projection for VAE input
        self.global_proj = nn.Sequential(
            nn.Linear(attn_dim, global_feat_dim),
            nn.LayerNorm(global_feat_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        src_point_feats: torch.Tensor,
        tgt_point_feats: torch.Tensor,
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
        tgt_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src_point_feats: (B, N, 1024) from encoder.forward_with_point_features
            tgt_point_feats: (B, M, 1024) cached from Stage 1
            src_points: (B, N, 3) current source positions (after Stage 1)
            tgt_points: (B, M, 3) target positions
            tgt_global: (B, 512) cached target global features

        Returns:
            R_res: (B, 3, 3) residual rotation
            t_res: (B, 3) residual translation
            combined: (B, 1024) combined global features for VAE
            confidence: (B, N) per-point confidence
            assignment: (B, N, M) assignment matrix
        """
        # 1. Project features to attention dim
        src_proj = self.src_feat_proj(src_point_feats)  # (B, N, 256)
        tgt_proj = self.tgt_feat_proj(tgt_point_feats)  # (B, M, 256)

        # 2. Cross-attention with inter-cloud geo bias
        geo_bias = self.geo_bias(src_points, tgt_points)  # (B, H, N, M)
        src_enriched = src_proj + self.cross_attn(
            self.norm(src_proj), self.norm(tgt_proj), geo_bias
        )
        src_enriched = src_enriched + self.ffn(self.norm_ff(src_enriched))

        # 3. Soft correspondences
        src_corr = self.src_corr_proj(src_enriched)  # (B, N, corr_proj_dim) L2-normalized
        tgt_corr = self.tgt_corr_proj(tgt_proj)       # (B, M, corr_proj_dim) L2-normalized

        tau = self.temperature.clamp(min=0.01, max=1.0)
        similarity = torch.bmm(src_corr, tgt_corr.transpose(1, 2)) / tau  # (B, N, M)

        # Sinkhorn normalization (3 iterations)
        log_S = similarity
        for _ in range(3):
            log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
            log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
        S = torch.exp(log_S)

        tgt_corr_points = torch.bmm(S, tgt_points)  # (B, N, 3)
        confidence = S.max(dim=-1).values  # (B, N)

        # 4. Kabsch SVD
        R_res, t_res = self.kabsch(src_points, tgt_corr_points, weights=confidence)

        # 5. Global features for VAE
        src_global_new = self.global_proj(src_enriched.max(dim=1).values)  # (B, 512)
        combined = torch.cat([src_global_new, tgt_global], dim=-1)  # (B, 1024)

        return R_res, t_res, combined, confidence, S
