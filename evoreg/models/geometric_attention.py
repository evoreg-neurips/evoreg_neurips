"""
Geometric Cross-Attention module for transformation-invariant feature learning.

Adds distance-based geometric structure embeddings and cross-attention
between source and target point clouds to the EvoReg encoder.

Key ideas (informed by GeoTransformer, Qin et al. CVPR 2022):
  1. Pair-wise distances are invariant to rigid transforms -- encoding them
     as attention biases gives the network geometric awareness for free.
  2. Cross-attention between source and target lets the encoder discover
     inter-cloud correspondences before aggregation.

This is NOT a copy of GeoTransformer. We adapt the geometric-bias and
cross-attention concepts into EvoReg's VAE-based registration framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Distance encoding
# ---------------------------------------------------------------------------

class GaussianSmearing(nn.Module):
    """Fixed Gaussian radial basis functions for encoding scalar distances.

    Maps continuous distances to a set of Gaussian basis values centered
    at evenly spaced points from 0 to *cutoff*. Parameters are fixed
    (non-learnable) so no gradient is stored for the basis itself.
    """

    def __init__(self, num_rbf: int = 16, cutoff: float = 2.0):
        super().__init__()
        self.num_rbf = num_rbf
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.inv_width = num_rbf / cutoff  # 1 / sigma

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: (...) tensor of non-negative distances.
        Returns:
            (..., num_rbf) Gaussian basis values in [0, 1].
        """
        # (..., 1) - (K,) -> (..., K)
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (diff * self.inv_width) ** 2)


class GeometricBiasComputer(nn.Module):
    """Converts point coordinates into per-head attention biases.

    Pipeline:  points -> pair-wise distances -> Gaussian RBF -> linear -> bias
    The output is a (B, H, N, N) tensor that gets added to the QK^T
    attention logits, making the self-attention distance-aware.
    """

    def __init__(self, num_heads: int = 4, num_rbf: int = 16, cutoff: float = 2.0):
        super().__init__()
        self.rbf = GaussianSmearing(num_rbf, cutoff)
        self.proj = nn.Linear(num_rbf, num_heads, bias=False)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) point coordinates.
        Returns:
            bias: (B, num_heads, N, N) geometric attention bias.
        """
        with torch.no_grad():
            distances = torch.cdist(points, points)  # (B, N, N)
            rbf_values = self.rbf(distances)          # (B, N, N, K)
        # Only the projection is learnable
        bias = self.proj(rbf_values)                  # (B, N, N, H)
        return bias.permute(0, 3, 1, 2)               # (B, H, N, N)


# ---------------------------------------------------------------------------
# Attention layers
# ---------------------------------------------------------------------------

class GeometricSelfAttention(nn.Module):
    """Multi-head self-attention augmented with geometric distance bias.

    Attention scores are computed as:
        score(i,j) = (q_i · k_j) / sqrt(d)  +  geo_bias(i,j)

    Since Euclidean distances are invariant to rigid transforms, the
    geo_bias term provides transformation-invariant positional information.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
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
        self, x: torch.Tensor, geo_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim) point features.
            geo_bias: (B, H, N, N) pre-computed geometric bias (optional).
        Returns:
            (B, N, dim) attention output.
        """
        B, N, _ = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        k = self.k_proj(x).reshape(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, H, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if geo_bias is not None:
            attn = attn + geo_bias
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention between two sets of features.

    Lets one point cloud attend to the other, enabling the network to
    discover inter-cloud correspondences and build correspondence-aware
    features.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
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
        self, x_query: torch.Tensor, x_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_query:   (B, N, dim) features to update.
            x_context: (B, M, dim) features to attend to.
        Returns:
            (B, N, dim) cross-attention output.
        """
        B, N, _ = x_query.shape
        M = x_context.shape[1]
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x_query).reshape(B, N, H, d).transpose(1, 2)
        k = self.k_proj(x_context).reshape(B, M, H, d).transpose(1, 2)
        v = self.v_proj(x_context).reshape(B, M, H, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Blocks and full module
# ---------------------------------------------------------------------------

class GeometricCrossAttentionBlock(nn.Module):
    """One round of geometric self-attention + cross-attention + FFN.

    Processing order (following GeoTransformer's interleaved design):
      1. Geometric self-attention on each cloud (with distance bias)
      2. Cross-attention between clouds (source <-> target)
      3. Feed-forward network on each cloud

    All sub-layers use pre-norm residual connections.  Parameters are
    shared between the two clouds (symmetric processing).
    """

    def __init__(
        self, dim: int, num_heads: int = 4, ffn_ratio: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        # Self-attention (shared for both clouds)
        self.self_attn = GeometricSelfAttention(dim, num_heads, dropout)
        self.norm_sa = nn.LayerNorm(dim)

        # Cross-attention (shared for both directions)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm_ca = nn.LayerNorm(dim)

        # Feed-forward (shared for both clouds)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(dim)

    def forward(
        self,
        src_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        src_geo_bias: torch.Tensor,
        tgt_geo_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_feats:    (B, N, dim) source point features.
            tgt_feats:    (B, M, dim) target point features.
            src_geo_bias: (B, H, N, N) geometric bias for source self-attn.
            tgt_geo_bias: (B, H, M, M) geometric bias for target self-attn.
        Returns:
            Updated (src_feats, tgt_feats).
        """
        # --- 1. Geometric self-attention (intra-cloud) ---
        src_feats = src_feats + self.self_attn(self.norm_sa(src_feats), src_geo_bias)
        tgt_feats = tgt_feats + self.self_attn(self.norm_sa(tgt_feats), tgt_geo_bias)

        # --- 2. Cross-attention (inter-cloud, symmetric) ---
        src_normed = self.norm_ca(src_feats)
        tgt_normed = self.norm_ca(tgt_feats)
        src_feats = src_feats + self.cross_attn(src_normed, tgt_normed)
        tgt_feats = tgt_feats + self.cross_attn(tgt_normed, src_normed)

        # --- 3. Feed-forward ---
        src_feats = src_feats + self.ffn(self.norm_ff(src_feats))
        tgt_feats = tgt_feats + self.ffn(self.norm_ff(tgt_feats))

        return src_feats, tgt_feats


class GeometricCrossAttentionModule(nn.Module):
    """Full geometric cross-attention encoder.

    Stacks *num_blocks* GeometricCrossAttentionBlocks.  The geometric
    distance biases are pre-computed once from the input coordinates and
    shared across all blocks (no redundant computation).

    Args:
        dim:          Feature dimension for the attention layers.
        num_heads:    Number of attention heads.
        num_blocks:   Number of stacked blocks.
        ffn_ratio:    FFN hidden-dim multiplier.
        num_rbf:      Number of Gaussian RBF centres for distance encoding.
        cutoff:       Maximum distance for the RBF (should cover point cloud diameter).
        dropout:      Dropout rate inside attention and FFN.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 3,
        ffn_ratio: int = 4,
        num_rbf: int = 16,
        cutoff: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.geo_bias = GeometricBiasComputer(num_heads, num_rbf, cutoff)
        self.blocks = nn.ModuleList(
            [
                GeometricCrossAttentionBlock(dim, num_heads, ffn_ratio, dropout)
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        src_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_feats:  (B, N, dim) initial per-point source features.
            tgt_feats:  (B, M, dim) initial per-point target features.
            src_points: (B, N, 3)   source coordinates (used for distance bias).
            tgt_points: (B, M, 3)   target coordinates (used for distance bias).
        Returns:
            Enriched (src_feats, tgt_feats) with same shapes.
        """
        # Pre-compute geometric biases once (shared across all blocks)
        src_geo_bias = self.geo_bias(src_points)  # (B, H, N, N)
        tgt_geo_bias = self.geo_bias(tgt_points)  # (B, H, M, M)

        for block in self.blocks:
            src_feats, tgt_feats = block(
                src_feats, tgt_feats, src_geo_bias, tgt_geo_bias
            )

        return src_feats, tgt_feats
