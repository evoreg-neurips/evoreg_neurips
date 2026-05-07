"""
GeoTransformer baseline reimplementation (Qin et al., CVPR 2022).

Pure PyTorch reimplementation of GeoTransformer for point cloud registration,
faithful to the paper architecture. No CUDA extensions required.

Architecture:
    Source + Target → KPConv-FPN backbone → Geometric Transformer
    → Superpoint Matching → Point Matching (Sinkhorn OT)
    → Local-to-Global Registration → R, t
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# A) Pure-PyTorch replacements for CUDA ops
# ---------------------------------------------------------------------------

def farthest_point_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Iterative farthest-point sampling.

    Args:
        points: (B, N, 3)
        n_samples: number of points to select

    Returns:
        indices: (B, n_samples) long tensor
    """
    B, N, _ = points.shape
    device = points.device
    indices = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), float('inf'), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    indices[:, 0] = farthest
    for i in range(1, n_samples):
        centroid = points[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        distances = torch.min(distances, dist)
        farthest = distances.argmax(dim=-1)
        indices[:, i] = farthest
    return indices


def knn_search(query: torch.Tensor, support: torch.Tensor, k: int) -> torch.Tensor:
    """k-nearest neighbor search via pairwise distances.

    Args:
        query: (B, N, 3)
        support: (B, M, 3)
        k: number of neighbors

    Returns:
        indices: (B, N, k) long tensor — indices into support
    """
    dists = torch.cdist(query, support, p=2.0)  # (B, N, M)
    _, indices = dists.topk(k, dim=-1, largest=False)
    return indices


# ---------------------------------------------------------------------------
# B) KPConv modules (pure PyTorch)
# ---------------------------------------------------------------------------

def _get_kernel_points(num_points: int = 15, dimension: int = 3) -> torch.Tensor:
    """Generate fixed kernel point dispositions.

    Uses Fibonacci-sphere layout for reproducibility. The original KPConv uses
    energy-minimized dispositions, but Fibonacci provides a reasonable approximation
    for a reimplementation (kernel weights are learned regardless).
    """
    points = [torch.zeros(dimension)]  # center point
    n = num_points - 1
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append(torch.tensor([x, y, z]))
    return torch.stack(points, dim=0)  # (num_points, 3)


class KPConv(nn.Module):
    """Kernel Point Convolution (Thomas et al., ICCV 2019).

    Aggregates neighbor features weighted by Gaussian kernels centered at
    fixed kernel point positions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        radius: float = 0.1,
        sigma: float = 0.04,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma

        kernel_points = _get_kernel_points(kernel_size, 3)
        self.register_buffer('kernel_points', kernel_points * radius * 0.7)

        self.weights = nn.Parameter(
            torch.empty(kernel_size, in_channels, out_channels)
        )
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        query_points: torch.Tensor,
        support_points: torch.Tensor,
        support_features: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_points: (B, N, 3)
            support_points: (B, M, 3)
            support_features: (B, M, C_in)
            neighbor_indices: (B, N, K) — indices into support_points

        Returns:
            output: (B, N, C_out)
        """
        B, N, K_neigh = neighbor_indices.shape

        # Clamp indices to valid range for safe gathering
        safe_idx = neighbor_indices.clamp(0, support_points.shape[1] - 1)

        # Gather neighbor coordinates: (B, N, K_neigh, 3)
        idx_3 = safe_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        neighbor_coords = torch.gather(
            support_points.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_3
        )
        # Gather neighbor features: (B, N, K_neigh, C_in)
        idx_feat = safe_idx.unsqueeze(-1).expand(-1, -1, -1, self.in_channels)
        neighbor_feats = torch.gather(
            support_features.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_feat
        )

        # Relative positions: (B, N, K_neigh, 3)
        rel_pos = neighbor_coords - query_points.unsqueeze(2)

        # Linear clamp influence (matching official KPConv, NOT Gaussian):
        # weight = clamp(1 - ||p - kp|| / sigma, min=0)
        diff = rel_pos.unsqueeze(3) - self.kernel_points  # (B, N, K_neigh, Kp, 3)
        dists = (diff ** 2).sum(-1).clamp(min=1e-10).sqrt()  # (B, N, K_neigh, Kp)
        kernel_weights = (1.0 - dists / self.sigma).clamp(min=0.0)
        # Transpose for matmul: (B, N, Kp, K_neigh)
        kernel_weights = kernel_weights.permute(0, 1, 3, 2)

        # Aggregate: weighted sum of neighbor features per kernel point
        weighted_feats = torch.matmul(kernel_weights, neighbor_feats)  # (B, N, Kp, C_in)

        # Per-kernel-point linear transform, sum over kernel points
        # weighted_feats: (B, N, Kp, C_in), weights: (Kp, C_in, C_out)
        output = torch.einsum('bnpc,pco->bno', weighted_feats, self.weights)

        # Normalize by number of non-empty neighbors (matching official)
        neighbor_has_feat = (neighbor_feats.abs().sum(-1) > 0).float()  # (B, N, K_neigh)
        neighbor_count = neighbor_has_feat.sum(-1).clamp(min=1.0)  # (B, N)
        output = output / neighbor_count.unsqueeze(-1)

        if self.bias is not None:
            output = output + self.bias

        return output


class UnaryBlock(nn.Module):
    """1x1 convolution equivalent: Linear + GroupNorm + LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        ng = min(num_groups, out_channels)
        while out_channels % ng != 0:
            ng -= 1
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.GroupNorm(ng, out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C)"""
        B, N, C = x.shape
        x = self.linear(x.reshape(B * N, C)).reshape(B, N, -1)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # GroupNorm on (B, C, N)
        return self.act(x)


class ConvBlock(nn.Module):
    """KPConv + GroupNorm + LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, radius: float, sigma: float,
                 num_groups: int = 32):
        super().__init__()
        self.conv = KPConv(in_channels, out_channels, radius=radius, sigma=sigma)
        ng = min(num_groups, out_channels)
        while out_channels % ng != 0:
            ng -= 1
        self.norm = nn.GroupNorm(ng, out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, query_pts, support_pts, support_feats, neighbor_idx):
        x = self.conv(query_pts, support_pts, support_feats, neighbor_idx)
        B, N, C = x.shape
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.act(x)


class ResidualBlock(nn.Module):
    """Bottleneck residual: UnaryBlock → ConvBlock → UnaryBlock + skip."""

    def __init__(self, in_channels: int, out_channels: int, radius: float, sigma: float):
        super().__init__()
        mid = out_channels // 4
        self.unary1 = UnaryBlock(in_channels, mid)
        self.conv = ConvBlock(mid, mid, radius=radius, sigma=sigma)
        self.unary2 = UnaryBlock(mid, out_channels)
        self.skip = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, query_pts, support_pts, support_feats, neighbor_idx):
        identity = self.skip(support_feats)
        x = self.unary1(support_feats)
        x = self.conv(query_pts, support_pts, x, neighbor_idx)
        x = self.unary2(x)
        return self.act(x + identity)


class KPConvFPN(nn.Module):
    """KPConv Feature Pyramid Network (matching official GeoTransformer config).

    Official ModelNet40 backbone:
        - 3 encoder stages producing dims 128, 256, 512 (init_dim * 2/4/8)
        - 2 decoder stages producing 256-dim fine features via skip connections
        - Coarse features = raw encoder stage 3 output (512-dim) -> transformer
        - Fine features = FPN-decoded output (256-dim) -> point matching

    For small point clouds (N~1024), uses FPS for downsampling and kNN for neighbors.
    """

    def __init__(
        self,
        in_channels: int = 1,
        init_dim: int = 64,
        encoder_dims: Tuple[int, ...] = (128, 256, 512),
        decoder_dims: Tuple[int, ...] = (256, 256),
        base_radius: float = 0.075,
        base_sigma: float = 0.03,
        k_neighbors: int = 16,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors

        # Initial feature embedding
        self.input_proj = UnaryBlock(in_channels, init_dim)

        # Encoder stages
        self.encoder_blocks = nn.ModuleList()
        dims = [init_dim] + list(encoder_dims)
        for i in range(len(encoder_dims)):
            radius = base_radius * (2 ** i)
            sigma = base_sigma * (2 ** i)
            self.encoder_blocks.append(
                ResidualBlock(dims[i], dims[i + 1], radius=radius, sigma=sigma)
            )

        # Decoder stages (upsampling with skip connections)
        # Official: decoder2 takes cat(upsample(512), 256_skip)=768 -> 256
        #           decoder1 takes cat(upsample(256), 128_skip)=384 -> 256
        self.decoder_blocks = nn.ModuleList()
        self.decoder_skips = nn.ModuleList()
        dec_in_dims = [encoder_dims[-1]] + list(decoder_dims[:-1])
        for i in range(len(decoder_dims)):
            skip_dim = encoder_dims[-(i + 2)] if (i + 2) <= len(encoder_dims) else init_dim
            self.decoder_skips.append(nn.Linear(skip_dim + dec_in_dims[i], decoder_dims[i]))
            self.decoder_blocks.append(UnaryBlock(decoder_dims[i], decoder_dims[i]))

        self.coarse_dim = encoder_dims[-1]  # 512 — raw encoder output for transformer
        self.fine_dim = decoder_dims[-1]     # 256 — decoded output for point matching

    def _downsample(self, points: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample via FPS."""
        B, N, _ = points.shape
        n_out = max(int(N * ratio), 1)
        idx = farthest_point_sampling(points, n_out)
        down_pts = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        return down_pts, idx

    def _upsample_features(
        self, coarse_pts: torch.Tensor, coarse_feats: torch.Tensor,
        fine_pts: torch.Tensor, k: int = 3
    ) -> torch.Tensor:
        """Interpolate features from coarse to fine via inverse-distance-weighted kNN."""
        nn_idx = knn_search(fine_pts, coarse_pts, k)  # (B, N_fine, k)
        B, N_fine, _ = fine_pts.shape
        C = coarse_feats.shape[-1]
        idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        nn_feats = torch.gather(
            coarse_feats.unsqueeze(1).expand(-1, N_fine, -1, -1), 2, idx_exp
        )  # (B, N_fine, k, C)
        nn_coords = torch.gather(
            coarse_pts.unsqueeze(1).expand(-1, N_fine, -1, -1), 2,
            nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        # Inverse-distance weighting (not inverse-squared-distance)
        dists = ((fine_pts.unsqueeze(2) - nn_coords) ** 2).sum(-1).clamp(min=1e-8).sqrt()
        weights = 1.0 / dists
        weights = weights / weights.sum(dim=-1, keepdim=True)
        interp = (weights.unsqueeze(-1) * nn_feats).sum(dim=2)
        return interp

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (B, N, 3)

        Returns:
            coarse_points: (B, N_c, 3) — superpoint positions
            coarse_feats: (B, N_c, D) — superpoint features
            fine_points: (B, N_f, 3) — fine point positions
            fine_feats: (B, N_f, D) — fine point features
        """
        B, N, _ = points.shape

        # Official GeoTransformer uses ones as initial features (input_dim=1).
        # Positions are used for convolution geometry, not as features.
        ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
        feats = self.input_proj(ones)  # (B, N, init_dim)

        # Encoder
        enc_points = [points]
        enc_feats = [feats]

        current_pts = points
        current_feats = feats

        for i, block in enumerate(self.encoder_blocks):
            if i > 0:
                down_pts, down_idx = self._downsample(current_pts, 0.25)
                down_feats = torch.gather(
                    current_feats, 1,
                    down_idx.unsqueeze(-1).expand(-1, -1, current_feats.shape[-1])
                )
                current_pts = down_pts
                current_feats = down_feats

            nn_idx = knn_search(current_pts, current_pts, self.k_neighbors)
            current_feats = block(current_pts, current_pts, current_feats, nn_idx)
            enc_points.append(current_pts)
            enc_feats.append(current_feats)

        coarse_points = enc_points[-1]
        coarse_feats = enc_feats[-1]

        # Decoder — upsample with skip connections
        dec_feats = coarse_feats
        for i, (skip_proj, dec_block) in enumerate(zip(self.decoder_skips, self.decoder_blocks)):
            target_idx = len(enc_points) - 2 - i
            target_idx = max(target_idx, 0)
            target_pts = enc_points[target_idx]
            target_skip_feats = enc_feats[target_idx]

            upsampled = self._upsample_features(
                enc_points[-1 - i] if i == 0 else enc_points[target_idx + 1],
                dec_feats, target_pts
            )

            cat_feats = torch.cat([upsampled, target_skip_feats], dim=-1)
            dec_feats = skip_proj(cat_feats)
            dec_feats = dec_block(dec_feats)

        fine_points = target_pts
        fine_feats = dec_feats

        return coarse_points, coarse_feats, fine_points, fine_feats


# ---------------------------------------------------------------------------
# C) Geometric Transformer
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding for scalar values (distances or angles)."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...) → (..., embed_dim)"""
        half = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=x.device, dtype=x.dtype) *
            -(math.log(10000.0) / half)
        )
        args = x.unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class GeometricStructureEmbedding(nn.Module):
    """Encodes pair-wise distances and triplet-wise angles (Eq. 5-9).

    Makes attention features transformation-invariant by using geometric
    relationships rather than absolute positions.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        sigma_d: float = 0.2,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        reduction: str = 'max',
    ):
        super().__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.angle_k = angle_k
        self.reduction = reduction

        # Factor converts radians to the scaled index matching official code:
        # factor_a = 180 / (sigma_a * pi)
        self.factor_a = 180.0 / (sigma_a * math.pi)

        self.dist_encoding = SinusoidalPositionalEncoding(hidden_dim)
        self.angle_encoding = SinusoidalPositionalEncoding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @torch.no_grad()
    def _get_embedding_indices(self, points: torch.Tensor):
        """Compute distance and angle indices (no gradients, matching official)."""
        B, N, _ = points.shape
        diffs = points.unsqueeze(2) - points.unsqueeze(1)  # (B, N, N, 3)
        dists = diffs.norm(dim=-1)  # (B, N, N)
        d_indices = dists / self.sigma_d

        a_indices = None
        if self.angle_k > 0 and N > 2:
            k = min(self.angle_k, N - 1)
            self_dists = dists.clone()
            self_dists[:, range(N), range(N)] = float('inf')
            _, nn_idx = self_dists.topk(k, dim=-1, largest=False)
            nn_idx_3 = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            nn_points = torch.gather(points.unsqueeze(1).expand(-1, N, -1, -1), 2, nn_idx_3)
            # ref: i→neighbor_x, anc: j→i (matching official)
            ref_vectors = nn_points - points.unsqueeze(2)  # (B, N, k, 3)
            anc_vectors = diffs  # (B, N, N, 3): points[i] - points[j]
            ref_vectors = ref_vectors.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, N, N, k, 3)
            anc_vectors = anc_vectors.unsqueeze(3).expand(-1, -1, -1, k, -1)  # (B, N, N, k, 3)
            sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)
            cos_values = (ref_vectors * anc_vectors).sum(dim=-1)
            angles = torch.atan2(sin_values, cos_values)
            a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) superpoint positions

        Returns:
            rpe: (B, N, N, hidden_dim) relative position embeddings
        """
        d_indices, a_indices = self._get_embedding_indices(points)

        # Sinusoidal embeddings are detached (matching official)
        dist_embed = self.proj_d(self.dist_encoding(d_indices).detach())

        if a_indices is not None:
            angle_embed = self.proj_a(self.angle_encoding(a_indices).detach())
            if self.reduction == 'max':
                angle_embed = angle_embed.max(dim=3).values
            else:
                angle_embed = angle_embed.mean(dim=3)
            rpe = dist_embed + angle_embed
        else:
            rpe = dist_embed

        return rpe


class RPEMultiHeadAttention(nn.Module):
    """Multi-head attention with Relative Position Embedding (Eq. 8).

    Content-dependent RPE: a_ij = (Q_i K_j^T + Q_i (W_R r_ij)^T) / sqrt(d)
    The RPE contribution depends on query content, not just position.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # Project RPE to full hidden_dim for content-dependent interaction with Q
        self.rpe_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, rpe: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            rpe: (B, N, N, D) — relative position embeddings
            mask: optional (B, N, N) bool mask

        Returns:
            (B, N, D)
        """
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        Q = self.q_proj(x).view(B, N, H, d).permute(0, 2, 1, 3)  # (B, H, N, d)
        K = self.k_proj(x).view(B, N, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, N, H, d).permute(0, 2, 1, 3)

        # Content-based attention: Q_i * K_j^T
        attn_content = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, N, N)

        # Content-dependent RPE: Q_i * (W_R * r_ij)^T
        # Project RPE and reshape to per-head: (B, N, N, H, d)
        rpe_proj = self.rpe_proj(rpe).view(B, N, N, H, d)
        # Permute to (B, H, N, N, d) for einsum with Q: (B, H, N, d)
        rpe_proj = rpe_proj.permute(0, 3, 1, 2, 4)  # (B, H, N, N, d)
        attn_position = torch.einsum('bhnd,bhnmd->bhnm', Q, rpe_proj)  # (B, H, N, N)

        attn = (attn_content + attn_position) / math.sqrt(d)

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, N, d)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


class GeometricSelfAttentionLayer(nn.Module):
    """Self-attention with geometric RPE + FFN.

    Uses post-norm (LayerNorm after residual) matching the official implementation.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, ffn_dim: int = 512):
        super().__init__()
        self.attn = RPEMultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, rpe: torch.Tensor) -> torch.Tensor:
        # Post-norm: LayerNorm(x + sublayer(x))
        x = self.norm1(x + self.attn(x, rpe))
        x = self.norm2(x + self.ffn(x))
        return x


class CrossAttentionLayer(nn.Module):
    """Standard cross-attention + FFN.

    Uses post-norm matching the official implementation.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, ffn_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_q: (B, N, D) query features
            x_kv: (B, M, D) key/value features

        Returns:
            (B, N, D)
        """
        B, N, D = x_q.shape
        M = x_kv.shape[1]
        H = self.num_heads
        d = self.head_dim

        Q = self.q_proj(x_q).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(x_kv).view(B, M, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(x_kv).view(B, M, H, d).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)

        # Post-norm
        x_q = self.norm_attn(x_q + out)
        x_q = self.norm_ffn(x_q + self.ffn(x_q))
        return x_q


class GeometricTransformerModule(nn.Module):
    """Interleaved [geometric self-attention, cross-attention] x num_blocks.

    Self-attention layers are shared between source and target (Siamese).
    Includes in_proj/out_proj linear layers matching the official implementation.
    Cross-attention is sequential: src updated first, tgt sees updated src.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 3,
        sigma_d: float = 0.2,
        sigma_a: float = 15.0,
        angle_k: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.gse = GeometricStructureEmbedding(
            hidden_dim=hidden_dim, sigma_d=sigma_d, sigma_a=sigma_a, angle_k=angle_k
        )
        self.self_attn_layers = nn.ModuleList([
            GeometricSelfAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        src_feats: torch.Tensor, src_points: torch.Tensor,
        tgt_feats: torch.Tensor, tgt_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_feats: (B, N_s, D_in) source superpoint features
            src_points: (B, N_s, 3)
            tgt_feats: (B, N_t, D_in) target superpoint features
            tgt_points: (B, N_t, 3)

        Returns:
            src_feats: (B, N_s, D_out) updated source features
            tgt_feats: (B, N_t, D_out) updated target features
        """
        # Project to hidden dim
        src_feats = self.in_proj(src_feats)
        tgt_feats = self.in_proj(tgt_feats)

        # Compute geometric RPEs once (positions don't change)
        rpe_src = self.gse(src_points)  # (B, N_s, N_s, D)
        rpe_tgt = self.gse(tgt_points)  # (B, N_t, N_t, D)

        for self_attn, cross_attn in zip(self.self_attn_layers, self.cross_attn_layers):
            # Self-attention with geometric RPE (shared weights for src/tgt)
            src_feats = self_attn(src_feats, rpe_src)
            tgt_feats = self_attn(tgt_feats, rpe_tgt)
            # Cross-attention: sequential (src updated first, tgt sees updated src)
            src_feats = cross_attn(src_feats, tgt_feats)
            tgt_feats = cross_attn(tgt_feats, src_feats)

        # Project to output dim
        src_feats = self.out_proj(src_feats)
        tgt_feats = self.out_proj(tgt_feats)

        return src_feats, tgt_feats


# ---------------------------------------------------------------------------
# D) Superpoint Matching
# ---------------------------------------------------------------------------

class SuperpointMatching(nn.Module):
    """Gaussian correlation + dual normalization + top-k (Eq. 12).

    Dual normalization: row-normalize × col-normalize on the Gaussian
    correlation matrix (matching official implementation).
    """

    def __init__(self, num_correspondences: int = 128, dual_norm: bool = True):
        super().__init__()
        self.num_correspondences = num_correspondences
        self.dual_norm = dual_norm

    def forward(
        self,
        src_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src_feats: (B, N_s, D)
            tgt_feats: (B, N_t, D)

        Returns:
            src_indices: (B, Nc) indices into source superpoints
            tgt_indices: (B, Nc) indices into target superpoints
            scores: (B, Nc) matching scores
        """
        # L2-normalize features
        src_norm = F.normalize(src_feats, dim=-1)
        tgt_norm = F.normalize(tgt_feats, dim=-1)

        # Gaussian correlation: exp(-||h_i - h_j||^2) = exp(-(2 - 2*cos_sim))
        cos_sim = torch.bmm(src_norm, tgt_norm.transpose(1, 2))  # (B, N_s, N_t)
        correlation = torch.exp(-(2.0 - 2.0 * cos_sim))

        if self.dual_norm:
            # Dual normalization: row-norm × col-norm (matching official code)
            row_norm = correlation / (correlation.sum(dim=-1, keepdim=True) + 1e-8)
            col_norm = correlation / (correlation.sum(dim=-2, keepdim=True) + 1e-8)
            correlation = row_norm * col_norm

        B = src_feats.shape[0]
        Nc = min(self.num_correspondences, correlation.shape[1] * correlation.shape[2])

        flat_corr = correlation.view(B, -1)
        topk_scores, topk_idx = flat_corr.topk(Nc, dim=-1)

        src_indices = topk_idx // correlation.shape[2]
        tgt_indices = topk_idx % correlation.shape[2]

        return src_indices, tgt_indices, topk_scores


# ---------------------------------------------------------------------------
# E) Point Matching (Sinkhorn Optimal Transport)
# ---------------------------------------------------------------------------

class LearnableLogOptimalTransport(nn.Module):
    """Sinkhorn optimal transport with learnable dustbin parameter."""

    def __init__(self, num_iters: int = 100):
        super().__init__()
        self.num_iters = num_iters
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_matrix: (B, N, M) score matrix (higher = better match)

        Returns:
            Z: (B, N+1, M+1) assignment matrix with dustbin rows/columns
        """
        B, N, M = cost_matrix.shape
        device = cost_matrix.device

        dustbin_row = self.alpha.expand(B, 1, M)
        dustbin_col = self.alpha.expand(B, N, 1)
        dustbin_corner = self.alpha.expand(B, 1, 1)

        augmented = torch.cat([
            torch.cat([cost_matrix, dustbin_col], dim=-1),
            torch.cat([dustbin_row, dustbin_corner], dim=-1),
        ], dim=-2)  # (B, N+1, M+1)

        # Log-space Sinkhorn iterations
        log_Z = augmented
        for _ in range(self.num_iters):
            log_Z = log_Z - torch.logsumexp(log_Z, dim=-1, keepdim=True)
            log_Z = log_Z - torch.logsumexp(log_Z, dim=-2, keepdim=True)

        return torch.exp(log_Z)


class PointMatching(nn.Module):
    """Per-superpoint-pair point matching using Sinkhorn OT.

    Assigns fine-level points within each matched superpoint pair using
    optimal transport, then extracts mutual top-k correspondences.
    """

    def __init__(self, num_sinkhorn_iters: int = 100, top_k: int = 3,
                 confidence_threshold: float = 0.05):
        super().__init__()
        self.ot = LearnableLogOptimalTransport(num_sinkhorn_iters)
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

    def _extract_mutual_topk(self, Z_core: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized mutual top-k extraction from assignment matrix.

        Args:
            Z_core: (n_s, n_t) assignment probabilities (no dustbin)

        Returns:
            src_idx: (K,) matched source indices
            tgt_idx: (K,) matched target indices
        """
        n_s, n_t = Z_core.shape
        k = min(k, n_s, n_t)
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=Z_core.device), \
                   torch.empty(0, dtype=torch.long, device=Z_core.device)

        # Row-wise top-k: for each source, best target matches
        _, row_topk = Z_core.topk(k, dim=-1)  # (n_s, k)
        # Column-wise top-k: for each target, best source matches
        _, col_topk = Z_core.topk(k, dim=-2)  # (k, n_t)

        # Build candidate pairs from row-wise top-k
        src_cands = torch.arange(n_s, device=Z_core.device).unsqueeze(1).expand(-1, k).reshape(-1)
        tgt_cands = row_topk.reshape(-1)

        # Check mutual: for each candidate (si, tj), check if si is in col_topk[:, tj]
        # col_topk[:, tgt_cands] gives (k, n_candidates) — top-k sources for each candidate target
        col_for_cands = col_topk[:, tgt_cands]  # (k, n_candidates)
        is_mutual = (col_for_cands == src_cands.unsqueeze(0)).any(dim=0)  # (n_candidates,)

        # Also filter by confidence threshold
        scores = Z_core[src_cands, tgt_cands]
        is_confident = scores > self.confidence_threshold

        valid = is_mutual & is_confident
        return src_cands[valid], tgt_cands[valid]

    def forward(
        self,
        src_fine_feats: torch.Tensor,
        tgt_fine_feats: torch.Tensor,
        src_fine_points: torch.Tensor,
        tgt_fine_points: torch.Tensor,
        src_sp_indices: torch.Tensor,
        tgt_sp_indices: torch.Tensor,
        src_coarse_points: torch.Tensor,
        tgt_coarse_points: torch.Tensor,
        patch_radius: float = 0.15,
        max_points_per_patch: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Match fine points within each superpoint pair (batched Sinkhorn).

        Gathers local patches for all superpoint pairs, pads to uniform size,
        and runs a single batched Sinkhorn call per batch item.

        Returns:
            corr_src: (B, K_total, 3) source correspondence points
            corr_tgt: (B, K_total, 3) target correspondence points
            corr_weights: (B, K_total) validity mask
            assignment_matrices: list of assignment matrices for loss
        """
        B = src_fine_feats.shape[0]
        device = src_fine_feats.device
        D = src_fine_feats.shape[-1]
        Nc = src_sp_indices.shape[1]

        all_corr_src = []
        all_corr_tgt = []
        assignment_matrices = []

        for b in range(B):
            # Vectorized: distances from all fine points to all superpoint centers
            src_centers = src_coarse_points[b, src_sp_indices[b]]  # (Nc, 3)
            tgt_centers = tgt_coarse_points[b, tgt_sp_indices[b]]  # (Nc, 3)
            src_dists = torch.cdist(src_centers, src_fine_points[b])  # (Nc, N_fine)
            tgt_dists = torch.cdist(tgt_centers, tgt_fine_points[b])  # (Nc, M_fine)

            # For each pair: find points within radius, cap at max_points_per_patch
            src_in_patch = src_dists < patch_radius  # (Nc, N_fine)
            tgt_in_patch = tgt_dists < patch_radius  # (Nc, M_fine)
            src_counts = src_in_patch.sum(dim=1)  # (Nc,)
            tgt_counts = tgt_in_patch.sum(dim=1)  # (Nc,)

            valid_mask = (src_counts >= 1) & (tgt_counts >= 1)
            if not valid_mask.any():
                all_corr_src.append(src_coarse_points[b, :1])
                all_corr_tgt.append(tgt_coarse_points[b, :1])
                continue

            valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            n_valid = valid_idx.shape[0]

            # Determine padded patch sizes
            max_ns = min(src_counts[valid_idx].max().item(), max_points_per_patch)
            max_nt = min(tgt_counts[valid_idx].max().item(), max_points_per_patch)

            # Gather padded patches (fast indexing loop — not the bottleneck)
            src_patch_feats = torch.zeros(n_valid, max_ns, D, device=device)
            tgt_patch_feats = torch.zeros(n_valid, max_nt, D, device=device)
            src_patch_pts = torch.zeros(n_valid, max_ns, 3, device=device)
            tgt_patch_pts = torch.zeros(n_valid, max_nt, 3, device=device)
            src_patch_mask = torch.zeros(n_valid, max_ns, dtype=torch.bool, device=device)
            tgt_patch_mask = torch.zeros(n_valid, max_nt, dtype=torch.bool, device=device)

            for i, vi in enumerate(valid_idx):
                # Source: closest points within radius, capped
                s_in = src_in_patch[vi].nonzero(as_tuple=True)[0]
                if s_in.shape[0] > max_points_per_patch:
                    # Take closest ones
                    _, closest = src_dists[vi, s_in].topk(max_points_per_patch, largest=False)
                    s_in = s_in[closest]
                ns = s_in.shape[0]
                src_patch_feats[i, :ns] = src_fine_feats[b, s_in]
                src_patch_pts[i, :ns] = src_fine_points[b, s_in]
                src_patch_mask[i, :ns] = True

                # Target
                t_in = tgt_in_patch[vi].nonzero(as_tuple=True)[0]
                if t_in.shape[0] > max_points_per_patch:
                    _, closest = tgt_dists[vi, t_in].topk(max_points_per_patch, largest=False)
                    t_in = t_in[closest]
                nt = t_in.shape[0]
                tgt_patch_feats[i, :nt] = tgt_fine_feats[b, t_in]
                tgt_patch_pts[i, :nt] = tgt_fine_points[b, t_in]
                tgt_patch_mask[i, :nt] = True

            # Batched cost matrix: (n_valid, max_ns, max_nt)
            src_fn = F.normalize(src_patch_feats, dim=-1)
            tgt_fn = F.normalize(tgt_patch_feats, dim=-1)
            cost = torch.bmm(src_fn, tgt_fn.transpose(1, 2))

            # Mask invalid (padded) entries
            pair_mask = src_patch_mask.unsqueeze(2) & tgt_patch_mask.unsqueeze(1)
            cost = cost.masked_fill(~pair_mask, -1e9)

            # Single batched Sinkhorn call for all n_valid pairs
            Z = self.ot(cost)  # (n_valid, max_ns+1, max_nt+1)

            # Extract correspondences per pair
            batch_corr_src = []
            batch_corr_tgt = []
            for i in range(n_valid):
                ns = src_patch_mask[i].sum().item()
                nt = tgt_patch_mask[i].sum().item()
                Z_i = Z[i]
                assignment_matrices.append(Z_i)

                Z_core = Z_i[:ns, :nt]
                src_idx, tgt_idx = self._extract_mutual_topk(Z_core, self.top_k)
                if src_idx.numel() > 0:
                    batch_corr_src.append(src_patch_pts[i, src_idx])
                    batch_corr_tgt.append(tgt_patch_pts[i, tgt_idx])

            if batch_corr_src:
                all_corr_src.append(torch.cat(batch_corr_src, dim=0))
                all_corr_tgt.append(torch.cat(batch_corr_tgt, dim=0))
            else:
                all_corr_src.append(src_coarse_points[b, :1])
                all_corr_tgt.append(tgt_coarse_points[b, :1])

        # Pad to same length across batch
        max_corr = max(c.shape[0] for c in all_corr_src)
        corr_src = torch.zeros(B, max_corr, 3, device=device)
        corr_tgt = torch.zeros(B, max_corr, 3, device=device)
        corr_weights = torch.zeros(B, max_corr, device=device)

        for b in range(B):
            n = all_corr_src[b].shape[0]
            corr_src[b, :n] = all_corr_src[b]
            corr_tgt[b, :n] = all_corr_tgt[b]
            corr_weights[b, :n] = 1.0

        return corr_src, corr_tgt, corr_weights, assignment_matrices


# ---------------------------------------------------------------------------
# F) Local-to-Global Registration (LGR)
# ---------------------------------------------------------------------------

class LocalToGlobalRegistration(nn.Module):
    """Per-superpoint-pair hypothesis generation → select best → refine (Eq. 17).

    For each superpoint correspondence with enough point matches, computes a
    local rigid transform via weighted SVD. Scores each hypothesis by global
    inlier count. Refines the best hypothesis iteratively.
    """

    def __init__(
        self,
        acceptance_radius: float = 0.1,
        num_refinement_steps: int = 5,
        min_correspondences: int = 3,
    ):
        super().__init__()
        self.acceptance_radius = acceptance_radius
        self.num_refinement_steps = num_refinement_steps
        self.min_correspondences = min_correspondences

    def _weighted_svd(
        self,
        src_pts: torch.Tensor,
        tgt_pts: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kabsch SVD for single instance (no batch dim).

        Args:
            src_pts: (K, 3)
            tgt_pts: (K, 3)
            weights: (K,) optional

        Returns:
            R: (3, 3), t: (3,)
        """
        K = src_pts.shape[0]
        if weights is None:
            weights = torch.ones(K, device=src_pts.device)
        w = weights / (weights.sum() + 1e-8)

        c_src = (w.unsqueeze(-1) * src_pts).sum(0)
        c_tgt = (w.unsqueeze(-1) * tgt_pts).sum(0)

        src_c = src_pts - c_src
        tgt_c = tgt_pts - c_tgt

        H = (w.unsqueeze(-1) * src_c).t() @ tgt_c
        H = H + 1e-6 * torch.eye(3, device=H.device)

        U, S, Vt = torch.linalg.svd(H)
        det = torch.det(Vt.t() @ U.t())
        diag = torch.ones(3, device=H.device)
        diag[-1] = det.sign()
        R = Vt.t() @ torch.diag(diag) @ U.t()
        t = c_tgt - R @ c_src
        return R, t

    def _count_inliers(
        self, R: torch.Tensor, t: torch.Tensor,
        src_pts: torch.Tensor, tgt_pts: torch.Tensor,
    ) -> int:
        """Count inliers: points where ||R*src + t - tgt|| < acceptance_radius."""
        transformed = (R @ src_pts.t()).t() + t
        residuals = (transformed - tgt_pts).norm(dim=-1)
        return (residuals < self.acceptance_radius).sum().item()

    def forward(
        self,
        corr_src: torch.Tensor,
        corr_tgt: torch.Tensor,
        corr_weights: torch.Tensor,
        source_points: torch.Tensor,
        corr_sp_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            corr_src: (B, K, 3) source correspondence points
            corr_tgt: (B, K, 3) target correspondence points
            corr_weights: (B, K) validity mask
            source_points: (B, N, 3) full source cloud
            corr_sp_labels: (B, K) superpoint pair index per correspondence
                            (which superpoint pair each correspondence came from)

        Returns:
            R: (B, 3, 3) estimated rotation
            t: (B, 3) estimated translation
        """
        B = corr_src.shape[0]
        device = corr_src.device
        R_all = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        t_all = torch.zeros(B, 3, device=device)

        for b in range(B):
            mask = corr_weights[b] > 0.5
            src_pts = corr_src[b][mask]
            tgt_pts = corr_tgt[b][mask]

            if src_pts.shape[0] < self.min_correspondences:
                continue

            # Generate hypotheses per superpoint pair
            if corr_sp_labels is not None:
                labels = corr_sp_labels[b][mask]
                unique_labels = labels.unique()
            else:
                # If no labels, treat all correspondences as one group
                unique_labels = torch.tensor([0], device=device)
                labels = torch.zeros(src_pts.shape[0], device=device, dtype=torch.long)

            best_inliers = -1
            R_best = torch.eye(3, device=device)
            t_best = torch.zeros(3, device=device)

            for label in unique_labels:
                group_mask = labels == label
                if group_mask.sum() < self.min_correspondences:
                    continue

                group_src = src_pts[group_mask]
                group_tgt = tgt_pts[group_mask]

                # Compute local rigid transform for this superpoint pair
                R_cand, t_cand = self._weighted_svd(group_src, group_tgt)

                # Score by global inlier count across ALL correspondences
                inliers = self._count_inliers(R_cand, t_cand, src_pts, tgt_pts)

                if inliers > best_inliers:
                    best_inliers = inliers
                    R_best = R_cand
                    t_best = t_cand

            # If no hypothesis worked, use all correspondences
            if best_inliers <= 0:
                R_best, t_best = self._weighted_svd(src_pts, tgt_pts)

            # Refinement: re-estimate using inliers of best hypothesis
            for _ in range(self.num_refinement_steps):
                transformed = (R_best @ src_pts.t()).t() + t_best
                residuals = (transformed - tgt_pts).norm(dim=-1)
                inlier_mask = residuals < self.acceptance_radius
                if inlier_mask.sum() >= self.min_correspondences:
                    R_best, t_best = self._weighted_svd(
                        src_pts[inlier_mask], tgt_pts[inlier_mask]
                    )

            R_all[b] = R_best
            t_all[b] = t_best

        return R_all, t_all


# ---------------------------------------------------------------------------
# G) Top-level GeoTransformerBaseline
# ---------------------------------------------------------------------------

class GeoTransformerBaseline(nn.Module):
    """GeoTransformer for point cloud registration (Qin et al., CVPR 2022).

    Complete reimplementation in pure PyTorch.
    """

    def __init__(
        self,
        # KPConv-FPN (matching official: 3 stages, dims 128/256/512, input_dim=1)
        init_dim: int = 64,
        encoder_dims: Tuple[int, ...] = (128, 256, 512),
        decoder_dims: Tuple[int, ...] = (256, 256),
        base_radius: float = 0.075,
        base_sigma: float = 0.03,
        k_neighbors: int = 16,
        # Geometric Transformer
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_transformer_blocks: int = 3,
        sigma_d: float = 0.2,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        # Superpoint Matching
        num_correspondences: int = 128,
        # Point Matching
        num_sinkhorn_iters: int = 100,
        point_match_top_k: int = 3,
        patch_radius: float = 0.15,
        max_points_per_patch: int = 128,
        # LGR
        acceptance_radius: float = 0.1,
        num_refinement_steps: int = 5,
    ):
        super().__init__()

        self.patch_radius = patch_radius
        self.max_points_per_patch = max_points_per_patch

        # Shared backbone (applied independently to source and target)
        self.backbone = KPConvFPN(
            in_channels=1,
            init_dim=init_dim,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            base_radius=base_radius,
            base_sigma=base_sigma,
            k_neighbors=k_neighbors,
        )

        # Geometric Transformer (coarse encoder features are 512-dim -> in_proj -> 256)
        self.transformer = GeometricTransformerModule(
            input_dim=encoder_dims[-1],  # 512 (raw encoder output)
            output_dim=hidden_dim,       # 256
            hidden_dim=hidden_dim,       # 256
            num_heads=num_heads,
            num_blocks=num_transformer_blocks,
            sigma_d=sigma_d,
            sigma_a=sigma_a,
            angle_k=angle_k,
        )

        # Superpoint Matching
        self.superpoint_matching = SuperpointMatching(
            num_correspondences=num_correspondences,
        )

        # Point Matching
        self.point_matching = PointMatching(
            num_sinkhorn_iters=num_sinkhorn_iters,
            top_k=point_match_top_k,
        )

        # Local-to-Global Registration
        self.lgr = LocalToGlobalRegistration(
            acceptance_radius=acceptance_radius,
            num_refinement_steps=num_refinement_steps,
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            source: (B, N, 3) source point cloud
            target: (B, M, 3) target point cloud

        Returns:
            dict with keys:
                transformed_source: (B, N, 3)
                est_R: (B, 3, 3)
                est_t: (B, 3)
                superpoint_features_src: (B, N_s, D) — for circle loss
                superpoint_features_ref: (B, N_t, D) — for circle loss
                src_coarse_points: (B, N_s, 3)
                tgt_coarse_points: (B, N_t, 3)
                assignment_matrices: list — for point matching loss
        """
        B = source.shape[0]

        # 1. KPConv-FPN backbone (shared weights, applied independently)
        src_coarse_pts, src_coarse_feats, src_fine_pts, src_fine_feats = self.backbone(source)
        tgt_coarse_pts, tgt_coarse_feats, tgt_fine_pts, tgt_fine_feats = self.backbone(target)

        # 2. Geometric Transformer on superpoints
        src_trans_feats, tgt_trans_feats = self.transformer(
            src_coarse_feats, src_coarse_pts,
            tgt_coarse_feats, tgt_coarse_pts,
        )

        # 3. Superpoint Matching
        src_sp_idx, tgt_sp_idx, sp_scores = self.superpoint_matching(
            src_trans_feats, tgt_trans_feats,
        )

        # 4. Point Matching
        corr_src, corr_tgt, corr_weights, assignment_matrices = self.point_matching(
            src_fine_feats, tgt_fine_feats,
            src_fine_pts, tgt_fine_pts,
            src_sp_idx, tgt_sp_idx,
            src_coarse_pts, tgt_coarse_pts,
            patch_radius=self.patch_radius,
            max_points_per_patch=self.max_points_per_patch,
        )

        # 5. LGR → R, t
        R, t = self.lgr(corr_src, corr_tgt, corr_weights, source)

        # 6. Apply R, t to source
        transformed_source = torch.bmm(source, R.transpose(1, 2)) + t.unsqueeze(1)

        return {
            'transformed_source': transformed_source,
            'est_R': R,
            'est_t': t,
            'superpoint_features_src': src_trans_feats,
            'superpoint_features_ref': tgt_trans_feats,
            'src_coarse_points': src_coarse_pts,
            'tgt_coarse_points': tgt_coarse_pts,
            'assignment_matrices': assignment_matrices,
        }
