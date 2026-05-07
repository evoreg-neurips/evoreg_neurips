"""
VFA-style cross-attention generator for point cloud registration.

For each source point (Query), attends over its k nearest neighbors in the
target (Keys). The attention-weighted sum of neighbor positions gives the
displacement, analogous to VFA's radial vector field attention.

Usage:
    generator = CrossAttentionGenerator(feature_dim=64, k=8)
    deformed = generator(source, target)  # (B, N, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionGenerator(nn.Module):
    """
    VFA-style cross-attention generator for point cloud registration.

    For each source point:
      1. Extract a feature vector (Query) via a shared MLP.
      2. Find its k nearest neighbors in the target point cloud.
      3. Extract feature vectors (Keys) for those neighbors.
      4. Compute dot-product attention between Q and each K.
      5. Take the attention-weighted sum of neighbor positions (Values).
      6. The result is the predicted target position for that source point.

    Args:
        feature_dim: Dimension of the per-point feature projection.
        k: Number of nearest neighbors in the target to attend over.
        use_residual: If True, output = source + displacement.
                      If False, output is the raw attended position.
    """

    def __init__(self, feature_dim: int = 64, k: int = 8, use_residual: bool = True):
        super().__init__()
        self.k = k
        self.use_residual = use_residual

        # Shared MLP: projects raw 3D coordinates to feature space.
        # Both source and target points go through the same projection,
        # mirroring VFA's shared encoder before the matching step.
        self.point_proj = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Learnable temperature scalar (analogous to VFA's beta).
        # Initialized small so early attention is diffuse.
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: (B, N, 3) — the point cloud to be registered.
            target: (B, M, 3) — the fixed reference point cloud.

        Returns:
            Deformed source: (B, N, 3)
        """
        B, N, _ = source.shape
        M = target.shape[1]

        # --- Step 1: project all points to feature space ---
        # Flatten to (B*N, 3) for the shared MLP, then restore shape.
        src_feat = self.point_proj(source.view(B * N, 3)).view(B, N, -1)   # (B, N, F)
        tgt_feat = self.point_proj(target.view(B * M, 3)).view(B, M, -1)   # (B, M, F)

        # --- Step 2: find k nearest neighbors in target for each source point ---
        # Pairwise squared L2 distances: (B, N, M)
        # ||s - t||^2 = ||s||^2 + ||t||^2 - 2 s·t
        src_sq = (source ** 2).sum(-1, keepdim=True)          # (B, N, 1)
        tgt_sq = (target ** 2).sum(-1).unsqueeze(1)           # (B, 1, M)
        cross  = torch.bmm(source, target.transpose(1, 2))    # (B, N, M)
        dists  = src_sq + tgt_sq - 2 * cross                  # (B, N, M)

        # knn_idx: (B, N, k) — indices of the k closest target points
        _, knn_idx = dists.topk(self.k, dim=-1, largest=False)

        # --- Step 3: gather neighbor features (Keys) and positions (Values) ---
        # Expand index for gathering along the M dimension.
        F_dim = src_feat.shape[-1]
        idx_feat = knn_idx.unsqueeze(-1).expand(B, N, self.k, F_dim)  # (B, N, k, F)
        idx_pos  = knn_idx.unsqueeze(-1).expand(B, N, self.k, 3)      # (B, N, k, 3)

        # tgt_feat and target need an N dimension inserted before gathering.
        K = tgt_feat.unsqueeze(1).expand(B, N, M, F_dim).gather(2, idx_feat)  # (B, N, k, F)
        V = target.unsqueeze(1).expand(B, N, M, 3).gather(2, idx_pos)         # (B, N, k, 3)

        # --- Step 4: dot-product attention (VFA's Attention module) ---
        # Q: (B, N, 1, F),  K: (B, N, k, F)
        Q = src_feat.unsqueeze(2)                                  # (B, N, 1, F)
        temperature = self.log_temp.exp() * (F_dim ** 0.5)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / temperature  # (B, N, 1, k)
        attn = F.softmax(attn, dim=-1)                             # (B, N, 1, k)

        # --- Step 5: weighted sum of neighbor positions ---
        attended_pos = torch.matmul(attn, V).squeeze(2)            # (B, N, 3)

        # --- Step 6: residual output ---
        if self.use_residual:
            return source + (attended_pos - source)  # equivalent to attended_pos,
                                                     # but written as source + displacement
                                                     # to match PointCloudGenerator convention
        return attended_pos


if __name__ == "__main__":
    B, N, M = 2, 512, 512
    source = torch.randn(B, N, 3)
    target = torch.randn(B, M, 3)

    model = CrossAttentionGenerator(feature_dim=64, k=8)
    out = model(source, target)
    print(f"source: {source.shape} -> deformed: {out.shape}")

    loss = out.sum()
    loss.backward()
    print("Gradient check passed.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
