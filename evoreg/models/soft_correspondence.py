"""
Soft correspondence and differentiable SVD for rigid alignment.

Computes soft correspondences between source and target point clouds
using projected per-point features, then solves for the optimal rigid
transform via weighted Kabsch SVD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FeatureProjector(nn.Module):
    """Projects per-point features to a lower-dim space and L2-normalizes."""

    def __init__(self, in_dim: int = 1024, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_features: (B, N, in_dim)
        Returns:
            (B, N, proj_dim) L2-normalized
        """
        B, N, D = point_features.shape
        out = self.net(point_features.reshape(B * N, D))  # (B*N, proj_dim)
        out = out.view(B, N, -1)
        return F.normalize(out, dim=-1)


class SoftCorrespondenceModule(nn.Module):
    """
    Computes soft correspondences between source and target point clouds.

    Uses projected per-point features to build a similarity matrix,
    applies Sinkhorn normalization, then produces pseudo-target
    correspondences and per-point confidence weights.
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        proj_dim: int = 256,
        temperature: float = 0.1,
        use_dual_softmax: bool = True,
        n_sinkhorn_iters: int = 3,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_dual_softmax = use_dual_softmax
        self.n_sinkhorn_iters = n_sinkhorn_iters

        self.src_projector = FeatureProjector(feat_dim, proj_dim)
        self.tgt_projector = FeatureProjector(feat_dim, proj_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        src_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src_feats:  (B, N, feat_dim) per-point features from encoder
            tgt_feats:  (B, M, feat_dim)
            src_points: (B, N, 3)
            tgt_points: (B, M, 3)
        Returns:
            tgt_corr:   (B, N, 3)  pseudo-target correspondences
            confidence:  (B, N)    per-point confidence weights
            S:           (B, N, M) soft assignment matrix
        """
        src_proj = self.src_projector(src_feats)  # (B, N, proj_dim)
        tgt_proj = self.tgt_projector(tgt_feats)  # (B, M, proj_dim)

        # Similarity matrix (cosine similarity scaled by temperature)
        tau = self.temperature.clamp(min=0.01, max=1.0)
        similarity = torch.bmm(src_proj, tgt_proj.transpose(1, 2)) / tau  # (B, N, M)

        if self.use_dual_softmax:
            # Sinkhorn normalization for balanced assignments
            log_S = similarity
            for _ in range(self.n_sinkhorn_iters):
                log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)   # row
                log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)   # col
            S = torch.exp(log_S)
        else:
            S = F.softmax(similarity, dim=-1)

        # Pseudo-target correspondences: weighted average of target points
        tgt_corr = torch.bmm(S, tgt_points)  # (B, N, 3)

        # Confidence: max assignment probability per source point
        confidence = S.max(dim=-1).values  # (B, N)

        return tgt_corr, confidence, S


class DifferentiableKabsch(nn.Module):
    """
    Differentiable weighted Kabsch algorithm for rigid alignment.

    Given source points, pseudo-target correspondences, and optional
    confidence weights, solves for the optimal rotation R and translation t
    via SVD of the weighted cross-covariance matrix.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        src_points: torch.Tensor,
        tgt_corr: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_points: (B, N, 3)
            tgt_corr:   (B, N, 3) pseudo-target correspondences
            weights:    (B, N)    optional confidence weights
        Returns:
            R: (B, 3, 3) rotation matrix
            t: (B, 3)    translation vector
        """
        B, N, _ = src_points.shape
        device = src_points.device

        if weights is None:
            weights = torch.ones(B, N, device=device)

        # Normalize weights
        w = weights / (weights.sum(dim=1, keepdim=True) + self.eps)  # (B, N)
        w_expand = w.unsqueeze(-1)  # (B, N, 1)

        # Weighted centroids
        c_src = (w_expand * src_points).sum(dim=1)   # (B, 3)
        c_tgt = (w_expand * tgt_corr).sum(dim=1)     # (B, 3)

        # Center point clouds
        src_centered = src_points - c_src.unsqueeze(1)  # (B, N, 3)
        tgt_centered = tgt_corr - c_tgt.unsqueeze(1)    # (B, N, 3)

        # Weighted cross-covariance: H = src^T @ diag(w) @ tgt
        H = torch.bmm(
            (w_expand * src_centered).transpose(1, 2),  # (B, 3, N)
            tgt_centered                                  # (B, N, 3)
        )  # (B, 3, 3)

        # Small perturbation for SVD stability
        H = H + self.eps * torch.eye(3, device=device).unsqueeze(0)

        # SVD
        U, S_vals, Vt = torch.linalg.svd(H)

        # Reflection correction: ensure det(R) = +1
        det = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
        diag = torch.ones(B, 3, device=device)
        diag[:, -1] = torch.sign(det)

        # R = V @ diag @ U^T
        R = torch.bmm(
            torch.bmm(Vt.transpose(1, 2), torch.diag_embed(diag)),
            U.transpose(1, 2)
        )  # (B, 3, 3)

        # t = c_tgt - R @ c_src
        t = c_tgt - torch.bmm(R, c_src.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        return R, t
