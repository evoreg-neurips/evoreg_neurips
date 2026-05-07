"""
Rigid transformation head for SE(3) alignment.

Provides rigid alignment (rotation + translation) that can be combined with
non-rigid deformation for hybrid rigid/non-rigid registration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to 3x3 rotation matrix.

    Uses Gram-Schmidt orthogonalization on first two columns.
    Reference: "On the Continuity of Rotation Representations in Neural Networks"

    Args:
        r6d: (B, 6) tensor representing rotation

    Returns:
        R: (B, 3, 3) rotation matrices
    """
    batch_size = r6d.shape[0]

    # Reshape to two 3D vectors
    a1 = r6d[:, :3]  # First column (B, 3)
    a2 = r6d[:, 3:]  # Second column (B, 3)

    # Normalize first column
    b1 = F.normalize(a1, dim=1)

    # Gram-Schmidt: make a2 orthogonal to b1
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)

    # Cross product for third column
    b3 = torch.cross(b1, b2, dim=1)

    # Stack into rotation matrix
    R = torch.stack([b1, b2, b3], dim=2)  # (B, 3, 3)

    return R


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts rotation matrix to axis-angle representation.

    Args:
        R: (B, 3, 3) rotation matrices

    Returns:
        axis: (B, 3) rotation axis
        angle: (B,) rotation angle in radians
    """
    batch_size = R.shape[0]

    # Compute rotation angle from trace
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))

    # Handle small angles (use approximation)
    eps = 1e-6
    small_angle_mask = angle.abs() < eps

    # For non-small angles, extract axis from skew-symmetric part
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1)  # (B, 3)

    # Normalize axis
    axis_norm = axis.norm(dim=1, keepdim=True) + eps
    axis = axis / axis_norm

    # For small angles, set arbitrary axis (doesn't matter)
    axis[small_angle_mask] = torch.tensor([1.0, 0.0, 0.0], device=R.device)
    angle[small_angle_mask] = 0.0

    return axis, angle


class RigidHead(nn.Module):
    """
    Predicts SE(3) transformation (rotation + translation) from global features.

    Uses 6D rotation representation for continuous, differentiable rotations.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        """
        Initialize rigid transformation head.

        Args:
            feat_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Separate heads for rotation and translation
        self.rot_head = nn.Linear(hidden_dim, 6)    # 6D rotation representation
        self.trans_head = nn.Linear(hidden_dim, 3)  # Translation vector

        # Initialize rotation head to identity (6D rep of identity is [1,0,0,0,1,0])
        nn.init.zeros_(self.rot_head.weight)
        nn.init.constant_(self.rot_head.bias, 0.0)
        self.rot_head.bias.data[0] = 1.0  # [1, 0, 0, ...]
        self.rot_head.bias.data[4] = 1.0  # [..., 1, 0]

        # Initialize translation head to zero
        nn.init.zeros_(self.trans_head.weight)
        nn.init.zeros_(self.trans_head.bias)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict rigid transformation from features.

        Args:
            feat: (B, F) global feature vector

        Returns:
            R_pred: (B, 3, 3) predicted rotation matrix
            t_pred: (B, 3) predicted translation vector
        """
        h = self.mlp(feat)  # (B, hidden_dim)

        r6d = self.rot_head(h)      # (B, 6)
        t_pred = self.trans_head(h)  # (B, 3)

        R_pred = rotation_6d_to_matrix(r6d)  # (B, 3, 3)

        return R_pred, t_pred


def compute_rotation_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation loss using geodesic distance on SO(3).

    L_R = arccos((trace(R_delta) - 1) / 2)
    where R_delta = R_pred^T @ R_gt

    Args:
        R_pred: (B, 3, 3) predicted rotation
        R_gt: (B, 3, 3) ground truth rotation

    Returns:
        loss: (B,) rotation loss per sample
    """
    # Compute relative rotation: R_delta = R_pred^T @ R_gt
    R_delta = torch.bmm(R_pred.transpose(1, 2), R_gt)  # (B, 3, 3)

    # Compute trace
    trace = R_delta[:, 0, 0] + R_delta[:, 1, 1] + R_delta[:, 2, 2]  # (B,)

    # Geodesic distance: arccos((trace - 1) / 2)
    # Clamp for numerical stability
    cos_angle = torch.clamp((trace - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7)
    loss = torch.acos(cos_angle)  # (B,)

    return loss


def compute_translation_loss(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute translation loss using L2 norm.

    L_t = ||t_pred - t_gt||_2

    Args:
        t_pred: (B, 3) predicted translation
        t_gt: (B, 3) ground truth translation

    Returns:
        loss: (B,) translation loss per sample
    """
    loss = torch.norm(t_pred - t_gt, p=2, dim=1)  # (B,)
    return loss


def compute_rmse_loss(
    Y: torch.Tensor,
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor
) -> torch.Tensor:
    """
    Compute RMSE between predicted and ground truth rigid transformations.

    RMSE = sqrt(1/N * sum_i ||(R_pred * Y_i + t_pred) - (R_gt * Y_i + t_gt)||^2)

    Args:
        Y: (B, N, 3) source points
        R_pred: (B, 3, 3) predicted rotation
        t_pred: (B, 3) predicted translation
        R_gt: (B, 3, 3) ground truth rotation
        t_gt: (B, 3) ground truth translation

    Returns:
        loss: (B,) RMSE per sample
    """
    # Apply predicted transformation
    Y_pred = torch.bmm(Y, R_pred.transpose(1, 2)) + t_pred.unsqueeze(1)  # (B, N, 3)

    # Apply ground truth transformation
    Y_gt = torch.bmm(Y, R_gt.transpose(1, 2)) + t_gt.unsqueeze(1)  # (B, N, 3)

    # Compute RMSE
    squared_diff = ((Y_pred - Y_gt) ** 2).sum(dim=2)  # (B, N)
    rmse = torch.sqrt(squared_diff.mean(dim=1))  # (B,)

    return rmse


def apply_rigid_transform(
    Y: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Apply rigid transformation to point cloud.

    Y_rigid = R @ Y^T + t

    Args:
        Y: (B, N, 3) source points
        R: (B, 3, 3) rotation matrix
        t: (B, 3) translation vector

    Returns:
        Y_rigid: (B, N, 3) transformed points
    """
    Y_rigid = torch.bmm(Y, R.transpose(1, 2)) + t.unsqueeze(1)  # (B, N, 3)
    return Y_rigid


def compute_alignment_loss(
    X_hat: torch.Tensor,
    X: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """
    Compute alignment loss between predicted and target points.

    If X_hat and X have same number of points (correspondence available):
        L_align = sum_i ||X_hat_i - X_i||^2

    Otherwise (no correspondence), use k-nearest neighbor matching:
        L_align = sum_i sum_j ||X_hat_i - X_j||^2 for j in kNN(X_hat_i)

    Args:
        X_hat: (B, N, 3) predicted aligned source
        X: (B, M, 3) target points (M may differ from N)
        k: Number of nearest neighbors for kNN matching

    Returns:
        loss: (B,) alignment loss per sample
    """
    B, N, _ = X_hat.shape
    M = X.shape[1]

    if N == M:
        # Exact correspondence available
        loss = ((X_hat - X) ** 2).sum(dim=2).mean(dim=1)  # (B,)
    else:
        # Use k-nearest neighbor matching
        # Compute pairwise distances: (B, N, M)
        X_hat_expanded = X_hat.unsqueeze(2)  # (B, N, 1, 3)
        X_expanded = X.unsqueeze(1)          # (B, 1, M, 3)
        dists = ((X_hat_expanded - X_expanded) ** 2).sum(dim=3)  # (B, N, M)

        # Find k nearest neighbors
        k_actual = min(k, M)
        knn_dists, _ = torch.topk(dists, k_actual, dim=2, largest=False)  # (B, N, k)

        # Average over k neighbors and all points
        loss = knn_dists.mean(dim=(1, 2))  # (B,)

    return loss


def compute_displacement_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    Compute displacement regularization loss.

    L_disp = (sum_i ||delta_i||^2) / N

    Encourages small deformations.

    Args:
        delta: (B, N, 3) displacement field

    Returns:
        loss: (B,) displacement loss per sample
    """
    loss = (delta ** 2).sum(dim=2).mean(dim=1)  # (B,)
    return loss


def build_knn_graph(points: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Build k-nearest neighbor graph for points.

    Args:
        points: (B, N, 3) point cloud
        k: Number of nearest neighbors

    Returns:
        knn_indices: (B, N, k) indices of k nearest neighbors for each point
    """
    B, N, _ = points.shape

    # Compute pairwise distances
    points_expanded1 = points.unsqueeze(2)  # (B, N, 1, 3)
    points_expanded2 = points.unsqueeze(1)  # (B, 1, N, 3)
    dists = ((points_expanded1 - points_expanded2) ** 2).sum(dim=3)  # (B, N, N)

    # Find k+1 nearest neighbors (including self)
    k_actual = min(k + 1, N)
    _, knn_indices = torch.topk(dists, k_actual, dim=2, largest=False)  # (B, N, k+1)

    # Exclude self (first neighbor is always the point itself)
    knn_indices = knn_indices[:, :, 1:]  # (B, N, k)

    return knn_indices


def compute_deformation_loss(
    Y_rigid: torch.Tensor,
    X_hat: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """
    Compute deformation regularization loss.

    L_deform = sum_{i,j in E} ||(X_hat_i - X_hat_j) - (Y_rigid_i - Y_rigid_j)||^2 / |E|

    Preserves local structure by penalizing changes in edge lengths.

    Args:
        Y_rigid: (B, N, 3) rigidly aligned source points
        X_hat: (B, N, 3) final deformed points
        k: Number of nearest neighbors

    Returns:
        loss: (B,) deformation loss per sample
    """
    B, N, _ = Y_rigid.shape

    # Build kNN graph on Y_rigid
    knn_indices = build_knn_graph(Y_rigid, k)  # (B, N, k)

    # Gather neighbor points
    batch_indices = torch.arange(B, device=Y_rigid.device).view(B, 1, 1).expand(B, N, k)

    Y_rigid_neighbors = Y_rigid[batch_indices, knn_indices]  # (B, N, k, 3)
    X_hat_neighbors = X_hat[batch_indices, knn_indices]      # (B, N, k, 3)

    # Compute edge vectors
    Y_rigid_i = Y_rigid.unsqueeze(2)  # (B, N, 1, 3)
    X_hat_i = X_hat.unsqueeze(2)      # (B, N, 1, 3)

    Y_edges = Y_rigid_neighbors - Y_rigid_i  # (B, N, k, 3)
    X_edges = X_hat_neighbors - X_hat_i      # (B, N, k, 3)

    # Compute difference in edge vectors
    edge_diff = X_edges - Y_edges  # (B, N, k, 3)

    # Compute loss
    loss = (edge_diff ** 2).sum(dim=3).mean(dim=(1, 2))  # (B,)

    return loss


def compute_laplacian_loss(
    delta: torch.Tensor,
    Y_rigid: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """
    Compute Laplacian smoothness regularization.

    L_lap = sum_i ||delta_i - (sum_{j in N(i)} delta_j) / k||^2

    Encourages smooth displacement field (similar displacements for neighbors).

    Args:
        delta: (B, N, 3) displacement field
        Y_rigid: (B, N, 3) points for building kNN graph
        k: Number of nearest neighbors

    Returns:
        loss: (B,) Laplacian loss per sample
    """
    B, N, _ = delta.shape

    # Build kNN graph
    knn_indices = build_knn_graph(Y_rigid, k)  # (B, N, k)

    # Gather neighbor displacements
    batch_indices = torch.arange(B, device=delta.device).view(B, 1, 1).expand(B, N, k)
    delta_neighbors = delta[batch_indices, knn_indices]  # (B, N, k, 3)

    # Compute mean neighbor displacement
    delta_mean = delta_neighbors.mean(dim=2)  # (B, N, 3)

    # Compute Laplacian: delta_i - mean(delta_j for j in N(i))
    laplacian = delta - delta_mean  # (B, N, 3)

    # Compute loss
    loss = (laplacian ** 2).sum(dim=2).mean(dim=1)  # (B,)

    return loss


class RigidNonRigidLoss(nn.Module):
    """
    Combined loss for rigid + non-rigid registration.

    Computes both rigid alignment losses and non-rigid deformation regularization.
    """

    def __init__(
        self,
        lambda_rot: float = 1.0,
        lambda_trans: float = 1.0,
        lambda_rmse: float = 1.0,
        lambda_align: float = 1.0,
        lambda_disp: float = 0.01,
        lambda_deform: float = 0.1,
        lambda_lap: float = 0.1,
        k_neighbors: int = 5
    ):
        """
        Initialize combined loss.

        Args:
            lambda_rot: Weight for rotation loss
            lambda_trans: Weight for translation loss
            lambda_rmse: Weight for RMSE loss
            lambda_align: Weight for alignment loss
            lambda_disp: Weight for displacement regularization
            lambda_deform: Weight for deformation regularization
            lambda_lap: Weight for Laplacian smoothness
            k_neighbors: Number of nearest neighbors for graph-based losses
        """
        super().__init__()

        self.lambda_rot = lambda_rot
        self.lambda_trans = lambda_trans
        self.lambda_rmse = lambda_rmse
        self.lambda_align = lambda_align
        self.lambda_disp = lambda_disp
        self.lambda_deform = lambda_deform
        self.lambda_lap = lambda_lap
        self.k_neighbors = k_neighbors

    def forward(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        R_pred: torch.Tensor,
        t_pred: torch.Tensor,
        R_gt: torch.Tensor,
        t_gt: torch.Tensor,
        X_hat: torch.Tensor,
        delta: torch.Tensor
    ) -> dict:
        """
        Compute all losses.

        Args:
            Y: (B, N, 3) source points
            X: (B, M, 3) target points
            R_pred: (B, 3, 3) predicted rotation
            t_pred: (B, 3) predicted translation
            R_gt: (B, 3, 3) ground truth rotation
            t_gt: (B, 3) ground truth translation
            X_hat: (B, N, 3) final aligned points (after non-rigid)
            delta: (B, N, 3) displacement field

        Returns:
            losses: Dictionary of individual and total losses
        """
        # Rigid losses
        L_rot = compute_rotation_loss(R_pred, R_gt).mean()
        L_trans = compute_translation_loss(t_pred, t_gt).mean()
        L_rmse = compute_rmse_loss(Y, R_pred, t_pred, R_gt, t_gt).mean()

        rigid_loss = (
            self.lambda_rot * L_rot +
            self.lambda_trans * L_trans +
            self.lambda_rmse * L_rmse
        )

        # Compute Y_rigid for non-rigid losses
        Y_rigid = apply_rigid_transform(Y, R_pred, t_pred)

        # Non-rigid losses
        L_align = compute_alignment_loss(X_hat, X, k=self.k_neighbors).mean()
        L_disp = compute_displacement_loss(delta).mean()
        L_deform = compute_deformation_loss(Y_rigid, X_hat, k=self.k_neighbors).mean()
        L_lap = compute_laplacian_loss(delta, Y_rigid, k=self.k_neighbors).mean()

        non_rigid_loss = (
            self.lambda_align * L_align +
            self.lambda_disp * L_disp +
            self.lambda_deform * L_deform +
            self.lambda_lap * L_lap
        )

        total_loss = rigid_loss + non_rigid_loss

        return {
            'total_loss': total_loss,
            'rigid_loss': rigid_loss,
            'non_rigid_loss': non_rigid_loss,
            'L_rot': L_rot,
            'L_trans': L_trans,
            'L_rmse': L_rmse,
            'L_align': L_align,
            'L_disp': L_disp,
            'L_deform': L_deform,
            'L_lap': L_lap
        }
