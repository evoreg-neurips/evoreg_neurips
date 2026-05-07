"""
Lightweight truncated Chamfer distance for NDP optimization.

Replaces pytorch3d's knn_points with torch.cdist to avoid the heavy dependency.
"""

import torch


def compute_truncated_chamfer_distance(x, y, trunc=1e9):
    """Bidirectional truncated Chamfer distance (L1 norm convention).

    Args:
        x: (1, N, 3) source points
        y: (1, M, 3) target points
        trunc: truncation threshold on squared distances

    Returns:
        Scalar Chamfer loss (sum of mean sqrt distances in both directions).
    """
    dist_sq = torch.cdist(x, y).pow(2)  # (1, N, M) squared L2

    cham_x = dist_sq.min(2)[0].squeeze(0)  # (N,)
    cham_y = dist_sq.min(1)[0].squeeze(0)  # (M,)

    cham_x = cham_x[cham_x < trunc]
    cham_y = cham_y[cham_y < trunc]

    return cham_x.sqrt().mean() + cham_y.sqrt().mean()
