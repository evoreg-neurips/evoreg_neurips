"""Utility functions for DefTransNet: distance computation and graph construction."""

import torch


def pdist(x, p=2):
    """Pairwise squared distance within a point cloud.

    Args:
        x: (B, N, D) tensor
        p: distance type (1 or 2)

    Returns:
        (B, N, N) distance matrix
    """
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p == 2:
        xx = (x ** 2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


def pdist2(x, y, p=2):
    """Pairwise squared distance between two point clouds.

    Args:
        x: (B, N, D) tensor
        y: (B, M, D) tensor
        p: distance type (1 or 2)

    Returns:
        (B, N, M) distance matrix
    """
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p == 2:
        xx = (x ** 2).sum(dim=2).unsqueeze(2)
        yy = (y ** 2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist


def knn_graph(kpts, k, include_self=False):
    """Build k-nearest neighbor graph.

    Args:
        kpts: (B, N, D) point cloud
        k: number of neighbors
        include_self: whether to include self-loops

    Returns:
        ind: (B, N, k) neighbor indices
        dist: (B, N, N) masked distance matrix
        A: (B, N, N) adjacency matrix
    """
    B, N, D = kpts.shape
    device = kpts.device

    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1

    return ind, dist * A, A


def lbp_graph(kpts, k):
    """Build graph for Loopy Belief Propagation.

    Args:
        kpts: (B, N, D) point cloud (uses first sample in batch)
        k: number of neighbors

    Returns:
        edges: (E, 2) edge list
        edges_reverse_idx: (E,) reverse edge indices
    """
    device = kpts.device
    A = knn_graph(kpts, k, include_self=False)[2][0]
    edges = A.nonzero()
    edges_idx = torch.zeros_like(A).long()
    edges_idx[A.bool()] = torch.arange(edges.shape[0]).to(device)
    edges_reverse_idx = edges_idx.t()[A.bool()]
    return edges, edges_reverse_idx
