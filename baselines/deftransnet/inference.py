"""Smooth Loopy Belief Propagation (sLBP) inference for DefTransNet."""

import torch
import torch.nn.functional as F

from .utils import pdist2, knn_graph, lbp_graph


def smooth_lbp(kpts_fixed, kpts_moving, net, k=10, k1=128, n_iter=5,
               cost_scale=50.0, alpha=0.1):
    """Compute displacement field via Smooth Loopy Belief Propagation.

    Uses DefTransNet features and kNN candidate matching to estimate per-point
    displacements from fixed (source) to moving (target) point cloud.

    Args:
        kpts_fixed: (1, N, 3) or (N, 3) source points
        kpts_moving: (1, M, 3) or (M, 3) target points
        net: DefTransNet model (feature extractor)
        k: kNN for feature extraction graph
        k1: number of displacement candidates
        n_iter: sLBP message passing iterations
        cost_scale: regularization cost scale
        alpha: softmax temperature for final displacement

    Returns:
        (1, N, 3) predicted displacement field
    """
    # Ensure batch dimension
    if kpts_fixed.dim() == 2:
        kpts_fixed = kpts_fixed.unsqueeze(0)
    if kpts_moving.dim() == 2:
        kpts_moving = kpts_moving.unsqueeze(0)

    N = kpts_fixed.shape[1]
    M = kpts_moving.shape[1]
    device = kpts_fixed.device

    # Clamp k1 to available target points
    k1_actual = min(k1, M)

    # Extract features
    kpts_fixed_feat, kpts_moving_feat = net(kpts_fixed, kpts_moving, k)

    # Find k1 nearest neighbors in feature space
    dist = pdist2(kpts_fixed_feat, kpts_moving_feat)
    ind = (-dist).topk(k1_actual, dim=-1)[1]  # (1, N, k1)

    # Candidate displacements: target_nn - source
    candidates = (
        -kpts_fixed.view(1, N, 1, 3)
        + kpts_moving[:, ind.view(-1), :].view(1, N, k1_actual, 3)
    )  # (1, N, k1, 3)

    # Unary cost: feature distance
    candidates_cost = (
        kpts_fixed_feat.view(1, N, 1, -1)
        - kpts_moving_feat[:, ind.view(-1), :].view(1, N, k1_actual, -1)
    ).pow(2).mean(3)  # (1, N, k1)

    # Build LBP graph on source points
    edges, edges_reverse_idx = lbp_graph(kpts_fixed, k)

    # Message passing
    messages = torch.zeros((edges.shape[0], k1_actual), device=device)
    candidates_edges0 = candidates[0, edges[:, 0], :, :]  # (E, k1, 3)
    candidates_edges1 = candidates[0, edges[:, 1], :, :]  # (E, k1, 3)

    for _ in range(n_iter):
        # Aggregate messages at each node
        temp_message = torch.zeros((N, k1_actual), device=device).scatter_add_(
            0, edges[:, 1].view(-1, 1).expand(-1, k1_actual), messages
        )

        # Data cost + aggregated messages for edge source nodes
        multi_data_cost = torch.gather(
            temp_message + candidates_cost.squeeze(),
            0,
            edges[:, 0].view(-1, 1).expand(-1, k1_actual),
        )

        # Subtract reverse message (avoid double-counting)
        reverse_messages = torch.gather(
            messages, 0, edges_reverse_idx.view(-1, 1).expand(-1, k1_actual)
        )
        multi_data_cost = multi_data_cost - reverse_messages

        # Compute new messages via min over candidates
        messages = torch.zeros_like(multi_data_cost)
        # Process in chunks for memory efficiency
        unroll_factor = min(32, multi_data_cost.shape[0])
        if unroll_factor > 0:
            split = torch.chunk(torch.arange(multi_data_cost.shape[0]), unroll_factor)
            for i in range(len(split)):
                messages[split[i]] = torch.min(
                    multi_data_cost[split[i]].unsqueeze(1)
                    + cost_scale
                    * (candidates_edges0[split[i]].unsqueeze(1)
                       - candidates_edges1[split[i]].unsqueeze(2))
                    .pow(2)
                    .sum(3),
                    2,
                )[0]

    # Final belief = messages + unary cost
    reg_candidates_cost = (temp_message + candidates_cost.view(-1, k1_actual)).unsqueeze(0)

    # Softmax weighted displacement
    sm = F.softmax(
        alpha * reg_candidates_cost.view(1, N, -1), 2
    ).unsqueeze(3)  # (1, N, k1, 1)

    kpts_fixed_disp_pred = (candidates * sm).sum(2)  # (1, N, 3)

    return kpts_fixed_disp_pred
