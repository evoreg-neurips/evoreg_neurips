"""
GeoTransformer losses (Qin et al., CVPR 2022).

Implements:
    - OverlapAwareCircleLoss: metric learning on superpoint features (Eq. 18)
    - PointMatchingLoss: NLL on Sinkhorn assignment matrix (Eq. 19)
    - GeoTransformerLoss: combined L = L_oc + L_p + L_rt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class OverlapAwareCircleLoss(nn.Module):
    """Overlap-aware circle loss for superpoint feature learning (Eq. 18).

    Positive pairs: superpoints with >= overlap_threshold overlap, weighted by sqrt(overlap).
    Negative pairs: non-overlapping superpoints.

    Circle loss formula:
        L = softplus(logsumexp(s * alpha_p * (d_p - delta_p) + log(w_p))
                    + logsumexp(s * alpha_n * (delta_n - d_n))) / s
    where alpha_p = clamp(d_p - O_p, min=0), alpha_n = clamp(O_n - d_n, min=0)
    """

    def __init__(
        self,
        pos_margin: float = 0.1,
        neg_margin: float = 1.4,
        pos_optimal: float = 0.1,
        neg_optimal: float = 1.4,
        log_scale: float = 24.0,
        overlap_threshold: float = 0.1,
    ):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale
        self.overlap_threshold = overlap_threshold

    def forward(
        self,
        src_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        overlaps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src_feats: (B, N_s, D) source superpoint features
            tgt_feats: (B, N_t, D) target superpoint features
            overlaps: (B, N_s, N_t) overlap ratio between superpoint patches

        Returns:
            loss: scalar
        """
        B = src_feats.shape[0]
        total_loss = torch.tensor(0.0, device=src_feats.device)
        n_valid = 0

        for b in range(B):
            src_f = F.normalize(src_feats[b], dim=-1)  # (N_s, D)
            tgt_f = F.normalize(tgt_feats[b], dim=-1)  # (N_t, D)

            # Pairwise L2 distances (full N_s x N_t matrix)
            feat_dists = torch.cdist(src_f, tgt_f, p=2.0)  # (N_s, N_t)

            pos_mask = overlaps[b] >= self.overlap_threshold  # (N_s, N_t)
            neg_mask = ~pos_mask

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # Overlap-based positive scales: sqrt(overlap) — matching official pos_scales
            pos_scales = torch.sqrt(overlaps[b].clamp(min=1e-8))  # (N_s, N_t)

            # Row-wise circle loss (source as anchor, target as pairs)
            row_loss = self._circle_loss_rows(feat_dists, pos_mask, neg_mask, pos_scales)
            # Column-wise (target as anchor, source as pairs)
            col_loss = self._circle_loss_rows(
                feat_dists.t(), pos_mask.t(), neg_mask.t(), pos_scales.t()
            )

            batch_loss = (row_loss + col_loss) / 2.0
            if batch_loss.isfinite():
                total_loss = total_loss + batch_loss
                n_valid += 1

        return total_loss / max(n_valid, 1)

    def _circle_loss_rows(
        self, feat_dists: torch.Tensor, pos_mask: torch.Tensor,
        neg_mask: torch.Tensor, pos_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized circle loss matching official implementation.

        Official pattern (from circle_loss.py):
            pos_weights = feat_dists - pos_optimal  (clamped >=0)
            pos_weights *= pos_scales  (overlap multiplied into alpha, inside exponent)
            pos_weights = pos_weights.detach()
            loss_pos = logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights)

        Args:
            feat_dists: (N_a, N_b) pairwise distances
            pos_mask: (N_a, N_b) positive pair mask
            neg_mask: (N_a, N_b) negative pair mask
            pos_scales: (N_a, N_b) overlap-based scales for positives
        """
        NEG_INF = -1e9

        # Positive weights: alpha_p = clamp(d - O_p, min=0) * pos_scales
        pos_weights = feat_dists - self.pos_optimal
        pos_weights = pos_weights.clamp(min=0.0)
        pos_weights = pos_weights * pos_scales  # overlap weight multiplied into alpha
        pos_weights = pos_weights.detach()

        # Negative weights: alpha_n = clamp(O_n - d, min=0)
        neg_weights = self.neg_optimal - feat_dists
        neg_weights = neg_weights.clamp(min=0.0)
        neg_weights = neg_weights.detach()

        # Positive term: logsumexp over positives per row
        pos_logits = self.log_scale * (feat_dists - self.pos_margin) * pos_weights
        pos_logits = pos_logits + NEG_INF * (~pos_mask).float()  # mask out non-positives
        loss_pos = torch.logsumexp(pos_logits, dim=-1)  # (N_a,)

        # Negative term: logsumexp over negatives per row
        neg_logits = self.log_scale * (self.neg_margin - feat_dists) * neg_weights
        neg_logits = neg_logits + NEG_INF * (~neg_mask).float()
        loss_neg = torch.logsumexp(neg_logits, dim=-1)  # (N_a,)

        # Only keep anchors with both positives and negatives
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid = has_pos & has_neg

        if not valid.any():
            return torch.tensor(0.0, device=feat_dists.device)

        loss = F.softplus(loss_pos[valid] + loss_neg[valid]) / self.log_scale
        return loss.mean()


class PointMatchingLoss(nn.Module):
    """Negative log-likelihood loss on Sinkhorn assignment (Eq. 19).

    For GT-matched pairs: penalize -log(Z[i, j_gt])
    For unmatched source points: penalize -log(Z[i, dustbin])
    For unmatched target points: penalize -log(Z[dustbin, j])
    """

    def __init__(self, matching_radius: float = 0.05):
        super().__init__()
        self.matching_radius = matching_radius

    def forward(
        self,
        assignment_matrices: List[torch.Tensor],
        src_patch_points_list: Optional[List[torch.Tensor]] = None,
        tgt_patch_points_list: Optional[List[torch.Tensor]] = None,
        gt_R: Optional[torch.Tensor] = None,
        gt_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            assignment_matrices: list of (N_i+1, M_i+1) Sinkhorn outputs
            src_patch_points_list: list of (N_i, 3) local source points per patch
            tgt_patch_points_list: list of (M_i, 3) local target points per patch
            gt_R: (3, 3) ground truth rotation (single sample, not batched)
            gt_t: (3,) ground truth translation

        Returns:
            loss: scalar NLL loss
        """
        if not assignment_matrices:
            return torch.tensor(0.0, device=assignment_matrices[0].device
                                if assignment_matrices else 'cpu')

        total_loss = torch.tensor(0.0, device=assignment_matrices[0].device)
        n_valid = 0

        for idx, Z in enumerate(assignment_matrices):
            N = Z.shape[0] - 1
            M = Z.shape[1] - 1

            if N == 0 or M == 0:
                continue

            log_Z = torch.log(Z.clamp(min=1e-12))

            # If GT info available, compute proper GT-aware NLL
            if (src_patch_points_list is not None and
                    tgt_patch_points_list is not None and
                    gt_R is not None and gt_t is not None and
                    idx < len(src_patch_points_list)):
                src_pts = src_patch_points_list[idx]  # (N, 3)
                tgt_pts = tgt_patch_points_list[idx]  # (M, 3)

                # Transform source by GT
                src_transformed = (gt_R @ src_pts.t()).t() + gt_t  # (N, 3)

                # Find GT matches within matching_radius
                pair_dists = torch.cdist(
                    src_transformed.unsqueeze(0), tgt_pts.unsqueeze(0)
                ).squeeze(0)  # (N, M)

                min_dists, gt_tgt_idx = pair_dists.min(dim=1)  # (N,)
                matched_src = min_dists < self.matching_radius  # (N,) bool

                nll = torch.tensor(0.0, device=Z.device)
                n_terms = 0

                # Matched source points: -log(Z[i, j_gt])
                if matched_src.any():
                    matched_i = matched_src.nonzero(as_tuple=True)[0]
                    matched_j = gt_tgt_idx[matched_src]
                    nll = nll - log_Z[matched_i, matched_j].mean()
                    n_terms += 1

                # Unmatched source points: -log(Z[i, dustbin])
                unmatched_src = ~matched_src
                if unmatched_src.any():
                    unmatched_i = unmatched_src.nonzero(as_tuple=True)[0]
                    nll = nll - log_Z[unmatched_i, M].mean()  # M = dustbin column
                    n_terms += 1

                # Unmatched target points: -log(Z[dustbin, j])
                matched_tgt_set = set(gt_tgt_idx[matched_src].tolist())
                all_tgt = set(range(M))
                unmatched_tgt = list(all_tgt - matched_tgt_set)
                if unmatched_tgt:
                    unmatched_j = torch.tensor(unmatched_tgt, device=Z.device)
                    nll = nll - log_Z[N, unmatched_j].mean()  # N = dustbin row
                    n_terms += 1

                if n_terms > 0:
                    total_loss = total_loss + nll / n_terms
                    n_valid += 1
            else:
                # Fallback without GT: encourage peaked assignments via entropy
                Z_core = Z[:N, :M]
                entropy = -(Z_core * torch.log(Z_core.clamp(min=1e-12))).sum() / max(N * M, 1)
                total_loss = total_loss + entropy
                n_valid += 1

        return total_loss / max(n_valid, 1)


class GeoTransformerLoss(nn.Module):
    """Combined GeoTransformer loss: L = L_oc + L_p + L_rt.

    L_oc: overlap-aware circle loss on superpoint features
    L_p: point matching loss on Sinkhorn assignments
    L_rt: R/t supervision (geodesic rotation + translation L2)
    """

    def __init__(
        self,
        pos_margin: float = 0.1,
        neg_margin: float = 1.4,
        overlap_threshold: float = 0.1,
        matching_radius: float = 0.05,
        lambda_circle: float = 1.0,
        lambda_point: float = 1.0,
        lambda_rot: float = 1.0,
        lambda_trans: float = 1.0,
    ):
        super().__init__()
        self.circle_loss = OverlapAwareCircleLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            overlap_threshold=overlap_threshold,
        )
        self.point_loss = PointMatchingLoss(matching_radius=matching_radius)
        self.lambda_circle = lambda_circle
        self.lambda_point = lambda_point
        self.lambda_rot = lambda_rot
        self.lambda_trans = lambda_trans

    def geodesic_rotation_error(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """Geodesic rotation distance in radians."""
        R_diff = torch.bmm(R_pred, R_gt.transpose(1, 2))
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos_angle).mean()

    def forward(
        self,
        model_output: Dict,
        gt_R: torch.Tensor,
        gt_t: torch.Tensor,
        overlaps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_output: dict from GeoTransformerBaseline.forward()
            gt_R: (B, 3, 3) ground truth rotation
            gt_t: (B, 3) ground truth translation
            overlaps: (B, N_s, N_t) patch overlap ratios (if None, circle loss skipped)

        Returns:
            dict with 'total', 'circle', 'point', 'rot', 'trans' losses
        """
        losses = {}

        # Circle loss
        if overlaps is not None:
            losses['circle'] = self.lambda_circle * self.circle_loss(
                model_output['superpoint_features_src'],
                model_output['superpoint_features_ref'],
                overlaps,
            )
        else:
            losses['circle'] = torch.tensor(0.0, device=gt_R.device)

        # Point matching loss
        losses['point'] = self.lambda_point * self.point_loss(
            model_output['assignment_matrices'],
        )

        # R/t supervision
        losses['rot'] = self.lambda_rot * self.geodesic_rotation_error(
            model_output['est_R'], gt_R
        )
        # Translation L2 error (Euclidean distance, not MSE)
        losses['trans'] = self.lambda_trans * (
            model_output['est_t'] - gt_t
        ).norm(dim=-1).mean()

        losses['total'] = losses['circle'] + losses['point'] + losses['rot'] + losses['trans']
        return losses
