"""
DefTransNet wrapper for the evaluation pipeline.

Wraps the vendored DefTransNet + sLBP inference as an nn.Module so it can
be called from evaluate_baselines.py like any other model.

DefTransNet extracts features via graph convolution + transformer attention,
then sLBP (Smooth Loopy Belief Propagation) computes per-point displacements.

Reference: Monji-Azad et al., "DefTransNet: A Transformer-based Method for
Non-Rigid Point Cloud Registration", 2025.
"""

import torch
import torch.nn as nn

from baselines.deftransnet import DefTransNet, smooth_lbp


class DefTransNetWrapper(nn.Module):
    """nn.Module wrapper bundling DefTransNet + sLBP inference."""

    def __init__(self, k=10, k1=128, slbp_iter=5, slbp_cost_scale=50.0,
                 slbp_alpha=0.1, emb_dims=64, n_heads=4, ff_dims=1024):
        super().__init__()
        self.net = DefTransNet(k=k, emb_dims=emb_dims, n_heads=n_heads, ff_dims=ff_dims)
        self.k = k
        self.k1 = k1
        self.slbp_iter = slbp_iter
        self.slbp_cost_scale = slbp_cost_scale
        self.slbp_alpha = slbp_alpha

    def forward(self, source, target):
        """Run DefTransNet + sLBP on a point cloud pair.

        Args:
            source: (B, N, 3) tensor — points to align
            target: (B, M, 3) tensor — reference points

        Returns:
            dict with 'transformed_source' (B, N, 3)
        """
        B = source.shape[0]
        displacements = []
        for i in range(B):
            disp = smooth_lbp(
                kpts_fixed=source[i:i+1],
                kpts_moving=target[i:i+1],
                net=self.net,
                k=self.k,
                k1=self.k1,
                n_iter=self.slbp_iter,
                cost_scale=self.slbp_cost_scale,
                alpha=self.slbp_alpha,
            )
            displacements.append(disp.squeeze(0))
        disp_batch = torch.stack(displacements, dim=0)
        transformed = source + disp_batch
        return {
            'transformed_source': transformed,
        }
