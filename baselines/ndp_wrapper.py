"""
NDP (Neural Deformation Pyramid) wrapper for the evaluation pipeline.

Wraps the vendored NDP implementation as an nn.Module so it can be called
from evaluate_baselines.py like any other model.

NDP is a test-time optimization method (no learned weights). For each pair,
a fresh Deformation_Pyramid is created and optimized coarse-to-fine against
bidirectional Chamfer distance.

Reference: Li et al., "Neural Deformation Pyramid", ECCV 2022.
"""

import torch
import torch.nn as nn
import numpy as np

from baselines.ndp.nets import Deformation_Pyramid
from baselines.ndp.loss import compute_truncated_chamfer_distance


class NDPWrapper(nn.Module):
    """Thin nn.Module wrapper around NDP test-time optimization."""

    def __init__(
        self,
        m=9,           # number of pyramid levels
        k0=-8,         # base frequency exponent for positional encoding
        depth=3,       # MLP depth per level
        width=128,     # MLP width per level
        iters=500,     # Adam iterations per level
        lr=0.01,       # learning rate
        samples=1024,  # subsample size for optimization
        motion='SE3',  # motion model: SE3, Sim3, or sflow
        rotation_format='axis_angle',
        patience=50,   # early stopping patience (iterations without improvement)
        device=None,   # override device (defaults to input tensor's device)
    ):
        super().__init__()
        self.m = m
        self.k0 = k0
        self.depth = depth
        self.width = width
        self.iters = iters
        self.lr = lr
        self.samples = samples
        self.motion = motion
        self.rotation_format = rotation_format
        self.patience = patience
        self._device = device

    def forward(self, source, target):
        """Run NDP test-time optimization on a single pair.

        Args:
            source: (1, N, 3) tensor — points to align
            target: (1, M, 3) tensor — reference points

        Returns:
            dict with 'transformed_source' (1, N, 3)
        """
        device = self._device or source.device
        src = source[0].detach().to(device).float()  # (N, 3)
        tgt = target[0].detach().to(device).float()  # (M, 3)

        try:
            transformed = self._optimize(src, tgt, device)
        except Exception:
            # Fallback: return source unchanged (like BCPD)
            transformed = src.clone()

        return {
            'transformed_source': transformed.unsqueeze(0).to(source.device),
        }

    def _optimize(self, src, tgt, device):
        """Core NDP coarse-to-fine optimization loop."""
        N = src.shape[0]
        M = tgt.shape[0]

        # Mean-center independently for numerical stability
        src_mean = src.mean(0, keepdim=True)
        tgt_mean = tgt.mean(0, keepdim=True)
        src_c = src - src_mean
        tgt_c = tgt - tgt_mean

        # Subsample for optimization (if needed)
        if N > self.samples:
            opt_idx = torch.randperm(N, device=device)[:self.samples]
            src_opt = src_c[opt_idx]
        else:
            src_opt = src_c

        if M > self.samples:
            tgt_idx = torch.randperm(M, device=device)[:self.samples]
            tgt_opt = tgt_c[tgt_idx]
        else:
            tgt_opt = tgt_c

        tgt_opt_batch = tgt_opt.unsqueeze(0)  # (1, S, 3)

        # Build fresh pyramid
        pyramid = Deformation_Pyramid(
            depth=self.depth,
            width=self.width,
            device=device,
            k0=self.k0,
            m=self.m,
            rotation_format=self.rotation_format,
            motion=self.motion,
        )

        # Coarse-to-fine: optimize one level at a time
        for level in range(self.m):
            pyramid.gradient_setup(level)
            params = list(pyramid.pyramid[level].parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr)

            best_loss = float('inf')
            patience_counter = 0

            for it in range(self.iters):
                optimizer.zero_grad()
                warped, _ = pyramid.warp(src_opt, max_level=level)
                loss = compute_truncated_chamfer_distance(
                    warped.unsqueeze(0), tgt_opt_batch
                )
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                if loss_val < best_loss - 1e-6:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # Final warp of ALL source points through the full pyramid
        with torch.no_grad():
            warped_full, _ = pyramid.warp(src_c)

        # Undo centering: apply target mean (not source mean) since
        # we want the result in target's coordinate frame
        return warped_full + tgt_mean
