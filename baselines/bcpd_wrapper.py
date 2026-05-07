"""
BCPD (Bayesian Coherent Point Drift) wrapper for the evaluation pipeline.

Wraps probreg's registration_bcpd as an nn.Module so it can be called from
evaluate_baselines.py like any other model.

BCPD is an optimization-based method (no learned weights), so load_model()
will correctly skip weight loading (no entry in pretrained_mapping).
"""

import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from probreg import bcpd


class BCPDWrapper(nn.Module):
    """Thin nn.Module wrapper around probreg's BCPD."""

    def __init__(self, w=0.0, maxiter=50, tol=1e-3, lmd=2.0, k=50):
        super().__init__()
        self.w = w
        self.maxiter = maxiter
        self.tol = tol
        self.lmd = lmd
        self.k = k  # low-rank kernel approximation (default 1e20 overflows at 1024 pts)

    def forward(self, source, target):
        """Run BCPD registration on a single pair.

        Args:
            source: (1, N, 3) tensor — points to align (may be on any device)
            target: (1, M, 3) tensor — reference points (may be on any device)

        Returns:
            dict with transformed_source. est_R/est_t are omitted so that the
            evaluation pipeline's SVD (Kabsch) fallback estimates the best-fit
            rigid transform from the source↔transformed_source correspondence.
        """
        input_device = source.device

        # probreg operates on CPU numpy float64 arrays via Open3D point clouds
        src_np = source[0].detach().cpu().numpy().astype(np.float64)  # (N, 3)
        tgt_np = target[0].detach().cpu().numpy().astype(np.float64)  # (M, 3)

        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(src_np)

        tgt_pc = o3d.geometry.PointCloud()
        tgt_pc.points = o3d.utility.Vector3dVector(tgt_np)

        try:
            # registration_bcpd returns a CombinedTransformation directly
            tf = bcpd.registration_bcpd(
                source=src_pc, target=tgt_pc,
                w=self.w, maxiter=self.maxiter, tol=self.tol,
                lmd=self.lmd, k=self.k,
            )
            transformed_np = np.asarray(tf.transform(src_np))

            # Safety: if BCPD diverged (NaN/Inf), fall back to source
            if np.any(~np.isfinite(transformed_np)):
                transformed_np = src_np.copy()
        except np.linalg.LinAlgError:
            # Kernel matrix can be singular when points have exact duplicates
            # or perfectly regular spacing. Add tiny jitter and retry.
            src_jitter = src_np + np.random.randn(*src_np.shape) * 1e-4
            src_pc_j = o3d.geometry.PointCloud()
            src_pc_j.points = o3d.utility.Vector3dVector(src_jitter)
            try:
                tf = bcpd.registration_bcpd(
                    source=src_pc_j, target=tgt_pc,
                    w=self.w, maxiter=self.maxiter, tol=self.tol,
                    lmd=self.lmd, k=self.k,
                )
                transformed_np = np.asarray(tf.transform(src_np))
                if np.any(~np.isfinite(transformed_np)):
                    transformed_np = src_np.copy()
            except np.linalg.LinAlgError:
                transformed_np = src_np.copy()

        transformed = torch.from_numpy(transformed_np.astype(np.float32)).unsqueeze(0).to(input_device)

        return {
            'transformed_source': transformed,
        }
