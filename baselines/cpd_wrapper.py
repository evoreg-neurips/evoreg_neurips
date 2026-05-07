"""
CPD (Coherent Point Drift) wrapper for the evaluation pipeline.

Wraps pycpd's RigidRegistration and DeformableRegistration as an nn.Module
so it can be called from evaluate_baselines.py like any other model.

CPD is an optimization-based method (no learned weights), so load_model()
will correctly skip weight loading (no entry in pretrained_mapping).
"""

import torch
import torch.nn as nn
import numpy as np
from pycpd import RigidRegistration, DeformableRegistration


class CPDWrapper(nn.Module):
    """Thin nn.Module wrapper around pycpd for rigid and deformable CPD."""

    def __init__(self, mode='rigid', max_iterations=50, tol=1e-3):
        super().__init__()
        assert mode in ('rigid', 'deformable'), f"Unknown CPD mode: {mode}"
        self.mode = mode
        self.max_iterations = max_iterations
        self.tol = tol

    def forward(self, source, target):
        """Run CPD registration on a single pair.

        Args:
            source: (1, N, 3) tensor — points to align (may be on any device)
            target: (1, M, 3) tensor — reference points (may be on any device)

        Returns:
            dict with transformed_source, and for rigid mode: est_R, est_t.
            For deformable mode, est_R and est_t are omitted so that the
            evaluation pipeline's SVD fallback estimates them from the
            source↔transformed_source point correspondence.
        """
        input_device = source.device

        # pycpd operates on CPU numpy float64 arrays
        src_np = source[0].detach().cpu().numpy().astype(np.float64)  # (N, 3)
        tgt_np = target[0].detach().cpu().numpy().astype(np.float64)  # (M, 3)

        if self.mode == 'rigid':
            reg = RigidRegistration(
                X=tgt_np, Y=src_np,
                max_iterations=self.max_iterations, tolerance=self.tol,
            )
            TY, (s, R, t) = reg.register()

            # pycpd rigid: TY = s * Y @ R + t  (row-vector form)
            # In column-vector form this is: TY_col = s * R^T @ Y_col + t
            # The pipeline convention is: target = R_gt @ source + t (column form)
            # So the rotation we report must be R^T to match the pipeline.
            est_R = torch.from_numpy(R.T.astype(np.float32)).unsqueeze(0).to(input_device)
            est_t = torch.from_numpy(t.astype(np.float32)).unsqueeze(0).to(input_device)
            transformed = torch.from_numpy(TY.astype(np.float32)).unsqueeze(0).to(input_device)

            return {
                'transformed_source': transformed,
                'est_R': est_R,
                'est_t': est_t,
            }
        else:
            reg = DeformableRegistration(
                X=tgt_np, Y=src_np,
                max_iterations=self.max_iterations, tolerance=self.tol,
            )
            TY, _ = reg.register()
            transformed = torch.from_numpy(TY.astype(np.float32)).unsqueeze(0).to(input_device)

            # Deformable CPD has no rigid R/t. By omitting est_R/est_t from
            # the result dict, evaluate_on_pair() will use its SVD (Kabsch)
            # fallback to estimate the best-fit rigid transform from the
            # source↔transformed_source point correspondence.
            return {
                'transformed_source': transformed,
            }
