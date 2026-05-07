"""
ICP (Iterative Closest Point) wrapper for the evaluation pipeline.

Wraps Open3D's registration_icp as an nn.Module so it can be called from
evaluate_baselines.py like any other model.

ICP is an optimization-based method (no learned weights), so load_model()
will correctly skip weight loading (no entry in pretrained_mapping).
"""

import torch
import torch.nn as nn
import numpy as np
import open3d as o3d


class ICPWrapper(nn.Module):
    """Thin nn.Module wrapper around Open3D's point-to-point ICP."""

    def __init__(self, max_iterations=50, max_correspondence_distance=0.2):
        """
        Args:
            max_iterations: Maximum ICP iterations.
            max_correspondence_distance: Maximum distance for a point pair to
                be considered a correspondence. For UnitBall-normalized clouds,
                0.2 is a reasonable default.
        """
        super().__init__()
        self.max_iterations = max_iterations
        self.max_correspondence_distance = max_correspondence_distance

    def forward(self, source, target):
        """Run ICP registration on a single pair.

        Args:
            source: (1, N, 3) tensor — points to align (may be on any device)
            target: (1, M, 3) tensor — reference points (may be on any device)

        Returns:
            dict with transformed_source, est_R, est_t.
        """
        input_device = source.device

        # Open3D operates on CPU numpy float64 arrays
        src_np = source[0].detach().cpu().numpy().astype(np.float64)  # (N, 3)
        tgt_np = target[0].detach().cpu().numpy().astype(np.float64)  # (M, 3)

        # Convert to Open3D point clouds
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_np)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_np)

        # Run point-to-point ICP (no scaling)
        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            self.max_correspondence_distance,
            np.eye(4),  # identity init
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                with_scaling=False
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations
            ),
        )

        # Extract R, t from 4x4 homogeneous transform (source-to-target)
        T = result.transformation  # (4, 4)
        R = T[:3, :3]  # (3, 3)
        t = T[:3, 3]   # (3,)

        # Apply transform to source points
        aligned = (R @ src_np.T).T + t  # (N, 3)

        est_R = torch.from_numpy(R.astype(np.float32)).unsqueeze(0).to(input_device)
        est_t = torch.from_numpy(t.astype(np.float32)).unsqueeze(0).to(input_device)
        transformed = torch.from_numpy(aligned.astype(np.float32)).unsqueeze(0).to(input_device)

        return {
            'transformed_source': transformed,
            'est_R': est_R,
            'est_t': est_t,
        }
