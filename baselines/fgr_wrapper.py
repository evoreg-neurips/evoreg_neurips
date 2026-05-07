"""
FGR (Fast Global Registration) wrapper for the evaluation pipeline.

Wraps Open3D's registration_fgr_based_on_feature_matching as an nn.Module
so it can be called from evaluate_baselines.py like any other model.

FGR is an optimization-based method (no learned weights) that uses FPFH
feature descriptors for global point cloud registration.

Reference: Zhou et al., "Fast Global Registration", ECCV 2016.
"""

import torch
import torch.nn as nn
import numpy as np
import open3d as o3d


class FGRWrapper(nn.Module):
    """Thin nn.Module wrapper around Open3D's Fast Global Registration."""

    def __init__(self, voxel_size=0.05):
        """
        Args:
            voxel_size: Voxel size for FPFH feature computation. Controls the
                scale of geometric features. For UnitBall-normalized clouds,
                0.05 is a reasonable default.
        """
        super().__init__()
        self.voxel_size = voxel_size

    def _compute_fpfh(self, pcd):
        """Compute FPFH features for a point cloud."""
        radius_normal = self.voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        radius_feature = self.voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        return fpfh

    def forward(self, source, target):
        """Run FGR registration on a single pair.

        Args:
            source: (1, N, 3) tensor
            target: (1, M, 3) tensor

        Returns:
            dict with transformed_source, est_R, est_t.
        """
        input_device = source.device

        src_np = source[0].detach().cpu().numpy().astype(np.float64)
        tgt_np = target[0].detach().cpu().numpy().astype(np.float64)

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_np)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_np)

        src_fpfh = self._compute_fpfh(src_pcd)
        tgt_fpfh = self._compute_fpfh(tgt_pcd)

        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_pcd,
            tgt_pcd,
            src_fpfh,
            tgt_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=self.voxel_size * 0.5,
            ),
        )

        T = result.transformation
        R = T[:3, :3]
        t = T[:3, 3]

        aligned = (R @ src_np.T).T + t

        est_R = torch.from_numpy(R.astype(np.float32)).unsqueeze(0).to(input_device)
        est_t = torch.from_numpy(t.astype(np.float32)).unsqueeze(0).to(input_device)
        transformed = torch.from_numpy(aligned.astype(np.float32)).unsqueeze(0).to(input_device)

        return {
            'transformed_source': transformed,
            'est_R': est_R,
            'est_t': est_t,
        }
