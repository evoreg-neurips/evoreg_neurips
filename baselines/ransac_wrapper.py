"""
RANSAC registration wrapper for the evaluation pipeline.

Wraps Open3D's registration_ransac_based_on_feature_matching as an nn.Module
so it can be called from evaluate_baselines.py like any other model.

RANSAC is an optimization-based method (no learned weights) that uses FPFH
feature descriptors with random sample consensus for robust global registration.
"""

import torch
import torch.nn as nn
import numpy as np
import open3d as o3d


class RANSACWrapper(nn.Module):
    """Thin nn.Module wrapper around Open3D's RANSAC-based registration."""

    def __init__(self, voxel_size=0.05, max_iterations=100000, confidence=0.999):
        """
        Args:
            voxel_size: Voxel size for FPFH feature computation.
            max_iterations: Maximum RANSAC iterations.
            confidence: RANSAC confidence threshold for early stopping.
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.max_iterations = max_iterations
        self.confidence = confidence

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
        """Run RANSAC registration on a single pair.

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

        distance_threshold = self.voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd,
            tgt_pcd,
            src_fpfh,
            tgt_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.max_iterations, self.confidence
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
