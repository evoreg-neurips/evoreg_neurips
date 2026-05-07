"""
FAUST dataset for point cloud registration.

Loads FAUST registration PLY files and creates registration pairs by applying
random transformations. Uses identical processing pipeline to ModelNet40 for
fair cross-dataset comparison.
"""

import glob
import os
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from typing import Optional

# Handle both relative and absolute imports
try:
    from .utils import generate_transformation, apply_nonrigid_deformation
except ImportError:
    from utils import generate_transformation, apply_nonrigid_deformation


class FAUSTRegistrationDataset(Dataset):
    """
    FAUST registration dataset with identical processing to ModelNet40.

    Loads PLY registration files (tr_reg_000.ply to tr_reg_099.ply), applies
    UnitBall normalization, and generates registration pairs using the same
    transform pipeline as ModelNet40RegistrationDataset.

    Split: first 80 files (subjects 0-7) for train, last 20 (subjects 8-9) for test.
    """

    def __init__(
        self,
        data_dir: str,
        n_samples: int = 1000,
        n_points: int = 1024,
        noise_std: float = 0.01,
        normalize: str = 'UnitBall',
        rotation_range: Optional[float] = None,
        translation_range: float = 0.2,
        non_rigid: bool = False,
        nonrigid_n_control: int = 8,
        nonrigid_scale: float = 0.05,
        nonrigid_rbf_sigma: float = 0.3,
        split: Optional[str] = None,
        use_natural_pairs: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing FAUST .ply registration files.
            n_samples: Number of registration pairs to generate per epoch.
            n_points: Number of points to sample from each point cloud.
            noise_std: Standard deviation of Gaussian noise added to target.
            normalize: Normalization method ('UnitBall', 'BoundingBox', 'Identity').
            rotation_range: If provided, constrains rotations to +/-N degrees.
                           If None, uses full SO(3) uniform random rotations.
            translation_range: Half-width of translation box (default: 0.2).
            non_rigid: If True, apply non-rigid deformation to target after rigid transform.
            nonrigid_n_control: Number of RBF control points for non-rigid deformation.
            nonrigid_scale: Magnitude of non-rigid displacements.
            nonrigid_rbf_sigma: RBF bandwidth controlling locality of deformation.
            split: 'train' (first 80 files), 'test' (last 20 files), or None (all).
            use_natural_pairs: If True, pair different meshes and compute Procrustes GT
                              instead of applying synthetic transforms to a single mesh.
        """
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise_std = noise_std
        self.normalize = normalize
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.non_rigid = non_rigid
        self.nonrigid_n_control = nonrigid_n_control
        self.nonrigid_scale = nonrigid_scale
        self.nonrigid_rbf_sigma = nonrigid_rbf_sigma
        self.use_natural_pairs = use_natural_pairs

        # Find all PLY files
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.ply")))
        if not all_files:
            # Try recursive search
            all_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.ply"), recursive=True))
        if not all_files:
            raise FileNotFoundError(f"No .ply files found in {data_dir}")

        # Split: 80 train / 20 test (10 subjects x 10 poses, last 2 subjects = test)
        if split is not None:
            split_lower = split.lower()
            if split_lower == 'train':
                all_files = all_files[:-20] if len(all_files) > 20 else all_files
            elif split_lower == 'test':
                all_files = all_files[-20:] if len(all_files) > 20 else all_files
            else:
                raise ValueError(f"split must be 'train', 'test', or None, got '{split}'")
            print(f"FAUST split '{split_lower}': {len(all_files)} files")

        self.files = all_files

        # Load and normalize all point clouds
        # In natural pairs mode, keep full meshes for Procrustes computation
        if self.use_natural_pairs:
            max_store = None  # Keep all vertices (6890 for FAUST SMPL topology)
        else:
            max_store = max(self.n_points * 10, 10000)  # Same as ModelNet40
        mode_str = "natural pairs" if self.use_natural_pairs else "synthetic"
        print(f"Loading {len(self.files)} FAUST PLY files (mode={mode_str})...")
        self.point_clouds = []
        for fp in self.files:
            pc = self._load_and_normalize(fp)
            if max_store is not None and len(pc) > max_store:
                indices = np.random.choice(len(pc), max_store, replace=False)
                pc = pc[indices]
            self.point_clouds.append(pc)

        print(f"Loaded {len(self.point_clouds)} FAUST point clouds "
              f"(split={split}, mode={mode_str}, verts={len(self.point_clouds[0])})")

    def _load_and_normalize(self, file_path: str) -> np.ndarray:
        """Loads PLY file and normalizes. Identical normalization to ModelNet40."""
        mesh = trimesh.load_mesh(file_path, process=False)
        points = np.asarray(mesh.vertices, dtype=np.float32)

        if len(points) == 0:
            raise ValueError(f"Loaded point cloud is empty: {file_path}")
        if points.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape {points.shape}, expected (N, 3)")

        # Apply normalization — identical to ModelNet40 (modelnet40_dataset.py:312-318)
        if self.normalize == 'UnitBall':
            centroid = np.mean(points, axis=0, keepdims=True)
            points = points - centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points = points / max_dist
        elif self.normalize == 'BoundingBox':
            mins = np.min(points, axis=0, keepdims=True)
            maxs = np.max(points, axis=0, keepdims=True)
            center = (mins + maxs) / 2
            points = points - center
            scale = np.max(maxs - mins) / 2
            if scale > 0:
                points = points / scale
        elif self.normalize == 'Identity':
            pass
        else:
            raise ValueError(f"Unknown normalization: {self.normalize}")

        return points

    def _sample_points(self, points: np.ndarray, n_points: int) -> np.ndarray:
        """Samples n_points from point cloud. Identical to ModelNet40."""
        n_available = len(points)
        if n_available >= n_points:
            indices = np.random.choice(n_available, n_points, replace=False)
        else:
            indices = np.random.choice(n_available, n_points, replace=True)
        return points[indices]

    def _compute_procrustes(self, source_full: np.ndarray, target_full: np.ndarray):
        """Compute optimal rigid alignment (R, t) via SVD on corresponding vertices.

        Both inputs should be UnitBall-normalized. Uses all vertices for best
        Procrustes fit; the residual after alignment is genuine non-rigid deformation.

        Returns:
            rotation: (3, 3) float32 rotation matrix
            translation: (3,) float32 translation vector
            transformation: (4, 4) float32 homogeneous transformation matrix
        """
        centroid_src = source_full.mean(axis=0)
        centroid_tgt = target_full.mean(axis=0)
        src_c = source_full - centroid_src
        tgt_c = target_full - centroid_tgt

        H = src_c.T @ tgt_c  # (3, 3)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:  # Handle reflection
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.astype(np.float32)
        T[:3, 3] = t.astype(np.float32)
        return R.astype(np.float32), t.astype(np.float32), T

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        if self.use_natural_pairs:
            return self._getitem_natural(idx)
        return self._getitem_synthetic(idx)

    def _getitem_natural(self, idx: int) -> dict:
        """Natural pairs mode: pair two different FAUST meshes, compute Procrustes GT."""
        # Pick two different meshes
        idx_a = np.random.randint(0, len(self.point_clouds))
        idx_b = np.random.randint(0, len(self.point_clouds) - 1)
        if idx_b >= idx_a:
            idx_b += 1

        source_full = self.point_clouds[idx_a]
        target_full = self.point_clouds[idx_b]

        # Procrustes GT from full meshes (all vertices, typically 6890 for FAUST)
        rotation, translation, transformation = self._compute_procrustes(
            source_full, target_full
        )

        # Sample n_points independently from each mesh
        source = self._sample_points(source_full, self.n_points)
        target = self._sample_points(target_full, self.n_points)

        # Add noise to target (same as synthetic mode for consistency)
        if self.noise_std > 0:
            noise = np.random.randn(self.n_points, 3).astype(np.float32) * self.noise_std
            target = target + noise

        # Convert to tensors
        source_tensor = torch.from_numpy(source).float()
        target_tensor = torch.from_numpy(target).float()
        transformation_tensor = torch.from_numpy(transformation).float()
        rotation_tensor = torch.from_numpy(rotation).float()
        translation_tensor = torch.from_numpy(translation).float()

        return {
            'source': source_tensor,
            'target': target_tensor,
            'transformation': transformation_tensor,
            'rotation': rotation_tensor,
            'translation': translation_tensor,
        }

    def _getitem_synthetic(self, idx: int) -> dict:
        """
        Synthetic pairs mode: apply random transforms to a single mesh.
        Identical flow to ModelNet40 (modelnet40_dataset.py:398-455).
        """
        # Randomly select a point cloud from available files
        pc_idx = np.random.randint(0, len(self.point_clouds))
        point_cloud = self.point_clouds[pc_idx]

        # Sample points for source
        source = self._sample_points(point_cloud, self.n_points)

        # Generate transformation (rotation, translation, scale) — reuse from utils.py
        rotation, translation, scale, transformation = generate_transformation(
            rotation_range=self.rotation_range,
            translation_range=self.translation_range
        )

        # Apply transformation: target = scale * (source @ R^T) + t
        target = scale * (source @ rotation.T) + translation.reshape(1, 3)

        # Apply non-rigid deformation if enabled
        if self.non_rigid:
            target = apply_nonrigid_deformation(
                target,
                n_control_points=self.nonrigid_n_control,
                deformation_scale=self.nonrigid_scale,
                rbf_sigma=self.nonrigid_rbf_sigma,
            )

        # Add noise to target
        if self.noise_std > 0:
            noise = np.random.randn(self.n_points, 3).astype(np.float32) * self.noise_std
            target = target + noise

        # Convert to tensors
        source_tensor = torch.from_numpy(source).float()
        target_tensor = torch.from_numpy(target).float()
        transformation_tensor = torch.from_numpy(transformation).float()
        rotation_tensor = torch.from_numpy(rotation).float()
        translation_tensor = torch.from_numpy(translation).float()

        return {
            'source': source_tensor,
            'target': target_tensor,
            'transformation': transformation_tensor,
            'rotation': rotation_tensor,
            'translation': translation_tensor,
        }
