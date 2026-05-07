import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Union, Dict, List, Optional
from torch.utils.data import Dataset
import json

from .utils import generate_transformation, apply_nonrigid_deformation


class ShapeNetRegistrationDataset(Dataset):
    """
    ShapeNet registration dataset following the unified ModelNet40/FAUST/3DMatch
    pipeline. Loads pre-sampled .npy point clouds and generates registration
    pairs on-the-fly via shared utilities.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        n_samples: int = 1000,
        n_points: int = 1024,
        noise_std: float = 0.01,
        rotation_range: float = None,
        translation_range: float = 0.2,
        normalize: str = 'UnitBall',
        non_rigid: bool = False,
        nonrigid_n_control: int = 8,
        nonrigid_scale: float = 0.05,
        nonrigid_rbf_sigma: float = 0.3,
        split: Optional[str] = None,
        class_choice: Optional[List[str]] = None,
        shapenet13: bool = False,
    ):
        """
        Initialize ShapeNet pair generation dataset.

        Args:
            data_dir: Root directory containing ShapeNetV1PointCloud data.
                      Expected structure: data_dir/ShapeNetV1PointCloud/<synsetId>/*.npy
                      OR data_dir/<synsetId>/*.npy
            n_samples: Number of pairs to generate per epoch
            n_points: Number of points to sample from point cloud
            noise_std: Standard deviation of Gaussian noise (applied to target only)
            rotation_range: Max rotation in degrees (None = full SO(3))
            translation_range: Half-width of translation box
            normalize: Normalization method ('UnitBall', 'BoundingBox', 'Identity')
            non_rigid: If True, apply RBF non-rigid deformations to target
            nonrigid_n_control: Number of RBF control points
            nonrigid_scale: Magnitude of non-rigid deformations
            nonrigid_rbf_sigma: RBF bandwidth controlling locality
            split: 'train' (first 80%) or 'test' (last 20%) or None (all)
            class_choice: List of category names to include (e.g. ['airplane', 'car'])
            shapenet13: If True, use the standard 13-category subset
        """
        self.data_dir = Path(data_dir).resolve()
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise_std = noise_std
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.normalize = normalize
        self.non_rigid = non_rigid
        self.nonrigid_n_control = nonrigid_n_control
        self.nonrigid_scale = nonrigid_scale
        self.nonrigid_rbf_sigma = nonrigid_rbf_sigma
        self.split = split

        # ShapeNet13 standard categories
        if shapenet13:
            class_choice = [
                "airplane", "bench", "cabinet", "car", "chair", "display",
                "lamp", "loudspeaker", "rifle", "sofa", "table", "telephone", "vessel"
            ]

        # Resolve point cloud root (handle nested ShapeNetV1PointCloud dir)
        pc_root = self.data_dir / "ShapeNetV1PointCloud"
        if not pc_root.exists():
            pc_root = self.data_dir

        # Load taxonomy for name->synsetId mapping
        self.id2names = {}
        self.names2id = {}
        taxonomy_path = self.data_dir / "taxonomy.json"
        if not taxonomy_path.exists():
            taxonomy_path = self.data_dir.parent / "taxonomy.json"
        if taxonomy_path.exists():
            with open(taxonomy_path, 'r') as f:
                taxonomy = json.load(f)
            for entry in taxonomy:
                name = entry['name'].split(',')[0].strip()
                self.id2names[entry['synsetId']] = name
                self.names2id[name] = entry['synsetId']

        # Determine which synset directories to use
        synset_dirs = sorted([
            d for d in pc_root.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        # Filter by class_choice if specified
        if class_choice:
            selected_ids = set()
            for name in class_choice:
                if name in self.names2id:
                    selected_ids.add(self.names2id[name])
                else:
                    selected_ids.add(name)
            synset_dirs = [d for d in synset_dirs if d.name in selected_ids]

        # Load all point clouds
        self.point_clouds = []
        self._load_point_clouds(synset_dirs)

        if not self.point_clouds:
            raise ValueError(f"No point clouds found in {data_dir}")

    def _load_point_clouds(self, synset_dirs: List[Path]):
        """Load all .npy point clouds from synset directories."""
        total = 0
        n_categories = 0

        for synset_dir in synset_dirs:
            npy_files = sorted(synset_dir.glob("*.npy"))
            if not npy_files:
                continue

            # Apply train/test split (80/20 by sorted file order)
            if self.split == 'train':
                npy_files = npy_files[:int(len(npy_files) * 0.8)]
            elif self.split == 'test':
                npy_files = npy_files[int(len(npy_files) * 0.8):]

            n_categories += 1
            for npy_file in npy_files:
                try:
                    points = self._load_and_normalize(npy_file)
                    self.point_clouds.append({
                        'points': points,
                        'category': synset_dir.name,
                        'file_path': npy_file,
                    })
                    total += 1
                except Exception:
                    pass

        cat_label = self.id2names.get(synset_dirs[0].name, '') if synset_dirs else ''
        split_label = f" ({self.split})" if self.split else ""
        print(f"Loaded {total} ShapeNet point clouds from {n_categories} categories{split_label}")

    def _load_and_normalize(self, file_path: Path) -> np.ndarray:
        """Load and normalize a point cloud.

        Supports configurable normalization (UnitBall/BoundingBox/Identity),
        matching the ModelNet40/FAUST/3DMatch pattern.
        """
        points = np.load(str(file_path)).astype(np.float32)

        if len(points) == 0:
            raise ValueError(f"Empty point cloud: {file_path}")

        # Only keep xyz (first 3 columns) if more columns exist
        if points.ndim == 2 and points.shape[1] > 3:
            points = points[:, :3]

        # Pre-downsample large clouds to cap memory usage
        max_stored_points = 10000
        if len(points) > max_stored_points:
            indices = np.random.choice(len(points), max_stored_points, replace=False)
            points = points[indices]

        # Apply normalization
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
        """Sample n_points from point cloud."""
        n_available = len(points)

        if n_available >= n_points:
            indices = np.random.choice(n_available, n_points, replace=False)
        else:
            indices = np.random.choice(n_available, n_points, replace=True)

        return points[indices]

    def __len__(self) -> int:
        """Returns number of samples to generate."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a registration pair.

        Uses shared generate_transformation() from evoreg.data.utils,
        matching the ModelNet40/FAUST/3DMatch pair generation pattern exactly.
        Noise is applied to target only (not source).
        """
        # Randomly select a point cloud
        pc_idx = np.random.randint(0, len(self.point_clouds))
        pc = self.point_clouds[pc_idx]
        points = pc['points']

        # Sample points for source
        source = self._sample_points(points, self.n_points)

        # Generate transformation using shared utility
        rotation, translation, scale, transformation = generate_transformation(
            rotation_range=self.rotation_range,
            translation_range=self.translation_range
        )

        # Apply transformation to create target
        target = scale * (source @ rotation.T) + translation.reshape(1, 3)

        # Apply non-rigid deformation if enabled
        if self.non_rigid:
            target = apply_nonrigid_deformation(
                target,
                n_control_points=self.nonrigid_n_control,
                deformation_scale=self.nonrigid_scale,
                rbf_sigma=self.nonrigid_rbf_sigma,
            )

        # Add noise to target only
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


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/path/to/Desktop/Datasets/ShapeNetV1PointCloud"

    dataset = ShapeNetRegistrationDataset(
        data_dir=data_dir,
        n_samples=10,
        n_points=1024,
        normalize='UnitBall',
        split='test',
        shapenet13=True,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Point clouds loaded: {len(dataset.point_clouds)}")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for k, v in sample.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
