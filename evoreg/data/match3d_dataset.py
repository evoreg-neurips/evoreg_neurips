import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Union, Dict, List, Optional
from torch.utils.data import Dataset
import open3d as o3d

from .utils import generate_transformation, apply_nonrigid_deformation


class Match3DRegistrationDataset(Dataset):
    """
    3DMatch registration dataset.
    
    Loads point cloud pairs from the 3DMatch dataset format:
    - fragments/cloud_bin_*.ply: Point cloud fragments
    - poses/cloud_bin_*.txt: Individual pose matrices for each fragment
    - cloud_bin_*.info.txt: Pairwise registration ground truth transformations
    
    The dataset creates registration pairs based on the .info.txt files which contain
    ground truth transformations between point cloud pairs.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        n_points: int = 1024,
        noise_std: float = 0.0,
        max_pairs_per_scene: int = None,
        train: bool = True
    ):
        """
        Initializes 3DMatch registration dataset.
        
        Args:
            data_dir: Root directory containing 3DMatch scenes (train/ or test/)
            n_points: Number of points to sample from point cloud
            noise_std: Standard deviation of Gaussian noise to add to point clouds
            max_pairs_per_scene: Maximum number of pairs to load per scene (None = all)
            train: Whether this is training data (affects data augmentation)
        """
        self.data_dir = Path(data_dir)
        self.n_points = n_points
        self.noise_std = noise_std
        self.max_pairs_per_scene = max_pairs_per_scene
        self.train = train
        
        # Folders
        self.scene_dirs = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                fragments_dir = item / "fragments"
                poses_dir = item / "poses"
                if fragments_dir.exists() and poses_dir.exists():
                    self.scene_dirs.append(item)
        
        if not self.scene_dirs:
            raise ValueError(f"No 3DMatch data found in {data_dir}")
        
        self.pairs = []
        self._load_registration_pairs()
    
    def _load_registration_pairs(self):
        """Load all registration pairs from .info.txt files across all scenes."""
        for scene_dir in self.scene_dirs:
            scene_pairs = []
            
            # Get all .info.txt files in this scene
            info_files = list(scene_dir.glob("*.info.txt"))
            
            for info_file in info_files:
                try:
                    pair_data = self._parse_info_file(info_file, scene_dir)
                    if pair_data is not None:
                        scene_pairs.append(pair_data)
                except Exception as e:
                    continue
            
            # Apply max_pairs_per_scene limit if specified
            if self.max_pairs_per_scene is not None and len(scene_pairs) > self.max_pairs_per_scene:
                scene_pairs = scene_pairs[:self.max_pairs_per_scene]
            
            self.pairs.extend(scene_pairs)
    
    def _parse_info_file(self, info_file: Path, scene_dir: Path) -> Dict:
        """
        Parse a .info.txt file to extract registration pair information.
        
        Args:
            info_file: Path to the .info.txt file
            scene_dir: Path to the scene directory
            
        Returns:
            Dictionary with source/target files and transformation matrix
        """
        with open(info_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 5:
            return None
        
        # Parse header line
        header = lines[0].strip().split()
        if len(header) < 4:
            return None
        
        source_id = int(header[2])
        target_id = int(header[3])

        # Find .ply files
        fragments_dir = scene_dir / "fragments"
        source_file = fragments_dir / f"cloud_bin_{source_id}.ply"
        target_file = fragments_dir / f"cloud_bin_{target_id}.ply"
        
        if not (source_file.exists() and target_file.exists()):
            return None
        
        # Parse transformation matrix
        transformation = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            row_data = lines[i + 1].strip().split()
            if len(row_data) != 4:
                return None
            transformation[i] = [float(x) for x in row_data]
        
        return {
            'scene_name': scene_dir.name,
            'source_file': source_file,
            'target_file': target_file,
            'source_id': source_id,
            'target_id': target_id,
            'transformation': transformation
        }
    
    def _load_point_cloud(self, file_path: Path) -> np.ndarray:
        """
        Load point cloud from .ply file.
        
        Args:
            file_path: Path to .ply file
            
        Returns:
            Point cloud as numpy array (N, 3)
        """
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points, dtype=np.float32)
            
            if len(points) == 0:
                raise ValueError(f"Empty point cloud: {file_path}")
            
            return points
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")
    
    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        """
        Sample n_points from point cloud.
        
        Args:
            points: Input point cloud (N, 3)
            
        Returns:
            Sampled point cloud (n_points, 3)
        """
        n_available = len(points)
        
        if n_available >= self.n_points:
            # Random sampling without replacement
            indices = np.random.choice(n_available, self.n_points, replace=False)
        else:
            # Random sampling with replacement if not enough points
            indices = np.random.choice(n_available, self.n_points, replace=True)
        
        return points[indices]
    
    def _normalize_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud to unit sphere centered at origin.
        
        Args:
            points: Input point cloud (N, 3)
            
        Returns:
            Normalized point cloud (N, 3)
        """
        # Center at origin
        centroid = np.mean(points, axis=0, keepdims=True)
        points_centered = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
        
        return points_normalized.astype(np.float32)
    
    def __len__(self) -> int:
        """Returns number of registration pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a registration pair.
        
        Args:
            idx: Index of the pair
            
        Returns:
            Dictionary with 'source', 'target', and 'transformation' tensors
        """
        pair = self.pairs[idx]
        
        # Load source and target point clouds
        source_points = self._load_point_cloud(pair['source_file'])
        target_points = self._load_point_cloud(pair['target_file'])
        
        # Normalize point clouds to unit sphere
        source_points = self._normalize_point_cloud(source_points)
        target_points = self._normalize_point_cloud(target_points)
        
        # Sample points
        source = self._sample_points(source_points)
        target = self._sample_points(target_points)
        
        # Add noise if specified
        if self.noise_std > 0:
            source_noise = np.random.randn(*source.shape).astype(np.float32) * self.noise_std
            target_noise = np.random.randn(*target.shape).astype(np.float32) * self.noise_std
            source = source + source_noise
            target = target + target_noise
        
        # Convert to tensors
        source_tensor = torch.from_numpy(source).float()
        target_tensor = torch.from_numpy(target).float()
        transformation_tensor = torch.from_numpy(pair['transformation']).float()
        
        return {
            'source': source_tensor,
            'target': target_tensor,
            'transformation': transformation_tensor,
            'scene_name': pair['scene_name'],
            'source_id': pair['source_id'],
            'target_id': pair['target_id']
        }


class Match3DPairDataset(Dataset):
    """
    3DMatch dataset that generates pairs on-the-fly from individual fragments.

    Loads all point cloud fragments and generates registration pairs dynamically
    by applying random transformations. Uses the same shared utilities as
    ModelNet40 and FAUST datasets for consistent cross-dataset comparison.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        n_samples: int = 1000,
        n_points: int = 1024,
        noise_std: float = 0.01,
        rotation_range: float = 45.0,
        translation_range: float = 0.2,
        normalize: str = 'UnitBall',
        non_rigid: bool = False,
        nonrigid_n_control: int = 8,
        nonrigid_scale: float = 0.05,
        nonrigid_rbf_sigma: float = 0.3,
    ):
        """
        Initialize 3DMatch pair generation dataset.

        Args:
            data_dir: Root directory containing 3DMatch scenes
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

        # Load all point cloud fragments from all scenes
        self.fragments = []
        self._load_fragments()

        if not self.fragments:
            raise ValueError(f"No point cloud fragments found in {data_dir}")

    def _load_fragments(self):
        """Load all point cloud fragments from all scenes."""
        if not self.data_dir.exists():
            return

        scene_count = 0
        total_fragments = 0

        for scene_dir in sorted(self.data_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith('.'):
                continue

            fragments_dir = scene_dir / "fragments"
            if not fragments_dir.exists():
                continue

            scene_count += 1

            for ply_file in sorted(fragments_dir.glob("*.ply")):
                try:
                    points = self._load_and_normalize(ply_file)
                    self.fragments.append({
                        'points': points,
                        'scene_name': scene_dir.name,
                        'file_path': ply_file
                    })
                    total_fragments += 1
                except Exception as e:
                    pass

        print(f"Loaded {total_fragments} fragments from {scene_count} scenes")

    def _load_and_normalize(self, file_path: Path) -> np.ndarray:
        """Load and normalize a point cloud fragment.

        Supports configurable normalization (UnitBall/BoundingBox/Identity),
        matching the ModelNet40/FAUST pattern.
        """
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points, dtype=np.float32)

        if len(points) == 0:
            raise ValueError(f"Empty point cloud: {file_path}")

        # Pre-downsample large fragments to cap memory usage
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
        matching the ModelNet40/FAUST pair generation pattern exactly.
        Noise is applied to target only (not source).
        """
        # Randomly select a fragment
        fragment_idx = np.random.randint(0, len(self.fragments))
        fragment = self.fragments[fragment_idx]
        points = fragment['points']

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
    # Test the dataset
    sample_data_dir = Path(__file__).parent.parent.parent / "baselines" / "pretrained_models" / "exp_geotransformer" / "3DMatch" / "data" / "train"
    
    if sample_data_dir.exists():
        try:
            # Test ground truth pairs dataset
            dataset = Match3DRegistrationDataset(
                data_dir=sample_data_dir,
                n_points=512,
                noise_std=0.01,
                max_pairs_per_scene=5
            )
            
            if len(dataset) > 0:
                # Test loading a sample
                sample = dataset[0]
                
            # Test pair generation dataset
            pair_dataset = Match3DPairDataset(
                data_dir=sample_data_dir,
                n_samples=10,
                n_points=512,
                normalize='UnitBall'
            )
            
            if len(pair_dataset) > 0:
                pair_sample = pair_dataset[0]
            
        except Exception as e:
            pass