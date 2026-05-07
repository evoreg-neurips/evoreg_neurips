"""
ModelNet40 dataset for point cloud registration.

Loads ModelNet40 point clouds and creates registration pairs by applying
random transformations. Supports train/test splits and caching for fast loading.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Union, Dict, Optional
from torch.utils.data import Dataset
import hashlib
try:
    import open3d as o3d
except ImportError:
    print('open3d!')

# Handle both relative and absolute imports
try:
    from .utils import generate_transformation, apply_nonrigid_deformation
except ImportError:
    from utils import generate_transformation, apply_nonrigid_deformation

class ModelNet40RegistrationDataset(Dataset):
    """
    ModelNet40 registration dataset.

    Supports two modes:
    1. Single file mode: Load one .off file and generate registration pairs on-the-fly
    2. Directory mode: Load multiple .off files and generate registration pairs on-the-fly

    Registration pairs are generated dynamically during training with random transformations.
    This provides infinite data augmentation.

    ModelNet40 directory structure:
        ModelNet40/
            airplane/
                train/airplane_0001.off, airplane_0002.off, ...
                test/airplane_0627.off, ...
            bathtub/
                train/...
                test/...
            ...
    """

    # Pattern for matching files in directory mode
    FILE_PATTERN = "**/*.off"

    def __init__(
        self,
        file_path: Union[str, Path] = None,
        data_dir: Union[str, Path] = None,
        n_samples: int = 1000,
        n_points: int = 1024,
        noise_std: float = 0.01,
        save_files: bool = False,
        split: Optional[str] = None,
        normalize: str = 'UnitBall',
        use_cache: bool = True,
        rotation_range: float = None,
        translation_range: float = 0.2,
        non_rigid: bool = False,
        nonrigid_n_control: int = 8,
        nonrigid_scale: float = 0.05,
        nonrigid_rbf_sigma: float = 0.3
    ):
        """
        Initializes ModelNet40 registration dataset.

        Args:
            file_path: Path to single .off file (e.g., 'airplane_0001.off')
            data_dir: Directory containing multiple .off files (mutually exclusive with file_path).
            n_samples: Number of samples (registration pairs) to generate per epoch
            n_points: Number of points to sample from point cloud
            noise_std: Standard deviation of Gaussian noise to add
            save_files: If True, save registration pairs to disk (default: False)
            split: 'train', 'test', or None (all files). Filters based on
                   ModelNet40 directory structure (*/train/*.off vs */test/*.off)
            normalize: Normalization method - 'UnitBall', 'BoundingBox', or 'Identity'
            use_cache: If True, cache loaded point clouds to .pt file for fast reloading
            rotation_range: If provided, constrains random rotations to ±rotation_range degrees.
                           If None, uses full SO(3) uniform random rotations.
            translation_range: Half-width of translation box (default: 0.2).
            non_rigid: If True, apply non-rigid deformation to target after rigid transform.
            nonrigid_n_control: Number of RBF control points for non-rigid deformation.
            nonrigid_scale: Magnitude of non-rigid displacements.
            nonrigid_rbf_sigma: RBF bandwidth controlling locality of deformation.
        """
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise_std = noise_std
        self.save_files = save_files
        self.split = split
        self.normalize = normalize
        self.use_cache = use_cache
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.non_rigid = non_rigid
        self.nonrigid_n_control = nonrigid_n_control
        self.nonrigid_scale = nonrigid_scale
        self.nonrigid_rbf_sigma = nonrigid_rbf_sigma

        # Load files
        if file_path is not None and data_dir is not None:
            raise ValueError("Specify either file_path or data_dir, not both")

        if file_path is not None:
            # Single file mode (no caching needed)
            self.file_path = Path(file_path)
            self.files = [self.file_path]
            self.point_clouds = [self._load_and_normalize(self.file_path)]
            print(f"Loaded {len(self.point_clouds[0])} points from {self.file_path.name}")

        elif data_dir is not None:
            self.data_dir = Path(data_dir)
            all_files = sorted(list(self.data_dir.glob(self.FILE_PATTERN)))

            # Filter by split (train/test)
            if split is not None:
                split_lower = split.lower()
                if split_lower not in ('train', 'test'):
                    raise ValueError(f"split must be 'train', 'test', or None, got '{split}'")
                all_files = [f for f in all_files if f'/{split_lower}/' in str(f)]
                print(f"Split filter: '{split_lower}' -> {len(all_files)} files")

            # Filter out generated output files (*_source.off, *_target.off)
            excluded_suffixes = ('_source.off', '_target.off')
            self.files = []
            excluded_count = 0

            for f in all_files:
                if f.name.endswith(excluded_suffixes):
                    excluded_count += 1
                else:
                    self.files.append(f)

            if len(self.files) == 0:
                raise ValueError(f"No .off files found in {self.data_dir} (split={split}, after filtering)")

            if excluded_count > 0:
                print(f"Excluded {excluded_count} generated output files")

            # Try loading from cache
            cache_loaded = False
            if self.use_cache:
                cache_loaded = self._load_from_cache()

            if not cache_loaded:
                # Load all point clouds from .off files
                # Pre-downsample to max 2x n_points to save memory
                # (we only ever sample n_points at __getitem__ time)
                max_store = max(self.n_points * 10, 10000)
                print(f"Loading {len(self.files)} .off files from {self.data_dir} (split={split})...")
                print(f"  Pre-downsampling to max {max_store} points per cloud to save memory")
                self.point_clouds = []
                for i, fp in enumerate(self.files):
                    pc = self._load_and_normalize(fp)
                    if len(pc) > max_store:
                        indices = np.random.choice(len(pc), max_store, replace=False)
                        pc = pc[indices]
                    self.point_clouds.append(pc)
                    if (i + 1) % 500 == 0:
                        print(f"  Loaded {i + 1}/{len(self.files)} files...")

                print(f"Loaded {len(self.files)} point clouds (split={split})")

                # Save cache
                if self.use_cache:
                    self._save_to_cache()

        else:
            raise ValueError("Must specify either file_path or data_dir")

        # Track which files have been saved (save only once per file)
        self.files_saved = set()

    def _cache_path(self) -> Path:
        """Returns the cache file path based on data_dir, split, and normalize."""
        cache_dir = self.data_dir / '.cache'
        split_str = self.split or 'all'
        norm_str = self.normalize.lower()
        return cache_dir / f'{split_str}_{norm_str}.pt'

    def _load_from_cache(self) -> bool:
        """Tries to load point clouds from cache. Returns True if successful."""
        cache_path = self._cache_path()
        if not cache_path.exists():
            return False

        try:
            print(f"Loading from cache: {cache_path}")
            cache = torch.load(cache_path, weights_only=False)

            # Verify cache matches current file list
            cached_files = cache.get('files', [])
            current_files = [str(f) for f in self.files]

            if cached_files != current_files:
                print(f"Cache stale (file list changed). Reloading from disk.")
                return False

            self.point_clouds = cache['point_clouds']
            print(f"Loaded {len(self.point_clouds)} point clouds from cache")
            return True
        except Exception as e:
            print(f"Cache load failed: {e}. Reloading from disk.")
            return False

    def _save_to_cache(self):
        """Saves loaded point clouds to cache file."""
        cache_path = self._cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache = {
            'files': [str(f) for f in self.files],
            'point_clouds': self.point_clouds,
            'normalize': self.normalize,
            'split': self.split
        }

        torch.save(cache, cache_path)
        print(f"Saved cache: {cache_path} ({len(self.point_clouds)} point clouds)")

    def _load_off_file_custom(self, file_path: Path) -> np.ndarray:
        """
        Custom OFF file reader that can handle malformed headers.

        Handles cases where the header is missing newlines, e.g., "OFF3514 3546 0" instead of:
        OFF
        3514 3546 0

        Args:
            file_path: Path to .off file

        Returns:
            Vertices as numpy array (N, 3)
        """
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()

            # Handle malformed header (e.g., "OFF3514 3546 0")
            if first_line.startswith('OFF') and len(first_line) > 3:
                # Header and counts are on same line
                remainder = first_line[3:].strip()  # Get "3514 3546 0"
                counts_line = remainder
            else:
                # Standard format: "OFF" on first line, counts on second
                counts_line = f.readline().strip()

            # Parse counts
            counts = counts_line.split()
            if len(counts) < 1:
                raise ValueError(f"Invalid OFF file: could not parse vertex count from '{counts_line}'")

            n_vertices = int(counts[0])
            n_faces = int(counts[1]) if len(counts) > 1 else 0

            # Read vertices
            vertices = []
            for i in range(n_vertices):
                line = f.readline().strip().split()
                if len(line) < 3:
                    raise ValueError(f"Invalid vertex data at line {i+1} in {file_path}")
                vertices.append([float(x) for x in line[:3]])

            return np.array(vertices, dtype=np.float32)

    def _load_and_normalize(self, file_path: Path) -> np.ndarray:
        """
        Loads .off file and normalizes based on self.normalize setting.

        Args:
            file_path: Path to .off file

        Returns:
            Normalized point cloud (N, 3)
        """
        points = None
        error_msg = None

        # Try Open3D first
        try:
            mesh = o3d.io.read_triangle_mesh(str(file_path))
            points = np.asarray(mesh.vertices, dtype=np.float32)

            if len(points) == 0:
                raise ValueError("Open3D returned empty point cloud")
        except Exception as e:
            error_msg = str(e)

        # Fallback to custom parser if Open3D failed
        if points is None or len(points) == 0:
            try:
                points = self._load_off_file_custom(file_path)
            except Exception as e:
                raise ValueError(
                    f"Failed to load {file_path.name} with both Open3D and custom parser.\n"
                    f"  Open3D error: {error_msg}\n"
                    f"  Custom parser error: {str(e)}\n"
                    f"  Please check if the file is a valid OFF format."
                )

        # Validate points
        if len(points) == 0:
            raise ValueError(f"Loaded point cloud is empty: {file_path}")

        if points.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape {points.shape}, expected (N, 3)")

        # Apply normalization
        if self.normalize == 'UnitBall':
            # Center at origin and scale to unit sphere
            centroid = np.mean(points, axis=0, keepdims=True)
            points = points - centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points = points / max_dist
        elif self.normalize == 'BoundingBox':
            # Scale to fit in [-1, 1] bounding box
            mins = np.min(points, axis=0, keepdims=True)
            maxs = np.max(points, axis=0, keepdims=True)
            center = (mins + maxs) / 2
            points = points - center
            scale = np.max(maxs - mins) / 2
            if scale > 0:
                points = points / scale
        elif self.normalize == 'Identity':
            pass  # No normalization
        else:
            raise ValueError(f"Unknown normalization: {self.normalize}")

        return points

    def _sample_points(self, points: np.ndarray, n_points: int) -> np.ndarray:
        """Samples n_points from point cloud."""
        n_available = len(points)

        if n_available >= n_points:
            indices = np.random.choice(n_available, n_points, replace=False)
        else:
            indices = np.random.choice(n_available, n_points, replace=True)

        return points[indices]

    def _save_registration_pair(
        self,
        source: np.ndarray,
        target: np.ndarray,
        transformation: np.ndarray,
        file_index: int
    ):
        """
        Saves a registration pair to the same folder as the source file.

        Args:
            source: Source point cloud (N, 3)
            target: Target point cloud (N, 3)
            transformation: 4x4 transformation matrix
            file_index: Index of the file used to generate this pair
        """
        # Get the file that was used
        source_file = self.files[file_index]

        # Extract base name from input file (e.g., airplane_0001 from airplane_0001.off)
        base_name = source_file.stem  # Gets filename without extension
        output_dir = source_file.parent  # Save in same directory as input file

        # Save source as .off
        source_path = output_dir / f"{base_name}_source.off"
        with open(source_path, 'w') as f:
            f.write("OFF\n")
            f.write(f"{len(source)} 0 0\n")
            for point in source:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

        # Save target as .off
        target_path = output_dir / f"{base_name}_target.off"
        with open(target_path, 'w') as f:
            f.write("OFF\n")
            f.write(f"{len(target)} 0 0\n")
            for point in target:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

        # Save transformation as .txt
        transform_path = output_dir / f"{base_name}_transformation.txt"
        np.savetxt(transform_path, transformation, fmt='%.6f')

        print(f"\nSaved registration pair for {source_file.name}:")
        print(f"  - {source_path.name}")
        print(f"  - {target_path.name}")
        print(f"  - {transform_path.name}")

    def __len__(self) -> int:
        """Returns number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generates a registration pair.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'source', 'target', and 'transformation' tensors
        """
        # Randomly select a point cloud from available files
        pc_idx = np.random.randint(0, len(self.point_clouds))
        point_cloud = self.point_clouds[pc_idx]

        # Sample points for source
        source = self._sample_points(point_cloud, self.n_points)

        # Generate transformation (rotation, translation, scale)
        rotation, translation, scale, transformation = generate_transformation(
            rotation_range=self.rotation_range,
            translation_range=self.translation_range
        )
        # Apply transformation to create target: scale * (source @ R^T) + t
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

        # Save registration pair for this file (only once per file, and only if save_files is True)
        if self.save_files and pc_idx not in self.files_saved:
            self._save_registration_pair(source, target, transformation, pc_idx)
            self.files_saved.add(pc_idx)

        return {
            'source': source_tensor,
            'target': target_tensor,
            'transformation': transformation_tensor,
            'rotation': rotation_tensor,
            'translation': translation_tensor,
        }


if __name__ == "__main__":
    # Test dataset with sample files
    print("Testing ModelNet40RegistrationDataset...")

    # Test 1: Directory mode with multiple files
    sample_dir = Path(__file__).parent / "ModelNet40Sample" / "airplane"

    if sample_dir.exists():
        print("\n=== Test 1: Directory mode (multiple files) ===")
        try:
            dataset = ModelNet40RegistrationDataset(
                data_dir=sample_dir,
                n_samples=5,
                n_points=1024,
                noise_std=0.01,
                save_files=False
            )

            print(f"Dataset size: {len(dataset)}")

            # Get first sample (will auto-save)
            sample = dataset[0]
            print(f"\nSample generated:")
            print(f"  Source shape: {sample['source'].shape}")
            print(f"  Target shape: {sample['target'].shape}")
            print(f"  Transformation shape: {sample['transformation'].shape}")

            print("\n[OK] Directory mode test passed!")

        except Exception as e:
            print(f"[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()

    # Test 2: Single file mode
    sample_file = Path(__file__).parent / "ModelNet40Sample" / "airplane_0001.off"

    if sample_file.exists():
        print("\n=== Test 2: Single file mode ===")
        try:
            dataset = ModelNet40RegistrationDataset(
                file_path=sample_file,
                n_samples=1024,
                n_points=1024,
                noise_std=0.01,
                save_files=False
            )

            print(f"Dataset size: {len(dataset)}")

            sample = dataset[0]
            print(f"\nSample generated:")
            print(f"  Source shape: {sample['source'].shape}")
            print(f"  Target shape: {sample['target'].shape}")
            print(f"  Transformation shape: {sample['transformation'].shape}")
            print("\n[OK] Single file mode test passed!")

        except Exception as e:
            print(f"[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n[OK] All tests completed!")
