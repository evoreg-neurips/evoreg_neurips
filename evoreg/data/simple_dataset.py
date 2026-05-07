"""
Simple dataset utilities for loading and handling point clouds.

Provides basic functionality for loading point clouds from various formats
and preparing them for EvoReg training and evaluation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
import warnings
import open3d as o3d


class SimplePointCloudDataset:
    """
    Simple dataset class for loading point cloud data.
    
    Supports loading from .npy, .npz, and .txt files with basic
    preprocessing capabilities.
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path],
        normalize: bool = True,
        center: bool = True,
        scale_mode: str = 'unit_sphere'
    ):
        """
        Initializes the point cloud dataset.
        
        Args:
            data_dir: Directory containing point cloud files
            normalize: Whether to normalize point clouds
            center: Whether to center point clouds at origin
            scale_mode: Scaling method ('unit_sphere' or 'unit_bbox')
        """
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.center = center
        self.scale_mode = scale_mode
        
        # Validates that data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
        # Finds all supported point cloud files
        self.files = self._find_point_cloud_files()
        
        if len(self.files) == 0:
            warnings.warn(f"No point cloud files found in {self.data_dir}")
    
    def _find_point_cloud_files(self) -> list:
        """
        Finds all supported point cloud files in the data directory.
        
        Returns:
            List of Path objects for valid point cloud files
        """
        # Defines supported file extensions
        supported_extensions = ['.npy', '.npz', '.txt', '.xyz', '.pts', '.ply', '.off']
        
        # Explicit list of filenames to exclude (correspondence/ground truth files)
        exclude_filenames = {'tr_gt_000.txt', 'tr_gt_001.txt'}
        
        # Collects all files with supported extensions
        files = []
        for ext in supported_extensions:
            files.extend(self.data_dir.glob(f'*{ext}'))
            files.extend(self.data_dir.glob(f'**/*{ext}'))  # Recursive search
        
        # Filters out excluded files based on explicit filename list
        filtered_files = []
        for f in files:
            filename = f.name
            if filename not in exclude_filenames:
                filtered_files.append(f)
            else:
                print(f"Excluding file: {filename}")
        
        # Sorts files for consistent ordering
        return sorted(list(set(filtered_files)))
    
    def load_point_cloud(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Loads a point cloud from file.
        
        Supports multiple file formats and returns point cloud
        as numpy array of shape (N, 3).
        
        Args:
            file_path: Path to point cloud file
            
        Returns:
            Point cloud as numpy array (N, 3)
        """
        file_path = Path(file_path)
        
        # Verifies file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        # Loads based on file extension
        ext = file_path.suffix.lower()
        
        if ext == '.npy':
            # Loads numpy array directly
            points = np.load(file_path)
        
        elif ext == '.npz':
            # Loads from numpy archive (assumes 'points' key)
            data = np.load(file_path)
            if 'points' in data:
                points = data['points']
            elif 'xyz' in data:
                points = data['xyz']
            else:
                # Uses first array if no standard key found
                key = list(data.keys())[0]
                points = data[key]
                warnings.warn(f"Using key '{key}' from npz file")
        
        elif ext in ['.txt', '.xyz', '.pts']:
            # Loads text-based formats
            points = np.loadtxt(file_path)
        
        elif ext in ['.ply', '.off']:
            points = self._load_mesh_file(file_path)
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Ensures points have correct shape
        if points.ndim == 1:
            # Validates that array size is divisible by 3
            if len(points) % 3 != 0:
                raise ValueError(
                    f"Array size must be divisible by 3. Invalid point cloud file: {file_path}"
                )
            # Reshapes flat array assuming 3 coordinates per point
            points = points.reshape(-1, 3)
        
        # Validates shape
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Invalid point cloud shape: {points.shape}")
        
        # Takes only first 3 columns if more exist (e.g., with normals/colors)
        if points.shape[1] > 3:
            warnings.warn(f"Using only first 3 columns (xyz) from {points.shape[1]} columns")
            points = points[:, :3]
        
        return points
    
    def _load_mesh_file(self, file_path: Path) -> np.ndarray:
        """
        Loads mesh file (PLY/OFF) and extracts vertices as point cloud.
        Uses Open3D to load mesh files and returns the vertices.
        
        Args:
            file_path: Path to mesh file (.ply or .off)
            
        Returns:
            Point cloud as numpy array (N, 3)
        """
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        points = np.asarray(mesh.vertices).astype(np.float32)
        
        return points
    
    def preprocess_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Preprocesses point cloud with centering and normalization.
        
        Args:
            points: Input point cloud (N, 3)
            
        Returns:
            Preprocessed point cloud (N, 3)
        """
        # Makes a copy to avoid modifying original
        points = points.copy()
        
        if self.center:
            # Centers point cloud at origin
            centroid = np.mean(points, axis=0, keepdims=True)
            points = points - centroid
        
        if self.normalize:
            # Normalizes based on specified mode
            if self.scale_mode == 'unit_sphere':
                # Scales to fit within unit sphere
                max_dist = np.max(np.linalg.norm(points, axis=1))
                if max_dist > 0:
                    points = points / max_dist
            
            elif self.scale_mode == 'unit_bbox':
                # Scales to fit within unit bounding box
                bbox_min = np.min(points, axis=0)
                bbox_max = np.max(points, axis=0)
                bbox_size = bbox_max - bbox_min
                max_size = np.max(bbox_size)
                if max_size > 0:
                    points = points / max_size
            
            else:
                raise ValueError(f"Unknown scale mode: {self.scale_mode}")
        
        return points
    
    def load_and_preprocess(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Loads and preprocesses a point cloud in one step.
        
        Args:
            file_path: Path to point cloud file
            
        Returns:
            Preprocessed point cloud (N, 3)
        """
        # Loads raw point cloud
        points = self.load_point_cloud(file_path)
        
        # Applies preprocessing
        points = self.preprocess_point_cloud(points)
        
        return points
    
    def __len__(self) -> int:
        """Returns number of point cloud files in dataset."""
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, str]]:
        """
        Gets a point cloud by index.
        
        Args:
            idx: Index of point cloud to load
            
        Returns:
            Dictionary with 'points' and 'file_path' keys
        """
        # Validates index
        if idx < 0 or idx >= len(self.files):
            raise IndexError(f"Index {idx} out of range [0, {len(self.files)})")
        
        # Loads and preprocesses point cloud
        file_path = self.files[idx]
        points = self.load_and_preprocess(file_path)
        
        return {
            'points': points,
            'file_path': str(file_path),
            'file_name': file_path.name
        }
    
    def to_torch_dataset(self) -> 'TorchPointCloudDataset':
        """
        Converts to PyTorch-compatible dataset.
        
        Returns:
            TorchPointCloudDataset wrapping this dataset
        """
        return TorchPointCloudDataset(self)


class TorchPointCloudDataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for SimplePointCloudDataset.
    
    Enables use with PyTorch DataLoader for batched loading
    and automatic tensor conversion.
    """
    
    def __init__(self, simple_dataset: SimplePointCloudDataset):
        """
        Initializes PyTorch dataset wrapper.
        
        Args:
            simple_dataset: SimplePointCloudDataset instance to wrap
        """
        self.dataset = simple_dataset
    
    def __len__(self) -> int:
        """Returns number of samples in dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Gets a sample and converts to PyTorch tensors.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with tensor data and metadata
        """
        # Gets data from underlying dataset
        data = self.dataset[idx]
        
        # Converts points to PyTorch tensor
        points_tensor = torch.from_numpy(data['points']).float()
        
        return {
            'points': points_tensor,
            'file_path': data['file_path'],
            'file_name': data['file_name']
        }


def create_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Creates a PyTorch DataLoader for point cloud data.
    
    Args:
        data_dir: Directory containing point cloud files
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of parallel data loading processes
        **dataset_kwargs: Additional arguments for SimplePointCloudDataset
        
    Returns:
        PyTorch DataLoader instance
    """
    # Creates dataset
    dataset = SimplePointCloudDataset(data_dir, **dataset_kwargs)
    torch_dataset = dataset.to_torch_dataset()
    
    # Creates DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_point_clouds
    )
    
    return dataloader


def collate_point_clouds(batch: list) -> Dict[str, Union[torch.Tensor, list]]:
    """
    Custom collate function for batching point clouds.
    
    Handles variable-sized point clouds by keeping them as a list
    rather than stacking into a tensor.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched data dictionary
    """
    # Collects all point clouds in a list
    points_list = [sample['points'] for sample in batch]
    
    # Collects metadata
    file_paths = [sample['file_path'] for sample in batch]
    file_names = [sample['file_name'] for sample in batch]
    
    return {
        'points': points_list,  # List of tensors
        'file_paths': file_paths,
        'file_names': file_names
    }


if __name__ == "__main__":
    # Tests basic functionality
    print("Testing SimplePointCloudDataset...")
    
    # Creates a test numpy file
    test_points = np.random.randn(1000, 3).astype(np.float32)
    test_file = Path("test_point_cloud.npy")
    np.save(test_file, test_points)
    
    try:
        # Tests dataset creation and loading
        dataset = SimplePointCloudDataset(".")
        print(f"Found {len(dataset)} point cloud files")
        
        if len(dataset) > 0:
            # Loads first point cloud
            data = dataset[0]
            print(f"Loaded point cloud shape: {data['points'].shape}")
            print(f"File: {data['file_name']}")
            
            # Tests PyTorch conversion
            torch_dataset = dataset.to_torch_dataset()
            torch_data = torch_dataset[0]
            print(f"Torch tensor shape: {torch_data['points'].shape}")
            print(f"Torch tensor dtype: {torch_data['points'].dtype}")
        
        print("[OK] Dataset tests passed!")
        
    finally:
        # Cleans up test file
        if test_file.exists():
            test_file.unlink()