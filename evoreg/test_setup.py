#!/usr/bin/env python3
"""
Test script to verify EvoReg environment setup.
Checks PyTorch installation, CUDA availability, and basic functionality.
"""

import sys
import importlib
from pathlib import Path


def test_imports():
    """
    Tests that all required packages can be imported.
    
    Returns:
        bool: True if all required packages import successfully
    """
    print("Testing package imports...")
    
    # Defines lists of required and optional packages
    required_packages = [
        'torch',
        'numpy',
        'scipy',
        'tqdm',
        'matplotlib',
        'yaml',
        'open3d',
        'trimesh',
        'plyfile'
    ]
    
    optional_packages = [
        'wandb',
        'pytest',
        'jupyter',
        'tensorboard'
    ]
    
    # Tracks packages that fail to import
    failed_imports = []
    
    # Tests each required package
    print("\nRequired packages:")
    for package in required_packages:
        try:
            # Attempts to import the package
            importlib.import_module(package)
            print(f"  [OK] {package}")
        except ImportError as e:
            # Records import failures
            print(f"  [FAIL] {package} - {str(e)}")
            failed_imports.append(package)
    
    # Tests optional packages without failing if missing
    print("\nOptional packages:")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"  [OK] {package}")
        except ImportError:
            # Notes package is not installed but doesn't fail
            print(f"  ○ {package} (not installed)")
    
    # Returns success if all required packages imported
    return len(failed_imports) == 0


def test_pytorch():
    """
    Tests PyTorch functionality and CUDA availability.
    
    Verifies PyTorch installation, checks for GPU support,
    and tests basic tensor operations including gradients.
    
    Returns:
        bool: True if all PyTorch tests pass
    """
    print("\n" + "="*50)
    print("Testing PyTorch setup...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Checks if CUDA is available for GPU acceleration
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            # Displays GPU information if available
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            # Notes CPU-only mode
            print("  Note: CUDA not available. CPU mode will be used.")
        
        # Tests basic tensor operations
        print("\nTesting tensor operations...")
        device = torch.device('cuda' if cuda_available else 'cpu')
        
        # Creates random point cloud tensors
        x = torch.randn(100, 3).to(device)
        y = torch.randn(100, 3).to(device)
        
        # Tests basic arithmetic operations
        z = x + y
        # Computes pairwise distances between point sets
        dist = torch.cdist(x, y)
        
        print(f"  [OK] Tensor creation and basic ops work on {device}")
        print(f"  [OK] Created tensors of shape {x.shape}")
        print(f"  [OK] Computed pairwise distances: shape {dist.shape}")
        
        # Tests automatic differentiation capabilities
        x_grad = torch.randn(10, 3, requires_grad=True).to(device)
        y_grad = (x_grad ** 2).sum()
        y_grad.backward()
        
        print(f"  [OK] Gradient computation works")
        
        return True
        
    except Exception as e:
        # Reports any failures in PyTorch testing
        print(f"  [FAIL] PyTorch test failed: {str(e)}")
        return False


def test_point_cloud_libs():
    """
    Tests point cloud processing libraries.
    
    Verifies Open3D installation and tests basic point cloud
    operations like creation and normal estimation.
    
    Returns:
        bool: True if point cloud libraries work correctly
    """
    print("\n" + "="*50)
    print("Testing point cloud libraries...")
    
    try:
        import numpy as np
        import open3d as o3d
        
        # Creates a random point cloud for testing
        points = np.random.randn(1000, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Displays library version and point count
        print(f"  [OK] Open3D version: {o3d.__version__}")
        print(f"  [OK] Created point cloud with {len(pcd.points)} points")
        
        # Tests normal estimation functionality
        pcd.estimate_normals()
        print(f"  [OK] Computed normals")
        
        return True
        
    except Exception as e:
        # Reports failures in point cloud library testing
        print(f"  [FAIL] Point cloud library test failed: {str(e)}")
        return False


def test_directory_structure():
    """
    Verifies EvoReg directory structure.
    
    Checks that all required subdirectories exist in the
    project structure.
    
    Returns:
        bool: True if all required directories exist
    """
    print("\n" + "="*50)
    print("Verifying directory structure...")
    
    # Gets the base directory relative to this script
    base_dir = Path(__file__).parent
    required_dirs = ['models', 'data', 'training', 'evaluation', 'utils', 'configs']
    
    # Tracks if all directories exist
    all_exist = True
    
    # Checks each required directory
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        exists = dir_path.exists() and dir_path.is_dir()
        status = "[OK]" if exists else "[FAIL]"
        print(f"  {status} {dir_name}/")
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """
    Runs all environment tests.
    
    Executes a suite of tests to verify the development environment
    is properly configured for EvoReg development.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("EvoReg Environment Setup Test")
    print("=" * 50)
    
    # Defines test suite with names and functions
    tests = [
        ("Package imports", test_imports),
        ("PyTorch setup", test_pytorch),
        ("Point cloud libraries", test_point_cloud_libs),
        ("Directory structure", test_directory_structure)
    ]
    
    # Stores results of each test
    results = []
    
    # Executes each test and records results
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            # Handles unexpected test failures
            print(f"\n[FAIL] {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Displays summary of all test results
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    
    # Checks if all tests passed
    all_passed = True
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "[OK]" if success else "[FAIL]"
        print(f"{symbol} {test_name}: {status}")
        if not success:
            all_passed = False
    
    # Provides final status and instructions
    if all_passed:
        print("\n[OK] All tests passed! Environment is ready for EvoReg development.")
    else:
        print("\n[FAIL] Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    # Returns appropriate exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())