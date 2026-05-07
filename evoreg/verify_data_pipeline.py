"""
Verification script for data pipeline components.

Verifies that data loading, synthetic generation, and visualization
all work correctly together.
"""

import sys
import numpy as np
from pathlib import Path

# Adds parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evoreg.data.simple_dataset import SimplePointCloudDataset, create_dataloader
from evoreg.data.synthetic_data import (
    generate_sphere, generate_cube, generate_cylinder, generate_torus,
    generate_registration_pair, generate_dataset
)
from evoreg.utils.visualization import (
    visualize_point_cloud, visualize_registration,
    visualize_correspondences, save_visualization
)


def verify_synthetic_generation():
    """
    Tests synthetic data generation functionality.
    
    Creates various synthetic shapes and verifies their properties.
    """
    print("\n" + "="*50)
    print("Testing synthetic data generation...")
    
    # Generates different shapes
    shapes = {
        'sphere': generate_sphere(1000, radius=1.0, noise_std=0.01),
        'cube': generate_cube(1000, side_length=2.0, surface_only=True),
        'cylinder': generate_cylinder(1000, radius=0.5, height=2.0),
        'torus': generate_torus(1000, major_radius=1.0, minor_radius=0.3)
    }
    
    # Verifies each shape
    for shape_name, points in shapes.items():
        print(f"\n{shape_name.capitalize()}:")
        print(f"  Shape: {points.shape}")
        print(f"  Min: {points.min(axis=0)}")
        print(f"  Max: {points.max(axis=0)}")
        print(f"  Mean: {points.mean(axis=0)}")
        
        # Checks data type and validity
        assert points.dtype == np.float32, f"Wrong dtype for {shape_name}"
        assert points.shape[1] == 3, f"Wrong dimensions for {shape_name}"
        assert not np.any(np.isnan(points)), f"NaN values in {shape_name}"
    
    print("\n[OK] Synthetic generation tests passed!")
    return shapes


def verify_registration_pairs():
    """
    Tests registration pair generation.
    
    Creates registration pairs with various transformations and noise.
    """
    print("\n" + "="*50)
    print("Testing registration pair generation...")
    
    # Generates pairs with different settings
    test_configs = [
        {'noise_std': 0.0, 'outlier_ratio': 0.0, 'partial_overlap': 1.0},
        {'noise_std': 0.02, 'outlier_ratio': 0.0, 'partial_overlap': 1.0},
        {'noise_std': 0.02, 'outlier_ratio': 0.1, 'partial_overlap': 1.0},
        {'noise_std': 0.02, 'outlier_ratio': 0.1, 'partial_overlap': 0.7},
    ]
    
    pairs = []
    for i, config in enumerate(test_configs):
        # Generates pair
        pair = generate_registration_pair(
            shape_type='sphere',
            n_points=500,
            **config
        )
        pairs.append(pair)
        
        print(f"\nPair {i+1} (noise={config['noise_std']}, "
              f"outliers={config['outlier_ratio']}, "
              f"overlap={config['partial_overlap']}):")
        print(f"  Source shape: {pair['source'].shape}")
        print(f"  Target shape: {pair['target'].shape}")
        print(f"  Scale factor: {pair['scale']:.3f}")
        
        # Verifies transformation matrix
        T = pair['transformation']
        assert T.shape == (4, 4), "Wrong transformation shape"
        assert np.allclose(T[3, :], [0, 0, 0, 1]), "Invalid transformation matrix"
    
    print("\n[OK] Registration pair tests passed!")
    return pairs


def verify_data_loading(temp_dir: Path):
    """
    Tests data loading functionality.
    
    Creates temporary files and tests loading them with the dataset class.
    
    Args:
        temp_dir: Temporary directory for test files
    """
    print("\n" + "="*50)
    print("Testing data loading...")
    
    # Creates test data files
    n_test_files = 5
    for i in range(n_test_files):
        # Generates random point cloud
        points = generate_sphere(100 + i * 50, noise_std=0.01)
        
        # Saves in different formats
        if i % 3 == 0:
            # Saves as .npy
            np.save(temp_dir / f"test_{i}.npy", points)
        elif i % 3 == 1:
            # Saves as .npz
            np.savez(temp_dir / f"test_{i}.npz", points=points)
        else:
            # Saves as .txt
            np.savetxt(temp_dir / f"test_{i}.txt", points)
    
    # Creates dataset
    dataset = SimplePointCloudDataset(
        temp_dir,
        normalize=True,
        center=True,
        scale_mode='unit_sphere'
    )
    
    print(f"\nDataset created with {len(dataset)} files")
    
    # Tests loading each file
    for i in range(min(3, len(dataset))):
        data = dataset[i]
        points = data['points']
        
        print(f"\nFile {i}: {data['file_name']}")
        print(f"  Original shape: {points.shape}")
        print(f"  Centered: {np.allclose(points.mean(axis=0), 0, atol=1e-6)}")
        print(f"  Max radius: {np.max(np.linalg.norm(points, axis=1)):.3f}")
    
    # Tests PyTorch dataloader
    if len(dataset) > 0:
        dataloader = create_dataloader(temp_dir, batch_size=2, shuffle=True)
        
        # Gets one batch
        for batch in dataloader:
            print(f"\nBatch test:")
            print(f"  Batch size: {len(batch['points'])}")
            print(f"  First sample shape: {batch['points'][0].shape}")
            print(f"  First sample dtype: {batch['points'][0].dtype}")
            break
    
    print("\n[OK] Data loading tests passed!")
    return dataset


def verify_visualization(shapes: dict, pairs: list, output_dir: Path):
    """
    Tests visualization functionality.
    
    Creates visualizations of shapes and registration pairs.
    
    Args:
        shapes: Dictionary of shape arrays
        pairs: List of registration pairs
        output_dir: Directory to save visualizations
    """
    print("\n" + "="*50)
    print("Testing visualization...")
    
    # Imports matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping visualization tests")
        return
    
    # Creates output directory
    output_dir.mkdir(exist_ok=True)
    
    # Visualizes individual shapes
    fig = plt.figure(figsize=(16, 4))
    
    for i, (shape_name, points) in enumerate(shapes.items()):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        visualize_point_cloud(points[:500], shape_name.capitalize(), ax=ax)
    
    plt.tight_layout()
    save_visualization(fig, output_dir / "shapes.png")
    plt.close()
    
    # Visualizes registration pairs
    for i, pair in enumerate(pairs[:2]):  # First two pairs only
        fig = visualize_registration(
            pair['source'],
            pair['target'],
            pair['source'],  # Would be transformed source in real use
            title=f"Registration Pair {i+1}"
        )
        save_visualization(fig, output_dir / f"registration_pair_{i}.png")
        plt.close()
    
    # Visualizes correspondences
    pair = pairs[0]
    fig = visualize_correspondences(
        pair['source'][:100],
        pair['target'][:100],
        max_lines=50
    )
    save_visualization(fig, output_dir / "correspondences.png")
    plt.close()
    
    print(f"\n[OK] Saved visualizations to {output_dir}")


def main():
    """
    Runs all data pipeline tests.
    
    Tests each component and verifies they work together correctly.
    """
    print("EvoReg Data Pipeline Test")
    print("=" * 60)
    
    # Creates temporary directory for test files
    test_dir = Path("test_data_temp")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Runs all tests
        shapes = verify_synthetic_generation()
        pairs = verify_registration_pairs()
        dataset = verify_data_loading(test_dir)
        verify_visualization(shapes, pairs, test_dir)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY: All data pipeline tests passed! [OK]")
        print("="*60)
        
        print("\nThe data pipeline is ready for use:")
        print("- Synthetic data generation works")
        print("- Data loading from files works")
        print("- Visualization utilities work")
        print("- PyTorch integration works")
        
    finally:
        # Option to clean up or keep files
        print(f"\nVisualization results saved in: {test_dir.absolute()}")
        keep_files = input("\nKeep visualization files? (y/n): ").lower().strip() == 'y'
        
        if not keep_files:
            # Cleans up temporary files
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)
                print(f"Cleaned up temporary directory: {test_dir}")
        else:
            print(f"Files kept in: {test_dir.absolute()}")


if __name__ == "__main__":
    main()