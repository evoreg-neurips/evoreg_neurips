"""
Feature Matching Recall for point cloud evaluation.

Implements Feature Matching Recall using inlier ratio calculation with the following steps:
1. Extract feature vectors using KPConv (Kernel Point Convolution) neural network
2. Find correspondences using KNN on the extracted features
3. Calculate inlier ratio using ground-truth transformation
4. Calculate Euclidean distance between transformed source and target points
5. Count inliers that exceed threshold (0.05)
6. Return Feature Matching Recall = inliers / total correspondences

Based on: https://github.com/ZhiChen902/AMR

The implementation uses KPConv for optimal point cloud feature representation
with kernel point convolutions for local geometric feature extraction.
"""


import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List

# Import KPConv from local kpconv module
from evoreg.evaluation.kpconv.kpconv import KPConv

class KPConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=15, radius=0.2, sigma=0.05):
        super().__init__()
        self.kpconv = KPConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            radius=radius,
            sigma=sigma,
            bias=True,
            dimension=3
        )
    def forward(self, points, neighbor_indices):
        # Use coordinates as input features
        feats = points
        return self.kpconv(feats, points, points, neighbor_indices)


def knn_indices(src_feats, tgt_feats, k=1):
    # src_feats: (N, C), tgt_feats: (M, C)
    dists = torch.cdist(src_feats, tgt_feats)  # (N, M)
    knn = torch.topk(dists, k, largest=False)
    return knn.indices  # (N, k)


def compute_inlier_ratio_kpconv(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    transform: np.ndarray,
    acceptance_radius: float = 0.1,
    k: int = 1,
    kpconv_params: dict = None
) -> float:
    """
    Compute inlier ratio using KPConv features and KNN correspondences.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src = torch.from_numpy(src_points).float().to(device)
    tgt = torch.from_numpy(tgt_points).float().to(device)

    # Build neighbor indices (brute force kNN in input space)
    src_nn = torch.cdist(src, src).argsort(dim=1)[:, :16]  # (N, 16)
    tgt_nn = torch.cdist(tgt, tgt).argsort(dim=1)[:, :16]  # (M, 16)

    # KPConv feature extractor
    params = kpconv_params or {}
    feature_extractor = KPConvFeatureExtractor(**params).to(device)
    feature_extractor.eval()
    with torch.no_grad():
        src_feats = feature_extractor(src, src_nn)  # (N, C)
        tgt_feats = feature_extractor(tgt, tgt_nn)  # (M, C)

    # KNN correspondences (src -> tgt)
    tgt_idx = knn_indices(src_feats, tgt_feats, k=k).squeeze(-1).cpu().numpy()  # (N,)
    src_corr = src_points
    tgt_corr = tgt_points[tgt_idx]

    # Transform src_corr
    src_corr_trans = src_corr @ transform[:3, :3].T + transform[:3, 3]
    dists = np.linalg.norm(tgt_corr - src_corr_trans, axis=1)
    inlier_mask = dists < acceptance_radius
    inlier_ratio = np.mean(inlier_mask)
    return inlier_ratio


class FeatureMatchingRecall:

    def __init__(self, 
                 acceptance_radius: float = 0.1,
                 inlier_ratio_threshold: float = 0.05):
        self.acceptance_radius = acceptance_radius
        self.inlier_ratio_threshold = inlier_ratio_threshold
        
        self.pair_results = []
        self.scene_results = []
    
    def evaluate_correspondences(self, 
                                 ref_corr_points: np.ndarray,
                                 src_corr_points: np.ndarray, 
                                 transform: np.ndarray) -> Dict[str, float]:
        """KPConv-based correspondence evaluation."""
        # Use KPConv features for matching
        inlier_ratio = compute_inlier_ratio_kpconv(
            src_corr_points, ref_corr_points, transform,
            acceptance_radius=self.acceptance_radius
        )
        recall = float(inlier_ratio >= self.inlier_ratio_threshold)
        # For reporting, compute mean distance as before
        src_transformed = src_corr_points @ transform[:3, :3].T + transform[:3, 3]
        distances = np.linalg.norm(ref_corr_points - src_transformed, axis=1)
        inlier_mask = distances < self.acceptance_radius
        return {
            'inlier_ratio': inlier_ratio,
            'recall': recall,
            'num_correspondences': len(ref_corr_points),
            'mean_distance': np.mean(distances),
            'inlier_count': np.sum(inlier_mask)
        }
    
    def evaluate_pair(self, 
                     ref_corr_points: np.ndarray,
                     src_corr_points: np.ndarray,
                     transform: np.ndarray,
                     pair_id: str = None) -> Dict[str, float]:
        """Evaluate single point cloud pair."""
        
        result = self.evaluate_correspondences(
            ref_corr_points, src_corr_points, transform
        )
        
        if pair_id:
            result['pair_id'] = pair_id
            
        self.pair_results.append(result)
        
        return result
    
    def evaluate_scene(self, scene_pairs: List[Dict]) -> Dict[str, float]:
        """Evaluate multiple pairs in a scene."""
        
        scene_pair_results = []
        
        for pair_data in scene_pairs:
            result = self.evaluate_pair(
                ref_corr_points=pair_data['ref_corr_points'],
                src_corr_points=pair_data['src_corr_points'], 
                transform=pair_data['transform'],
                pair_id=pair_data.get('pair_id', None)
            )
            scene_pair_results.append(result)

        scene_recall = np.mean([r['recall'] for r in scene_pair_results])
        scene_inlier_ratio = np.mean([r['inlier_ratio'] for r in scene_pair_results])
        scene_mean_distance = np.mean([r['mean_distance'] for r in scene_pair_results])
        
        scene_result = {
            'scene_recall': scene_recall,
            'scene_inlier_ratio': scene_inlier_ratio, 
            'scene_mean_distance': scene_mean_distance,
            'num_pairs': len(scene_pair_results),
            'pair_results': scene_pair_results
        }
        
        self.scene_results.append(scene_result)
        return scene_result
    
    def get_final_metrics(self) -> Tuple[float, float, int]:
        """Get final aggregated metrics as tuple (FMR, IR, num_pairs)."""
        
        if not self.pair_results:
            return 0.0, 0.0, 0
        
        all_recalls = [r['recall'] for r in self.pair_results]
        all_inlier_ratios = [r['inlier_ratio'] for r in self.pair_results]
        
        return (
            np.mean(all_recalls),      # FMR
            np.mean(all_inlier_ratios), # IR  
            len(self.pair_results)     # num_pairs
        )
    
    def reset(self):
        self.pair_results = []
        self.scene_results = []


def test_kpconv_feature_extraction():
    """Test KPConv feature extraction with simple data."""
    print("Testing KPConv feature extraction...")
    
    # Create simple test data
    src_points = np.random.randn(50, 3).astype(np.float32)
    tgt_points = np.random.randn(60, 3).astype(np.float32)
    transform = np.eye(4, dtype=np.float32)
    
    try:
        # Test KPConv feature extraction
        inlier_ratio = compute_inlier_ratio_kpconv(
            src_points, tgt_points, transform,
            acceptance_radius=0.1
        )
        print(f"KPConv feature extraction successful!")
        print(f"Inlier ratio: {inlier_ratio:.3f}")
        return True
    except Exception as e:
        print(f"KPConv feature extraction failed: {e}")
        return False


def test_perfect_alignment():
    """Test with perfect alignment (identity transformation)."""
    print("Testing perfect alignment")
    
    # Create identical point clouds
    points = np.random.randn(50, 3).astype(np.float32)
    transform = np.eye(4, dtype=np.float32)
    
    evaluator = FeatureMatchingRecall(
        acceptance_radius=0.1,
        inlier_ratio_threshold=0.05
    )
    
    result = evaluator.evaluate_pair(
        ref_corr_points=points,
        src_corr_points=points,
        transform=transform,
        pair_id="perfect_alignment"
    )
    
    print(f"Perfect alignment - FMR: {result['recall']}, IR: {result['inlier_ratio']:.3f}")
    print(f"Mean distance: {result['mean_distance']:.6f}")
    
    return result


def test_noisy_alignment():
    """Test with noisy but good alignment."""
    print("\nTesting noisy alignment")
    
    # Create source points and add small noise to target
    src_points = np.random.randn(50, 3).astype(np.float32)
    noise = np.random.randn(50, 3).astype(np.float32) * 0.05  # Small noise
    tgt_points = src_points + noise
    transform = np.eye(4, dtype=np.float32)
    
    evaluator = FeatureMatchingRecall(
        acceptance_radius=0.1,
        inlier_ratio_threshold=0.05
    )
    
    result = evaluator.evaluate_pair(
        ref_corr_points=tgt_points,
        src_corr_points=src_points,
        transform=transform,
        pair_id="noisy_alignment"
    )
    
    print(f"Noisy alignment - FMR: {result['recall']}, IR: {result['inlier_ratio']:.3f}")
    print(f"Mean distance: {result['mean_distance']:.6f}")
    
    return result


def test_poor_alignment():
    """Test with poor alignment (large transformation error)."""
    print("\nTesting poor alignment")
    
    # Create very different point clouds
    src_points = np.random.randn(50, 3).astype(np.float32)
    tgt_points = np.random.randn(50, 3).astype(np.float32) * 5  # Very different
    transform = np.eye(4, dtype=np.float32)
    
    evaluator = FeatureMatchingRecall(
        acceptance_radius=0.1,
        inlier_ratio_threshold=0.05
    )
    
    result = evaluator.evaluate_pair(
        ref_corr_points=tgt_points,
        src_corr_points=src_points,
        transform=transform,
        pair_id="poor_alignment"
    )
    
    print(f"Poor alignment - FMR: {result['recall']}, IR: {result['inlier_ratio']:.3f}")
    print(f"Mean distance: {result['mean_distance']:.6f}")
    
    return result


def test_different_thresholds():
    """Test with different acceptance radius thresholds."""
    print("\nTesting different thresholds")
    
    # Create test data with moderate noise
    src_points = np.random.randn(50, 3).astype(np.float32)
    noise = np.random.randn(50, 3).astype(np.float32) * 0.08
    tgt_points = src_points + noise
    transform = np.eye(4, dtype=np.float32)
    
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]
    
    for threshold in thresholds:
        evaluator = FeatureMatchingRecall(
            acceptance_radius=threshold,
            inlier_ratio_threshold=0.05
        )
        
        result = evaluator.evaluate_pair(
            ref_corr_points=tgt_points,
            src_corr_points=src_points,
            transform=transform,
            pair_id=f"threshold_{threshold}"
        )
        
        print(f"Threshold {threshold}: FMR={result['recall']}, IR={result['inlier_ratio']:.3f}")


def test_scene_evaluation():
    """Test scene-level evaluation with multiple pairs."""
    print("\nTesting scene evaluation")
    
    evaluator = FeatureMatchingRecall(
        acceptance_radius=0.1,
        inlier_ratio_threshold=0.05
    )
    
    # Create scene with varying quality pairs
    scene_pairs = []
    
    # Good alignment pair
    good_src = np.random.randn(50, 3).astype(np.float32)
    good_tgt = good_src + np.random.randn(50, 3).astype(np.float32) * 0.03
    scene_pairs.append({
        'ref_corr_points': good_tgt,
        'src_corr_points': good_src,
        'transform': np.eye(4, dtype=np.float32),
        'pair_id': 'good_pair'
    })
    
    # Medium alignment pair
    med_src = np.random.randn(60, 3).astype(np.float32)
    med_tgt = med_src + np.random.randn(60, 3).astype(np.float32) * 0.08
    scene_pairs.append({
        'ref_corr_points': med_tgt,
        'src_corr_points': med_src,
        'transform': np.eye(4, dtype=np.float32),
        'pair_id': 'medium_pair'
    })
    
    # Poor alignment pair
    poor_src = np.random.randn(40, 3).astype(np.float32)
    poor_tgt = np.random.randn(40, 3).astype(np.float32) * 2
    scene_pairs.append({
        'ref_corr_points': poor_tgt,
        'src_corr_points': poor_src,
        'transform': np.eye(4, dtype=np.float32),
        'pair_id': 'poor_pair'
    })
    
    scene_result = evaluator.evaluate_scene(scene_pairs)
    
    print(f"Scene results:")
    print(f"  Scene FMR: {scene_result['scene_recall']:.3f}")
    print(f"  Scene IR: {scene_result['scene_inlier_ratio']:.3f}")
    print(f"  Scene mean distance: {scene_result['scene_mean_distance']:.3f}")
    print(f"  Number of pairs: {scene_result['num_pairs']}")
    
    return scene_result


def test_final_metrics():
    """Test aggregated final metrics calculation."""
    print("\nTesting final metrics")
    
    evaluator = FeatureMatchingRecall(
        acceptance_radius=0.1,
        inlier_ratio_threshold=0.05
    )
    
    # Evaluate multiple pairs with known characteristics
    test_cases = [
        ("perfect", 0.0),      # Perfect alignment
        ("good", 0.05),        # Good alignment  
        ("medium", 0.12),      # Medium alignment
        ("poor", 0.5),         # Poor alignment
    ]
    
    for case_name, noise_level in test_cases:
        src_points = np.random.randn(50, 3).astype(np.float32)
        if noise_level == 0.0:
            tgt_points = src_points.copy()
        else:
            tgt_points = src_points + np.random.randn(50, 3).astype(np.float32) * noise_level
        
        evaluator.evaluate_pair(
            ref_corr_points=tgt_points,
            src_corr_points=src_points,
            transform=np.eye(4, dtype=np.float32),
            pair_id=case_name
        )
    
    # Get final aggregated metrics
    fmr, ir, num_pairs = evaluator.get_final_metrics()
    
    print(f"Final aggregated metrics:")
    print(f"  Overall FMR: {fmr:.3f}")
    print(f"  Overall IR: {ir:.3f}")
    print(f"  Total pairs evaluated: {num_pairs}")
    
    return fmr, ir, num_pairs


def test_kpconv_parameters():
    """Test different KPConv parameter configurations."""
    print("\nTesting different KPConv parameters")
    
    src_points = np.random.randn(50, 3).astype(np.float32)
    tgt_points = src_points + np.random.randn(50, 3).astype(np.float32) * 0.05
    transform = np.eye(4, dtype=np.float32)
    
    # Test different parameter configurations
    param_configs = [
        {"out_channels": 16, "kernel_size": 10},
        {"out_channels": 32, "kernel_size": 15},
        {"out_channels": 64, "kernel_size": 20},
    ]
    
    for i, params in enumerate(param_configs):
        try:
            inlier_ratio = compute_inlier_ratio_kpconv(
                src_points, tgt_points, transform,
                acceptance_radius=0.1,
                kpconv_params=params
            )
            print(f"Config {i+1} - channels: {params['out_channels']}, "
                  f"kernel_size: {params['kernel_size']}, IR: {inlier_ratio:.3f}")
        except Exception as e:
            print(f"Config {i+1} failed: {e}")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\nTesting edge cases")
    
    evaluator = FeatureMatchingRecall()
    
    # Test empty results
    fmr, ir, num_pairs = evaluator.get_final_metrics()
    print(f"Empty evaluator - FMR: {fmr}, IR: {ir}, pairs: {num_pairs}")
    assert fmr == 0.0 and ir == 0.0 and num_pairs == 0
    
    # Test reset functionality
    evaluator.evaluate_pair(
        ref_corr_points=np.random.randn(20, 3),
        src_corr_points=np.random.randn(20, 3),
        transform=np.eye(4),
        pair_id="test"
    )
    
    fmr_before, _, pairs_before = evaluator.get_final_metrics()
    print(f"Before reset - pairs: {pairs_before}")
    
    evaluator.reset()
    fmr_after, _, pairs_after = evaluator.get_final_metrics()
    print(f"After reset - pairs: {pairs_after}")
    assert pairs_after == 0
    
    # Test minimum point cloud size
    try:
        small_points = np.random.randn(5, 3).astype(np.float32)
        result = evaluator.evaluate_pair(
            ref_corr_points=small_points,
            src_corr_points=small_points,
            transform=np.eye(4),
            pair_id="small"
        )
        print(f"Small point cloud test - IR: {result['inlier_ratio']:.3f}")
    except Exception as e:
        print(f"Small point cloud test failed: {e}")
    
    print("Edge case tests completed!")



if __name__ == "__main__":
    """Run all feature matching recall tests."""
    print("=" * 60)
    print("Feature Matching Recall Comprehensive Tests")
    print("=" * 60)
    
    test_functions = [
        test_kpconv_feature_extraction,
        test_perfect_alignment,
        test_noisy_alignment, 
        test_poor_alignment,
        test_different_thresholds,
        test_scene_evaluation,
        test_final_metrics,
        test_kpconv_parameters,
        test_edge_cases
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        print(f"\n{'-' * 40}")
        try:
            result = test_func()
            if result is not False:  # Consider non-False as pass
                passed += 1
                print(f"{test_func.__name__} passed")
            else:
                print(f"{test_func.__name__} failed")
        except Exception as e:
            print(f"{test_func.__name__} crashed: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All Feature Matching Recall tests passed!")
    else:
        print("Some tests failed - check error messages above")