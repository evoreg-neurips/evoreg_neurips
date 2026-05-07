"""
Registration Recall for point cloud evaluation.

Implements Registration Recall metric that processes datasets of point cloud pairs,
calculates one-way RMSE for each registration attempt, and computes recall based on 
successful registrations (RMSE < threshold).

Registration Recall = (Number of successful registrations) / (Total registration attempts)
where success is defined as RMSE < threshold using KNN and Euclidean distances.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
import numpy as np


def calculate_rmse_knn(
    source: torch.Tensor,
    target: torch.Tensor,
    k: int = 1
) -> torch.Tensor:
    """
    Calculate one-way RMSE using K-Nearest Neighbors from source to target.
    
    This measures how well the transformed source point cloud aligns with
    the ground truth target point cloud.
    
    Args:
        source: Transformed source point cloud of shape (N, 3)
        target: Ground truth target point cloud of shape (M, 3)
        k: Number of nearest neighbors to consider (default: 1)
        
    Returns:
        RMSE value as scalar tensor
    """
    # Compute pairwise distances
    source_expanded = source.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    
    # Euclidean distances (N, M)
    distances = torch.norm(source_expanded - target_expanded, p=2, dim=2)
    
    # Find k nearest neighbors for each source point
    k_distances, _ = torch.topk(distances, k, dim=1, largest=False)  # (N, k)
    
    # Average over k neighbors, then compute RMSE
    mean_distances = k_distances.mean(dim=1)  # (N,)
    rmse = torch.sqrt((mean_distances ** 2).mean())
    
    return rmse


def registration_recall_dataset(
    source_list: List[torch.Tensor],
    target_list: List[torch.Tensor],
    threshold: float = 0.1,
    k: int = 1,
    return_details: bool = False
) -> Union[float, Tuple[float, List[float], List[bool]]]:
    """
    Calculate Registration Recall across a dataset of point cloud pairs using one-way RMSE.
    
    Args:
        source_list: List of transformed source point clouds, each of shape (N, 3)
        target_list: List of ground truth target point clouds, each of shape (M, 3)
        threshold: RMSE threshold for successful registration
        k: Number of nearest neighbors for RMSE calculation
        return_details: If True, return detailed results
        
    Returns:
        Registration recall (float), optionally with RMSE values and success flags
    """
    if len(source_list) != len(target_list):
        raise ValueError(f"Mismatched lengths: {len(source_list)} vs {len(target_list)}")
    
    total_pairs = len(source_list)
    successful_registrations = 0
    rmse_values = []
    success_flags = []
    
    for i, (source, target) in enumerate(zip(source_list, target_list)):
        # Calculate one-way RMSE for this pair
        rmse = calculate_rmse_knn(source, target, k)
        
        rmse_val = rmse.item()
        rmse_values.append(rmse_val)
        
        # Check if registration is successful
        is_successful = rmse_val < threshold
        success_flags.append(is_successful)
        
        if is_successful:
            successful_registrations += 1
    
    # Calculate recall
    recall = successful_registrations / total_pairs if total_pairs > 0 else 0.0
    
    if return_details:
        return recall, rmse_values, success_flags
    else:
        return recall
        
        # Check if registration is successful
        is_successful = rmse_val < threshold
        success_flags.append(is_successful)
        
        if is_successful:
            successful_registrations += 1
    
    # Calculate recall
    recall = successful_registrations / total_pairs if total_pairs > 0 else 0.0
    
    if return_details:
        return recall, rmse_values, success_flags
    else:
        return recall


class RegistrationRecall(nn.Module):
    """
    Registration Recall module for point cloud evaluation.

    Evaluates registration success across datasets by calculating one-way RMSE
    from transformed source to ground truth target point clouds, determining 
    recall based on success threshold.

    """

    def __init__(
        self,
        threshold: float = 0.1,
        k: int = 1,
        reduction: str = 'mean'
    ):
        """
        Initializes the Registration Recall module.

        Args:
            threshold: RMSE threshold for successful registration
            k: Number of nearest neighbors for RMSE calculation
            reduction: Reduction method ('mean', 'sum', or 'none') - for compatibility
        """
        super(RegistrationRecall, self).__init__()
        self.threshold = threshold
        self.k = k
        self.reduction = reduction

    def forward(
        self,
        source: Union[torch.Tensor, List[torch.Tensor]],
        target: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Computes Registration Recall for single pair or dataset.

        Args:
            source: Source point cloud(s) - single tensor (N, 3) or list of tensors
            target: Target point cloud(s) - single tensor (M, 3) or list of tensors

        Returns:
            Registration recall value as tensor
        """
        if isinstance(source, list) and isinstance(target, list):
            # Dataset mode: process multiple pairs
            recall = registration_recall_dataset(
                source, target, self.threshold, self.k
            )
            return torch.tensor(recall, dtype=torch.float32)
        
        elif isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
            # Multi batch mode: get mean rmse for batch tensors
            if source.dim() == 3 and target.dim() == 3:
                rmse_values = torch.stack([
                    calculate_rmse_knn(source[i], target[i], self.k)
                    for i in range(source.size(0))
                ])
                rmse = rmse_values.mean()
            else:
                # Single pair mode: calculate one-way RMSE and return success (0 or 1)
                rmse = calculate_rmse_knn(source, target, self.k)
            
            # Return 1.0 if successful, 0.0 if not
            success = (rmse < self.threshold).float()
            return success
        
        else:
            raise ValueError("Source and target must both be tensors or both be lists")

    def evaluate_dataset(
        self,
        source_list: List[torch.Tensor],
        target_list: List[torch.Tensor],
        return_details: bool = False
    ) -> Union[float, Tuple[float, List[float], List[bool]]]:
        """
        Evaluate registration recall on a dataset with detailed results.
        
        Args:
            source_list: List of source point clouds
            target_list: List of target point clouds
            return_details: Whether to return detailed RMSE values and success flags
            
        Returns:
            Registration recall and optionally detailed results
        """
        return registration_recall_dataset(
            source_list, target_list, self.threshold, 
            self.k, return_details
        )

    def extra_repr(self) -> str:
        """Returns extra representation string for printing."""
        return (f'threshold={self.threshold}, '
                f'k={self.k}, reduction={self.reduction}')


def create_test_dataset(n_pairs: int = 10, n_points: int = 100) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Create a test dataset of point cloud pairs with varying registration quality.
    
    Args:
        n_pairs: Number of point cloud pairs to generate
        n_points: Number of points per cloud
        
    Returns:
        Tuple of (source_list, target_list)
    """
    source_list = []
    target_list = []
    
    for i in range(n_pairs):
        # Generate source point cloud
        source = torch.randn(n_points, 3)
        
        # Create target with varying amounts of noise/deformation
        noise_level = 0.1 + (i / n_pairs) * 0.5  # Increasing noise
        target = source + noise_level * torch.randn_like(source)
        
        source_list.append(source)
        target_list.append(target)
    
    return source_list, target_list




if __name__ == "__main__":
    # Test the Registration Recall implementation
    print("Testing Registration Recall implementation...")
    
    # Test 1: Single pair RMSE calculation
    print("\n1. Testing one-way RMSE calculation for single pairs:")
    source_single = torch.randn(50, 3)
    target_single = source_single + 0.1 * torch.randn_like(source_single)  # Small noise
    
    rmse = calculate_rmse_knn(source_single, target_single, k=1)
    
    print(f"One-way RMSE (source->target): {rmse.item():.4f}")
    
    # Test 2: Create test dataset
    print("\n2. Creating test dataset:")
    source_list, target_list = create_test_dataset(n_pairs=10, n_points=100)
    print(f"Created {len(source_list)} point cloud pairs")
    
    # Test 3: Dataset-level registration recall
    print("\n3. Testing dataset-level registration recall:")
    recall, rmse_values, success_flags = registration_recall_dataset(
        source_list, target_list, threshold=0.3, return_details=True
    )
    
    print(f"Overall Registration Recall: {recall:.4f}")
    print(f"Successful registrations: {sum(success_flags)}/{len(success_flags)}")
    print(f"RMSE values: {[f'{rmse:.3f}' for rmse in rmse_values[:5]]}... (showing first 5)")
    
    # Test 4: Module interface - dataset mode
    print("\n4. Testing module interface (dataset mode):")
    rr_module = RegistrationRecall(threshold=0.3, k=1)
    recall_module = rr_module(source_list, target_list)
    print(f"Registration Recall (module): {recall_module.item():.4f}")
    
    # Test 5: Module interface - single pair mode
    print("\n5. Testing module interface (single pair mode):")
    success_single = rr_module(source_single, target_single)
    print(f"Single pair success: {success_single.item():.0f} (1=success, 0=failure)")
    
    # Test 6: Different thresholds
    print("\n6. Testing different thresholds:")
    thresholds = [0.1, 0.2, 0.3, 0.5, 1.0]
    for threshold in thresholds:
        recall_thresh = registration_recall_dataset(source_list, target_list, threshold=threshold)
        print(f"Threshold {threshold}: Recall = {recall_thresh:.4f}")
    
    # Test 7: Detailed evaluation
    print("\n7. Testing detailed evaluation:")
    rr_eval = RegistrationRecall(threshold=0.25, k=1)
    recall_detailed, rmse_detailed, success_detailed = rr_eval.evaluate_dataset(
        source_list, target_list, return_details=True
    )
    
    print(f"Detailed evaluation - Recall: {recall_detailed:.4f}")
    print(f"Success pattern: {success_detailed}")
    print(f"Mean RMSE: {np.mean(rmse_detailed):.4f}")
    print(f"Std RMSE: {np.std(rmse_detailed):.4f}")
    
    # Test 8: Different k values for KNN
    print("\n8. Testing different k values:")
    k_values = [1, 3, 5]
    for k in k_values:
        rmse_k = calculate_rmse_knn(source_single, target_single, k=k)
        print(f"k={k}: RMSE = {rmse_k.item():.4f}")
    
    # Test 9: Perfect alignment test
    print("\n9. Testing perfect alignment:")
    perfect_recall = registration_recall_dataset([source_single], [source_single], threshold=0.1)
    print(f"Perfect alignment recall: {perfect_recall:.4f} (should be 1.0)")
    
    # Test 10: Very poor alignment test
    print("\n10. Testing very poor alignment:")
    poor_target = torch.randn_like(source_single) * 10  # Very different
    poor_recall = registration_recall_dataset([source_single], [poor_target], threshold=0.1)
    print(f"Poor alignment recall: {poor_recall:.4f} (should be 0.0)")
    
    print("\nAll Registration Recall tests completed!")


   
