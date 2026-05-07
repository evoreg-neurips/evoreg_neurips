"""
Translation error.
"""

import math
from typing import Optional, Literal

import torch
import torch.nn as nn

def translation_error(t, t_gt):
    return torch.norm(t - t_gt, dim=1)

class TranslationError(nn.Module):
    """
    Translation Error loss module for point cloud registration.
    
    Provides a PyTorch module wrapper around the translation_error function
    for use in neural network training pipelines.
    """
    
    def __init__(
        self,
        reduction: str = 'mean'
    ):
        super(TranslationError, self).__init__()
        self.reduction = reduction

    def forward(
        self, 
        predicted_rot: torch.Tensor, 
        target_rot: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the translation error loss.
        
        Args:
            predicted_rot: Predicted translation matrices of shape (B, 3, 3) or (3, 3)
            target_rot: Target translation matrices of shape (B, 3, 3) or (3, 3)
            
        Returns:
            translation error value in degrees
        """
        # Handles both batched and unbatched inputs
        if predicted_rot.dim() == 2:
            predicted_rot = predicted_rot.unsqueeze(0)
        if target_rot.dim() == 2:
            target_rot = target_rot.unsqueeze(0)
        
        # Computes rotation error using the existing function
        error = translation_error(predicted_rot, target_rot)
        
        # Applies reduction
        if self.reduction == 'mean':
            return torch.mean(error)
        elif self.reduction == 'sum':
            return torch.sum(error)
        elif self.reduction == 'none':
            return error
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")

if __name__ == "__main__":
    # Basic correctness tests
    print("Testing translation_error implementation...")

    t_gt = torch.tensor([[0.0, 0.0, 0.0]])

    # Case 1: Move along x by +5
    t_x = torch.tensor([[5.0, 0.0, 0.0]])
    error_x = translation_error(t_x, t_gt)
    print("Translation error (x):", error_x.tolist())
    assert torch.allclose(error_x, torch.tensor([5.0]))

    # Case 2: Move along y by +5
    t_y = torch.tensor([[0.0, 5.0, 0.0]])
    error_y = translation_error(t_y, t_gt)
    print("Translation error (y):", error_y.tolist())
    assert torch.allclose(error_y, torch.tensor([5.0]))

    # Case 3: Move along z by +5
    t_z = torch.tensor([[0.0, 0.0, 5.0]])
    error_z = translation_error(t_z, t_gt)
    print("Translation error (z):", error_z.tolist())
    assert torch.allclose(error_z, torch.tensor([5.0]))

    # Case 4: Batch test (all together)
    t_batch = torch.tensor([
        [5.0, 0.0, 0.0],  # x-shift
        [0.0, 5.0, 0.0],  # y-shift
        [0.0, 0.0, 5.0],  # z-shift
    ])
    t_gt_batch = torch.zeros_like(t_batch)
    error_batch = translation_error(t_batch, t_gt_batch)
    print("Translation error (batch):", error_batch.tolist())
    expected_batch = torch.tensor([5.0, 5.0, 5.0])
    assert torch.allclose(error_batch, expected_batch)

    # Tests module interface
    translation_error_module = TranslationError(reduction='mean')
    t_gt = torch.tensor([[0.0, 3.0, 0.0]])
    t_x = torch.tensor([[5.0, 0.0, 1.0]])
    translation_error = translation_error_module(t_x, t_gt)
    print(f"Translation error (module): {translation_error.item():.4f}")

    print("\nAll translation error tests passed!")