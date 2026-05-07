"""
Rotation error.
"""

import math
from typing import Optional, Literal

import torch
import torch.nn as nn

def _rot_x(angles: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    """
    Batch rotation matrices around the x-axis.

    Args:
        angles: Tensor of shape (B,) or scalar () with angles.
        degrees: If True, 'angles' are in degrees; otherwise radians.

    Returns:
        R: Tensor of shape (B, 3, 3)
    """
    if angles.ndim == 0:
        angles = angles.unsqueeze(0)
    if degrees:
        angles = angles * (torch.pi / 180.0)

    c = torch.cos(angles)
    s = torch.sin(angles)
    B = angles.shape[0]

    R = torch.zeros(B, 3, 3, dtype=angles.dtype, device=angles.device)
    R[:, 0, 0] = 1
    R[:, 1, 1] =  c
    R[:, 1, 2] = -s
    R[:, 2, 1] =  s
    R[:, 2, 2] =  c
    return R

def _rot_y(angles: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    """
    Batch rotation matrices around the y-axis.

    Args:
        angles: Tensor of shape (B,) or scalar () with angles.
        degrees: If True, 'angles' are in degrees; otherwise radians.

    Returns:
        R: Tensor of shape (B, 3, 3)
    """
    if angles.ndim == 0:
        angles = angles.unsqueeze(0)
    if degrees:
        angles = angles * (torch.pi / 180.0)

    c = torch.cos(angles)
    s = torch.sin(angles)
    B = angles.shape[0]

    R = torch.zeros(B, 3, 3, dtype=angles.dtype, device=angles.device)
    R[:, 0, 0] =  c
    R[:, 0, 2] =  s
    R[:, 1, 1] =  1
    R[:, 2, 0] = -s
    R[:, 2, 2] =  c
    return R

def _rot_z(angles: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    """
    Batch rotation matrices around the z-axis.

    Args:
        angles: Tensor of shape (B,) or scalar () with angles.
        degrees: If True, 'angles' are in degrees; otherwise radians.

    Returns:
        R: Tensor of shape (B, 3, 3)
    """
    if angles.ndim == 0:
        angles = angles.unsqueeze(0)
    if degrees:
        angles = angles * (torch.pi / 180.0)

    c = torch.cos(angles)
    s = torch.sin(angles)
    B = angles.shape[0]

    R = torch.zeros(B, 3, 3, dtype=angles.dtype, device=angles.device)
    R[:, 0, 0] =  c
    R[:, 0, 1] = -s
    R[:, 1, 0] =  s
    R[:, 1, 1] =  c
    R[:, 2, 2] =  1
    return R

def rotation_error(R: torch.Tensor, R_gt: torch.Tensor):
    cos_theta = (torch.einsum('bij,bij->b', R, R_gt) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    return torch.acos(cos_theta) * 180 / math.pi

class RotationError(nn.Module):
    """
    Rotation Error loss module for point cloud registration.
    
    Provides a PyTorch module wrapper around the rotation_error function
    for use in neural network training pipelines.
    """
    
    def __init__(
        self,
        reduction: str = 'mean'
    ):
        super(RotationError, self).__init__()
        self.reduction = reduction

    def forward(
        self, 
        predicted_rot: torch.Tensor, 
        target_rot: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the rotation error loss.
        
        Args:
            predicted_rot: Predicted rotation matrices of shape (B, 3, 3) or (3, 3)
            target_rot: Target rotation matrices of shape (B, 3, 3) or (3, 3)
            
        Returns:
            rotation error value in degrees
        """
        # Handles both batched and unbatched inputs
        if predicted_rot.dim() == 2:
            predicted_rot = predicted_rot.unsqueeze(0)
        if target_rot.dim() == 2:
            target_rot = target_rot.unsqueeze(0)
        
        # Computes rotation error using the existing function
        error = rotation_error(predicted_rot, target_rot)
        
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
    print("Testing rotation_error implementation...")

    I = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)  # (4,3,3)
    # Case 1: identical -> 0 deg
    err0 = rotation_error(I, I)
    print("Identical rotations (deg):", err0.tolist())
    assert torch.allclose(err0, torch.zeros_like(err0), atol=1e-5)

    # Case 2: 90° rotation around x
    angle = torch.tensor([torch.pi/2])
    R_pred = _rot_x(angle)
    R_gt   = _rot_x(torch.zeros(1))
    err90 = rotation_error(R_pred, R_gt)
    print("90° rotation around x error (deg):", err90.tolist())
    assert torch.allclose(err90, torch.full_like(err90, 90.0), atol=1e-5)

    # Case 3: 180° rotation around x
    angle = torch.tensor([torch.pi])
    R_pred = _rot_x(angle)
    R_gt   = _rot_x(torch.zeros(1))
    err180 = rotation_error(R_pred, R_gt)
    print("180° rotation around x error (deg):", err180.tolist())
    assert torch.allclose(err180, torch.full_like(err180, 180.0), atol=1e-5)

    # Case 4: Batch with 4 different angles: 0°, 30°, 90°, 180°
    angles_deg = torch.tensor([0.0, 30.0, 90.0, 180.0])
    angles_rad = torch.deg2rad(angles_deg)
    
    R_pred = _rot_x(angles_rad)
    R_gt   = _rot_x(torch.zeros(4))

    err = rotation_error(R_pred, R_gt)
    print("Mixed-angle x-rot errors (deg):", err.tolist())
    expected = angles_deg
    assert torch.allclose(err, expected, atol=1e-5), \
        f"Expected {expected.tolist()}, got {err.tolist()}"

    # Case 5: 180° rotation around y
    angle = torch.tensor([torch.pi])
    R_pred = _rot_y(angle)
    R_gt   = _rot_y(torch.zeros(1))
    err180 = rotation_error(R_pred, R_gt)
    print("180° rotation around y error (deg):", err180.tolist())
    assert torch.allclose(err180, torch.full_like(err180, 180.0), atol=1e-5)

    # Case 6: 90° rotation around z
    angle = torch.tensor([torch.pi/2])
    R_pred = _rot_z(angle)
    R_gt   = _rot_z(torch.zeros(1))
    err90 = rotation_error(R_pred, R_gt)
    print("90° rotation around z error (deg):", err90.tolist())
    assert torch.allclose(err90, torch.full_like(err90, 90.0), atol=1e-5)

    # Tests module interface
    rotation_error_module = RotationError(reduction='mean')
    loss = rotation_error_module(R_pred, R_gt)
    print(f"Rotation error (module): {loss.item():.4f}")

    print("\nAll rotation error tests passed!")