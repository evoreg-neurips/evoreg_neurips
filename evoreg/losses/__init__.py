"""
Loss functions for EvoReg training.

Provides various loss functions for point cloud registration including
Chamfer distance, Earth Mover's distance, and KL divergence.
"""

from .chamfer_distance import ChamferDistance, chamfer_distance
from .kl_divergence import KLDivergenceLoss, kl_divergence

__all__ = [
    'ChamferDistance', 
    'chamfer_distance',
    'KLDivergenceLoss',
    'kl_divergence',
]