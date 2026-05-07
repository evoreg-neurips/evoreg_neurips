"""
EvoReg: Versatile and Robust Point Cloud Registration via Multi-Stage Alignment.

A unified coarse-to-fine pipeline covering all four (rigid / non-rigid) x
(supervised / self-supervised) registration settings in a single architecture.
The pipeline composes a gradient-free CMA-ES pre-alignment over SE(3),
iterative Sinkhorn soft correspondences with confidence-weighted Kabsch
refinement, a residual MLP rigid head, and a conditional VAE that predicts a
residual non-rigid deformation field on the rigidly-aligned source. A
diffusion-based score-matching objective is used as an auxiliary loss during
training, and an optional suite of training-free refinement modules can be
composed at inference time.
"""

__version__ = '0.1.0'
__author__ = 'Anonymous'

# Will be populated as modules are implemented
__all__ = []
