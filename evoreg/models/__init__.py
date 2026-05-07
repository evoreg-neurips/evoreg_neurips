"""
Neural network models for EvoReg.

Provides the point-cloud encoder, residual MLP rigid head, conditional VAE
generator, and score-based diffusion architectures used in the multi-stage
EvoReg pipeline.
"""

from .pointnet_encoder import PointNetEncoder
from .vae_encoder import VAEEncoder, ConditionalVAEEncoder
from .generator import PointCloudGenerator, AttentionGenerator
from .cross_attention_generator import CrossAttentionGenerator
from .evoreg_model import EvoReg, EvoRegGeometric, EvoRegWithLosses, EvoRegWithDiffusion, create_evoreg
from .score_network import ScoreNetwork, UNetScoreNetwork
from .diffusion import DiffusionProcess, SimplifiedDiffusion
from .rigid_head import RigidHead, RigidNonRigidLoss, apply_rigid_transform
from .soft_correspondence import SoftCorrespondenceModule, DifferentiableKabsch
from .coarse_to_fine import EvoRegCoarseToFine

__all__ = [
    'PointNetEncoder',
    'VAEEncoder',
    'ConditionalVAEEncoder',
    'PointCloudGenerator',
    'AttentionGenerator',
    'CrossAttentionGenerator',
    'EvoReg',
    'EvoRegGeometric',
    'EvoRegWithLosses',
    'EvoRegWithDiffusion',
    'create_evoreg',
    'ScoreNetwork',
    'UNetScoreNetwork',
    'DiffusionProcess',
    'SimplifiedDiffusion',
    'RigidHead',
    'RigidNonRigidLoss',
    'apply_rigid_transform',
    'SoftCorrespondenceModule',
    'DifferentiableKabsch',
    'EvoRegCoarseToFine',
]
