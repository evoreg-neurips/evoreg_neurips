"""
Training script for EvoReg model.

Provides complete training pipeline for EvoReg on point cloud registration tasks.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import glob, random
import socket
import subprocess
import trimesh

# Adds EvoReg to path
sys.path.append(str(Path(__file__).parent))

from evoreg.models import create_evoreg
from evoreg.models.rigid_head import RigidNonRigidLoss
from evoreg.data.synthetic_data import generate_sphere, generate_cube, generate_cylinder, generate_torus, generate_registration_pair
from evoreg.data.modelnet40_dataset import ModelNet40RegistrationDataset
try:
    from evoreg.data.faust_dataset import FAUSTRegistrationDataset as FaustDataset
except ImportError:
    FaustDataset = None
from evoreg.data.spare_dataset import SpareDataset
from evoreg.data.shapenet_dataset import ShapeNetRegistrationDataset
from evoreg.data.match3d_dataset import Match3DRegistrationDataset, Match3DPairDataset
from evoreg.data.utils import generate_transformation, apply_occlusion_augmentation
from evoreg.utils.visualization import visualize_registration, save_visualization
from evoreg.losses import ChamferDistance


class SyntheticRegistrationDataset(Dataset):
    """
    Dataset for synthetic point cloud registration pairs.
    
    Generates registration pairs on-the-fly during training.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_points: int = 1000,
        shape_types: Tuple[str, ...] = ('sphere', 'cube', 'cylinder', 'torus'),
        noise_std: float = 0.01,
        outlier_ratio: float = 0.0,
        partial_overlap: float = 1.0
    ):
        """
        Initializes synthetic dataset.
        
        Args:
            n_samples: Number of samples in dataset
            n_points: Number of points per cloud
            shape_types: Types of shapes to generate
            noise_std: Noise standard deviation
            outlier_ratio: Ratio of outliers
            partial_overlap: Overlap between source and target
        """
        self.n_samples = n_samples
        self.n_points = n_points
        self.shape_types = shape_types
        self.noise_std = noise_std
        self.outlier_ratio = outlier_ratio
        self.partial_overlap = partial_overlap
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generates a registration pair
        shape_type = self.shape_types[idx % len(self.shape_types)]
        
        pair = generate_registration_pair(
            shape_type=shape_type,
            n_points=self.n_points,
            noise_std=self.noise_std,
            outlier_ratio=self.outlier_ratio,
            partial_overlap=self.partial_overlap
        )
        
        # Converts to torch tensors
        source = torch.from_numpy(pair['source']).float()
        target = torch.from_numpy(pair['target']).float()
        transformation = torch.from_numpy(pair['transformation']).float()
        
        return {
            'source': source,
            'target': target,
            'transformation': transformation
        }

def _normalize(P: np.ndarray):
    c = P.mean(axis=0, keepdims=True)
    P0 = P - c
    s = np.sqrt((P0**2).sum(axis=1).mean()) + 1e-9
    return (P0 / s).astype(np.float32)

def _rand_rot(max_deg: float, rng: np.random.Generator):
    axis = rng.normal(size=3); axis /= (np.linalg.norm(axis) + 1e-9)
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
    return R



class EvoRegTrainer:
    """
    Trainer class for EvoReg model.
    
    Handles training loop, validation, checkpointing, and logging.
    Supports both VAE-only and VAE+Diffusion training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        chamfer_weight: float = 1.0,
        kl_weight: float = 0.001,
        diffusion_weight: float = 1.0,
        kl_annealing: bool = True,
        with_diffusion: bool = False,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        use_weighted_loss: bool = True,
        use_lr_schedule: bool = False,
        lr_schedule_step: int = 5,
        lr_schedule_gamma: float = 0.5,
        use_occlusion: bool = False,
        occlusion_type: str = 'dropout',
        occlusion_ratio: float = 0.2,
        occlusion_prob: float = 0.5,
        occlusion_noise_std: float = 0.01,
        use_rigid_head: bool = False,
        use_rt_supervision: bool = False,
        lambda_rot: float = 1.0,
        lambda_trans: float = 1.0,
        lambda_rmse: float = 1.0,
        lambda_align: float = 1.0,
        lambda_disp: float = 0.01,
        lambda_deform: float = 0.1,
        lambda_lap: float = 0.1,
        use_correspondence_loss: bool = False,
        lambda_correspondence: float = 0.1,
        lambda_rigid_chamfer: float = 0.0,
        lambda_cycle: float = 0.0,
        lambda_intermediate_chamfer: float = 0.0,
        lambda_centroid: float = 0.0,
        lambda_centroid_s1: Optional[float] = None,
        lambda_centroid_s2: Optional[float] = None,
        lambda_centroid_s3: Optional[float] = None,
    ):
        """
        Initializes trainer.

        Args:
            model: EvoReg model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            chamfer_weight: Weight for Chamfer loss
            kl_weight: Weight for KL loss
            diffusion_weight: Weight for diffusion loss
            kl_annealing: Whether to use KL annealing
            with_diffusion: Whether model includes diffusion
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_interval: Logging frequency
            use_weighted_loss: Whether to use weighted Chamfer loss
            use_lr_schedule: Whether to use learning rate scheduling
            lr_schedule_step: Epochs between LR decay steps
            lr_schedule_gamma: LR decay factor
            use_occlusion: Whether to use occlusion augmentation
            occlusion_type: Type of occlusion ('dropout', 'region', 'noise', 'outlier')
            occlusion_ratio: Ratio for occlusion (e.g., 0.2 = 20% dropout)
            occlusion_prob: Probability of applying occlusion to a batch
            occlusion_noise_std: Standard deviation for Gaussian noise (if using 'noise' type)
            use_rigid_head: Whether model uses rigid SE(3) head
            use_rt_supervision: Whether to use R/t supervision losses against ground truth
            lambda_rot: Weight for rotation loss
            lambda_trans: Weight for translation loss
            lambda_rmse: Weight for RMSE loss
            lambda_align: Weight for alignment loss
            lambda_disp: Weight for displacement regularization
            lambda_deform: Weight for deformation smoothness
            lambda_lap: Weight for Laplacian smoothness
            use_correspondence_loss: Whether to use point-to-point correspondence loss
            lambda_correspondence: Weight for correspondence loss (MSE)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.with_diffusion = with_diffusion
        self.use_weighted_loss = use_weighted_loss

        # Occlusion augmentation settings (inspired by PRNet)
        self.use_occlusion = use_occlusion
        self.occlusion_type = occlusion_type
        self.occlusion_ratio = occlusion_ratio
        self.occlusion_prob = occlusion_prob
        self.occlusion_noise_std = occlusion_noise_std

        # Creates checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Sets up optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Sets up learning rate scheduler (inspired by PRNet)
        self.use_lr_schedule = use_lr_schedule
        self.scheduler = None
        if use_lr_schedule:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_schedule_step,
                gamma=lr_schedule_gamma
            )
            print(f"LR scheduler enabled: step_size={lr_schedule_step}, gamma={lr_schedule_gamma}")

        # Sets up loss functions
        # Always create chamfer_loss — with diffusion+c2f, losses are computed
        # manually by the trainer (not by EvoRegWithDiffusion)
        self.chamfer_loss = ChamferDistance(reduction='mean')
        self.chamfer_weight = chamfer_weight
        self.kl_weight = kl_weight
        self.diffusion_weight = diffusion_weight
        self.kl_annealing = kl_annealing

        # Rigid head settings
        self.use_rigid_head = use_rigid_head
        self.use_rt_supervision = use_rt_supervision
        if use_rigid_head:
            self.rigid_loss = RigidNonRigidLoss(
                lambda_rot=lambda_rot,
                lambda_trans=lambda_trans,
                lambda_rmse=lambda_rmse,
                lambda_align=lambda_align,
                lambda_disp=lambda_disp,
                lambda_deform=lambda_deform,
                lambda_lap=lambda_lap
            ).to(device)
            print(f"Rigid head enabled with loss weights: rot={lambda_rot}, trans={lambda_trans}, "
                  f"rmse={lambda_rmse}, align={lambda_align}, disp={lambda_disp}, "
                  f"deform={lambda_deform}, lap={lambda_lap}")

        # Correspondence loss settings
        self.use_correspondence_loss = use_correspondence_loss
        self.lambda_correspondence = lambda_correspondence
        self.lambda_rigid_chamfer = lambda_rigid_chamfer
        self.lambda_cycle = lambda_cycle
        self.lambda_intermediate_chamfer = lambda_intermediate_chamfer
        # Centroid loss weights: either a single shared lambda or per-stage lambdas.
        self.lambda_centroid_s1 = lambda_centroid if lambda_centroid_s1 is None else lambda_centroid_s1
        self.lambda_centroid_s2 = lambda_centroid if lambda_centroid_s2 is None else lambda_centroid_s2
        self.lambda_centroid_s3 = lambda_centroid if lambda_centroid_s3 is None else lambda_centroid_s3
        self.lambda_centroid = self.lambda_centroid_s1 + self.lambda_centroid_s2 + self.lambda_centroid_s3
        if lambda_cycle > 0:
            print(f"Cycle consistency loss enabled with weight: lambda_cycle={lambda_cycle}")
        if lambda_intermediate_chamfer > 0:
            print(f"Intermediate Chamfer loss enabled with weight: lambda_intermediate_chamfer={lambda_intermediate_chamfer}")
        if use_correspondence_loss:
            print(f"Correspondence loss enabled with weight: lambda_correspondence={lambda_correspondence}")
        if (self.lambda_centroid_s1 + self.lambda_centroid_s2 + self.lambda_centroid_s3) > 0:
            print(
                f"Centroid loss enabled with weights: "
                f"lambda_centroid_s1={self.lambda_centroid_s1}, "
                f"lambda_centroid_s2={self.lambda_centroid_s2}, "
                f"lambda_centroid_s3={self.lambda_centroid_s3}"
            )

        # Tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        # Track additional metrics that appear in tqdm postfix (epoch averages)
        self.train_metrics = {
            'chamfer': [],
            'kl': [],
            'diffusion_loss': [],
            'rigid_total': [],
            'correspondence': [],
            'cycle': [],
            'centroid': [],
            'centroid_s1': [],
            'centroid_s2': [],
            'centroid_s3': [],
        }
        self.val_metrics = {
            'chamfer': [],
            'kl': [],
            'diffusion_loss': [],
            'rigid_total': [],
            'correspondence': [],
            'cycle': [],
            'centroid': [],
            'centroid_s1': [],
            'centroid_s2': [],
            'centroid_s3': [],
        }

    def compute_point_weights(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes importance weights for points (inspired by PRNet).

        Weights points based on their distance from centroid to emphasize
        boundary/discriminative regions similar to PRNet's weighted loss.

        Args:
            source: Source point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)

        Returns:
            Point weights (B, N) with higher values for important regions
        """
        batch_size, n_points = source.shape[0], source.shape[1]

        # Compute centroid
        centroid = source.mean(dim=1, keepdim=True)  # (B, 1, 3)

        # Distance from centroid (boundary points get higher weight)
        dist_to_center = torch.norm(source - centroid, dim=-1)  # (B, N)

        # Normalize to [0, 1] range per batch
        dist_normalized = dist_to_center / (dist_to_center.max(dim=1, keepdim=True)[0] + 1e-8)

        # Weight formula inspired by PRNet's region-based weighting
        # Base weight of 3.0, with up to 4.0 additional weight for boundary points
        # This roughly approximates PRNet's 16:4:3:0 ratio for landmarks:features:face:neck
        weights = 3.0 + 4.0 * dist_normalized

        return weights

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        step: Optional[int] = None,
        source: Optional[torch.Tensor] = None,
        R_gt: Optional[torch.Tensor] = None,
        t_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes training losses.

        Args:
            outputs: Model outputs dictionary
            target: Target point cloud
            step: Current training step
            source: Source point cloud (needed for weighted loss and rigid losses)
            R_gt: Ground truth rotation matrix (B, 3, 3) - for rigid head supervision
            t_gt: Ground truth translation vector (B, 3) - for rigid head supervision

        Returns:
            Dictionary of losses
        """
        # All loss computation is done here (works for VAE-only, c2f, and diffusion models)
        # Compute point weights if using weighted loss
        weights = None
        if self.use_weighted_loss and source is not None:
            weights = self.compute_point_weights(source, target)

        # Computes Chamfer distance (with optional weights)
        chamfer = self.chamfer_loss(outputs['output'], target, weights=weights)
        
        # Computes KL divergence
        mu = outputs['mu']
        log_var = outputs['log_var']
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Applies KL annealing if enabled
        if self.kl_annealing and step is not None:
            kl_weight = min(self.kl_weight, self.kl_weight * step / 10000)
        else:
            kl_weight = self.kl_weight

        # Initialize rigid loss components
        rigid_loss_dict = {}
        rigid_total = torch.tensor(0.0, device=self.device)

        # Computes rigid losses if rigid head is enabled
        if self.use_rigid_head and 'R_pred' in outputs and 't_pred' in outputs:
            # Check if ground truth is available
            if R_gt is not None and t_gt is not None:
                # Compute rigid losses using the RigidNonRigidLoss module
                rigid_loss_dict = self.rigid_loss(
                    Y=source,  # Original source before rigid transformation
                    X=target,  # Target point cloud
                    R_pred=outputs['R_pred'],  # Predicted rotation
                    t_pred=outputs['t_pred'],  # Predicted translation
                    R_gt=R_gt,  # Ground truth rotation
                    t_gt=t_gt,  # Ground truth translation
                    X_hat=outputs['output'],  # Final output (after non-rigid)
                    delta=outputs.get('delta')  # Displacement field
                )
                rigid_total = rigid_loss_dict['total_loss']
            else:
                # No ground truth: compute only non-rigid losses
                # (rigid losses L_rot, L_trans, L_rmse need GT)
                from evoreg.models.rigid_head import (
                    compute_alignment_loss, compute_displacement_loss,
                    compute_deformation_loss, compute_laplacian_loss
                )

                # Use source_rigid_2 (after residual rigid head) if available (c2f model),
                # otherwise fall back to source_rigid (base EvoReg)
                Y_rigid = outputs.get('source_rigid_2', outputs.get('source_rigid'))
                X_hat = outputs['output']
                delta = outputs.get('delta')

                L_align = compute_alignment_loss(X_hat, target, k=self.rigid_loss.k_neighbors).mean()
                L_disp = compute_displacement_loss(delta).mean()
                L_deform = compute_deformation_loss(Y_rigid, X_hat, k=self.rigid_loss.k_neighbors).mean()
                L_lap = compute_laplacian_loss(delta, Y_rigid, k=self.rigid_loss.k_neighbors).mean()

                rigid_total = (
                    self.rigid_loss.lambda_align * L_align +
                    self.rigid_loss.lambda_disp * L_disp +
                    self.rigid_loss.lambda_deform * L_deform +
                    self.rigid_loss.lambda_lap * L_lap
                )

                rigid_loss_dict = {
                    'total_loss': rigid_total,
                    'non_rigid_loss': rigid_total,
                    'L_align': L_align,
                    'L_disp': L_disp,
                    'L_deform': L_deform,
                    'L_lap': L_lap
                }

        # Computes correspondence loss (MSE between output and target)
        correspondence_loss = torch.tensor(0.0, device=target.device)
        if self.use_correspondence_loss:
            # Direct point-to-point MSE loss to enforce i->i correspondence
            correspondence_loss = torch.mean((outputs['output'] - target) ** 2)

        # Coarse-to-fine: Stage 1 rigid Chamfer loss
        rigid_chamfer = torch.tensor(0.0, device=target.device)
        if self.lambda_rigid_chamfer > 0 and 'source_rigid' in outputs:
            rigid_chamfer = self.chamfer_loss(outputs['source_rigid'], target, weights=weights)

        # Cycle consistency loss on correspondences
        cycle_loss = torch.tensor(0.0, device=target.device)
        if self.lambda_cycle > 0 and 'cycle_matrix' in outputs:
            cycle_matrix = outputs['cycle_matrix']  # (B, N, N)
            N = cycle_matrix.shape[1]
            identity = torch.eye(N, device=target.device).unsqueeze(0)  # (1, N, N)
            cycle_loss = torch.mean((cycle_matrix - identity) ** 2)

        # Intermediate Chamfer supervision at each SVD iteration
        intermediate_chamfer = torch.tensor(0.0, device=target.device)
        if self.lambda_intermediate_chamfer > 0 and 'intermediate_sources' in outputs:
            intermediates = outputs['intermediate_sources']
            n_iters = len(intermediates)
            for i, src_i in enumerate(intermediates[:-1]):  # skip last (already in rigid_chamfer)
                weight = (i + 1) / n_iters  # 0.33, 0.67 for 3 iters
                intermediate_chamfer = intermediate_chamfer + weight * self.chamfer_loss(src_i, target, weights=weights)

        # Centroid losses (self-supervised): encourage aligned source centroid to match target centroid at each stage.
        centroid_loss_stage1 = torch.tensor(0.0, device=target.device)
        centroid_loss_stage2 = torch.tensor(0.0, device=target.device)
        centroid_loss_stage3 = torch.tensor(0.0, device=target.device)

        if (self.lambda_centroid_s1 + self.lambda_centroid_s2 + self.lambda_centroid_s3) > 0:
            tgt_centroid = target.mean(dim=1)  # (B, 3)

            if 'source_rigid' in outputs:
                src1_centroid = outputs['source_rigid'].mean(dim=1)
                centroid_loss_stage1 = torch.norm(src1_centroid - tgt_centroid, dim=1).mean()

            if 'source_rigid_2' in outputs:
                src2_centroid = outputs['source_rigid_2'].mean(dim=1)
                centroid_loss_stage2 = torch.norm(src2_centroid - tgt_centroid, dim=1).mean()

            # Final output always present
            out_centroid = outputs['output'].mean(dim=1)
            centroid_loss_stage3 = torch.norm(out_centroid - tgt_centroid, dim=1).mean()

        centroid_loss = (
            self.lambda_centroid_s1 * centroid_loss_stage1
            + self.lambda_centroid_s2 * centroid_loss_stage2
            + self.lambda_centroid_s3 * centroid_loss_stage3
        )

        # Diffusion score-matching loss (when wrapped with EvoRegWithDiffusion)
        diffusion_loss = torch.tensor(0.0, device=target.device)
        if 'diffusion_loss' in outputs:
            diffusion_loss = outputs['diffusion_loss']

        # Computes total loss
        total_loss = (
            self.chamfer_weight * chamfer + kl_weight * kl + rigid_total
            + self.lambda_correspondence * correspondence_loss
            + self.lambda_rigid_chamfer * rigid_chamfer
            + self.diffusion_weight * diffusion_loss
            + self.lambda_cycle * cycle_loss
            + self.lambda_intermediate_chamfer * intermediate_chamfer
            + centroid_loss
        )

        # Prepare return dictionary
        result = {
            'loss': total_loss,
            'chamfer': chamfer,
            'kl': kl,
            'kl_weight': kl_weight,
        }

        # Add correspondence loss if used
        if self.use_correspondence_loss:
            result['correspondence'] = correspondence_loss

        # Add diffusion loss if present
        if 'diffusion_loss' in outputs:
            result['diffusion_loss'] = diffusion_loss

        # Add cycle consistency loss if used
        if self.lambda_cycle > 0:
            result['cycle'] = cycle_loss

        # Add intermediate Chamfer loss if used
        if self.lambda_intermediate_chamfer > 0:
            result['inter_chamfer'] = intermediate_chamfer

        # Add coarse-to-fine rigid Chamfer loss
        if self.lambda_rigid_chamfer > 0:
            result['rigid_chamfer'] = rigid_chamfer

        # Add rigid loss components if computed
        if rigid_loss_dict:
            result['rigid_total'] = rigid_total
            result.update(rigid_loss_dict)

        # Add centroid losses if enabled
        if (self.lambda_centroid_s1 + self.lambda_centroid_s2 + self.lambda_centroid_s3) > 0:
            result['centroid'] = centroid_loss
            result['centroid_s1'] = centroid_loss_stage1
            result['centroid_s2'] = centroid_loss_stage2
            result['centroid_s3'] = centroid_loss_stage3

        return result
    
    def train_epoch(self):
        """
        Trains for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        epoch_losses = []
        metric_sums = {k: 0.0 for k in self.train_metrics.keys()}
        metric_counts = {k: 0 for k in self.train_metrics.keys()}

        # Creates progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Moves data to device
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            # Apply occlusion augmentation to source (inspired by PRNet)
            # Only applied during training with specified probability
            if self.use_occlusion and np.random.rand() < self.occlusion_prob:
                # Convert to numpy for augmentation
                source_np = source.cpu().numpy()

                # Apply occlusion to each sample in batch
                augmented_sources = []
                for i in range(source_np.shape[0]):
                    augmented = apply_occlusion_augmentation(
                        source_np[i],
                        occlusion_type=self.occlusion_type,
                        occlusion_ratio=self.occlusion_ratio,
                        noise_std=self.occlusion_noise_std
                    )

                    # Pad or sample to maintain consistent point count
                    n_points = source_np.shape[1]
                    if len(augmented) < n_points:
                        # Pad with random sampling if too few points
                        indices = np.random.choice(len(augmented), n_points, replace=True)
                        augmented = augmented[indices]
                    elif len(augmented) > n_points:
                        # Sample if too many points (for outlier injection)
                        indices = np.random.choice(len(augmented), n_points, replace=False)
                        augmented = augmented[indices]

                    augmented_sources.append(augmented)

                # Convert back to tensor
                source = torch.from_numpy(np.stack(augmented_sources)).float().to(self.device)

            # Forward pass (return_latent works for all model types including diffusion wrapper)
            outputs = self.model(source, target, return_latent=True)

            # Extract ground truth R/t when supervised mode is enabled
            R_gt = None
            t_gt = None
            if self.use_rt_supervision:
                R_gt = batch.get('rotation', None)
                t_gt = batch.get('translation', None)
                if R_gt is not None:
                    R_gt = R_gt.to(self.device)
                if t_gt is not None:
                    t_gt = t_gt.to(self.device)

            # Computes losses
            losses = self.compute_loss(
                outputs, target, self.global_step,
                source=source, R_gt=R_gt, t_gt=t_gt
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Updates tracking
            epoch_losses.append(losses['loss'].item())
            self.global_step += 1

            # Accumulate metrics for epoch averages
            for k in metric_sums.keys():
                if k in losses:
                    metric_sums[k] += float(losses[k].item())
                    metric_counts[k] += 1

            # Updates progress bar
            if batch_idx % self.log_interval == 0:
                postfix = {
                    'loss': f"{losses['loss'].item():.4f}",
                    'chamfer': f"{losses['chamfer'].item():.4f}",
                    'kl': f"{losses['kl'].item():.4f}",
                }

                # Show diffusion loss when present
                if 'diffusion_loss' in losses:
                    postfix['diff'] = f"{losses['diffusion_loss'].item():.4f}"

                # Add rigid loss info if computed
                if 'rigid_total' in losses:
                    postfix['rigid'] = f"{losses['rigid_total'].item():.4f}"

                # Add correspondence loss info if computed
                if 'correspondence' in losses:
                    postfix['corr'] = f"{losses['correspondence'].item():.4f}"

                # Add cycle consistency loss info if computed
                if 'cycle' in losses:
                    postfix['cycle'] = f"{losses['cycle'].item():.4f}"

                if 'centroid' in losses:
                    postfix['cent'] = f"{losses['centroid'].item():.4f}"
                    postfix['c1'] = f"{losses['centroid_s1'].item():.4f}"
                    postfix['c2'] = f"{losses['centroid_s2'].item():.4f}"
                    postfix['c3'] = f"{losses['centroid_s3'].item():.4f}"

                pbar.set_postfix(postfix)

        epoch_avg_loss = float(np.mean(epoch_losses))
        epoch_metrics = {}
        for k in metric_sums.keys():
            if metric_counts[k] > 0:
                epoch_metrics[k] = metric_sums[k] / metric_counts[k]
            else:
                epoch_metrics[k] = None
        return epoch_avg_loss, epoch_metrics
    
    def validate(self):
        """
        Validates the model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        val_losses = []
        metric_sums = {k: 0.0 for k in self.val_metrics.keys()}
        metric_counts = {k: 0 for k in self.val_metrics.keys()}
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Moves data to device
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)

                # Forward pass
                outputs = self.model(source, target, return_latent=True)

                # Computes losses
                losses = self.compute_loss(outputs, target, source=source)
                val_losses.append(losses['loss'].item())
                for k in metric_sums.keys():
                    if k in losses:
                        metric_sums[k] += float(losses[k].item())
                        metric_counts[k] += 1
        epoch_avg_loss = float(np.mean(val_losses))
        epoch_metrics = {}
        for k in metric_sums.keys():
            if metric_counts[k] > 0:
                epoch_metrics[k] = metric_sums[k] / metric_counts[k]
            else:
                epoch_metrics[k] = None
        return epoch_avg_loss, epoch_metrics
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """
        Saves model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_config': getattr(self, 'model_config', {}),
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pth'):
        """
        Loads model checkpoint.

        Args:
            filename: Checkpoint filename or path
        """
        # Handle both absolute/relative paths and just filenames
        checkpoint_path = Path(filename)
        if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
            # Try in checkpoint_dir if not absolute and doesn't exist as-is
            checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            print(f"Checkpoint {filename} not found at {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, will resume from epoch {self.current_epoch}")
    
    def train(self, n_epochs: int):
        """
        Trains the model for specified epochs.

        Args:
            n_epochs: Number of epochs to train (total, not additional)
        """
        start_epoch = self.current_epoch
        print(f"Starting training from epoch {start_epoch} to {n_epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(start_epoch, n_epochs):
            self.current_epoch = epoch

            # Training
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            for k, v in train_metrics.items():
                self.train_metrics[k].append(v)

            # Validation
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            for k, v in val_metrics.items():
                self.val_metrics[k].append(v)

            # Step learning rate scheduler if enabled
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            else:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Saves checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')

            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

        # Saves final model
        self.save_checkpoint('final_model.pth')

        print("Training completed!")


def main():
    """
    Main training function.
    """
    # Parses arguments
    parser = argparse.ArgumentParser(description='Train EvoReg model')
    parser.add_argument('--lambda_centroid', type=float, default=0.0,
                        help='Shared centroid loss weight for all stages (default: 0.0)')
    parser.add_argument('--lambda_centroid_s1', type=float, default=None,
                        help='Stage-1 centroid loss weight (overrides --lambda_centroid if set)')
    parser.add_argument('--lambda_centroid_s2', type=float, default=None,
                        help='Stage-2 centroid loss weight (overrides --lambda_centroid if set)')
    parser.add_argument('--lambda_centroid_s3', type=float, default=None,
                        help='Stage-3 centroid loss weight (overrides --lambda_centroid if set)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--n_points', type=int, default=1000, help='Points per cloud')
    parser.add_argument('--noise', type=float, default=0.01, help='Noise level')
    parser.add_argument('--with_diffusion', action='store_true', help='Train with diffusion')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Diffusion timesteps')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'modelnet40', 'faust', 'shapenet', '3dmatch'],
                        help='Dataset to use for training')
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='Directory containing training data files')
    parser.add_argument('--val_data_dir', type=str, default=None,
                        help='Directory containing validation data files')
    parser.add_argument('--file_dir', type=str, default=None,
                        help='Path to ModelNet40 .off file (e.g., airplane_train.off)')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of training samples for ModelNet40 (default: 1000)')
    
    # 3DMatch specific arguments
    parser.add_argument('--max_pairs_per_scene', type=int, default=None,
                        help='Maximum pairs per scene for 3DMatch (None = all pairs)')
    parser.add_argument('--use_gt_pairs', action='store_true',
                        help='Use ground truth pairs for 3DMatch (vs generated pairs)')
    
    parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object during training, and generated by atlasnet')
    parser.add_argument("--demo", action="store_true", help="run demo autoencoder or single-view")
    parser.add_argument("--dir", type=str, help="data directory for shapenet")
    parser.add_argument("--folder_name", type=str, help="folder name for shapenet", default="non_existing_folder")    
    parser.add_argument("--shapenet13", action="store_true", help="Load 13 usual shapenet categories")
    parser.add_argument("--SVR", action="store_true", help="Single_view Reconstruction")
    parser.add_argument('--class_choice', nargs='+', default=["airplane"], type=str)
    parser.add_argument('--normalization', type=str, default="UnitBall",
                        choices=['UnitBall', 'BoundingBox', 'Identity'])
    parser.add_argument('--rotation_range', type=float, default=None,
                        help='Constrain rotations to ±N degrees (default: None = full SO(3))')
    parser.add_argument('--translation_range', type=float, default=0.2,
                        help='Translation range ±N (default: 0.2)')
    parser.add_argument("--sample", action="store_false", help="Sample the input pointclouds")
    parser.add_argument("--save_data", type=str, help="Path to save data (folder name)", default=None)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')

    # Weighted loss arguments (PRNet-inspired)
    parser.add_argument('--use_weighted_loss', action='store_true', default=True,
                        help='Use weighted Chamfer loss (inspired by PRNet)')
    parser.add_argument('--no_weighted_loss', dest='use_weighted_loss', action='store_false',
                        help='Disable weighted loss (use uniform weights)')

    # Learning rate scheduling arguments (PRNet-inspired)
    parser.add_argument('--use_lr_schedule', action='store_true', default=False,
                        help='Enable learning rate scheduling (decay every N epochs)')
    parser.add_argument('--lr_schedule_step', type=int, default=5,
                        help='Number of epochs between LR decay steps (default: 5)')
    parser.add_argument('--lr_schedule_gamma', type=float, default=0.5,
                        help='LR decay factor (default: 0.5, i.e., halve LR)')

    # Occlusion augmentation arguments (PRNet-inspired)
    parser.add_argument('--use_occlusion', action='store_true', default=False,
                        help='Enable occlusion augmentation during training')
    parser.add_argument('--occlusion_type', type=str, default='dropout',
                        choices=['dropout', 'region', 'noise', 'outlier'],
                        help='Type of occlusion augmentation (default: dropout)')
    parser.add_argument('--occlusion_ratio', type=float, default=0.2,
                        help='Occlusion ratio: for dropout/region (fraction removed), '
                             'for outlier (fraction added) (default: 0.2)')
    parser.add_argument('--occlusion_prob', type=float, default=0.5,
                        help='Probability of applying occlusion to a batch (default: 0.5)')
    parser.add_argument('--occlusion_noise_std', type=float, default=0.01,
                        help='Standard deviation for Gaussian noise (only used with --occlusion_type noise)')

    # Rigid head arguments (for rigid + non-rigid registration)
    parser.add_argument('--use_rigid_head', action='store_true', default=False,
                        help='Enable rigid SE(3) head for hybrid rigid + non-rigid registration')
    parser.add_argument('--lambda_rot', type=float, default=1.0,
                        help='Weight for rotation loss (default: 1.0)')
    parser.add_argument('--lambda_trans', type=float, default=1.0,
                        help='Weight for translation loss (default: 1.0)')
    parser.add_argument('--lambda_rmse', type=float, default=1.0,
                        help='Weight for RMSE loss (default: 1.0)')
    parser.add_argument('--lambda_align', type=float, default=1.0,
                        help='Weight for alignment loss (default: 1.0)')
    parser.add_argument('--lambda_disp', type=float, default=0.01,
                        help='Weight for displacement regularization (default: 0.01)')
    parser.add_argument('--lambda_deform', type=float, default=0.1,
                        help='Weight for deformation smoothness (default: 0.1)')
    parser.add_argument('--lambda_lap', type=float, default=0.1,
                        help='Weight for Laplacian smoothness (default: 0.1)')

    # Correspondence loss arguments (for enforcing point-to-point correspondence)
    parser.add_argument('--use_correspondence_loss', action='store_true', default=False,
                        help='Enable point-to-point correspondence loss (MSE) to enforce i->i matching')
    parser.add_argument('--lambda_correspondence', type=float, default=0.1,
                        help='Weight for correspondence loss (default: 0.1)')

    # Geometric cross-attention arguments (experiment)
    parser.add_argument('--use_geometric_attention', action='store_true', default=False,
                        help='Use geometric cross-attention encoder (experiment branch)')
    parser.add_argument('--geo_attn_heads', type=int, default=4,
                        help='Number of attention heads for geometric cross-attention (default: 4)')
    parser.add_argument('--geo_attn_blocks', type=int, default=3,
                        help='Number of geometric cross-attention blocks (default: 3)')
    parser.add_argument('--geo_num_rbf', type=int, default=16,
                        help='Number of Gaussian RBF centres for distance encoding (default: 16)')
    parser.add_argument('--geo_rbf_cutoff', type=float, default=2.0,
                        help='RBF cutoff distance — should cover point cloud diameter (default: 2.0)')

    # Coarse-to-fine arguments
    parser.add_argument('--use_coarse_to_fine', action='store_true', default=False,
                        help='Use coarse-to-fine registration (SVD + residual rigid + VAE)')
    parser.add_argument('--corr_proj_dim', type=int, default=256,
                        help='Projection dim for soft correspondences (default: 256)')
    parser.add_argument('--corr_temperature', type=float, default=0.1,
                        help='Temperature for correspondence softmax (default: 0.1)')
    parser.add_argument('--lambda_rigid_chamfer', type=float, default=0.5,
                        help='Weight for Stage 1 rigid Chamfer loss (default: 0.5)')
    parser.add_argument('--lambda_cycle', type=float, default=0.0,
                        help='Weight for cycle consistency loss on correspondences (default: 0.0)')
    parser.add_argument('--lambda_intermediate_chamfer', type=float, default=0.0,
                        help='Weight for intermediate Chamfer loss at each SVD iteration (default: 0.0)')
    parser.add_argument('--n_svd_iterations', type=int, default=3,
                        help='Number of iterative SVD passes in Stage 1 of c2f (default: 3)')
    parser.add_argument('--n_stage2a_iterations', type=int, default=1,
                        help='Number of iterative Stage 2 refinement passes (default: 1)')
    parser.add_argument('--use_local_features', action='store_true', default=False,
                        help='Add k-NN local feature enrichment before correspondences')
    parser.add_argument('--local_k', type=int, default=20,
                        help='Number of nearest neighbors for local feature enrichment (default: 20)')
    parser.add_argument('--use_geo_consistency', action='store_true', default=False,
                        help='Enable geometric consistency reweighting for SVD correspondences')
    parser.add_argument('--geo_consistency_alpha', type=float, default=5.0,
                        help='Strictness of geometric consistency check (higher = stricter, default: 5.0)')
    parser.add_argument('--use_pso', action='store_true', default=False,
                        help='Enable NIA coarse pre-alignment before learned pipeline')
    parser.add_argument('--nia_type', type=str, default='pso',
                        choices=['pso', 'de', 'cmaes', 'firefly', 'gwo', 'pso_de', 'de_cmaes', 'pso_cmaes'],
                        help='NIA type for pre-alignment (default: pso)')
    parser.add_argument('--pso_particles', type=int, default=50,
                        help='Number of NIA particles/population (default: 50)')
    parser.add_argument('--pso_iterations', type=int, default=30,
                        help='Number of NIA iterations (default: 30)')
    parser.add_argument('--fitness_subsample', type=int, default=128,
                        help='Number of points subsampled for NIA fitness evaluation (default: 128)')
    parser.add_argument('--pso_translation_range', type=float, default=1.0,
                        help='Stage 0 CMA-ES translation search range (default: 1.0)')
    parser.add_argument('--use_inter_stage_nia', action='store_true', default=False,
                        help='Enable inter-stage NIA refinement after Stages 1, 2, and 3')
    parser.add_argument('--inter_stage_nia_particles', type=int, default=25,
                        help='Number of particles for inter-stage NIA (default: 25)')
    parser.add_argument('--inter_stage_nia_iterations', type=int, default=15,
                        help='Number of iterations for inter-stage NIA (default: 15)')
    parser.add_argument('--inter_stage_nia_rot_s1', type=float, default=20.0,
                        help='Rotation range (deg) for NIA after Stage 1 (default: 20.0)')
    parser.add_argument('--inter_stage_nia_trans_s1', type=float, default=0.5,
                        help='Translation range for NIA after Stage 1 (default: 0.5)')
    parser.add_argument('--inter_stage_nia_rot_s2', type=float, default=10.0,
                        help='Rotation range (deg) for NIA after Stage 2 (default: 10.0)')
    parser.add_argument('--inter_stage_nia_trans_s2', type=float, default=0.2,
                        help='Translation range for NIA after Stage 2 (default: 0.2)')
    parser.add_argument('--inter_stage_nia_rot_s3', type=float, default=5.0,
                        help='Rotation range (deg) for NIA after Stage 3 (default: 5.0)')
    parser.add_argument('--inter_stage_nia_trans_s3', type=float, default=0.1,
                        help='Translation range for NIA after Stage 3 (default: 0.1)')
    parser.add_argument('--diffusion_weight', type=float, default=1.0,
                        help='Weight for diffusion score-matching loss (default: 1.0)')
    parser.add_argument('--two_phase_diffusion', action='store_true', default=False,
                        help='Two-phase training: load pretrained c2f, freeze it, train only diffusion')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Path to pretrained c2f checkpoint for two-phase diffusion training')
    parser.add_argument('--use_rt_supervision', action='store_true', default=False,
                        help='Enable R/t supervision losses (geodesic rotation + translation L2) against ground truth')

    # Stage ablation flags (for ablation experiments)
    parser.add_argument('--no_stage0', action='store_true', default=False,
                        help='Ablation: disable Stage 0 (CMA-ES pre-alignment)')
    parser.add_argument('--no_stage1', action='store_true', default=False,
                        help='Ablation: disable Stage 1 (iterative SVD)')
    parser.add_argument('--no_stage2', action='store_true', default=False,
                        help='Ablation: disable Stage 2 (residual rigid head)')
    parser.add_argument('--no_stage3', action='store_true', default=False,
                        help='Ablation: disable Stage 3 (VAE non-rigid deformation)')

    # Control-point deformation (alternative Stage 3)
    parser.add_argument('--use_control_points', action='store_true', default=False,
                        help='Use control-point deformation for Stage 3 instead of VAE+Generator')
    parser.add_argument('--n_control_points', type=int, default=128,
                        help='Number of FPS control points for deformation (default: 128)')
    parser.add_argument('--rbf_sigma', type=float, default=0.2,
                        help='Gaussian RBF bandwidth for control point interpolation (default: 0.2)')

    # Non-rigid deformation arguments
    parser.add_argument('--non_rigid', action='store_true', default=False,
                        help='Apply non-rigid deformations to target during training')
    parser.add_argument('--faust_natural_pairs', action='store_true', default=False,
                        help='Use natural cross-pose pairs for FAUST training (instead of synthetic transforms)')
    parser.add_argument('--nonrigid_n_control', type=int, default=8,
                        help='Number of RBF control points for non-rigid deformation (default: 8)')
    parser.add_argument('--nonrigid_scale', type=float, default=0.05,
                        help='Magnitude of non-rigid displacements (default: 0.05)')
    parser.add_argument('--nonrigid_rbf_sigma', type=float, default=0.3,
                        help='RBF bandwidth for non-rigid deformation locality (default: 0.3)')

    # Generator type
    parser.add_argument('--generator_type', type=str, default='mlp',
                        choices=['mlp', 'attention', 'cross_attention'],
                        help='Generator variant: mlp (default), attention, or '
                             'cross_attention (VFA-style kNN cross-attention between source and target)')
    parser.add_argument('--ca_feature_dim', type=int, default=64,
                        help='Feature projection dim for cross_attention generator (default: 64)')
    parser.add_argument('--ca_k', type=int, default=8,
                        help='kNN neighbours for cross_attention generator (default: 8)')

    # Loss weight arguments
    parser.add_argument('--chamfer_weight', type=float, default=1.0,
                        help='Weight for Chamfer loss (default: 1.0)')
    parser.add_argument('--kl_weight', type=float, default=0.001,
                        help='Weight for KL divergence loss (default: 0.001)')

    args = parser.parse_args()

    # Seed all RNGs for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    #Validates arguments for ShapeNet dataset
    if args.dataset == 'shapenet':
        if args.dir is None:
            parser.error("--dir is required when --dataset is 'shapenet'")
    
    # Creates datasets
    print("Creating datasets...")
    
    if args.dataset == 'synthetic':
        train_dataset = SyntheticRegistrationDataset(
            n_samples=1000,
            n_points=args.n_points,
            noise_std=args.noise,
            outlier_ratio=0.05
        )
        
        val_dataset = SyntheticRegistrationDataset(
            n_samples=200,
            n_points=args.n_points,
            noise_std=args.noise,
            outlier_ratio=0.05
        )
    
    elif args.dataset == 'modelnet40':
        if args.train_data_dir is None:
            raise ValueError("--train_data_dir must be specified for ModelNet40 dataset")
        if args.val_data_dir is None:
            raise ValueError("--val_data_dir must be specified for ModelNet40 dataset")
        
        train_dataset = ModelNet40RegistrationDataset(
            data_dir=args.train_data_dir,
            n_samples=args.n_samples,
            n_points=args.n_points,
            noise_std=args.noise,
            split='train',
            normalize=args.normalization,
            rotation_range=args.rotation_range,
            translation_range=args.translation_range,
            non_rigid=args.non_rigid,
            nonrigid_n_control=args.nonrigid_n_control,
            nonrigid_scale=args.nonrigid_scale,
            nonrigid_rbf_sigma=args.nonrigid_rbf_sigma,
        )

        val_dataset = ModelNet40RegistrationDataset(
            data_dir=args.val_data_dir,
            n_samples=args.n_samples // 5,
            n_points=args.n_points,
            noise_std=args.noise,
            split='test',
            normalize=args.normalization,
            rotation_range=args.rotation_range,
            translation_range=args.translation_range,
            non_rigid=args.non_rigid,
            nonrigid_n_control=args.nonrigid_n_control,
            nonrigid_scale=args.nonrigid_scale,
            nonrigid_rbf_sigma=args.nonrigid_rbf_sigma,
        )
    elif args.dataset == 'faust':
        if args.train_data_dir is None:
            raise ValueError("--train_data_dir must be specified for faust dataset")
        if args.val_data_dir is None:
            args.val_data_dir = args.train_data_dir

        train_dataset = FaustDataset(
            data_dir=args.train_data_dir,
            n_samples=args.n_samples,
            n_points=args.n_points,
            noise_std=args.noise,
            split='train',
            normalize=args.normalization,
            rotation_range=args.rotation_range,
            translation_range=args.translation_range,
            non_rigid=args.non_rigid,
            nonrigid_n_control=args.nonrigid_n_control,
            nonrigid_scale=args.nonrigid_scale,
            nonrigid_rbf_sigma=args.nonrigid_rbf_sigma,
            use_natural_pairs=args.faust_natural_pairs,
        )

        val_dataset = FaustDataset(
            data_dir=args.val_data_dir,
            n_samples=args.n_samples // 5,
            n_points=args.n_points,
            noise_std=args.noise,
            split='test',
            normalize=args.normalization,
            rotation_range=args.rotation_range,
            translation_range=args.translation_range,
            non_rigid=args.non_rigid,
            nonrigid_n_control=args.nonrigid_n_control,
            nonrigid_scale=args.nonrigid_scale,
            nonrigid_rbf_sigma=args.nonrigid_rbf_sigma,
            use_natural_pairs=args.faust_natural_pairs,
        )

    elif args.dataset == 'SPARE':
        if args.data_dir is None:
            raise ValueError("--data_dir must be specified for SPARE dataset")
        
        train_dataset = SpareDataset(
            Spare_Dataset_Path=args.train_data_dir,
            Indices= 20,
            Train = True
        )
        
        val_dataset = SpareDataset(
            Spare_Dataset_Path=args.val_data_dir,
            Indices= 20,
            Train= False
        )
    elif args.dataset == '3dmatch':
        if args.train_data_dir is None:
            raise ValueError("--train_data_dir must be specified for 3DMatch dataset")
        
        if args.use_gt_pairs:
            # Use ground truth registration pairs
            print(f"Loading 3DMatch ground truth pairs from: {args.train_data_dir}")
            train_dataset = Match3DRegistrationDataset(
                data_dir=args.train_data_dir,
                n_points=args.n_points,
                noise_std=args.noise,
                max_pairs_per_scene=args.max_pairs_per_scene,
                train=True
            )
            
            val_data_dir = args.val_data_dir if args.val_data_dir else args.train_data_dir
            val_dataset = Match3DRegistrationDataset(
                data_dir=val_data_dir,
                n_points=args.n_points,
                noise_std=0.0,  # No noise for validation
                max_pairs_per_scene=args.max_pairs_per_scene,
                train=False
            )
        else:
            # Generate pairs on-the-fly from fragments
            print(f"Loading 3DMatch fragments for pair generation from: {args.train_data_dir}")
            train_dataset = Match3DPairDataset(
                data_dir=args.train_data_dir,
                n_samples=args.n_samples,
                n_points=args.n_points,
                noise_std=args.noise,
                rotation_range=args.rotation_range,
                translation_range=args.translation_range,
                normalize=args.normalization,
                non_rigid=getattr(args, 'non_rigid', False),
            )

            val_data_dir = args.val_data_dir if args.val_data_dir else args.train_data_dir
            val_dataset = Match3DPairDataset(
                data_dir=val_data_dir,
                n_samples=200,  # Smaller validation set
                n_points=args.n_points,
                noise_std=0.0,  # No noise for validation
                rotation_range=args.rotation_range,
                translation_range=args.translation_range,
                normalize=args.normalization,
                non_rigid=getattr(args, 'non_rigid', False),
            )
    elif args.dataset == 'shapenet':
        train_dataset = ShapeNetRegistrationDataset(
            data_dir=args.train_data_dir,
            n_samples=args.n_samples,
            n_points=args.n_points,
            noise_std=args.noise,
            rotation_range=args.rotation_range,
            translation_range=args.translation_range,
            normalize=args.normalization,
            non_rigid=getattr(args, 'non_rigid', False),
            nonrigid_n_control=getattr(args, 'nonrigid_n_control', 8),
            nonrigid_scale=getattr(args, 'nonrigid_scale', 0.05),
            nonrigid_rbf_sigma=getattr(args, 'nonrigid_rbf_sigma', 0.3),
            split='train',
            shapenet13=getattr(args, 'shapenet13', False),
            class_choice=getattr(args, 'class_choice', None),
        )

        val_data_dir = args.val_data_dir if args.val_data_dir else args.train_data_dir
        val_dataset = ShapeNetRegistrationDataset(
            data_dir=val_data_dir,
            n_samples=200,
            n_points=args.n_points,
            noise_std=0.0,
            rotation_range=args.rotation_range,
            translation_range=args.translation_range,
            normalize=args.normalization,
            non_rigid=getattr(args, 'non_rigid', False),
            split='test',
            shapenet13=getattr(args, 'shapenet13', False),
            class_choice=getattr(args, 'class_choice', None),
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Creates data loaders
    def _worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    # Creates model
    model_desc = []
    if args.use_geometric_attention:
        model_desc.append('geometric cross-attention')
    if args.with_diffusion:
        model_desc.append('diffusion')
    if args.use_rigid_head:
        model_desc.append('rigid head')
    desc_str = ' with ' + ', '.join(model_desc) if model_desc else ''
    print(f"Creating model{desc_str}...")
    model = create_evoreg(
        latent_dim=args.latent_dim,
        feature_dim=512,
        with_diffusion=args.with_diffusion,
        diffusion_n_steps=args.diffusion_steps if args.with_diffusion else None,
        use_rigid_head=args.use_rigid_head,
        use_geometric_attention=args.use_geometric_attention,
        geo_num_heads=args.geo_attn_heads,
        geo_num_blocks=args.geo_attn_blocks,
        geo_num_rbf=args.geo_num_rbf,
        geo_rbf_cutoff=args.geo_rbf_cutoff,
        use_coarse_to_fine=args.use_coarse_to_fine,
        corr_proj_dim=args.corr_proj_dim,
        corr_temperature=args.corr_temperature,
        n_svd_iterations=args.n_svd_iterations,
        n_stage2a_iterations=args.n_stage2a_iterations if args.use_coarse_to_fine else 1,
        use_local_features=args.use_local_features if args.use_coarse_to_fine else False,
        local_k=args.local_k,
        use_geo_consistency=args.use_geo_consistency if args.use_coarse_to_fine else False,
        geo_consistency_alpha=args.geo_consistency_alpha,
        use_pso=args.use_pso if args.use_coarse_to_fine else False,
        nia_type=args.nia_type,
        pso_particles=args.pso_particles,
        pso_iterations=args.pso_iterations,
        fitness_subsample=args.fitness_subsample,
        pso_translation_range=args.pso_translation_range,
        use_inter_stage_nia=args.use_inter_stage_nia if args.use_coarse_to_fine else False,
        inter_stage_nia_particles=args.inter_stage_nia_particles,
        inter_stage_nia_iterations=args.inter_stage_nia_iterations,
        inter_stage_nia_rot_s1=args.inter_stage_nia_rot_s1,
        inter_stage_nia_trans_s1=args.inter_stage_nia_trans_s1,
        inter_stage_nia_rot_s2=args.inter_stage_nia_rot_s2,
        inter_stage_nia_trans_s2=args.inter_stage_nia_trans_s2,
        inter_stage_nia_rot_s3=args.inter_stage_nia_rot_s3,
        inter_stage_nia_trans_s3=args.inter_stage_nia_trans_s3,
        no_stage0=args.no_stage0 if args.use_coarse_to_fine else False,
        no_stage1=args.no_stage1 if args.use_coarse_to_fine else False,
        no_stage2=args.no_stage2 if args.use_coarse_to_fine else False,
        no_stage3=args.no_stage3 if args.use_coarse_to_fine else False,
        use_control_points=args.use_control_points if args.use_coarse_to_fine else False,
        n_control_points=args.n_control_points,
        rbf_sigma=args.rbf_sigma,
        generator_type=args.generator_type,
        ca_feature_dim=args.ca_feature_dim,
        ca_k=args.ca_k,
    )

    # Two-phase diffusion: load pretrained c2f, freeze inner model, train only diffusion
    if args.two_phase_diffusion:
        if not args.with_diffusion:
            raise ValueError("--two_phase_diffusion requires --with_diffusion")
        if not args.pretrained_checkpoint:
            raise ValueError("--two_phase_diffusion requires --pretrained_checkpoint")

        pretrained_path = Path(args.pretrained_checkpoint)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

        print(f"Two-phase diffusion: loading pretrained c2f from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

        # Load only the inner model (vae_model) weights, skip score_network/diffusion
        inner_sd = {k.replace('vae_model.', ''): v for k, v in sd.items()
                    if k.startswith('vae_model.') or (not k.startswith('score_network.') and not k.startswith('diffusion.'))}
        # If checkpoint was saved without diffusion wrapper, keys won't have vae_model. prefix
        if not any(k.startswith('vae_model.') for k in sd.keys()):
            inner_sd = {k: v for k, v in sd.items()
                        if not k.startswith('score_network.') and not k.startswith('diffusion.')}

        missing, unexpected = model.vae_model.load_state_dict(inner_sd, strict=False)
        if missing:
            print(f"  Warning: missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Warning: unexpected keys: {unexpected[:5]}...")
        print(f"  Loaded {len(inner_sd)} weights into inner c2f model")

        # Freeze all inner model parameters
        frozen_count = 0
        for param in model.vae_model.parameters():
            param.requires_grad = False
            frozen_count += 1
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  Froze {frozen_count} inner model params, {trainable} trainable (score network + diffusion)")

    # Creates trainer
    trainer = EvoRegTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        chamfer_weight=args.chamfer_weight,
        kl_weight=args.kl_weight,
        diffusion_weight=args.diffusion_weight,
        with_diffusion=args.with_diffusion,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_weighted_loss=args.use_weighted_loss,
        use_lr_schedule=args.use_lr_schedule,
        lr_schedule_step=args.lr_schedule_step,
        lr_schedule_gamma=args.lr_schedule_gamma,
        use_occlusion=args.use_occlusion,
        occlusion_type=args.occlusion_type,
        occlusion_ratio=args.occlusion_ratio,
        occlusion_prob=args.occlusion_prob,
        occlusion_noise_std=args.occlusion_noise_std,
        use_rigid_head=args.use_rigid_head,
        use_rt_supervision=args.use_rt_supervision,
        lambda_rot=args.lambda_rot,
        lambda_trans=args.lambda_trans,
        lambda_rmse=args.lambda_rmse,
        lambda_align=args.lambda_align,
        lambda_disp=args.lambda_disp,
        lambda_deform=args.lambda_deform,
        lambda_lap=args.lambda_lap,
        use_correspondence_loss=args.use_correspondence_loss,
        lambda_correspondence=args.lambda_correspondence,
        lambda_rigid_chamfer=args.lambda_rigid_chamfer if args.use_coarse_to_fine else 0.0,
        lambda_cycle=args.lambda_cycle if args.use_coarse_to_fine else 0.0,
        lambda_intermediate_chamfer=args.lambda_intermediate_chamfer if args.use_coarse_to_fine else 0.0,
        lambda_centroid=args.lambda_centroid if args.use_coarse_to_fine else 0.0,
        lambda_centroid_s1=args.lambda_centroid_s1 if args.use_coarse_to_fine else None,
        lambda_centroid_s2=args.lambda_centroid_s2 if args.use_coarse_to_fine else None,
        lambda_centroid_s3=args.lambda_centroid_s3 if args.use_coarse_to_fine else None,
    )

    # Store model config for checkpoint metadata
    trainer.model_config = {
        'n_svd_iterations': args.n_svd_iterations,
        'n_stage2a_iterations': args.n_stage2a_iterations if args.use_coarse_to_fine else 1,
        'use_coarse_to_fine': args.use_coarse_to_fine,
        'two_phase_diffusion': args.two_phase_diffusion,
        'use_geo_consistency': args.use_geo_consistency if args.use_coarse_to_fine else False,
        'geo_consistency_alpha': args.geo_consistency_alpha,
        'use_pso': args.use_pso if args.use_coarse_to_fine else False,
        'nia_type': args.nia_type,
        'pso_particles': args.pso_particles,
        'pso_iterations': args.pso_iterations,
        'fitness_subsample': args.fitness_subsample,
        'pso_translation_range': args.pso_translation_range,
        'use_inter_stage_nia': args.use_inter_stage_nia if args.use_coarse_to_fine else False,
        'inter_stage_nia_particles': args.inter_stage_nia_particles,
        'inter_stage_nia_iterations': args.inter_stage_nia_iterations,
        'inter_stage_nia_rot_s1': args.inter_stage_nia_rot_s1,
        'inter_stage_nia_trans_s1': args.inter_stage_nia_trans_s1,
        'inter_stage_nia_rot_s2': args.inter_stage_nia_rot_s2,
        'inter_stage_nia_trans_s2': args.inter_stage_nia_trans_s2,
        'inter_stage_nia_rot_s3': args.inter_stage_nia_rot_s3,
        'inter_stage_nia_trans_s3': args.inter_stage_nia_trans_s3,
        'no_stage0': args.no_stage0,
        'no_stage1': args.no_stage1,
        'no_stage2': args.no_stage2,
        'no_stage3': args.no_stage3,
        'use_control_points': args.use_control_points if args.use_coarse_to_fine else False,
        'n_control_points': args.n_control_points,
        'rbf_sigma': args.rbf_sigma,
    }

    # Loads checkpoint if resuming
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Prepare config dictionary (store all CLI arguments)
    config_dict = vars(args)

    # Experiment metadata
    start_time = datetime.now().isoformat()
    hostname = socket.gethostname()

    # Try to obtain git commit hash (if repository exists)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    def save_training_history():
        end_time = datetime.now().isoformat()

        history = {
            'config': config_dict,
            'git_commit': git_commit,
            'hostname': hostname,
            'start_time': start_time,
            'end_time': end_time,
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_metrics': trainer.train_metrics,
            'val_metrics': trainer.val_metrics,
        }
        history_path = Path(args.checkpoint_dir) / 'training_history.json'
        try:
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            print(f"Training history saved to {history_path}")
        except Exception as e:
            print(f"Failed to save training history: {e}")

    # Run training with Ctrl+C safety
    try:
        trainer.train(n_epochs=args.epochs)
        print("Training complete!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving progress...")
    finally:
        save_training_history()


if __name__ == "__main__":
    main()