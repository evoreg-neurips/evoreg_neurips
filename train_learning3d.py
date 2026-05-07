import sys, os
import sys
import os
import argparse 

# directory containing train_learning3d.py
CURRENT = os.path.dirname(os.path.abspath(__file__))

# go up one directory → /home/.../anon-user/
ROOT = os.path.dirname(CURRENT)

# add parent directory to PYTHONPATH so "learning3d" is found
sys.path.append(ROOT)

# optional: add learning3d folder directly
sys.path.append(os.path.join(ROOT, "learning3d"))
from evoreg.data.faust_dataset import FaustDataset
from evoreg.data.modelnet40_dataset import ModelNet40RegistrationDataset
from learning3d.models import PointNet, PointNetLK, DCP, iPCRNet, PRNet, PPFNet, RPMNet
import torch 
import torch.nn as nn 
import torch.optim as optim
from evoreg.losses import ChamferDistance
from torch.utils.data import DataLoader, Dataset
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

class OpenSourceRegistrationTrainer:
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
        swap_input_order: bool = False
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
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.with_diffusion = with_diffusion
        self.use_weighted_loss = use_weighted_loss
        self.swap_input_order = swap_input_order

        # Creates checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Sets up optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Sets up loss functions (if not using integrated model)
        if not with_diffusion:
            self.chamfer_loss = ChamferDistance(reduction='mean')
            self.chamfer_weight = chamfer_weight
            self.kl_weight = kl_weight
            self.diffusion_weight = diffusion_weight
        self.kl_annealing = kl_annealing
        
        # Tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

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
        source: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes training losses.

        Args:
            outputs: Model outputs dictionary
            target: Target point cloud
            step: Current training step
            source: Source point cloud (needed for weighted loss)

        Returns:
            Dictionary of losses
        """
        # If using diffusion model, losses are already computed
        if self.with_diffusion:
            # Model already computed all losses
            if self.kl_annealing and step is not None:
                # Update KL weight in model if using annealing
                kl_weight = min(self.model.kl_weight, self.model.kl_weight * step / 10000)
                # Recompute total loss with annealed KL weight
                total_loss = (
                    self.model.chamfer_weight * outputs['chamfer_loss'] +
                    kl_weight * outputs['kl_loss'] +
                    self.model.diffusion_weight * outputs.get('diffusion_loss', 0)
                )
                outputs['loss'] = total_loss
                outputs['kl_weight'] = kl_weight
            return outputs

        # VAE-only model: compute losses manually
        # Compute point weights if using weighted loss
        weights = None
        if self.use_weighted_loss and source is not None:
            weights = self.compute_point_weights(source, target)

        # Computes Chamfer distance (with optional weights)
        chamfer = self.chamfer_loss(outputs['transformed_source'], target, weights=weights)
        
        
        self.chamfer_weight = 1
        # Computes total loss
        total_loss = self.chamfer_weight * chamfer
        
        return {
            'loss': total_loss,
            'chamfer': chamfer
            
            
        }
    
    def train_epoch(self) -> float:
        """
        Trains for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        epoch_losses = []
        
        # Creates progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Moves data to device
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            # iPCRNet/PointNetLK: forward(template, source) — template=target
            # PRNet: forward(src, tgt) — src=source
            if self.with_diffusion:
                outputs = self.model(source, target, return_components=True)
            elif self.swap_input_order:
                outputs = self.model(target, source)
            else:
                outputs = self.model(source, target)

            # Computes losses
            losses = self.compute_loss(outputs, target, self.global_step, source=source)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Updates tracking
            epoch_losses.append(losses['loss'].item())
            self.global_step += 1
            
            # Updates progress bar
            if batch_idx % self.log_interval == 0:
                postfix = {
                    'loss': f"{losses['loss'].item():.4f}"
                }
                
                if self.with_diffusion:
                    postfix.update({
                        'chamfer': f"{losses.get('chamfer_loss', losses.get('chamfer', 0)).item():.4f}",
                        
                        'diff': f"{losses.get('diffusion_loss', 0).item():.4f}"
                    })
                else:
                    postfix.update({
                        'chamfer': f"{losses['chamfer'].item():.4f}"
                        
                    })
                
                pbar.set_postfix(postfix)
        
        return np.mean(epoch_losses)
    
    def validate(self) -> float:
        """
        Validates the model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Moves data to device
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)

                # Forward pass (same ordering as training)
                if self.with_diffusion:
                    outputs = self.model(source, target, return_components=False)
                elif self.swap_input_order:
                    outputs = self.model(target, source)
                else:
                    outputs = self.model(source, target)

                # Computes losses
                losses = self.compute_loss(outputs, target, source=source)
                val_losses.append(losses['loss'].item())
        
        return np.mean(val_losses)
    
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
            'best_val_loss': self.best_val_loss
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
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Saves checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
            
            # Regular checkpoint
            if epoch % 1000 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # Saves final model
        self.save_checkpoint('final_model.pth')
        
        # Saves training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history, f)
        
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Learning3d Registration models')
    parser.add_argument('--model', type=str, default='pnlk', choices=['pnlk', 'pcrnet', 'rpmnet', 'prnet'])
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['faust', 'modelnet40'])
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_points', type=int, default=1024, help='Points per cloud')
    parser.add_argument('--n_samples', type=int, default=1000, help='Samples per epoch')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Noise standard deviation')
    parser.add_argument('--train_data_dir', type=str, required=True,
                        help='Directory containing training data files')
    parser.add_argument('--val_data_dir', type=str, default=None,
                        help='Directory containing validation data files')
    parser.add_argument('--normalization', type=str, default='UnitBall',
                        choices=['UnitBall', 'BoundingBox', 'Identity'])
    parser.add_argument('--rotation_range', type=float, default=None,
                        help='Constrain rotations to ±N degrees (default: None = full SO(3))')
    parser.add_argument('--translation_range', type=float, default=0.2,
                        help='Translation range ±N (default: 0.2)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    args = parser.parse_args()

    if args.val_data_dir is None:
        args.val_data_dir = args.train_data_dir

    if args.dataset == 'modelnet40':
        train_dataset = ModelNet40RegistrationDataset(
            data_dir=args.train_data_dir,
            n_samples=args.n_samples,
            n_points=args.n_points,
            noise_std=args.noise_std,
            split='train',
            normalize=args.normalization,
            rotation_range=args.rotation_range,
            translation_range=args.translation_range
        )
        val_dataset = ModelNet40RegistrationDataset(
            data_dir=args.val_data_dir,
            n_samples=args.n_samples // 5,
            n_points=args.n_points,
            noise_std=args.noise_std,
            split='test',
            normalize=args.normalization,
            rotation_range=args.rotation_range,
            translation_range=args.translation_range
        )
    elif args.dataset == 'faust':
        train_dataset = FaustDataset(FAUST_Dataset_Path=args.train_data_dir)
        val_dataset = FaustDataset(FAUST_Dataset_Path=args.val_data_dir, Train=False)

    # Creates data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )


    # iPCRNet and PointNetLK use forward(template, source) convention
    # PRNet uses forward(src, tgt) convention
    swap_input = False

    if args.model == 'pnlk':
        model = PointNetLK(feature_model=PointNet(), delta=1e-02, xtol=1e-07, p0_zero_mean=True, p1_zero_mean=True, pooling='max')
        swap_input = True
    elif args.model == 'dcp':
        model = DCP(feature_model=PointNet(), pointer_='transformer', head='svd')
    elif args.model == 'pcrnet':
        model = iPCRNet(feature_model=PointNet(), pooling='max')
        swap_input = True
    elif args.model == 'rpmnet':
        model = RPMNet(feature_model=PPFNet())
    elif args.model == 'prnet':
        model = PRNet()

    # Creates trainer
    trainer = OpenSourceRegistrationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        with_diffusion=False,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_weighted_loss=False,
        swap_input_order=swap_input
    )

    # Loads checkpoint if resuming
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Trains model
    trainer.train(n_epochs=args.epochs)

if __name__ == "__main__":
    main()