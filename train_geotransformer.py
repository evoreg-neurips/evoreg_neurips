"""
Training script for GeoTransformer baseline.

Reuses our data infrastructure (ModelNet40RegistrationDataset) but uses
GeoTransformer's architecture and losses. Follows the paper's training recipe:
Adam, lr=1e-4, exponential decay, 200 epochs, batch_size=1.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent))

from baselines.geotransformer_baseline import GeoTransformerBaseline
from baselines.geotransformer_losses import GeoTransformerLoss
from evoreg.data.modelnet40_dataset import ModelNet40RegistrationDataset
from evoreg.losses import ChamferDistance


def compute_patch_overlaps(
    src_coarse_pts: torch.Tensor,
    tgt_coarse_pts: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    gt_R: torch.Tensor,
    gt_t: torch.Tensor,
    patch_radius: float = 0.15,
) -> torch.Tensor:
    """Compute ground-truth overlap ratios between source and target superpoints.

    Transforms source points by GT R,t then checks how many points in each
    source patch overlap with each target patch.

    Args:
        src_coarse_pts: (B, N_s, 3) source superpoint positions
        tgt_coarse_pts: (B, N_t, 3) target superpoint positions
        source: (B, N, 3) full source cloud
        target: (B, M, 3) full target cloud
        gt_R: (B, 3, 3) ground truth rotation
        gt_t: (B, 3) ground truth translation
        patch_radius: radius for patch membership

    Returns:
        overlaps: (B, N_s, N_t) overlap ratios in [0, 1]
    """
    B = source.shape[0]
    N_s = src_coarse_pts.shape[1]
    N_t = tgt_coarse_pts.shape[1]
    device = source.device

    # Transform source superpoints by GT
    src_transformed = torch.bmm(src_coarse_pts, gt_R.transpose(1, 2)) + gt_t.unsqueeze(1)

    # Overlap = fraction of transformed source patch points that fall
    # within a target patch. Simplified: use superpoint distance as proxy.
    # If distance between transformed src superpoint and tgt superpoint < 2*radius,
    # they overlap proportionally.
    dists = torch.cdist(src_transformed, tgt_coarse_pts, p=2.0)  # (B, N_s, N_t)
    max_dist = 2.0 * patch_radius
    overlaps = (1.0 - dists / max_dist).clamp(min=0.0)

    return overlaps


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: GeoTransformerLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    patch_radius: float = 0.15,
) -> Dict[str, float]:
    model.train()
    total_losses = {'total': 0, 'circle': 0, 'point': 0, 'rot': 0, 'trans': 0}
    n_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        gt_R = batch['rotation'].to(device)
        gt_t = batch['translation'].to(device)

        optimizer.zero_grad()

        output = model(source, target)

        # Compute GT overlaps for circle loss
        overlaps = compute_patch_overlaps(
            output['src_coarse_points'], output['tgt_coarse_points'],
            source, target, gt_R, gt_t, patch_radius,
        )

        losses = criterion(output, gt_R, gt_t, overlaps=overlaps)
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'rot': f"{losses['rot'].item():.4f}",
            'trans': f"{losses['trans'].item():.4f}",
        })

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: GeoTransformerLoss,
    device: str,
    patch_radius: float = 0.15,
) -> Dict[str, float]:
    model.eval()
    total_losses = {'total': 0, 'circle': 0, 'point': 0, 'rot': 0, 'trans': 0}
    chamfer_fn = ChamferDistance(reduction='mean')
    total_chamfer = 0.0
    total_rot_err = 0.0
    total_trans_err = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc='Validating'):
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        gt_R = batch['rotation'].to(device)
        gt_t = batch['translation'].to(device)

        output = model(source, target)

        overlaps = compute_patch_overlaps(
            output['src_coarse_points'], output['tgt_coarse_points'],
            source, target, gt_R, gt_t, patch_radius,
        )
        losses = criterion(output, gt_R, gt_t, overlaps=overlaps)

        for k in total_losses:
            total_losses[k] += losses[k].item()

        # Chamfer distance
        total_chamfer += chamfer_fn(output['transformed_source'], target).item()

        # Rotation error (degrees)
        R_diff = torch.bmm(output['est_R'], gt_R.transpose(1, 2))
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = ((trace - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)
        rot_err_deg = torch.acos(cos_angle) * 180.0 / np.pi
        total_rot_err += rot_err_deg.mean().item()

        # Translation error
        total_trans_err += (output['est_t'] - gt_t).norm(dim=-1).mean().item()

        n_batches += 1

    n = max(n_batches, 1)
    metrics = {k: v / n for k, v in total_losses.items()}
    metrics['chamfer'] = total_chamfer / n
    metrics['rot_err_deg'] = total_rot_err / n
    metrics['trans_err'] = total_trans_err / n
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train GeoTransformer baseline')
    parser.add_argument('--dataset', type=str, default='modelnet40')
    parser.add_argument('--data_dir', type=str, default='data/ModelNet40')
    parser.add_argument('--val_data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Learning rate (default: 4e-4, sqrt-scaled from paper 1e-4 for batch_size=16)')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='Exponential LR decay factor per epoch (default: 0.95, matching paper)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for Adam optimizer (default: 1e-6, matching paper)')
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--n_samples', type=int, default=9843)
    parser.add_argument('--rotation_range', type=float, default=45.0)
    parser.add_argument('--translation_range', type=float, default=1.0)
    parser.add_argument('--normalization', type=str, default='UnitBall')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None)
    # Loss weights
    parser.add_argument('--lambda_circle', type=float, default=1.0)
    parser.add_argument('--lambda_point', type=float, default=1.0)
    parser.add_argument('--lambda_rot', type=float, default=1.0)
    parser.add_argument('--lambda_trans', type=float, default=1.0)
    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_transformer_blocks', type=int, default=3)
    parser.add_argument('--num_correspondences', type=int, default=128)
    parser.add_argument('--num_sinkhorn_iters', type=int, default=100)
    parser.add_argument('--acceptance_radius', type=float, default=0.1)
    parser.add_argument('--patch_radius', type=float, default=0.15)
    parser.add_argument('--max_points_per_patch', type=int, default=128)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g. experiments/.../last_model.pth)')

    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Datasets
    val_data_dir = args.val_data_dir or args.data_dir
    train_dataset = ModelNet40RegistrationDataset(
        data_dir=args.data_dir,
        n_points=args.n_points,
        n_samples=args.n_samples,
        split='train',
        rotation_range=args.rotation_range,
        translation_range=args.translation_range,
        normalize=args.normalization,
    )
    val_dataset = ModelNet40RegistrationDataset(
        data_dir=val_data_dir,
        n_points=args.n_points,
        n_samples=args.n_samples // 5,
        split='test',
        rotation_range=args.rotation_range,
        translation_range=args.translation_range,
        normalize=args.normalization,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Model
    model = GeoTransformerBaseline(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_transformer_blocks,
        num_correspondences=args.num_correspondences,
        num_sinkhorn_iters=args.num_sinkhorn_iters,
        acceptance_radius=args.acceptance_radius,
        patch_radius=args.patch_radius,
        max_points_per_patch=args.max_points_per_patch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GeoTransformer parameters: {n_params:,}")

    # Loss
    criterion = GeoTransformerLoss(
        lambda_circle=args.lambda_circle,
        lambda_point=args.lambda_point,
        lambda_rot=args.lambda_rot,
        lambda_trans=args.lambda_trans,
    )

    # Optimizer + scheduler (matching paper: Adam, weight_decay=1e-6, ExponentialLR gamma=0.95)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # Save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = args.save_dir or f'experiments/geotransformer_{args.epochs}ep_{timestamp}'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['n_params'] = n_params
    config['device'] = device
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    history = []

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')))
            history = ckpt.get('history', [])
            print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"WARNING: Resume checkpoint not found: {resume_path}, training from scratch")

    for epoch in range(start_epoch, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} | LR: {lr:.2e}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            patch_radius=args.patch_radius,
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            patch_radius=args.patch_radius,
        )
        scheduler.step()

        print(f"  Train loss: {train_metrics['total']:.4f}")
        print(f"  Val loss: {val_metrics['total']:.4f} | "
              f"Chamfer: {val_metrics['chamfer']:.4f} | "
              f"RotErr: {val_metrics['rot_err_deg']:.2f}deg | "
              f"TransErr: {val_metrics['trans_err']:.4f}")

        epoch_record = {
            'epoch': epoch, 'lr': lr,
            'train': train_metrics, 'val': val_metrics,
        }
        history.append(epoch_record)

        # Checkpoint
        is_best = val_metrics['total'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total']
            print(f"  -> New best val loss: {best_val_loss:.4f}")

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_metrics['total'],
            'best_val_loss': best_val_loss,
            'history': history,
            'config': config,
        }
        torch.save(ckpt, save_dir / 'last_model.pth')
        if is_best:
            torch.save(ckpt, save_dir / 'best_model.pth')
        if epoch % 50 == 0:
            torch.save(ckpt, save_dir / f'model_epoch_{epoch}.pth')

        # Save history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
