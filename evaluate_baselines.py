"""
Unified evaluation script for point cloud registration.

Evaluates EvoReg and a broad set of rigid and non-rigid baselines (CPD, BCPD,
NDP, DefTransNet, FLOT, ICP, FGR, RANSAC, GeoTransformer, PRNet, DCP, RPMNet,
PointNetLK, DeepGMR, iPCRNet) under a unified protocol on ModelNet40,
ShapeNet, FAUST, and 3DMatch.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
import time
import subprocess
import socket
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from evoreg.models import create_evoreg
from evoreg.models.evoreg_model import EvoReg
from safetensors.torch import load_file
# Adds EvoReg to path
sys.path.append(str(Path(__file__).parent))

from evoreg.data.synthetic_data import generate_registration_pair
from evoreg.data.shapenet_dataset import ShapeNetRegistrationDataset
from evoreg.data.modelnet40_dataset import ModelNet40RegistrationDataset
from evoreg.data.faust_dataset import FAUSTRegistrationDataset
from evoreg.data.spare_dataset import SpareDataset
from evoreg.data.match3d_dataset import Match3DPairDataset, Match3DRegistrationDataset
from evoreg.losses import ChamferDistance
from evoreg.evaluation import (
    registration_error,
    RegistrationError,
    earth_movers_distance,
    approximate_emd_sinkhorn,
    sliced_wasserstein_distance,
    Point_to_Point_Error,
    RegistrationRecall,
    Correspondence_Error,
    Geodesic_Distance,
    rotation_error,
    RotationError,
    translation_error,
    TranslationError
)
from baselines.cpd_wrapper import CPDWrapper
from baselines.bcpd_wrapper import BCPDWrapper
from baselines.ndp_wrapper import NDPWrapper
from baselines.icp_wrapper import ICPWrapper
from baselines.fgr_wrapper import FGRWrapper
from baselines.ransac_wrapper import RANSACWrapper
from baselines.pretrained_models.exp_atlasnet.AtlasNet.model.atlasnet import Atlasnet
from baselines.pretrained_models.exp_atlasnet.AtlasNet.auxiliary.argument_parser import parser as atlasnet_parser
import baselines.pretrained_models.exp_atlasnet.AtlasNet.auxiliary.my_utils as my_utils
from easydict import EasyDict

def sinkhorn_tto_refine(x_hat, target, steps=15, lr=0.01, lambda_reg=10.0,
                         sinkhorn_eps=0.01, sinkhorn_iters=50):
    """Test-time optimization: refine point cloud via Sinkhorn EMD minimization.

    Optimizes per-point displacements epsilon to minimize:
        L_sink^EMD(x_hat + epsilon, target) + lambda * ||epsilon||^2
    """
    with torch.enable_grad():
        x_hat_detached = x_hat.detach()
        epsilon = torch.zeros_like(x_hat_detached, requires_grad=True)
        optimizer = torch.optim.Adam([epsilon], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = (approximate_emd_sinkhorn(x_hat_detached + epsilon, target.detach(),
                        epsilon=sinkhorn_eps, max_iter=sinkhorn_iters, reduction='mean')
                    + lambda_reg * (epsilon ** 2).mean())
            loss.backward()
            optimizer.step()
    return (x_hat_detached + epsilon).detach()


def _axis_angle_to_matrix(axis_angle):
    """Convert axis-angle (B, 3) to rotation matrix (B, 3, 3) via Rodrigues."""
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
    axis = axis_angle / theta  # (B, 3)
    K = torch.zeros(axis.shape[0], 3, 3, device=axis.device, dtype=axis.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    theta = theta.unsqueeze(-1)  # (B, 1, 1)
    I = torch.eye(3, device=axis.device).unsqueeze(0)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def sinkhorn_tto_rigid_refine(source, target, R_init, t_init,
                               steps=15, lr=0.01, lambda_reg=10.0,
                               sinkhorn_eps=0.01, sinkhorn_iters=50):
    """Test-time optimization: refine R, t via Sinkhorn EMD minimization.

    Optimizes rotation and translation corrections to minimize:
        L_sink^EMD(R_new @ source + t_new, target) + lambda * (||delta_rot||^2 + ||delta_t||^2)

    where R_new = delta_R @ R_init, t_new = t_init + delta_t.

    Args:
        source: Original source point cloud (B, N, 3)
        target: Target point cloud (B, M, 3)
        R_init: Initial rotation from model (B, 3, 3)
        t_init: Initial translation from model (B, 3)
        steps: Number of gradient descent steps
        lr: Learning rate
        lambda_reg: L2 regularization on corrections
        sinkhorn_eps: Sinkhorn entropic regularization
        sinkhorn_iters: Sinkhorn iterations per step

    Returns:
        transformed: Refined point cloud (B, N, 3)
        R_final: Refined rotation (B, 3, 3)
        t_final: Refined translation (B, 3)
    """
    with torch.enable_grad():
        source_d = source.detach()
        target_d = target.detach()
        R_init_d = R_init.detach()
        t_init_d = t_init.detach()

        delta_rot = torch.zeros(source_d.shape[0], 3, device=source_d.device,
                                dtype=source_d.dtype, requires_grad=True)
        delta_t = torch.zeros(source_d.shape[0], 3, device=source_d.device,
                              dtype=source_d.dtype, requires_grad=True)

        optimizer = torch.optim.Adam([delta_rot, delta_t], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            R_delta = _axis_angle_to_matrix(delta_rot)
            R_new = R_delta @ R_init_d
            t_new = t_init_d + delta_t
            transformed = torch.bmm(source_d, R_new.transpose(1, 2)) + t_new.unsqueeze(1)
            loss = (approximate_emd_sinkhorn(transformed, target_d,
                        epsilon=sinkhorn_eps, max_iter=sinkhorn_iters, reduction='mean')
                    + lambda_reg * ((delta_rot ** 2).mean() + (delta_t ** 2).mean()))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            R_delta = _axis_angle_to_matrix(delta_rot)
            R_final = R_delta @ R_init_d
            t_final = t_init_d + delta_t
            transformed_final = torch.bmm(source_d, R_final.transpose(1, 2)) + t_final.unsqueeze(1)

    return transformed_final.detach(), R_final.detach(), t_final.detach()


def translation_only_icp_refine(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    steps: int = 10,
    distance_weight_tau: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refine a registered point cloud using translation-only ICP.

    Keeps the current rotation fixed and iteratively updates only a global
    translation by snapping each source point to its nearest target point.
    Translation updates are weighted by nearest-neighbor distance so closer
    correspondences contribute more.

    Returns:
        refined_points: Translation-refined point cloud
        total_delta_t: Accumulated translation update (B, 3)
    """
    refined = x_hat.detach().clone()
    total_delta = torch.zeros(refined.shape[0], 3, device=refined.device, dtype=refined.dtype)

    for _ in range(steps):
        dists = torch.cdist(refined, target)
        nn_idx = dists.argmin(dim=-1)  # (B, N)
        nn_dists = torch.gather(dists, 2, nn_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
        matched_target = torch.gather(
            target,
            1,
            nn_idx.unsqueeze(-1).expand(-1, -1, target.shape[-1]),
        )
        residuals = matched_target - refined  # (B, N, 3)
        tau = max(float(distance_weight_tau), 1e-8)
        weights = torch.exp(-nn_dists / tau)  # (B, N)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        delta_t = (residuals * weights.unsqueeze(-1)).sum(dim=1)  # (B, 3)
        refined = refined + delta_t.unsqueeze(1)
        total_delta = total_delta + delta_t

    return refined, total_delta


MODEL_NET_40 = "ModelNet40"
FAUST = "FAUST"
ATLASNET = "AtlasNet"

# Imports learning3d models (only working ones)
try:
    from learning3d.models import PointNetLK, iPCRNet, PRNet, DCP, RPMNet, PPFNet, DeepGMR, PointNet
    from learning3d.models.deepgmr import PointNet as DeepGMRPointNet
    LEARNING3D_AVAILABLE = True
except ImportError:
    print("Warning: learning3d not available. Install with: pip install learning3d")
    LEARNING3D_AVAILABLE = False
if LEARNING3D_AVAILABLE == False:
    # directory containing train_learning3d.py
    CURRENT = os.path.dirname(os.path.abspath(__file__))

    # go up one directory → /home/.../anon-user/
    ROOT = os.path.dirname(CURRENT)

    # add parent directory to PYTHONPATH so "learning3d" is found
    sys.path.append(ROOT)

    # optional: add learning3d folder directly
    sys.path.append(os.path.join(ROOT, "learning3d"))
    from learning3d.models import PointNetLK, iPCRNet, PRNet, DCP, RPMNet, PPFNet, DeepGMR, PointNet
    from learning3d.models.deepgmr import PointNet as DeepGMRPointNet
    LEARNING3D_AVAILABLE = True

# Imports Open3D for data loading
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: open3d not available. Sample data evaluation will be disabled")
    OPEN3D_AVAILABLE = False

def compute_rri_features(pts_np, k=20):
    """Compute Rotation and Reflection Invariant (RRI) features for DeepGMR.

    Follows the RRI computation from learning3d (data_utils/dataloaders.py::get_rri).
    For each point, computes 4 features per neighbor (rp, rq, theta, phi) using
    k nearest neighbors, yielding k*4 features per point.

    Args:
        pts_np: (N, 3) numpy array of point coordinates
        k: number of nearest neighbors (must match model's nearest_neighbors param)

    Returns:
        (N, 3 + k*4) numpy array with XYZ concatenated with RRI features
    """
    from scipy.spatial import cKDTree
    kdt = cKDTree(pts_np)
    _, idx = kdt.query(pts_np, k=k+1)
    idx = idx[:, 1:]  # exclude self
    q = pts_np[idx]  # (N, K, 3) neighbor coords
    p = np.repeat(pts_np[:, None], k, axis=1)  # (N, K, 3) repeated query points
    rp = np.linalg.norm(p, axis=-1, keepdims=True)
    rq = np.linalg.norm(q, axis=-1, keepdims=True)
    pn = p / (rp + 1e-8)
    qn = q / (rq + 1e-8)
    dot = np.sum(pn * qn, -1, keepdims=True)
    theta = np.arccos(np.clip(dot, -1, 1))
    T_q = q - dot * p
    sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
    cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
    psi = np.arctan2(sin_psi, cos_psi) % (2 * np.pi)
    idx2 = np.argpartition(psi, 1)[:, :, 1:2]
    phi = np.take_along_axis(psi, idx2, axis=-1)
    feat = np.concatenate([rp, rq, theta, phi], axis=-1).reshape(-1, k * 4)
    return np.concatenate([pts_np, feat], axis=-1)


def _run_command(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run a command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    output = result.stdout.strip()
    return output or None


def get_git_commit(repo_dir: Optional[Path] = None) -> str:
    """Collect git commit hash for reproducibility."""
    repo_dir = repo_dir or Path(__file__).resolve().parent
    git_commit = _run_command(['git', 'rev-parse', 'HEAD'], cwd=repo_dir)
    return git_commit or 'unknown'


def get_gpu_metadata() -> Dict[str, object]:
    """Collect accelerator metadata without failing on unavailable backends."""
    gpu_info: Dict[str, object] = {
        'device_requested': None,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_devices': [],
        'mps_available': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()),
    }

    if torch.cuda.is_available():
        cuda_devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_memory_gb = round(props.total_memory / (1024 ** 3), 2)
            cuda_devices.append({
                'index': index,
                'name': torch.cuda.get_device_name(index),
                'total_memory_gb': total_memory_gb,
                'capability': f'{props.major}.{props.minor}',
            })
        gpu_info['cuda_devices'] = cuda_devices

    return gpu_info


# Imports trimesh for mesh-based metrics
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("Warning: trimesh not available. Mesh-based metrics will be disabled")
    TRIMESH_AVAILABLE = False


class BaselineEvaluator:
    """
    Evaluates pretrained baseline models on registration tasks.

    Loads pretrained models from learning3d and evaluates them on
    synthetic or FAUST or ModelNet40 datasets using standardized metrics.
    """

    def __init__(
        self,
        device: str = 'cpu',
        pretrained_dir: str = 'baselines/pretrained_models',
        results_dir: str = 'baselines/results',
        recall_threshold: float = 0.1
    ):
        """
        Initializes the baseline evaluator.

        Args:
            device: Device to run evaluation on ('cpu', 'cuda', or 'mps')
            pretrained_dir: Directory containing pretrained model weights
            results_dir: Directory to save evaluation results
            recall_threshold: RMSE threshold for registration recall metric
        """
        self.device = device
        self.pretrained_dir = Path(pretrained_dir)
        self.results_dir = Path(results_dir)
        self.recall_threshold = recall_threshold

        # Creates directories if they don't exist
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initializes metrics
        self.chamfer_metric = ChamferDistance(reduction='mean')
        self.registration_metric = RegistrationError(reduction='mean', squared=False)
        self.registration_recall = RegistrationRecall(
            threshold=recall_threshold,
            k=1,
            reduction='mean'
        )

        # Stores available models
        self.models = {}
        self.results = {}

    def get_pretrained_path(self, model_name: str) -> Optional[Path]:
        """
        Gets the default pretrained model path for a given model.

        Args:
            model_name: Name of the model

        Returns:
            Path to pretrained weights or None if not found
        """
        # Maps model names to their pretrained checkpoint paths (only working models)
        pretrained_mapping = {
            'evoreg': 'checkpoints/best_model.pth',
            'pointnetlk': 'exp_pnlk/models/best_model.t7',
            'ipcrnet': 'exp_ipcrnet/models/best_model.t7',
            'prnet': 'exp_prnet/models/best_model.t7',
            'dcp': 'exp_dcp/models/best_model.t7',
            'rpmnet': 'exp_rpmnet/models/best_model.pth',
            'deepgmr': 'exp_deepgmr/models/best_model.pth',
            'atlasnet': 'exp_atlasnet/models/atlasnet_autoencoder_25_squares/network.pth',
            'geotransformer': 'exp_geotransformer/best_model.pth',
            'deftransnet': 'exp_deftransnet/best_model_DefTransNet.pth',
            'flot': 'exp_flot/best_model_FLOT.pt',
        }

        if model_name.lower() in pretrained_mapping:
            pretrained_path = self.pretrained_dir / pretrained_mapping[model_name.lower()]
            if pretrained_path.exists():
                return pretrained_path

        return None

    def load_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        use_pretrained: bool = True
    ) -> Optional[nn.Module]:
        """
        Loads a pretrained baseline model.

        Args:
            model_name: Name of the model ('pointnetlk', 'ipcrnet', 'prnet')
            checkpoint_path: Path to pretrained weights (optional, overrides auto-detection)
            use_pretrained: Whether to automatically load pretrained weights

        Returns:
            Loaded model or None if loading fails
        """
        #if not LEARNING3D_AVAILABLE:
        #    print(f"Cannot load {model_name}: learning3d not available")
        #    return None

        try:
            # Creates model based on name (only working models)
            if model_name.lower() == 'pointnetlk':
                model = PointNetLK()
            elif model_name.lower() == 'ipcrnet':
                model = iPCRNet()
            elif model_name.lower() == 'prnet':
                model = PRNet()
                # Note: PRNet has MPS compatibility issues. Use CPU or CUDA devices.
                if self.device == 'mps':
                    print("WARNING: PRNet has known issues with MPS device. Consider using --device cpu or cuda.")
            elif model_name.lower() == 'dcp':
                from learning3d.models.dgcnn import DGCNN
                model = DCP(feature_model=DGCNN(emb_dims=512))
            elif model_name.lower() == 'rpmnet':
                model = RPMNet(feature_model=PPFNet())
            elif model_name.lower() == 'deepgmr':
                # learning3d's DeepGMR PointNet references a module-level `args`
                # for d_model and n_clusters. We inject these to match the
                # pretrained checkpoint (encoder out=1024, clusters=16).
                # Also, DeepGMR.__init__ has a bug (`feature_model if not None`
                # always evaluates to feature_model), so we pass it explicitly.
                import types, learning3d.models.deepgmr as _dgmr_mod
                _dgmr_mod.args = types.SimpleNamespace(d_model=1024, n_clusters=16)
                model = DeepGMR(feature_model=DeepGMRPointNet(use_rri=True, nearest_neighbors=20))
            elif model_name.lower() == 'atlasnet':
                self.opt = atlasnet_parser()
                self.opt.device = self.device
                self.opt.batch_size = 1
                self.opt.batch_size_test = 1
                model = Atlasnet(opt=self.opt)
            elif model_name.lower() in ('cpd_rigid', 'cpd_nonrigid'):
                mode = 'rigid' if model_name.lower() == 'cpd_rigid' else 'deformable'
                model = CPDWrapper(mode=mode)
            elif model_name.lower() == 'bcpd':
                model = BCPDWrapper()
            elif model_name.lower() == 'ndp':
                model = NDPWrapper(device=self.device)
            elif model_name.lower() == 'icp':
                model = ICPWrapper()
            elif model_name.lower() == 'fgr':
                model = FGRWrapper()
            elif model_name.lower() == 'ransac':
                model = RANSACWrapper()
            elif model_name.lower() == 'deftransnet':
                from baselines.deftransnet_wrapper import DefTransNetWrapper
                model = DefTransNetWrapper(k=10, k1=128, slbp_iter=3, slbp_cost_scale=10.0, slbp_alpha=-50.0)
            elif model_name.lower() == 'flot':
                from baselines.flot_wrapper import FLOTWrapper
                model = FLOTWrapper(nb_iter=10)
            elif model_name.lower() in ('evoreg', 'evoreg_rigid'):
                # Auto-detect rigid head from checkpoint if available
                use_rigid = model_name.lower() == 'evoreg_rigid'
                if not use_rigid and checkpoint_path is not None:
                    cp = Path(checkpoint_path)
                    if cp.exists():
                        try:
                            ckpt = torch.load(cp, map_location='cpu', weights_only=False)
                            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
                            if any(k.startswith('rigid_head.') for k in sd.keys()):
                                use_rigid = True
                                print("Auto-detected rigid head in checkpoint")
                        except Exception:
                            pass
                model = EvoReg(use_rigid_head=use_rigid)
            elif model_name.lower() in ('evoreg_geo', 'evoreg_geo_rigid'):
                from evoreg.models.evoreg_model import EvoRegGeometric
                use_rigid = model_name.lower() == 'evoreg_geo_rigid'
                model = EvoRegGeometric(use_rigid_head=use_rigid)
            elif model_name.lower() == 'evoreg_c2f':
                from evoreg.models.coarse_to_fine import EvoRegCoarseToFine
                from evoreg.models.evoreg_model import EvoRegWithDiffusion
                # Auto-detect config from checkpoint
                n_svd_iters = 3
                n_stage2a_iters = 1
                has_diffusion = False
                has_local_features = False
                has_geo_consistency = False
                geo_alpha = 5.0
                has_pso = False
                nia_type = 'pso'
                pso_particles = 50
                pso_iterations = 30
                has_inter_stage_nia = False
                inter_stage_nia_particles = 25
                inter_stage_nia_iterations = 15
                inter_stage_nia_rot_s1 = 20.0
                inter_stage_nia_trans_s1 = 0.5
                inter_stage_nia_rot_s2 = 10.0
                inter_stage_nia_trans_s2 = 0.2
                inter_stage_nia_rot_s3 = 5.0
                inter_stage_nia_trans_s3 = 0.1
                has_control_points = False
                n_control_points = 128
                rbf_sigma = 0.2
                if checkpoint_path is not None:
                    cp = Path(checkpoint_path)
                    if cp.exists():
                        try:
                            ckpt = torch.load(cp, map_location='cpu', weights_only=False)
                            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
                            has_diffusion = any(k.startswith('score_network.') or k.startswith('diffusion.') for k in sd.keys())
                            if has_diffusion:
                                print("Auto-detected diffusion weights in c2f checkpoint")
                            has_local_features = any('local_enrichment.' in k for k in sd.keys())
                            if has_local_features:
                                print("Auto-detected local feature enrichment in c2f checkpoint")
                            # Detect control-point head from state dict keys
                            if any('control_point_head.' in k for k in sd.keys()):
                                has_control_points = True
                                print("Auto-detected control-point deformation head in c2f checkpoint")
                            # Read model config if stored
                            model_cfg = ckpt.get('model_config', {})
                            if model_cfg:
                                n_svd_iters = model_cfg.get('n_svd_iterations', 3)
                                n_stage2a_iters = model_cfg.get('n_stage2a_iterations', 1)
                                if n_stage2a_iters > 1:
                                    print(f"Auto-detected iterative Stage 2a: {n_stage2a_iters} iterations")
                                has_geo_consistency = model_cfg.get('use_geo_consistency', False)
                                geo_alpha = model_cfg.get('geo_consistency_alpha', 5.0)
                                if has_geo_consistency:
                                    print(f"Auto-detected geometric consistency reweighting (alpha={geo_alpha})")
                                has_pso = model_cfg.get('use_pso', False)
                                nia_type = model_cfg.get('nia_type', 'pso')
                                pso_particles = model_cfg.get('pso_particles', 50)
                                pso_iterations = model_cfg.get('pso_iterations', 30)
                                if has_pso:
                                    print(f"Auto-detected NIA pre-alignment: {nia_type} ({pso_particles} particles, {pso_iterations} iters)")
                                has_inter_stage_nia = model_cfg.get('use_inter_stage_nia', False)
                                inter_stage_nia_particles = model_cfg.get('inter_stage_nia_particles', 25)
                                inter_stage_nia_iterations = model_cfg.get('inter_stage_nia_iterations', 15)
                                inter_stage_nia_rot_s1 = model_cfg.get('inter_stage_nia_rot_s1', 20.0)
                                inter_stage_nia_trans_s1 = model_cfg.get('inter_stage_nia_trans_s1', 0.5)
                                inter_stage_nia_rot_s2 = model_cfg.get('inter_stage_nia_rot_s2', 10.0)
                                inter_stage_nia_trans_s2 = model_cfg.get('inter_stage_nia_trans_s2', 0.2)
                                inter_stage_nia_rot_s3 = model_cfg.get('inter_stage_nia_rot_s3', 5.0)
                                inter_stage_nia_trans_s3 = model_cfg.get('inter_stage_nia_trans_s3', 0.1)
                                if has_inter_stage_nia:
                                    print(f"Auto-detected inter-stage NIA: S1={inter_stage_nia_rot_s1}°/{inter_stage_nia_trans_s1}, S2={inter_stage_nia_rot_s2}°/{inter_stage_nia_trans_s2}, S3={inter_stage_nia_rot_s3}°/{inter_stage_nia_trans_s3}")
                                has_control_points = model_cfg.get('use_control_points', False)
                                n_control_points = model_cfg.get('n_control_points', 128)
                                rbf_sigma = model_cfg.get('rbf_sigma', 0.2)
                                if has_control_points:
                                    print(f"Auto-detected control-point deformation (K={n_control_points}, sigma={rbf_sigma})")
                        except Exception:
                            pass
                base_model = EvoRegCoarseToFine(
                    n_svd_iterations=n_svd_iters,
                    n_stage2a_iterations=n_stage2a_iters,
                    use_local_features=has_local_features,
                    use_geo_consistency=has_geo_consistency,
                    geo_consistency_alpha=geo_alpha,
                    use_pso=has_pso,
                    nia_type=nia_type,
                    pso_particles=pso_particles,
                    pso_iterations=pso_iterations,
                    use_inter_stage_nia=has_inter_stage_nia,
                    inter_stage_nia_particles=inter_stage_nia_particles,
                    inter_stage_nia_iterations=inter_stage_nia_iterations,
                    inter_stage_nia_rot_s1=inter_stage_nia_rot_s1,
                    inter_stage_nia_trans_s1=inter_stage_nia_trans_s1,
                    inter_stage_nia_rot_s2=inter_stage_nia_rot_s2,
                    inter_stage_nia_trans_s2=inter_stage_nia_trans_s2,
                    inter_stage_nia_rot_s3=inter_stage_nia_rot_s3,
                    inter_stage_nia_trans_s3=inter_stage_nia_trans_s3,
                )
                if has_diffusion:
                    model = EvoRegWithDiffusion(vae_model=base_model)
                else:
                    model = base_model
            elif model_name.lower() == 'geotransformer':
                from baselines.geotransformer_baseline import GeoTransformerBaseline
                # Auto-detect config from checkpoint
                gt_kwargs = {}
                if checkpoint_path is not None:
                    cp = Path(checkpoint_path)
                    if cp.exists():
                        try:
                            ckpt = torch.load(cp, map_location='cpu', weights_only=False)
                            cfg = ckpt.get('config', {})
                            if cfg:
                                for k in ('hidden_dim', 'num_heads', 'num_transformer_blocks',
                                          'num_correspondences', 'num_sinkhorn_iters',
                                          'acceptance_radius', 'patch_radius'):
                                    if k in cfg:
                                        gt_kwargs[k] = cfg[k]
                                if gt_kwargs:
                                    print(f"Auto-detected GeoTransformer config: {gt_kwargs}")
                        except Exception:
                            pass
                model = GeoTransformerBaseline(**gt_kwargs)
            else:
                print(f"Unknown or unsupported model: {model_name}")
                print(f"Supported models: pointnetlk, ipcrnet, prnet, geotransformer")
                return None

            # Determines checkpoint path
            # Only apply --model_checkpoint to EvoReg variants; baselines use their own pretrained weights
            # CPD is optimization-based (no learned weights) — skip weight loading entirely
            is_evoreg = model_name.lower().startswith('evoreg')
            is_geotransformer = model_name.lower() == 'geotransformer'
            is_deftransnet = model_name.lower() == 'deftransnet'
            is_flot = model_name.lower() == 'flot'
            is_cpd = model_name.lower().startswith('cpd') or model_name.lower() in ('bcpd', 'ndp', 'icp', 'fgr', 'ransac')
            if is_cpd:
                checkpoint_path = None
            elif checkpoint_path is not None and (is_evoreg or is_geotransformer or is_deftransnet or is_flot):
                checkpoint_path = Path(checkpoint_path)
            elif use_pretrained:
                checkpoint_path = self.get_pretrained_path(model_name)
            else:
                checkpoint_path = None

            # Loads pretrained weights if available
            if checkpoint_path is not None and checkpoint_path.exists():
                print(f"Loading pretrained weights from {checkpoint_path}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)

                    # Handles different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    print(f"[OK] Successfully loaded pretrained weights")
                except Exception as e:
                    print(f"Warning: Could not load weights: {str(e)}")
                    print("Attemtping to use safetensors")
                    try:
                        # Attempt to load as safetensors
                        safetensors_path = checkpoint_path
                        state_dict = load_file(safetensors_path, device=self.device)
                        model.load_state_dict(state_dict, strict=False)
                        print(f"Model loaded with safetensors!")
                    except Exception as e2:
                        print(f"Warning: Could not load weights: {str(e2)}")
                        print("Using randomly initialized weights")
            else:
                if is_cpd:
                    print(f"{model_name} is optimization-based (no learned weights)")
                else:
                    if use_pretrained:
                        print(f"Warning: No pretrained weights found for {model_name}")
                    print(f"Using randomly initialized weights")

            # Moves model to device
            model = model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            import traceback
            print(f"Error loading {model_name}: {str(e)}")
            traceback.print_exc()
            return None

    # Models where forward(template, source) transforms the 2nd arg to match the 1st
    TEMPLATE_SOURCE_MODELS = {'ipcrnet', 'pointnetlk', 'dcp', 'rpmnet', 'deepgmr'}

    @staticmethod
    def _estimate_rigid_transform_from_points(
        source: torch.Tensor,
        registered: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the best-fit rigid transform from source to registered points.

        Uses a batched Kabsch solve assuming source and registered have the same
        point ordering. This is intended for evaluation-time R/t reporting and
        does not modify the registered point cloud used by geometric metrics.
        """
        source_mean = source.mean(dim=1, keepdim=True)
        registered_mean = registered.mean(dim=1, keepdim=True)

        source_centered = source - source_mean
        registered_centered = registered - registered_mean
        H = torch.bmm(source_centered.transpose(1, 2), registered_centered)

        U, _, Vt = torch.linalg.svd(H)
        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

        det = torch.det(R)
        neg_mask = det < 0
        if neg_mask.any():
            Vt_fixed = Vt.clone()
            Vt_fixed[neg_mask, -1, :] *= -1
            R = torch.bmm(Vt_fixed.transpose(1, 2), U.transpose(1, 2))

        t = registered_mean.squeeze(1) - torch.bmm(
            source_mean, R.transpose(1, 2)
        ).squeeze(1)
        return R, t

    def evaluate_on_pair(
        self,
        model: nn.Module,
        source: torch.Tensor,
        target: torch.Tensor,
        ground_truth_transform: Optional[np.ndarray] = None,
        keep_idxs: torch.Tensor = None,
        model_name: str = None,
    ) -> Dict[str, float]:
        """
        Evaluates a model on a single registration pair.

        Args:
            model: Baseline model to evaluate
            source: Source point cloud (N, 3) - already normalized by dataset
            target: Target point cloud (M, 3) - already normalized by dataset
            ground_truth_transform: Ground truth transformation (4, 4)
            keep_idxs: Indices to keep for evaluation (for occlusion handling)
            model_name: Name of the model (used to determine input ordering)

        Returns:
            Dictionary of metric values
        """
        # Convert to tensors if needed and prepare inputs
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()

        source = source.unsqueeze(0).to(self.device) if source.dim() == 2 else source.to(self.device)  # (1, N, 3)
        target = target.unsqueeze(0).to(self.device) if target.dim() == 2 else target.to(self.device)  # (1, M, 3)

        # Timing: synchronize GPU before starting measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        try:
            # DeepGMR requires RRI (Rotation-Reflection Invariant) features
            # appended to each point cloud before the forward pass
            if model_name and model_name.lower() == 'deepgmr':
                src_rri = compute_rri_features(source.squeeze(0).cpu().numpy())
                tgt_rri = compute_rri_features(target.squeeze(0).cpu().numpy())
                source_rri = torch.from_numpy(src_rri).float().unsqueeze(0).to(self.device)
                target_rri = torch.from_numpy(tgt_rri).float().unsqueeze(0).to(self.device)

            # Runs model inference on normalized point clouds
            # iPCRNet and PointNetLK use forward(template, source) convention
            # where template=target, source=source_to_transform
            # NDP requires gradients for test-time optimization — run outside no_grad
            is_optim_baseline = model_name and model_name.lower() == 'ndp'
            if is_optim_baseline:
                result = model(source, target)
            with torch.no_grad():
                if not is_optim_baseline:
                    if model_name and model_name.lower() == 'deepgmr':
                        result = model(target_rri, source_rri)
                    elif model_name and model_name.lower() in self.TEMPLATE_SOURCE_MODELS:
                        result = model(target, source)
                    else:
                        result = model(source, target)

                # Optional: NIA refinement after Stage 3 at inference
                if (getattr(self, 'use_eval_nia_s3', False)
                        and isinstance(result, dict) and 'output' in result):
                    inner = model.vae_model if hasattr(model, 'vae_model') else model
                    if hasattr(inner, 'inter_nia_s3'):
                        _, _, refined = inner.inter_nia_s3(result['output'], target)
                        result['output'] = refined
                        result['transformed_source'] = refined

                # Optional: diffusion refinement at inference (experimental)
                if (getattr(self, 'use_diffusion_refinement', False)
                        and hasattr(model, 'diffusion')
                        and isinstance(result, dict) and 'output' in result):
                    result['output'] = model.diffusion.refine(
                        result['output'], target,
                        num_steps=getattr(self, 'diffusion_refine_steps', 50),
                        noise_level=getattr(self, 'diffusion_noise_level', 0.01),
                    )
                    result['transformed_source'] = result['output']

                # Optional: Sinkhorn rigid TTO (refine R, t)
                if (getattr(self, 'use_sinkhorn_tto_rigid', False)
                        and isinstance(result, dict) and 'output' in result):
                    # Get initial R, t from model output
                    R_init = result.get('est_R')
                    if R_init is None:
                        R_init = result.get('R_pred')
                    t_init = result.get('est_t')
                    if t_init is None:
                        t_init = result.get('t_pred')
                    if R_init is not None and t_init is not None:
                        if t_init.dim() == 3 and t_init.shape[1] == 1:
                            t_init = t_init.squeeze(1)
                        refined_pts, R_new, t_new = sinkhorn_tto_rigid_refine(
                            source, target, R_init, t_init,
                            steps=getattr(self, 'sinkhorn_tto_rigid_steps', 15),
                            lr=getattr(self, 'sinkhorn_tto_rigid_lr', 0.01),
                            lambda_reg=getattr(self, 'sinkhorn_tto_rigid_lambda_', 10.0),
                            sinkhorn_eps=getattr(self, 'sinkhorn_tto_rigid_sinkhorn_eps', 0.01),
                            sinkhorn_iters=getattr(self, 'sinkhorn_tto_rigid_sinkhorn_iters', 50),
                        )
                        result['output'] = refined_pts
                        result['transformed_source'] = refined_pts
                        if 'est_R' in result:
                            result['est_R'] = R_new
                        if 'R_pred' in result:
                            result['R_pred'] = R_new
                        if 'est_t' in result:
                            result['est_t'] = t_new
                        if 't_pred' in result:
                            result['t_pred'] = t_new

                # Optional: Sinkhorn per-point TTO (refine point positions)
                if (getattr(self, 'use_sinkhorn_tto', False)
                        and isinstance(result, dict) and 'output' in result):
                    result['output'] = sinkhorn_tto_refine(
                        result['output'], target,
                        steps=getattr(self, 'sinkhorn_tto_steps', 15),
                        lr=getattr(self, 'sinkhorn_tto_lr', 0.01),
                        lambda_reg=getattr(self, 'sinkhorn_tto_lambda_', 10.0),
                        sinkhorn_eps=getattr(self, 'sinkhorn_tto_sinkhorn_eps', 0.01),
                        sinkhorn_iters=getattr(self, 'sinkhorn_tto_sinkhorn_iters', 50),
                    )
                    result['transformed_source'] = result['output']

                # Optional: translation-only ICP refinement
                if (getattr(self, 'use_translation_only_icp', False)
                        and isinstance(result, dict) and 'output' in result):
                    refined_output, delta_t = translation_only_icp_refine(
                        result['output'],
                        target,
                        steps=getattr(self, 'translation_only_icp_steps', 10),
                        distance_weight_tau=getattr(self, 'translation_only_icp_tau', 0.01),
                    )
                    result['output'] = refined_output
                    result['transformed_source'] = refined_output
                    if 't_pred' in result and result['t_pred'] is not None:
                        result['t_pred'] = result['t_pred'] + delta_t

            # Timing: synchronize GPU after all inference (including NIA/diffusion refinement)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - t_start

            #If using occlusion for registrations target.shape != source.shape need to evaluate on kept indices
            if keep_idxs is not None:
                source = source.squeeze(0)[keep_idxs]
                source = source.unsqueeze(0)
            # Extracts registered point cloud (X_hat)
            if isinstance(result, dict):
                registered = result.get('transformed_source', source)
            elif isinstance(result, tuple):
                registered = result[0]
            else:
                registered = result
            
            # Ensure registered has correct shape for metrics
            if registered.dim() == 2:
                registered = registered.unsqueeze(0)  # (1, N, 3)

            # Optional: summarize the final registered cloud via a best-fit rigid transform.
            if (
                getattr(self, 'use_kabsch', False)
                and isinstance(result, dict)
                and model_name
                and model_name.lower().startswith('evoreg')
                and source.shape[1] == registered.shape[1]
            ):
                est_R, est_t = self._estimate_rigid_transform_from_points(source, registered)
                result['est_R'] = est_R
                result['est_t'] = est_t

            # Computes basic metrics on normalized point clouds
            chamfer = self.chamfer_metric(registered, target).item()
            reg_error = self.registration_metric(registered, target).item()

            # Computes Earth Mover's Distance (using Sinkhorn approximation for speed)
            try:
                emd = approximate_emd_sinkhorn(registered, target, epsilon=0.01).item()
            except Exception as e:
                print(f"Warning: EMD computation failed: {str(e)}")
                emd = float('inf')

            # Computes Sliced Wasserstein Distance
            try:
                swd = sliced_wasserstein_distance(registered, target, n_projections=100).item()
            except Exception as e:
                print(f"Warning: SWD computation failed: {str(e)}")
                swd = float('inf')

            # Computes Point-to-Point Error (assumes 1-to-1 correspondence)
            try:
                p2p = Point_to_Point_Error(registered, target, reduction='mean').item()
            except Exception as e:
                # If point counts don't match, use nearest neighbor
                p2p = float('nan')

            # Computes Registration Recall
            try:
                recall = self.registration_recall(registered, target).item()
            except Exception as e:
                print(f"Warning: Registration Recall computation failed: {str(e)}")
                recall = float('nan')

            # Computes Rotation and Translation Errors (if ground truth is available)
            rot_err = float('nan')
            trans_err = float('nan')
            if ground_truth_transform is not None:
                try:
                    if ground_truth_transform.ndim == 3:
                        ground_truth_transform = ground_truth_transform.squeeze(0)
                    # Extracts ground truth rotation and translation
                    # Note: transformation matrix may include scale, so we normalize to get pure rotation
                    gt_R_raw = torch.from_numpy(ground_truth_transform[:3, :3]).float().unsqueeze(0).to(self.device)  # (1, 3, 3)
                    gt_t = torch.from_numpy(ground_truth_transform[:3, 3]).float().unsqueeze(0).to(self.device)  # (1, 3)
                    
                    # Normalize rotation matrix to handle scaling (use SVD to extract pure rotation)
                    # This handles cases where scale is applied to the rotation matrix
                    U_gt, S_gt, Vt_gt = torch.linalg.svd(gt_R_raw)
                    gt_R = torch.bmm(U_gt, Vt_gt)  # Pure rotation matrix (1, 3, 3)

                    # Extracts predicted rotation and translation
                    pred_R, pred_t = None, None
                    
                    # First, try to get R and t directly from model output
                    if isinstance(result, dict):
                        pred_R = result.get('est_R')  # learning3d models (PRNet, DCP, etc.)
                        if pred_R is None:
                            pred_R = result.get('R_pred')  # EvoReg c2f model
                        pred_t = result.get('est_t')
                        if pred_t is None:
                            pred_t = result.get('t_pred')

                        # Handles different translation shapes
                        if pred_t is not None and pred_t.dim() == 3 and pred_t.shape[1] == 1:
                            pred_t = pred_t.squeeze(1)  # (1, 1, 3) -> (1, 3)
                    
                    # If model doesn't provide R and t, estimate them from point clouds using SVD (Kabsch algorithm)
                    if pred_R is None or pred_t is None:
                        # Assumes source and registered have same number of points and are in correspondence
                        if source.shape[1] == registered.shape[1]:  # Same number of points
                            # Center the point clouds
                            source_centered = source - source.mean(dim=1, keepdim=True)  # (1, N, 3)
                            registered_centered = registered - registered.mean(dim=1, keepdim=True)  # (1, N, 3)
                            
                            # Compute covariance matrix
                            H = torch.bmm(source_centered.transpose(1, 2), registered_centered)  # (1, 3, 3)
                            
                            # SVD
                            U, S, Vt = torch.linalg.svd(H)
                            
                            # Compute rotation matrix
                            R_svd = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))  # (1, 3, 3)
                            
                            # Ensure proper rotation (det(R) = 1)
                            det = torch.det(R_svd)
                            if det < 0:
                                # Flip the last column of Vt
                                Vt[:, -1, :] *= -1
                                R_svd = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
                            
                            # Compute translation
                            source_mean = source.mean(dim=1)  # (1, 3)
                            registered_mean = registered.mean(dim=1)  # (1, 3)
                            t_svd = registered_mean - torch.bmm(source_mean.unsqueeze(1), R_svd.transpose(1, 2)).squeeze(1)  # (1, 3)
                            
                            if pred_R is None:
                                pred_R = R_svd
                            if pred_t is None:
                                pred_t = t_svd
                    
                    # Computes errors if we have both R and t
                    if pred_R is not None and pred_t is not None:
                        rot_err = rotation_error(pred_R, gt_R).item()
                        trans_err = translation_error(pred_t, gt_t).item()
                except Exception as e:
                    print(f"Warning: Rotation/Translation error computation failed: {str(e)}")

            metrics = {
                'chamfer_distance': chamfer,
                'registration_error': reg_error,
                'earth_movers_distance': emd,
                'sliced_wasserstein_distance': swd,
                'point_to_point_error': p2p,
                'registration_recall': recall,
                'rotation_error': rot_err,
                'translation_error': trans_err,
                'inference_time': inference_time,
                'success': True
            }

            return metrics

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - t_start
            print(f"Error during evaluation: {str(e)}")
            return {
                'chamfer_distance': float('inf'),
                'registration_error': float('inf'),
                'earth_movers_distance': float('inf'),
                'sliced_wasserstein_distance': float('inf'),
                'point_to_point_error': float('inf'),
                'registration_recall': float('nan'),
                'rotation_error': float('nan'),
                'translation_error': float('nan'),
                'inference_time': inference_time,
                'success': False,
                'error': str(e)
            }

    def find_source_path(self, data_path) -> str:
        return data_path / 'test_scan_000.ply'
        
    def find_target_path(self, data_path, index) -> str:
        return data_path / f'tr_reg_{index:03d}.ply'
        
    def find_gt_path(self, data_path, index) -> str:
        return data_path / f'tr_gt_{index:03d}.txt'
    
    def read_mesh_file(self, file_path) -> np.ndarray:
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        points = np.asarray(mesh.vertices).astype(np.float32)
        return points
    
    def read_off_file(self, file_path) -> np.ndarray:
        """
        Read OFF file format (handles both mesh and point cloud).
        
        Args:
            file_path: Path to the OFF file
            
        Returns:
            Numpy array of vertices (N, 3)
        """
        with open(file_path, 'r') as f:
            # Read header
            header = f.readline().strip()
            if header != 'OFF':
                raise ValueError(f"Invalid OFF file: {file_path}")
            
            # Read counts
            counts = f.readline().strip().split()
            n_vertices, n_faces = int(counts[0]), int(counts[1])
            
            # Read vertices
            vertices = []
            for i in range(n_vertices):
                line = f.readline().strip().split()
                vertices.append([float(x) for x in line[:3]])  # Only take x, y, z
            
            return np.array(vertices, dtype=np.float32)

    def load_sample_data(
        self,
        dir: str,
        load_meshes: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """
        Loads sample data for evaluation.

        Args:
            dir: Directory containing sample data files
            load_meshes: Whether to load full mesh data (for mesh-based metrics)

        Returns:
            List of dictionaries containing source, target pairs
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Cannot load sample data.")
            return []

        data_path = Path(dir)
        if not data_path.exists():
            print(f"data directory not found: {dir}")
            return []

        # Loads meshes
        pairs = []

        try:
            # Loads test scan (source)
            source_path = self.find_source_path(data_path)
            source_points = self.read_mesh_file(source_path)

            # Loads source mesh if requested
            source_trimesh = None
            if load_meshes and TRIMESH_AVAILABLE:
                source_trimesh = trimesh.load_mesh(str(source_path))

            # Loads registration targets
            for i in range(2):
                target_path = self.find_target_path(data_path, i)
                if target_path.exists():
                    # Check if it's an OFF file (which may be point cloud only)
                    target_points = self.read_mesh_file(target_path)

                    # Loads target mesh if requested
                    target_trimesh = None
                    if load_meshes and TRIMESH_AVAILABLE:
                        target_trimesh = trimesh.load_mesh(str(target_path))

                    # Loads ground truth correspondence if available
                    gt_path = self.find_gt_path(data_path, i)
                    gt_correspondence = None
                    if gt_path and gt_path.exists():
                        gt_correspondence = np.loadtxt(str(gt_path), dtype=np.int32)

                    pairs.append({
                        'source': source_points,
                        'target': target_points,
                        'source_mesh': source_trimesh,
                        'target_mesh': target_trimesh,
                        'correspondence': gt_correspondence,
                        'index': i
                    })

            print(f"Loaded {len(pairs)} sample pairs")
            print(f"Source shape: {source_points.shape}")
            if len(pairs) > 0:
                print(f"  Target shape: {pairs[0]['target'].shape}")
            if load_meshes and TRIMESH_AVAILABLE:
                print(f"  Mesh data loaded for mesh-based metrics")

        except Exception as e:
            print(f"Error loading sample data: {str(e)}")
            return []

        return pairs

    def load_sample_shapenet_data(
        self,
        data_dir: str,
        n_samples: int = 100,
        n_points: int = 1024,
        noise_std: float = 0.01,
        normalize: str = 'UnitBall',
        rotation_range: float = None,
        translation_range: float = 0.2,
        non_rigid: bool = False,
        shapenet13: bool = False,
        class_choice: list = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Loads ShapeNet data using ShapeNetRegistrationDataset.
        Mirrors load_sample_modelnet40_data() / load_sample_faust_data() exactly.
        """
        try:
            dataset = ShapeNetRegistrationDataset(
                data_dir=data_dir,
                n_samples=n_samples,
                n_points=n_points,
                noise_std=noise_std,
                normalize=normalize,
                rotation_range=rotation_range,
                translation_range=translation_range,
                non_rigid=non_rigid,
                split='test',
                shapenet13=shapenet13,
                class_choice=class_choice,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            pbar = tqdm(loader, desc=f'ShapeNet Registration dataset loading ...')

            pairs = []

            for idx, batch in enumerate(pbar):
                source_points = batch['source'].squeeze(0).cpu().numpy()
                target_points = batch['target'].squeeze(0).cpu().numpy()
                transformation = batch['transformation'].squeeze(0).cpu().numpy()

                pairs.append({
                    'source': source_points,
                    'target': target_points,
                    'source_mesh': None,
                    'target_mesh': None,
                    'correspondence': transformation,
                    'index': idx
                })
        except Exception as e:
            print(f"Error loading ShapeNet data: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

        return pairs
    
    def load_sample_modelnet40_data(
        self,
        data_dir: str,
        n_samples: int = 100,
        n_points: int = 1024,
        noise_std: float = 0.01,
        save_files: bool = False,
        normalize: str = 'UnitBall',
        rotation_range: float = None,
        translation_range: float = 0.2,
        non_rigid: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """
        Loads ModelNet40 data using the ModelNet40RegistrationDataset.

        Args:
            data_dir: Directory containing .off files
            n_samples: Number of registration pairs to generate
            n_points: Number of points per cloud
            noise_std: Standard deviation of noise to add
            save_files: Whether to save generated pairs to disk
            normalize: Normalization method ('UnitBall', 'BoundingBox', 'Identity')
            rotation_range: If provided, constrains rotations to ±N degrees
            translation_range: Half-width of translation box
            non_rigid: If True, apply non-rigid deformations to eval pairs

        Returns:
            List of dictionaries containing source, target pairs
        """
        try:
            dataset = ModelNet40RegistrationDataset(
                data_dir=data_dir,
                n_samples=n_samples,
                n_points=n_points,
                noise_std=noise_std,
                save_files=save_files,
                split='test',
                normalize=normalize,
                rotation_range=rotation_range,
                translation_range=translation_range,
                non_rigid=non_rigid,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            pbar = tqdm(loader, desc=f'ModelNet40 Registration dataset loading ...')
            
            pairs = []

            for idx, batch in enumerate(pbar):
                # Extract data from batch
                source_points = batch['source'].squeeze(0).cpu().numpy()  # (N, 3)
                target_points = batch['target'].squeeze(0).cpu().numpy()  # (N, 3)
                transformation = batch['transformation'].squeeze(0).cpu().numpy()  # (4, 4)
                
                pairs.append({
                    'source': source_points,
                    'target': target_points,
                    'source_mesh': None,
                    'target_mesh': None,
                    'correspondence': transformation,
                    'index': idx
                })
        except Exception as e:
            print(f"Error loading ModelNet40 data: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

        return pairs

    def load_sample_faust_data(
        self,
        data_dir: str,
        n_samples: int = 100,
        n_points: int = 1024,
        noise_std: float = 0.01,
        normalize: str = 'UnitBall',
        rotation_range: float = None,
        translation_range: float = 0.2,
        non_rigid: bool = False,
        use_natural_pairs: bool = False,
        full_dataset: bool = False,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Loads FAUST data using FAUSTRegistrationDataset.
        full_dataset=True uses all 100 FAUST registrations; valid because FAUST
        is never seen during training, so this is OOD eval with no leakage.
        """
        try:
            dataset = FAUSTRegistrationDataset(
                data_dir=data_dir,
                n_samples=n_samples,
                n_points=n_points,
                noise_std=noise_std,
                normalize=normalize,
                split=None if full_dataset else 'test',
                rotation_range=rotation_range,
                translation_range=translation_range,
                non_rigid=non_rigid,
                use_natural_pairs=use_natural_pairs,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            pbar = tqdm(loader, desc=f'FAUST Registration dataset loading ...')

            pairs = []

            for idx, batch in enumerate(pbar):
                source_points = batch['source'].squeeze(0).cpu().numpy()
                target_points = batch['target'].squeeze(0).cpu().numpy()
                transformation = batch['transformation'].squeeze(0).cpu().numpy()

                pairs.append({
                    'source': source_points,
                    'target': target_points,
                    'source_mesh': None,
                    'target_mesh': None,
                    'correspondence': transformation,
                    'index': idx
                })
        except Exception as e:
            print(f"Error loading FAUST data: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

        return pairs

    def load_sample_3dmatch_data(
        self,
        data_dir: str,
        n_samples: int = 100,
        n_points: int = 1024,
        noise_std: float = 0.01,
        normalize: str = 'UnitBall',
        rotation_range: float = None,
        translation_range: float = 0.2,
        non_rigid: bool = False,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Loads 3DMatch data using Match3DPairDataset.
        Mirrors load_sample_modelnet40_data() / load_sample_faust_data() exactly.
        """
        try:
            dataset = Match3DPairDataset(
                data_dir=data_dir,
                n_samples=n_samples,
                n_points=n_points,
                noise_std=noise_std,
                normalize=normalize,
                rotation_range=rotation_range,
                translation_range=translation_range,
                non_rigid=non_rigid,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            pbar = tqdm(loader, desc=f'3DMatch Registration dataset loading ...')

            pairs = []

            for idx, batch in enumerate(pbar):
                source_points = batch['source'].squeeze(0).cpu().numpy()
                target_points = batch['target'].squeeze(0).cpu().numpy()
                transformation = batch['transformation'].squeeze(0).cpu().numpy()

                pairs.append({
                    'source': source_points,
                    'target': target_points,
                    'source_mesh': None,
                    'target_mesh': None,
                    'correspondence': transformation,
                    'index': idx
                })
        except Exception as e:
            print(f"Error loading 3DMatch data: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

        return pairs

    def evaluate_on_sample_data(
        self,
        model_name: str,
        model: nn.Module,
        args,
        opt,
        dir: str,
        downsample_source: Optional[int] = None,
        downsample_target: Optional[int] = None,
        compute_mesh_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Evaluates a model on sample data.

        Args:
            model_name: Name of the model being evaluated
            model: Model to evaluate
            dir: Directory containing sample data files
            downsample_source: Number of points to downsample source to (optional)
            downsample_target: Number of points to downsample target to (optional)
            compute_mesh_metrics: Whether to compute mesh-based metrics (requires trimesh)

        Returns:
            Dictionary of aggregated metrics
        """
        print(f"\nEvaluating {model_name} on sample data...")

        # Seed RNG so every model is evaluated on the exact same pairs
        seed = getattr(args, 'seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Loads sample pairs (with meshes if computing mesh metrics)
        if args.dataset.lower() == 'atlasnet':
            pairs = self.load_sample_shapenet_data(opt)
        elif args.dataset.lower() == 'modelnet40':
            # Use ModelNet40RegistrationDataset to generate pairs
            pairs = self.load_sample_modelnet40_data(
                data_dir=dir,
                n_samples=args.n_samples,
                n_points=args.n_points,
                noise_std=args.noise_std,
                save_files=args.save_files,
                normalize=args.normalization,
                rotation_range=args.rotation_range,
                translation_range=args.translation_range,
                non_rigid=args.non_rigid,
            )
        elif args.dataset.lower() == 'faust':
            pairs = self.load_sample_faust_data(
                data_dir=dir,
                n_samples=args.n_samples,
                n_points=args.n_points,
                noise_std=args.noise_std,
                normalize=args.normalization,
                rotation_range=args.rotation_range,
                translation_range=args.translation_range,
                non_rigid=args.non_rigid,
                use_natural_pairs=getattr(args, 'faust_natural_pairs', False),
                full_dataset=getattr(args, 'faust_full_dataset', False),
            )
        elif args.dataset.lower() == '3dmatch':
            pairs = self.load_sample_3dmatch_data(
                data_dir=dir,
                n_samples=args.n_samples,
                n_points=args.n_points,
                noise_std=args.noise_std,
                normalize=args.normalization,
                rotation_range=args.rotation_range,
                translation_range=args.translation_range,
                non_rigid=args.non_rigid,
            )
        elif args.dataset.lower() == 'shapenet':
            pairs = self.load_sample_shapenet_data(
                data_dir=dir,
                n_samples=args.n_samples,
                n_points=args.n_points,
                noise_std=args.noise_std,
                normalize=args.normalization,
                rotation_range=args.rotation_range,
                translation_range=args.translation_range,
                non_rigid=args.non_rigid,
                shapenet13=getattr(args, 'shapenet13', False),
                class_choice=getattr(args, 'class_choice', None),
            )

        if len(pairs) == 0:
            empty_stat = {'mean': float('nan'), 'std': 0.0, 'median': float('nan'), 'min': float('nan'), 'max': float('nan')}
            return {
                'model': model_name,
                'dataset': 'sample',
                'n_samples': 0,
                'success_rate': 0.0,
                'chamfer_distance': empty_stat,
                'registration_error': empty_stat,
                'earth_movers_distance': empty_stat,
                'sliced_wasserstein_distance': empty_stat,
                'point_to_point_error': empty_stat,
                'registration_recall': empty_stat,
                'rotation_error': empty_stat,
                'translation_error': empty_stat,
                'inference_time': empty_stat,
            }

        # Initializes metric accumulators
        chamfer_distances = []
        registration_errors = []
        emd_values = []
        swd_values = []
        p2p_errors = []
        recall_values = []
        correspondence_errors = []
        geodesic_distances = []
        rotation_errors = []
        translation_errors = []
        inference_times = []
        successes = []

        # Per-sample arrays for bootstrapping / significance testing
        n_total = len(pairs)
        per_sample = {
            'sample_indices': list(range(n_total)),
            'success': [False] * n_total,
            'chamfer_distance': [float('nan')] * n_total,
            'registration_error': [float('nan')] * n_total,
            'earth_movers_distance': [float('nan')] * n_total,
            'sliced_wasserstein_distance': [float('nan')] * n_total,
            'point_to_point_error': [float('nan')] * n_total,
            'registration_recall': [float('nan')] * n_total,
            'rotation_error': [float('nan')] * n_total,
            'translation_error': [float('nan')] * n_total,
            'inference_time': [float('nan')] * n_total,
        }

        # Creates progress bar
        pbar = tqdm(pairs, desc=f'{model_name}')

        for sample_idx, pair in enumerate(pbar):
            # Extracts point clouds
            source = pair['source']
            target = pair['target']

            # Downsamples if requested
            if downsample_source is not None and len(source) > downsample_source:
                indices = np.random.choice(len(source), downsample_source, replace=False)
                source = source[indices]

            if downsample_target is not None and len(target) > downsample_target:
                indices = np.random.choice(len(target), downsample_target, replace=False)
                target = target[indices]

            # Converts to tensors
            source_tensor = torch.from_numpy(source).float() if isinstance(source, np.ndarray) else torch.tensor(source)
            target_tensor = torch.from_numpy(target).float() if isinstance(target, np.ndarray) else torch.tensor(target)
            gt_transform = pair.get('correspondence')
            gt_transform = gt_transform.cpu().numpy() if isinstance(gt_transform, torch.Tensor) else gt_transform

            # Evaluates on this pair
            metrics = self.evaluate_on_pair(model, source_tensor, target_tensor, ground_truth_transform=gt_transform, model_name=model_name)

            # Accumulates metrics
            if metrics['success']:
                chamfer_distances.append(metrics['chamfer_distance'])
                registration_errors.append(metrics['registration_error'])
                if not np.isinf(metrics['earth_movers_distance']):
                    emd_values.append(metrics['earth_movers_distance'])
                if not np.isinf(metrics['sliced_wasserstein_distance']):
                    swd_values.append(metrics['sliced_wasserstein_distance'])
                if not np.isnan(metrics['point_to_point_error']):
                    p2p_errors.append(metrics['point_to_point_error'])
                if not np.isnan(metrics['registration_recall']):
                    recall_values.append(metrics['registration_recall'])
                if not np.isnan(metrics['rotation_error']):
                    rotation_errors.append(metrics['rotation_error'])
                if not np.isnan(metrics['translation_error']):
                    translation_errors.append(metrics['translation_error'])
                if 'inference_time' in metrics:
                    inference_times.append(metrics['inference_time'])

                # Record per-sample values (NaN stays for filtered/failed)
                per_sample['success'][sample_idx] = True
                per_sample['chamfer_distance'][sample_idx] = metrics['chamfer_distance']
                per_sample['registration_error'][sample_idx] = metrics['registration_error']
                if not np.isinf(metrics['earth_movers_distance']):
                    per_sample['earth_movers_distance'][sample_idx] = metrics['earth_movers_distance']
                if not np.isinf(metrics['sliced_wasserstein_distance']):
                    per_sample['sliced_wasserstein_distance'][sample_idx] = metrics['sliced_wasserstein_distance']
                if not np.isnan(metrics['point_to_point_error']):
                    per_sample['point_to_point_error'][sample_idx] = metrics['point_to_point_error']
                if not np.isnan(metrics['registration_recall']):
                    per_sample['registration_recall'][sample_idx] = metrics['registration_recall']
                if not np.isnan(metrics['rotation_error']):
                    per_sample['rotation_error'][sample_idx] = metrics['rotation_error']
                if not np.isnan(metrics['translation_error']):
                    per_sample['translation_error'][sample_idx] = metrics['translation_error']
                if 'inference_time' in metrics:
                    per_sample['inference_time'][sample_idx] = metrics['inference_time']

                # Computes mesh-based metrics if available
                # Note: These metrics require matching topology (no downsampling)
                if (compute_mesh_metrics and TRIMESH_AVAILABLE and
                    pair.get('target_mesh') is not None and
                    downsample_source is None and downsample_target is None):
                    try:
                        # Gets registered source points (convert back to numpy)
                        with torch.no_grad():
                            result = model(source_tensor.unsqueeze(0).to(self.device),
                                         target_tensor.unsqueeze(0).to(self.device))

                        if isinstance(result, dict):
                            registered = result.get('transformed_source', source_tensor.unsqueeze(0))
                        elif isinstance(result, tuple):
                            registered = result[0]
                        else:
                            registered = result

                        registered_np = registered.squeeze(0).cpu().numpy()

                        # Computes Correspondence Error using predicted vertices
                        # (requires same number of vertices as target - FAUST topology)
                        if registered_np.shape[0] == pair['target_mesh'].vertices.shape[0]:
                            corr_error = Correspondence_Error(
                                target=pair['target_mesh'],
                                radius=0.05,
                                V_pred=registered_np
                            ).item()
                            correspondence_errors.append(corr_error)

                        # Computes Geodesic Distance (requires identical topology)
                        if (registered_np.shape[0] == pair['target_mesh'].vertices.shape[0] and
                            pair['source_mesh'].faces.shape == pair['target_mesh'].faces.shape):
                            geod_dist, _ = Geodesic_Distance(
                                source=pair['source_mesh'],
                                target=pair['target_mesh'],
                                seeds=256
                            )
                            geodesic_distances.append(geod_dist.item())

                    except Exception as e:
                        print(f"Warning: Mesh-based metric computation failed: {str(e)}")

                successes.append(1)
            else:
                successes.append(0)

            # Updates progress bar
            if len(chamfer_distances) > 0:
                pbar.set_postfix({
                    'chamfer': f"{np.mean(chamfer_distances):.4f}",
                    'reg_error': f"{np.mean(registration_errors):.4f}",
                    'emd': f"{np.mean(emd_values):.4f}" if emd_values else 'N/A',
                    'success': f"{np.mean(successes)*100:.1f}%"
                })

        # Helper function to compute statistics
        def compute_stats(values):
            if values:
                return {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                return {
                    'mean': float('nan'),
                    'std': 0.0,
                    'median': float('nan'),
                    'min': float('nan'),
                    'max': float('nan')
                }

        # Computes summary statistics
        results = {
            'model': model_name,
            'dataset': 'sample',
            'n_samples': len(pairs),
            'source_points': len(pairs[0]['source']) if downsample_source is None else downsample_source,
            'target_points': len(pairs[0]['target']) if downsample_target is None else downsample_target,
            'success_rate': np.mean(successes),
            'chamfer_distance': compute_stats(chamfer_distances),
            'registration_error': compute_stats(registration_errors),
            'earth_movers_distance': compute_stats(emd_values),
            'sliced_wasserstein_distance': compute_stats(swd_values),
            'point_to_point_error': compute_stats(p2p_errors),
            'registration_recall': compute_stats(recall_values),
            'correspondence_error': compute_stats(correspondence_errors),
            'geodesic_distance': compute_stats(geodesic_distances),
            'rotation_error': compute_stats(rotation_errors),
            'translation_error': compute_stats(translation_errors),
            'inference_time': compute_stats(inference_times),
            'per_sample_metrics': per_sample,
        }

        return results

    def evaluate_on_shapenet_data(
        self,
        model_name: str,
        model: nn.Module,
        dir: str,
        opt = None,
    ) -> Dict[str, float]:
        """
        Evaluates a model on shapenet data.

        Args:
            model_name: Name of the model being evaluated
            model: Model to evaluate
            dir: Directory containing sample data files
            opt: EasyDict for options to build AtlasNet

        Returns:
            Dictionary of aggregated metrics
        """
        print(f"\nEvaluating {model_name} on shapenet data...")

        # Load Trainer
        from baselines.pretrained_models.exp_atlasnet.AtlasNet.training.trainer import Trainer

        my_utils.plant_seeds(random_seed=opt.random_seed)
        trainer = Trainer(opt)
        trainer.build_dataset()
        trainer.build_network()
        trainer.flags.train = False
        trainer.network.eval()
        
        if trainer.datasets.len_dataset_test == 0:
            return {
                'model': model_name,
                'dataset': 'sample',
                'success_rate': 0.0,
                'chamfer_distance': {'mean': float('inf')},
            }
        
        # Initializes metric accumulators
        chamfer_distances = []
        registration_errors = []
        emd_values = []
        swd_values = []
        p2p_errors = []
        recall_values = []
        correspondence_errors = []
        geodesic_distances = []
        rotation_errors = []
        translation_errors = []
        successes = []

        with torch.no_grad():
            iterator = trainer.datasets.dataloader_test.__iter__()
            pbar = tqdm(iterator, desc=f'{model_name}')
            trainer.reset_iteration()
            for data in pbar:
                trainer.increment_iteration()
                trainer.data = EasyDict(data)
                trainer.data.points = trainer.data.points.to(self.opt.device)
                trainer.test_iteration()
                trainer.data.pointsReconstructed = trainer.data.pointsReconstructed_prims.transpose(2, 3).contiguous()
                trainer.data.pointsReconstructed = trainer.data.pointsReconstructed.view(trainer.batch_size, -1, 3)

                # Computes chamfer metrics
                chamfer = self.chamfer_metric(trainer.data.points.squeeze(0),
                                            trainer.data.pointsReconstructed.squeeze(0)).item()
                
                metrics = {
                    'chamfer_distance': chamfer,
                    'registration_error': None,
                    'earth_movers_distance': None,
                    'sliced_wasserstein_distance': None,
                    'point_to_point_error': None,
                    'registration_recall': None,
                    'rotation_error': None,
                    'translation_error': None,
                    'success': True
                }

                # Accumulates metrics
                if metrics['success']:
                    chamfer_distances.append(metrics['chamfer_distance'])
                    successes.append(1)
                else:
                    successes.append(0)

            # Updates progress bar
            if len(chamfer_distances) > 0:
                pbar.set_postfix({
                    'chamfer': f"{np.mean(chamfer_distances):.4f}",
                    'reg_error': f"{np.mean(registration_errors):.4f}",
                    'emd': f"{np.mean(emd_values):.4f}" if emd_values else 'N/A',
                    'success': f"{np.mean(successes)*100:.1f}%"
                })

        # Helper function to compute statistics
        def compute_stats(values):
            if values:
                return {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                return {
                    'mean': float('nan'),
                    'std': 0.0,
                    'median': float('nan'),
                    'min': float('nan'),
                    'max': float('nan')
                }

        # Computes summary statistics
        results = {
            'model': model_name,
            'dataset': 'sample',
            'n_samples': len(pbar),
            'success_rate': np.mean(successes),
            'chamfer_distance': compute_stats(chamfer_distances),
            'registration_error': compute_stats(registration_errors),
            'earth_movers_distance': compute_stats(emd_values),
            'sliced_wasserstein_distance': compute_stats(swd_values),
            'point_to_point_error': compute_stats(p2p_errors),
            'registration_recall': compute_stats(recall_values),
            'correspondence_error': compute_stats(correspondence_errors),
            'geodesic_distance': compute_stats(geodesic_distances),
            'rotation_error': compute_stats(rotation_errors),
            'translation_error': compute_stats(translation_errors)
        }

        return results
    
    def evaluate_on_dataset(
        self,
        model_name: str,
        model: nn.Module,
        n_samples: int = 100,
        n_points: int = 1000,
        noise_std: float = 0.01,
        shape_types: Tuple[str, ...] = ('sphere', 'cube', 'cylinder', 'torus')
    ) -> Dict[str, float]:
        """
        Evaluates a model on synthetic dataset.

        Args:
            model_name: Name of the model being evaluated
            model: Model to evaluate
            n_samples: Number of test samples
            n_points: Number of points per cloud
            noise_std: Noise level for synthetic data
            shape_types: Types of shapes to test on

        Returns:
            Dictionary of aggregated metrics
        """
        print(f"\nEvaluating {model_name} on {n_samples} synthetic samples...")

        # Initializes metric accumulators
        chamfer_distances = []
        registration_errors = []
        emd_values = []
        swd_values = []
        p2p_errors = []
        recall_values = []
        rotation_errors = []
        translation_errors = []
        successes = []

        # Creates progress bar
        pbar = tqdm(range(n_samples), desc=f'{model_name}')

        for i in pbar:
            # Generates synthetic registration pair
            shape_type = shape_types[i % len(shape_types)]
            pair = generate_registration_pair(
                shape_type=shape_type,
                n_points=n_points,
                noise_std=noise_std,
                outlier_ratio=0.05
            )

            # Converts to tensors
            source = torch.from_numpy(pair['source']).float()
            target = torch.from_numpy(pair['target']).float()

            # Evaluates on this pair
            metrics = self.evaluate_on_pair(
                model, source, target,
                ground_truth_transform=pair['transformation'],
                model_name=model_name
            )

            # Accumulates metrics
            if metrics['success']:
                chamfer_distances.append(metrics['chamfer_distance'])
                registration_errors.append(metrics['registration_error'])
                if not np.isinf(metrics['earth_movers_distance']):
                    emd_values.append(metrics['earth_movers_distance'])
                if not np.isinf(metrics['sliced_wasserstein_distance']):
                    swd_values.append(metrics['sliced_wasserstein_distance'])
                if not np.isnan(metrics['point_to_point_error']):
                    p2p_errors.append(metrics['point_to_point_error'])
                if not np.isnan(metrics['registration_recall']):
                    recall_values.append(metrics['registration_recall'])
                if not np.isnan(metrics['rotation_error']):
                    rotation_errors.append(metrics['rotation_error'])
                if not np.isnan(metrics['translation_error']):
                    translation_errors.append(metrics['translation_error'])
                successes.append(1)
            else:
                successes.append(0)

            # Updates progress bar
            if len(chamfer_distances) > 0:
                pbar.set_postfix({
                    'chamfer': f"{np.mean(chamfer_distances):.4f}",
                    'reg_error': f"{np.mean(registration_errors):.4f}",
                    'rot_err': f"{np.mean(rotation_errors):.2f}°" if rotation_errors else 'N/A',
                    'success': f"{np.mean(successes)*100:.1f}%"
                })

        # Helper function to compute statistics
        def compute_stats(values):
            if values:
                return {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                return {
                    'mean': float('nan'),
                    'std': 0.0,
                    'median': float('nan'),
                    'min': float('nan'),
                    'max': float('nan')
                }

        # Computes summary statistics
        results = {
            'model': model_name,
            'n_samples': n_samples,
            'n_points': n_points,
            'success_rate': np.mean(successes),
            'chamfer_distance': compute_stats(chamfer_distances),
            'registration_error': compute_stats(registration_errors),
            'earth_movers_distance': compute_stats(emd_values),
            'sliced_wasserstein_distance': compute_stats(swd_values),
            'point_to_point_error': compute_stats(p2p_errors),
            'registration_recall': compute_stats(recall_values),
            'rotation_error': compute_stats(rotation_errors),
            'translation_error': compute_stats(translation_errors)
        }

        return results

    def run_baseline_comparison(
        self,
        models_to_evaluate: List[str],
        checkpoint_paths: Optional[Dict[str, str]] = None,
        n_samples: int = 100,
        n_points: int = 1000,
        save_results: bool = True,
        config: Optional[Dict[str, object]] = None,
        git_commit: str = 'unknown',
        hostname: Optional[str] = None,
        start_time: Optional[str] = None,
        gpu_info: Optional[Dict[str, object]] = None
    ) -> Dict[str, Dict]:
        """
        Runs comparison of multiple baseline models.

        Args:
            models_to_evaluate: List of model names to evaluate
            checkpoint_paths: Dictionary mapping model names to checkpoint paths
            n_samples: Number of samples to evaluate on
            n_points: Number of points per cloud
            save_results: Whether to save results to file
            config: Evaluation CLI config
            git_commit: Git commit hash
            hostname: Hostname where evaluation ran
            start_time: Evaluation start timestamp
            gpu_info: GPU metadata

        Returns:
            Dictionary containing all evaluation results
        """
        if checkpoint_paths is None:
            checkpoint_paths = {}

        # Stores all results
        all_results = {
            'config': config or {},
            'git_commit': git_commit,
            'hostname': hostname or socket.gethostname(),
            'start_time': start_time or datetime.now().isoformat(),
            'end_time': None,
            'evaluation_config': {
                'n_samples': n_samples,
                'n_points': n_points,
                'device': self.device
            },
            'gpu_info': gpu_info or {},
            'models': {}
        }

        print("=" * 80)
        print("BASELINE MODEL EVALUATION")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Samples: {n_samples}")
        print(f"Points per cloud: {n_points}")
        print(f"Models to evaluate: {', '.join(models_to_evaluate)}")
        print("=" * 80)

        # Evaluates each model
        for model_name in models_to_evaluate:
            print(f"\n{'='*80}")
            print(f"Evaluating {model_name.upper()}")
            print('='*80)

            # Loads model
            checkpoint_path = checkpoint_paths.get(model_name)
            model = self.load_model(model_name, checkpoint_path)

            if model is None:
                print(f"Skipping {model_name} (failed to load)")
                continue

            # Evaluates model
            results = self.evaluate_on_dataset(
                model_name, model,
                n_samples=n_samples,
                n_points=n_points
            )

            # Stores results
            all_results['models'][model_name] = results

            # Prints summary
            print(f"\n{model_name.upper()} Results:")
            print(f"  Success Rate: {results['success_rate']*100:.1f}%")
            print(f"  Chamfer Distance: {results['chamfer_distance']['mean']:.6f} ± {results['chamfer_distance']['std']:.6f}")
            print(f"  Registration Error: {results['registration_error']['mean']:.6f} ± {results['registration_error']['std']:.6f}")

            # Prints additional metrics if available
            if 'earth_movers_distance' in results and not np.isnan(results['earth_movers_distance']['mean']):
                print(f"  Earth Mover's Distance: {results['earth_movers_distance']['mean']:.6f} ± {results['earth_movers_distance']['std']:.6f}")
            if 'sliced_wasserstein_distance' in results and not np.isnan(results['sliced_wasserstein_distance']['mean']):
                print(f"  Sliced Wasserstein Distance: {results['sliced_wasserstein_distance']['mean']:.6f} ± {results['sliced_wasserstein_distance']['std']:.6f}")
            if 'registration_recall' in results and not np.isnan(results['registration_recall']['mean']):
                print(f"  Registration Recall: {results['registration_recall']['mean']*100:.1f}%")
            if 'correspondence_error' in results and not np.isnan(results['correspondence_error']['mean']):
                print(f"  Correspondence Error: {results['correspondence_error']['mean']:.6f} ± {results['correspondence_error']['std']:.6f}")
            if 'geodesic_distance' in results and not np.isnan(results['geodesic_distance']['mean']):
                print(f"  Geodesic Distance: {results['geodesic_distance']['mean']:.6f} ± {results['geodesic_distance']['std']:.6f}")
            if 'rotation_error' in results and not np.isnan(results['rotation_error']['mean']):
                print(f"  Rotation Error: {results['rotation_error']['mean']:.6f}° ± {results['rotation_error']['std']:.6f}°")
            if 'translation_error' in results and not np.isnan(results['translation_error']['mean']):
                print(f"  Translation Error: {results['translation_error']['mean']:.6f} ± {results['translation_error']['std']:.6f}")

        # Saves results if requested
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.results_dir / f'baseline_results_{timestamp}.json'
            all_results['end_time'] = datetime.now().isoformat()

            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            print(f"\n{'='*80}")
            print(f"Results saved to: {results_file}")
            print('='*80)

        return all_results

    def print_comparison_table(self, results: Dict[str, Dict]):
        """
        Prints a formatted comparison table of results.

        Args:
            results: Dictionary of evaluation results
        """
        # Log number of samples evaluated per model
        print(f"\n{'='*170}")
        print("EVALUATION SUMMARY")
        print('='*170)
        for model_name, model_results in results['models'].items():
            n_eval = model_results.get('n_samples', 'unknown')
            print(f"  {model_name}: {n_eval} samples evaluated")
        print('='*170)

        print(f"\n{'='*170}")
        print("BASELINE COMPARISON TABLE - ALL METRICS")
        print('='*170)

        # Prints header
        print(f"{'Model':<12} {'Success':<8} {'Chamfer':<16} {'RegError':<16} "
              f"{'EMD':<16} {'SWD':<16} {'P2P':<16} {'Recall':<8} {'RotErr':<16} {'TransErr':<16} {'Geod':<16} {'Time(s)':<16}")
        print('-'*170)

        # Prints each model's results
        for model_name, model_results in results['models'].items():
            success_rate = model_results['success_rate'] * 100

            # Helper to format metric
            def fmt(metric_dict):
                mean = metric_dict['mean']
                std = metric_dict['std']
                if np.isnan(mean):
                    return "N/A".ljust(16)
                return f"{mean:.4f}±{std:.4f}".ljust(16)

            chamfer = fmt(model_results['chamfer_distance'])
            reg_error = fmt(model_results['registration_error'])
            emd = fmt(model_results.get('earth_movers_distance', {'mean': float('nan'), 'std': 0}))
            swd = fmt(model_results.get('sliced_wasserstein_distance', {'mean': float('nan'), 'std': 0}))
            p2p = fmt(model_results.get('point_to_point_error', {'mean': float('nan'), 'std': 0}))
            rot_err = fmt(model_results.get('rotation_error', {'mean': float('nan'), 'std': 0}))
            trans_err = fmt(model_results.get('translation_error', {'mean': float('nan'), 'std': 0}))
            geod = fmt(model_results.get('geodesic_distance', {'mean': float('nan'), 'std': 0}))

            recall_dict = model_results.get('registration_recall', {'mean': float('nan')})
            recall_mean = recall_dict['mean']
            recall_str = f"{recall_mean*100:.1f}%" if not np.isnan(recall_mean) else "N/A"

            inf_time = fmt(model_results.get('inference_time', {'mean': float('nan'), 'std': 0}))

            print(f"{model_name:<12} {success_rate:<8.1f} {chamfer} {reg_error} "
                  f"{emd} {swd} {p2p} {recall_str:<8} {rot_err} {trans_err} {geod} {inf_time}")

        print('='*170)

        # Prints metric descriptions
        print("\nMetric Descriptions:")
        print("  Chamfer   : Chamfer Distance (bidirectional nearest neighbor)")
        print("  RegError  : Registration Error (RMSE from CPD paper)")
        print("  EMD       : Earth Mover's Distance (Sinkhorn approximation)")
        print("  SWD       : Sliced Wasserstein Distance")
        print("  P2P       : Point-to-Point Error (1-to-1 correspondence)")
        print("  Recall    : Registration Recall (success rate at threshold)")
        print("  RotErr    : Rotation Error in degrees (requires ground truth)")
        print("  TransErr  : Translation Error (requires ground truth)")
        print("  Geod      : Geodesic Distance (surface-aware distortion)")
        print('='*170)

    # DEPRECATED: Use evaluate_on_sample_data() with dataset='3dmatch' instead.
    # Kept for backward compatibility only. The unified path through
    # evaluate_on_sample_data -> load_sample_3dmatch_data is preferred.
    def evaluate_on_3dmatch_data(
        self,
        model_name: str,
        model: nn.Module,
        data_dir: str,
        n_samples: int = 100,
        n_points: int = 1024,
        use_gt_pairs: bool = False,
        max_pairs_per_scene: int = None
    ) -> Dict[str, float]:
        """
        Evaluates a model on 3DMatch dataset.

        Args:
            model_name: Name of the model being evaluated
            model: Model to evaluate
            data_dir: Directory containing 3DMatch scenes
            n_samples: Number of test samples (for generated pairs)
            n_points: Number of points per cloud
            use_gt_pairs: Whether to use ground truth pairs or generate pairs
            max_pairs_per_scene: Maximum pairs per scene for ground truth pairs

        Returns:
            Dictionary of aggregated metrics
        """
        print(f"\nEvaluating {model_name} on 3DMatch data...")
        print(f"Data directory: {data_dir}")
        print(f"Use ground truth pairs: {use_gt_pairs}")
        print(f"Points per cloud: {n_points}")

        # Create appropriate dataset
        print(f"DEBUG: data_dir type = {type(data_dir)}")
        print(f"DEBUG: data_dir value = {data_dir}")
        print(f"DEBUG: data_dir exists? {Path(data_dir).exists()}")
        
        if use_gt_pairs:
            dataset = Match3DRegistrationDataset(
                data_dir=data_dir,
                n_points=n_points,
                max_pairs_per_scene=max_pairs_per_scene,
                train=False
            )
            print(f"Loaded {len(dataset)} ground truth pairs")
        else:
            dataset = Match3DPairDataset(
                data_dir=data_dir,
                n_samples=n_samples,
                n_points=n_points,
                noise_std=0.01
            )
            print(f"Generated {len(dataset)} pairs from fragments")

        # Create data loader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Initialize metric accumulators
        chamfer_distances = []
        registration_errors = []
        emd_values = []
        swd_values = []
        p2p_errors = []
        recall_values = []
        correspondence_errors = []
        geodesic_distances = []
        rotation_errors = []
        translation_errors = []
        successes = []

        # Creates progress bar
        pbar = tqdm(dataloader, desc=f'{model_name}')

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                if i >= n_samples:
                    break

                try:
                    # Extract data from batch
                    source = batch['source'].to(self.device).squeeze(0)  # Remove batch dimension
                    target = batch['target'].to(self.device).squeeze(0)
                    gt_transform = batch['transformation'].squeeze(0).cpu().numpy()

                    # Evaluate on this pair
                    metrics = self.evaluate_on_pair(
                        model, source, target,
                        ground_truth_transform=gt_transform,
                        model_name=model_name
                    )

                    # Accumulates metrics
                    if metrics['success']:
                        chamfer_distances.append(metrics['chamfer_distance'])
                        registration_errors.append(metrics['registration_error'])
                        if not np.isinf(metrics['earth_movers_distance']):
                            emd_values.append(metrics['earth_movers_distance'])
                        if not np.isinf(metrics['sliced_wasserstein_distance']):
                            swd_values.append(metrics['sliced_wasserstein_distance'])
                        if not np.isnan(metrics['point_to_point_error']):
                            p2p_errors.append(metrics['point_to_point_error'])
                        if not np.isnan(metrics['registration_recall']):
                            recall_values.append(metrics['registration_recall'])
                        if not np.isnan(metrics['rotation_error']):
                            rotation_errors.append(metrics['rotation_error'])
                        if not np.isnan(metrics['translation_error']):
                            translation_errors.append(metrics['translation_error'])
                        successes.append(1)
                    else:
                        successes.append(0)

                    # Updates progress bar
                    if len(chamfer_distances) > 0:
                        pbar.set_postfix({
                            'chamfer': f"{np.mean(chamfer_distances):.4f}",
                            'reg_error': f"{np.mean(registration_errors):.4f}",
                            'emd': f"{np.mean(emd_values):.4f}" if emd_values else 'N/A',
                            'success': f"{np.mean(successes)*100:.1f}%"
                        })

                except Exception as e:
                    print(f"Error evaluating pair {i}: {e}")
                    successes.append(0)
                    continue

        # Helper function to compute statistics
        def compute_stats(values):
            if values:
                return {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                return {
                    'mean': float('nan'),
                    'std': 0.0,
                    'median': float('nan'),
                    'min': float('nan'),
                    'max': float('nan')
                }

        # Computes summary statistics
        results = {
            'model': model_name,
            'dataset': '3dmatch',
            'n_samples': min(n_samples, len(dataset)),
            'source_points': n_points,
            'target_points': n_points,
            'success_rate': np.mean(successes),
            'chamfer_distance': compute_stats(chamfer_distances),
            'registration_error': compute_stats(registration_errors),
            'earth_movers_distance': compute_stats(emd_values),
            'sliced_wasserstein_distance': compute_stats(swd_values),
            'point_to_point_error': compute_stats(p2p_errors),
            'registration_recall': compute_stats(recall_values),
            'correspondence_error': compute_stats(correspondence_errors),
            'geodesic_distance': compute_stats(geodesic_distances),
            'rotation_error': compute_stats(rotation_errors),
            'translation_error': compute_stats(translation_errors)
        }

        return results

    def evaluate_on_Faust_data(
            self,
            model_name: str,
            model: nn.Module,
            data_dir: str,
            n_samples: int = 20,
            n_points = 1024,
        ) -> Dict[str, float]:
            """
            Evaluates a model on 3DMatch dataset.

            Args:
                model_name: Name of the model being evaluated
                model: Model to evaluate
                data_dir: Directory containing 3DMatch scenes
                n_samples: Number of test samples (for generated pairs)
                n_points: Number of points per cloud
                use_gt_pairs: Whether to use ground truth pairs or generate pairs
                max_pairs_per_scene: Maximum pairs per scene for ground truth pairs

            Returns:
                Dictionary of aggregated metrics
            """
            print(f"\nEvaluating {model_name} on Faust data...")
            print(f"Data directory: {data_dir}")
            

            
            dataset = FaustDataset(
                FAUST_Dataset_Path=data_dir,
                Train= False
            )
            print(f"Generated {len(dataset)} pairs from fragments")

            # Create data loader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # Initialize metric accumulators
            chamfer_distances = []
            registration_errors = []
            emd_values = []
            swd_values = []
            p2p_errors = []
            recall_values = []
            correspondence_errors = []
            geodesic_distances = []
            rotation_errors = []
            translation_errors = []
            successes = []

            # Creates progress bar
            pbar = tqdm(dataloader, desc=f'{model_name}')

            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(pbar):
                    if i >= n_samples:
                        break

                    try:
                        keep_idxs = batch['keep_idx'].to(self.device).squeeze(0)
                        # Extract data from batch
                        source = batch['source'].to(self.device).squeeze(0)  # Remove batch dimension
                        target = batch['target'].to(self.device).squeeze(0)
                        
                        gt_transform = np.array(source[keep_idxs])

                        # Evaluate on this pair
                        metrics = self.evaluate_on_pair(
                            model, source, target,
                            ground_truth_transform=gt_transform,
                            keep_idxs=keep_idxs,
                            model_name=model_name
                        )

                        # Accumulates metrics
                        if metrics['success']:
                            chamfer_distances.append(metrics['chamfer_distance'])
                            registration_errors.append(metrics['registration_error'])
                            if not np.isinf(metrics['earth_movers_distance']):
                                emd_values.append(metrics['earth_movers_distance'])
                            if not np.isinf(metrics['sliced_wasserstein_distance']):
                                swd_values.append(metrics['sliced_wasserstein_distance'])
                            if not np.isnan(metrics['point_to_point_error']):
                                p2p_errors.append(metrics['point_to_point_error'])
                            if not np.isnan(metrics['registration_recall']):
                                recall_values.append(metrics['registration_recall'])
                            if not np.isnan(metrics['rotation_error']):
                                rotation_errors.append(metrics['rotation_error'])
                            if not np.isnan(metrics['translation_error']):
                                translation_errors.append(metrics['translation_error'])
                            successes.append(1)
                        else:
                            successes.append(0)

                        # Updates progress bar
                        if len(chamfer_distances) > 0:
                            pbar.set_postfix({
                                'chamfer': f"{np.mean(chamfer_distances):.4f}",
                                'reg_error': f"{np.mean(registration_errors):.4f}",
                                'emd': f"{np.mean(emd_values):.4f}" if emd_values else 'N/A',
                                'success': f"{np.mean(successes)*100:.1f}%"
                            })

                    except Exception as e:
                        print(f"Error evaluating pair {i}: {e}")
                        successes.append(0)
                        continue

            # Helper function to compute statistics
            def compute_stats(values):
                if values:
                    return {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                else:
                    return {
                        'mean': float('nan'),
                        'std': 0.0,
                        'median': float('nan'),
                        'min': float('nan'),
                        'max': float('nan')
                    }

            # Computes summary statistics
            results = {
                'model': model_name,
                'dataset': 'Faust',
                'n_samples': min(n_samples, len(dataset)),
                'source_points': n_points,
                'target_points': n_points,
                'success_rate': np.mean(successes),
                'chamfer_distance': compute_stats(chamfer_distances),
                'registration_error': compute_stats(registration_errors),
                'earth_movers_distance': compute_stats(emd_values),
                'sliced_wasserstein_distance': compute_stats(swd_values),
                'point_to_point_error': compute_stats(p2p_errors),
                'registration_recall': compute_stats(recall_values),
                'correspondence_error': compute_stats(correspondence_errors),
                'geodesic_distance': compute_stats(geodesic_distances),
                'rotation_error': compute_stats(rotation_errors),
                'translation_error': compute_stats(translation_errors)
            }

            return results

    def evaluate_on_spare_data(
            self,
            model_name: str,
            model: nn.Module,
            data_dir: str,
        ) -> Dict[str, float]:
            """
            Evaluates a model on 3DMatch dataset.

            Args:
                model_name: Name of the model being evaluated
                model: Model to evaluate
                data_dir: Directory containing 3DMatch scenes
                n_samples: Number of test samples (for generated pairs)
                n_points: Number of points per cloud
                use_gt_pairs: Whether to use ground truth pairs or generate pairs
                max_pairs_per_scene: Maximum pairs per scene for ground truth pairs

            Returns:
                Dictionary of aggregated metrics
            """
            print(f"\nEvaluating {model_name} on 3DMatch data...")
            print(f"Data directory: {data_dir}")
            

            # Create appropriate dataset
            
            dataset = Match3DPairDataset(
                data_dir=data_dir,
                n_samples=n_samples,
                n_points=n_points,
                noise_std=0.01
            )
            print(f"Generated {len(dataset)} pairs from fragments")

            # Create data loader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # Initialize metric accumulators
            chamfer_distances = []
            registration_errors = []
            emd_values = []
            swd_values = []
            p2p_errors = []
            recall_values = []
            correspondence_errors = []
            geodesic_distances = []
            rotation_errors = []
            translation_errors = []
            successes = []

            # Creates progress bar
            pbar = tqdm(dataloader, desc=f'{model_name}')

            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(pbar):
                    if i >= n_samples:
                        break

                    try:
                        # Extract data from batch
                        source = batch['source'].to(self.device).squeeze(0)  # Remove batch dimension
                        target = batch['target'].to(self.device).squeeze(0)
                        gt_transform = batch['transformation'].squeeze(0).cpu().numpy()

                        # Evaluate on this pair
                        metrics = self.evaluate_on_pair(
                            model, source, target,
                            ground_truth_transform=gt_transform,
                            model_name=model_name
                        )

                        # Accumulates metrics
                        if metrics['success']:
                            chamfer_distances.append(metrics['chamfer_distance'])
                            registration_errors.append(metrics['registration_error'])
                            if not np.isinf(metrics['earth_movers_distance']):
                                emd_values.append(metrics['earth_movers_distance'])
                            if not np.isinf(metrics['sliced_wasserstein_distance']):
                                swd_values.append(metrics['sliced_wasserstein_distance'])
                            if not np.isnan(metrics['point_to_point_error']):
                                p2p_errors.append(metrics['point_to_point_error'])
                            if not np.isnan(metrics['registration_recall']):
                                recall_values.append(metrics['registration_recall'])
                            if not np.isnan(metrics['rotation_error']):
                                rotation_errors.append(metrics['rotation_error'])
                            if not np.isnan(metrics['translation_error']):
                                translation_errors.append(metrics['translation_error'])
                            successes.append(1)
                        else:
                            successes.append(0)

                        # Updates progress bar
                        if len(chamfer_distances) > 0:
                            pbar.set_postfix({
                                'chamfer': f"{np.mean(chamfer_distances):.4f}",
                                'reg_error': f"{np.mean(registration_errors):.4f}",
                                'emd': f"{np.mean(emd_values):.4f}" if emd_values else 'N/A',
                                'success': f"{np.mean(successes)*100:.1f}%"
                            })

                    except Exception as e:
                        print(f"Error evaluating pair {i}: {e}")
                        successes.append(0)
                        continue

            # Helper function to compute statistics
            def compute_stats(values):
                if values:
                    return {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                else:
                    return {
                        'mean': float('nan'),
                        'std': 0.0,
                        'median': float('nan'),
                        'min': float('nan'),
                        'max': float('nan')
                    }

            # Computes summary statistics
            results = {
                'model': model_name,
                'dataset': 'Faust',
                'n_samples': min(n_samples, len(dataset)),
                'source_points': n_points,
                'target_points': n_points,
                'success_rate': np.mean(successes),
                'chamfer_distance': compute_stats(chamfer_distances),
                'registration_error': compute_stats(registration_errors),
                'earth_movers_distance': compute_stats(emd_values),
                'sliced_wasserstein_distance': compute_stats(swd_values),
                'point_to_point_error': compute_stats(p2p_errors),
                'registration_recall': compute_stats(recall_values),
                'correspondence_error': compute_stats(correspondence_errors),
                'geodesic_distance': compute_stats(geodesic_distances),
                'rotation_error': compute_stats(rotation_errors),
                'translation_error': compute_stats(translation_errors)
            }

            return results
            
def main():
    """
    Main evaluation function.
    """
    # Parses arguments
    parser = argparse.ArgumentParser(description='Evaluate baseline registration models')
    parser.add_argument('--models', nargs='+',
                       default=['evoreg', 'pointnetlk', 'ipcrnet', 'prnet'],
                       help='Models to evaluate (supported: evoreg, pointnetlk, ipcrnet, prnet, atlasnet)')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'faust', 'modelnet40', 'atlasnet', '3dmatch', 'shapenet'],
                       help='Dataset to evaluate on')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of test samples (for synthetic/ModelNet40)')
    parser.add_argument('--n_points', type=int, default=1000,
                       help='Points per cloud (for synthetic/ModelNet40)')
    parser.add_argument('--noise_std', type=float, default=0.01,
                       help='Noise standard deviation for ModelNet40 dataset')
    parser.add_argument('--save_files', action='store_true', default=True,
                       help='Save generated registration pairs to disk (for ModelNet40)')
    parser.add_argument('--dir', type=str, help='Directory containing sample data (FAUST/ModelNet40)')
    parser.add_argument('--downsample_source', type=int, default=None,
                       help='Downsample source to N points (for FAUST/ModelNet40)')
    parser.add_argument('--downsample_target', type=int, default=None,
                       help='Downsample target to N points (for FAUST/ModelNet40)')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--pretrained_dir', type=str, default='baselines/pretrained_models',
                       help='Directory with pretrained models')
    parser.add_argument('--results_dir', type=str, default='baselines/results',
                       help='Directory to save results')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                       help='path to model checkpoint .pth')
    # 3DMatch specific arguments
    parser.add_argument('--use_gt_pairs', action='store_true',
                       help='Use ground truth pairs for 3DMatch (vs generated pairs)')
    parser.add_argument('--max_pairs_per_scene', type=int, default=None,
                       help='Maximum pairs per scene for 3DMatch (None = all pairs)')
    parser.add_argument('--recall_threshold', type=float, default=0.1,
                       help='RMSE threshold for registration recall metric (default: 0.1)')
    parser.add_argument('--normalization', type=str, default='UnitBall',
                       choices=['UnitBall', 'BoundingBox', 'Identity'],
                       help='Normalization method for point clouds (default: UnitBall)')
    parser.add_argument('--rotation_range', type=float, default=None,
                       help='Constrain eval rotations to ±N degrees (default: None = full SO(3))')
    parser.add_argument('--translation_range', type=float, default=0.2,
                       help='Eval translation range ±N (default: 0.2)')
    parser.add_argument('--non_rigid', action='store_true', default=False,
                       help='Apply non-rigid deformations to eval pairs')
    parser.add_argument('--faust_natural_pairs', action='store_true', default=False,
                       help='Use natural FAUST cross-pose pairs instead of synthetic transforms')
    parser.add_argument('--faust_full_dataset', action='store_true', default=False,
                       help='Use all 100 FAUST files instead of 20-file test split (OOD eval, no leakage)')
    parser.add_argument('--shapenet13', action='store_true', default=False,
                       help='Use standard 13-category ShapeNet subset')
    # Optional NIA refinement after Stage 3 at inference
    parser.add_argument('--use_eval_nia_s3', action='store_true', default=False,
                       help='Apply NIA refinement after Stage 3 at inference')
    parser.add_argument('--eval_nia_s3_particles', type=int, default=None,
                       help='Override NIA ref 3 particles at inference')
    parser.add_argument('--eval_nia_s3_iterations', type=int, default=None,
                       help='Override NIA ref 3 iterations at inference')
    parser.add_argument('--eval_nia_s3_rot_range', type=float, default=None,
                       help='Override NIA ref 3 rotation search range (degrees) at inference')
    parser.add_argument('--eval_nia_s3_trans_range', type=float, default=None,
                       help='Override NIA ref 3 translation search range at inference')
    # Optional diffusion refinement at inference (experimental)
    parser.add_argument('--use_diffusion_refinement', action='store_true', default=False,
                       help='Apply diffusion reverse process at inference (experimental)')
    parser.add_argument('--diffusion_noise_level', type=float, default=0.01,
                       help='Noise level for diffusion refinement (0-1, fraction of schedule)')
    parser.add_argument('--diffusion_refine_steps', type=int, default=50,
                       help='Number of reverse diffusion steps')
    # Sinkhorn test-time optimization
    parser.add_argument('--use_sinkhorn_tto', action='store_true', default=False,
                       help='Apply Sinkhorn per-point TTO at inference (improves Chamfer/EMD)')
    parser.add_argument('--sinkhorn_tto_steps', type=int, default=15,
                       help='Number of per-point TTO gradient descent steps')
    parser.add_argument('--sinkhorn_tto_lr', type=float, default=0.01,
                       help='Per-point TTO learning rate')
    parser.add_argument('--sinkhorn_tto_lambda', type=float, default=10.0,
                       help='Per-point TTO L2 regularization weight')
    parser.add_argument('--sinkhorn_tto_sinkhorn_eps', type=float, default=0.01,
                       help='Sinkhorn entropic regularization')
    parser.add_argument('--sinkhorn_tto_sinkhorn_iters', type=int, default=50,
                       help='Sinkhorn iterations per TTO step')
    # Sinkhorn rigid TTO (refines R, t)
    parser.add_argument('--use_sinkhorn_tto_rigid', action='store_true', default=False,
                       help='Apply Sinkhorn rigid TTO at inference (improves RotErr/TransErr)')
    parser.add_argument('--sinkhorn_tto_rigid_steps', type=int, default=15,
                       help='Number of rigid TTO gradient descent steps')
    parser.add_argument('--sinkhorn_tto_rigid_lr', type=float, default=0.01,
                       help='Rigid TTO learning rate')
    parser.add_argument('--sinkhorn_tto_rigid_lambda', type=float, default=10.0,
                       help='Rigid TTO L2 regularization weight on corrections')
    parser.add_argument('--sinkhorn_tto_rigid_sinkhorn_eps', type=float, default=0.01,
                       help='Sinkhorn entropic regularization for rigid TTO')
    parser.add_argument('--sinkhorn_tto_rigid_sinkhorn_iters', type=int, default=50,
                       help='Sinkhorn iterations per rigid TTO step')
    parser.add_argument('--use_translation_only_icp', action='store_true', default=False,
                       help='Apply translation-only ICP refinement at inference')
    parser.add_argument('--translation_only_icp_steps', type=int, default=10,
                       help='Number of translation-only ICP iterations')
    parser.add_argument('--translation_only_icp_tau', type=float, default=0.01,
                       help='Distance weighting temperature for translation-only ICP')
    parser.add_argument('--use_kabsch', action='store_true', default=False,
                       help='Summarize final EvoReg outputs with a best-fit rigid transform for R/t metrics')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for reproducible eval pairs (default: 123)')
    args, unknown = parser.parse_known_args()
    config_dict = vars(args).copy()
    if unknown:
        config_dict['unknown_cli_args'] = unknown
    start_time = datetime.now().isoformat()
    hostname = socket.gethostname()
    git_commit = get_git_commit()
    gpu_info = get_gpu_metadata()
    gpu_info['device_requested'] = args.device

    if args.dataset in ['faust', 'modelnet40', 'atlasnet', '3dmatch', 'shapenet'] and args.dir is None:
        parser.error(f"--dir is required when --dataset is '{args.dataset}'")

    # Creates evaluator
    evaluator = BaselineEvaluator(
        device=args.device,
        pretrained_dir=args.pretrained_dir,
        results_dir=args.results_dir,
        recall_threshold=args.recall_threshold
    )

    # Optional: NIA refinement after Stage 3 at inference (off by default)
    evaluator.use_eval_nia_s3 = args.use_eval_nia_s3
    if args.use_eval_nia_s3:
        # Override NIA ref 3 particles/iterations if specified
        if (args.eval_nia_s3_particles is not None or args.eval_nia_s3_iterations is not None
                or args.eval_nia_s3_rot_range is not None or args.eval_nia_s3_trans_range is not None):
            for m in evaluator.models.values():
                inner = m.vae_model if hasattr(m, 'vae_model') else m
                if hasattr(inner, 'inter_nia_s3'):
                    if args.eval_nia_s3_particles is not None:
                        inner.inter_nia_s3.n_particles = args.eval_nia_s3_particles
                    if args.eval_nia_s3_iterations is not None:
                        inner.inter_nia_s3.n_iterations = args.eval_nia_s3_iterations
                    if args.eval_nia_s3_rot_range is not None:
                        import math
                        inner.inter_nia_s3.rotation_range = math.radians(args.eval_nia_s3_rot_range)
                    if args.eval_nia_s3_trans_range is not None:
                        inner.inter_nia_s3.translation_range = args.eval_nia_s3_trans_range
        p = args.eval_nia_s3_particles or 'default'
        it = args.eval_nia_s3_iterations or 'default'
        rot = args.eval_nia_s3_rot_range or 'default'
        trans = args.eval_nia_s3_trans_range or 'default'
        print(f"NIA Stage 3 refinement enabled at inference (particles={p}, iterations={it}, rot_range={rot}, trans_range={trans})")

    # Optional: diffusion refinement at inference (experimental, off by default)
    evaluator.use_diffusion_refinement = args.use_diffusion_refinement
    evaluator.diffusion_noise_level = args.diffusion_noise_level
    evaluator.diffusion_refine_steps = args.diffusion_refine_steps
    if args.use_diffusion_refinement:
        print(f"[EXPERIMENTAL] Diffusion refinement enabled: noise_level={args.diffusion_noise_level}, steps={args.diffusion_refine_steps}")

    # Optional: Sinkhorn test-time optimization at inference
    evaluator.use_sinkhorn_tto = args.use_sinkhorn_tto
    evaluator.sinkhorn_tto_steps = args.sinkhorn_tto_steps
    evaluator.sinkhorn_tto_lr = args.sinkhorn_tto_lr
    evaluator.sinkhorn_tto_lambda_ = args.sinkhorn_tto_lambda
    evaluator.sinkhorn_tto_sinkhorn_eps = args.sinkhorn_tto_sinkhorn_eps
    evaluator.sinkhorn_tto_sinkhorn_iters = args.sinkhorn_tto_sinkhorn_iters
    if args.use_sinkhorn_tto:
        print(f"Sinkhorn per-point TTO enabled: steps={args.sinkhorn_tto_steps}, lr={args.sinkhorn_tto_lr}, lambda={args.sinkhorn_tto_lambda}, eps={args.sinkhorn_tto_sinkhorn_eps}, iters={args.sinkhorn_tto_sinkhorn_iters}")

    # Optional: Sinkhorn rigid TTO at inference (refines R, t)
    evaluator.use_sinkhorn_tto_rigid = args.use_sinkhorn_tto_rigid
    evaluator.sinkhorn_tto_rigid_steps = args.sinkhorn_tto_rigid_steps
    evaluator.sinkhorn_tto_rigid_lr = args.sinkhorn_tto_rigid_lr
    evaluator.sinkhorn_tto_rigid_lambda_ = args.sinkhorn_tto_rigid_lambda
    evaluator.sinkhorn_tto_rigid_sinkhorn_eps = args.sinkhorn_tto_rigid_sinkhorn_eps
    evaluator.sinkhorn_tto_rigid_sinkhorn_iters = args.sinkhorn_tto_rigid_sinkhorn_iters
    if args.use_sinkhorn_tto_rigid:
        print(f"Sinkhorn rigid TTO enabled: steps={args.sinkhorn_tto_rigid_steps}, lr={args.sinkhorn_tto_rigid_lr}, lambda={args.sinkhorn_tto_rigid_lambda}, eps={args.sinkhorn_tto_rigid_sinkhorn_eps}, iters={args.sinkhorn_tto_rigid_sinkhorn_iters}")

    evaluator.use_translation_only_icp = args.use_translation_only_icp
    evaluator.translation_only_icp_steps = args.translation_only_icp_steps
    evaluator.translation_only_icp_tau = args.translation_only_icp_tau
    if args.use_translation_only_icp:
        print(
            "Translation-only ICP enabled: "
            f"steps={args.translation_only_icp_steps}, tau={args.translation_only_icp_tau}"
        )
    evaluator.use_kabsch = args.use_kabsch
    if args.use_kabsch:
        print("Kabsch rigid summary enabled for EvoReg-family R/t metrics")
    # Runs evaluation based on dataset choice
    if args.dataset in ['faust', 'modelnet40', 'atlasnet', '3dmatch', 'shapenet']:
        print("=" * 80)
        print("BASELINE MODEL EVALUATION ON SAMPLE DATA")
        print("=" * 80)
        print(f"Device: {args.device}")
        print(f"directory: {args.dir}")
        if args.downsample_source:
            print(f"Source downsampling: {args.downsample_source} points")
        if args.downsample_target:
            print(f"Target downsampling: {args.downsample_target} points")
        print(f"Models to evaluate: {', '.join(args.models)}")
        print(f"Recall threshold: {args.recall_threshold}")
        print("=" * 80)

        # Stores all results
        all_results = {
            'config': config_dict,
            'git_commit': git_commit,
            'hostname': hostname,
            'start_time': start_time,
            'end_time': None,
            'evaluation_config': {
                'dataset': args.dataset,
                'dir': args.dir,
                'downsample_source': args.downsample_source,
                'downsample_target': args.downsample_target,
                'device': args.device
            },
            'gpu_info': gpu_info,
            'models': {}
        }

        # Evaluates each model
        for model_name in args.models:
            print(f"\n{'='*80}")
            print(f"Evaluating {model_name.upper()}")
            print('='*80)

            # Loads model
            model = evaluator.load_model(model_name, args.model_checkpoint)
            #--dataset faust --dir FAUST/MPI-FAUST/training/registrations --model_checkpoint /path/to/EvoReg_Point_Set_Registration/checkpoints/FAUST_checkpoints_No_Weighted_Loss/best_model.pth
            if model is None:
                print(f"Skipping {model_name} (failed to load)")
                continue

            # Evaluates on Sample Data
            if args.dataset == 'faust':
                opt = atlasnet_parser()
                opt.device = evaluator.device
                opt.batch_size = 1
                opt.batch_size_test = 1
                results = evaluator.evaluate_on_sample_data(
                    model_name, model, args,
                    dir=args.dir,
                    opt=opt,
                    downsample_source=args.downsample_source,
                    downsample_target=args.downsample_target
                )
            elif model_name.lower() == 'atlasnet' and args.dataset == 'atlasnet':
                results = evaluator.evaluate_on_shapenet_data(
                    model_name, model,
                    dir=args.dir,
                    opt=evaluator.opt,
                )
            else:
                opt = atlasnet_parser()
                opt.device = evaluator.device
                opt.batch_size = 1
                opt.batch_size_test = 1
                results = evaluator.evaluate_on_sample_data(
                    model_name, model, args,
                    dir=args.dir,
                    opt = opt,
                    downsample_source=args.downsample_source,
                    downsample_target=args.downsample_target
                )

            # Stores results
            all_results['models'][model_name] = results

            # Prints summary
            print(f"\n{model_name.upper()} Results:")
            print(f"  Success Rate: {results['success_rate']*100:.1f}%")
            print(f"  Chamfer Distance: {results['chamfer_distance']['mean']:.6f} ± {results['chamfer_distance']['std']:.6f}")
            print(f"  Registration Error: {results['registration_error']['mean']:.6f} ± {results['registration_error']['std']:.6f}")

            # Prints additional metrics if available
            if 'earth_movers_distance' in results and not np.isnan(results['earth_movers_distance']['mean']):
                print(f"  Earth Mover's Distance: {results['earth_movers_distance']['mean']:.6f} ± {results['earth_movers_distance']['std']:.6f}")
            if 'sliced_wasserstein_distance' in results and not np.isnan(results['sliced_wasserstein_distance']['mean']):
                print(f"  Sliced Wasserstein Distance: {results['sliced_wasserstein_distance']['mean']:.6f} ± {results['sliced_wasserstein_distance']['std']:.6f}")
            if 'registration_recall' in results and not np.isnan(results['registration_recall']['mean']):
                print(f"  Registration Recall: {results['registration_recall']['mean']*100:.1f}%")
            if 'correspondence_error' in results and not np.isnan(results['correspondence_error']['mean']):
                print(f"  Correspondence Error: {results['correspondence_error']['mean']:.6f} ± {results['correspondence_error']['std']:.6f}")
            if 'geodesic_distance' in results and not np.isnan(results['geodesic_distance']['mean']):
                print(f"  Geodesic Distance: {results['geodesic_distance']['mean']:.6f} ± {results['geodesic_distance']['std']:.6f}")
            if 'rotation_error' in results and not np.isnan(results['rotation_error']['mean']):
                print(f"  Rotation Error: {results['rotation_error']['mean']:.6f}° ± {results['rotation_error']['std']:.6f}°")
            if 'translation_error' in results and not np.isnan(results['translation_error']['mean']):
                print(f"  Translation Error: {results['translation_error']['mean']:.6f} ± {results['translation_error']['std']:.6f}")
            if 'inference_time' in results and not np.isnan(results['inference_time']['mean']):
                print(f"  Inference Time: {results['inference_time']['mean']:.6f}s ± {results['inference_time']['std']:.6f}s")

        # Saves results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = evaluator.results_dir / f'{args.dataset}_baseline_results_{timestamp}.json'
        all_results['end_time'] = datetime.now().isoformat()

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Results saved to: {results_file}")
        print('='*80)

        # Prints comparison table
        evaluator.print_comparison_table(all_results)

    else:
        # Runs evaluation on synthetic data
        results = evaluator.run_baseline_comparison(
            models_to_evaluate=args.models,
            n_samples=args.n_samples,
            n_points=args.n_points,
            save_results=True,
            config=config_dict,
            git_commit=git_commit,
            hostname=hostname,
            start_time=start_time,
            gpu_info=gpu_info
        )

        # Prints comparison table
        evaluator.print_comparison_table(results)

    print("\nBaseline evaluation completed!")


if __name__ == "__main__":
    main()
    #--models pointnetlk --dataset faust --dir FAUST/MPI-FAUST/training/registrations --model_checkpoint /path/to/checkpoints/best_model.pth
