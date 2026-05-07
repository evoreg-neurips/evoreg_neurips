"""
Complete EvoReg model for point cloud registration.

Bundles the multi-stage coarse-to-fine architecture: a learned point-cloud
encoder, iterative Sinkhorn correspondences with confidence-weighted Kabsch
refinement, a residual MLP rigid head, and a conditional VAE that predicts a
residual non-rigid deformation field. A score-based diffusion module is
jointly trained as an auxiliary objective.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .pointnet_encoder import PointNetEncoder
from .vae_encoder import VAEEncoder
from .generator import PointCloudGenerator
from .cross_attention_generator import CrossAttentionGenerator
from .score_network import ScoreNetwork
from .diffusion import DiffusionProcess
from .rigid_head import RigidHead, apply_rigid_transform
# GeometricCrossAttentionModule imported lazily below (abandoned experiment)


class EvoReg(nn.Module):
    """
    EvoReg: Versatile and Robust Point Cloud Registration via Multi-Stage Alignment.

    Combines all stages of the EvoReg pipeline:
    - Point-cloud encoder for feature extraction
    - Iterative Sinkhorn soft correspondences + confidence-weighted Kabsch
    - Residual MLP head correcting remaining rigid error
    - Conditional VAE generator for residual non-rigid deformation
    - Score-based diffusion module (auxiliary training objective)
    """
    
    def __init__(
        self,
        point_dim: int = 3,
        feature_dim: int = 512,
        latent_dim: int = 128,
        encoder_hidden_dims: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        vae_hidden_dims: Tuple[int, ...] = (512, 256),
        generator_hidden_dims: Tuple[int, ...] = (256, 512, 512, 256),
        share_point_encoder: bool = True,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        use_rigid_head: bool = False,
        rigid_head_hidden_dim: int = 256,
        generator_type: str = 'mlp',
        ca_feature_dim: int = 64,
        ca_k: int = 8,
    ):
        """
        Initializes the complete EvoReg model.

        Args:
            point_dim: Dimension of points (3 for 3D)
            feature_dim: Dimension of point cloud features
            latent_dim: Dimension of latent space
            encoder_hidden_dims: Hidden dimensions for PointNet encoder
            vae_hidden_dims: Hidden dimensions for VAE encoder
            generator_hidden_dims: Hidden dimensions for generator
            share_point_encoder: Whether to share encoder for source/target
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            use_rigid_head: Whether to use rigid SE(3) head for rigid alignment
            rigid_head_hidden_dim: Hidden dimension for rigid head MLP
            generator_type: Which generator to use ('mlp', 'attention', or
                'cross_attention'). 'cross_attention' uses VFA-style kNN
                attention directly between source and target; the VAE still
                runs and contributes a KL term for regularisation.
            ca_feature_dim: Feature projection dimension for cross-attention
                generator (only used when generator_type='cross_attention').
            ca_k: Number of nearest neighbours in target to attend over
                (only used when generator_type='cross_attention').
        """
        super(EvoReg, self).__init__()

        self.point_dim = point_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.use_rigid_head = use_rigid_head
        self.generator_type = generator_type
        
        # Creates PointNet encoder for source
        self.source_encoder = PointNetEncoder(
            input_dim=point_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=feature_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
        
        # Shares encoder or creates separate one for target
        if share_point_encoder:
            self.target_encoder = self.source_encoder
        else:
            self.target_encoder = PointNetEncoder(
                input_dim=point_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=feature_dim,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
        
        # Creates VAE encoder for latent distribution
        self.vae_encoder = VAEEncoder(
            input_dim=feature_dim * 2,  # Concatenated features
            hidden_dims=vae_hidden_dims,
            latent_dim=latent_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
        
        # Creates generator for point deformation
        if generator_type == 'cross_attention':
            self.generator = CrossAttentionGenerator(
                feature_dim=ca_feature_dim,
                k=ca_k,
                use_residual=True,
            )
        else:
            self.generator = PointCloudGenerator(
                latent_dim=latent_dim,
                point_dim=point_dim,
                hidden_dims=generator_hidden_dims,
                use_residual=True,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )

        # Creates rigid head if enabled (for SE(3) alignment)
        if use_rigid_head:
            self.rigid_head = RigidHead(
                feat_dim=feature_dim * 2,  # Concatenated source + target features
                hidden_dim=rigid_head_hidden_dim
            )
        else:
            self.rigid_head = None
    
    def encode(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes source and target point clouds to latent distribution.
        
        Args:
            source: Source point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
        
        Returns:
            Tuple of (mu, log_var) for latent distribution
        """
        # Extracts features from both point clouds
        source_features = self.source_encoder(source)
        target_features = self.target_encoder(target)
        
        # Concatenates features
        combined_features = torch.cat([source_features, target_features], dim=-1)
        
        # Encodes to latent distribution
        mu, log_var = self.vae_encoder(combined_features)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        
        Returns:
            Sampled latent code
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def decode(
        self,
        source: torch.Tensor,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decodes latent code to generate deformed point cloud.

        Args:
            source: Source point cloud (B, N, 3)
            z: Latent deformation code (B, latent_dim) — used by 'mlp' and
               'attention' generators; ignored by 'cross_attention'.
            target: Target point cloud (B, M, 3) — required when
               generator_type='cross_attention'.

        Returns:
            Deformed point cloud (B, N, 3)
        """
        if self.generator_type == 'cross_attention':
            if target is None:
                raise ValueError("target must be provided when using cross_attention generator")
            return self.generator(source, target)
        return self.generator(source, z)
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through EvoReg.

        Args:
            source: Source point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            return_latent: Whether to return latent codes

        Returns:
            Dictionary containing:
            - 'output': Registered/deformed point cloud
            - 'mu': Mean of latent distribution
            - 'log_var': Log variance of latent distribution
            - 'z': Sampled latent code (if return_latent=True)
            - 'R_pred': Predicted rotation (if using rigid head)
            - 't_pred': Predicted translation (if using rigid head)
            - 'source_rigid': Rigidly aligned source (if using rigid head)
            - 'delta': Displacement field (if using rigid head)
        """
        # Step 1: Extract features (needed for both rigid and non-rigid)
        source_features = self.source_encoder(source)
        target_features = self.target_encoder(target)
        combined_features = torch.cat([source_features, target_features], dim=-1)

        # Step 2: Rigid alignment (if enabled)
        if self.use_rigid_head:
            # Predict rigid transformation from features
            R_pred, t_pred = self.rigid_head(combined_features)

            # Apply rigid transformation to source
            source_rigid = apply_rigid_transform(source, R_pred, t_pred)

            # Use rigidly aligned source for non-rigid deformation
            source_for_deformation = source_rigid
        else:
            R_pred = None
            t_pred = None
            source_rigid = None
            source_for_deformation = source

        # Step 3: Non-rigid deformation (VAE + Generator)
        # Encode to latent distribution
        mu, log_var = self.vae_encoder(combined_features)

        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)

        # Generate deformed point cloud
        output = self.decode(source_for_deformation, z, target=target)

        # Calculate displacement field (for loss computation)
        if self.use_rigid_head:
            delta = output - source_rigid
        else:
            delta = output - source

        # Prepare output dictionary
        results = {
            'output': output,
            'mu': mu,
            'log_var': log_var
        }

        if return_latent:
            results['z'] = z

        if self.use_rigid_head:
            results['R_pred'] = R_pred
            results['t_pred'] = t_pred
            results['source_rigid'] = source_rigid
            results['delta'] = delta

        return results
    
    def register(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Performs point cloud registration (inference mode).
        
        Args:
            source: Source point cloud to register
            target: Target point cloud to match
            n_samples: Number of samples to average (for uncertainty)
        
        Returns:
            Registered point cloud
        """
        self.eval()  # Sets to evaluation mode
        
        with torch.no_grad():
            # Encodes to latent distribution
            mu, log_var = self.encode(source, target)
            
            if n_samples == 1:
                # Uses mean for deterministic output
                z = mu
            else:
                # Samples multiple times and averages
                outputs = []
                for _ in range(n_samples):
                    z = self.reparameterize(mu, log_var)
                    output = self.decode(source, z, target=target)
                    outputs.append(output)

                return torch.stack(outputs).mean(dim=0)

            # Generates registered point cloud
            output = self.decode(source, z, target=target)
            
        return output


class EvoRegWithLosses(nn.Module):
    """
    EvoReg model with integrated loss computation.
    
    Wraps the EvoReg model and includes loss functions for training.
    """
    
    def __init__(
        self,
        model: EvoReg,
        chamfer_weight: float = 1.0,
        kl_weight: float = 0.001,
        kl_annealing: bool = True
    ):
        """
        Initializes EvoReg with losses.
        
        Args:
            model: EvoReg model instance
            chamfer_weight: Weight for Chamfer distance loss
            kl_weight: Weight for KL divergence loss
            kl_annealing: Whether to use KL annealing during training
        """
        super(EvoRegWithLosses, self).__init__()
        
        self.model = model
        self.chamfer_weight = chamfer_weight
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.current_kl_weight = 0.0 if kl_annealing else kl_weight
        
        # Imports loss functions
        from ..losses import ChamferDistance, KLDivergenceLoss
        
        self.chamfer_loss = ChamferDistance(reduction='mean')
        self.kl_loss = KLDivergenceLoss(reduction='batchmean')
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            step: Current training step (for KL annealing)
        
        Returns:
            Dictionary with outputs and losses
        """
        # Gets model outputs
        outputs = self.model(source, target, return_latent=True)
        
        # Computes Chamfer distance loss
        chamfer = self.chamfer_loss(outputs['output'], target)
        
        # Computes KL divergence loss
        kl = self.kl_loss(outputs['mu'], outputs['log_var'])
        
        # Updates KL weight if using annealing
        if self.kl_annealing and step is not None:
            # Linear annealing over first 10000 steps
            self.current_kl_weight = min(self.kl_weight, self.kl_weight * step / 10000)
        else:
            self.current_kl_weight = self.kl_weight
        
        # Computes total loss
        total_loss = self.chamfer_weight * chamfer + self.current_kl_weight * kl
        
        # Adds losses to output dictionary
        outputs['loss'] = total_loss
        outputs['chamfer_loss'] = chamfer
        outputs['kl_loss'] = kl
        outputs['weighted_chamfer'] = self.chamfer_weight * chamfer
        outputs['weighted_kl'] = self.current_kl_weight * kl
        
        return outputs


class EvoRegWithDiffusion(nn.Module):
    """
    Complete EvoReg model with VAE and Diffusion.
    
    Combines VAE for initial registration with diffusion for
    iterative refinement.
    """
    
    def __init__(
        self,
        vae_model: EvoReg,
        score_hidden_dim: int = 256,
        n_steps: int = 1000,
        schedule_type: str = 'linear',
        chamfer_weight: float = 1.0,
        kl_weight: float = 0.001,
        diffusion_weight: float = 1.0
    ):
        """
        Initializes EvoReg with diffusion.
        
        Args:
            vae_model: Base VAE model
            score_hidden_dim: Hidden dimension for score network
            n_steps: Number of diffusion timesteps
            schedule_type: Noise schedule type
            chamfer_weight: Weight for Chamfer loss
            kl_weight: Weight for KL loss
            diffusion_weight: Weight for diffusion loss
        """
        super(EvoRegWithDiffusion, self).__init__()
        
        self.vae_model = vae_model
        self.chamfer_weight = chamfer_weight
        self.kl_weight = kl_weight
        self.diffusion_weight = diffusion_weight
        
        # Creates score network
        self.score_network = ScoreNetwork(
            point_dim=vae_model.point_dim,
            hidden_dim=score_hidden_dim,
            n_layers=4
        )
        
        # Creates diffusion process
        self.diffusion = DiffusionProcess(
            score_network=self.score_network,
            n_timesteps=n_steps,
            schedule_type=schedule_type
        )
        
        # Loss functions
        from ..losses import ChamferDistance, KLDivergenceLoss
        self.chamfer_loss = ChamferDistance(reduction='mean')
        self.kl_loss = KLDivergenceLoss(reduction='batchmean')
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_latent: bool = False,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with VAE/C2F and diffusion training.

        Passes through ALL inner model outputs (R_pred, t_pred, source_rigid,
        etc. for c2f) and adds diffusion_loss + noise_pred_error.  Loss
        aggregation is left to the trainer so that KL annealing, rigid losses,
        and other terms are handled in one place.

        Args:
            source: Source point cloud
            target: Target point cloud
            return_latent: Forwarded to inner model
            return_components: Kept for backward compat (treated as return_latent)

        Returns:
            Dictionary with all inner model outputs plus diffusion losses
        """
        # Inner model forward (works for both EvoReg and EvoRegCoarseToFine)
        vae_outputs = self.vae_model(source, target, return_latent=(return_latent or return_components))

        # Diffusion score-matching loss on the VAE/generator output
        diffusion_losses = self.diffusion.compute_loss(
            vae_outputs['output'],
            target
        )

        # Pass through ALL inner model outputs, then add diffusion fields
        results = dict(vae_outputs)
        results['diffusion_loss'] = diffusion_losses['diffusion_loss']
        results['noise_pred_error'] = diffusion_losses['noise_pred_error']
        return results
    
    def register(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        use_diffusion: bool = True,
        diffusion_steps: int = 50,
        noise_level: float = 0.1
    ) -> torch.Tensor:
        """
        Performs registration with optional diffusion refinement.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            use_diffusion: Whether to use diffusion refinement
            diffusion_steps: Number of diffusion steps
            noise_level: Initial noise level for refinement
        
        Returns:
            Registered point cloud
        """
        self.eval()
        
        with torch.no_grad():
            # VAE registration
            vae_output = self.vae_model.register(source, target)
            
            if not use_diffusion:
                return vae_output
            
            # Diffusion refinement
            refined = self.diffusion.refine(
                vae_output,
                target,
                num_steps=diffusion_steps,
                noise_level=noise_level,
                use_ddim=True
            )
            
            return refined


class EvoRegGeometric(nn.Module):
    """
    EvoReg variant with Geometric Cross-Attention in the encoder.

    Replaces independent PointNet encoding with a paired encoding scheme:
      1. Shared per-point MLP extracts initial features from each cloud.
      2. Geometric cross-attention enriches features with cross-cloud
         awareness and transformation-invariant distance embeddings.
      3. Max-pooled global features are fed to the same VAE + generator.

    This addresses two key limitations of the base EvoReg encoder:
      - Source and target are encoded independently (no cross-interaction).
      - Raw coordinates are not invariant to rigid transforms.
    """

    def __init__(
        self,
        point_dim: int = 3,
        feature_dim: int = 512,
        latent_dim: int = 128,
        initial_mlp_dims: Tuple[int, ...] = (64, 128, 256),
        final_mlp_dims: Tuple[int, ...] = (512, 1024),
        num_heads: int = 4,
        num_attn_blocks: int = 3,
        num_rbf: int = 16,
        rbf_cutoff: float = 2.0,
        vae_hidden_dims: Tuple[int, ...] = (512, 256),
        generator_hidden_dims: Tuple[int, ...] = (256, 512, 512, 256),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        use_rigid_head: bool = False,
        rigid_head_hidden_dim: int = 256,
    ):
        super().__init__()

        self.point_dim = point_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.use_rigid_head = use_rigid_head

        attn_dim = initial_mlp_dims[-1]

        # ----- Initial per-point MLP (shared for source and target) -----
        init_layers = []
        in_dim = point_dim
        for h_dim in initial_mlp_dims:
            init_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                init_layers.append(nn.BatchNorm1d(h_dim))
            init_layers.append(nn.ReLU())
            in_dim = h_dim
        self.initial_mlp = nn.Sequential(*init_layers)

        # ----- Geometric cross-attention -----
        from .geometric_attention import GeometricCrossAttentionModule
        self.geo_cross_attn = GeometricCrossAttentionModule(
            dim=attn_dim,
            num_heads=num_heads,
            num_blocks=num_attn_blocks,
            num_rbf=num_rbf,
            cutoff=rbf_cutoff,
            dropout=dropout_rate,
        )

        # ----- Final per-point MLP (shared, after attention) -----
        final_layers = []
        in_dim = attn_dim
        for h_dim in final_mlp_dims:
            final_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                final_layers.append(nn.BatchNorm1d(h_dim))
            final_layers.append(nn.ReLU())
            in_dim = h_dim
        self.final_mlp = nn.Sequential(*final_layers)

        # ----- Global feature MLP (after max-pool) -----
        self.global_mlp = nn.Sequential(
            nn.Linear(final_mlp_dims[-1], feature_dim),
            nn.BatchNorm1d(feature_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # ----- VAE encoder -----
        self.vae_encoder = VAEEncoder(
            input_dim=feature_dim * 2,
            hidden_dims=vae_hidden_dims,
            latent_dim=latent_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        # ----- Generator (same as base EvoReg) -----
        self.generator = PointCloudGenerator(
            latent_dim=latent_dim,
            point_dim=point_dim,
            hidden_dims=generator_hidden_dims,
            use_residual=True,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        # ----- Rigid head (optional) -----
        if use_rigid_head:
            self.rigid_head = RigidHead(
                feat_dim=feature_dim * 2,
                hidden_dim=rigid_head_hidden_dim,
            )
        else:
            self.rigid_head = None

    # -- helpers ----------------------------------------------------------

    def _pointwise_mlp(self, mlp: nn.Sequential, points: torch.Tensor) -> torch.Tensor:
        """Run a shared MLP on each point independently.

        Reshapes (B, N, D_in) -> (B*N, D_in) for BatchNorm compatibility,
        then back to (B, N, D_out).
        """
        B, N, D = points.shape
        flat = points.reshape(-1, D)
        out = mlp(flat)
        return out.reshape(B, N, -1)

    def _pool_global(self, feats: torch.Tensor) -> torch.Tensor:
        """Max-pool per-point features and run global MLP.

        Args:
            feats: (B, N, D) per-point features.
        Returns:
            (B, feature_dim) global feature.
        """
        global_feat, _ = torch.max(feats, dim=1)  # (B, D)
        # Handle batch_size=1 with BatchNorm
        if global_feat.shape[0] == 1 and self.training:
            self.global_mlp.eval()
            out = self.global_mlp(global_feat)
            self.global_mlp.train()
            return out
        return self.global_mlp(global_feat)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, source: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.generator(source, z)

    # -- main interface ---------------------------------------------------

    def _encode_paired(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Paired encoding with geometric cross-attention.

        Returns:
            (src_global, tgt_global) each (B, feature_dim).
        """
        # Per-point features (shared MLP)
        src_feats = self._pointwise_mlp(self.initial_mlp, source)  # (B, N, attn_dim)
        tgt_feats = self._pointwise_mlp(self.initial_mlp, target)  # (B, M, attn_dim)

        # Geometric cross-attention
        src_feats, tgt_feats = self.geo_cross_attn(
            src_feats, tgt_feats, source, target
        )

        # Finish MLP + pool
        src_feats = self._pointwise_mlp(self.final_mlp, src_feats)
        tgt_feats = self._pointwise_mlp(self.final_mlp, tgt_feats)

        src_global = self._pool_global(src_feats)
        tgt_global = self._pool_global(tgt_feats)
        return src_global, tgt_global

    def encode(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode source and target to latent distribution parameters."""
        src_global, tgt_global = self._encode_paired(source, target)
        combined = torch.cat([src_global, tgt_global], dim=-1)
        mu, log_var = self.vae_encoder(combined)
        return mu, log_var

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass (same return dict interface as EvoReg)."""
        src_global, tgt_global = self._encode_paired(source, target)
        combined = torch.cat([src_global, tgt_global], dim=-1)

        # Rigid alignment (optional)
        if self.use_rigid_head:
            R_pred, t_pred = self.rigid_head(combined)
            source_rigid = apply_rigid_transform(source, R_pred, t_pred)
            source_for_deformation = source_rigid
        else:
            R_pred = t_pred = source_rigid = None
            source_for_deformation = source

        # VAE
        mu, log_var = self.vae_encoder(combined)
        z = self.reparameterize(mu, log_var)

        # Generator
        output = self.decode(source_for_deformation, z)

        # Build result dict (matches EvoReg interface exactly)
        results: Dict[str, torch.Tensor] = {
            "output": output,
            "mu": mu,
            "log_var": log_var,
        }
        if return_latent:
            results["z"] = z
        if self.use_rigid_head:
            delta = output - source_rigid
            results.update(
                {
                    "R_pred": R_pred,
                    "t_pred": t_pred,
                    "source_rigid": source_rigid,
                    "delta": delta,
                }
            )
        return results

    def register(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Inference-time registration (matches EvoReg.register interface)."""
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encode(source, target)
            if n_samples == 1:
                z = mu
            else:
                outputs = []
                for _ in range(n_samples):
                    z = self.reparameterize(mu, log_var)
                    outputs.append(self.decode(source, z))
                return torch.stack(outputs).mean(dim=0)
            output = self.decode(source, z)
        return output


def create_evoreg(
    latent_dim: int = 128,
    feature_dim: int = 512,
    with_losses: bool = False,
    with_diffusion: bool = False,
    use_rigid_head: bool = False,
    rigid_head_hidden_dim: int = 256,
    generator_type: str = 'mlp',
    ca_feature_dim: int = 64,
    ca_k: int = 8,
    use_geometric_attention: bool = False,
    geo_num_heads: int = 4,
    geo_num_blocks: int = 3,
    geo_num_rbf: int = 16,
    geo_rbf_cutoff: float = 2.0,
    use_coarse_to_fine: bool = False,
    corr_proj_dim: int = 256,
    corr_temperature: float = 0.1,
    use_dual_softmax: bool = True,
    n_sinkhorn_iters: int = 3,
    n_svd_iterations: int = 3,
    n_stage2a_iterations: int = 1,
    use_local_features: bool = False,
    local_k: int = 20,
    use_geo_consistency: bool = False,
    geo_consistency_alpha: float = 5.0,
    use_pso: bool = False,
    nia_type: str = 'pso',
    pso_particles: int = 50,
    pso_iterations: int = 30,
    fitness_subsample: int = 128,
    use_inter_stage_nia: bool = False,
    inter_stage_nia_particles: int = 25,
    inter_stage_nia_iterations: int = 15,
    inter_stage_nia_rot_s1: float = 20.0,
    inter_stage_nia_trans_s1: float = 0.5,
    inter_stage_nia_rot_s2: float = 10.0,
    inter_stage_nia_trans_s2: float = 0.2,
    inter_stage_nia_rot_s3: float = 5.0,
    inter_stage_nia_trans_s3: float = 0.1,
    no_stage0: bool = False,
    no_stage1: bool = False,
    no_stage2: bool = False,
    no_stage3: bool = False,
    use_control_points: bool = False,
    n_control_points: int = 128,
    rbf_sigma: float = 0.2,
    diffusion_weight: float = 1.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create EvoReg model.

    Args:
        latent_dim: Dimension of latent space
        feature_dim: Dimension of features
        with_losses: Whether to include loss computation
        with_diffusion: Whether to include diffusion module
        use_rigid_head: Whether to use rigid SE(3) head for rigid alignment
        rigid_head_hidden_dim: Hidden dimension for rigid head MLP
        use_geometric_attention: Whether to use geometric cross-attention encoder
        geo_num_heads: Number of attention heads (geometric variant)
        geo_num_blocks: Number of cross-attention blocks (geometric variant)
        geo_num_rbf: Number of Gaussian RBF centres for distance encoding
        geo_rbf_cutoff: RBF cutoff distance (should cover point cloud diameter)
        use_coarse_to_fine: Whether to use coarse-to-fine registration (SVD + residual rigid + VAE)
        corr_proj_dim: Projection dimension for soft correspondences
        corr_temperature: Temperature for correspondence softmax
        use_dual_softmax: Whether to use Sinkhorn normalization
        n_sinkhorn_iters: Number of Sinkhorn iterations
        generator_type: Generator variant — 'mlp', 'attention', or
            'cross_attention' (VFA-style kNN cross-attention).
        ca_feature_dim: Feature dim for cross-attention generator.
        ca_k: kNN count for cross-attention generator.
        **kwargs: Additional arguments for EvoReg

    Returns:
        EvoReg model (with optional losses and/or diffusion)
    """
    # Separates diffusion kwargs
    diffusion_kwargs = {}
    vae_kwargs = {}

    for key, value in kwargs.items():
        if key.startswith('diffusion_'):
            diffusion_kwargs[key.replace('diffusion_', '')] = value
        else:
            vae_kwargs[key] = value

    # Creates base model
    if use_coarse_to_fine:
        from .coarse_to_fine import EvoRegCoarseToFine
        model = EvoRegCoarseToFine(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            use_batch_norm=vae_kwargs.pop('use_batch_norm', True),
            dropout_rate=vae_kwargs.pop('dropout_rate', 0.1),
            corr_proj_dim=corr_proj_dim,
            corr_temperature=corr_temperature,
            use_dual_softmax=use_dual_softmax,
            n_sinkhorn_iters=n_sinkhorn_iters,
            rigid_hidden_dim=rigid_head_hidden_dim,
            n_svd_iterations=n_svd_iterations,
            n_stage2a_iterations=n_stage2a_iterations,
            use_local_features=use_local_features,
            local_k=local_k,
            use_geo_consistency=use_geo_consistency,
            geo_consistency_alpha=geo_consistency_alpha,
            use_pso=use_pso,
            nia_type=nia_type,
            pso_particles=pso_particles,
            pso_iterations=pso_iterations,
            fitness_subsample=fitness_subsample,
            use_inter_stage_nia=use_inter_stage_nia,
            inter_stage_nia_particles=inter_stage_nia_particles,
            inter_stage_nia_iterations=inter_stage_nia_iterations,
            inter_stage_nia_rot_s1=inter_stage_nia_rot_s1,
            inter_stage_nia_trans_s1=inter_stage_nia_trans_s1,
            inter_stage_nia_rot_s2=inter_stage_nia_rot_s2,
            inter_stage_nia_trans_s2=inter_stage_nia_trans_s2,
            inter_stage_nia_rot_s3=inter_stage_nia_rot_s3,
            inter_stage_nia_trans_s3=inter_stage_nia_trans_s3,
            no_stage0=no_stage0,
            no_stage1=no_stage1,
            no_stage2=no_stage2,
            no_stage3=no_stage3,
        )
    elif use_geometric_attention:
        model = EvoRegGeometric(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            use_rigid_head=use_rigid_head,
            rigid_head_hidden_dim=rigid_head_hidden_dim,
            num_heads=geo_num_heads,
            num_attn_blocks=geo_num_blocks,
            num_rbf=geo_num_rbf,
            rbf_cutoff=geo_rbf_cutoff,
            dropout_rate=vae_kwargs.pop('dropout_rate', 0.1),
        )
    else:
        model = EvoReg(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            use_rigid_head=use_rigid_head,
            rigid_head_hidden_dim=rigid_head_hidden_dim,
            generator_type=generator_type,
            ca_feature_dim=ca_feature_dim,
            ca_k=ca_k,
            **vae_kwargs
        )

    # Adds diffusion if requested
    if with_diffusion:
        model = EvoRegWithDiffusion(model, diffusion_weight=diffusion_weight, **diffusion_kwargs)
    elif with_losses:
        model = EvoRegWithLosses(model)

    return model


if __name__ == "__main__":
    # Tests the complete EvoReg model
    print("Testing complete EvoReg model...")
    
    # Creates model
    model = create_evoreg(
        latent_dim=128,
        feature_dim=512,
        with_losses=False
    )
    
    # Creates test data
    batch_size = 2
    n_points = 1000
    source = torch.randn(batch_size, n_points, 3)
    target = torch.randn(batch_size, n_points, 3)
    
    # Tests forward pass
    print("\n1. Testing forward pass:")
    outputs = model(source, target, return_latent=True)
    
    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Output shape: {outputs['output'].shape}")
    print(f"Mu shape: {outputs['mu'].shape}")
    print(f"Log var shape: {outputs['log_var'].shape}")
    print(f"Latent z shape: {outputs['z'].shape}")
    
    # Tests registration (inference)
    print("\n2. Testing registration (inference):")
    registered = model.register(source, target)
    print(f"Registered shape: {registered.shape}")
    
    # Tests with losses
    print("\n3. Testing model with losses:")
    model_with_losses = create_evoreg(with_losses=True)
    
    outputs_with_losses = model_with_losses(source, target, step=100)
    print(f"Total loss: {outputs_with_losses['loss'].item():.4f}")
    print(f"Chamfer loss: {outputs_with_losses['chamfer_loss'].item():.4f}")
    print(f"KL loss: {outputs_with_losses['kl_loss'].item():.4f}")
    
    # Tests gradient flow
    loss = outputs_with_losses['loss']
    loss.backward()
    print("\nGradient check passed!")
    
    # Counts parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal EvoReg parameters: {n_params:,}")
    
    # Tests with single sample
    print("\n4. Testing single sample:")
    single_source = torch.randn(500, 3)
    single_target = torch.randn(600, 3)
    single_output = model.register(
        single_source.unsqueeze(0),
        single_target.unsqueeze(0)
    )
    print(f"Single registration shape: {single_output.shape}")
    
    # Tests VAE + Diffusion model
    print("\n5. Testing EvoReg with Diffusion:")
    model_with_diffusion = create_evoreg(
        latent_dim=128,
        feature_dim=512,
        with_diffusion=True,
        diffusion_n_diffusion_steps=100
    )
    
    outputs_diffusion = model_with_diffusion(source, target)
    print(f"Total loss: {outputs_diffusion['loss'].item():.4f}")
    print(f"VAE Chamfer loss: {outputs_diffusion['chamfer_loss'].item():.4f}")
    print(f"KL loss: {outputs_diffusion['kl_loss'].item():.4f}")
    print(f"Diffusion loss: {outputs_diffusion['diffusion_loss'].item():.4f}")
    
    # Tests inference with diffusion
    print("\n6. Testing registration with diffusion refinement:")
    refined = model_with_diffusion.register(
        source,
        target,
        use_diffusion=True,
        diffusion_steps=10
    )
    print(f"Refined output shape: {refined.shape}")
    
    print("\nAll EvoReg tests passed!")