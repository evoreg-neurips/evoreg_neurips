"""
Coarse-to-fine registration model for EvoReg.

Stage 1:  Per-point features -> soft correspondences -> differentiable SVD -> R1, t1
Stage 2:  Re-encode aligned pair -> residual rigid head -> R2, t2
Stage 3:  VAE -> generator -> small non-rigid residual displacements
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .pointnet_encoder import PointNetEncoder
from .vae_encoder import VAEEncoder
from .generator import PointCloudGenerator
from .rigid_head import RigidHead, apply_rigid_transform
from .soft_correspondence import SoftCorrespondenceModule, DifferentiableKabsch
from .pso_alignment import PSOAlignment
from .de_alignment import DEAlignment
from .cmaes_alignment import CMAESAlignment
from .firefly_alignment import FireflyAlignment
from .hybrid_alignment import PSODEAlignment, DECMAESAlignment, PSOCMAESAlignment
from .gwo_alignment import GWOAlignment


class EvoRegCoarseToFine(nn.Module):
    """
    Coarse-to-fine point cloud registration model.

    Combines SVD-based rigid alignment (from learned soft correspondences)
    with VAE-based non-rigid deformation refinement.
    """

    def __init__(
        self,
        point_dim: int = 3,
        feature_dim: int = 512,
        latent_dim: int = 128,
        encoder_hidden_dims: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        vae_hidden_dims: Tuple[int, ...] = (512, 256),
        generator_hidden_dims: Tuple[int, ...] = (256, 512, 512, 256),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        # Soft correspondence params
        corr_proj_dim: int = 256,
        corr_temperature: float = 0.1,
        use_dual_softmax: bool = True,
        n_sinkhorn_iters: int = 3,
        # Residual rigid head
        rigid_hidden_dim: int = 256,
        # Iterative SVD
        n_svd_iterations: int = 3,
        # Iterative Stage 2 refinement
        n_stage2a_iterations: int = 1,
        # Local feature enrichment
        use_local_features: bool = False,
        local_k: int = 20,
        # Geometric consistency reweighting
        use_geo_consistency: bool = False,
        geo_consistency_alpha: float = 5.0,
        # NIA pre-alignment (PSO, DE, CMA-ES, Firefly)
        use_pso: bool = False,
        nia_type: str = 'pso',
        pso_particles: int = 50,
        pso_iterations: int = 30,
        fitness_subsample: int = 128,
        pso_rotation_range: float = 45.0,
        pso_translation_range: float = 1.0,
        # Inter-stage NIA refinement (progressive tightening)
        use_inter_stage_nia: bool = False,
        inter_stage_nia_particles: int = 25,
        inter_stage_nia_iterations: int = 15,
        inter_stage_nia_rot_s1: float = 20.0,   # after Stage 1
        inter_stage_nia_trans_s1: float = 0.5,
        inter_stage_nia_rot_s2: float = 10.0,   # after Stage 2
        inter_stage_nia_trans_s2: float = 0.2,
        inter_stage_nia_rot_s3: float = 5.0,    # after Stage 3 (inference only)
        inter_stage_nia_trans_s3: float = 0.1,
        # Stage ablation flags
        no_stage0: bool = False,
        no_stage1: bool = False,
        no_stage2: bool = False,
        no_stage3: bool = False,
    ):
        super().__init__()

        self.point_dim = point_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.n_svd_iterations = n_svd_iterations
        self.n_stage2a_iterations = n_stage2a_iterations
        self.use_local_features = use_local_features
        self.use_geo_consistency = use_geo_consistency
        self.geo_consistency_alpha = geo_consistency_alpha
        self.use_pso = use_pso if not no_stage0 else False
        self.nia_type = nia_type
        self.use_inter_stage_nia = use_inter_stage_nia
        self.no_stage1 = no_stage1
        self.no_stage2 = no_stage2
        self.no_stage3 = no_stage3

        if self.use_pso:
            nia_kwargs = dict(
                n_particles=pso_particles,
                n_iterations=pso_iterations,
                fitness_subsample=fitness_subsample,
                rotation_range=pso_rotation_range,
                translation_range=pso_translation_range,
            )
            nia_classes = {
                'pso': PSOAlignment,
                'de': DEAlignment,
                'cmaes': CMAESAlignment,
                'firefly': FireflyAlignment,
                'gwo': GWOAlignment,
                'pso_de': PSODEAlignment,
                'de_cmaes': DECMAESAlignment,
                'pso_cmaes': PSOCMAESAlignment,
            }
            if nia_type not in nia_classes:
                raise ValueError(f"Unknown NIA type: {nia_type}. Choose from: {list(nia_classes.keys())}")
            self.pso = nia_classes[nia_type](**nia_kwargs)

        if self.use_inter_stage_nia:
            nia_classes = {
                'pso': PSOAlignment, 'de': DEAlignment, 'cmaes': CMAESAlignment,
                'firefly': FireflyAlignment, 'gwo': GWOAlignment,
                'pso_de': PSODEAlignment, 'de_cmaes': DECMAESAlignment, 'pso_cmaes': PSOCMAESAlignment,
            }
            common = dict(n_particles=inter_stage_nia_particles, n_iterations=inter_stage_nia_iterations, fitness_subsample=fitness_subsample)
            self.inter_nia_s1 = nia_classes[nia_type](rotation_range=inter_stage_nia_rot_s1, translation_range=inter_stage_nia_trans_s1, **common)
            self.inter_nia_s2 = nia_classes[nia_type](rotation_range=inter_stage_nia_rot_s2, translation_range=inter_stage_nia_trans_s2, **common)
            self.inter_nia_s3 = nia_classes[nia_type](rotation_range=inter_stage_nia_rot_s3, translation_range=inter_stage_nia_trans_s3, **common)

        # Shared PointNet encoder (source and target)
        self.encoder = PointNetEncoder(
            input_dim=point_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=feature_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        # Per-point feature dim is the last hidden dim of the encoder
        per_point_dim = encoder_hidden_dims[-1]  # 1024

        # Local feature enrichment (k-NN edge convolution) — abandoned, kept for reference
        if use_local_features:
            from .local_feature import LocalFeatureEnrichment
            self.local_enrichment = LocalFeatureEnrichment(
                feat_dim=per_point_dim, k=local_k,
            )

        # Stage 1: Soft correspondence + SVD
        self.soft_correspondence = SoftCorrespondenceModule(
            feat_dim=per_point_dim,
            proj_dim=corr_proj_dim,
            temperature=corr_temperature,
            use_dual_softmax=use_dual_softmax,
            n_sinkhorn_iters=n_sinkhorn_iters,
        )
        self.kabsch = DifferentiableKabsch()

        # Stage 2: Residual rigid head
        self.residual_rigid_head = RigidHead(
            feat_dim=feature_dim * 2,
            hidden_dim=rigid_hidden_dim,
        )

        # Stage 3: VAE + Generator
        self.vae_encoder = VAEEncoder(
            input_dim=feature_dim * 2,
            hidden_dims=vae_hidden_dims,
            latent_dim=latent_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        self.generator = PointCloudGenerator(
            latent_dim=latent_dim,
            point_dim=point_dim,
            hidden_dims=generator_hidden_dims,
            use_residual=True,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full coarse-to-fine forward pass.

        Args:
            source: (B, N, 3) source point cloud
            target: (B, M, 3) target point cloud
            return_latent: whether to include z in results

        Returns:
            Dict with output, mu, log_var, R_pred, t_pred, source_rigid, etc.
        """
        # ===== Stage 0: PSO coarse pre-alignment (optional) =====
        B = source.shape[0]
        if self.use_pso:
            R_pso, t_pso, current_source = self.pso(source, target)
            R_acc = R_pso.clone()
            t_acc = t_pso.clone()
        else:
            R_acc = torch.eye(3, device=source.device).unsqueeze(0).expand(B, -1, -1).clone()
            t_acc = torch.zeros(B, 3, device=source.device)
            current_source = source

        # ===== Stage 1: Iterative SVD (correspond -> SVD -> align, N times) =====
        # Encode target ONCE (features reused across all iterations)
        tgt_global, tgt_point_feats = self.encoder.forward_with_point_features(target)
        if self.use_local_features:
            tgt_point_feats = self.local_enrichment(target, tgt_point_feats)

        intermediate_sources = []

        if not self.no_stage1:
            for _svd_iter in range(self.n_svd_iterations):
                src_global, src_point_feats = self.encoder.forward_with_point_features(current_source)
                if self.use_local_features:
                    src_point_feats = self.local_enrichment(current_source, src_point_feats)

                tgt_corr, confidence, assignment = self.soft_correspondence(
                    src_point_feats, tgt_point_feats, current_source, target,
                )

                # Geometric consistency reweighting: upweight correspondences
                # that preserve pairwise distances (inliers under rigid transform)
                if self.use_geo_consistency:
                    d_src = torch.cdist(current_source, current_source)  # (B, N, N)
                    d_tgt = torch.cdist(tgt_corr, tgt_corr)              # (B, N, N)
                    consistency = torch.exp(-self.geo_consistency_alpha * (d_src - d_tgt).abs())
                    geo_score = consistency.mean(dim=-1)  # (B, N)
                    weights = confidence * geo_score
                else:
                    weights = confidence

                R_i, t_i = self.kabsch(current_source, tgt_corr, weights=weights)

                # Accumulate: R_acc = R_i @ R_acc, t_acc = R_i @ t_acc + t_i
                R_acc = torch.bmm(R_i, R_acc)
                t_acc = torch.bmm(R_i, t_acc.unsqueeze(-1)).squeeze(-1) + t_i
                current_source = apply_rigid_transform(current_source, R_i, t_i)
                intermediate_sources.append(current_source)
        else:
            # Stage 1 skipped — dummy outputs for loss computation
            N = source.shape[1]
            M = target.shape[1]
            confidence = torch.ones(B, N, device=source.device)
            assignment = torch.zeros(B, N, M, device=source.device)

        source_rigid = current_source
        R_svd, t_svd = R_acc, t_acc

        # ===== Inter-stage NIA after Stage 1 (optional, 20°/0.5) =====
        if self.use_inter_stage_nia:
            R_nia1, t_nia1, source_rigid = self.inter_nia_s1(source_rigid, target)
            R_svd = torch.bmm(R_nia1, R_svd)
            t_svd = torch.bmm(R_nia1, t_svd.unsqueeze(-1)).squeeze(-1) + t_nia1

        # ===== Cycle consistency (on last iteration's features) =====
        # Forward assignment S_fwd: (B, N, M) — already computed above
        # Backward: run correspondences target→source
        if not self.no_stage1:
            _, _, assignment_bwd = self.soft_correspondence(
                tgt_point_feats, src_point_feats, target, current_source,
            )
            # assignment_bwd: (B, M, N)
            # Cycle: S_fwd @ S_bwd should ≈ I(N×N)
            cycle_matrix = torch.bmm(assignment, assignment_bwd)  # (B, N, N)
        else:
            N = source.shape[1]
            cycle_matrix = torch.eye(N, device=source.device).unsqueeze(0).expand(B, -1, -1)

        # ===== Stage 2: Iterative residual rigid refinement =====
        if not self.no_stage2:
            current_source_2a = source_rigid
            R_res_acc = torch.eye(3, device=source.device).unsqueeze(0).expand(B, -1, -1).clone()
            t_res_acc = torch.zeros(B, 3, device=source.device)

            for _s2a_iter in range(self.n_stage2a_iterations):
                src_global_2 = self.encoder(current_source_2a)   # (B, feature_dim)
                combined = torch.cat([src_global_2, tgt_global], dim=-1)  # (B, feature_dim*2)
                R_res_i, t_res_i = self.residual_rigid_head(combined)

                # Accumulate residual rigid transforms
                R_res_acc = torch.bmm(R_res_i, R_res_acc)
                t_res_acc = torch.bmm(R_res_i, t_res_acc.unsqueeze(-1)).squeeze(-1) + t_res_i
                current_source_2a = apply_rigid_transform(current_source_2a, R_res_i, t_res_i)

            source_rigid_2 = current_source_2a
            R_res, t_res = R_res_acc, t_res_acc
        else:
            # Stage 2 skipped — pass through from Stage 1
            source_rigid_2 = source_rigid
            R_res = torch.eye(3, device=source.device).unsqueeze(0).expand(B, -1, -1)
            t_res = torch.zeros(B, 3, device=source.device)

        # ===== Inter-stage NIA after Stage 2 (optional, 10°/0.2) =====
        if self.use_inter_stage_nia:
            R_nia2, t_nia2, source_rigid_2 = self.inter_nia_s2(source_rigid_2, target)
            R_res = torch.bmm(R_nia2, R_res)
            t_res = torch.bmm(R_nia2, t_res.unsqueeze(-1)).squeeze(-1) + t_nia2

        # Re-encode after final Stage 2 iteration for VAE input
        src_global_2 = self.encoder(source_rigid_2)
        combined = torch.cat([src_global_2, tgt_global], dim=-1)

        R_total = torch.bmm(R_res, R_svd)
        t_total = torch.bmm(R_res, t_svd.unsqueeze(-1)).squeeze(-1) + t_res

        # ===== Stage 3: VAE -> generator -> non-rigid residual =====
        if not self.no_stage3:
            mu, log_var = self.vae_encoder(combined)
            z = self.reparameterize(mu, log_var)

            output = self.generator(source_rigid_2, z)  # (B, N, 3)
            delta = output - source_rigid_2
        else:
            # Stage 3 skipped — rigid output is final, dummy VAE outputs
            output = source_rigid_2
            delta = torch.zeros_like(source_rigid_2)
            mu = torch.zeros(B, self.latent_dim, device=source.device)
            log_var = torch.zeros(B, self.latent_dim, device=source.device)
            z = torch.zeros(B, self.latent_dim, device=source.device)

        # ===== Build results =====
        results = {
            'output': output,
            'transformed_source': output,
            'mu': mu,
            'log_var': log_var,
            # Total rigid (for loss computation against GT)
            'R_pred': R_total,
            't_pred': t_total,
            # SVD stage outputs (for Stage 1 loss)
            'R_svd': R_svd,
            't_svd': t_svd,
            'source_rigid': source_rigid,
            # Residual rigid outputs
            'R_res': R_res,
            't_res': t_res,
            'source_rigid_2': source_rigid_2,
            # Non-rigid
            'delta': delta,
            # Correspondence info
            'confidence': confidence,
            'assignment_matrix': assignment,
            'cycle_matrix': cycle_matrix,
            'intermediate_sources': intermediate_sources,
        }

        if return_latent:
            results['z'] = z

        return results

    @torch.no_grad()
    def register(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Inference-mode registration (uses mu, no sampling).

        Args:
            source: (B, N, 3)
            target: (B, M, 3)
            n_samples: unused, kept for interface compat

        Returns:
            Registered point cloud (B, N, 3)
        """
        self.eval()

        B = source.shape[0]

        # Stage 0: PSO coarse pre-alignment (optional)
        if self.use_pso:
            _, _, current_source = self.pso(source, target)
        else:
            current_source = source

        # Stage 1: Iterative SVD
        tgt_global, tgt_point_feats = self.encoder.forward_with_point_features(target)
        if self.use_local_features:
            tgt_point_feats = self.local_enrichment(target, tgt_point_feats)

        R_acc = torch.eye(3, device=source.device).unsqueeze(0).expand(B, -1, -1).clone()
        t_acc = torch.zeros(B, 3, device=source.device)

        for _svd_iter in range(self.n_svd_iterations):
            _, src_point_feats = self.encoder.forward_with_point_features(current_source)
            if self.use_local_features:
                src_point_feats = self.local_enrichment(current_source, src_point_feats)

            tgt_corr, confidence, _ = self.soft_correspondence(
                src_point_feats, tgt_point_feats, current_source, target,
            )

            if self.use_geo_consistency:
                d_src = torch.cdist(current_source, current_source)
                d_tgt = torch.cdist(tgt_corr, tgt_corr)
                consistency = torch.exp(-self.geo_consistency_alpha * (d_src - d_tgt).abs())
                geo_score = consistency.mean(dim=-1)
                weights = confidence * geo_score
            else:
                weights = confidence

            R_i, t_i = self.kabsch(current_source, tgt_corr, weights=weights)
            R_acc = torch.bmm(R_i, R_acc)
            t_acc = torch.bmm(R_i, t_acc.unsqueeze(-1)).squeeze(-1) + t_i
            current_source = apply_rigid_transform(current_source, R_i, t_i)

        source_rigid = current_source

        # Inter-stage NIA after Stage 1 (optional, 20°/0.5)
        if self.use_inter_stage_nia:
            _, _, source_rigid = self.inter_nia_s1(source_rigid, target)

        # Stage 2: Iterative residual rigid refinement
        current_source_2a = source_rigid
        for _s2a_iter in range(self.n_stage2a_iterations):
            src_global_2 = self.encoder(current_source_2a)
            combined = torch.cat([src_global_2, tgt_global], dim=-1)
            R_res_i, t_res_i = self.residual_rigid_head(combined)
            current_source_2a = apply_rigid_transform(current_source_2a, R_res_i, t_res_i)
        source_rigid_2 = current_source_2a

        # Inter-stage NIA after Stage 2 (optional, 10°/0.2)
        if self.use_inter_stage_nia:
            _, _, source_rigid_2 = self.inter_nia_s2(source_rigid_2, target)

        # Re-encode after final Stage 2 for VAE input
        src_global_2 = self.encoder(source_rigid_2)
        combined = torch.cat([src_global_2, tgt_global], dim=-1)

        # Stage 3: VAE (deterministic — use mu)
        mu, log_var = self.vae_encoder(combined)
        output = self.generator(source_rigid_2, mu)

        # Inter-stage NIA after Stage 3 (inference only, 5°/0.1)
        if self.use_inter_stage_nia:
            _, _, output = self.inter_nia_s3(output, target)

        return output
