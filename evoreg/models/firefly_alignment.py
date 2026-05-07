"""
Firefly Algorithm-based coarse alignment for point cloud registration.

Fireflies are attracted to brighter (better fitness) individuals with
distance-dependent attraction strength. This naturally handles multimodal
landscapes as fireflies cluster around multiple local optima.
"""

import torch
import torch.nn as nn
import math

from .pso_alignment import axis_angle_to_rotation_matrix, batch_chamfer_fast


class FireflyAlignment(nn.Module):
    """
    Firefly Algorithm for coarse rigid alignment.

    Searches the 6-DOF space (axis-angle rotation + translation) to minimize
    Chamfer distance. Fireflies move toward brighter (lower Chamfer) neighbors
    with attraction that decays with distance.

    No learnable parameters — pure optimization module.
    """

    def __init__(
        self,
        n_particles: int = 50,
        n_iterations: int = 30,
        rotation_range: float = 45.0,
        translation_range: float = 1.0,
        alpha: float = 0.2,       # randomization strength
        beta0: float = 1.0,       # base attractiveness
        gamma: float = 1.0,       # absorption coefficient (controls attraction decay)
        fitness_subsample: int = 128,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.rotation_range = math.radians(rotation_range)
        self.translation_range = translation_range
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.fitness_subsample = fitness_subsample

    @torch.no_grad()
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple:
        B, N, _ = source.shape
        device = source.device
        P = self.n_particles

        # Subsample points for fitness evaluation
        S = min(self.fitness_subsample, N)
        if S < N:
            idx = torch.randperm(N, device=device)[:S]
            source_sub = source[:, idx, :]
            M = target.shape[1]
            idx_t = torch.randperm(M, device=device)[:S]
            target_sub = target[:, idx_t, :]
        else:
            source_sub = source
            target_sub = target

        # Initialize firefly positions: (B, P, 6)
        positions = torch.zeros(B, P, 6, device=device)
        positions[:, :, :3].uniform_(-self.rotation_range, self.rotation_range)
        positions[:, :, 3:].uniform_(-self.translation_range, self.translation_range)
        positions[:, 0, :] = 0.0  # identity

        # Evaluate initial fitness
        fitness = self._evaluate_batch(source_sub, target_sub, positions)

        # Track global best
        gbest_idx = fitness.argmin(dim=1)
        gbest_pos = positions[torch.arange(B, device=device), gbest_idx].clone()
        gbest_fit = fitness[torch.arange(B, device=device), gbest_idx].clone()

        # Scale factors for randomization
        scale = torch.zeros(6, device=device)
        scale[:3] = self.rotation_range * 0.1
        scale[3:] = self.translation_range * 0.1

        alpha = self.alpha

        for _ in range(self.n_iterations):
            new_positions = positions.clone()

            # For each firefly i, move toward all brighter fireflies j
            # Vectorized: compute pairwise distances and brightness comparisons
            # positions: (B, P, 6)

            # Pairwise distances in parameter space: (B, P, P)
            diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, P, P, 6)
            r_sq = (diff ** 2).sum(dim=-1)  # (B, P, P)

            # Attractiveness: beta = beta0 * exp(-gamma * r^2)
            beta = self.beta0 * torch.exp(-self.gamma * r_sq)  # (B, P, P)

            # Brightness comparison: j is brighter than i if fitness[j] < fitness[i]
            # fitness: (B, P) — lower is better (Chamfer distance)
            brighter = fitness.unsqueeze(1) > fitness.unsqueeze(2)  # (B, P_i, P_j)
            # brighter[b, i, j] = True if firefly j is brighter than i

            # Weighted movement toward brighter fireflies
            # attraction = sum_j [brighter_ij * beta_ij * (x_j - x_i)]
            attraction_weights = brighter.float() * beta  # (B, P, P)
            # Normalize: divide by number of brighter fireflies (avoid div by zero)
            n_brighter = attraction_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
            attraction_weights = attraction_weights / n_brighter

            # Weighted sum of movements toward brighter fireflies
            movement = torch.bmm(
                attraction_weights.reshape(B * P, 1, P).float(),
                diff.reshape(B * P, P, 6).float() * -1  # diff is (i - j), we want (j - i)
            ).reshape(B, P, 6) * -1  # wait, diff = pos_i - pos_j, so -(diff) = pos_j - pos_i

            # Actually: diff[b,i,j] = pos[b,i] - pos[b,j], so movement toward j is -diff
            movement = torch.einsum('bij,bijd->bid', attraction_weights, -diff)

            # Add randomization
            rand_step = alpha * scale.unsqueeze(0).unsqueeze(0) * \
                        (torch.rand(B, P, 6, device=device) - 0.5)

            new_positions = positions + movement + rand_step

            # Clamp
            new_positions[:, :, :3].clamp_(-self.rotation_range, self.rotation_range)
            new_positions[:, :, 3:].clamp_(-self.translation_range, self.translation_range)

            positions = new_positions

            # Re-evaluate fitness
            fitness = self._evaluate_batch(source_sub, target_sub, positions)

            # Update global best
            batch_best_idx = fitness.argmin(dim=1)
            batch_best_fit = fitness[torch.arange(B, device=device), batch_best_idx]
            improved = batch_best_fit < gbest_fit
            gbest_pos[improved] = positions[
                torch.arange(B, device=device), batch_best_idx
            ][improved]
            gbest_fit[improved] = batch_best_fit[improved]

            # Reduce randomization over time
            alpha *= 0.97

        # Extract best R, t
        R_best = axis_angle_to_rotation_matrix(gbest_pos[:, :3])
        t_best = gbest_pos[:, 3:]

        aligned_source = torch.bmm(source, R_best.transpose(1, 2)) + t_best.unsqueeze(1)

        return R_best, t_best, aligned_source

    def _evaluate_batch(self, source, target, positions):
        B, N, _ = source.shape
        P = positions.shape[1]

        axis_angles = positions[:, :, :3].reshape(B * P, 3)
        R_all = axis_angle_to_rotation_matrix(axis_angles).reshape(B, P, 3, 3)
        t_all = positions[:, :, 3:]

        source_exp = source.unsqueeze(1).expand(-1, P, -1, -1)
        transformed = torch.einsum('bpnj,bpkj->bpnk', source_exp, R_all) + t_all.unsqueeze(2)

        transformed_flat = transformed.reshape(B * P, N, 3)
        target_flat = target.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, -1, 3)

        fitness = batch_chamfer_fast(transformed_flat, target_flat)
        return fitness.reshape(B, P)
