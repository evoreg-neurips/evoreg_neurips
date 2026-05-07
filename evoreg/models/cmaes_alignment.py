"""
CMA-ES-based coarse alignment for point cloud registration.

Covariance Matrix Adaptation Evolution Strategy adapts the search distribution's
covariance matrix over generations, learning which directions in the 6-DOF space
are most promising. Considered the gold standard for continuous black-box optimization.

Simplified implementation suitable for batched GPU execution.
"""

import torch
import torch.nn as nn
import math

from .pso_alignment import axis_angle_to_rotation_matrix, batch_chamfer_fast


class CMAESAlignment(nn.Module):
    """
    CMA-ES for coarse rigid alignment.

    Simplified CMA-ES that maintains a diagonal covariance approximation
    for GPU-friendly batched execution. Adapts step sizes per dimension.

    No learnable parameters — pure optimization module.
    """

    def __init__(
        self,
        n_particles: int = 50,
        n_iterations: int = 30,
        rotation_range: float = 45.0,
        translation_range: float = 1.0,
        fitness_subsample: int = 128,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.rotation_range = math.radians(rotation_range)
        self.translation_range = translation_range
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
        D = 6  # 3 rotation (axis-angle) + 3 translation

        # Subsample points for fitness evaluation
        S = min(self.fitness_subsample, N)
        if S < N:
            idx = torch.randperm(N, device=device)[:S]
            source_sub = source[:, idx, :]
            M_pts = target.shape[1]
            idx_t = torch.randperm(M_pts, device=device)[:S]
            target_sub = target[:, idx_t, :]
        else:
            source_sub = source
            target_sub = target

        # CMA-ES parameters (per batch)
        # Mean: start at identity (zero axis-angle, zero translation)
        mean = torch.zeros(B, D, device=device)

        # Step size (sigma) per dimension
        sigma = torch.zeros(B, D, device=device)
        sigma[:, :3] = self.rotation_range * 0.3  # rotation
        sigma[:, 3:] = self.translation_range * 0.3  # translation

        # Number of parents (top mu individuals used for update)
        mu = P // 2

        # Recombination weights (log-weighted)
        weights = torch.log(torch.tensor(mu + 0.5, device=device)) - \
                  torch.log(torch.arange(1, mu + 1, device=device, dtype=torch.float32))
        weights = weights / weights.sum()  # (mu,)

        # Learning rates
        c_sigma = 0.3  # step-size adaptation rate
        d_sigma = 1.0  # step-size damping

        # Track global best
        gbest_pos = mean.clone()
        gbest_fit = torch.full((B,), float('inf'), device=device)

        for _ in range(self.n_iterations):
            # Sample offspring: z ~ N(0, I), x = mean + sigma * z
            z = torch.randn(B, P, D, device=device)
            offspring = mean.unsqueeze(1) + sigma.unsqueeze(1) * z  # (B, P, D)

            # Clamp
            offspring[:, :, :3].clamp_(-self.rotation_range, self.rotation_range)
            offspring[:, :, 3:].clamp_(-self.translation_range, self.translation_range)

            # Evaluate fitness
            fitness = self._evaluate_batch(source_sub, target_sub, offspring)  # (B, P)

            # Sort by fitness (ascending — lower is better)
            sorted_idx = fitness.argsort(dim=1)  # (B, P)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, mu)

            # Select top-mu parents
            parent_idx = sorted_idx[:, :mu]  # (B, mu)
            parent_z = z[batch_idx, parent_idx]  # (B, mu, D)
            parent_pos = offspring[batch_idx, parent_idx]  # (B, mu, D)

            # Update mean: weighted recombination of parents
            mean_new = (weights.unsqueeze(0).unsqueeze(-1) * parent_pos).sum(dim=1)  # (B, D)

            # Update sigma: based on step length
            mean_z = (weights.unsqueeze(0).unsqueeze(-1) * parent_z).sum(dim=1)  # (B, D)
            sigma = sigma * torch.exp(c_sigma / d_sigma * (mean_z.abs().mean(dim=-1, keepdim=True) - 1))

            # Clamp sigma to prevent collapse or explosion
            sigma[:, :3].clamp_(self.rotation_range * 0.01, self.rotation_range * 0.5)
            sigma[:, 3:].clamp_(self.translation_range * 0.01, self.translation_range * 0.5)

            mean = mean_new

            # Track global best
            best_idx = sorted_idx[:, 0]  # (B,)
            best_fit = fitness[torch.arange(B, device=device), best_idx]
            best_pos = offspring[torch.arange(B, device=device), best_idx]
            improved = best_fit < gbest_fit
            gbest_pos[improved] = best_pos[improved]
            gbest_fit[improved] = best_fit[improved]

        # Also evaluate the final mean as a candidate
        mean_fitness = self._evaluate_batch(
            source_sub, target_sub, mean.unsqueeze(1)
        ).squeeze(1)  # (B,)
        improved = mean_fitness < gbest_fit
        gbest_pos[improved] = mean[improved]

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
