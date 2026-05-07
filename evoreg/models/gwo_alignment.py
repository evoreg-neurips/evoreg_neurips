"""
Grey Wolf Optimizer-based coarse alignment for point cloud registration.

GWO mimics the leadership hierarchy and hunting behavior of grey wolves.
Alpha (best), Beta (2nd best), and Delta (3rd best) guide the rest of
the pack (Omega wolves) toward prey. The encircling and hunting mechanism
provides a natural balance between exploration and exploitation.
"""

import torch
import torch.nn as nn
import math

from .pso_alignment import axis_angle_to_rotation_matrix, batch_chamfer_fast


class GWOAlignment(nn.Module):
    """
    Grey Wolf Optimizer for coarse rigid alignment.

    Searches the 6-DOF space (axis-angle rotation + translation) to minimize
    Chamfer distance. Alpha, Beta, Delta wolves guide the pack.

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

        # Initialize wolf positions: (B, P, 6)
        positions = torch.zeros(B, P, 6, device=device)
        positions[:, :, :3].uniform_(-self.rotation_range, self.rotation_range)
        positions[:, :, 3:].uniform_(-self.translation_range, self.translation_range)
        positions[:, 0, :] = 0.0  # identity

        # Evaluate initial fitness
        fitness = self._evaluate_batch(source_sub, target_sub, positions)

        # Find Alpha, Beta, Delta (top 3 wolves per batch)
        sorted_idx = fitness.argsort(dim=1)
        batch_range = torch.arange(B, device=device)

        alpha_pos = positions[batch_range, sorted_idx[:, 0]].clone()  # (B, 6)
        alpha_fit = fitness[batch_range, sorted_idx[:, 0]].clone()
        beta_pos = positions[batch_range, sorted_idx[:, 1]].clone()
        delta_pos = positions[batch_range, sorted_idx[:, 2]].clone()

        for iteration in range(self.n_iterations):
            # Linearly decrease 'a' from 2 to 0 over iterations
            a = 2.0 * (1.0 - iteration / self.n_iterations)

            # For each wolf, compute new position guided by Alpha, Beta, Delta
            # A = 2 * a * r1 - a (coefficient vector)
            # C = 2 * r2 (coefficient vector)
            # D = |C * X_leader - X_wolf|
            # X_new = X_leader - A * D

            # Alpha influence
            r1 = torch.rand(B, P, 6, device=device)
            r2 = torch.rand(B, P, 6, device=device)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = (C1 * alpha_pos.unsqueeze(1) - positions).abs()
            X1 = alpha_pos.unsqueeze(1) - A1 * D_alpha

            # Beta influence
            r1 = torch.rand(B, P, 6, device=device)
            r2 = torch.rand(B, P, 6, device=device)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = (C2 * beta_pos.unsqueeze(1) - positions).abs()
            X2 = beta_pos.unsqueeze(1) - A2 * D_beta

            # Delta influence
            r1 = torch.rand(B, P, 6, device=device)
            r2 = torch.rand(B, P, 6, device=device)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = (C3 * delta_pos.unsqueeze(1) - positions).abs()
            X3 = delta_pos.unsqueeze(1) - A3 * D_delta

            # New position = average of three leader-guided positions
            positions = (X1 + X2 + X3) / 3.0

            # Clamp
            positions[:, :, :3].clamp_(-self.rotation_range, self.rotation_range)
            positions[:, :, 3:].clamp_(-self.translation_range, self.translation_range)

            # Evaluate fitness
            fitness = self._evaluate_batch(source_sub, target_sub, positions)

            # Update Alpha, Beta, Delta
            sorted_idx = fitness.argsort(dim=1)

            new_alpha_pos = positions[batch_range, sorted_idx[:, 0]]
            new_alpha_fit = fitness[batch_range, sorted_idx[:, 0]]

            # Keep global best alpha across all iterations
            improved = new_alpha_fit < alpha_fit
            alpha_pos[improved] = new_alpha_pos[improved]
            alpha_fit[improved] = new_alpha_fit[improved]

            beta_pos = positions[batch_range, sorted_idx[:, 1]].clone()
            delta_pos = positions[batch_range, sorted_idx[:, 2]].clone()

        # Extract best R, t from Alpha
        R_best = axis_angle_to_rotation_matrix(alpha_pos[:, :3])
        t_best = alpha_pos[:, 3:]

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
