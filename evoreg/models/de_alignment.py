"""
Differential Evolution-based coarse alignment for point cloud registration.

DE uses mutation (combine 3 random individuals) + crossover + greedy selection
to search the 6-DOF rigid transformation space. Often outperforms PSO on
multimodal fitness landscapes due to its fundamentally different search dynamics.
"""

import torch
import torch.nn as nn
import math

from .pso_alignment import axis_angle_to_rotation_matrix, batch_chamfer_fast


class DEAlignment(nn.Module):
    """
    Differential Evolution for coarse rigid alignment.

    Searches the 6-DOF space (axis-angle rotation + translation) to minimize
    Chamfer distance between transformed source and target.

    No learnable parameters — pure optimization module.
    """

    def __init__(
        self,
        n_particles: int = 50,
        n_iterations: int = 30,
        rotation_range: float = 45.0,
        translation_range: float = 1.0,
        F: float = 0.8,          # mutation scale factor
        CR: float = 0.9,         # crossover probability
        fitness_subsample: int = 128,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.rotation_range = math.radians(rotation_range)
        self.translation_range = translation_range
        self.F = F
        self.CR = CR
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

        # Initialize population: (B, P, 6)
        population = torch.zeros(B, P, 6, device=device)
        population[:, :, :3].uniform_(-self.rotation_range, self.rotation_range)
        population[:, :, 3:].uniform_(-self.translation_range, self.translation_range)
        population[:, 0, :] = 0.0  # identity

        # Evaluate initial fitness
        fitness = self._evaluate_batch(source_sub, target_sub, population)  # (B, P)

        # DE iterations
        for _ in range(self.n_iterations):
            # Mutation: DE/rand/1 — for each individual, pick 3 random others
            # v_i = x_r1 + F * (x_r2 - x_r3)
            indices = torch.stack([
                torch.randperm(P, device=device)[:P] for _ in range(B * 3)
            ]).reshape(3, B, P)
            r1, r2, r3 = indices[0], indices[1], indices[2]

            x_r1 = population[torch.arange(B, device=device).unsqueeze(1), r1]
            x_r2 = population[torch.arange(B, device=device).unsqueeze(1), r2]
            x_r3 = population[torch.arange(B, device=device).unsqueeze(1), r3]

            mutant = x_r1 + self.F * (x_r2 - x_r3)

            # Clamp mutant
            mutant[:, :, :3].clamp_(-self.rotation_range, self.rotation_range)
            mutant[:, :, 3:].clamp_(-self.translation_range, self.translation_range)

            # Crossover: binomial crossover
            # For each dimension, use mutant with probability CR, else keep original
            cross_mask = torch.rand(B, P, 6, device=device) < self.CR
            # Ensure at least one dimension comes from mutant
            j_rand = torch.randint(0, 6, (B, P), device=device)
            for d in range(6):
                cross_mask[:, :, d] |= (j_rand == d)

            trial = torch.where(cross_mask, mutant, population)

            # Selection: greedy — keep trial if better
            trial_fitness = self._evaluate_batch(source_sub, target_sub, trial)
            improved = trial_fitness < fitness
            population[improved] = trial[improved]
            fitness[improved] = trial_fitness[improved]

        # Extract best individual per batch
        best_idx = fitness.argmin(dim=1)  # (B,)
        best_pos = population[torch.arange(B, device=device), best_idx]  # (B, 6)

        R_best = axis_angle_to_rotation_matrix(best_pos[:, :3])
        t_best = best_pos[:, 3:]

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
