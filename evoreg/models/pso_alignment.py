"""
PSO-based coarse alignment for point cloud registration.

Particle Swarm Optimization searches the 6-DOF rigid transformation space
(axis-angle rotation + translation) to find a coarse alignment that minimizes
Chamfer distance. This provides a good initialization for the learned pipeline.
"""

import torch
import torch.nn as nn
import math


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix via Rodrigues' formula.

    Args:
        axis_angle: (*, 3) axis-angle vectors

    Returns:
        (*, 3, 3) rotation matrices
    """
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)  # (*, 1)
    axis = axis_angle / theta  # (*, 3)

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of the axis
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    sin_theta = torch.sin(theta).unsqueeze(-1)  # (*, 1, 1)
    cos_theta = torch.cos(theta).unsqueeze(-1)  # (*, 1, 1)

    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    R = eye + sin_theta * K + (1 - cos_theta) * torch.bmm(
        K.reshape(-1, 3, 3), K.reshape(-1, 3, 3)
    ).reshape_as(K)

    return R


def batch_chamfer_fast(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Fast batched Chamfer distance (mean of bidirectional nearest-neighbor distances).

    Args:
        source: (B, N, 3)
        target: (B, M, 3)

    Returns:
        (B,) Chamfer distance per sample
    """
    # source→target
    dists_st = torch.cdist(source, target)  # (B, N, M)
    min_st = dists_st.min(dim=-1).values     # (B, N)

    # target→source
    min_ts = dists_st.min(dim=-2).values     # (B, M)

    return min_st.mean(dim=-1) + min_ts.mean(dim=-1)  # (B,)


class PSOAlignment(nn.Module):
    """
    Particle Swarm Optimization for coarse rigid alignment.

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
        w: float = 0.7,       # inertia weight
        c1: float = 1.5,      # cognitive coefficient
        c2: float = 1.5,      # social coefficient
        fitness_subsample: int = 128,  # subsample points for Chamfer fitness
    ):
        super().__init__()
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.rotation_range = math.radians(rotation_range)
        self.translation_range = translation_range
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.fitness_subsample = fitness_subsample

    @torch.no_grad()
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple:
        """
        Find coarse alignment via PSO.

        Args:
            source: (B, N, 3) source point cloud
            target: (B, M, 3) target point cloud

        Returns:
            R_pso: (B, 3, 3) rotation matrix
            t_pso: (B, 3) translation vector
            aligned_source: (B, N, 3) source after PSO alignment
        """
        B, N, _ = source.shape
        device = source.device
        P = self.n_particles

        # Subsample points for fitness evaluation (coarse alignment doesn't need all points)
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

        # Initialize particles: 6-DOF [axis_angle(3), translation(3)]
        # Rotation: uniform in axis-angle space within rotation_range
        # Translation: uniform within translation_range
        positions = torch.zeros(B, P, 6, device=device)
        positions[:, :, :3].uniform_(-self.rotation_range, self.rotation_range)
        positions[:, :, 3:].uniform_(-self.translation_range, self.translation_range)

        # Include identity as one particle (no transformation)
        positions[:, 0, :] = 0.0

        # Initialize velocities
        velocities = torch.zeros_like(positions)
        velocities[:, :, :3].uniform_(-self.rotation_range * 0.1, self.rotation_range * 0.1)
        velocities[:, :, 3:].uniform_(-self.translation_range * 0.1, self.translation_range * 0.1)

        # Evaluate initial fitness for all particles (using subsampled points)
        fitness = self._evaluate_batch(source_sub, target_sub, positions)  # (B, P)

        # Personal best
        pbest_pos = positions.clone()
        pbest_fit = fitness.clone()

        # Global best
        gbest_idx = fitness.argmin(dim=1)  # (B,)
        gbest_pos = positions[torch.arange(B, device=device), gbest_idx]  # (B, 6)
        gbest_fit = fitness[torch.arange(B, device=device), gbest_idx]    # (B,)

        # PSO iterations
        for _ in range(self.n_iterations):
            r1 = torch.rand(B, P, 6, device=device)
            r2 = torch.rand(B, P, 6, device=device)

            # Update velocities
            cognitive = self.c1 * r1 * (pbest_pos - positions)
            social = self.c2 * r2 * (gbest_pos.unsqueeze(1) - positions)
            velocities = self.w * velocities + cognitive + social

            # Clamp velocities
            max_v_rot = self.rotation_range * 0.2
            max_v_trans = self.translation_range * 0.2
            velocities[:, :, :3].clamp_(-max_v_rot, max_v_rot)
            velocities[:, :, 3:].clamp_(-max_v_trans, max_v_trans)

            # Update positions
            positions = positions + velocities

            # Clamp positions
            positions[:, :, :3].clamp_(-self.rotation_range, self.rotation_range)
            positions[:, :, 3:].clamp_(-self.translation_range, self.translation_range)

            # Evaluate fitness (using subsampled points)
            fitness = self._evaluate_batch(source_sub, target_sub, positions)

            # Update personal best
            improved = fitness < pbest_fit
            pbest_pos[improved] = positions[improved]
            pbest_fit[improved] = fitness[improved]

            # Update global best
            batch_best_idx = fitness.argmin(dim=1)
            batch_best_fit = fitness[torch.arange(B, device=device), batch_best_idx]
            improved_global = batch_best_fit < gbest_fit
            gbest_pos[improved_global] = positions[
                torch.arange(B, device=device), batch_best_idx
            ][improved_global]
            gbest_fit[improved_global] = batch_best_fit[improved_global]

        # Extract best R, t
        R_pso = axis_angle_to_rotation_matrix(gbest_pos[:, :3])  # (B, 3, 3)
        t_pso = gbest_pos[:, 3:]  # (B, 3)

        # Apply to source
        aligned_source = torch.bmm(source, R_pso.transpose(1, 2)) + t_pso.unsqueeze(1)

        return R_pso, t_pso, aligned_source

    def _evaluate_batch(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate Chamfer distance for all particles in all batches.

        Args:
            source: (B, N, 3)
            target: (B, M, 3)
            positions: (B, P, 6)

        Returns:
            fitness: (B, P) Chamfer distance for each particle
        """
        B, N, _ = source.shape
        P = positions.shape[1]

        # Convert axis-angle to rotation matrices: (B, P, 3, 3)
        axis_angles = positions[:, :, :3].reshape(B * P, 3)
        R_all = axis_angle_to_rotation_matrix(axis_angles).reshape(B, P, 3, 3)
        t_all = positions[:, :, 3:]  # (B, P, 3)

        # Transform source by each particle: (B, P, N, 3)
        # source: (B, 1, N, 3), R: (B, P, 3, 3) → (B, P, N, 3)
        source_exp = source.unsqueeze(1).expand(-1, P, -1, -1)  # (B, P, N, 3)
        transformed = torch.einsum('bpnj,bpkj->bpnk', source_exp, R_all) + t_all.unsqueeze(2)

        # Compute Chamfer for each particle
        # Reshape to (B*P, N, 3) and (B*P, M, 3) for batch Chamfer
        transformed_flat = transformed.reshape(B * P, N, 3)
        target_flat = target.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, -1, 3)

        fitness = batch_chamfer_fast(transformed_flat, target_flat)  # (B*P,)
        return fitness.reshape(B, P)
