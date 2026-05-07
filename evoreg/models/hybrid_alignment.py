"""
Hybrid NIA-based coarse alignment for point cloud registration.

Chains two NIAs sequentially: the first performs broad exploration,
then the second refines around the best solution found. Each NIA
gets half the total iteration budget.
"""

import torch
import torch.nn as nn
import math

from .pso_alignment import PSOAlignment, axis_angle_to_rotation_matrix, batch_chamfer_fast
from .de_alignment import DEAlignment
from .cmaes_alignment import CMAESAlignment


class HybridAlignment(nn.Module):
    """
    Hybrid NIA that chains two optimization algorithms sequentially.

    Phase 1: First NIA explores broadly with half the iteration budget.
    Phase 2: Second NIA refines around Phase 1's best solution with
             a tighter search range and the remaining iterations.

    No learnable parameters — pure optimization module.
    """

    def __init__(
        self,
        nia1_class,
        nia2_class,
        n_particles: int = 50,
        n_iterations: int = 30,
        rotation_range: float = 45.0,
        translation_range: float = 1.0,
        fitness_subsample: int = 128,
        refinement_shrink: float = 0.3,  # Phase 2 searches within 30% of original range
    ):
        super().__init__()
        self.fitness_subsample = fitness_subsample
        self.rotation_range_rad = math.radians(rotation_range)
        self.translation_range = translation_range
        self.refinement_shrink = refinement_shrink

        iters_phase1 = n_iterations // 2
        iters_phase2 = n_iterations - iters_phase1

        # Phase 1: broad exploration
        self.nia1 = nia1_class(
            n_particles=n_particles,
            n_iterations=iters_phase1,
            rotation_range=rotation_range,
            translation_range=translation_range,
            fitness_subsample=fitness_subsample,
        )

        # Phase 2: refinement around Phase 1's best
        # Uses tighter range — will be re-centered at runtime
        self.nia2 = nia2_class(
            n_particles=n_particles,
            n_iterations=iters_phase2,
            rotation_range=rotation_range * refinement_shrink,
            translation_range=translation_range * refinement_shrink,
            fitness_subsample=fitness_subsample,
        )

        self.n_particles = n_particles
        self.n_iterations = n_iterations

    @torch.no_grad()
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple:
        """
        Two-phase hybrid alignment.

        Phase 1: Run NIA1 on original source/target.
        Phase 2: Apply Phase 1's best transform, then run NIA2
                 on the partially-aligned source to refine.

        Returns composed R, t from both phases applied to original source.
        """
        B = source.shape[0]
        device = source.device

        # Phase 1: broad exploration
        R1, t1, aligned_phase1 = self.nia1(source, target)

        # Phase 2: refine from Phase 1's alignment
        R2, t2, _ = self.nia2(aligned_phase1, target)

        # Compose transforms: R_total = R2 @ R1, t_total = R2 @ t1 + t2
        R_total = torch.bmm(R2, R1)
        t_total = torch.bmm(R2, t1.unsqueeze(-1)).squeeze(-1) + t2

        # Apply composed transform to original source
        aligned_source = torch.bmm(source, R_total.transpose(1, 2)) + t_total.unsqueeze(1)

        return R_total, t_total, aligned_source


class PSODEAlignment(HybridAlignment):
    """PSO exploration → DE refinement."""
    def __init__(self, n_particles=50, n_iterations=30, rotation_range=45.0,
                 translation_range=1.0, fitness_subsample=128):
        super().__init__(
            nia1_class=PSOAlignment, nia2_class=DEAlignment,
            n_particles=n_particles, n_iterations=n_iterations,
            rotation_range=rotation_range, translation_range=translation_range,
            fitness_subsample=fitness_subsample,
        )


class DECMAESAlignment(HybridAlignment):
    """DE exploration → CMA-ES refinement."""
    def __init__(self, n_particles=50, n_iterations=30, rotation_range=45.0,
                 translation_range=1.0, fitness_subsample=128):
        super().__init__(
            nia1_class=DEAlignment, nia2_class=CMAESAlignment,
            n_particles=n_particles, n_iterations=n_iterations,
            rotation_range=rotation_range, translation_range=translation_range,
            fitness_subsample=fitness_subsample,
        )


class PSOCMAESAlignment(HybridAlignment):
    """PSO exploration → CMA-ES refinement."""
    def __init__(self, n_particles=50, n_iterations=30, rotation_range=45.0,
                 translation_range=1.0, fitness_subsample=128):
        super().__init__(
            nia1_class=PSOAlignment, nia2_class=CMAESAlignment,
            n_particles=n_particles, n_iterations=n_iterations,
            rotation_range=rotation_range, translation_range=translation_range,
            fitness_subsample=fitness_subsample,
        )
