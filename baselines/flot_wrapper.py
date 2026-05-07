"""
FLOT wrapper for the evaluation pipeline.

Wraps the FLOT scene-flow model as an nn.Module so it can be called from
evaluate_baselines.py like any other model.

FLOT predicts a per-point scene flow vector that, when added to the source
points, yields their corresponding positions in the target frame.

Reference: Puy et al., "FLOT: Scene Flow on Point Clouds Guided by
Optimal Transport", ECCV 2020.
"""

import torch
import torch.nn as nn

from evoreg.models.FLOT_model import FLOT


class FLOTWrapper(nn.Module):
    """nn.Module wrapper around FLOT for the standard registration interface."""

    def __init__(self, nb_iter: int = 10):
        """
        Args:
            nb_iter: Number of unrolled Sinkhorn iterations in FLOT's
                optimal-transport correspondence step. Default 10
                matches the original FLOT paper.
        """
        super().__init__()
        self.net = FLOT(nb_iter=nb_iter)

    def forward(self, source, target):
        """Run FLOT on a point cloud pair and return the transformed source.

        Args:
            source: (B, N, 3) tensor — points to align
            target: (B, M, 3) tensor — reference points

        Returns:
            dict with 'transformed_source' (B, N, 3). FLOT does not predict
            an explicit rigid transform, so 'est_R' and 'est_t' are omitted;
            the evaluation pipeline's Kabsch fallback can recover them from
            the source --> transformed_source correspondence if needed.
        """
        # FLOT expects a tuple (pc1, pc2) where pc1 is source and pc2 is target
        flow = self.net((source, target))   # (B, N, 3)
        transformed = source + flow

        return {
            'transformed_source': transformed,
        }
