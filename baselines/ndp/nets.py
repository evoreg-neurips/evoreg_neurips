"""
Neural Deformation Pyramid (NDP) — core network components.

Vendored from https://github.com/rabbityl/DeformationPyramid (Li et al., ECCV 2022)
with the following modifications:
  - Inlined rotation helpers (exp_so3, skew) to remove rigid_body.py dependency
  - Removed unused classes (Nerfies_Deformation, Neural_Prior)
  - Removed unused rotation formats (euler, quaternion, 6D) — kept axis_angle only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotation helpers (inlined from rigid_body.py)
# ---------------------------------------------------------------------------

def skew(w):
    """Build skew-symmetric matrix from (*, 3) vector."""
    zeros = torch.zeros_like(w[..., 0])
    return torch.stack([
        zeros, -w[..., 2], w[..., 1],
        w[..., 2], zeros, -w[..., 0],
        -w[..., 1], w[..., 0], zeros,
    ], dim=-1).reshape(*w.shape[:-1], 3, 3)


def exp_so3(w, theta):
    """Rodrigues formula: exponential map SO(3).

    Args:
        w: (*, 3) unit axis
        theta: (*, 1) angle

    Returns:
        (*, 3, 3) rotation matrix
    """
    K = skew(w)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return I + torch.sin(theta[..., None]) * K + (1 - torch.cos(theta[..., None])) * (K @ K)


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple fully-connected ReLU network."""

    def __init__(self, depth, width):
        super().__init__()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 1)]
        )

    def forward(self, x):
        for linear in self.pts_linears:
            x = F.relu(linear(x))
        return x


class NDPLayer(nn.Module):
    """Single level of the Neural Deformation Pyramid.

    Each level has its own positional encoding frequency, MLP, and
    per-point SE(3) prediction (rotation + translation).
    """

    def __init__(self, depth, width, k0, m, rotation_format="axis_angle",
                 nonrigidity_est=False, motion='SE3'):
        super().__init__()
        self.k0 = k0
        self.m = m
        self.motion = motion
        self.nonrigidity_est = nonrigidity_est
        self.rotation_format = rotation_format

        dim_x = 6  # positional encoding: sin+cos for each of x, y, z
        self.input = nn.Sequential(nn.Linear(dim_x, width), nn.ReLU())
        self.mlp = MLP(depth=depth, width=width)

        # Rotation branch
        if self.motion in ("Sim3", "SE3"):
            self.rot_branch = nn.Linear(width, 3)  # axis-angle
            if self.motion == "Sim3":
                self.s_branch = nn.Linear(width, 1)

        # Translation branch
        self.trn_branch = nn.Linear(width, 3)

        # Nonrigidity estimation branch
        if self.nonrigidity_est:
            self.nr_branch = nn.Linear(width, 1)
            self.sigmoid = nn.Sigmoid()

        self.mlp_scale = 0.001
        self._reset_parameters()

    def forward(self, x):
        fea = self._posenc(x)
        fea = self.input(fea)
        fea = self.mlp(fea)
        t = self.mlp_scale * self.trn_branch(fea)

        if self.motion == "SE3":
            R = self._get_rotation(fea)
            x_ = (R @ x[..., None]).squeeze(-1) + t
        elif self.motion == "Sim3":
            R = self._get_rotation(fea)
            s = self.mlp_scale * self.s_branch(fea) + 1
            x_ = s * (R @ x[..., None]).squeeze(-1) + t
        else:  # scene flow
            x_ = x + t

        if self.nonrigidity_est:
            nonrigidity = self.sigmoid(self.mlp_scale * self.nr_branch(fea))
            x_ = x + nonrigidity * (x_ - x)
            nonrigidity = nonrigidity.squeeze(-1)
        else:
            nonrigidity = None

        return x_, nonrigidity

    def _get_rotation(self, fea):
        R = self.mlp_scale * self.rot_branch(fea)
        theta = torch.norm(R, dim=-1, keepdim=True).clamp(min=1e-8)
        w = R / theta
        return exp_so3(w, theta)

    def _posenc(self, pos):
        """Single-frequency positional encoding at this pyramid level."""
        mul = 2 ** (self.m + self.k0)
        sx = torch.sin(pos[..., 0:1] * mul)
        cx = torch.cos(pos[..., 0:1] * mul)
        sy = torch.sin(pos[..., 1:2] * mul)
        cy = torch.cos(pos[..., 1:2] * mul)
        sz = torch.sin(pos[..., 2:3] * mul)
        cz = torch.cos(pos[..., 2:3] * mul)
        return torch.cat([sx, cx, sy, cy, sz, cz], dim=-1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Deformation_Pyramid:
    """Multi-level deformation pyramid (coarse-to-fine test-time optimization).

    This is NOT an nn.Module — it manages a list of NDPLayer modules and
    provides utilities for hierarchical optimization.
    """

    def __init__(self, depth, width, device, k0, m, rotation_format,
                 nonrigidity_est=False, motion='SE3'):
        assert motion in ("Sim3", "SE3", "sflow")
        self.pyramid = []
        for i in range(m):
            layer = NDPLayer(
                depth, width, k0, m=i + 1,
                rotation_format=rotation_format,
                nonrigidity_est=nonrigidity_est and (i != 0),
                motion=motion,
            ).to(device)
            self.pyramid.append(layer)
        self.n_hierarchy = m

    def warp(self, x, max_level=None, min_level=0):
        if max_level is None:
            max_level = self.n_hierarchy - 1
        assert max_level < self.n_hierarchy
        data = {}
        for i in range(min_level, max_level + 1):
            x, nonrigidity = self.pyramid[i](x)
            data[i] = (x, nonrigidity)
        return x, data

    def gradient_setup(self, optimized_level):
        """Freeze all levels except the one being optimized."""
        assert optimized_level < self.n_hierarchy
        for i in range(self.n_hierarchy):
            for param in self.pyramid[i].parameters():
                param.requires_grad = (i == optimized_level)
