"""Core DefTransNet model: Tnet, EdgeConv, and DefTransNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import knn_graph
from .transformer import Transformer


class Tnet(nn.Module):
    """Spatial transformer network that learns a 3x3 affine matrix."""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

    def forward(self, input):
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool_size = int(xb.size(-1))
        pool = F.max_pool1d(xb, pool_size).squeeze(-1)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


class EdgeConv(nn.Module):
    """Graph convolution with PointNet++-like neighbor aggregation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x, ind):
        B, N, D = x.shape
        k = ind.shape[2]

        y = x.reshape(B * N, D)[ind.reshape(B * N, k)].reshape(B, N, k, D)
        x = x.reshape(B, N, 1, D).expand(B, N, k, D)

        x = torch.cat([y - x, x], dim=3)

        x = self.conv(x.permute(0, 3, 1, 2))
        x = F.max_pool2d(x, (1, k))
        x = x.squeeze(3).permute(0, 2, 1)

        return x


class DefTransNet(nn.Module):
    """DefTransNet feature extractor.

    Extracts per-point features from two point clouds using graph convolution
    and cross-attention transformer. Returns feature tuples, NOT displacements.
    Displacement inference is handled by smooth_lbp in inference.py.
    """

    def __init__(self, k=10, emb_dims=64, n_heads=4, ff_dims=1024, n_layers=1):
        super().__init__()
        self.k = k
        self.input_transform = Tnet(k=3)

        self.conv1 = EdgeConv(3, 32)
        self.conv2 = EdgeConv(32, 32)
        self.conv3 = EdgeConv(32, emb_dims)

        self.conv4 = nn.Sequential(
            nn.Conv1d(emb_dims, emb_dims, 1, bias=False),
            nn.InstanceNorm1d(emb_dims),
            nn.Conv1d(emb_dims, emb_dims, 1),
        )

        self.transform = Transformer(
            emb_dims=emb_dims, n_heads=n_heads, ff_dims=ff_dims, n_layers=n_layers,
        )

    def forward(self, x, y, k=None):
        """Extract features from source and target point clouds.

        Args:
            x: (B, N, 3) source points
            y: (B, M, 3) target points
            k: kNN neighbors (defaults to self.k)

        Returns:
            (x_feat, y_feat): feature tuples, each (B, N/M, emb_dims)
        """
        if k is None:
            k = self.k

        # Input transform
        matrix3x3x = self.input_transform(x.transpose(1, 2))
        x = torch.bmm(x, matrix3x3x)

        matrix3x3y = self.input_transform(y.transpose(1, 2))
        y = torch.bmm(y, matrix3x3y)

        # Graph convolution on source
        fixed_ind = knn_graph(x, k, include_self=True)[0]
        x = self.conv1(x, fixed_ind)
        x = self.conv2(x, fixed_ind)
        x = self.conv3(x, fixed_ind)

        # Graph convolution on target (3x more neighbors)
        moving_ind = knn_graph(y, k * 3, include_self=True)[0]
        y = self.conv1(y, moving_ind)
        y = self.conv2(y, moving_ind)
        y = self.conv3(y, moving_ind)

        # Cross-attention transformer
        xfp, yfp = self.transform(x.transpose(1, 2), y.transpose(1, 2))
        x = x + xfp.transpose(1, 2)
        y = y + yfp.transpose(1, 2)

        # Final MLP
        x = self.conv4(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.conv4(y.permute(0, 2, 1)).permute(0, 2, 1)

        return x, y
