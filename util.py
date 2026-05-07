import numpy as np

def normalize_point_cloud(pc, eps=1e-9):
    """
    pc: (N,3) numpy array
    Returns: normalized pc, center, scale
    """
    center = pc.mean(axis=0, keepdims=True)
    pc_centered = pc - center
    scale = np.linalg.norm(pc_centered, axis=1).max() + eps # radius
    pc_norm = pc_centered / scale
    return pc_norm, center, scale
