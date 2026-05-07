"""
Implements one metric:

Correspondence accuracy within a geodesic radius (FAUST-style):
  For each source vertex i, measure the shortest-path distance
  along the TARGET surface between the predicted target location f(i)
  and the ground-truth target location f_true(i)=i. Report Acc@radius.
"""

from typing import Optional, Sequence, Dict
import numpy as np
import torch
import trimesh
import potpourri3d as pp3d
from scipy.spatial import cKDTree


def Correspondence_Error(
    target: trimesh.Trimesh,
    *,
    radius: float = 0.05,                   # same units as the mesh (e.g., meters → 0.05m)
    pi_hat: Optional[np.ndarray] = None,    # (N,) predicted target indices (mapping)
    V_pred: Optional[np.ndarray] = None,    # OR (N,3) predicted vertices to snap to target
    point_indices: Optional[Sequence[int]] = None  # optional subset of vertices to evaluate
) -> torch.Tensor:
    """
    Computes FAUST-style *geodesic correspondence* accuracy on the TARGET surface.

    - target: trimesh.Trimesh of the target mesh (M2). Must be triangular.
    - radius: threshold for Acc@radius (same units as target vertices).
    - pi_hat: predicted mapping (indices on target) of shape (N,).
    - V_pred: predicted vertices (N,3). If provided, we snap to nearest target vertex to
              obtain pi_hat.
    - point_indices: optional iterable of vertex indices to evaluate (default: all).

    Returns:
        torch.Tensor scalar in [0,1]: Acc@radius (fraction of evaluated vertices with
        geodesic error ≤ radius).
    """
    # --- Validate mesh ---
    V2 = np.asarray(target.vertices, dtype=np.float64)
    F2 = np.asarray(target.faces,    dtype=np.int32)
    if V2.ndim != 2 or V2.shape[1] != 3 or F2.ndim != 2 or F2.shape[1] != 3:
        raise ValueError("Target mesh must be triangular with V:(N,3), F:(F,3).")
    N = V2.shape[0]

    # --- Build predicted indices on the target (if needed) ---
    if (pi_hat is None) == (V_pred is None):
        raise ValueError("Provide exactly one of: pi_hat (indices) OR V_pred (predicted vertices).")

    if V_pred is not None:
        if V_pred.shape != V2.shape:
            raise ValueError("V_pred must have shape (N,3) to match target.vertices.")
        kd = cKDTree(V2)
        _, pi_hat_full = kd.query(V_pred, k=1)  # snap to nearest target vertex
        pi_hat_full = pi_hat_full.astype(np.int32)
    else:
        pi_hat = np.asarray(pi_hat, dtype=np.int32)
        if pi_hat.shape[0] != N:
            raise ValueError("pi_hat must have length N to match target vertices.")
        pi_hat_full = pi_hat

    # --- Choose which source vertices (on M1) to evaluate; GT mapping is identity (i -> i) ---
    if point_indices is None:
        eval_idx = np.arange(N, dtype=np.int32)
    else:
        eval_idx = np.asarray(point_indices, dtype=np.int32)
        if eval_idx.ndim != 1:
            raise ValueError("point_indices must be a 1D sequence of ints.")

    # --- Geodesics on the TARGET only (M2) ---
    solver = pp3d.MeshHeatMethodDistanceSolver(V2, F2)

    # Compute distance fields only from the UNIQUE predicted targets we’ll query
    uniq_pred = np.unique(pi_hat_full[eval_idx])
    dist_from_pred: Dict[int, np.ndarray] = {
        int(j): solver.compute_distance(int(j)) for j in uniq_pred
    }

    # Per-vertex errors: e_i = d_M2( predicted j , true i )
    e = np.empty(eval_idx.shape[0], dtype=np.float64)
    for k, i in enumerate(eval_idx):
        j = int(pi_hat_full[i])
        e[k] = dist_from_pred[j][int(i)]

    acc = float((e <= float(radius)).mean())
    return torch.tensor(acc, dtype=torch.float32)


# --------------------
# Quick tests
# --------------------
if __name__ == "__main__":


    
    PLY_PATH = r"C:\Users\ben-p\CCDM\EvoReg_Point_Set_Registration\evoreg\data\FAUSTSample\tr_reg_000.ply"

    mesh = trimesh.load_mesh(PLY_PATH, process=False)
    N = len(mesh.vertices)

    # Test 1: identity mapping → Acc@tiny radius ≈ 1.0
    pi_id = np.arange(N, dtype=np.int32)
    acc_id = Correspondence_Error(mesh, radius=1e-9, pi_hat=pi_id).item()
    print(f"[Test 1] Identity mapping, Acc@1e-9: {acc_id:.4f}")
    assert abs(acc_id - 1.0) < 1e-6

    # Build a length scale to pick a reasonable radius for noisy vertices
    E = mesh.edges_unique
    if len(E) == 0:
        # fallback: crude scale from bbox
        med_edge = np.linalg.norm(mesh.bounding_box.extents) / 100.0
    else:
        med_edge = float(np.median(np.linalg.norm(
            mesh.vertices[E[:, 0]] - mesh.vertices[E[:, 1]], axis=1)))

    # Test 2: small vertex noise, radius a few× noise → high accuracy
    rng = np.random.default_rng(0)
    noisy = mesh.copy()
    V_noisy = noisy.vertices.copy()
    noise = 0.02 * med_edge
    V_noisy += rng.normal(scale=noise, size=V_noisy.shape)
    noisy.vertices = V_noisy

    acc_noisy = Correspondence_Error(mesh, radius=3 * noise, V_pred=noisy.vertices).item()
    print(f"[Test 2] Noisy target, Acc@3*noise: {acc_noisy:.4f}")
    assert acc_noisy > 0.7  # should be high with generous radius

    # Test 3: subset evaluation
    subset = np.arange(0, N, max(1, N // 100))  # ~1% of vertices
    acc_subset = Correspondence_Error(mesh, radius=1e-9, pi_hat=pi_id, point_indices=subset).item()
    print(f"[Test 3] Subset eval (identity), Acc@1e-9: {acc_subset:.4f}")
    assert abs(acc_subset - 1.0) < 1e-6

    print("All correspondence-error tests passed.")
