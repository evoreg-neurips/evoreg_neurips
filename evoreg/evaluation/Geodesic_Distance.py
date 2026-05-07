"""
Implements two error metrics:
Geodesic Distance (Normalized by volume)
Correspondce error within a geodesic radius
"""
from typing import Optional, Sequence, Union, Tuple
import numpy as np
import torch
import trimesh
import potpourri3d as pp3d

def _np_vertices_faces(m: trimesh.Trimesh):
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces,    dtype=np.int32)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("Mesh must be triangular with V:(N,3), F:(F,3).")
    return V, F

def _length_scale(m: trimesh.Trimesh) -> float:
    return float(np.sqrt(m.area + 1e-12))

def Geodesic_Distance(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    seeds: Optional[Union[int, Sequence[int], np.ndarray]] = 256,
    rng_seed: int = 0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Geodesic distortion between meshes (Pred vs GT), normalized by volume^(1/3).

    seeds:
      - int  -> sample this many seeds uniformly at random
      - list/array[int] -> use exactly these vertex indices
      - None -> use ALL vertices as seeds

    Returns:
      (normalized_mae: torch.scalar, seeds_used: np.ndarray[int])
    """
    V_gt, F_gt   = _np_vertices_faces(source)
    V_pr, F_pred = _np_vertices_faces(target)

    if F_gt.shape != F_pred.shape or not np.array_equal(F_gt, F_pred):
        raise ValueError("source and target must share IDENTICAL faces (same topology).")

    N = V_gt.shape[0]

    # Resolve seeds
    if seeds is None:
        seeds_used = np.arange(N, dtype=np.int32)
    elif isinstance(seeds, int):
        k = max(1, min(seeds, N))
        rng = np.random.default_rng(rng_seed)
        seeds_used = rng.choice(N, size=k, replace=False).astype(np.int32)
    else:
        seeds_used = np.asarray(seeds, dtype=np.int32)
        if seeds_used.ndim != 1:
            raise ValueError("seeds list/array must be 1D.")
        if (seeds_used < 0).any() or (seeds_used >= N).any():
            raise ValueError("seeds contain out-of-range vertex indices.")
        # dedupe & keep stable order
        _, first_idx = np.unique(seeds_used, return_index=True)
        seeds_used = seeds_used[np.sort(first_idx)]

    # Geodesic solvers
    solver_gt   = pp3d.MeshHeatMethodDistanceSolver(V_gt, F_gt)
    solver_pred = pp3d.MeshHeatMethodDistanceSolver(V_pr, F_gt)

    # Distance tables
    D_gt   = np.stack([solver_gt.compute_distance(int(s))   for s in seeds_used], axis=0)  # (K,N)
    D_pred = np.stack([solver_pred.compute_distance(int(s)) for s in seeds_used], axis=0)  # (K,N)

    # Absolute difference → normalize by length scale
    diff = np.abs(D_pred - D_gt)
    L = _length_scale(source) + 1e-12
    mae_norm = float((diff / L).mean())

    return torch.tensor(mae_norm, dtype=torch.float32), seeds_used
    
    


if __name__ == "__main__":
    import numpy as np
    import trimesh

    # >>>> edit this to your local file <<<<
    PLY_PATH = r"C:\Users\ben-p\CCDM\EvoReg_Point_Set_Registration\evoreg\data\FAUSTSample\tr_reg_000.ply"

    mesh = trimesh.load_mesh(PLY_PATH, process=False)
    N = len(mesh.vertices)


    # Identity: distortion should be ~0
    d0, seeds_used = Geodesic_Distance(mesh, mesh.copy(), seeds=[0, 10, 42, 1337])
    d0 = d0.item()
    print(f"[GEO Test 1] Distortion (identity, seeds={seeds_used.tolist()}): {d0:.6f}")
    assert d0 <= 1e-6


    # Build a length scale to pick a reasonable radius for noisy vertices
    E = mesh.edges_unique
    if len(E) == 0:
        # fallback: crude scale from bbox
        med_edge = np.linalg.norm(mesh.bounding_box.extents) / 100.0
    else:
        med_edge = float(np.median(
            np.linalg.norm(mesh.vertices[E[:, 0]] - mesh.vertices[E[:, 1]], axis=1)
        ))

    # Small noise: distortion small but > 0
    rng = np.random.default_rng(0)
    noise = 0.02 * med_edge
    mp = mesh.copy()
    Vp = mp.vertices.copy()
    Vp += rng.normal(scale=noise, size=Vp.shape)
    mp.vertices = Vp
    d1, used_auto = Geodesic_Distance(mesh, mp, seeds=128, rng_seed=123)
    d1 = d1.item()
    print(f"[GEO Test 2] Distortion (noisy, 128 seeds): {d1:.6f}")
    assert d1 > 0.0

    # All vertices as seeds (might be slower on large meshes)
    d2, used_all = Geodesic_Distance(mesh, mp, seeds=None)
    d2 = d2.item()
    print(f"[GEO Test 3] Distortion (noisy, all seeds={len(used_all)}): {d2:.6f}")
    assert d2 >= d1 * 0.5  # sanity: using all seeds shouldn't be wildly smaller

    print("All geodesic-distance tests passed.")
  