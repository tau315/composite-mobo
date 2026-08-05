"""2D SIMP structural topology-optimization benchmark (1152 dimensions).

Linear-elastic compliance minimization of a half-MBB beam via the Solid
Isotropic Material with Penalization (SIMP) method, using the exact element
stiffness matrix, DOF numbering, boundary conditions, and density filter from
the widely-used, peer-reviewed "88 lines" MATLAB code (Andreassen et al.
2011) -- self-contained here in pure NumPy/SciPy (a real sparse finite-element
solve every evaluation), no external solver dependency.

Design variables (1152 = 48x24 element grid): per-element material density in
[0, 1], passed through the code's own rmin-radius density filter before the
FE solve. The filter both suppresses checkerboarding (its usual role) and,
here, tames the compliance outliers that iid-random densities would otherwise
produce by isolating a near-zero-stiffness element on the sole load path.

Composite structure: the raw response is the per-element strain-energy /
compliance field, aggregated to a coarse 8x6 = 48-cell block grid (each block
summing a 6x4 patch), returned in log-space -- the linear-scale block
compliances span several orders of magnitude across blocks (blocks near the
load dominate), which is severely ill-conditioned for a joint GP; log-space
brings the raw covariance condition number from ~2e8 down to ~2e3. The known
reduction is total compliance (sum over blocks) and peak block compliance
(max over blocks), both minimized -- a genuine trade-off, since the
compliance-optimal SIMP criterion concentrates material exactly where local
strain energy is highest, which is what drives up the peak.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from benchmark_common import BenchmarkProblem, run_benchmark

NELX = 48
NELY = 24
DIM = NELX * NELY  # 1152
_E0 = 1.0
_EMIN = 1e-9
_NU = 0.3
_PENAL = 3.0
_RMIN = 1.5
_BLOCK_ROWS = 6
_BLOCK_COLS = 8
_BH = NELY // _BLOCK_ROWS
_BW = NELX // _BLOCK_COLS
RAW_DIM = _BLOCK_ROWS * _BLOCK_COLS  # 48
_LOG_EPS = 1e-6


def _element_stiffness(nu: float) -> np.ndarray:
    """8x8 bilinear plane-stress quad element stiffness, verbatim from top88.m."""
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], dtype=float)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], dtype=float)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], dtype=float)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], dtype=float)
    KE_A = np.vstack([np.hstack([A11, A12]), np.hstack([A12.T, A11])])
    KE_B = np.vstack([np.hstack([B11, B12]), np.hstack([B12.T, B11])])
    return (KE_A + nu * KE_B) / (1 - nu**2) / 24


def _build_density_filter(nelx: int, nely: int, rmin: float) -> sp.csr_matrix:
    """top88.m's H/Hs density filter (radius-rmin neighbor weighting)."""
    iH, jH, sH = [], [], []
    for i1 in range(1, nelx + 1):
        for j1 in range(1, nely + 1):
            e1 = (i1 - 1) * nely + j1
            for i2 in range(max(i1 - (int(np.ceil(rmin)) - 1), 1), min(i1 + (int(np.ceil(rmin)) - 1), nelx) + 1):
                for j2 in range(max(j1 - (int(np.ceil(rmin)) - 1), 1), min(j1 + (int(np.ceil(rmin)) - 1), nely) + 1):
                    e2 = (i2 - 1) * nely + j2
                    w = max(0.0, rmin - np.hypot(i1 - i2, j1 - j2))
                    if w > 0:
                        iH.append(e1 - 1)
                        jH.append(e2 - 1)
                        sH.append(w)
    H = sp.coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).flatten()
    return sp.diags(1.0 / Hs) @ H


_FILTER = _build_density_filter(NELX, NELY, _RMIN)


def _build_dof_maps(nelx: int, nely: int):
    """top88.m's edofMat (1-indexed in MATLAB, 0-indexed here)."""
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape((nely + 1, nelx + 1), order="F")
    edofVec = (2 * nodenrs[:-1, :-1] + 1).flatten(order="F")
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    return edofVec[:, None] + offsets[None, :] - 1


def _solve(x_density: np.ndarray) -> np.ndarray:
    """(NELY, NELX) density in [0, 1] -> (NELY, NELX) per-element compliance."""
    nelx, nely = NELX, NELY
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = _element_stiffness(_NU)
    edofMat = _build_dof_maps(nelx, nely)
    x_flat = x_density.flatten(order="F")
    xPhys = _FILTER @ x_flat
    scale = _EMIN + xPhys**_PENAL * (_E0 - _EMIN)
    rows = np.repeat(edofMat, 8, axis=1).reshape(-1)
    cols = np.tile(edofMat, (1, 8)).reshape(-1)
    vals = (scale[:, None] * KE.flatten()[None, :]).reshape(-1)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K = (K + K.T) / 2
    F = np.zeros(ndof)
    F[1] = -1.0  # unit downward load at top-left node
    fixed_x = np.arange(0, 2 * (nely + 1), 2)  # left-edge x-dofs
    fixed_corner = np.array([2 * (nelx + 1) * (nely + 1) - 1])  # bottom-right y-dof
    fixeddofs = np.union1d(fixed_x, fixed_corner)
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    U = np.zeros(ndof)
    U[freedofs] = spla.spsolve(K[freedofs, :][:, freedofs].tocsc(), F[freedofs])
    Ue = U[edofMat]
    ce = scale * np.einsum("ij,jk,ik->i", Ue, KE, Ue)
    return ce.reshape((nely, nelx), order="F")


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Raw block-compliance response (48 components), in log-space. X is
    normalized [0, 1]^1152 and used directly as per-element densities."""
    X_flat = X.double().reshape(-1, DIM)
    rows = []
    for row in X_flat.detach().cpu().numpy():
        density = np.clip(row, 0.0, 1.0).reshape(NELY, NELX, order="F")
        ce = _solve(density)
        blocks = ce.reshape(_BLOCK_ROWS, _BH, _BLOCK_COLS, _BW).sum(axis=(1, 3)).reshape(-1)
        rows.append(np.log(blocks + _LOG_EPS))
    return torch.tensor(np.stack(rows), dtype=torch.double)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Total compliance (sum over blocks) and peak block compliance (max),
    both minimized. Inverts the log-space raw response exactly first."""
    blocks = H.exp() - _LOG_EPS
    total = blocks.sum(dim=-1)
    peak = blocks.amax(dim=-1)
    return torch.stack([total, peak], dim=-1)


PROBLEM = BenchmarkProblem(
    name="SIMP topology optimization, half-MBB beam (2 objectives, 1152 dimensions)",
    slug="topopt_simp_2obj_1152d",
    dim=DIM,
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.tensor([0.0, 0.0], dtype=torch.double),
    ref_point=torch.tensor([6000.0, 5000.0], dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
