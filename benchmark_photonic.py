"""2D photonic-crystal complete-bandgap design benchmark (1024 dimensions).

A square-lattice 2D photonic crystal whose unit cell is a free-form 32x32
pixelated dielectric-density map (the standard topology-optimization
continuous relaxation used in real inverse photonic design), solved via the
plane-wave expansion (PWE) method -- the same physics MPB (MIT Photonic
Bands) implements, self-contained here in pure NumPy/SciPy rather than
depending on MPB/MEEP (compiled C/MPI/HDF5 tools).

Physics (Joannopoulos, Johnson, Winn, Meade, "Photonic Crystals", 2nd ed.,
Ch. 5): for a 2D crystal periodic in the x-y plane, Maxwell's equations
decouple into two scalar polarizations, each a Hermitian (generalized)
eigenvalue problem in a truncated plane-wave basis. The PWE solver here was
verified against the classic textbook validation case (square lattice of
dielectric rods, eps=8.9, r=0.2a in air): it reproduces the known E-pol
band-1/2 gap (~0.32 to ~0.44, in units of 2*pi*c/a) and correctly shows no
H-pol gap for that structure -- both textbook-known, not tuned to match.

Composite structure: the raw response is the lowest 4 bands' frequencies at
6 k-points along the irreducible Brillouin-zone path (Gamma-X-M-Gamma) for
both polarizations (6*4*2 = 48 components). The known reduction is the
complete bandgap width between bands 1 and 2 for each polarization,
max_k(band_1) subtracted from min_k(band_2) -- the standard textbook
definition. Both are maximized (a wider gap is better); this module returns
them negated for the minimize convention. Finding a structure with both a
wide E-pol and a wide H-pol gap (a "complete" bandgap) is a genuine
two-objective trade-off, since the two polarizations couple to the same
dielectric structure through different Fourier coefficients.
"""

from __future__ import annotations

import numpy as np
import torch

from benchmark_common import BenchmarkProblem, run_benchmark

GRID = 32
DIM = GRID * GRID  # 1024
_A = 1.0
_GCUT = 4  # reciprocal-lattice truncation: G=(2pi/a)(m,n), m,n in [-GCUT, GCUT]
_N_BANDS = 4
_EPS_BG = 1.0
_EPS_HI = 8.9  # silicon-like, matches the textbook validation case
_G_LIST = [(m, n) for m in range(-_GCUT, _GCUT + 1) for n in range(-_GCUT, _GCUT + 1)]
_NG = len(_G_LIST)


def _kpath(n_points: int = 6):
    """Evenly-spaced points along Gamma-X-M-Gamma (2D square lattice)."""
    pts_sym = [(0.0, 0.0), (np.pi / _A, 0.0), (np.pi / _A, np.pi / _A), (0.0, 0.0)]
    seg = [np.hypot(pts_sym[i + 1][0] - pts_sym[i][0], pts_sym[i + 1][1] - pts_sym[i][1]) for i in range(3)]
    total = sum(seg)
    pts = []
    for i in range(n_points):
        s = total * i / (n_points - 1)
        acc = 0.0
        for k in range(3):
            if s <= acc + seg[k] or k == 2:
                t = 0.0 if seg[k] == 0 else (s - acc) / seg[k]
                t = min(max(t, 0.0), 1.0)
                k0, k1 = pts_sym[k], pts_sym[k + 1]
                pts.append((k0[0] + t * (k1[0] - k0[0]), k0[1] + t * (k1[1] - k0[1])))
                break
            acc += seg[k]
    return pts


_KPTS = _kpath(6)


def _fourier_coeffs(eps_grid: np.ndarray) -> np.ndarray:
    n = eps_grid.shape[0]
    return np.fft.fftshift(np.fft.fft2(eps_grid) / (n * n))


def _get_coeff(F: np.ndarray, n: int, dm: int, dn: int) -> complex:
    i, j = dm + n // 2, dn + n // 2
    if 0 <= i < n and 0 <= j < n:
        return F[i, j]
    return 0.0 + 0.0j


def _bands_at_k(kx: float, ky: float, F: np.ndarray, n: int, polarization: str) -> np.ndarray:
    kGx = np.array([kx + 2 * np.pi / _A * m for (m, _n) in _G_LIST])
    kGy = np.array([ky + 2 * np.pi / _A * _n for (m, _n) in _G_LIST])
    cache = {}
    for a in range(_NG):
        for b in range(_NG):
            dm = _G_LIST[a][0] - _G_LIST[b][0]
            dn = _G_LIST[a][1] - _G_LIST[b][1]
            if (dm, dn) not in cache:
                cache[(dm, dn)] = _get_coeff(F, n, dm, dn)
    if polarization == "H":
        M = np.zeros((_NG, _NG), dtype=complex)
        for a in range(_NG):
            for b in range(_NG):
                dm = _G_LIST[a][0] - _G_LIST[b][0]
                dn = _G_LIST[a][1] - _G_LIST[b][1]
                M[a, b] = (kGx[a] * kGx[b] + kGy[a] * kGy[b]) * cache[(dm, dn)]
        w2 = np.linalg.eigvalsh(M)[:_N_BANDS]
    else:
        from scipy.linalg import eigh as sp_eigh

        Adiag = kGx**2 + kGy**2
        B = np.zeros((_NG, _NG), dtype=complex)
        for a in range(_NG):
            for b in range(_NG):
                dm = _G_LIST[a][0] - _G_LIST[b][0]
                dn = _G_LIST[a][1] - _G_LIST[b][1]
                B[a, b] = cache[(dm, dn)]
        w2 = sp_eigh(np.diag(Adiag).astype(complex), B, eigvals_only=True, subset_by_index=[0, _N_BANDS - 1])
    return np.sqrt(np.clip(w2, 0, None)) / (2 * np.pi)


def _bands_for_density(density: np.ndarray) -> np.ndarray:
    eps_grid = _EPS_BG + density * (_EPS_HI - _EPS_BG)
    F_eps = _fourier_coeffs(eps_grid)
    F_eta = _fourier_coeffs(1.0 / eps_grid)
    out = np.zeros((len(_KPTS), _N_BANDS, 2))
    for ki, (kx, ky) in enumerate(_KPTS):
        out[ki, :, 0] = _bands_at_k(kx, ky, F_eps, GRID, "E")
        out[ki, :, 1] = _bands_at_k(kx, ky, F_eta, GRID, "H")
    return out


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Raw band-structure response: 6 k-points x 4 bands x 2 polarizations
    (E, H), flattened -- 48 components. X is normalized [0, 1]^1024 and used
    directly as the per-pixel dielectric-density map."""
    X_flat = X.double().reshape(-1, DIM)
    rows = []
    for row in X_flat.detach().cpu().numpy():
        density = np.clip(row, 0.0, 1.0).reshape(GRID, GRID)
        rows.append(_bands_for_density(density).reshape(-1))
    return torch.tensor(np.stack(rows), dtype=torch.double)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Complete bandgap width between bands 1 and 2 per polarization:
    max_k(band_1) subtracted from min_k(band_2). Negated for the minimize
    convention (a wider gap is a smaller/more-negative objective)."""
    n_k = len(_KPTS)
    Y = H.view(*H.shape[:-1], n_k, _N_BANDS, 2)
    max_band1 = Y[..., :, 0, :].amax(dim=-2)  # (..., 2)
    min_band2 = Y[..., :, 1, :].amin(dim=-2)  # (..., 2)
    gap = min_band2 - max_band1  # [gap_E, gap_H], maximize
    return -gap  # minimize convention


PROBLEM = BenchmarkProblem(
    name="Photonic-crystal complete bandgap (2 objectives, 1024 dimensions)",
    slug="photonic_bandgap_2obj_1024d",
    dim=DIM,
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.tensor([-0.6, -0.6], dtype=torch.double),
    ref_point=torch.tensor([0.3, 0.3], dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
