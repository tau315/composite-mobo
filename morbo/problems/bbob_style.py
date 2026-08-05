#!/usr/bin/env python3
r"""BBOB-style bi-objective benchmarks: an external landscape taxonomy.

Motivated by two gaps identified late in this project's literature review
(`LITERATURE_REVIEW.md`'s "Follow-up review" section): (1) LABCAT
(Visser et al. 2023/24) -- the closest prior art to `pca_ellipsoid`/
`ard_box` -- was evaluated on COCO/BBOB, a benchmark family we had never
touched; (2) the `ard_box` landscape-dependence finding (RESULTS.md
sec 10e/11a: wins on rugged-`g` DTLZ3/7, fails catastrophically on
smooth-`g` DTLZ2/5) rests on only 4 hand-picked functions, where BBOB's
own 24-function taxonomy is organized into 5 curated landscape categories
specifically so a method's behavior can be characterized *by category*
rather than by function name -- separable, low/moderate conditioning,
high conditioning unimodal, multimodal with adequate global structure,
multimodal with weak global structure (Hansen et al. 2021, COCO: A
platform for comparing continuous optimizers, Optimization Methods and
Software 36).

HONESTY NOTE (read before citing numbers from this module): these are
**faithful-in-spirit reimplementations**, not the official `cocoex`
package. We apply the same non-linear distortions the real BBOB functions
use (`T_osz`, `T_asy`, the `Lambda^alpha` ill-conditioning diagonal, a
random rotation) in the same qualitative composition, verified against the
published formulas (Hansen et al., "Real-Parameter Black-Box Optimization
Benchmarking 2009: Noiseless Functions Definitions", INRIA RR-6829) to the
extent practical without the reference C implementation to check
bit-exactness against. Two consequences: (a) our hypervolume numbers are
internally comparable across methods run here, but NOT directly comparable
to externally published COCO/LABCAT hypervolume tables; (b) the weak-
global-structure representative (`peaks`) is a custom multi-Gaussian-peak
function in the spirit of BBOB's Gallagher functions (f21/f22), not a
literal reimplementation of Gallagher's more involved peak-placement
algorithm. What *is* preserved, and is what this module is actually for:
the qualitative landscape taxonomy (separable/smooth vs.\ ill-conditioned
vs.\ genuinely multimodal-and-deceptive), a random rotation applied the
same way BBOB itself applies one to most of its functions, and the
bi-objective combination method real `bbob-biobj` uses -- literally pair
two base functions' outputs (Tušar et al. 2016, "COCO: The bi-objective
black box optimization benchmarking testbed").

Five base landscape representatives, one per BBOB category:
  sphere        -- separable, smooth, unimodal (category 1)
  rosenbrock    -- moderate conditioning, curved non-separable valley (2)
  ellipsoidal    -- high conditioning, unimodal, rotated (3)
  rastrigin     -- multimodal with adequate global structure (4)
  peaks         -- multimodal with weak global structure (5; custom)

Each base function optionally accepts `k_eff`: dims beyond `k_eff` are
pinned to their optimum value before evaluation (same no-op-padding
semantics as `sparse_dtlz2.py`), so this module also supports an
effective-dimension dose-response test on genuinely non-algebraic
landscapes, mirroring `SparseDTLZ2` (RESULTS.md sec 7) but without DTLZ2's
smooth closed-form `g`.
"""
import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

_BBOB_BASE_SEED = 20260716  # fixed: same instance for every run seed


def _random_rotation(dim: int, seed: int, dtype, device) -> Tensor:
    """Deterministic random orthogonal matrix via QR of a seeded Gaussian."""
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=gen, dtype=torch.double)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q.to(dtype=dtype, device=device)


def _random_shift(dim: int, seed: int, dtype, device, low=-4.0, high=4.0) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (low + (high - low) * torch.rand(dim, generator=gen, dtype=torch.double)).to(
        dtype=dtype, device=device
    )


def _t_osz(z: Tensor) -> Tensor:
    """Oscillation transform (BBOB's T_osz): smooth non-linear distortion
    that breaks perfect symmetry without changing the location of optima."""
    sgn = torch.sign(z)
    hat = torch.where(z == 0, torch.zeros_like(z), torch.log(z.abs()))
    c1 = torch.where(z > 0, torch.full_like(z, 10.0), torch.full_like(z, 5.5))
    c2 = torch.where(z > 0, torch.full_like(z, 7.9), torch.full_like(z, 3.1))
    return sgn * torch.exp(hat + 0.049 * (torch.sin(c1 * hat) + torch.sin(c2 * hat)))


def _t_asy(z: Tensor, beta: float) -> Tensor:
    """Asymmetry transform (BBOB's T_asy^beta): only distorts positive
    components, exponent grows with dimension index -- breaks symmetry."""
    dim = z.shape[-1]
    idx = torch.arange(dim, dtype=z.dtype, device=z.device)
    exponent = 1.0 + beta * (idx / max(dim - 1, 1)) * z.clamp_min(0.0).sqrt()
    pos = z.clamp_min(1e-12).pow(exponent)
    return torch.where(z > 0, pos, z)


def _lambda_alpha(dim: int, alpha: float, dtype, device) -> Tensor:
    """Diagonal ill-conditioning matrix (BBOB's Lambda^alpha)."""
    idx = torch.arange(dim, dtype=dtype, device=device)
    return alpha ** (0.5 * idx / max(dim - 1, 1))


def _apply_k_eff_mask(X: Tensor, x_opt: Tensor, k_eff: Optional[int]) -> Tensor:
    """Pin dims beyond k_eff to their optimum value -- literal no-ops,
    same semantics as sparse_dtlz2.py's masking."""
    if k_eff is None or k_eff >= X.shape[-1]:
        return X
    out = X.clone()
    out[..., k_eff:] = x_opt[k_eff:]
    return out


def _make_sphere(dim, seed, dtype, device, k_eff=None) -> Callable[[Tensor], Tensor]:
    x_opt = _random_shift(dim, seed, dtype, device)

    def f(X: Tensor) -> Tensor:
        X = _apply_k_eff_mask(X, x_opt, k_eff)
        z = X - x_opt
        return (z**2).sum(dim=-1)

    return f


def _make_rosenbrock(dim, seed, dtype, device, k_eff=None) -> Callable[[Tensor], Tensor]:
    x_opt = _random_shift(dim, seed, dtype, device)
    R = _random_rotation(dim, seed + 1, dtype, device)
    scale = max(1.0, math.sqrt(dim) / 8.0)

    def f(X: Tensor) -> Tensor:
        X = _apply_k_eff_mask(X, x_opt, k_eff)
        z = scale * (X - x_opt) @ R.T + 1.0
        return (100.0 * (z[..., :-1] ** 2 - z[..., 1:]) ** 2 + (z[..., :-1] - 1.0) ** 2).sum(
            dim=-1
        )

    return f


def _make_ellipsoidal(dim, seed, dtype, device, k_eff=None) -> Callable[[Tensor], Tensor]:
    x_opt = _random_shift(dim, seed, dtype, device)
    R = _random_rotation(dim, seed + 1, dtype, device)
    idx = torch.arange(dim, dtype=dtype, device=device)
    # Official BBOB uses a 1e6 conditioning ratio; reduced to 1e4 here to
    # keep the raw objective's dynamic range numerically tractable for GP
    # fitting at the dimensions this project tests (60-300) -- still
    # unambiguously "high conditioning" (10000x axis-to-axis curvature
    # ratio), just not so extreme it risks Cholesky/model-fit instability.
    weights = 1.0e4 ** (idx / max(dim - 1, 1))

    def f(X: Tensor) -> Tensor:
        X = _apply_k_eff_mask(X, x_opt, k_eff)
        z = _t_osz((X - x_opt) @ R.T)
        return (weights * z**2).sum(dim=-1)

    return f


def _make_rastrigin(dim, seed, dtype, device, k_eff=None) -> Callable[[Tensor], Tensor]:
    x_opt = _random_shift(dim, seed, dtype, device)
    R = _random_rotation(dim, seed + 1, dtype, device)
    lam = _lambda_alpha(dim, 10.0, dtype, device)

    def f(X: Tensor) -> Tensor:
        X = _apply_k_eff_mask(X, x_opt, k_eff)
        z = _t_osz((X - x_opt) @ R.T)
        z = _t_asy(z, beta=0.2)
        z = (lam * z) @ R.T
        d = z.shape[-1]
        return 10.0 * (d - torch.cos(2 * math.pi * z).sum(dim=-1)) + (z**2).sum(dim=-1)

    return f


def _make_peaks(
    dim, seed, dtype, device, k_eff=None, n_peaks: int = 10
) -> Callable[[Tensor], Tensor]:
    """Custom multi-Gaussian-peak function: many deceptive local optima with
    little global correlation, in the spirit of BBOB's Gallagher functions
    (NOT a literal reimplementation -- see module docstring)."""
    gen = torch.Generator().manual_seed(seed)
    centers = -4.0 + 8.0 * torch.rand(n_peaks, dim, generator=gen, dtype=torch.double)
    weights = 0.2 + 0.8 * torch.rand(n_peaks, generator=gen, dtype=torch.double)
    sigmas = 0.5 + 1.5 * torch.rand(n_peaks, generator=gen, dtype=torch.double)
    centers, weights, sigmas = (
        centers.to(dtype=dtype, device=device),
        weights.to(dtype=dtype, device=device),
        sigmas.to(dtype=dtype, device=device),
    )
    x_opt = centers[weights.argmax()]

    def f(X: Tensor) -> Tensor:
        X = _apply_k_eff_mask(X, x_opt, k_eff)
        diffs = X.unsqueeze(-2) - centers  # (..., n_peaks, dim)
        sq_dist = (diffs**2).sum(dim=-1)  # (..., n_peaks)
        peak_vals = weights * torch.exp(-sq_dist / (2 * sigmas**2))  # (..., n_peaks)
        return -100.0 * peak_vals.max(dim=-1).values + 0.01 * ((X - x_opt) ** 2).sum(dim=-1)

    return f


_BASE_FACTORIES = {
    "sphere": _make_sphere,
    "rosenbrock": _make_rosenbrock,
    "ellipsoidal": _make_ellipsoidal,
    "rastrigin": _make_rastrigin,
    "peaks": _make_peaks,
}

# Curated pairs spanning: trivial control, rotation-robustness (unimodal),
# curved-valley, structured-multimodal (DTLZ3/7 analog), weak-structure
# (Rover/Gallagher analog), and a genuinely mixed-landscape bi-objective
# tradeoff (one smooth objective, one deceptive one -- the case no DTLZ
# variant in this study tests, since DTLZ's M objectives always share one g).
PRESET_PAIRS = {
    "sphere_sphere": ("sphere", "sphere"),
    "ellipsoidal_ellipsoidal": ("ellipsoidal", "ellipsoidal"),
    "rosenbrock_rosenbrock": ("rosenbrock", "rosenbrock"),
    "rastrigin_rastrigin": ("rastrigin", "rastrigin"),
    "peaks_peaks": ("peaks", "peaks"),
    "sphere_peaks": ("sphere", "peaks"),
}


def get_bbob_biobj_fn(
    dim: int,
    f1_name: str,
    f2_name: str,
    k_eff: Optional[int] = None,
    dtype=torch.double,
    device=None,
) -> Tuple[callable, Tensor]:
    r"""Bi-objective BBOB-style function: literally pair two base functions'
    outputs (the actual `bbob-biobj` construction method).

    Args:
        dim: nominal input dimension.
        f1_name, f2_name: keys into `_BASE_FACTORIES` (see
            `PRESET_PAIRS` for curated combinations spanning the landscape
            taxonomy). The two objectives get independent random
            instances (different rotation/shift seeds) even when
            f1_name == f2_name.
        k_eff: if given, dims beyond k_eff are pinned to their optimum
            value for BOTH objectives (literal no-ops), enabling an
            effective-dimension dose-response test analogous to
            SparseDTLZ2 (RESULTS.md sec 7) on non-algebraic landscapes.
        dtype, device: for the returned bounds.

    Returns:
        `(f, bounds)`: `f` maps `n x dim` -> `n x 2` raw MINIMIZATION
        objectives (wrap with `BenchmarkFunction(..., negate=True)`, as
        every other raw-minimization evalfn in this repo does).
        `bounds`: `2 x dim`, `[-5, 5]^dim` (BBOB's own domain convention).
    """
    if f1_name not in _BASE_FACTORIES or f2_name not in _BASE_FACTORIES:
        raise ValueError(
            f"unknown BBOB-style function name(s): {f1_name!r}, {f2_name!r}; "
            f"available: {sorted(_BASE_FACTORIES)}"
        )
    f1 = _BASE_FACTORIES[f1_name](
        dim, _BBOB_BASE_SEED + dim, dtype, device, k_eff=k_eff
    )
    f2 = _BASE_FACTORIES[f2_name](
        dim, _BBOB_BASE_SEED + dim + 10_000, dtype, device, k_eff=k_eff
    )

    def f(X: Tensor) -> Tensor:
        return torch.stack([f1(X), f2(X)], dim=-1)

    bounds = torch.empty(2, dim, dtype=dtype, device=device)
    bounds[0] = -5.0
    bounds[1] = 5.0
    return f, bounds
