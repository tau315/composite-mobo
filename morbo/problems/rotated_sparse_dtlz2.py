#!/usr/bin/env python3
r"""SparseDTLZ2 with a rotated (non-axis-aligned) effective subspace.

Closes the most important logical gap in the SparseDTLZ2 study: there, the
informative dimensions are AXIS-ALIGNED -- the one geometry `ard_box`
could in principle exploit, and where rotation buys nothing beyond
per-axis rescaling. Applying a fixed random rotation to the inputs before
evaluating makes the effective subspace a random linear subspace instead,
cleanly separating two hypotheses the axis-aligned version can't:

  - "shape adaptation finds the informative SUBSPACE" -- rotation-based
    shapes (pca/ard_pca/cma) should be nearly invariant to this change,
    since PCA/CMA never privileged the coordinate axes to begin with;
  - "shape adaptation finds the informative AXES" -- `ard_box` (axis-
    aligned by construction) should get strictly worse, and `isotropic`
    should be unaffected (a hypercube is rotation-blind in distribution).

Construction: y = clamp(R (x - 0.5) + 0.5, 0, 1), then standard
SparseDTLZ2(y). The rotation R is a fixed orthogonal matrix drawn once
from a seeded generator (seed depends only on `dim`, NOT the run seed --
every method and every run seed faces the identical rotated problem, so
cross-method comparisons stay controlled). Rotating about the domain
center keeps the optimum region (informative distance dims at 0.5, i.e.
y = x = center) interior and reachable; the clamp only distorts far
corners of the cube, away from the optimum.
"""
import math
from typing import Tuple

import torch
from torch import Tensor

from morbo.problems.sparse_dtlz2 import get_sparse_dtlz2_fn

_ROTATION_BASE_SEED = 20260712  # fixed: same rotation for every run seed


def _random_rotation(dim: int, dtype, device) -> Tensor:
    """Deterministic random orthogonal matrix via QR of a seeded Gaussian."""
    gen = torch.Generator().manual_seed(_ROTATION_BASE_SEED + dim)
    A = torch.randn(dim, dim, generator=gen, dtype=torch.double)
    Q, R = torch.linalg.qr(A)
    # Fix QR sign ambiguity so the result is deterministic and det(Q)=+1-ish
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q.to(dtype=dtype, device=device)


def get_rotated_sparse_dtlz2_fn(
    dim: int,
    num_objectives: int,
    k_eff: int,
    dtype=torch.double,
    device=None,
) -> Tuple[callable, Tensor]:
    r"""Rotated-input SparseDTLZ2. Same signature/convention as
    `get_sparse_dtlz2_fn` (raw minimization values; wrap with
    `BenchmarkFunction(..., negate=True)`)."""
    base_f, bounds = get_sparse_dtlz2_fn(
        dim=dim,
        num_objectives=num_objectives,
        k_eff=k_eff,
        dtype=dtype,
        device=device,
    )
    Q = _random_rotation(dim, dtype=dtype, device=device)

    def f(X: Tensor) -> Tensor:
        Y = (X - 0.5) @ Q.to(X).T + 0.5
        return base_f(Y.clamp(0.0, 1.0))

    return f, bounds
