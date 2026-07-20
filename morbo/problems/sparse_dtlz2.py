#!/usr/bin/env python3
r"""DTLZ2 variant with a controllable gap between nominal and effective dimension.

Standard DTLZ2 already has low *effective* dimensionality relative to its
nominal `dim`: only the first `num_objectives - 1` "position" dims and the
scalar `g(x) = sum_{i in distance dims} (x_i - 0.5)^2` (a sum over the
remaining `k = dim - num_objectives + 1` "distance" dims) matter -- but every
one of those `k` distance dims contributes to `g`, so `k` itself still grows
with nominal `dim`.

This variant masks all but `k_eff` of the `k` distance dims out of `g`
entirely -- the masked-out dims are literal no-ops on every objective, not
merely "less informative" -- so nominal dimension can be scaled up while
holding the true number of informative dims fixed at
`(num_objectives - 1) + k_eff`. This directly tests whether trust-region
shape adaptation's benefit tracks the *gap* between nominal and effective
dimension (the mechanism this project's dimension sweep implicated), as
opposed to nominal dimension alone: standard DTLZ2 confounds the two, since
`k` (and hence the amount of "unused" volume an isotropic box wastes)
necessarily grows in lockstep with nominal `dim` there.
"""
import math
from typing import Tuple

import torch
from torch import Tensor


def get_sparse_dtlz2_fn(
    dim: int,
    num_objectives: int,
    k_eff: int,
    dtype=torch.double,
    device=None,
) -> Tuple[callable, Tensor]:
    r"""Construct the raw (un-negated, minimization-convention) objective
    function and bounds for sparse/partial-effective-dimension DTLZ2.

    Args:
        dim: nominal input dimension.
        num_objectives: number of objectives `M`.
        k_eff: number of the `k = dim - num_objectives + 1` distance
            dimensions that actually affect `g` (and hence every objective).
            The remaining `k - k_eff` distance dims are literal no-ops.
            Must satisfy `0 <= k_eff <= k`.
        dtype: dtype for the returned bounds.
        device: device for the returned bounds.

    Returns:
        A tuple `(f, bounds)`:
            f: a callable mapping a `... x dim`-dim tensor in `[0, 1]^dim`
                to a `... x num_objectives`-dim tensor of raw (positive,
                un-negated) DTLZ2-convention objective values -- same
                convention as `botorch.test_functions.multi_objective.DTLZ2`,
                so it is meant to be wrapped the same way (via
                `BenchmarkFunction(..., negate=True)`).
            bounds: a `2 x dim`-dim tensor, `[0, 1]^dim`.
    """
    if dim <= num_objectives:
        raise ValueError(
            f"dim must be > num_objectives, got {dim} and {num_objectives}."
        )
    k = dim - num_objectives + 1
    if not (0 <= k_eff <= k):
        raise ValueError(f"k_eff must be in [0, {k}], got {k_eff}.")

    mask = torch.zeros(k, dtype=dtype, device=device)
    mask[:k_eff] = 1.0
    pi_over_2 = math.pi / 2

    def f(X: Tensor) -> Tensor:
        X_m = X[..., -k:]
        g = ((X_m - 0.5).pow(2) * mask.to(X_m)).sum(dim=-1)
        g_plus1 = 1 + g
        fs = []
        for i in range(num_objectives):
            idx = num_objectives - 1 - i
            f_i = g_plus1.clone()
            if idx > 0:
                f_i = f_i * torch.cos(X[..., :idx] * pi_over_2).prod(dim=-1)
            if i > 0:
                f_i = f_i * torch.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        return torch.stack(fs, dim=-1)

    bounds = torch.zeros(2, dim, dtype=dtype, device=device)
    bounds[1] = 1.0
    return f, bounds
