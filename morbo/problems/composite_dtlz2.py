#!/usr/bin/env python3
r"""Composite-structure version of the 2-objective DTLZ2 test problem.

Standard DTLZ2 (M=2) computes:
    f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
    f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
    g(x) = sum_{i=m}^{d-1} (x_i - 0.5)^2

`get_composite_dtlz2_fn` exposes the raw intermediate quantities
`[g, cos(x_0*pi/2), sin(x_0*pi/2)]` as the "raw response" a composite-GP
would model directly, and `composite_dtlz2_reduction` is the known,
deterministic formula that reduces that raw response to the final
objectives -- mathematically identical to `botorch`'s `DTLZ2`, so this is
usable as an apples-to-apples A/B against modeling the objectives directly.
"""
import math
from typing import Tuple

import torch
from torch import Tensor


def get_composite_dtlz2_fn(
    dim: int, num_objectives: int = 2, dtype=torch.double, device=None
) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response function and bounds for composite DTLZ2.

    Args:
        dim: input dimension.
        num_objectives: must be 2 (the only case `composite_dtlz2_reduction`
            implements).
        dtype: dtype for the returned bounds.
        device: device for the returned bounds.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: a callable mapping a `... x dim`-dim tensor in
                `[0, 1]^dim` to a `... x 3`-dim tensor
                `[g, cos(x_0*pi/2), sin(x_0*pi/2)]`.
            bounds: a `2 x dim`-dim tensor, `[0, 1]^dim`.
    """
    if num_objectives != 2:
        raise ValueError(
            "composite_dtlz2_reduction only implements the M=2 case, got "
            f"num_objectives={num_objectives}."
        )
    if dim <= num_objectives:
        raise ValueError(f"dim must be > num_objectives, got {dim} and {num_objectives}.")
    k = dim - num_objectives + 1

    def raw_response(X: Tensor) -> Tensor:
        X_m = X[..., -k:]
        g = (X_m - 0.5).pow(2).sum(dim=-1)
        pi_over_2 = math.pi / 2
        cos_term = torch.cos(X[..., 0] * pi_over_2)
        sin_term = torch.sin(X[..., 0] * pi_over_2)
        return torch.stack([g, cos_term, sin_term], dim=-1)

    bounds = torch.zeros(2, dim, dtype=dtype, device=device)
    bounds[1] = 1.0
    return raw_response, bounds


def composite_dtlz2_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction from the raw DTLZ2 response to final objectives.

    Args:
        Y_raw: a `... x 3`-dim tensor `[g, cos_term, sin_term]`, as produced
            by `raw_response` above (un-negated, true minimization-convention
            DTLZ2 internals).

    Returns:
        A `... x 2`-dim tensor of *maximized* objectives -- already negated
        relative to standard (minimization) DTLZ2, matching the sign
        convention every other `evalfn` branch produces after
        `BenchmarkFunction(..., negate=True)`. Composite DTLZ2 passes
        `negate=False` to `BenchmarkFunction` instead, since this reduction
        performs the negation itself.
    """
    g = Y_raw[..., 0]
    cos_term = Y_raw[..., 1]
    sin_term = Y_raw[..., 2]
    g_plus1 = 1 + g
    f0 = g_plus1 * cos_term
    f1 = g_plus1 * sin_term
    return -torch.stack([f0, f1], dim=-1)
