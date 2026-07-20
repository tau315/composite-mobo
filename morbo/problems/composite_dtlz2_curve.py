#!/usr/bin/env python3
r"""A *correlated* composite-structure DTLZ2 variant, for isolating whether
correlation-aware raw-response modeling (e.g. a Kronecker-structured
multi-task GP) helps over independent per-dimension GPs.

`morbo/problems/composite_dtlz2.py`'s raw response `[g, cos_term, sin_term]`
is *not* meaningfully correlated across its 3 components -- it can't
distinguish "decoupling into independent GPs is enough" from "modeling
cross-dimension correlation matters", since there's essentially no
correlation structure to exploit either way. This module instead exposes a
raw response that is a genuine discretized *curve*:

    h(theta; x) = (1 + g(x)) * cos(theta - x_0 * pi/2),   theta in [0, pi/2]

sampled at K equally spaced points. Since h is smooth in theta, adjacent
raw-response components are strongly correlated by construction -- the same
kind of structure a real image (Maddox et al. 2021) or a real time-series
trajectory (see `composite_penicillin.py`) would have.

The curve is constructed so its *endpoints* are exactly DTLZ2's own two
objectives (using cos(-a) = cos(a) and cos(pi/2 - a) = sin(a)):
    h(0)     = (1+g) * cos(x_0*pi/2)  = f_0
    h(pi/2)  = (1+g) * sin(x_0*pi/2)  = f_1
so this remains mathematically equivalent to standard DTLZ2 -- same true
objectives, same Pareto front -- with the interior K-2 curve points as pure
"extra correlated raw structure" for the ablation to exploit or ignore.
"""
import math
from typing import Tuple

import torch
from torch import Tensor


def get_composite_dtlz2_curve_fn(
    dim: int,
    num_objectives: int = 2,
    n_curve_points: int = 8,
    dtype=torch.double,
    device=None,
) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response (curve) function and bounds.

    Args:
        dim: input dimension.
        num_objectives: must be 2.
        n_curve_points: number of points `K` at which the curve is sampled
            (raw response dimension). `K >= 2`; the first and last points
            are exactly the two DTLZ2 objectives.
        dtype, device: for the returned bounds.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping a `... x dim`-dim tensor in
                `[0, 1]^dim` to a `... x n_curve_points`-dim tensor, the
                curve `h(theta_j; x)` for `j = 0, ..., K-1`.
            bounds: a `2 x dim`-dim tensor, `[0, 1]^dim`.
    """
    if num_objectives != 2:
        raise ValueError(
            f"composite_dtlz2_curve only implements M=2, got {num_objectives}."
        )
    if n_curve_points < 2:
        raise ValueError(f"n_curve_points must be >= 2, got {n_curve_points}.")
    if dim <= num_objectives:
        raise ValueError(f"dim must be > num_objectives, got {dim} and {num_objectives}.")
    k = dim - num_objectives + 1
    thetas = torch.linspace(0.0, math.pi / 2, n_curve_points, dtype=dtype, device=device)

    def raw_response(X: Tensor) -> Tensor:
        X_m = X[..., -k:]
        g = (X_m - 0.5).pow(2).sum(dim=-1, keepdim=True)  # `... x 1`
        g_plus1 = 1 + g
        angle = thetas.to(X) - X[..., 0:1] * (math.pi / 2)  # `... x n_curve_points`
        return g_plus1 * torch.cos(angle)

    bounds = torch.zeros(2, dim, dtype=dtype, device=device)
    bounds[1] = 1.0
    return raw_response, bounds


def composite_dtlz2_curve_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: the curve's endpoints are exactly the 2 objectives.

    Args:
        Y_raw: a `... x K`-dim tensor, the sampled curve (un-negated,
            true minimization-convention DTLZ2 internals).

    Returns:
        A `... x 2`-dim tensor of *maximized* objectives (negated relative
        to standard minimization DTLZ2, matching every other evalfn's
        convention after `BenchmarkFunction(..., negate=True)` -- this
        reduction performs that negation itself, so `negate=False` should be
        passed to `BenchmarkFunction` for this evalfn, same as
        `composite_dtlz2.py`).
    """
    return -torch.stack([Y_raw[..., 0], Y_raw[..., -1]], dim=-1)
