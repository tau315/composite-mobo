#!/usr/bin/env python3
r"""Controlled decoupling experiment: DTLZ2 whose objective is held FIXED
while the effective input-dimension of the composite raw response is dialed
from fully decoupled to fully coupled.

Motivation. Across this project's benchmarks, a single measured property --
the effective input-dimension of the raw response (mean fraction of inputs
each raw component depends on) -- separates every composite win from every
loss (Spearman rho approx -0.9; a clean threshold at ~0.4). But that is an
*observational* correlation across heterogeneous benchmarks, where effective
dimension is confounded with input-dimension, objective count, and problem
family. This benchmark removes the confounds: it holds the input dimension
`dim`, the objective count (M=2), and the *exact objective function* (hence
the Pareto front and its difficulty) constant, and varies ONLY the effective
dimension of the raw components, via a block-size knob `b`. If the composite
advantage tracks `b`, decoupling *causes* the effect rather than merely
correlating with it.

Construction. Standard bi-objective DTLZ2 on `x in [0,1]^dim`:
    g(x)   = sum_{j in distance vars} (x_j - 0.5)^2      (k = dim - 1 vars)
    theta  = x_0 * pi/2
    f_1    = (1 + g) cos(theta),  f_2 = (1 + g) sin(theta)   (both minimized)
The single position variable `x_0` sets the angle; the remaining `k` distance
variables set `g`. The objective depends on x only through `g` and `x_0`.

The composite raw response partitions the `k` distance variables into
`K = ceil(k / b)` contiguous blocks and exposes each block's partial sum
    block_i = sum_{j in block_i} (x_j - 0.5)^2
alongside the pass-through position variable `x_0`. Because
`sum_i block_i = g` for EVERY block size `b`, the reduction reconstructs the
identical objective regardless of `b` -- only the granularity of the raw
response changes:
    b = 1        -> K = k blocks, each a 1-D function of a single input
                    (fully decoupled, effective dim ~ 1/dim)
    b = k        -> K = 1 block = g itself, a full-D function of all inputs
                    (fully coupled; the composite GP then models g directly,
                    i.e. no decoupling advantage over direct modeling)
Sweeping `b` from 1 to k traces the composite advantage from its maximum to
~zero with everything else held fixed.
"""
from typing import Tuple

import math

import torch
from torch import Tensor

DTLZ2B_NUM_OBJECTIVES = 2  # this controlled study fixes M=2


def _blocks(k: int, b: int):
    """Contiguous partition of range(k) into blocks of size <= b."""
    return [list(range(s, min(s + b, k))) for s in range(0, k, b)]


def dtlz2_blocked_objective(X: Tensor, dim: int) -> Tensor:
    r"""The M=2 DTLZ2 objectives directly (maximize convention: returns
    `[-f_1, -f_2]`), for the direct baseline. Identical for every block
    size -- the block size only affects the composite raw response."""
    X = X.reshape(-1, dim).double()
    x0 = X[:, 0]
    g = ((X[:, 1:] - 0.5) ** 2).sum(dim=-1)
    theta = x0 * (math.pi / 2)
    f1 = (1 + g) * torch.cos(theta)
    f2 = (1 + g) * torch.sin(theta)
    return torch.stack([-f1, -f2], dim=-1)


def get_composite_dtlz2_blocked_fn(
    dim: int, block_size: int, dtype=torch.double, device=None
) -> Tuple[callable, Tensor]:
    r"""Raw-response function and bounds for block-structured composite DTLZ2.

    Raw response: `[x_0, block_1, ..., block_K]` (position pass-through plus
    `K = ceil((dim-1)/block_size)` partial sums of `(x_j - 0.5)^2`). Each
    block depends on `block_size` inputs -> effective input-dimension knob.
    """
    k = dim - 1
    blocks = _blocks(k, block_size)

    def raw_response(X_input: Tensor) -> Tensor:
        X = X_input.reshape(-1, dim).to(dtype)
        x0 = X[:, 0:1]
        dvars = (X[:, 1:] - 0.5) ** 2  # (n, k)
        block_sums = torch.stack([dvars[:, idx].sum(dim=-1) for idx in blocks], dim=-1)
        out = torch.cat([x0, block_sums], dim=-1)
        return out.view(*X_input.shape[:-1], 1 + len(blocks))

    bounds = torch.stack(
        [torch.zeros(dim, dtype=dtype, device=device), torch.ones(dim, dtype=dtype, device=device)]
    )
    return raw_response, bounds


def composite_dtlz2_blocked_reduction(Y_raw: Tensor) -> Tensor:
    r"""Reduction: reconstruct `g = sum_i block_i` and `theta = x_0 pi/2`,
    then the two DTLZ2 objectives (maximize convention `[-f_1, -f_2]`).
    Bit-identical to `dtlz2_blocked_objective` regardless of block size,
    since the block sums always total `g`."""
    x0 = Y_raw[..., 0]
    g = Y_raw[..., 1:].sum(dim=-1)
    theta = x0 * (math.pi / 2)
    f1 = (1 + g) * torch.cos(theta)
    f2 = (1 + g) * torch.sin(theta)
    return torch.stack([-f1, -f2], dim=-1)
