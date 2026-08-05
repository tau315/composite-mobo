#!/usr/bin/env python3
r"""Composite-structure DTLZ2, generalized to any number of objectives.

This project's own `composite_dtlz2.py` (not copied into this folder --
see below) only implements the textbook M=2 special case: raw response
`[g, cos(x_0*pi/2), sin(x_0*pi/2)]`, reduced to `f_0=(1+g)*cos`,
`f_1=(1+g)*sin`. Composite-modeling benchmarks with more objectives don't
exist elsewhere in this project, so this is new, standalone code (not
copied from anywhere) -- verified numerically against BoTorch's own
`DTLZ2` below rather than merely by inspection.

Standard DTLZ2 (Deb et al.), M objectives, D input dims: with M-1
"position" variables `p_0, ..., p_{M-2}` (the first M-1 input dims) and
`g(x) = sum` of squared deviations from 0.5 over the remaining
`k = D - M + 1` "distance" dims,

    f_j(x) = (1+g) * prod_{i=0}^{M-2-j} cos(p_i * pi/2)
                    * (sin(p_{M-1-j} * pi/2) if j > 0 else 1),   j = 0, ..., M-1

(the M=2 case is exactly `composite_dtlz2.py`'s `f_0 = (1+g)*cos(p_0)`,
`f_1 = (1+g)*sin(p_0)`.) The raw response this module exposes is the
minimal set of intermediate quantities the reduction needs:
`[g, cos(p_0), sin(p_0), cos(p_1), sin(p_1), ..., cos(p_{M-2}), sin(p_{M-2})]`
-- `1 + 2*(M-1)` raw components a composite GP would model directly,
instead of the M final objectives a direct-modeling GP would model.
"""
import math
from typing import Callable, Tuple

import torch
from torch import Tensor


def get_composite_dtlz2_general_fn(
    dim: int, num_objectives: int, dtype=torch.double, device=None
) -> Tuple[Callable, Tensor]:
    r"""Construct the raw-response function and bounds for
    general-M composite DTLZ2.

    Args:
        dim: input dimension. Must satisfy ``dim >= num_objectives`` (so
            at least one "distance" variable exists).
        num_objectives: number of objectives ``M`` (any ``M >= 2``, unlike
            ``composite_dtlz2.py``'s hardcoded ``M=2``).
        dtype, device: for the returned bounds.

    Returns:
        ``(raw_response, bounds)``: ``raw_response`` maps a
        ``... x dim`` tensor in ``[0, 1]^dim`` to a
        ``... x (1 + 2*(num_objectives - 1))`` tensor
        ``[g, cos(p_0), sin(p_0), ..., cos(p_{M-2}), sin(p_{M-2})]``.
        ``bounds`` is ``[0, 1]^dim``.
    """
    M = num_objectives
    if M < 2:
        raise ValueError(f"num_objectives must be >= 2, got {M}.")
    if dim < M:
        raise ValueError(f"dim must be >= num_objectives, got dim={dim}, M={M}.")
    k = dim - M + 1

    def raw_response(X: Tensor) -> Tensor:
        X_m = X[..., -k:]
        g = (X_m - 0.5).pow(2).sum(dim=-1, keepdim=True)  # ... x 1
        pos = X[..., : M - 1] * (math.pi / 2)  # ... x (M-1)
        cos_terms = torch.cos(pos)
        sin_terms = torch.sin(pos)
        interleaved = torch.stack([cos_terms, sin_terms], dim=-1)  # ... x (M-1) x 2
        interleaved = interleaved.reshape(*interleaved.shape[:-2], 2 * (M - 1))
        return torch.cat([g, interleaved], dim=-1)  # ... x (1 + 2*(M-1))

    bounds = torch.zeros(2, dim, dtype=dtype, device=device)
    bounds[1] = 1.0
    return raw_response, bounds


def composite_dtlz2_general_reduction(Y_raw: Tensor, num_objectives: int) -> Tensor:
    r"""Known reduction from the general-M raw response to final objectives.

    Args:
        Y_raw: ``... x (1 + 2*(num_objectives-1))`` tensor, as produced by
            ``get_composite_dtlz2_general_fn``'s ``raw_response`` (un-negated,
            true minimization-convention DTLZ2 internals).
        num_objectives: ``M``, matching what ``Y_raw`` was constructed with.

    Returns:
        ``... x M`` tensor of *maximized* objectives (negated relative to
        standard minimization DTLZ2 -- this reduction performs that
        negation itself, matching ``composite_dtlz2_reduction``'s
        convention).
    """
    M = num_objectives
    g_plus1 = 1 + Y_raw[..., 0]
    rest = Y_raw[..., 1:].reshape(*Y_raw.shape[:-1], M - 1, 2)
    cos_terms, sin_terms = rest[..., 0], rest[..., 1]  # ... x (M-1) each

    objectives = []
    for j in range(M):
        upper = M - 1 - j  # prod_{i=0}^{upper-1} cos_terms[i]
        prod_cos = cos_terms[..., :upper].prod(dim=-1) if upper > 0 else torch.ones_like(g_plus1)
        val = prod_cos * sin_terms[..., M - 1 - j] if j > 0 else prod_cos
        objectives.append(g_plus1 * val)
    return -torch.stack(objectives, dim=-1)
