#!/usr/bin/env python3
r"""Rover with dummy dimensions: a REAL problem with a controllable
nominal-vs-effective dimension gap.

Plain Rover (d=60, all dims genuinely matter) is this project's canonical
"no low-dimensional structure to exploit" problem -- trust-region shape
adaptation shows no robust effect there (near-coin-flip win rates across
seeds). SparseDTLZ2 established, synthetically, that effective dimension
relative to budget governs when shape adaptation helps. This problem
bridges the two: the SAME real Rover trajectory objective, embedded in a
larger nominal space where the extra dimensions are literal no-ops (the
trajectory only reads the first `base_dim` coordinates).

Prediction from the SparseDTLZ2 mechanism: at base_dim=60 real dims inside
nominal d=120/180, the effective-dimension story says shape adaptation
should now HELP on Rover -- the isotropic box wastes volume on the dummy
half of the space, and a PCA/CMA shape can discover that the data only
varies meaningfully in the real half. If it still shows no effect, the
"all real dims matter" property dominates and the mechanism is more
subtle than "effective dim relative to budget" alone.

Budget/protocol matches tr_shape_rover (2000 evals, batch 50, 5 seeds).
"""
from typing import Tuple

import torch
from torch import Tensor

from morbo.problems.rover import get_rover_fn


def get_sparse_rover_fn(
    dim: int,
    base_dim: int = 60,
    dtype=torch.double,
    device=None,
    **rover_kwargs,
) -> Tuple[callable, Tensor]:
    r"""Rover reading only the first `base_dim` of `dim` nominal dims.

    Args:
        dim: nominal input dimension (must be > base_dim).
        base_dim: how many leading dimensions the rover trajectory actually
            uses (must be even, matching rover's own constraint).
        dtype, device: for the returned bounds.
        rover_kwargs: forwarded to `get_rover_fn` (force_goal, force_start).

    Returns:
        `(f, bounds)` with f mapping `n x dim` -> `n x 2` (same raw
        convention as `get_rover_fn`), bounds `2 x dim`.
    """
    if dim <= base_dim:
        raise ValueError(f"dim must be > base_dim, got {dim} <= {base_dim}.")
    if base_dim % 2 != 0:
        raise ValueError(f"base_dim must be even, got {base_dim}.")
    rover_f, rover_bounds = get_rover_fn(
        base_dim, device=device, dtype=dtype, **rover_kwargs
    )

    def f(X: Tensor) -> Tensor:
        return rover_f(X[..., :base_dim])

    # Real dims keep rover's own bounds; dummy dims get [0, 1].
    bounds = torch.zeros(2, dim, dtype=dtype, device=device)
    bounds[1] = 1.0
    bounds[:, :base_dim] = rover_bounds
    return f, bounds
