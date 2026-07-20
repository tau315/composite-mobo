#!/usr/bin/env python3
r"""SparseDTLZ2 whose informative dimensions CHANGE partway through the run.

Probes the one regime no current benchmark tests: re-adaptation. The
informative distance-dimension subset switches from the FIRST k_eff
distance dims to the LAST k_eff (disjoint) at a configured fraction of the
eval budget. Predictions per method:

  - `pca_ellipsoid` (memoryless, recomputed from scratch each iteration):
    should re-adapt within a few iterations of the switch.
  - `cma_ellipsoid` (persistent covariance, temporally smoothed): its
    memory -- the exact property that made it the only method to break
    through at d=200/600ev -- should now HURT, since it must first
    un-learn the stale subspace at its decay rate.
  - `mab_shape`: gets to re-learn which arm suits the new landscape.
  - `morbo` (isotropic): blind to the switch by construction; a useful
    control for how much raw difficulty the switch itself adds.

METRIC CAVEAT (important, deliberate): this is a dynamic benchmark. The
saved `objective_history` records each point's value AT EVALUATION TIME --
pre-switch evaluations were scored under mask A, post-switch under mask B,
so a post-hoc hypervolume over the full mixed history is not a statement
about any single fixed function. The meaningful comparison is the
HV-trajectory recovery AFTER the switch (slope/level of `true_hv` past the
switch point, per-method), not the final mixed-history number. Analysis
should slice histories at the switch eval index.

Note the achievable Pareto front itself is IDENTICAL under both masks
(g=0 is reachable either way, at informative-dims=0.5), so the switch
changes which coordinates matter without moving the target front.
"""
from typing import Tuple

import math

import torch
from torch import Tensor


def get_time_varying_sparse_dtlz2_fn(
    dim: int,
    num_objectives: int,
    k_eff: int,
    switch_at_eval: int,
    dtype=torch.double,
    device=None,
) -> Tuple[callable, Tensor]:
    r"""Stateful SparseDTLZ2 with a mid-run informative-subset switch.

    Args:
        dim, num_objectives, k_eff: as in `get_sparse_dtlz2_fn`; requires
            `2 * k_eff <= k = dim - num_objectives + 1` so the pre- and
            post-switch informative subsets can be disjoint.
        switch_at_eval: total evaluation count after which the informative
            subset flips from the first `k_eff` distance dims to the last
            `k_eff`. The function COUNTS ITS OWN CALLS (row count), so this
            wrapper is stateful and must be constructed fresh per run.

    Returns:
        `(f, bounds)`, raw minimization convention (wrap with
        `BenchmarkFunction(..., negate=True)`).
    """
    if dim <= num_objectives:
        raise ValueError(f"dim must be > num_objectives, got {dim} and {num_objectives}.")
    k = dim - num_objectives + 1
    if 2 * k_eff > k:
        raise ValueError(
            f"need 2*k_eff <= k for disjoint pre/post subsets, got k_eff={k_eff}, k={k}."
        )

    mask_pre = torch.zeros(k, dtype=dtype, device=device)
    mask_pre[:k_eff] = 1.0
    mask_post = torch.zeros(k, dtype=dtype, device=device)
    mask_post[-k_eff:] = 1.0
    pi_over_2 = math.pi / 2
    n_evals_seen = 0

    def f(X: Tensor) -> Tensor:
        nonlocal n_evals_seen
        n = X.shape[0] if X.dim() > 1 else 1
        mask = mask_pre if n_evals_seen < switch_at_eval else mask_post
        n_evals_seen += n

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
