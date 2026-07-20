#!/usr/bin/env python3
r"""Bi-objective LassoBench: validation loss vs. coefficient sparsity.

LassoBench (Sehic et al., AutoML 2022, arXiv:2111.02790,
github.com/ksehic/LassoBench) tunes per-feature Lasso regularization
weights in [-1, 1]^d. Its synthetic benchmarks have KNOWN, tunable
effective dimensionality (synt_simple d=60/de=3, synt_medium d=100/de=5,
synt_high d=300/de=15, synt_hard d=1000/de=50) -- the closest available
real(istic)-problem bridge to this project's SparseDTLZ2 result that
effective dimension relative to budget, not nominal dimension, governs
when trust-region shape adaptation helps. Real-world variants (e.g. DNA:
d=180/de=43) ground it further.

LassoBench itself is single-objective (validation loss). This wrapper adds
the natural second objective its own machinery already computes: the
FRACTION OF ACTIVE (nonzero) Lasso coefficients of the fitted solution --
accuracy vs. model sparsity, a genuine tradeoff every Lasso practitioner
cares about. Objective 1 remains exactly LassoBench's own `evaluate()`
loss, so best-loss-so-far from our saved histories is directly comparable
to the single-objective numbers in their paper (Table 2: 30 repetitions,
1000 evals for synt_simple/synt_medium/RCV1/DNA-scale, 5000 for
synt_high/synt_hard).

Requires `LassoBench` to be installed (not a default dependency of this
repo -- see cluster/setup_env.sh's optional section):
    git clone https://github.com/ksehic/LassoBench.git && pip install -e LassoBench/
"""
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

_INSTALL_MSG = (
    "LassoBench is not installed. Install it with:\n"
    "  git clone https://github.com/ksehic/LassoBench.git\n"
    "  pip install -e LassoBench/\n"
    "(on the cluster: run inside the morbo-env conda env; see cluster/setup_env.sh)"
)

_SYNT_BENCH_NAMES = {"synt_simple", "synt_medium", "synt_high", "synt_hard"}


def get_lasso_bench_mo_fn(
    bench_name: str, dtype=torch.double, device=None
) -> Tuple[callable, Tensor, int]:
    r"""Construct the bi-objective LassoBench function.

    Args:
        bench_name: a LassoBench benchmark name -- one of the synthetic
            benchmarks ("synt_simple", "synt_medium", "synt_high",
            "synt_hard") or a real-world dataset name for
            `RealBenchmark(pick_data=...)` (e.g. "DNA", "Leukemia", "RCV1",
            "Breast_cancer", "Diabetes").
        dtype: dtype for the returned bounds.
        device: device for the returned bounds.

    Returns:
        A tuple `(f, bounds, dim)`:
            f: callable mapping a `n x dim` tensor in [-1, 1]^dim to a
                `n x 2` tensor of MINIMIZATION objectives
                `[validation_loss, active_coefficient_fraction]` -- meant
                to be wrapped via `BenchmarkFunction(..., negate=True)`
                like every other raw-minimization evalfn here.
            bounds: `2 x dim` tensor, [-1, 1]^dim.
            dim: the benchmark's own dimensionality (`n_features`) --
                callers should validate their configured dim against this.
    """
    try:
        import LassoBench
    except ImportError as e:
        raise ImportError(_INSTALL_MSG) from e

    if bench_name in _SYNT_BENCH_NAMES:
        bench = LassoBench.SyntheticBenchmark(pick_bench=bench_name)
    else:
        try:
            bench = LassoBench.RealBenchmark(pick_data=bench_name)
        except Exception:
            # Dataset naming differs across LassoBench versions
            # (e.g. "DNA" vs "dna") -- retry lowercased before giving up.
            bench = LassoBench.RealBenchmark(pick_data=bench_name.lower())
    dim = int(bench.n_features)

    def f(X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy()
        if X_np.ndim == 1:
            X_np = X_np[None, :]
        out = np.empty((X_np.shape[0], 2))
        for i, cfg in enumerate(X_np):
            # LassoBench raises on configs epsilon-outside [-1, 1].
            cfg = np.clip(cfg.astype(np.float64), -1.0, 1.0)
            loss = float(bench.evaluate(cfg))
            # `test()` fits the same Lasso and stores the coefficient
            # vector on the benchmark object -- the only public path to
            # the fitted coefficients. Its test-set MSE output is unused.
            bench.test(cfg)
            coef = np.asarray(bench.reg_coef)
            active_frac = float(np.count_nonzero(coef)) / dim
            out[i, 0] = loss
            out[i, 1] = active_frac
        return torch.tensor(out, dtype=X.dtype, device=X.device)

    bounds = torch.empty(2, dim, dtype=dtype, device=device)
    bounds[0] = -1.0
    bounds[1] = 1.0
    return f, bounds, dim
